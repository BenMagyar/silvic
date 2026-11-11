//! The CPU backend: every device operation on host memory, parallel over rayon's pool. It is the CPU training path
//! and the reference the GPU backends are tested against.
//!
//! Histograms follow LightGBM's row-wise construction with "ordered gradients": a work item is one feature group
//! over one block of the leaf's rows; it gathers the rows' gradients into a contiguous buffer, then adds them at
//! each row's bins into an accumulator of the group's bins (at most `GROUP_BIN_BUDGET`, so it stays in L1). Blocks
//! are sized from the leaf's row count and the group's shape only, and each group's blocks are merged in block
//! order, so float histograms are deterministic whatever the thread count. Quantized histograms add each row's
//! packed `(h_q << 32) + g_q` in one i64 add per feature and unpack before the gradient half could overflow, so
//! they are exact integers. As LightGBM, histograms skip the features the learner will not evaluate
//! (`set_histogram_features`). Rows of small leaves, spread thinly over the dataset, are prefetched ahead.

use std::borrow::Cow;
use std::sync::Arc;

use rayon::prelude::*;
use rayon::ThreadPool;

use crate::backend::{Backend, BagSpec, BinSplit, DeviceTree, GradientStats, LeafRange, ObjectiveSetup};
use crate::data::BinnedDataset;
use crate::error::{Result, SilvicError};
use crate::objective;
use crate::quantize::{self, QuantSpec};

/// The fewest adds per accumulator bin a histogram block does, so zeroing and merging its accumulator stay cheap.
const MIN_ADDS_PER_BIN: usize = 16;
/// The most row blocks per group of a histogram: enough to balance the cores on a large leaf. A constant, so float
/// sums do not depend on the thread count.
const MAX_BLOCKS_PER_GROUP: usize = 64;
/// Accumulator slots past a group's bins, so a bin byte beyond its feature's bins never indexes past the accumulator.
const SCRATCH_BINS: usize = 256;
/// How many rows ahead the scattered-row loops prefetch, enough to cover a DRAM miss.
const PREFETCH_ROWS: usize = 16;
/// Rows whose ids spread over more than this many dataset rows per row are prefetched by hand.
const SCATTERED_SPAN: usize = 4;
/// Bins per work item when merging histogram blocks.
const MERGE_CHUNK_BINS: usize = 1024;
/// A group's histogram adds its whole row when at least `DENSE_RUN_NUM / DENSE_RUN_DEN` of its columns are used.
const DENSE_RUN_NUM: usize = 3;
const DENSE_RUN_DEN: usize = 4;
/// Rows per work item of the partition, gradient, quantization and score passes.
const ROW_CHUNK: usize = 8192;
/// Rows `add_tree` walks down the tree together.
const TRAVERSE_LANES: usize = 8;

/// One feature group as the hot loops read it.
#[derive(Debug, Clone)]
struct GroupLayout {
    group: usize,
    stride: usize,
    /// Index of the group's bin 0 in the flat histogram.
    hist_offset: usize,
    bins: usize,
    /// Per column up to the last used one: its feature, if any.
    column_features: Vec<Option<usize>>,
    /// Per column: its feature's first bin within the group (`bins`, the scratch slots, for a column no feature
    /// uses).
    offsets: Vec<usize>,
    /// Runs `start..end` of consecutive columns whose features histograms currently need.
    runs: Vec<(u32, u32)>,
    /// Features in `runs`.
    adds: usize,
}

impl GroupLayout {
    /// Keeps the columns of the features with `used[feature]` (all of them with `None`) as runs, so a row's bins
    /// are read run by run, several bytes per load. When most columns are used, one run over the whole row is
    /// cheaper than the gaps save: adding the bins of an unused feature is harmless.
    fn activate(&mut self, used: Option<&[bool]>) {
        self.runs.clear();
        let wanted = |f: usize| used.is_none_or(|used| used.get(f).copied().unwrap_or(true));
        let width = self.column_features.len();
        let active = self.column_features.iter().filter(|f| f.is_some_and(wanted)).count();
        if active == 0 {
            self.adds = 0;
            return;
        }
        if active * DENSE_RUN_DEN >= width * DENSE_RUN_NUM {
            self.runs.push((0, width as u32));
            self.adds = width;
            return;
        }
        self.adds = active;
        let mut start = None;
        for (column, feature) in self.column_features.iter().enumerate() {
            match (feature.is_some_and(wanted), start) {
                (true, None) => start = Some(column),
                (false, Some(first)) => {
                    self.runs.push((first as u32, column as u32));
                    start = None;
                }
                _ => {}
            }
        }
        if let Some(first) = start {
            self.runs.push((first as u32, width as u32));
        }
    }
}

/// The reference backend. [`CpuBackend::borrowing`] reads the datasets in place; [`CpuBackend::new`] copies them on
/// `load`.
#[derive(Debug, Default)]
pub struct CpuBackend<'a> {
    pool: Option<Arc<ThreadPool>>,
    borrowed: Vec<&'a BinnedDataset>,
    /// Training set first, then validation sets.
    datasets: Vec<Cow<'a, BinnedDataset>>,
    layout: Vec<GroupLayout>,
    total_bins: usize,
    objective: Option<ObjectiveSetup>,
    scores: Vec<Vec<f32>>,
    grad: Vec<f32>,
    hess: Vec<f32>,
    packed: Vec<i32>,
    /// `max(grad_max_q, hess_max_q)` of the last quantization: bounds rows per packed accumulator.
    max_quant: i64,
    partition: Vec<u32>,
    scratch: Vec<u32>,
    hist: HistBuffers,
    /// The last Bernoulli bag and its rows.
    bag: Option<(BagSpec, Vec<u32>)>,
}

impl<'a> CpuBackend<'a> {
    /// A backend that copies the datasets it is loaded with.
    pub fn new() -> CpuBackend<'a> {
        CpuBackend::default()
    }

    /// A backend that reads `train` and `valid` in place when `load` gets these same datasets (by address), so the
    /// bins are not copied; other datasets are copied.
    pub fn borrowing(train: &'a BinnedDataset, valid: &[&'a BinnedDataset]) -> CpuBackend<'a> {
        let mut borrowed = vec![train];
        borrowed.extend_from_slice(valid);
        CpuBackend {
            borrowed,
            ..CpuBackend::default()
        }
    }

    fn adopt(&self, data: &BinnedDataset) -> Cow<'a, BinnedDataset> {
        match self.borrowed.iter().find(|held| std::ptr::eq(**held, data)) {
            Some(held) => Cow::Borrowed(*held),
            None => Cow::Owned(data.clone()),
        }
    }

    fn train(&self) -> Result<&BinnedDataset> {
        self.datasets
            .first()
            .map(|d| d.as_ref())
            .ok_or_else(|| SilvicError::device("cpu backend used before load"))
    }

    fn check_range(&self, leaf: LeafRange) -> Result<()> {
        let end = leaf.begin as usize + leaf.count as usize;
        if end > self.partition.len() {
            return Err(SilvicError::device(format!(
                "leaf rows {}..{end} outside the partition of {} rows",
                leaf.begin,
                self.partition.len()
            )));
        }
        Ok(())
    }

    /// Overwrites the training gradients, for tests that grow trees without an objective.
    #[cfg(test)]
    pub(crate) fn set_gradients(&mut self, grad: &[f32], hess: &[f32]) {
        self.grad.copy_from_slice(grad);
        self.hess.copy_from_slice(hess);
    }
}

/// The hot-loop layout of each group, checking the invariants `accumulate` relies on: every feature's bins lie
/// inside its group's and every column inside the row.
fn group_layout(data: &BinnedDataset) -> Result<Vec<GroupLayout>> {
    data.groups
        .iter()
        .enumerate()
        .map(|(group, g)| {
            let bins = g.total_bins as usize;
            let stride = g.row_stride as usize;
            let width = g
                .features
                .iter()
                .map(|&f| data.features[f].column as usize + 1)
                .max()
                .unwrap_or(0);
            let mut column_features = vec![None; width];
            let mut offsets = vec![bins; width];
            for &f in &g.features {
                let meta = &data.features[f];
                let offset = (meta.hist_offset as usize).checked_sub(g.hist_offset as usize);
                match offset {
                    Some(offset) if offset + meta.num_bins as usize <= bins && (meta.column as usize) < stride => {
                        column_features[meta.column as usize] = Some(f);
                        offsets[meta.column as usize] = offset;
                    }
                    _ => {
                        return Err(SilvicError::data(format!(
                            "feature `{}` lies outside its group's bins or row",
                            meta.name
                        )))
                    }
                }
            }
            let mut layout = GroupLayout {
                group,
                stride,
                hist_offset: g.hist_offset as usize,
                bins,
                column_features,
                offsets,
                runs: Vec::new(),
                adds: 0,
            };
            layout.activate(None);
            Ok(layout)
        })
        .collect()
}

/// One group's rows and layout, borrowed for a histogram pass.
struct GroupView<'d> {
    data: &'d [u8],
    stride: usize,
    hist_offset: usize,
    bins: usize,
    offsets: &'d [usize],
    /// Column runs to add.
    runs: &'d [(u32, u32)],
    /// Features to add per row.
    adds: usize,
    /// One past the last column read: the bytes of a row to prefetch.
    width: usize,
}

fn group_views<'d>(layout: &'d [GroupLayout], data: &'d BinnedDataset) -> Vec<GroupView<'d>> {
    layout
        .iter()
        .map(|g| GroupView {
            data: &data.groups[g.group].data,
            stride: g.stride,
            hist_offset: g.hist_offset,
            bins: g.bins,
            offsets: &g.offsets,
            runs: &g.runs,
            adds: g.adds,
            width: g.runs.last().map_or(0, |&(_, end)| end as usize),
        })
        .collect()
}

/// Adds `values[i]` at every feature bin of row `rows[i]` of `group`; `acc` holds the group's bins and
/// `SCRATCH_BINS` more. Rows spread thinly over the dataset are prefetched ahead; dense runs (the root, large
/// leaves) are left to the hardware prefetcher.
#[inline]
fn accumulate<T: Copy, A>(group: &GroupView<'_>, rows: &[u32], values: &[T], acc: &mut [A], add: impl Fn(&mut A, T)) {
    assert!(
        acc.len() >= group.bins + SCRATCH_BINS,
        "accumulator smaller than its group"
    );
    let span = match (rows.first(), rows.last()) {
        (Some(&first), Some(&last)) => last.abs_diff(first) as usize,
        _ => return,
    };
    if span > SCATTERED_SPAN * rows.len() {
        accumulate_rows::<true, T, A>(group, rows, values, acc, add);
    } else {
        accumulate_rows::<false, T, A>(group, rows, values, acc, add);
    }
}

#[inline(always)]
fn accumulate_rows<const PREFETCH: bool, T: Copy, A>(
    group: &GroupView<'_>,
    rows: &[u32],
    values: &[T],
    acc: &mut [A],
    add: impl Fn(&mut A, T),
) {
    let width = group.width;
    for (i, (&row, &value)) in rows.iter().zip(values).enumerate() {
        if PREFETCH {
            if let Some(&ahead) = rows.get(i + PREFETCH_ROWS) {
                let start = group.data.as_ptr().wrapping_add(ahead as usize * group.stride);
                prefetch(start);
                prefetch(start.wrapping_add(width.saturating_sub(1)));
            }
        }
        let base = row as usize * group.stride;
        let bins = &group.data[base..base + width];
        if group.runs.len() == 1 {
            add_run(bins, &group.offsets[..width], acc, &add, value);
        } else {
            for &(start, end) in group.runs {
                let range = start as usize..end as usize;
                add_run(&bins[range.clone()], &group.offsets[range], acc, &add, value);
            }
        }
    }
}

#[inline(always)]
fn add_run<T: Copy, A>(bins: &[u8], offsets: &[usize], acc: &mut [A], add: &impl Fn(&mut A, T), value: T) {
    for (&bin, &offset) in bins.iter().zip(offsets) {
        // SAFETY: `group_layout` keeps every offset at most `group.bins` and a bin is below 256, so the index is
        // below `group.bins + SCRATCH_BINS <= acc.len()` (asserted by `accumulate`). This is the histogram's
        // innermost loop; the bounds check costs a sizeable share of it.
        add(unsafe { acc.get_unchecked_mut(offset + usize::from(bin)) }, value);
    }
}

/// Gathers `value(row)` of each row into `out`, calling `ahead(row)` (prefetches) for a row further on: a leaf's
/// rows are scattered over the dataset.
#[inline]
fn gather<T>(rows: &[u32], out: &mut [T], ahead: impl Fn(usize), value: impl Fn(usize) -> T) {
    for (i, (slot, &row)) in out.iter_mut().zip(rows).enumerate() {
        if let Some(&next) = rows.get(i + PREFETCH_ROWS) {
            ahead(next as usize);
        }
        *slot = value(row as usize);
    }
}

/// A hint to load the cache line at `address`; never faults, whatever the address.
#[inline(always)]
fn prefetch(address: *const u8) {
    // SAFETY: prefetch instructions only hint the cache; they neither fault nor read into a register, for any
    // address.
    #[cfg(target_arch = "aarch64")]
    unsafe {
        std::arch::asm!("prfm pldl1keep, [{0}]", in(reg) address, options(nostack, readonly, preserves_flags));
    }
    #[cfg(target_arch = "x86_64")]
    unsafe {
        std::arch::x86_64::_mm_prefetch::<{ std::arch::x86_64::_MM_HINT_T0 }>(address.cast());
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    let _ = address;
}

/// Rows per block of `group` for a leaf of `rows` rows: at least `MIN_ADDS_PER_BIN` adds per bin, at most
/// `MAX_BLOCKS_PER_GROUP` blocks.
fn block_rows(rows: usize, group: &GroupView<'_>) -> usize {
    let adds_per_row = group.adds.max(1);
    let min_rows = (MIN_ADDS_PER_BIN * group.bins).div_ceil(adds_per_row);
    min_rows.max(rows.div_ceil(MAX_BLOCKS_PER_GROUP)).max(1)
}

/// One (group, row block) work item of a histogram pass.
#[derive(Debug, Clone, Copy)]
struct BlockTask {
    group: usize,
    start: usize,
    end: usize,
}

/// Buffers the histogram passes reuse from call to call.
#[derive(Debug, Default)]
struct HistBuffers {
    /// Each block's sums: `2 * (bins + SCRATCH_BINS)` values per block, `[g, h]` per bin of its group.
    blocks: Vec<f64>,
    packed_gh: Vec<i64>,
    float_gh: Vec<[f64; 2]>,
    tasks: Vec<BlockTask>,
}

/// Runs `block(group, rows, values, sums)` over every (group, row block) in parallel, then writes each group's
/// block sums, added in block order, into its bins of `out`. `values[i]` belongs to `rows[i]`; `sums` (the group's
/// bins plus `SCRATCH_BINS`) is zero on entry.
fn histogram_blocks<T: Sync>(
    groups: &[GroupView<'_>],
    rows: &[u32],
    values: &[T],
    buffers: (&mut Vec<f64>, &mut Vec<BlockTask>),
    out: &mut [f64],
    block: impl Fn(&GroupView<'_>, &[u32], &[T], &mut [f64]) + Sync,
) {
    let (sums, tasks) = buffers;
    tasks.clear();
    for (group, view) in groups.iter().enumerate() {
        if view.adds == 0 {
            continue;
        }
        let size = block_rows(rows.len(), view);
        let count = rows.len().div_ceil(size).max(1);
        tasks.extend((0..count).map(|b| BlockTask {
            group,
            start: (b * size).min(rows.len()),
            end: ((b + 1) * size).min(rows.len()),
        }));
    }
    let slot = |task: &BlockTask| 2 * (groups[task.group].bins + SCRATCH_BINS);
    sums.resize(tasks.iter().map(slot).sum(), 0.0);
    let mut slots = Vec::with_capacity(tasks.len());
    let mut rest = sums.as_mut_slice();
    for task in tasks.iter() {
        let (head, tail) = std::mem::take(&mut rest).split_at_mut(slot(task));
        slots.push(head);
        rest = tail;
    }
    tasks.par_iter().zip(slots).for_each(|(task, sums)| {
        sums.fill(0.0);
        block(
            &groups[task.group],
            &rows[task.start..task.end],
            &values[task.start..task.end],
            sums,
        );
    });
    merge_blocks(groups, tasks, sums, out);
}

/// Writes each group's bins of `out` as the sum of its blocks in `sums`, in block order, in parallel over bin
/// ranges; bins outside every group, and of groups without blocks, are zeroed.
fn merge_blocks(groups: &[GroupView<'_>], tasks: &[BlockTask], sums: &[f64], out: &mut [f64]) {
    // Offsets of each group's first block in `sums`, and its block count.
    let mut firsts = vec![(0usize, 0usize); groups.len()];
    let mut offset = 0;
    for task in tasks {
        let entry = &mut firsts[task.group];
        if entry.1 == 0 {
            entry.0 = offset;
        }
        entry.1 += 1;
        offset += 2 * (groups[task.group].bins + SCRATCH_BINS);
    }
    let mut order: Vec<usize> = (0..groups.len()).collect();
    order.sort_by_key(|&g| groups[g].hist_offset);
    let mut pieces = Vec::new();
    let mut rest = out;
    let mut position = 0;
    for g in order {
        let (gap, tail) = std::mem::take(&mut rest).split_at_mut(2 * (groups[g].hist_offset - position));
        gap.fill(0.0);
        let (target, tail) = tail.split_at_mut(2 * groups[g].bins);
        rest = tail;
        position = groups[g].hist_offset + groups[g].bins;
        for (chunk, piece) in target.chunks_mut(2 * MERGE_CHUNK_BINS).enumerate() {
            pieces.push((g, chunk * 2 * MERGE_CHUNK_BINS, piece));
        }
    }
    rest.fill(0.0);
    pieces.into_par_iter().for_each(|(g, begin, piece)| {
        let (first, count) = firsts[g];
        if count == 0 {
            piece.fill(0.0);
            return;
        }
        let stride = 2 * (groups[g].bins + SCRATCH_BINS);
        let len = piece.len();
        let block = |b: usize| &sums[first + b * stride + begin..][..len];
        piece.copy_from_slice(block(0));
        for b in 1..count {
            for (value, sum) in piece.iter_mut().zip(block(b)) {
                *value += sum;
            }
        }
    });
}

/// LightGBM's "ordered gradients": each row's value, in the leaf's row order, gathered once for every group.
fn ordered<T: Copy + Default + Send>(
    rows: &[u32],
    out: &mut Vec<T>,
    ahead: impl Fn(usize) + Sync,
    value: impl Fn(usize) -> T + Sync,
) {
    out.resize(rows.len(), T::default());
    out.par_chunks_mut(ROW_CHUNK)
        .zip(rows.par_chunks(ROW_CHUNK))
        .for_each(|(out, rows)| gather(rows, out, &ahead, &value));
}

/// Quantized histogram of `rows`: exact integer `[g, h]` sums.
fn histogram_quantized(
    groups: &[GroupView<'_>],
    packed: &[i32],
    max_quant: i64,
    rows: &[u32],
    buffers: &mut HistBuffers,
    out: &mut [f64],
) {
    // Rows per packed accumulator before the gradient half could leave i32.
    let row_limit = (i64::from(i32::MAX) / max_quant.max(1)) as usize;
    ordered(
        rows,
        &mut buffers.packed_gh,
        |row| prefetch(packed.as_ptr().wrapping_add(row).cast()),
        |row| {
            let (g, h) = quantize::unpack(packed[row]);
            (i64::from(h) << 32) + i64::from(g)
        },
    );
    let blocks = (&mut buffers.blocks, &mut buffers.tasks);
    histogram_blocks(
        groups,
        rows,
        &buffers.packed_gh,
        blocks,
        out,
        |group, rows, gh, sums| {
            let mut acc = vec![0i64; group.bins + SCRATCH_BINS];
            for (rows, gh) in rows.chunks(row_limit).zip(gh.chunks(row_limit)) {
                accumulate(group, rows, gh, &mut acc, |a, v| *a += v);
                flush_packed(&mut acc, sums);
            }
        },
    );
}

/// Adds packed `(H << 32) + G` sums into the `[g, h]` pairs of `sums` (exact: integers far below 2^53) and zeroes
/// the accumulator.
fn flush_packed(acc: &mut [i64], sums: &mut [f64]) {
    for (value, pair) in acc.iter_mut().zip(sums.as_chunks_mut::<2>().0) {
        // G stays inside i32 by the row bound, so the low half is G and the rest is H.
        let g = i64::from(*value as i32);
        pair[0] += g as f64;
        pair[1] += ((*value - g) >> 32) as f64;
        *value = 0;
    }
}

/// Float histogram of `rows`: f64 sums of the f32 gradients and hessians.
fn histogram_float(
    groups: &[GroupView<'_>],
    grad: &[f32],
    hess: &[f32],
    rows: &[u32],
    buffers: &mut HistBuffers,
    out: &mut [f64],
) {
    ordered(
        rows,
        &mut buffers.float_gh,
        |row| {
            prefetch(grad.as_ptr().wrapping_add(row).cast());
            prefetch(hess.as_ptr().wrapping_add(row).cast());
        },
        |row| [f64::from(grad[row]), f64::from(hess[row])],
    );
    let blocks = (&mut buffers.blocks, &mut buffers.tasks);
    histogram_blocks(groups, rows, &buffers.float_gh, blocks, out, |group, rows, gh, sums| {
        accumulate(group, rows, gh, sums.as_chunks_mut::<2>().0, |a, v| {
            a[0] += v[0];
            a[1] += v[1];
        });
    });
}

/// Stable partition of `rows` by `goes_left`, through `scratch`; returns the left count. `ahead(row)` runs
/// (prefetches) for rows further on.
fn stable_partition(
    rows: &mut [u32],
    scratch: &mut [u32],
    ahead: impl Fn(u32) + Sync,
    goes_left: impl Fn(u32) -> bool + Sync,
) -> usize {
    let lefts: Vec<usize> = rows
        .par_chunks(ROW_CHUNK)
        .zip(scratch.par_chunks_mut(ROW_CHUNK))
        .map(|(src, dst)| {
            let (mut left, mut right) = (0, dst.len());
            for (i, &row) in src.iter().enumerate() {
                if let Some(&next) = src.get(i + PREFETCH_ROWS) {
                    ahead(next);
                }
                if goes_left(row) {
                    dst[left] = row;
                    left += 1;
                } else {
                    right -= 1;
                    dst[right] = row;
                }
            }
            dst[left..].reverse();
            left
        })
        .collect();
    let total_left: usize = lefts.iter().sum();
    let (mut left_rest, mut right_rest) = rows.split_at_mut(total_left);
    let mut targets = Vec::with_capacity(lefts.len());
    for (block, &left) in lefts.iter().enumerate() {
        let len = rows_len_of_block(block, ROW_CHUNK, scratch.len());
        let (l, lr) = std::mem::take(&mut left_rest).split_at_mut(left);
        let (r, rr) = std::mem::take(&mut right_rest).split_at_mut(len - left);
        left_rest = lr;
        right_rest = rr;
        targets.push((l, r));
    }
    scratch.par_chunks(ROW_CHUNK).zip(targets).for_each(|(src, (l, r))| {
        l.copy_from_slice(&src[..l.len()]);
        r.copy_from_slice(&src[l.len()..]);
    });
    total_left
}

fn rows_len_of_block(block: usize, chunk: usize, total: usize) -> usize {
    chunk.min(total - block * chunk)
}

/// A node of the branch-free traversal: leaves are nodes whose children are themselves, so every row takes the same
/// number of steps.
struct FlatNode<'d> {
    data: &'d [u8],
    stride: usize,
    column: usize,
    left_bins: [u32; 8],
    /// `[right, left]`.
    children: [u32; 2],
}

/// The tree as flat nodes (internal nodes, then one per leaf) and its depth; checks that children come after their
/// parent and every index is in range.
fn flatten<'d>(data: &'d BinnedDataset, tree: &DeviceTree) -> Result<(Vec<FlatNode<'d>>, usize)> {
    let internal = tree.nodes.len();
    let malformed = || SilvicError::device("malformed device tree");
    let mut depth = vec![0usize; internal];
    let mut max_depth = 0;
    let mut nodes = Vec::with_capacity(internal + tree.leaf_values.len());
    for (k, node) in tree.nodes.iter().enumerate() {
        let group = data.groups.get(node.split.group as usize).ok_or_else(malformed)?;
        let mut child = |c: i32| -> Result<u32> {
            if c >= 0 {
                let c = c as usize;
                if c <= k || c >= internal {
                    return Err(malformed());
                }
                depth[c] = depth[k] + 1;
                Ok(c as u32)
            } else if (!c as usize) < tree.leaf_values.len() {
                max_depth = max_depth.max(depth[k] + 1);
                Ok((internal + !c as usize) as u32)
            } else {
                Err(malformed())
            }
        };
        let children = [child(node.right)?, child(node.left)?];
        nodes.push(FlatNode {
            data: &group.data,
            stride: group.row_stride as usize,
            column: node.split.column as usize,
            left_bins: node.split.left_bins,
            children,
        });
    }
    let (data0, stride0, column0) = (nodes[0].data, nodes[0].stride, nodes[0].column);
    for leaf in 0..tree.leaf_values.len() {
        let this = (internal + leaf) as u32;
        nodes.push(FlatNode {
            data: data0,
            stride: stride0,
            column: column0,
            left_bins: [0; 8],
            children: [this, this],
        });
    }
    Ok((nodes, max_depth))
}

/// Adds `tree` to `scores` by traversing the rows' bins, `TRAVERSE_LANES` rows at a time: their node-to-node
/// chains of dependent loads are independent, so they overlap, and no step branches on the data.
fn add_tree_to(data: &BinnedDataset, tree: &DeviceTree, scores: &mut [f32]) -> Result<()> {
    if tree.nodes.is_empty() {
        let value = tree.leaf_values[0];
        scores.par_iter_mut().for_each(|s| *s += value);
        return Ok(());
    }
    let (nodes, depth) = flatten(data, tree)?;
    let internal = tree.nodes.len();
    scores
        .par_chunks_mut(ROW_CHUNK)
        .enumerate()
        .for_each(|(chunk, scores)| {
            for (lanes, scores) in scores.chunks_mut(TRAVERSE_LANES).enumerate() {
                let first = chunk * ROW_CHUNK + lanes * TRAVERSE_LANES;
                let mut current = [0u32; TRAVERSE_LANES];
                for _ in 0..depth {
                    for (lane, node) in current.iter_mut().enumerate().take(scores.len()) {
                        let n = &nodes[*node as usize];
                        let bin = n.data[(first + lane) * n.stride + n.column];
                        let left = (n.left_bins[usize::from(bin >> 5)] >> (bin & 31)) & 1;
                        *node = n.children[left as usize];
                    }
                }
                for (score, &node) in scores.iter_mut().zip(&current) {
                    *score += tree.leaf_values[node as usize - internal];
                }
            }
        });
    Ok(())
}

impl Backend for CpuBackend<'_> {
    fn name(&self) -> String {
        let threads = install(&self.pool, rayon::current_num_threads);
        format!("cpu ({threads} threads)")
    }

    fn set_thread_pool(&mut self, pool: Arc<ThreadPool>) {
        self.pool = Some(pool);
    }

    fn load(&mut self, train: &BinnedDataset, valid: &[&BinnedDataset], objective: &ObjectiveSetup) -> Result<()> {
        let mut datasets = vec![self.adopt(train)];
        datasets.extend(valid.iter().map(|v| self.adopt(v)));
        for data in &datasets {
            if data.groups.len() != train.groups.len() || data.features.len() != train.features.len() {
                return Err(SilvicError::data(
                    "validation set binned with a different layout than the training set",
                ));
            }
            if let Some(group) = data
                .groups
                .iter()
                .find(|g| g.data.len() != data.num_rows * g.row_stride as usize)
            {
                return Err(SilvicError::data(format!(
                    "group of {} bytes does not hold {} rows of {} bytes",
                    group.data.len(),
                    data.num_rows,
                    group.row_stride
                )));
            }
        }
        let rows = train.num_rows;
        if rows > u32::MAX as usize {
            return Err(SilvicError::unsupported(format!(
                "{rows} rows on one rank; the limit is 2^32 - 1"
            )));
        }
        self.layout = group_layout(train)?;
        self.total_bins = train.total_bins;
        self.scores = datasets.iter().map(|d| vec![0.0; d.num_rows]).collect();
        self.datasets = datasets;
        self.objective = Some(objective.clone());
        self.grad = vec![0.0; rows];
        self.hess = vec![0.0; rows];
        self.packed = vec![0; rows];
        self.partition = Vec::with_capacity(rows);
        self.scratch = vec![0; rows];
        Ok(())
    }

    fn init_scores(&mut self, dataset: usize, value: f64) -> Result<()> {
        let scores = self
            .scores
            .get_mut(dataset)
            .ok_or_else(|| SilvicError::device(format!("no dataset {dataset}")))?;
        scores.fill(value as f32);
        Ok(())
    }

    fn compute_gradients(&mut self) -> Result<GradientStats> {
        let setup = self
            .objective
            .as_ref()
            .ok_or_else(|| SilvicError::device("cpu backend used before load"))?;
        let train = self.datasets[0].as_ref();
        let (scores, grad, hess) = (&self.scores[0], &mut self.grad, &mut self.hess);
        Ok(install(&self.pool, || {
            objective::gradients(
                setup,
                scores,
                &train.label,
                train.weight.as_deref(),
                train.query_boundaries.as_deref(),
                grad,
                hess,
            );
            let max_abs = |v: &[f32]| v.par_iter().map(|x| f64::from(x.abs())).reduce(|| 0.0, f64::max);
            GradientStats {
                max_abs_grad: max_abs(grad),
                max_hess: max_abs(hess),
            }
        }))
    }

    fn quantize(&mut self, spec: &QuantSpec) -> Result<()> {
        let row_offset = self.train()?.row_offset;
        let (grad, hess) = (&self.grad, &self.hess);
        let packed = &mut self.packed;
        install(&self.pool, || {
            packed
                .par_chunks_mut(ROW_CHUNK)
                .enumerate()
                .for_each(|(chunk, packed)| {
                    let first = chunk * ROW_CHUNK;
                    for (offset, slot) in packed.iter_mut().enumerate() {
                        let row = first + offset;
                        *slot = quantize::quantize_row(spec, grad[row], hess[row], row_offset + row as u64);
                    }
                })
        });
        self.max_quant = i64::from(spec.grad_max_q.max(spec.hess_max_q));
        Ok(())
    }

    fn reset_partition(&mut self, bag: &BagSpec) -> Result<u32> {
        let train = self
            .datasets
            .first()
            .ok_or_else(|| SilvicError::device("cpu backend used before load"))?;
        let (rows, row_offset) = (train.num_rows as u32, train.row_offset);
        let partition = &mut self.partition;
        partition.clear();
        match *bag {
            BagSpec::All => install(&self.pool, || partition.par_extend(0..rows)),
            // A bag holds for `bagging_freq` iterations; drawing it once saves a hash per row in the others.
            BagSpec::Bernoulli { .. } if self.bag.as_ref().is_some_and(|(spec, _)| spec == bag) => {
                partition.extend_from_slice(&self.bag.as_ref().expect("cached bag").1);
            }
            BagSpec::Bernoulli { seed, round, fraction } => {
                install(&self.pool, || {
                    partition.par_extend(
                        (0..rows)
                            .into_par_iter()
                            .filter(|&row| quantize::in_bag(seed, round, row_offset + u64::from(row), fraction)),
                    )
                });
                let mut cached = self.bag.take().map(|(_, rows)| rows).unwrap_or_default();
                cached.clone_from(partition);
                self.bag = Some((*bag, cached));
            }
        }
        Ok(self.partition.len() as u32)
    }

    fn set_histogram_features(&mut self, used: &[bool]) {
        for group in &mut self.layout {
            group.activate(Some(used));
        }
    }

    fn histogram(&mut self, leaf: LeafRange, quantized: bool, out: &mut [f64]) -> Result<()> {
        self.check_range(leaf)?;
        if out.len() != 2 * self.total_bins {
            return Err(SilvicError::device(format!(
                "histogram buffer of {} values, expected {}",
                out.len(),
                2 * self.total_bins
            )));
        }
        let train = self
            .datasets
            .first()
            .ok_or_else(|| SilvicError::device("cpu backend used before load"))?;
        let groups = group_views(&self.layout, train);
        let rows = &self.partition[leaf.begin as usize..(leaf.begin + leaf.count) as usize];
        let buffers = &mut self.hist;
        if quantized {
            let (packed, max_quant) = (&self.packed, self.max_quant);
            install(&self.pool, || {
                histogram_quantized(&groups, packed, max_quant, rows, buffers, out)
            });
        } else {
            let (grad, hess) = (&self.grad, &self.hess);
            install(&self.pool, || histogram_float(&groups, grad, hess, rows, buffers, out));
        }
        Ok(())
    }

    fn partition(&mut self, leaf: LeafRange, split: &BinSplit) -> Result<u32> {
        self.check_range(leaf)?;
        let train = self
            .datasets
            .first()
            .ok_or_else(|| SilvicError::device("cpu backend used before load"))?;
        let group = train
            .groups
            .get(split.group as usize)
            .ok_or_else(|| SilvicError::device(format!("split on missing group {}", split.group)))?;
        let (data, stride, column) = (&group.data, group.row_stride as usize, split.column as usize);
        let goes_left = |row: u32| split.goes_left(data[row as usize * stride + column]);
        let ahead = |row: u32| prefetch(data.as_ptr().wrapping_add(row as usize * stride + column));
        let range = leaf.begin as usize..(leaf.begin + leaf.count) as usize;
        let rows = &mut self.partition[range];
        let scratch = &mut self.scratch[..rows.len()];
        let left = install(&self.pool, || stable_partition(rows, scratch, ahead, goes_left));
        Ok(left as u32)
    }

    fn leaf_sums(&mut self, leaves: &[LeafRange], grad_scale: f32, hess_scale: f32) -> Result<Vec<[i64; 2]>> {
        for &leaf in leaves {
            self.check_range(leaf)?;
        }
        let (grad, hess, partition) = (&self.grad, &self.hess, &self.partition);
        Ok(install(&self.pool, || {
            leaves
                .par_iter()
                .map(|leaf| {
                    partition[leaf.begin as usize..(leaf.begin + leaf.count) as usize]
                        .par_iter()
                        .map(|&row| {
                            let row = row as usize;
                            [
                                quantize::to_fixed(grad[row], grad_scale),
                                quantize::to_fixed(hess[row], hess_scale),
                            ]
                        })
                        .reduce(|| [0, 0], |a, b| [a[0] + b[0], a[1] + b[1]])
                })
                .collect()
        }))
    }

    fn add_tree(&mut self, tree: &DeviceTree) -> Result<()> {
        if tree.leaf_values.is_empty() || tree.leaf_values.len() != tree.nodes.len() + 1 {
            return Err(SilvicError::device(format!(
                "tree with {} nodes and {} leaves",
                tree.nodes.len(),
                tree.leaf_values.len()
            )));
        }
        let datasets = &self.datasets;
        let scores = &mut self.scores;
        // Datasets share the training set's groups (checked at load), so the first one validates the tree for all.
        install(&self.pool, || {
            datasets
                .iter()
                .zip(scores.iter_mut())
                .try_for_each(|(data, scores)| add_tree_to(data, tree, scores))
        })
    }

    fn scores(&mut self, dataset: usize) -> Result<Vec<f32>> {
        self.scores
            .get(dataset)
            .cloned()
            .ok_or_else(|| SilvicError::device(format!("no dataset {dataset}")))
    }
}

/// Runs `op` on `pool` when one is set, else on the current (global) pool.
fn install<R: Send>(pool: &Option<Arc<ThreadPool>>, op: impl FnOnce() -> R + Send) -> R {
    match pool {
        Some(pool) => pool.install(op),
        None => op(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{BinMapper, FeatureGroup, FeatureMeta};

    /// Two groups: features 0-1 (bins 7, 3) and feature 2 (bins 5), rows from a hash.
    fn dataset(rows: usize) -> BinnedDataset {
        let bins = [7u32, 3, 5];
        let bin = |row: usize, f: usize| (quantize::mix32((row * 3 + f) as u32) % bins[f]) as u8;
        let meta = |f: usize, group: u32, column: u32, hist_offset: u32| FeatureMeta {
            name: format!("f{f}"),
            bins: BinMapper::Categorical {
                categories: (1..bins[f] as i32).collect(),
            },
            num_bins: bins[f],
            group,
            column,
            hist_offset,
        };
        BinnedDataset {
            num_rows: rows,
            row_offset: 100,
            features: vec![meta(0, 0, 0, 0), meta(1, 0, 1, 7), meta(2, 1, 0, 10)],
            groups: vec![
                FeatureGroup {
                    features: vec![0, 1],
                    row_stride: 4,
                    hist_offset: 0,
                    total_bins: 10,
                    data: (0..rows).flat_map(|r| [bin(r, 0), bin(r, 1), 0, 0]).collect(),
                },
                FeatureGroup {
                    features: vec![2],
                    row_stride: 4,
                    hist_offset: 10,
                    total_bins: 5,
                    data: (0..rows).flat_map(|r| [bin(r, 2), 0, 0, 0]).collect(),
                },
            ],
            total_bins: 15,
            label: vec![0.0; rows],
            weight: None,
            query_boundaries: None,
        }
    }

    fn loaded(data: &BinnedDataset) -> CpuBackend<'_> {
        let mut backend = CpuBackend::borrowing(data, &[]);
        backend.load(data, &[], &ObjectiveSetup::Regression).expect("load");
        let grad: Vec<f32> = (0..data.num_rows).map(|r| (r % 13) as f32 / 4.0 - 1.5).collect();
        let hess: Vec<f32> = (0..data.num_rows).map(|r| 0.25 + (r % 5) as f32 / 8.0).collect();
        backend.set_gradients(&grad, &hess);
        backend
    }

    /// Direct per-bin sums of `value(row)` over `rows`.
    fn naive(data: &BinnedDataset, rows: &[u32], value: impl Fn(usize) -> (f64, f64)) -> Vec<f64> {
        let mut out = vec![0.0; 2 * data.total_bins];
        for &row in rows {
            let (g, h) = value(row as usize);
            for f in 0..data.features.len() {
                let index = data.features[f].hist_offset as usize + usize::from(data.bin(f, row as usize));
                out[2 * index] += g;
                out[2 * index + 1] += h;
            }
        }
        out
    }

    #[test]
    fn histograms_match_direct_sums() {
        let data = dataset(50_000);
        let mut backend = loaded(&data);
        let spec = QuantSpec::new(1.5, 0.75, None, 16, true, 9, 3);
        backend.quantize(&spec).expect("quantize");
        backend
            .reset_partition(&BagSpec::Bernoulli {
                seed: 4,
                round: 1,
                fraction: 0.7,
            })
            .expect("bag");
        let leaf = LeafRange {
            begin: 1000,
            count: 30_000,
        };
        let rows = backend.partition[1000..31_000].to_vec();
        let mut out = vec![0.0; 30];
        backend.histogram(leaf, true, &mut out).expect("quantized");
        let packed = backend.packed.clone();
        let expected = naive(&data, &rows, |r| {
            let (g, h) = quantize::unpack(packed[r]);
            (f64::from(g), f64::from(h))
        });
        assert_eq!(out, expected, "quantized");
        // With a feature mask, the bins of the used features stay exact; the others may hold anything.
        for used in [[true, false, true], [false, true, false], [false, false, true]] {
            backend.set_histogram_features(&used);
            backend.histogram(leaf, true, &mut out).expect("masked");
            for (f, meta) in data.features.iter().enumerate().filter(|(f, _)| used[*f]) {
                let bins = 2 * meta.hist_offset as usize..2 * (meta.hist_offset + meta.num_bins) as usize;
                assert_eq!(out[bins.clone()], expected[bins], "mask {used:?}: feature {f}");
            }
        }
        backend.set_histogram_features(&[true; 3]);
        backend.histogram(leaf, false, &mut out).expect("float");
        let (grad, hess) = (backend.grad.clone(), backend.hess.clone());
        let expected = naive(&data, &rows, |r| (f64::from(grad[r]), f64::from(hess[r])));
        for (bin, (a, b)) in out.iter().zip(&expected).enumerate() {
            assert!((a - b).abs() < 1e-9 * b.abs().max(1.0), "float bin {bin}: {a} vs {b}");
        }
    }

    #[test]
    fn packed_accumulators_flush_before_overflow() {
        let rows_count = 400_000;
        let data = dataset(rows_count);
        let backend = loaded(&data);
        let rows: Vec<u32> = (0..rows_count as u32).collect();
        // g_q = -32000, h_q = 3 on every row: one accumulator over all rows would take the gradient half past i32.
        let packed = vec![3 * 65_536 - 32_000; rows_count];
        let mut out = vec![0.0; 30];
        let groups = group_views(&backend.layout, &data);
        let pool = rayon::ThreadPoolBuilder::new().num_threads(1).build().expect("pool");
        // A bound of 1000 rows per packed accumulator forces flushes inside every block.
        let mut buffers = HistBuffers::default();
        pool.install(|| {
            histogram_quantized(
                &groups,
                &packed,
                i64::from(i32::MAX) / 1000,
                &rows,
                &mut buffers,
                &mut out,
            )
        });
        let expected = naive(&data, &rows, |r| {
            let (g, h) = quantize::unpack(packed[r]);
            (f64::from(g), f64::from(h))
        });
        assert_eq!(out, expected);
        assert!(
            out.iter().step_by(2).any(|&g| g < f64::from(i32::MIN)),
            "sums leave i32"
        );
    }

    #[test]
    fn partition_is_stable() {
        let data = dataset(30_000);
        let mut backend = loaded(&data);
        backend.reset_partition(&BagSpec::All).expect("all");
        let mut left_bins = [0u32; 8];
        left_bins[0] = 0b10110;
        let split = BinSplit {
            group: 0,
            column: 0,
            left_bins,
        };
        let leaf = LeafRange {
            begin: 500,
            count: 25_000,
        };
        let left = backend.partition(leaf, &split).expect("partition") as usize;
        let rows = &backend.partition[500..25_500];
        let goes_left = |row: u32| split.goes_left(data.bin(0, row as usize));
        let expected_left: Vec<u32> = (500..25_500).filter(|&r| goes_left(r)).collect();
        let expected_right: Vec<u32> = (500..25_500).filter(|&r| !goes_left(r)).collect();
        assert_eq!(left, expected_left.len());
        assert_eq!(&rows[..left], &expected_left[..]);
        assert_eq!(&rows[left..], &expected_right[..]);
        assert_eq!(
            &backend.partition[..500],
            &(0..500).collect::<Vec<u32>>()[..],
            "rows before the leaf"
        );
    }

    #[test]
    fn add_tree_and_leaf_sums() {
        let data = dataset(10_000);
        let mut backend = loaded(&data);
        backend.reset_partition(&BagSpec::All).expect("all");
        let mut left_bins = [0u32; 8];
        left_bins[0] = 0b11;
        let split = BinSplit {
            group: 1,
            column: 0,
            left_bins,
        };
        let left = backend
            .partition(
                LeafRange {
                    begin: 0,
                    count: 10_000,
                },
                &split,
            )
            .expect("partition");
        let tree = DeviceTree {
            nodes: vec![crate::backend::DeviceNode {
                split,
                left: !0,
                right: !1,
            }],
            leaf_values: vec![0.5, -0.25],
        };
        backend.init_scores(0, 1.0).expect("init");
        backend.add_tree(&tree).expect("add");
        let scores = backend.scores(0).expect("scores");
        for (row, score) in scores.iter().enumerate() {
            let expected = if data.bin(2, row) < 2 { 1.5 } else { 0.75 };
            assert_eq!(*score, expected, "row {row}");
        }
        let leaves = [
            LeafRange { begin: 0, count: left },
            LeafRange {
                begin: left,
                count: 10_000 - left,
            },
        ];
        let sums = backend.leaf_sums(&leaves, 1024.0, 256.0).expect("sums");
        for (leaf, range) in leaves.iter().enumerate() {
            let rows = &backend.partition[range.begin as usize..(range.begin + range.count) as usize];
            let g: i64 = rows
                .iter()
                .map(|&r| quantize::to_fixed(backend.grad[r as usize], 1024.0))
                .sum();
            let h: i64 = rows
                .iter()
                .map(|&r| quantize::to_fixed(backend.hess[r as usize], 256.0))
                .sum();
            assert_eq!(sums[leaf], [g, h], "leaf {leaf}");
        }
    }
}
