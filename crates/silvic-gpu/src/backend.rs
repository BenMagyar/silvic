//! `silvic_core::Backend` over any [`GpuDevice`]: the training set's binned rows, gradients, quantized gradients and
//! partition live in device memory, and every operation is a few kernel launches from `kernels/kernels.kern`.
//!
//! Host synchronization is limited to the results the learner reads, one wait each: gradient maxima, the bag count,
//! leaf sums, scores, and per split one wait for partition and the child's histogram together
//! (`partition_and_histogram`: the histogram kernel reads the child's rows from the device-side left count).
//! Everything else (gradients, quantization, tree updates) is queued. The training set's update by a new tree is
//! deferred until validation scores are read and then started without waiting, so it overlaps the host's metric
//! evaluation.
//!
//! Quantized histograms, partitions, bags, leaf sums and score updates are integer or the same f32 operations as
//! `silvic_core::CpuBackend`, so they agree exactly; lambdarank and binary gradients agree to f32 rounding. Queries
//! longer than the largest lambdarank kernel holds are computed on the host with
//! `silvic_core::objective::lambdarank_query`.

use bytemuck::{Pod, Zeroable};
use rayon::prelude::*;
use silvic_core::backend::{Backend, BagSpec, BinSplit, DeviceTree, GradientStats, LeafRange, ObjectiveSetup, Side};
use silvic_core::data::{BinnedDataset, FeatureGroup};
use silvic_core::objective::{self, lambdarank::SigmoidTable};
use silvic_core::quantize::QuantSpec;
use silvic_core::{Result, SilvicError};

use crate::device::{Arg, GpuDevice};
use crate::params::{
    kernel_source, BagParams, CopyParams, FillParams, GradParams, GroupLayout, HistParams, HistUnit, LeafSumParams,
    PartitionParams, QuantParams, RankParams, ScanParams, StatsParams, TreeNode, TreeParams, HIST_FLOAT_BINS,
    HIST_QUANT_BINS, HIST_SIDE_LEAF, HIST_SIDE_LEFT, HIST_SIDE_RIGHT, LAMBDARANK_CLASSES, REDUCE_THREADS, SCAN_THREADS,
};

/// The kernels of `kernels.kern`, in [`KERNEL_NAMES`] order.
#[derive(Debug, Clone, Copy)]
enum Kernel {
    FillU32,
    IotaU32,
    CopyU32,
    GradRegression,
    GradBinary,
    GradStats,
    LambdarankSmall,
    LambdarankMedium,
    LambdarankLarge,
    Quantize,
    BagCount,
    ScanBlocks,
    BagScatter,
    PartitionCount,
    PartitionScatter,
    HistQuant,
    HistFloat,
    LeafSums,
    AddTree,
}

const KERNEL_NAMES: [&str; 19] = [
    "fill_u32",
    "iota_u32",
    "copy_u32",
    "grad_regression",
    "grad_binary",
    "grad_stats",
    "lambdarank_small",
    "lambdarank_medium",
    "lambdarank_large",
    "quantize",
    "bag_count",
    "scan_blocks",
    "bag_scatter",
    "partition_count",
    "partition_scatter",
    "hist_quant",
    "hist_float",
    "leaf_sums",
    "add_tree",
];

const LAMBDARANK_KERNELS: [Kernel; 3] = [
    Kernel::LambdarankSmall,
    Kernel::LambdarankMedium,
    Kernel::LambdarankLarge,
];

/// Threadgroups of a grid-stride kernel: enough to fill any current GPU, within every API's grid limit.
const MAX_GROUPS: u32 = 65_535;
/// Threads of the elementwise kernels.
const ELEMENT_THREADS: u32 = 256;
/// Threadgroups of `grad_stats`; the host reduces this many partials.
const STATS_GROUPS: u32 = 1024;
/// Fewest rows one compaction (bag, partition) threadgroup handles; more when a leaf needs more than
/// `MAX_COMPACT_BLOCKS` such chunks.
const COMPACT_CHUNK: u32 = 4096;
/// Most threadgroups of one compaction: the length of the block-count buffer `scan_blocks` scans.
const MAX_COMPACT_BLOCKS: u32 = 65_536;
/// Threads of the histogram kernels, when the device allows that many: with 32 KiB of threadgroup memory per
/// threadgroup, a full-size group hides the most memory latency (measured on an M4 Pro: 1024 beats 512 and 256).
const HIST_THREADS: u32 = 1024;
/// Histogram threadgroups (row blocks times work units) a launch aims for: enough to occupy the GPU, few enough
/// that zeroing and flushing the shared histograms stays small next to the row work.
const HIST_TARGET_GROUPS: u32 = 1024;
/// Fewest rows of a histogram row block, below which flushing dominates.
const HIST_MIN_ROWS_PER_BLOCK: u32 = 1024;
/// Most rows of a float histogram row block, bounding each f32 partial sum.
const HIST_FLOAT_MAX_ROWS_PER_BLOCK: u32 = 8192;
/// Rows of one `leaf_sums` threadgroup.
const LEAF_SUM_ROWS: u32 = REDUCE_THREADS * 16;
/// Rows per task when counting each column's most frequent bin at load.
const MFB_CHUNK_ROWS: usize = 65_536;

/// A dataset's device buffers.
struct DatasetBuffers<B> {
    num_rows: u32,
    /// Every group's row-major bins, one group after another.
    bins: B,
    /// `GroupLayout` per group.
    groups: B,
    layouts: Vec<GroupLayout>,
    scores: B,
}

/// Histogram work units of the training set for one shared-histogram capacity.
struct HistUnits<B> {
    units: B,
    col_map: B,
    count: u32,
}

/// Lambdarank inputs, with queries sorted into the kernels' size classes.
struct RankSetup<B> {
    query_bounds: B,
    queries: B,
    /// `(first index into queries, number of queries)` per entry of `LAMBDARANK_CLASSES`.
    classes: [(u32, u32); 3],
    inverse_max_dcg: B,
    label_gain: B,
    discount: B,
    /// LightGBM's pair sigmoid table as f32.
    sigmoid_table: B,
    params: RankParams,
    /// Queries longer than every kernel class holds, computed on the host.
    host_queries: Vec<usize>,
    host_bounds: Vec<u32>,
    host_label: Vec<f32>,
    host_weight: Option<Vec<f32>>,
}

/// A device buffer that grows to the largest size asked of it.
struct Scratch<B> {
    buffer: B,
    capacity: usize,
}

struct Loaded<B> {
    datasets: Vec<DatasetBuffers<B>>,
    num_rows: u32,
    row_offset: u64,
    total_bins: usize,
    objective: ObjectiveSetup,
    label: B,
    weight: Option<B>,
    /// Bound in place of an absent weight buffer.
    dummy: B,
    grad: B,
    hess: B,
    packed: B,
    partition: B,
    scratch_rows: B,
    /// One byte per row of the leaf being partitioned: whether it goes left.
    split_flags: B,
    block_counts: B,
    scan_total: B,
    stats_partials: B,
    /// `2 * total_bins` i32 (quantized) or f32 (float) histogram.
    hist: B,
    quant_units: HistUnits<B>,
    float_units: HistUnits<B>,
    leaf_partials: Scratch<B>,
    /// `(begin, end)` partition positions of each `leaf_sums` threadgroup.
    leaf_jobs: Scratch<B>,
    nodes: Scratch<B>,
    leaf_values: Scratch<B>,
    rank: Option<RankSetup<B>>,
    quant: Option<QuantSpec>,
    /// Node count of the last tree when its update of the training scores is not encoded yet (`add_tree`).
    deferred_tree: Option<u32>,
}

/// `silvic_core::Backend` over any [`GpuDevice`].
pub struct GpuBackend<D: GpuDevice> {
    state: Option<Loaded<D::Buffer>>,
    compiled: bool,
    device: D,
}

impl<D: GpuDevice> GpuBackend<D> {
    /// Wraps `device`; kernels compile on [`GpuBackend::compile`] or the first `load`.
    pub fn new(device: D) -> Self {
        GpuBackend {
            state: None,
            compiled: false,
            device,
        }
    }

    pub fn device(&self) -> &D {
        &self.device
    }

    /// Compiles the kernels for the device, once.
    pub fn compile(&mut self) -> Result<()> {
        if !self.compiled {
            self.device.compile(&kernel_source(D::PRELUDE), &KERNEL_NAMES)?;
            self.compiled = true;
        }
        Ok(())
    }

    /// The training gradients and hessians of the last `compute_gradients`, read back for inspection.
    pub fn gradients(&self) -> Result<(Vec<f32>, Vec<f32>)> {
        let state = self.loaded()?;
        let n = state.num_rows as usize;
        Ok((
            download_vec(&self.device, &state.grad, 0, n)?,
            download_vec(&self.device, &state.hess, 0, n)?,
        ))
    }

    fn loaded(&self) -> Result<&Loaded<D::Buffer>> {
        self.state
            .as_ref()
            .ok_or_else(|| SilvicError::device("backend used before load"))
    }
}

fn launch<D: GpuDevice>(
    device: &D,
    kernel: Kernel,
    grid: [u32; 2],
    threads: u32,
    args: &[Arg<'_, D::Buffer>],
) -> Result<()> {
    device.launch(kernel as usize, [grid[0], grid[1], 1], [threads, 1, 1], args)
}

/// Threadgroups of a grid-stride kernel over `n` items.
fn groups_for(n: u32, threads: u32) -> u32 {
    n.div_ceil(threads).clamp(1, MAX_GROUPS)
}

/// A parameter struct argument.
fn params<B, T: Pod>(value: &T) -> Arg<'_, B> {
    Arg::Bytes(bytemuck::bytes_of(value))
}

fn new_buffer<D: GpuDevice>(device: &D, data: &[u8]) -> Result<D::Buffer> {
    let buffer = device.alloc(data.len())?;
    if !data.is_empty() {
        device.upload(&buffer, 0, data)?;
    }
    Ok(buffer)
}

fn zeroed_vec<T: Pod + Zeroable>(len: usize) -> Vec<T> {
    vec![T::zeroed(); len]
}

fn download_vec<D: GpuDevice, T: Pod>(
    device: &D,
    buffer: &D::Buffer,
    offset_items: usize,
    len: usize,
) -> Result<Vec<T>> {
    let mut out: Vec<T> = zeroed_vec(len);
    device.download(
        buffer,
        offset_items * std::mem::size_of::<T>(),
        bytemuck::cast_slice_mut(&mut out),
    )?;
    Ok(out)
}

fn split_u64(value: u64) -> (u32, u32) {
    (value as u32, (value >> 32) as u32)
}

/// `value` as an f32 pair `(hi, lo)` with `hi + lo` within ~2^-48 relative of it, for double-float kernel arithmetic.
fn split_f64(value: f64) -> (f32, f32) {
    let hi = value as f32;
    (hi, (value - f64::from(hi)) as f32)
}

fn rows_u32(rows: usize) -> Result<u32> {
    u32::try_from(rows).map_err(|_| SilvicError::unsupported(format!("{rows} rows on one device (at most 2^32 - 1)")))
}

fn upload_dataset<D: GpuDevice>(device: &D, data: &BinnedDataset) -> Result<DatasetBuffers<D::Buffer>> {
    let num_rows = rows_u32(data.num_rows)?;
    let mut layouts = Vec::with_capacity(data.groups.len());
    let mut offset = 0u64;
    for group in &data.groups {
        let len = data.num_rows as u64 * u64::from(group.row_stride);
        if group.data.len() as u64 != len || group.row_stride % 4 != 0 {
            return Err(SilvicError::data(format!(
                "feature group of {} bytes for {} rows of stride {}",
                group.data.len(),
                data.num_rows,
                group.row_stride
            )));
        }
        let (base_lo, base_hi) = split_u64(offset);
        layouts.push(GroupLayout {
            base_lo,
            base_hi,
            row_stride: group.row_stride,
            pad: 0,
        });
        offset += len;
    }
    let total =
        usize::try_from(offset).map_err(|_| SilvicError::unsupported("binned data exceeds the address space"))?;
    let bins = device.alloc(total)?;
    for (group, layout) in data.groups.iter().zip(&layouts) {
        let base = (u64::from(layout.base_hi) << 32) | u64::from(layout.base_lo);
        if !group.data.is_empty() {
            device.upload(&bins, base as usize, &group.data)?;
        }
    }
    Ok(DatasetBuffers {
        num_rows,
        groups: new_buffer(device, bytemuck::cast_slice(&layouts))?,
        layouts,
        bins,
        scores: device.alloc(data.num_rows * 4)?,
    })
}

/// The most frequent bin of every column of `group` (LightGBM's `most_freq_bin`).
fn most_frequent_bins(group: &FeatureGroup) -> Vec<u8> {
    let stride = group.row_stride as usize;
    if stride == 0 {
        return Vec::new();
    }
    let counts = group
        .data
        .par_chunks(stride * MFB_CHUNK_ROWS)
        .map(|rows| {
            let mut counts = vec![0u32; stride * 256];
            for row in rows.chunks_exact(stride) {
                for (column, &bin) in row.iter().enumerate() {
                    counts[column * 256 + usize::from(bin)] += 1;
                }
            }
            counts
        })
        .reduce(
            || vec![0u32; stride * 256],
            |mut total, part| {
                total.iter_mut().zip(part).for_each(|(t, p)| *t += p);
                total
            },
        );
    counts
        .as_chunks::<256>()
        .0
        .iter()
        .map(|column| {
            // The lowest of equally frequent bins.
            let best = column
                .iter()
                .enumerate()
                .max_by_key(|&(bin, &count)| (count, std::cmp::Reverse(bin)));
            best.map_or(0, |(bin, _)| bin as u8)
        })
        .collect()
}

/// Splits every group's features (in bin order) into runs whose bins fit `max_bins`. A `col_map` entry packs the
/// column's unit-relative bin offset with its most frequent bin in bits 16..24 (`kernels.kern`, `hist_*`).
fn hist_units(data: &BinnedDataset, max_bins: u32, most_frequent: &[Vec<u8>]) -> Result<(Vec<HistUnit>, Vec<i32>)> {
    let mut units = Vec::new();
    let mut col_map = Vec::new();
    let mut words_base = 0u64;
    for (group, group_mfb) in data.groups.iter().zip(most_frequent) {
        let mut features = group.features.clone();
        features.sort_by_key(|&f| data.features[f].hist_offset);
        let mut start = 0;
        while start < features.len() {
            let first = data.features[features[start]].hist_offset;
            let mut end = start;
            let mut extent = 0;
            while end < features.len() {
                let meta = &data.features[features[end]];
                let next_extent = meta.hist_offset + meta.num_bins - first;
                if next_extent > max_bins {
                    break;
                }
                extent = next_extent;
                end += 1;
            }
            if end == start {
                return Err(SilvicError::data(format!("a feature has more than {max_bins} bins")));
            }
            let cols_begin = col_map.len();
            col_map.resize(cols_begin + group.row_stride as usize, -1);
            for &feature in &features[start..end] {
                let meta = &data.features[feature];
                let column = meta.column as usize;
                col_map[cols_begin + column] =
                    ((meta.hist_offset - first) | (u32::from(group_mfb[column]) << 16)) as i32;
            }
            let (words_base_lo, words_base_hi) = split_u64(words_base);
            units.push(HistUnit {
                words_base_lo,
                words_base_hi,
                row_words: group.row_stride / 4,
                bin_begin: first,
                num_bins: extent,
                cols_begin: cols_begin as u32,
                pad: [0; 2],
            });
            start = end;
        }
        words_base += data.num_rows as u64 * u64::from(group.row_stride / 4);
    }
    Ok((units, col_map))
}

fn upload_units<D: GpuDevice>(
    device: &D,
    data: &BinnedDataset,
    max_bins: u32,
    most_frequent: &[Vec<u8>],
) -> Result<HistUnits<D::Buffer>> {
    let (units, col_map) = hist_units(data, max_bins, most_frequent)?;
    Ok(HistUnits {
        units: new_buffer(device, bytemuck::cast_slice(&units))?,
        col_map: new_buffer(device, bytemuck::cast_slice(&col_map))?,
        count: units.len() as u32,
    })
}

/// `objective::lambdarank::discount` (LightGBM `DCGCalculator::GetDiscount`) for every rank a lambdarank kernel
/// holds.
fn discounts() -> Vec<f32> {
    let ranks = LAMBDARANK_CLASSES[LAMBDARANK_CLASSES.len() - 1].0 as usize;
    (0..ranks)
        .map(|rank| objective::lambdarank::discount(rank) as f32)
        .collect()
}

fn check_rank_labels(label: &[f32], label_gain: &[f64]) -> Result<()> {
    let valid = |&value: &f32| value >= 0.0 && value.fract() == 0.0 && (value as usize) < label_gain.len();
    match label.iter().find(|value| !valid(value)) {
        Some(value) => Err(SilvicError::data(format!(
            "lambdarank label {value} is not an integer index into label_gain ({} entries)",
            label_gain.len()
        ))),
        None => Ok(()),
    }
}

/// Queries per lambdarank kernel class, and the queries no class holds.
fn classify_queries(bounds: &[u32]) -> ([Vec<u32>; 3], Vec<usize>) {
    let mut classes: [Vec<u32>; 3] = Default::default();
    let mut host = Vec::new();
    for query in 0..bounds.len() - 1 {
        let size = bounds[query + 1] - bounds[query];
        match LAMBDARANK_CLASSES.iter().position(|&(capacity, _)| size <= capacity) {
            Some(class) => classes[class].push(query as u32),
            None => host.push(query),
        }
    }
    (classes, host)
}

fn rank_setup<D: GpuDevice>(
    device: &D,
    train: &BinnedDataset,
    setup: &ObjectiveSetup,
) -> Result<Option<RankSetup<D::Buffer>>> {
    let ObjectiveSetup::LambdaRank {
        sigmoid,
        norm,
        truncation_level,
        label_gain,
        inverse_max_dcg,
    } = setup
    else {
        return Ok(None);
    };
    let bounds = train
        .query_boundaries
        .as_deref()
        .ok_or_else(|| SilvicError::data("lambdarank needs query boundaries"))?;
    if bounds.len() != inverse_max_dcg.len() + 1 || bounds.last().copied() != Some(train.num_rows as u32) {
        return Err(SilvicError::data(
            "query boundaries do not match the rows and inverse max DCGs",
        ));
    }
    check_rank_labels(&train.label, label_gain)?;
    let (lists, host_queries) = classify_queries(bounds);
    let mut classes = [(0u32, 0u32); 3];
    let mut queries = Vec::with_capacity(lists.iter().map(Vec::len).sum());
    for (class, list) in classes.iter_mut().zip(&lists) {
        *class = (queries.len() as u32, list.len() as u32);
        queries.extend_from_slice(list);
    }
    let gains: Vec<f32> = label_gain.iter().map(|&g| g as f32).collect();
    let inverse: Vec<f32> = inverse_max_dcg.iter().map(|&v| v as f32).collect();
    let table = SigmoidTable::shared(*sigmoid);
    let table_values: Vec<f32> = table.values().iter().map(|&v| v as f32).collect();
    let keep_host = !host_queries.is_empty();
    Ok(Some(RankSetup {
        query_bounds: new_buffer(device, bytemuck::cast_slice(bounds))?,
        queries: new_buffer(device, bytemuck::cast_slice(&queries))?,
        classes,
        inverse_max_dcg: new_buffer(device, bytemuck::cast_slice(&inverse))?,
        label_gain: new_buffer(device, bytemuck::cast_slice(&gains))?,
        discount: new_buffer(device, bytemuck::cast_slice(&discounts()))?,
        sigmoid_table: new_buffer(device, bytemuck::cast_slice(&table_values))?,
        params: RankParams {
            query_begin: 0,
            num_queries: 0,
            has_weight: u32::from(train.weight.is_some()),
            norm: u32::from(*norm),
            truncation_level: *truncation_level,
            sigmoid: *sigmoid as f32,
            sigmoid_min_hi: split_f64(table.min_input()).0,
            sigmoid_min_lo: split_f64(table.min_input()).1,
            sigmoid_factor_hi: split_f64(table.idx_factor()).0,
            sigmoid_factor_lo: split_f64(table.idx_factor()).1,
            sigmoid_last: (table.values().len() - 1) as u32,
            pad: 0,
        },
        host_queries,
        host_bounds: if keep_host { bounds.to_vec() } else { Vec::new() },
        host_label: if keep_host { train.label.clone() } else { Vec::new() },
        host_weight: if keep_host { train.weight.clone() } else { None },
    }))
}

fn scratch<D: GpuDevice>(device: &D, capacity: usize, item_bytes: usize) -> Result<Scratch<D::Buffer>> {
    Ok(Scratch {
        buffer: device.alloc(capacity * item_bytes)?,
        capacity,
    })
}

/// Grows `scratch` to hold `len` items (contents are not kept).
fn ensure_capacity<D: GpuDevice>(
    device: &D,
    scratch: &mut Scratch<D::Buffer>,
    len: usize,
    item_bytes: usize,
) -> Result<()> {
    if len > scratch.capacity {
        let capacity = len.next_power_of_two();
        scratch.buffer = device.alloc(capacity * item_bytes)?;
        scratch.capacity = capacity;
    }
    Ok(())
}

fn load_train<D: GpuDevice>(
    device: &D,
    train: &BinnedDataset,
    datasets: Vec<DatasetBuffers<D::Buffer>>,
    objective: &ObjectiveSetup,
) -> Result<Loaded<D::Buffer>> {
    let n = train.num_rows;
    if train.label.len() != n || train.weight.as_ref().is_some_and(|w| w.len() != n) {
        return Err(SilvicError::data("label or weight length differs from the row count"));
    }
    let most_frequent: Vec<Vec<u8>> = train.groups.iter().map(most_frequent_bins).collect();
    Ok(Loaded {
        num_rows: rows_u32(n)?,
        row_offset: train.row_offset,
        total_bins: train.total_bins,
        objective: objective.clone(),
        label: new_buffer(device, bytemuck::cast_slice(&train.label))?,
        weight: train
            .weight
            .as_ref()
            .map(|w| new_buffer(device, bytemuck::cast_slice(w)))
            .transpose()?,
        dummy: device.alloc(4)?,
        grad: device.alloc(n * 4)?,
        hess: device.alloc(n * 4)?,
        packed: device.alloc(n * 4)?,
        partition: device.alloc(n * 4)?,
        scratch_rows: device.alloc(n * 4)?,
        split_flags: device.alloc(n)?,
        block_counts: device.alloc(MAX_COMPACT_BLOCKS as usize * 4)?,
        scan_total: device.alloc(4)?,
        stats_partials: device.alloc(STATS_GROUPS as usize * 8)?,
        hist: device.alloc(train.total_bins * 8)?,
        quant_units: upload_units(device, train, HIST_QUANT_BINS, &most_frequent)?,
        float_units: upload_units(device, train, HIST_FLOAT_BINS, &most_frequent)?,
        leaf_partials: scratch(device, 1024, 16)?,
        leaf_jobs: scratch(device, 1024, 8)?,
        nodes: scratch(device, 64, std::mem::size_of::<TreeNode>())?,
        leaf_values: scratch(device, 64, 4)?,
        rank: rank_setup(device, train, objective)?,
        quant: None,
        deferred_tree: None,
        datasets,
    })
}

/// Rows per threadgroup of a compaction over `count` rows: a multiple of the threadgroup size, so that at most
/// `MAX_COMPACT_BLOCKS` chunks cover the rows.
fn compact_chunk(count: u32) -> u32 {
    let chunk = count.div_ceil(MAX_COMPACT_BLOCKS).max(COMPACT_CHUNK);
    chunk.div_ceil(SCAN_THREADS) * SCAN_THREADS
}

/// Rows per histogram row block of a `count`-row leaf over `units` work units, at most `cap`.
fn hist_rows_per_block(count: u32, units: u32, cap: u32) -> u32 {
    let blocks = HIST_TARGET_GROUPS.div_ceil(units.max(1));
    count.div_ceil(blocks).max(HIST_MIN_ROWS_PER_BLOCK).min(cap).max(1)
}

/// Rows one packed shared histogram takes before an i16 half could overflow: `QuantSpec::max_rows_per_packed_block`,
/// except that a constant hessian adds exactly 1 per row, so only the gradient half bounds it.
fn packed_block_rows(spec: &QuantSpec) -> u32 {
    let rows = if spec.constant_hessian {
        (i32::from(i16::MAX) / spec.grad_max_q.max(1)) as u32
    } else {
        spec.max_rows_per_packed_block()
    };
    rows.max(1)
}

impl<B> Loaded<B> {
    fn check_leaf(&self, leaf: LeafRange) -> Result<()> {
        match leaf.begin.checked_add(leaf.count) {
            Some(end) if end <= self.num_rows => Ok(()),
            _ => Err(SilvicError::device(format!(
                "leaf {}+{} outside the {} training rows",
                leaf.begin, leaf.count, self.num_rows
            ))),
        }
    }

    fn check_hist_out(&self, out: &[f64]) -> Result<()> {
        if out.len() == 2 * self.total_bins {
            Ok(())
        } else {
            Err(SilvicError::device(format!(
                "histogram buffer of {} values for {} bins",
                out.len(),
                self.total_bins
            )))
        }
    }

    fn weight_or_dummy(&self) -> &B {
        self.weight.as_ref().unwrap_or(&self.dummy)
    }

    fn dataset(&self, dataset: usize) -> Result<&DatasetBuffers<B>> {
        self.datasets
            .get(dataset)
            .ok_or_else(|| SilvicError::device(format!("dataset {dataset} was not loaded")))
    }
}

impl<D: GpuDevice> GpuBackend<D> {
    fn pointwise_gradients(&self, kernel: Kernel, sigmoid: f32) -> Result<()> {
        let state = self.loaded()?;
        let p = GradParams {
            n: state.num_rows,
            has_weight: u32::from(state.weight.is_some()),
            sigmoid,
            pad: 0,
        };
        let train = state.dataset(0)?;
        launch(
            &self.device,
            kernel,
            [groups_for(state.num_rows, ELEMENT_THREADS), 1],
            ELEMENT_THREADS,
            &[
                Arg::Buffer(&train.scores),
                Arg::Buffer(&state.label),
                Arg::Buffer(state.weight_or_dummy()),
                Arg::Buffer(&state.grad),
                Arg::Buffer(&state.hess),
                params(&p),
            ],
        )
    }

    fn lambdarank_gradients(&self) -> Result<()> {
        let state = self.loaded()?;
        let rank = state
            .rank
            .as_ref()
            .ok_or_else(|| SilvicError::device("lambdarank was not set up"))?;
        let train = state.dataset(0)?;
        for (class, &(query_begin, num_queries)) in rank.classes.iter().enumerate() {
            if num_queries == 0 {
                continue;
            }
            let p = RankParams {
                query_begin,
                num_queries,
                ..rank.params
            };
            launch(
                &self.device,
                LAMBDARANK_KERNELS[class],
                [num_queries.min(MAX_GROUPS), 1],
                LAMBDARANK_CLASSES[class].1,
                &[
                    Arg::Buffer(&train.scores),
                    Arg::Buffer(&state.label),
                    Arg::Buffer(state.weight_or_dummy()),
                    Arg::Buffer(&rank.query_bounds),
                    Arg::Buffer(&rank.queries),
                    Arg::Buffer(&rank.inverse_max_dcg),
                    Arg::Buffer(&rank.label_gain),
                    Arg::Buffer(&rank.discount),
                    Arg::Buffer(&rank.sigmoid_table),
                    Arg::Buffer(&state.grad),
                    Arg::Buffer(&state.hess),
                    params(&p),
                ],
            )?;
        }
        self.host_lambdarank(state, rank)
    }

    /// `objective::lambdarank_query` for the queries too long for the kernels: their scores come down, their
    /// gradients go back up.
    fn host_lambdarank(&self, state: &Loaded<D::Buffer>, rank: &RankSetup<D::Buffer>) -> Result<()> {
        let train = state.dataset(0)?;
        for &query in &rank.host_queries {
            let start = rank.host_bounds[query] as usize;
            let end = rank.host_bounds[query + 1] as usize;
            let scores: Vec<f32> = download_vec(&self.device, &train.scores, start, end - start)?;
            let mut grad = vec![0f32; end - start];
            let mut hess = vec![0f32; end - start];
            let weight = rank.host_weight.as_ref().map(|w| &w[start..end]);
            objective::lambdarank_query(
                &state.objective,
                query,
                &scores,
                &rank.host_label[start..end],
                weight,
                &mut grad,
                &mut hess,
            );
            self.device
                .upload(&state.grad, start * 4, bytemuck::cast_slice(&grad))?;
            self.device
                .upload(&state.hess, start * 4, bytemuck::cast_slice(&hess))?;
        }
        Ok(())
    }

    fn gradient_stats(&self) -> Result<GradientStats> {
        let state = self.loaded()?;
        let groups = state.num_rows.div_ceil(REDUCE_THREADS).clamp(1, STATS_GROUPS);
        let p = StatsParams {
            n: state.num_rows,
            pad: [0; 3],
        };
        launch(
            &self.device,
            Kernel::GradStats,
            [groups, 1],
            REDUCE_THREADS,
            &[
                Arg::Buffer(&state.grad),
                Arg::Buffer(&state.hess),
                Arg::Buffer(&state.stats_partials),
                params(&p),
            ],
        )?;
        let partials: Vec<f32> = download_vec(&self.device, &state.stats_partials, 0, 2 * groups as usize)?;
        let mut stats = GradientStats::default();
        for pair in partials.as_chunks::<2>().0 {
            stats.max_abs_grad = stats.max_abs_grad.max(f64::from(pair[0]));
            stats.max_hess = stats.max_hess.max(f64::from(pair[1]));
        }
        Ok(stats)
    }

    /// Encodes `add_tree` for `dataset` with the uploaded tree of `num_nodes` nodes.
    fn encode_add_tree(&self, state: &Loaded<D::Buffer>, dataset: usize, num_nodes: u32) -> Result<()> {
        let data = state.dataset(dataset)?;
        let p = TreeParams {
            n: data.num_rows,
            num_nodes,
            pad: [0; 2],
        };
        launch(
            &self.device,
            Kernel::AddTree,
            [groups_for(data.num_rows, ELEMENT_THREADS), 1],
            ELEMENT_THREADS,
            &[
                Arg::Buffer(&data.bins),
                Arg::Buffer(&data.groups),
                Arg::Buffer(&state.nodes.buffer),
                Arg::Buffer(&state.leaf_values.buffer),
                Arg::Buffer(&data.scores),
                params(&p),
            ],
        )
    }

    /// Encodes the training set's update by the last tree, when `add_tree` deferred it.
    fn encode_deferred_tree(&mut self) -> Result<()> {
        let Some(num_nodes) = self.state.as_mut().and_then(|state| state.deferred_tree.take()) else {
            return Ok(());
        };
        self.encode_add_tree(self.loaded()?, 0, num_nodes)
    }

    fn hist_threads(&self) -> u32 {
        HIST_THREADS.min(self.device.max_threads_per_group())
    }

    fn fill(&self, buffer: &D::Buffer, n: u32, value: u32) -> Result<()> {
        let p = FillParams { n, value };
        launch(
            &self.device,
            Kernel::FillU32,
            [groups_for(n, ELEMENT_THREADS), 1],
            ELEMENT_THREADS,
            &[Arg::Buffer(buffer), params(&p)],
        )
    }

    /// Exclusive scan of the first `blocks` block counts; the total lands in `scan_total`.
    fn scan_block_counts(&self, state: &Loaded<D::Buffer>, blocks: u32) -> Result<()> {
        let p = ScanParams { n: blocks, pad: [0; 3] };
        launch(
            &self.device,
            Kernel::ScanBlocks,
            [1, 1],
            SCAN_THREADS,
            &[
                Arg::Buffer(&state.block_counts),
                Arg::Buffer(&state.scan_total),
                params(&p),
            ],
        )
    }

    fn read_scan_total(&self, state: &Loaded<D::Buffer>) -> Result<u32> {
        let total: Vec<u32> = download_vec(&self.device, &state.scan_total, 0, 1)?;
        Ok(total[0])
    }

    fn bernoulli_bag(&self, seed: u32, round: u32, fraction: f64) -> Result<u32> {
        let state = self.loaded()?;
        let n = state.num_rows;
        if n == 0 {
            return Ok(0);
        }
        let chunk = compact_chunk(n);
        let blocks = n.div_ceil(chunk);
        let (row_offset_lo, row_offset_hi) = split_u64(state.row_offset);
        let p = BagParams {
            n,
            chunk,
            row_offset_lo,
            row_offset_hi,
            seed,
            round,
            threshold: bag_threshold(fraction),
            pad: 0,
        };
        launch(
            &self.device,
            Kernel::BagCount,
            [blocks, 1],
            SCAN_THREADS,
            &[Arg::Buffer(&state.block_counts), params(&p)],
        )?;
        self.scan_block_counts(state, blocks)?;
        launch(
            &self.device,
            Kernel::BagScatter,
            [blocks, 1],
            SCAN_THREADS,
            &[
                Arg::Buffer(&state.block_counts),
                Arg::Buffer(&state.partition),
                params(&p),
            ],
        )?;
        self.read_scan_total(state)
    }

    /// Encodes the histogram of `leaf`'s rows (`side` `HIST_SIDE_LEAF`), or of one child of `leaf` after a partition
    /// encoded just before (the kernel reads the left count from `scan_total`). Quantized launches must cover at
    /// most `rows_per_launch` rows.
    fn encode_histogram(&self, state: &Loaded<D::Buffer>, leaf: LeafRange, side: u32, quantized: bool) -> Result<()> {
        let (kernel, units, cap) = if quantized {
            let spec = state
                .quant
                .ok_or_else(|| SilvicError::device("quantized histogram before quantize"))?;
            (Kernel::HistQuant, &state.quant_units, packed_block_rows(&spec))
        } else {
            (Kernel::HistFloat, &state.float_units, HIST_FLOAT_MAX_ROWS_PER_BLOCK)
        };
        // A child's size is unknown until the partition runs: blocks are sized for half the parent and the grid
        // covers all of it.
        let expected = if side == HIST_SIDE_LEAF {
            leaf.count
        } else {
            leaf.count.div_ceil(2)
        };
        let rows_per_block = hist_rows_per_block(expected, units.count, cap);
        let p = HistParams {
            begin: leaf.begin,
            count: leaf.count,
            rows_per_block,
            side,
        };
        self.fill(&state.hist, (2 * state.total_bins) as u32, 0)?;
        let grid = [leaf.count.div_ceil(rows_per_block), units.count];
        let bins = &state.datasets[0].bins;
        let tail = [
            Arg::Buffer(bins),
            Arg::Buffer(&units.units),
            Arg::Buffer(&units.col_map),
            Arg::Buffer(&state.hist),
            Arg::Buffer(&state.scan_total),
            params(&p),
        ];
        let args: Vec<Arg<'_, D::Buffer>> = if quantized {
            [Arg::Buffer(&state.partition), Arg::Buffer(&state.packed)]
                .into_iter()
                .chain(tail)
                .collect()
        } else {
            [
                Arg::Buffer(&state.partition),
                Arg::Buffer(&state.grad),
                Arg::Buffer(&state.hess),
            ]
            .into_iter()
            .chain(tail)
            .collect()
        };
        launch(&self.device, kernel, grid, self.hist_threads(), &args)
    }

    /// Adds the histogram the last `encode_histogram` built into `out`.
    fn read_histogram(&self, state: &Loaded<D::Buffer>, quantized: bool, out: &mut [f64]) -> Result<()> {
        let bins = 2 * state.total_bins;
        if quantized {
            let sums: Vec<i32> = download_vec(&self.device, &state.hist, 0, bins)?;
            out.iter_mut().zip(&sums).for_each(|(out, &sum)| *out += f64::from(sum));
        } else {
            let sums: Vec<f32> = download_vec(&self.device, &state.hist, 0, bins)?;
            out.iter_mut().zip(&sums).for_each(|(out, &sum)| *out += f64::from(sum));
        }
        Ok(())
    }

    /// Rows one histogram launch may cover: a quantized bin's global i32 sum is at most `rows * max_q` in magnitude.
    fn rows_per_launch(&self, state: &Loaded<D::Buffer>, quantized: bool) -> Result<u32> {
        if !quantized {
            return Ok(u32::MAX);
        }
        let spec = state
            .quant
            .ok_or_else(|| SilvicError::device("quantized histogram before quantize"))?;
        Ok(i32::MAX as u32 / spec.grad_max_q.max(spec.hess_max_q).max(1) as u32)
    }

    /// The histogram of `leaf` into `out`, in as many launches as `rows_per_launch` needs.
    fn leaf_histogram(
        &self,
        state: &Loaded<D::Buffer>,
        leaf: LeafRange,
        quantized: bool,
        out: &mut [f64],
    ) -> Result<()> {
        let rows_per_launch = self.rows_per_launch(state, quantized)?;
        out.fill(0.0);
        let end = leaf.begin + leaf.count;
        let mut begin = leaf.begin;
        while begin < end {
            let count = (end - begin).min(rows_per_launch);
            self.encode_histogram(state, LeafRange { begin, count }, HIST_SIDE_LEAF, quantized)?;
            self.read_histogram(state, quantized, out)?;
            begin += count;
        }
        Ok(())
    }

    /// Encodes the stable partition of `leaf` (non-empty) by `split`; the left count lands in `scan_total`.
    fn encode_partition(&self, state: &Loaded<D::Buffer>, leaf: LeafRange, split: &BinSplit) -> Result<()> {
        let p = self.partition_params(state, leaf, split)?;
        let blocks = leaf.count.div_ceil(p.chunk);
        let bins = &state.datasets[0].bins;
        launch(
            &self.device,
            Kernel::PartitionCount,
            [blocks, 1],
            SCAN_THREADS,
            &[
                Arg::Buffer(&state.partition),
                Arg::Buffer(bins),
                Arg::Buffer(&state.block_counts),
                Arg::Buffer(&state.split_flags),
                params(&p),
            ],
        )?;
        self.scan_block_counts(state, blocks)?;
        launch(
            &self.device,
            Kernel::PartitionScatter,
            [blocks, 1],
            SCAN_THREADS,
            &[
                Arg::Buffer(&state.partition),
                Arg::Buffer(&state.split_flags),
                Arg::Buffer(&state.block_counts),
                Arg::Buffer(&state.scan_total),
                Arg::Buffer(&state.scratch_rows),
                params(&p),
            ],
        )?;
        let copy = CopyParams {
            begin: leaf.begin,
            count: leaf.count,
            pad: [0; 2],
        };
        launch(
            &self.device,
            Kernel::CopyU32,
            [groups_for(leaf.count, ELEMENT_THREADS), 1],
            ELEMENT_THREADS,
            &[
                Arg::Buffer(&state.scratch_rows),
                Arg::Buffer(&state.partition),
                params(&copy),
            ],
        )
    }

    fn partition_params(
        &self,
        state: &Loaded<D::Buffer>,
        leaf: LeafRange,
        split: &BinSplit,
    ) -> Result<PartitionParams> {
        let layout = state.datasets[0].layouts.get(split.group as usize).ok_or_else(|| {
            SilvicError::device(format!(
                "split on group {} of {}",
                split.group,
                state.datasets[0].layouts.len()
            ))
        })?;
        if split.column >= layout.row_stride {
            return Err(SilvicError::device(format!(
                "split on column {} of a {}-byte row",
                split.column, layout.row_stride
            )));
        }
        Ok(PartitionParams {
            begin: leaf.begin,
            count: leaf.count,
            chunk: compact_chunk(leaf.count),
            row_stride: layout.row_stride,
            group_base_lo: layout.base_lo,
            group_base_hi: layout.base_hi,
            column: split.column,
            pad: 0,
            left_bins: split.left_bins,
        })
    }

    fn tree_nodes(&self, state: &Loaded<D::Buffer>, tree: &DeviceTree) -> Result<Vec<TreeNode>> {
        let layouts = &state.datasets[0].layouts;
        let child_ok = |node: usize, child: i32| {
            if child >= 0 {
                (child as usize) > node && (child as usize) < tree.nodes.len()
            } else {
                ((!child) as usize) < tree.leaf_values.len()
            }
        };
        tree.nodes
            .iter()
            .enumerate()
            .map(|(index, node)| {
                let split = &node.split;
                let layout_ok = layouts
                    .get(split.group as usize)
                    .is_some_and(|l| split.column < l.row_stride);
                if !layout_ok || !child_ok(index, node.left) || !child_ok(index, node.right) {
                    return Err(SilvicError::device(format!("malformed device tree node {index}")));
                }
                Ok(TreeNode {
                    group: split.group,
                    column: split.column,
                    left: node.left,
                    right: node.right,
                    left_bins: split.left_bins,
                })
            })
            .collect()
    }
}

/// `ceil(fraction * 2^24)`: a draw `k / 2^24` is below `fraction` exactly when `k` is below this.
fn bag_threshold(fraction: f64) -> u32 {
    const DRAWS: f64 = 16_777_216.0;
    if fraction.is_nan() || fraction <= 0.0 {
        0
    } else if fraction >= 1.0 {
        DRAWS as u32
    } else {
        (fraction * DRAWS).ceil() as u32
    }
}

impl<D: GpuDevice> Backend for GpuBackend<D> {
    fn name(&self) -> String {
        self.device.name()
    }

    fn load(&mut self, train: &BinnedDataset, valid: &[&BinnedDataset], objective: &ObjectiveSetup) -> Result<()> {
        self.compile()?;
        self.state = None;
        let device = &self.device;
        let datasets = std::iter::once(train)
            .chain(valid.iter().copied())
            .map(|data| upload_dataset(device, data))
            .collect::<Result<Vec<_>>>()?;
        if datasets.iter().any(|d| d.layouts.len() != datasets[0].layouts.len()) {
            return Err(SilvicError::data(
                "validation sets must share the training set's feature groups",
            ));
        }
        self.state = Some(load_train(device, train, datasets, objective)?);
        Ok(())
    }

    fn init_scores(&mut self, dataset: usize, value: f64) -> Result<()> {
        if dataset == 0 {
            // A deferred tree update would land on the refilled scores.
            if let Some(state) = self.state.as_mut() {
                state.deferred_tree = None;
            }
        }
        let state = self.loaded()?;
        let data = state.dataset(dataset)?;
        self.fill(&data.scores, data.num_rows, (value as f32).to_bits())
    }

    fn compute_gradients(&mut self) -> Result<GradientStats> {
        self.encode_deferred_tree()?;
        let state = self.loaded()?;
        match state.objective {
            ObjectiveSetup::Regression => self.pointwise_gradients(Kernel::GradRegression, 1.0)?,
            ObjectiveSetup::Binary { sigmoid } => self.pointwise_gradients(Kernel::GradBinary, sigmoid as f32)?,
            ObjectiveSetup::LambdaRank { .. } => self.lambdarank_gradients()?,
        }
        self.gradient_stats()
    }

    fn quantize(&mut self, spec: &QuantSpec) -> Result<()> {
        let state = self.loaded()?;
        let (row_offset_lo, row_offset_hi) = split_u64(state.row_offset);
        let p = QuantParams {
            n: state.num_rows,
            row_offset_lo,
            row_offset_hi,
            inv_grad_scale: spec.inv_grad_scale,
            inv_hess_scale: spec.inv_hess_scale,
            grad_max_q: spec.grad_max_q,
            hess_max_q: spec.hess_max_q,
            constant_hessian: u32::from(spec.constant_hessian),
            stochastic: u32::from(spec.stochastic),
            seed: spec.seed,
            iteration: spec.iteration,
            pad: 0,
        };
        launch(
            &self.device,
            Kernel::Quantize,
            [groups_for(state.num_rows, ELEMENT_THREADS), 1],
            ELEMENT_THREADS,
            &[
                Arg::Buffer(&state.grad),
                Arg::Buffer(&state.hess),
                Arg::Buffer(&state.packed),
                params(&p),
            ],
        )?;
        if let Some(state) = self.state.as_mut() {
            state.quant = Some(*spec);
        }
        Ok(())
    }

    fn reset_partition(&mut self, bag: &BagSpec) -> Result<u32> {
        match *bag {
            BagSpec::All => {
                let state = self.loaded()?;
                let p = FillParams {
                    n: state.num_rows,
                    value: 0,
                };
                launch(
                    &self.device,
                    Kernel::IotaU32,
                    [groups_for(state.num_rows, ELEMENT_THREADS), 1],
                    ELEMENT_THREADS,
                    &[Arg::Buffer(&state.partition), params(&p)],
                )?;
                Ok(state.num_rows)
            }
            BagSpec::Bernoulli { seed, round, fraction } => self.bernoulli_bag(seed, round, fraction),
        }
    }

    fn histogram(&mut self, leaf: LeafRange, quantized: bool, out: &mut [f64]) -> Result<()> {
        let state = self.loaded()?;
        state.check_hist_out(out)?;
        state.check_leaf(leaf)?;
        if leaf.count == 0 || state.total_bins == 0 {
            out.fill(0.0);
            return Ok(());
        }
        self.leaf_histogram(state, leaf, quantized, out)
    }

    fn partition(&mut self, leaf: LeafRange, split: &BinSplit) -> Result<u32> {
        let state = self.loaded()?;
        state.check_leaf(leaf)?;
        if leaf.count == 0 {
            return Ok(0);
        }
        self.encode_partition(state, leaf, split)?;
        self.read_scan_total(state)
    }

    /// Partition and the child's histogram in one command buffer: the histogram kernel takes the child's rows from
    /// the device-side left count, so the host waits once.
    fn partition_and_histogram(
        &mut self,
        leaf: LeafRange,
        split: &BinSplit,
        side: Side,
        quantized: bool,
        out: &mut [f64],
    ) -> Result<u32> {
        let state = self.loaded()?;
        state.check_hist_out(out)?;
        state.check_leaf(leaf)?;
        if leaf.count == 0 || state.total_bins == 0 || leaf.count > self.rows_per_launch(state, quantized)? {
            let left = self.partition(leaf, split)?;
            self.histogram(side.range(leaf, left), quantized, out)?;
            return Ok(left);
        }
        self.encode_partition(state, leaf, split)?;
        let side = match side {
            Side::Left => HIST_SIDE_LEFT,
            Side::Right => HIST_SIDE_RIGHT,
        };
        self.encode_histogram(state, leaf, side, quantized)?;
        out.fill(0.0);
        self.read_histogram(state, quantized, out)?;
        self.read_scan_total(state)
    }

    fn leaf_sums(&mut self, leaves: &[LeafRange], grad_scale: f32, hess_scale: f32) -> Result<Vec<[i64; 2]>> {
        let blocks: Vec<u32> = leaves.iter().map(|leaf| leaf.count.div_ceil(LEAF_SUM_ROWS)).collect();
        let total_blocks: usize = blocks.iter().map(|&b| b as usize).sum();
        let mut jobs = Vec::with_capacity(2 * total_blocks);
        for leaf in leaves {
            self.loaded()?.check_leaf(*leaf)?;
            let end = leaf.begin + leaf.count;
            for start in (leaf.begin..end).step_by(LEAF_SUM_ROWS as usize) {
                jobs.extend([start, (start + LEAF_SUM_ROWS).min(end)]);
            }
        }
        let state = self
            .state
            .as_mut()
            .ok_or_else(|| SilvicError::device("backend used before load"))?;
        ensure_capacity(&self.device, &mut state.leaf_partials, total_blocks, 16)?;
        ensure_capacity(&self.device, &mut state.leaf_jobs, total_blocks, 8)?;
        let state = self.loaded()?;
        let partials: Vec<i64> = if total_blocks == 0 {
            Vec::new()
        } else {
            self.device
                .upload(&state.leaf_jobs.buffer, 0, bytemuck::cast_slice(&jobs))?;
            let p = LeafSumParams {
                grad_scale,
                hess_scale,
                pad: [0; 2],
            };
            launch(
                &self.device,
                Kernel::LeafSums,
                [total_blocks as u32, 1],
                REDUCE_THREADS,
                &[
                    Arg::Buffer(&state.partition),
                    Arg::Buffer(&state.grad),
                    Arg::Buffer(&state.hess),
                    Arg::Buffer(&state.leaf_jobs.buffer),
                    Arg::Buffer(&state.leaf_partials.buffer),
                    params(&p),
                ],
            )?;
            download_vec(&self.device, &state.leaf_partials.buffer, 0, 2 * total_blocks)?
        };
        let mut sums = Vec::with_capacity(leaves.len());
        let mut pairs = partials.as_chunks::<2>().0.iter();
        for &leaf_blocks in &blocks {
            let sum = pairs
                .by_ref()
                .take(leaf_blocks as usize)
                .fold([0i64; 2], |acc, pair| [acc[0] + pair[0], acc[1] + pair[1]]);
            sums.push(sum);
        }
        Ok(sums)
    }

    fn add_tree(&mut self, tree: &DeviceTree) -> Result<()> {
        let state = self.loaded()?;
        if tree.leaf_values.len() != tree.nodes.len() + 1 {
            return Err(SilvicError::device(format!(
                "device tree with {} nodes and {} leaves",
                tree.nodes.len(),
                tree.leaf_values.len()
            )));
        }
        let nodes = self.tree_nodes(state, tree)?;
        // The previous tree's update of the training scores reads the node buffers about to be overwritten.
        self.encode_deferred_tree()?;
        let state = self
            .state
            .as_mut()
            .ok_or_else(|| SilvicError::device("backend used before load"))?;
        ensure_capacity(
            &self.device,
            &mut state.nodes,
            nodes.len(),
            std::mem::size_of::<TreeNode>(),
        )?;
        ensure_capacity(&self.device, &mut state.leaf_values, tree.leaf_values.len(), 4)?;
        let state = self.loaded()?;
        if !nodes.is_empty() {
            self.device
                .upload(&state.nodes.buffer, 0, bytemuck::cast_slice(&nodes))?;
        }
        self.device
            .upload(&state.leaf_values.buffer, 0, bytemuck::cast_slice(&tree.leaf_values))?;
        let num_nodes = nodes.len() as u32;
        for dataset in 1..state.datasets.len() {
            self.encode_add_tree(state, dataset, num_nodes)?;
        }
        if state.datasets.len() == 1 {
            return self.encode_add_tree(state, 0, num_nodes);
        }
        // The training set's update waits until the validation scores are read (or the training scores are next
        // used), so reading validation scores does not wait for the much larger training set.
        if let Some(state) = self.state.as_mut() {
            state.deferred_tree = Some(num_nodes);
        }
        Ok(())
    }

    fn scores(&mut self, dataset: usize) -> Result<Vec<f32>> {
        if dataset == 0 {
            self.encode_deferred_tree()?;
        }
        let state = self.loaded()?;
        let data = state.dataset(dataset)?;
        let scores = download_vec(&self.device, &data.scores, 0, data.num_rows as usize)?;
        // Start the training set's update now: it runs while the host evaluates these scores.
        self.encode_deferred_tree()?;
        self.device.flush()?;
        Ok(scores)
    }
}
