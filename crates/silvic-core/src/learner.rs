//! Leaf-wise tree growth: LightGBM 3.3.5 `SerialTreeLearner::Train`/`SplitInner`
//! (`src/treelearner/serial_tree_learner.cpp`) with the reductions of `DataParallelTreeLearner`.
//!
//! Every rank builds histograms of its own rows and allreduces them, so split finding sees global histograms and
//! every rank picks the same split. Each split costs one backend call and one collective: the child to build a
//! histogram for is chosen before partitioning, as the smaller by the split's estimated counts (LightGBM's, from
//! hessians; identical on every rank; the left one on a tie), and `Backend::partition_and_histogram` partitions and
//! builds it together. Its allreduce carries the left child's row count in a trailing slot, so leaf counts stay
//! exact (unlike LightGBM's data-parallel learner, which keeps the estimates). The other child's histogram is its
//! parent's minus the built one's, exact in quantized mode whichever child is built.
//!
//! Features with no valid candidate in a leaf are not evaluated in its descendants (LightGBM
//! `FeatureHistogram::is_splittable`, inherited by both children), and a pair of children both under
//! `2 * min_data_in_leaf` rows or at `max_depth` is not searched (`BeforeFindBestSplit`).
//!
//! Quantized leaf renewal follows LightGBM 4 `GradientDiscretizer::RenewIntGradTreeOutput`, with the float sums
//! replaced by exact fixed-point sums so they reduce identically on any number of ranks.

use rayon::ThreadPool;

use crate::backend::{Backend, BinSplit, DeviceNode, DeviceTree, LeafRange, Side};
use crate::collective::Collective;
use crate::config::Params;
use crate::data::{BinMapper, FeatureMeta};
use crate::error::{Result, SilvicError};
use crate::quantize;
use crate::split::{self, HistScale, LeafSums, SplitConfig, SplitInfo, SplitRule};
use crate::tree::{MissingType, SplitChildren, Tree};

/// Grows trees over one training set's features.
pub struct Learner<'a> {
    features: &'a [FeatureMeta],
    total_bins: usize,
    cfg: SplitConfig,
    num_leaves: usize,
    max_depth: i32,
    pool: Option<&'a ThreadPool>,
    /// Feature whose bins give a leaf's totals: every row falls in exactly one of its bins.
    sums_feature: Option<usize>,
}

/// One tree's inputs beyond the learner's.
#[derive(Debug, Clone, Copy)]
pub struct TreeSpec<'a> {
    pub quantized: bool,
    /// Histogram units to real sums (`HistScale::UNIT` in float mode).
    pub scale: HistScale,
    /// The features this tree may split on (`split::feature_mask`).
    pub feature_mask: &'a [bool],
    /// This rank's bagged rows, as `Backend::reset_partition` laid them out.
    pub root: LeafRange,
    /// Bagged rows over every rank.
    pub root_count: i64,
}

/// A grown tree with what the device needs to apply it.
#[derive(Debug, Clone, PartialEq)]
pub struct GrownTree {
    pub tree: Tree,
    /// Device form of each internal node's split, by node index.
    pub splits: Vec<BinSplit>,
    /// This rank's rows of each leaf in the backend's partition.
    pub leaves: Vec<LeafRange>,
}

impl GrownTree {
    /// The tree as `Backend::add_tree` takes it, from the current (shrunk) leaf values.
    pub fn device_tree(&self) -> DeviceTree {
        let nodes = self
            .splits
            .iter()
            .enumerate()
            .map(|(node, &split)| DeviceNode {
                split,
                left: self.tree.left_child[node],
                right: self.tree.right_child[node],
            })
            .collect();
        DeviceTree {
            nodes,
            leaf_values: self.tree.leaf_value.iter().map(|&v| v as f32).collect(),
        }
    }
}

/// A leaf being grown.
struct Leaf {
    range: LeafRange,
    sums: LeafSums,
    hist: Option<Vec<f64>>,
    /// By feature: whether the leaf had any valid candidate; descendants skip features without one.
    splittable: Vec<bool>,
    best: Option<SplitInfo>,
}

/// The growing tree and its bookkeeping.
struct Growth<'s> {
    spec: TreeSpec<'s>,
    tree: Tree,
    leaves: Vec<Leaf>,
    splits: Vec<BinSplit>,
    /// Histogram buffers of split leaves, reused for new ones.
    spare: Vec<Vec<f64>>,
}

impl<'a> Learner<'a> {
    pub fn new(params: &Params, features: &'a [FeatureMeta], total_bins: usize, pool: Option<&'a ThreadPool>) -> Self {
        Learner {
            features,
            total_bins,
            cfg: SplitConfig::new(params),
            num_leaves: params.num_leaves,
            max_depth: params.max_depth,
            pool,
            sums_feature: features.iter().position(|f| !f.is_trivial()),
        }
    }

    /// `SerialTreeLearner::Train`: grows one tree leaf-wise. Every rank must call this with the same spec except
    /// `root`, and gets the same tree.
    pub fn grow(
        &self,
        backend: &mut dyn Backend,
        collective: &mut dyn Collective,
        spec: TreeSpec<'_>,
    ) -> Result<GrownTree> {
        let mut growth = Growth {
            spec,
            tree: Tree::new(),
            leaves: vec![Leaf {
                range: spec.root,
                sums: LeafSums {
                    count: spec.root_count,
                    ..LeafSums::default()
                },
                hist: None,
                splittable: vec![true; self.features.len()],
                best: None,
            }],
            splits: Vec::new(),
            spare: Vec::new(),
        };
        if let Some(sums_feature) = self.sums_feature {
            if self.may_split(0, spec.root_count, 0) {
                // The root's sums come from `sums_feature`'s bins, so they are needed whatever the mask.
                let mut root_features = spec.feature_mask.to_vec();
                root_features[sums_feature] = true;
                backend.set_histogram_features(&root_features);
                let hist = self.histogram(backend, collective, &mut growth, spec.root)?;
                let meta = &self.features[sums_feature];
                let begin = 2 * meta.hist_offset as usize;
                let feature_hist = &hist[begin..begin + 2 * meta.num_bins as usize];
                growth.leaves[0].sums = LeafSums::from_feature_hist(feature_hist, spec.root_count);
                let evaluate = spec.feature_mask.to_vec();
                self.find_split(&mut growth.leaves[0], hist, &evaluate, spec.scale);
            }
        }
        while growth.tree.num_leaves < self.num_leaves {
            let Some(best) = best_leaf(&growth.leaves) else {
                break;
            };
            self.split_leaf(backend, collective, &mut growth, best)?;
        }
        Ok(GrownTree {
            tree: growth.tree,
            splits: growth.splits,
            leaves: growth.leaves.iter().map(|leaf| leaf.range).collect(),
        })
    }

    /// `BeforeFindBestSplit`: whether a pair of new leaves (or the root, `right_count = 0`) is searched at all.
    fn may_split(&self, depth: i32, left_count: i64, right_count: i64) -> bool {
        let min_pair = 2 * self.cfg.min_data_in_leaf;
        self.depth_allows(depth) && (left_count >= min_pair || right_count >= min_pair)
    }

    /// The part of `may_split` known before a split's exact counts.
    fn depth_allows(&self, depth: i32) -> bool {
        self.max_depth <= 0 || depth < self.max_depth
    }

    /// A histogram buffer: `2 * total_bins` values and one trailing slot that carries a row count through the same
    /// allreduce.
    fn hist_buffer(&self, growth: &mut Growth<'_>) -> Vec<f64> {
        growth.spare.pop().unwrap_or_else(|| vec![0.0; 2 * self.total_bins + 1])
    }

    /// The global histogram of a leaf's rows.
    fn histogram(
        &self,
        backend: &mut dyn Backend,
        collective: &mut dyn Collective,
        growth: &mut Growth<'_>,
        range: LeafRange,
    ) -> Result<Vec<f64>> {
        let mut hist = self.hist_buffer(growth);
        let bins = 2 * self.total_bins;
        backend.histogram(range, growth.spec.quantized, &mut hist[..bins])?;
        collective.allreduce_sum_f64(&mut hist[..bins])?;
        Ok(hist)
    }

    /// Partitions a leaf and builds the global histogram of its `side` child in one backend call and one
    /// allreduce, the left count riding in the histogram's trailing slot (exact: an integer below 2^53). Returns
    /// the local and global left counts and the histogram.
    fn partition_with_histogram(
        &self,
        backend: &mut dyn Backend,
        collective: &mut dyn Collective,
        growth: &mut Growth<'_>,
        (range, split, side): (LeafRange, &BinSplit, Side),
    ) -> Result<(u32, i64, Vec<f64>)> {
        let mut hist = self.hist_buffer(growth);
        let bins = 2 * self.total_bins;
        let left_local =
            backend.partition_and_histogram(range, split, side, growth.spec.quantized, &mut hist[..bins])?;
        hist[bins] = f64::from(left_local);
        collective.allreduce_sum_f64(&mut hist)?;
        Ok((left_local, hist[bins] as i64, hist))
    }

    /// Partitions a leaf whose children are not searched: only the left count is reduced.
    fn partition_only(
        &self,
        backend: &mut dyn Backend,
        collective: &mut dyn Collective,
        range: LeafRange,
        split: &BinSplit,
    ) -> Result<(u32, i64)> {
        let left_local = backend.partition(range, split)?;
        let mut left_count = [i64::from(left_local)];
        collective.allreduce_sum_i64(&mut left_count)?;
        Ok((left_local, left_count[0]))
    }

    fn find_split(&self, leaf: &mut Leaf, hist: Vec<f64>, evaluate: &[bool], scale: HistScale) {
        let bins = &hist[..2 * self.total_bins];
        let search = || split::find_best_split(bins, self.features, evaluate, leaf.sums, scale, &self.cfg);
        let found = match self.pool {
            Some(pool) => pool.install(search),
            None => search(),
        };
        leaf.splittable = found.splittable;
        leaf.best = found.best;
        leaf.hist = Some(hist);
    }

    /// `SerialTreeLearner::SplitInner` plus the next round's histograms and split search for both children.
    fn split_leaf(
        &self,
        backend: &mut dyn Backend,
        collective: &mut dyn Collective,
        growth: &mut Growth<'_>,
        index: usize,
    ) -> Result<()> {
        let info = growth.leaves[index]
            .best
            .take()
            .ok_or_else(|| SilvicError::internal("the leaf chosen to split has no split"))?;
        let meta = &self.features[info.feature];
        let bin_split = bin_split(meta, &info.rule);
        let parent_range = growth.leaves[index].range;
        let parent_count = growth.leaves[index].sums.count;
        // Whether the children can be searched, as far as is known before their exact counts: a leaf with a best
        // split always has a histogram.
        let may_search = growth.leaves[index].hist.is_some()
            && growth.tree.num_leaves + 1 < self.num_leaves
            && self.depth_allows(growth.tree.leaf_depth[index] + 1);
        // The child that gets a fresh histogram: the smaller by the split's estimated counts (the left on a tie),
        // which every rank computes identically, so it is chosen before the partition.
        let built = if info.left.count <= info.right.count {
            Side::Left
        } else {
            Side::Right
        };
        // The children evaluate the features their parent could split on; only those bins of their histograms are
        // ever read, here and in every descendant.
        let evaluate: Vec<bool> = growth
            .spec
            .feature_mask
            .iter()
            .zip(&growth.leaves[index].splittable)
            .map(|(&used, &splittable)| used && splittable)
            .collect();
        let (left_local, left_count, built_hist) = if may_search {
            backend.set_histogram_features(&evaluate);
            let (local, global, hist) =
                self.partition_with_histogram(backend, collective, growth, (parent_range, &bin_split, built))?;
            (local, global, Some(hist))
        } else {
            let (local, global) = self.partition_only(backend, collective, parent_range, &bin_split)?;
            (local, global, None)
        };
        let right_count = parent_count - left_count;
        if left_count <= 0 || right_count <= 0 {
            return Err(SilvicError::internal(format!(
                "split of leaf {index} on feature {} sends {left_count} rows left and {right_count} right",
                info.feature
            )));
        }
        let scale = growth.spec.scale;
        let children = SplitChildren {
            left_value: info.left_output,
            right_value: info.right_output,
            left_count,
            right_count,
            left_weight: info.left.hess * scale.hess,
            right_weight: info.right.hess * scale.hess,
            // LightGBM stores the gain over the parent's without `min_gain_to_split`.
            gain: (info.gain + self.cfg.min_gain_to_split) as f32,
        };
        let right_index = apply_split(&mut growth.tree, index, &info, meta, &children);
        if right_index != growth.leaves.len() {
            return Err(SilvicError::internal(format!(
                "tree numbered the new leaf {right_index}, expected {}",
                growth.leaves.len()
            )));
        }
        growth.splits.push(bin_split);
        let parent_hist = growth.leaves[index].hist.take();
        growth.leaves[index] = new_leaf(
            Side::Left.range(parent_range, left_local),
            LeafSums {
                count: left_count,
                ..info.left
            },
        );
        growth.leaves.push(new_leaf(
            Side::Right.range(parent_range, left_local),
            LeafSums {
                count: right_count,
                ..info.right
            },
        ));
        let depth = growth.tree.leaf_depth[index];
        let searched = self.may_split(depth, left_count, right_count);
        match (parent_hist, built_hist) {
            (Some(parent_hist), Some(built_hist)) if searched => {
                let children = match built {
                    Side::Left => [index, right_index],
                    Side::Right => [right_index, index],
                };
                self.search_children(growth, parent_hist, built_hist, &evaluate, children);
                Ok(())
            }
            (parent_hist, built_hist) => {
                growth.spare.extend(parent_hist);
                growth.spare.extend(built_hist);
                Ok(())
            }
        }
    }

    /// The best split of each child: `built` from its own histogram, `other` from the parent's minus `built`'s
    /// (exact in quantized mode, so which child is built does not change the result).
    fn search_children(
        &self,
        growth: &mut Growth<'_>,
        mut parent_hist: Vec<f64>,
        built_hist: Vec<f64>,
        evaluate: &[bool],
        [built, other]: [usize; 2],
    ) {
        let bins = 2 * self.total_bins;
        for (parent, child) in parent_hist[..bins].iter_mut().zip(&built_hist[..bins]) {
            *parent -= child;
        }
        let scale = growth.spec.scale;
        self.find_split(&mut growth.leaves[built], built_hist, evaluate, scale);
        self.find_split(&mut growth.leaves[other], parent_hist, evaluate, scale);
    }

    /// Quantized-mode leaf renewal: each leaf's output from exact fixed-point sums of its rows' float gradients
    /// over every rank, with power-of-two scales from the global maxima. Leaf weights become the renewed hessian
    /// sums.
    pub fn renew_leaves(
        &self,
        backend: &mut dyn Backend,
        collective: &mut dyn Collective,
        grown: &mut GrownTree,
        max_abs_grad: f64,
        max_hess: f64,
    ) -> Result<()> {
        let grad_scale = quantize::fixed_point_scale(max_abs_grad);
        let hess_scale = quantize::fixed_point_scale(max_hess);
        let sums = backend.leaf_sums(&grown.leaves, grad_scale, hess_scale)?;
        if sums.len() != grown.leaves.len() {
            return Err(SilvicError::device(format!(
                "leaf_sums returned {} leaves, expected {}",
                sums.len(),
                grown.leaves.len()
            )));
        }
        let mut flat: Vec<i64> = sums.iter().flatten().copied().collect();
        collective.allreduce_sum_i64(&mut flat)?;
        for (leaf, pair) in flat.as_chunks::<2>().0.iter().enumerate() {
            let sum_grad = pair[0] as f64 / f64::from(grad_scale);
            let sum_hess = pair[1] as f64 / f64::from(hess_scale);
            let output = split::leaf_output(sum_grad, sum_hess, self.cfg.lambda_l1, self.cfg.lambda_l2);
            grown.tree.set_leaf_output(leaf, output);
            grown.tree.leaf_weight[leaf] = sum_hess;
        }
        Ok(())
    }
}

fn new_leaf(range: LeafRange, sums: LeafSums) -> Leaf {
    Leaf {
        range,
        sums,
        hist: None,
        splittable: Vec::new(),
        best: None,
    }
}

/// `ArrayArgs::ArgMax` over the leaves' best gains (the first on ties), when that gain is positive.
fn best_leaf(leaves: &[Leaf]) -> Option<usize> {
    let mut best: Option<(usize, f64)> = None;
    for (index, leaf) in leaves.iter().enumerate() {
        let Some(split) = &leaf.best else {
            continue;
        };
        if best.is_none_or(|(_, gain)| split.gain > gain) {
            best = Some((index, split.gain));
        }
    }
    best.filter(|&(_, gain)| gain > 0.0).map(|(index, _)| index)
}

/// The split in device form: the bins of the feature that go left.
fn bin_split(meta: &FeatureMeta, rule: &SplitRule) -> BinSplit {
    let mut left_bins = [0u32; 8];
    let mut set = |bin: u32| left_bins[bin as usize / 32] |= 1 << (bin % 32);
    match rule {
        SplitRule::Numerical {
            threshold,
            default_left,
        } => {
            (0..=*threshold).for_each(&mut set);
            if *default_left && has_nan_bin(meta) {
                set(meta.num_bins - 1);
            }
        }
        SplitRule::Categorical { bins } => bins.iter().copied().for_each(set),
    }
    BinSplit {
        group: meta.group,
        column: meta.column,
        left_bins,
    }
}

fn has_nan_bin(meta: &FeatureMeta) -> bool {
    matches!(meta.bins, BinMapper::Numerical { nan_bin: true, .. })
}

/// `Tree::Split`/`Tree::SplitCategorical` with the real threshold or category values; returns the right leaf.
fn apply_split(tree: &mut Tree, leaf: usize, info: &SplitInfo, meta: &FeatureMeta, children: &SplitChildren) -> usize {
    match (&info.rule, &meta.bins) {
        (
            SplitRule::Numerical {
                threshold,
                default_left,
            },
            BinMapper::Numerical {
                upper_bounds, nan_bin, ..
            },
        ) => {
            // The last numeric bin's bound is +inf: such a split is NaN against everything else, which LightGBM
            // writes with a finite threshold.
            let bound = upper_bounds[*threshold as usize];
            let threshold = if bound.is_finite() { bound } else { f64::MAX };
            let missing = if *nan_bin { MissingType::NaN } else { MissingType::None };
            tree.split_numerical(leaf, info.feature, threshold, *default_left, missing, children)
        }
        (SplitRule::Categorical { bins }, BinMapper::Categorical { categories }) => {
            let values: Vec<i32> = bins.iter().map(|&bin| categories[bin as usize - 1]).collect();
            // Bin 0 (negative, NaN, unseen) always goes right; `MissingType::NaN` makes prediction send NaN right
            // too, where `None` would map it to category 0.
            tree.split_categorical(leaf, info.feature, &values, MissingType::NaN, children)
        }
        _ => unreachable!("split rules come from the feature's own bin mapper"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{BagSpec, CpuBackend, ObjectiveSetup};
    use crate::collective::LocalCollective;
    use crate::data::{BinnedDataset, FeatureGroup};
    use crate::quantize::QuantSpec;
    use crate::test_util::run_ranks;

    /// Bins per feature: numerical without NaN, numerical with a NaN bin (last), categorical.
    const BINS: [u32; 3] = [8, 5, 6];

    fn features() -> Vec<FeatureMeta> {
        let numerical = |bins: u32, nan_bin: bool| {
            let numeric = bins - u32::from(nan_bin);
            let mut upper_bounds: Vec<f64> = (0..numeric).map(|b| f64::from(b) + 0.5).collect();
            *upper_bounds.last_mut().expect("bins") = f64::INFINITY;
            BinMapper::Numerical {
                upper_bounds,
                nan_bin,
                min: 0.0,
                max: f64::from(numeric),
            }
        };
        let mappers = [
            numerical(BINS[0], false),
            numerical(BINS[1], true),
            BinMapper::Categorical {
                categories: vec![7, 3, 9, 1, 4],
            },
        ];
        let mut offset = 0;
        mappers
            .into_iter()
            .enumerate()
            .map(|(f, bins)| {
                let meta = FeatureMeta {
                    name: format!("f{f}"),
                    bins,
                    num_bins: BINS[f],
                    group: 0,
                    column: f as u32,
                    hist_offset: offset,
                };
                offset += BINS[f];
                meta
            })
            .collect()
    }

    /// The bins of global row `row`, from a fixed hash.
    fn row_bins(row: u64) -> [u8; 3] {
        let h = |stream: u32| quantize::mix32(row as u32 ^ quantize::mix32(stream));
        [0, 1, 2].map(|f| (h(f) % BINS[f as usize]) as u8)
    }

    /// Gradients that depend on the bins, with every third row at g = h = 0 when `zeros`.
    fn row_gradient(row: u64, zeros: bool) -> (f32, f32) {
        if zeros && !row.is_multiple_of(3) {
            return (0.0, 0.0);
        }
        let bins = row_bins(row);
        let noise = (quantize::mix32(row as u32 ^ 0x5bd1_e995) % 1000) as f32 / 1000.0 - 0.5;
        let signal = if bins[0] > 3 { 1.0 } else { -1.0 } + if bins[1] == 4 { 2.0 } else { 0.0 };
        let cat = if bins[2] == 2 || bins[2] == 4 { -1.5 } else { 0.0 };
        (signal + cat + noise, 0.5 + (row % 7) as f32 / 7.0)
    }

    fn dataset(rows: std::ops::Range<u64>) -> BinnedDataset {
        let data = rows.clone().flat_map(|row| {
            let [a, b, c] = row_bins(row);
            [a, b, c, 0]
        });
        let num_rows = (rows.end - rows.start) as usize;
        BinnedDataset {
            num_rows,
            row_offset: rows.start,
            features: features(),
            groups: vec![FeatureGroup {
                features: vec![0, 1, 2],
                row_stride: 4,
                hist_offset: 0,
                total_bins: BINS.iter().sum(),
                data: data.collect(),
            }],
            total_bins: BINS.iter().sum::<u32>() as usize,
            label: vec![0.0; num_rows],
            weight: None,
            query_boundaries: None,
        }
    }

    fn params() -> Params {
        Params {
            num_leaves: 8,
            min_data_in_leaf: 20,
            max_cat_to_onehot: 4,
            min_data_per_group: 10,
            cat_smooth: 1.0,
            // Categorical splits add `cat_l2` to their outputs' denominators; 0 keeps every leaf at `-G / H`.
            cat_l2: 0.0,
            ..Params::default()
        }
    }

    struct Run {
        grown: GrownTree,
        grad: Vec<f32>,
        hess: Vec<f32>,
    }

    /// Grows one tree on global rows `rows` through a `CpuBackend`; quantized when `quant` holds the global maxima.
    fn grow(
        params: &Params,
        data: &BinnedDataset,
        collective: &mut dyn Collective,
        quant: Option<(f64, f64)>,
        zeros: bool,
    ) -> Run {
        let mut backend = CpuBackend::borrowing(data, &[]);
        backend.load(data, &[], &ObjectiveSetup::Regression).expect("load");
        let rows = data.row_offset..data.row_offset + data.num_rows as u64;
        let (grad, hess): (Vec<f32>, Vec<f32>) = rows.map(|row| row_gradient(row, zeros)).unzip();
        backend.set_gradients(&grad, &hess);
        let scale = match quant {
            Some((max_g, max_h)) => {
                let spec = QuantSpec::new(max_g, max_h, None, 4, true, 1, 0);
                backend.quantize(&spec).expect("quantize");
                HistScale {
                    grad: spec.grad_scale,
                    hess: spec.hess_scale,
                }
            }
            None => HistScale::UNIT,
        };
        let local = backend.reset_partition(&BagSpec::All).expect("partition");
        let mut count = [i64::from(local)];
        collective.allreduce_sum_i64(&mut count).expect("count");
        let learner = Learner::new(params, &data.features, data.total_bins, None);
        let mask = vec![true; 3];
        let spec = TreeSpec {
            quantized: quant.is_some(),
            scale,
            feature_mask: &mask,
            root: LeafRange { begin: 0, count: local },
            root_count: count[0],
        };
        let mut grown = learner.grow(&mut backend, collective, spec).expect("grow");
        if let Some((max_g, max_h)) = quant {
            learner
                .renew_leaves(&mut backend, collective, &mut grown, max_g, max_h)
                .expect("renew");
        }
        Run { grown, grad, hess }
    }

    /// The leaf of local row `row` by walking the device tree over the dataset's bins.
    fn leaf_of(data: &BinnedDataset, tree: &DeviceTree, row: usize) -> usize {
        let mut node = 0usize;
        loop {
            let n = &tree.nodes[node];
            let bin = data.groups[n.split.group as usize].data[row * 4 + n.split.column as usize];
            let next = if n.split.goes_left(bin) { n.left } else { n.right };
            if next < 0 {
                return !next as usize;
            }
            node = next as usize;
        }
    }

    #[test]
    fn leaves_match_the_partition_and_their_sums() {
        let params = params();
        let data = dataset(0..3000);
        let run = grow(&params, &data, &mut LocalCollective, None, false);
        let tree = &run.grown.tree;
        assert_eq!(tree.num_leaves, 8, "grows to num_leaves");
        assert!(tree.split_feature.contains(&2), "the categorical feature splits");
        let device = run.grown.device_tree();
        let mut members = vec![Vec::new(); tree.num_leaves];
        for row in 0..data.num_rows {
            members[leaf_of(&data, &device, row)].push(row);
        }
        for (leaf, rows) in members.iter().enumerate() {
            assert_eq!(
                rows.len() as u32,
                run.grown.leaves[leaf].count,
                "leaf {leaf}: partition count"
            );
            assert_eq!(rows.len() as i64, tree.leaf_count[leaf], "leaf {leaf}: tree count");
            assert!(rows.len() >= params.min_data_in_leaf, "leaf {leaf}: min_data_in_leaf");
            let g: f64 = rows.iter().map(|&r| f64::from(run.grad[r])).sum();
            let h: f64 = rows.iter().map(|&r| f64::from(run.hess[r])).sum();
            let expected = -g / h;
            assert!(
                (tree.leaf_value[leaf] - expected).abs() < 1e-9 * expected.abs().max(1.0),
                "leaf {leaf}: value {} vs {expected}",
                tree.leaf_value[leaf]
            );
            assert!((tree.leaf_weight[leaf] - h).abs() < 1e-9 * h, "leaf {leaf}: weight");
        }
    }

    #[test]
    fn max_depth_limits_growth() {
        let params = Params {
            max_depth: 2,
            ..params()
        };
        let run = grow(&params, &dataset(0..3000), &mut LocalCollective, None, false);
        assert_eq!(run.grown.tree.num_leaves, 4);
        assert_eq!(run.grown.tree.max_depth(), 2);
    }

    /// Quantized maxima of the gradients over all rows.
    fn maxima(rows: std::ops::Range<u64>, zeros: bool) -> (f64, f64) {
        rows.map(|row| row_gradient(row, zeros))
            .fold((0.0, 0.0), |(g, h), (rg, rh)| {
                (f64::max(g, f64::from(rg.abs())), f64::max(h, f64::from(rh)))
            })
    }

    #[test]
    fn two_ranks_grow_the_tree_of_one() {
        let params = params();
        let quant = Some(maxima(0..4000, false));
        let single = grow(&params, &dataset(0..4000), &mut LocalCollective, quant, false);
        assert_eq!(single.grown.tree.num_leaves, 8);
        let ranks = run_ranks(2, |rank, collective| {
            let rows = if rank == 0 { 0..1500 } else { 1500..4000 };
            grow(&params, &dataset(rows), collective, quant, false).grown.tree
        });
        for (rank, tree) in ranks.iter().enumerate() {
            assert_eq!(tree, &single.grown.tree, "rank {rank}");
        }
    }

    #[test]
    fn zero_gradient_rows_never_make_empty_leaves() {
        let params = Params {
            min_data_in_leaf: 0,
            min_sum_hessian_in_leaf: 0.0,
            num_leaves: 31,
            ..params()
        };
        let quant = Some(maxima(0..6000, true));
        let run = grow(&params, &dataset(0..6000), &mut LocalCollective, quant, true);
        let tree = &run.grown.tree;
        assert!(tree.num_leaves > 8, "grows past the signal: {} leaves", tree.num_leaves);
        for leaf in 0..tree.num_leaves {
            assert!(tree.leaf_count[leaf] > 0, "leaf {leaf} is empty");
            assert!(tree.leaf_value[leaf].is_finite(), "leaf {leaf} value");
        }
    }

    /// A backend whose partition sends every row right.
    struct NothingLeft<'a>(CpuBackend<'a>);

    impl Backend for NothingLeft<'_> {
        fn name(&self) -> String {
            self.0.name()
        }

        fn load(&mut self, train: &BinnedDataset, valid: &[&BinnedDataset], objective: &ObjectiveSetup) -> Result<()> {
            self.0.load(train, valid, objective)
        }

        fn init_scores(&mut self, dataset: usize, value: f64) -> Result<()> {
            self.0.init_scores(dataset, value)
        }

        fn compute_gradients(&mut self) -> Result<crate::backend::GradientStats> {
            self.0.compute_gradients()
        }

        fn quantize(&mut self, spec: &QuantSpec) -> Result<()> {
            self.0.quantize(spec)
        }

        fn reset_partition(&mut self, bag: &BagSpec) -> Result<u32> {
            self.0.reset_partition(bag)
        }

        fn histogram(&mut self, leaf: LeafRange, quantized: bool, out: &mut [f64]) -> Result<()> {
            self.0.histogram(leaf, quantized, out)
        }

        fn partition(&mut self, _leaf: LeafRange, _split: &BinSplit) -> Result<u32> {
            Ok(0)
        }

        fn leaf_sums(&mut self, leaves: &[LeafRange], grad_scale: f32, hess_scale: f32) -> Result<Vec<[i64; 2]>> {
            self.0.leaf_sums(leaves, grad_scale, hess_scale)
        }

        fn add_tree(&mut self, tree: &DeviceTree) -> Result<()> {
            self.0.add_tree(tree)
        }

        fn scores(&mut self, dataset: usize) -> Result<Vec<f32>> {
            self.0.scores(dataset)
        }
    }

    #[test]
    fn an_empty_child_is_an_internal_error() {
        let params = params();
        let data = dataset(0..3000);
        let mut backend = NothingLeft(CpuBackend::borrowing(&data, &[]));
        backend.load(&data, &[], &ObjectiveSetup::Regression).expect("load");
        let (grad, hess): (Vec<f32>, Vec<f32>) = (0..3000).map(|row| row_gradient(row, false)).unzip();
        backend.0.set_gradients(&grad, &hess);
        let local = backend.reset_partition(&BagSpec::All).expect("partition");
        let mask = vec![true; 3];
        let spec = TreeSpec {
            quantized: false,
            scale: HistScale::UNIT,
            feature_mask: &mask,
            root: LeafRange { begin: 0, count: local },
            root_count: i64::from(local),
        };
        let learner = Learner::new(&params, &data.features, data.total_bins, None);
        let error = learner
            .grow(&mut backend, &mut LocalCollective, spec)
            .expect_err("empty child");
        assert_eq!(error.kind(), "internal", "{error}");
    }
}
