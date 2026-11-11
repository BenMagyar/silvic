//! The device operations the learner drives. A backend holds the training set (dataset 0) and the validation sets
//! (1..) in its own memory, their f32 scores, the training gradients and a partition array of training row indices
//! in which every leaf of the tree being grown is a contiguous range.
//!
//! Histograms come back in the host layout (`2 * total_bins` f64, `[g, h]` per bin): integer counts of quantized
//! units in quantized mode, real sums in float mode. Everything a backend computes from quantized inputs is integer
//! and exact, so every backend returns the same numbers for the same inputs.

mod cpu;

pub use cpu::CpuBackend;

use crate::data::BinnedDataset;
use crate::error::Result;
use crate::quantize::QuantSpec;

/// A leaf's rows: `partition[begin..begin + count]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct LeafRange {
    pub begin: u32,
    pub count: u32,
}

/// A split as the device applies it: rows whose bin of the feature at (`group`, `column`) is set in `left_bins` go
/// left. Numerical splits set bins `0..=threshold` and, when missing values go left, the NaN bin.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BinSplit {
    pub group: u32,
    pub column: u32,
    pub left_bins: [u32; 8],
}

impl BinSplit {
    #[inline]
    pub fn goes_left(&self, bin: u8) -> bool {
        (self.left_bins[usize::from(bin) / 32] >> (bin % 32)) & 1 == 1
    }
}

/// One node of a tree in device form. Children `>= 0` are nodes, `< 0` are `!leaf`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceNode {
    pub split: BinSplit,
    pub left: i32,
    pub right: i32,
}

/// A trained tree as the device adds it to scores: `leaf_values` are already shrunk and exclude the first tree's bias.
#[derive(Debug, Clone, PartialEq)]
pub struct DeviceTree {
    /// Empty for a single-leaf tree.
    pub nodes: Vec<DeviceNode>,
    pub leaf_values: Vec<f32>,
}

/// Objective-specific inputs the gradient computation needs beyond labels, weights and query boundaries.
#[derive(Debug, Clone, PartialEq)]
pub enum ObjectiveSetup {
    Regression,
    Binary {
        sigmoid: f64,
    },
    LambdaRank {
        sigmoid: f64,
        norm: bool,
        truncation_level: u32,
        label_gain: Vec<f64>,
        /// Per training query (LightGBM `inverse_max_dcgs_`).
        inverse_max_dcg: Vec<f64>,
    },
}

/// Local maxima for the quantization scales; the learner reduces them over ranks.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct GradientStats {
    pub max_abs_grad: f64,
    pub max_hess: f64,
}

/// Which training rows the next tree sees.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BagSpec {
    All,
    /// Rows with `quantize::in_bag(seed, round, global_row, fraction)`.
    Bernoulli {
        seed: u32,
        round: u32,
        fraction: f64,
    },
}

/// A child of a split leaf.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Side {
    Left,
    Right,
}

impl Side {
    /// The child's rows after `leaf` was partitioned with `left` rows going left.
    pub fn range(self, leaf: LeafRange, left: u32) -> LeafRange {
        match self {
            Side::Left => LeafRange {
                begin: leaf.begin,
                count: left,
            },
            Side::Right => LeafRange {
                begin: leaf.begin + left,
                count: leaf.count - left,
            },
        }
    }
}

pub trait Backend {
    /// Human-readable device name for logs (`cpu (14 threads)`, `Apple M4 Pro (metal)`).
    fn name(&self) -> String;

    /// The pool host-side parallel work runs on when `num_threads` is set; backends without host-side parallelism
    /// ignore it. Called before `load`.
    fn set_thread_pool(&mut self, pool: std::sync::Arc<rayon::ThreadPool>) {
        let _ = pool;
    }

    /// Takes the training set and validation sets; called once, before anything else.
    fn load(&mut self, train: &BinnedDataset, valid: &[&BinnedDataset], objective: &ObjectiveSetup) -> Result<()>;

    /// Sets every score of `dataset` to `value`.
    fn init_scores(&mut self, dataset: usize, value: f64) -> Result<()>;

    /// Computes f32 gradients and hessians of the training scores; returns their local maxima.
    fn compute_gradients(&mut self) -> Result<GradientStats>;

    /// Quantizes the gradients and hessians with `spec` (`quantize::quantize_row`, keyed by global row id).
    fn quantize(&mut self, spec: &QuantSpec) -> Result<()>;

    /// Refills the partition with the bagged training rows in ascending order; returns their count.
    fn reset_partition(&mut self, bag: &BagSpec) -> Result<u32>;

    /// Histograms from now on only need the bins of the features with `used[feature]` (LightGBM `is_feature_used`:
    /// the tree's feature mask and the parent's splittable features); the other bins may hold anything. Every rank
    /// passes the same mask. Backends may ignore it.
    fn set_histogram_features(&mut self, used: &[bool]) {
        let _ = used;
    }

    /// Histogram of the rows of `leaf` into `out` (`2 * total_bins`, overwritten): quantized units when `quantized`,
    /// else float sums of gradients and hessians.
    fn histogram(&mut self, leaf: LeafRange, quantized: bool, out: &mut [f64]) -> Result<()>;

    /// Stable in-place partition of `leaf`'s rows by `split`; returns the number of rows going left.
    fn partition(&mut self, leaf: LeafRange, split: &BinSplit) -> Result<u32>;

    /// `partition`, then the histogram of the `side` child into `out`; returns the left count. A device overrides it
    /// to run both without waiting for the count in between: the side is chosen before partitioning (from the
    /// split's estimated counts, the same on every rank), so the histogram's row range can stay on the device.
    fn partition_and_histogram(
        &mut self,
        leaf: LeafRange,
        split: &BinSplit,
        side: Side,
        quantized: bool,
        out: &mut [f64],
    ) -> Result<u32> {
        let left = self.partition(leaf, split)?;
        self.histogram(side.range(leaf, left), quantized, out)?;
        Ok(left)
    }

    /// Exact fixed-point sums `[sum rint(g * scale), sum rint(h * scale)]` of each leaf's float gradients
    /// (`quantize::to_fixed`), for leaf renewal.
    fn leaf_sums(&mut self, leaves: &[LeafRange], grad_scale: f32, hess_scale: f32) -> Result<Vec<[i64; 2]>>;

    /// Adds `tree` to the scores of every dataset, traversing binned rows.
    fn add_tree(&mut self, tree: &DeviceTree) -> Result<()>;

    /// The current scores of `dataset`.
    fn scores(&mut self, dataset: usize) -> Result<Vec<f32>>;
}
