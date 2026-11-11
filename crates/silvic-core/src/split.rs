//! Best-split search over one leaf's histogram: a port of LightGBM 3.3.5 `FeatureHistogram`
//! (`src/treelearner/feature_histogram.hpp`) without monotone constraints, extra trees, path smoothing or
//! `max_delta_step`, which silvic rejects at parameter parsing.
//!
//! Histograms hold `[g, h]` per bin in histogram units: quantized units in quantized mode, which a [`HistScale`]
//! turns into real sums as LightGBM 4's `FindBestThresholdInt` does with its gradient and hessian scales, and real
//! sums in float mode (scale 1). Scans accumulate units, which are exact integers in quantized mode, and convert to
//! real sums for gains and outputs, so the child sums handed to the learner stay exact.
//!
//! Every bin is stored, LightGBM's `offset` (most-frequent bin 0 left out of the histogram) does not exist, so the
//! scans below are LightGBM's with `offset = 0`:
//! - `FindBestThresholdSequentially` with `REVERSE` (right side grows from the top bin down, missing values go left)
//!   and, when the feature has a NaN bin and more than two bins, the forward scan with `NA_AS_MISSING` (missing
//!   values go right; its last candidate is NaN against everything else).
//! - `FindBestThresholdCategoricalInner`: one-hot below `max_cat_to_onehot`, else the `sum_g / (sum_h + cat_smooth)`
//!   sorted scan from both ends. Bin 0 ("other") never goes left.
//!
//! A numerical scan runs in passes over a stack buffer: running sums (the only loop-carried work), then every
//! candidate's gain two at a time (no dependency from one candidate to the next), then the first maximum. Each value
//! is the same IEEE operation on the same operands as LightGBM's single loop, so splits are bit for bit the same. A
//! leaf's features are searched on rayon in contiguous ranges of about `SEARCH_CHUNK_WORK` bins.
//!
//! Two guards go beyond LightGBM, which aborts (`Check failed: left_count > 0`) on leaves whose rows all have zero
//! hessian, as lambdarank queries without a relevant item do: a leaf with a zero hessian sum has no split, and a
//! candidate whose child has no estimated rows or a zero hessian sum is rejected whatever `min_data_in_leaf` and
//! `min_sum_hessian_in_leaf` are.

use std::cmp::Ordering;

use rayon::prelude::*;

use crate::config::Params;
use crate::data::{BinMapper, FeatureMeta};
use crate::quantize;

/// LightGBM `kEpsilon`: `1e-15f`, a float literal widened to double.
pub const K_EPSILON: f64 = 1e-15_f32 as f64;
/// LightGBM `kMinScore`.
const MIN_SCORE: f64 = f64::NEG_INFINITY;
/// Random-number stream of the per-tree feature sample; `quantize` uses 0-2.
pub const STREAM_FEATURE_FRACTION: u32 = 3;
/// Bins a feature may have (bins are bytes).
const MAX_BINS: usize = 256;
/// Bin visits per parallel work item of the split search: enough to amortize a rayon task, small enough that a
/// bench-shaped leaf (136 features, about 22k visits) spreads over the cores.
const SEARCH_CHUNK_WORK: usize = 2048;

/// The split-finding parameters (LightGBM `Config` fields read by `FeatureHistogram`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SplitConfig {
    pub min_data_in_leaf: i64,
    pub min_sum_hessian_in_leaf: f64,
    pub lambda_l1: f64,
    pub lambda_l2: f64,
    pub min_gain_to_split: f64,
    pub max_cat_to_onehot: usize,
    pub max_cat_threshold: usize,
    pub cat_l2: f64,
    pub cat_smooth: f64,
    pub min_data_per_group: i64,
}

impl SplitConfig {
    pub fn new(params: &Params) -> SplitConfig {
        SplitConfig {
            min_data_in_leaf: params.min_data_in_leaf as i64,
            min_sum_hessian_in_leaf: params.min_sum_hessian_in_leaf,
            lambda_l1: params.lambda_l1,
            lambda_l2: params.lambda_l2,
            min_gain_to_split: params.min_gain_to_split,
            max_cat_to_onehot: params.max_cat_to_onehot,
            max_cat_threshold: params.max_cat_threshold,
            cat_l2: params.cat_l2,
            cat_smooth: params.cat_smooth,
            min_data_per_group: params.min_data_per_group as i64,
        }
    }
}

/// Real sum per histogram unit: the iteration's `grad_scale`/`hess_scale` in quantized mode, 1 in float mode.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HistScale {
    pub grad: f64,
    pub hess: f64,
}

impl HistScale {
    pub const UNIT: HistScale = HistScale { grad: 1.0, hess: 1.0 };
}

/// A leaf's or child's gradient and hessian sums in histogram units, and its row count over every rank.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct LeafSums {
    pub grad: f64,
    pub hess: f64,
    pub count: i64,
}

impl LeafSums {
    /// Sums of the bins of one feature, which every row of the leaf falls in exactly once.
    pub fn from_feature_hist(feature_hist: &[f64], count: i64) -> LeafSums {
        let (grad, hess) = feature_hist
            .as_chunks::<2>()
            .0
            .iter()
            .fold((0.0, 0.0), |(g, h), bin| (g + bin[0], h + bin[1]));
        LeafSums { grad, hess, count }
    }
}

/// Which bins of the split feature go left.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SplitRule {
    /// Bins `0..=threshold` go left; the NaN bin, when the feature has one, goes left iff `default_left`.
    Numerical { threshold: u32, default_left: bool },
    /// These bins (all in `1..num_bins`) go left, in LightGBM's `cat_threshold` order; every other bin goes right.
    Categorical { bins: Vec<u32> },
}

/// LightGBM `SplitInfo`: the best split of one leaf on one feature.
#[derive(Debug, Clone, PartialEq)]
pub struct SplitInfo {
    pub feature: usize,
    /// `best_gain - min_gain_shift`, where `min_gain_shift = parent_gain + min_gain_to_split`; positive.
    pub gain: f64,
    pub rule: SplitRule,
    /// Children sums in histogram units; counts are LightGBM's estimate from hessians.
    pub left: LeafSums,
    pub right: LeafSums,
    pub left_output: f64,
    pub right_output: f64,
}

/// A leaf's best split over the evaluated features, and which evaluated features had any valid candidate
/// (LightGBM `FeatureHistogram::is_splittable`, inherited by both children).
#[derive(Debug, Clone, PartialEq)]
pub struct LeafSplit {
    pub best: Option<SplitInfo>,
    /// Indexed by feature; `false` for features not evaluated.
    pub splittable: Vec<bool>,
}

/// LightGBM `FeatureHistogram::ThresholdL1`.
#[inline]
fn threshold_l1(s: f64, l1: f64) -> f64 {
    let reg = (s.abs() - l1).max(0.0);
    sign(s) * reg
}

/// LightGBM `Common::Sign`: 0 for 0.
#[inline]
fn sign(x: f64) -> f64 {
    f64::from(i8::from(x > 0.0) - i8::from(x < 0.0))
}

/// LightGBM `GetLeafGain` without `max_delta_step` and smoothing. `ThresholdL1(G)^2` is computed as the square of
/// its magnitude: the sign is exactly ±1 (or 0 with a 0 magnitude), so the square is bit for bit the same.
#[inline]
pub fn leaf_gain(sum_grad: f64, sum_hess: f64, l1: f64, l2: f64) -> f64 {
    let g = (sum_grad.abs() - l1).max(0.0);
    g * g / (sum_hess + l2)
}

/// `leaf_gain` of both children, summed: the two lanes in one vector division where the target has one, bit for
/// bit the scalar result (the same IEEE operations per lane, and `max` as IEEE `maxNum` like `f64::max`).
#[inline(always)]
fn split_gain(grad: [f64; 2], hess: [f64; 2], l1: f64, l2: f64) -> f64 {
    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::{
            vabsq_f64, vaddq_f64, vdivq_f64, vdupq_n_f64, vgetq_lane_f64, vld1q_f64, vmaxnmq_f64, vmulq_f64, vsubq_f64,
        };
        // SAFETY: NEON is part of the aarch64 baseline, and the loads read two-element arrays.
        unsafe {
            let g = vld1q_f64(grad.as_ptr());
            let h = vld1q_f64(hess.as_ptr());
            let r = vmaxnmq_f64(vsubq_f64(vabsq_f64(g), vdupq_n_f64(l1)), vdupq_n_f64(0.0));
            let q = vdivq_f64(vmulq_f64(r, r), vaddq_f64(h, vdupq_n_f64(l2)));
            vgetq_lane_f64::<0>(q) + vgetq_lane_f64::<1>(q)
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        leaf_gain(grad[0], hess[0], l1, l2) + leaf_gain(grad[1], hess[1], l1, l2)
    }
}

/// LightGBM `CalculateSplittedLeafOutput` without `max_delta_step` and smoothing.
#[inline]
pub fn leaf_output(sum_grad: f64, sum_hess: f64, l1: f64, l2: f64) -> f64 {
    -threshold_l1(sum_grad, l1) / (sum_hess + l2)
}

/// LightGBM `Common::RoundInt`.
#[inline]
fn round_int(x: f64) -> i64 {
    (x + 0.5) as i64
}

/// One leaf as `FindBestThreshold` sees it.
struct LeafCtx<'a> {
    cfg: &'a SplitConfig,
    scale: HistScale,
    units: LeafSums,
    /// Real sums as LightGBM passes them in: the hessian plus `2 * kEpsilon`.
    sum_grad: f64,
    sum_hess: f64,
    num_data: i64,
    cnt_factor: f64,
    /// `min_gain_shift` of the numerical scans, also the categorical one's (with the plain `lambda_l2`).
    min_gain_shift: f64,
}

impl<'a> LeafCtx<'a> {
    /// `None` when the leaf has no hessian, so no count estimate exists and no split is possible.
    fn new(cfg: &'a SplitConfig, scale: HistScale, units: LeafSums) -> Option<LeafCtx<'a>> {
        if units.hess.is_nan() || units.hess <= 0.0 || units.count <= 0 {
            return None;
        }
        let sum_grad = units.grad * scale.grad;
        let sum_hess = units.hess * scale.hess + 2.0 * K_EPSILON;
        let gain_shift = leaf_gain(sum_grad, sum_hess, cfg.lambda_l1, cfg.lambda_l2);
        Some(LeafCtx {
            cfg,
            scale,
            units,
            sum_grad,
            sum_hess,
            num_data: units.count,
            cnt_factor: units.count as f64 / sum_hess,
            min_gain_shift: gain_shift + cfg.min_gain_to_split,
        })
    }

    #[inline]
    fn bin_count(&self, hess_units: f64) -> i64 {
        round_int(hess_units * self.scale.hess * self.cnt_factor)
    }

    /// The side whose constraint fails first: LightGBM `continue`s while the growing side is too small and
    /// `break`s once the shrinking side is. Zero rows or zero hessian on either side always fail.
    #[inline]
    fn check(&self, grow: &Side, shrink_count: i64, shrink_hess: f64) -> Check {
        if grow.count < self.cfg.min_data_in_leaf
            || grow.count <= 0
            || grow.hess < self.cfg.min_sum_hessian_in_leaf
            || grow.units[1] <= 0.0
        {
            return Check::Continue;
        }
        if shrink_count < self.cfg.min_data_in_leaf || shrink_count <= 0 {
            return Check::Break;
        }
        if shrink_hess < self.cfg.min_sum_hessian_in_leaf || self.units.hess - grow.units[1] <= 0.0 {
            return Check::Break;
        }
        Check::Evaluate
    }
}

enum Check {
    Continue,
    Break,
    Evaluate,
}

/// The side of a scan that accumulates bins.
#[derive(Clone, Copy, Default)]
struct Side {
    units: [f64; 2],
    count: i64,
    /// Real hessian, starting at `kEpsilon` as in LightGBM.
    hess: f64,
}

impl Side {
    fn new() -> Side {
        Side {
            units: [0.0, 0.0],
            count: 0,
            hess: K_EPSILON,
        }
    }

    #[inline]
    fn add(&mut self, ctx: &LeafCtx<'_>, bin: &[f64]) {
        self.units[0] += bin[0];
        self.units[1] += bin[1];
        self.count += ctx.bin_count(bin[1]);
        self.hess = K_EPSILON + self.units[1] * ctx.scale.hess;
    }

    fn grad(&self, ctx: &LeafCtx<'_>) -> f64 {
        self.units[0] * ctx.scale.grad
    }
}

/// The best candidate of a scan, with the left child's real sums as LightGBM keeps them.
#[derive(Clone, Copy)]
struct Best {
    gain: f64,
    threshold: u32,
    left_grad: f64,
    left_hess: f64,
    left_count: i64,
    left_units: [f64; 2],
}

/// A feature's best candidate, kept `Copy` until the leaf's winner becomes a [`SplitInfo`].
#[derive(Clone, Copy)]
struct Candidate {
    feature: usize,
    /// `best.gain - min_gain_shift`, as `SplitInfo::gain`.
    gain: f64,
    best: Best,
    kind: CandidateKind,
}

#[derive(Clone, Copy)]
enum CandidateKind {
    Numerical {
        default_left: bool,
    },
    OneHot,
    /// The sorted categorical scan, from the low (`forward`) or the high end of the ctr order.
    Sorted {
        forward: bool,
    },
}

/// The result of one feature: whether any candidate beat `min_gain_shift`, and the best one.
struct FeatureResult {
    splittable: bool,
    best: Option<Candidate>,
}

impl FeatureResult {
    const NONE: FeatureResult = FeatureResult {
        splittable: false,
        best: None,
    };
}

/// The growing side of a numerical scan at its best candidate.
#[derive(Clone, Copy)]
struct Grown {
    /// The last bin added.
    bin: usize,
    units: [f64; 2],
    count: i64,
    gain: f64,
}

/// A leaf's constants as the candidate evaluation reads them.
struct EvalConsts {
    /// `max(min_data_in_leaf, 1)`: a side needs at least this many rows (and never none).
    min_count: i64,
    min_hess: f64,
    l1: f64,
    l2: f64,
    sum_grad: f64,
    sum_hess: f64,
    grad_scale: f64,
    hess_scale: f64,
}

impl EvalConsts {
    fn new(ctx: &LeafCtx<'_>) -> EvalConsts {
        let cfg = ctx.cfg;
        EvalConsts {
            min_count: cfg.min_data_in_leaf.max(1),
            min_hess: cfg.min_sum_hessian_in_leaf,
            l1: cfg.lambda_l1,
            l2: cfg.lambda_l2,
            sum_grad: ctx.sum_grad,
            sum_hess: ctx.sum_hess,
            grad_scale: ctx.scale.grad,
            hess_scale: ctx.scale.hess,
        }
    }
}

/// One scan's candidates in scan order, computed in passes so the per-candidate work (the bulk of it, two
/// divisions included) has no loop-carried dependency and vectorizes. Every value is the same IEEE operation on
/// the same operands as LightGBM's single loop, so the result is bit for bit the same.
struct ScanBuffer {
    bin: [u16; MAX_BINS],
    /// The growing side's running sums in histogram units, and its estimated row count.
    grad: [f64; MAX_BINS],
    hess: [f64; MAX_BINS],
    count: [i64; MAX_BINS],
    /// The candidate's gain, `-inf` when the growing side is too small.
    gain: [f64; MAX_BINS],
}

impl ScanBuffer {
    /// The loop both `FindBestThresholdSequentially` directions share, over the `len` candidates `fill` stored
    /// (up to where the scan stops): candidates the growing side is too small for are skipped, and the best is the
    /// first strict maximum above `min_gain_shift`.
    fn complete(&mut self, ctx: &LeafCtx<'_>, len: usize, splittable: &mut bool) -> Option<Grown> {
        self.evaluate(ctx, len);
        let best = self.best(ctx, len);
        *splittable |= best.is_some();
        best
    }

    fn new() -> ScanBuffer {
        ScanBuffer {
            bin: [0; MAX_BINS],
            grad: [0.0; MAX_BINS],
            hess: [0.0; MAX_BINS],
            count: [0; MAX_BINS],
            gain: [0.0; MAX_BINS],
        }
    }

    /// Running sums (`Side::add`) in scan order, up to (excluding) the first candidate where the scan stops: the
    /// other side is too small while the growing side is not (LightGBM's `break`). Returns the candidates stored.
    #[inline(always)]
    fn fill<'h>(&mut self, ctx: &LeafCtx<'_>, bins: impl Iterator<Item = (usize, &'h [f64; 2])>) -> usize {
        let cfg = ctx.cfg;
        let (min_count, min_hess) = (cfg.min_data_in_leaf.max(1), cfg.min_sum_hessian_in_leaf);
        let (num_data, sum_hess, units_hess) = (ctx.num_data, ctx.sum_hess, ctx.units.hess);
        let (hess_scale, cnt_factor) = (ctx.scale.hess, ctx.cnt_factor);
        let (mut grad, mut units_h, mut count) = (0.0f64, 0.0f64, 0i64);
        let mut len = 0;
        for (k, (t, bin)) in bins.take(MAX_BINS).enumerate() {
            grad += bin[0];
            units_h += bin[1];
            count += round_int(bin[1] * hess_scale * cnt_factor);
            let hess = K_EPSILON + units_h * hess_scale;
            let skip = count < min_count || hess < min_hess || units_h <= 0.0;
            let other_small = num_data - count < min_count || sum_hess - hess < min_hess || units_hess - units_h <= 0.0;
            if other_small && !skip {
                break;
            }
            self.bin[k] = t as u16;
            self.grad[k] = grad;
            self.hess[k] = units_h;
            self.count[k] = count;
            len = k + 1;
        }
        len
    }

    /// Each candidate's gain, `-inf` where the growing side is too small.
    #[inline(always)]
    fn evaluate(&mut self, ctx: &LeafCtx<'_>, len: usize) {
        let c = EvalConsts::new(ctx);
        #[cfg(target_arch = "aarch64")]
        let first = self.evaluate_pairs(&c, len);
        #[cfg(not(target_arch = "aarch64"))]
        let first = 0;
        for k in first..len {
            let (units_g, units_h, count) = (self.grad[k], self.hess[k], self.count[k]);
            let hess = K_EPSILON + units_h * c.hess_scale;
            let skip = count < c.min_count || hess < c.min_hess || units_h <= 0.0;
            let grad = units_g * c.grad_scale;
            let gain = split_gain([c.sum_grad - grad, grad], [c.sum_hess - hess, hess], c.l1, c.l2);
            self.gain[k] = if skip { f64::NEG_INFINITY } else { gain };
        }
    }

    /// The scalar loop of `evaluate` two candidates per NEON vector, operation for operation (no fused
    /// multiply-adds), so every lane is bit for bit the scalar result; returns the candidates done.
    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    fn evaluate_pairs(&mut self, c: &EvalConsts, len: usize) -> usize {
        use std::arch::aarch64::*;
        let pairs = len / 2;
        // SAFETY: NEON is part of the aarch64 baseline; every load and store touches lanes `k, k + 1 < len`, inside
        // the buffers' `MAX_BINS` entries.
        unsafe {
            let dup = vdupq_n_f64;
            let (eps, hess_scale, grad_scale) = (dup(K_EPSILON), dup(c.hess_scale), dup(c.grad_scale));
            let (min_hess, sum_hess, sum_grad) = (dup(c.min_hess), dup(c.sum_hess), dup(c.sum_grad));
            let (l1, l2, zero, neg_inf) = (dup(c.l1), dup(c.l2), dup(0.0), dup(f64::NEG_INFINITY));
            let min_count = vdupq_n_s64(c.min_count);
            let leaf = |g: float64x2_t, h: float64x2_t| {
                let r = vmaxnmq_f64(vsubq_f64(vabsq_f64(g), l1), zero);
                vdivq_f64(vmulq_f64(r, r), vaddq_f64(h, l2))
            };
            for pair in 0..pairs {
                let k = 2 * pair;
                let units_g = vld1q_f64(self.grad.as_ptr().add(k));
                let units_h = vld1q_f64(self.hess.as_ptr().add(k));
                let count = vld1q_s64(self.count.as_ptr().add(k));
                let hess = vaddq_f64(eps, vmulq_f64(units_h, hess_scale));
                let skip = vorrq_u64(
                    vcltq_s64(count, min_count),
                    vorrq_u64(vcltq_f64(hess, min_hess), vclezq_f64(units_h)),
                );
                let grad = vmulq_f64(units_g, grad_scale);
                let gain = vaddq_f64(
                    leaf(vsubq_f64(sum_grad, grad), vsubq_f64(sum_hess, hess)),
                    leaf(grad, hess),
                );
                vst1q_f64(self.gain.as_mut_ptr().add(k), vbslq_f64(skip, neg_inf, gain));
            }
        }
        2 * pairs
    }

    /// The first strict maximum above `min_gain_shift` before the stop (NaN never wins, as LightGBM's
    /// `current_gain <= min_gain_shift` skip): the maximum, then its first occurrence, which is what a sequential
    /// `gain > best` scan keeps, without its compare-select chain from one candidate to the next.
    #[inline(always)]
    fn best(&self, ctx: &LeafCtx<'_>, stop: usize) -> Option<Grown> {
        let gains = &self.gain[..stop];
        let mut lanes = [f64::NEG_INFINITY; 4];
        let quads = gains.as_chunks::<4>();
        for quad in quads.0 {
            for (lane, &gain) in lanes.iter_mut().zip(quad) {
                *lane = lane.max(gain);
            }
        }
        let max = quads
            .1
            .iter()
            .fold(lanes[0].max(lanes[1]).max(lanes[2].max(lanes[3])), |m, &g| m.max(g));
        if max.partial_cmp(&ctx.min_gain_shift) != Some(Ordering::Greater) {
            return None;
        }
        let k = gains.iter().position(|&g| g == max)?;
        Some(Grown {
            bin: usize::from(self.bin[k]),
            units: [self.grad[k], self.hess[k]],
            count: self.count[k],
            gain: gains[k],
        })
    }
}

/// A feature's histogram as `[g, h]` pairs.
fn pairs(hist: &[f64]) -> &[[f64; 2]] {
    hist.as_chunks::<2>().0
}

/// The scans of one feature: the reverse scan (bins above the threshold go right) and the forward one (bins up to
/// the threshold go left), each in its own buffer.
struct Scans {
    reverse: ScanBuffer,
    forward: ScanBuffer,
}

impl Scans {
    fn new() -> Scans {
        Scans {
            reverse: ScanBuffer::new(),
            forward: ScanBuffer::new(),
        }
    }
}

/// LightGBM `FindBestThresholdSequentially<REVERSE=true>`'s result: the right side grew from the top bin down, and
/// everything else (bin 0, and the NaN bin when the scan excluded it) goes left.
fn reverse_best(ctx: &LeafCtx<'_>, right: Grown) -> Best {
    let right_hess = K_EPSILON + right.units[1] * ctx.scale.hess;
    Best {
        gain: right.gain,
        threshold: (right.bin - 1) as u32,
        left_grad: ctx.sum_grad - right.units[0] * ctx.scale.grad,
        left_hess: ctx.sum_hess - right_hess,
        left_count: ctx.num_data - right.count,
        left_units: [ctx.units.grad - right.units[0], ctx.units.hess - right.units[1]],
    }
}

/// LightGBM `FindBestThresholdSequentially<REVERSE=false, NA_AS_MISSING=true>` with `offset = 0`: bins
/// `0..=threshold` go left, everything else (the NaN bin included) goes right.
fn forward_best(ctx: &LeafCtx<'_>, left: Grown) -> Best {
    Best {
        gain: left.gain,
        threshold: left.bin as u32,
        left_grad: left.units[0] * ctx.scale.grad,
        left_hess: K_EPSILON + left.units[1] * ctx.scale.hess,
        left_count: left.count,
        left_units: left.units,
    }
}

/// LightGBM `FuncForNumrical` with `MissingType::NaN` or `None`: the scans a feature gets and the side missing
/// values take in each.
fn find_numerical(ctx: &LeafCtx<'_>, scans: &mut Scans, feature: usize, hist: &[f64], nan_bin: bool) -> FeatureResult {
    let bins = pairs(hist);
    let num_bins = bins.len();
    let mut splittable = false;
    let mut result: Option<(Best, bool)> = None;
    if nan_bin && num_bins > 2 {
        let last = num_bins - 2;
        let reverse_len = scans
            .reverse
            .fill(ctx, bins[1..last + 1].iter().enumerate().rev().map(|(i, p)| (i + 1, p)));
        let forward_len = scans.forward.fill(ctx, bins[..last + 1].iter().enumerate());
        let mut output_gain = MIN_SCORE;
        if let Some(right) = scans.reverse.complete(ctx, reverse_len, &mut splittable) {
            let best = reverse_best(ctx, right);
            output_gain = best.gain - ctx.min_gain_shift;
            result = Some((best, true));
        }
        // The second scan replaces the first only when strictly better, compared as LightGBM does through the
        // already shifted output gain.
        if let Some(left) = scans.forward.complete(ctx, forward_len, &mut splittable) {
            let best = forward_best(ctx, left);
            if best.gain > output_gain + ctx.min_gain_shift {
                result = Some((best, false));
            }
        }
    } else if num_bins >= 2 {
        // With a NaN bin but no other split point, the reverse scan puts NaN on the right on its own.
        let scan = &mut scans.reverse;
        let len = scan.fill(ctx, bins[1..].iter().enumerate().rev().map(|(i, p)| (i + 1, p)));
        result = scan
            .complete(ctx, len, &mut splittable)
            .map(|right| (reverse_best(ctx, right), !nan_bin));
    }
    let best =
        result.map(|(best, default_left)| candidate(ctx, feature, best, CandidateKind::Numerical { default_left }));
    FeatureResult { splittable, best }
}

fn candidate(ctx: &LeafCtx<'_>, feature: usize, best: Best, kind: CandidateKind) -> Candidate {
    Candidate {
        feature,
        gain: best.gain - ctx.min_gain_shift,
        best,
        kind,
    }
}

/// The `SplitInfo` LightGBM fills from a scan's best candidate.
fn finish(ctx: &LeafCtx<'_>, feature: usize, best: &Best, l2: f64, rule: SplitRule) -> SplitInfo {
    let l1 = ctx.cfg.lambda_l1;
    SplitInfo {
        feature,
        gain: best.gain - ctx.min_gain_shift,
        rule,
        left: LeafSums {
            grad: best.left_units[0],
            hess: best.left_units[1],
            count: best.left_count,
        },
        right: LeafSums {
            grad: ctx.units.grad - best.left_units[0],
            hess: ctx.units.hess - best.left_units[1],
            count: ctx.num_data - best.left_count,
        },
        left_output: leaf_output(best.left_grad, best.left_hess, l1, l2),
        right_output: leaf_output(ctx.sum_grad - best.left_grad, ctx.sum_hess - best.left_hess, l1, l2),
    }
}

/// The winning candidate as a [`SplitInfo`]; `hist` is its feature's histogram slice.
fn split_info(ctx: &LeafCtx<'_>, hist: &[f64], c: &Candidate) -> SplitInfo {
    let best = &c.best;
    match c.kind {
        CandidateKind::Numerical { default_left } => {
            let rule = SplitRule::Numerical {
                threshold: best.threshold,
                default_left,
            };
            finish(ctx, c.feature, best, ctx.cfg.lambda_l2, rule)
        }
        CandidateKind::OneHot => {
            let rule = SplitRule::Categorical {
                bins: vec![best.threshold],
            };
            finish(ctx, c.feature, best, ctx.cfg.lambda_l2, rule)
        }
        CandidateKind::Sorted { forward } => {
            let mut order = [0u16; MAX_BINS];
            let sorted = sorted_categories(ctx, hist, &mut order);
            let take = best.threshold as usize + 1;
            let bins = (0..take)
                .map(|i| {
                    u32::from(if forward {
                        sorted[i]
                    } else {
                        sorted[sorted.len() - 1 - i]
                    })
                })
                .collect();
            finish(
                ctx,
                c.feature,
                best,
                ctx.cfg.lambda_l2 + ctx.cfg.cat_l2,
                SplitRule::Categorical { bins },
            )
        }
    }
}

/// LightGBM `FindBestThresholdCategoricalInner`, one-hot branch: one category left, the rest right.
fn find_onehot(ctx: &LeafCtx<'_>, feature: usize, hist: &[f64]) -> FeatureResult {
    let cfg = ctx.cfg;
    let num_bins = hist.len() / 2;
    let mut splittable = false;
    let mut best: Option<Best> = None;
    for t in 1..num_bins {
        let bin = &hist[2 * t..2 * t + 2];
        let mut left = Side::new();
        left.add(ctx, bin);
        let other_count = ctx.num_data - left.count;
        // LightGBM's one-hot branch tests the bin's raw hessian, and takes kEpsilon off the other side.
        let raw_hess = bin[1] * ctx.scale.hess;
        let other_hess = ctx.sum_hess - raw_hess - K_EPSILON;
        let grow = Side { hess: raw_hess, ..left };
        if !matches!(ctx.check(&grow, other_count, other_hess), Check::Evaluate) {
            continue;
        }
        let grad = left.grad(ctx);
        let gain = split_gain(
            [ctx.sum_grad - grad, grad],
            [other_hess, raw_hess + K_EPSILON],
            cfg.lambda_l1,
            cfg.lambda_l2,
        );
        if gain.is_nan() || gain <= ctx.min_gain_shift {
            continue;
        }
        splittable = true;
        if best.is_none_or(|b| gain > b.gain) {
            best = Some(Best {
                gain,
                threshold: t as u32,
                left_grad: grad,
                left_hess: raw_hess + K_EPSILON,
                left_count: left.count,
                left_units: left.units,
            });
        }
    }
    let best = best.map(|b| candidate(ctx, feature, b, CandidateKind::OneHot));
    FeatureResult { splittable, best }
}

/// The categories (bins `1..`) with at least `cat_smooth` estimated rows, sorted by `sum_g / (sum_h +
/// cat_smooth)`, written into `order`; returns the used prefix.
fn sorted_categories<'o>(ctx: &LeafCtx<'_>, hist: &[f64], order: &'o mut [u16; MAX_BINS]) -> &'o [u16] {
    let cfg = ctx.cfg;
    let num_bins = (hist.len() / 2).min(MAX_BINS);
    let mut used = 0;
    for t in 1..num_bins {
        if ctx.bin_count(hist[2 * t + 1]) as f64 >= cfg.cat_smooth {
            order[used] = t as u16;
            used += 1;
        }
    }
    let ctr = |t: u16| {
        let t = usize::from(t);
        (hist[2 * t] * ctx.scale.grad) / (hist[2 * t + 1] * ctx.scale.hess + cfg.cat_smooth)
    };
    // LightGBM's `std::stable_sort`: equal ratios keep ascending bin order, so an unstable sort with the bin as
    // tie-break gives the same order without the stable sort's buffer. `total_cmp` keeps a total order should a
    // ratio be NaN.
    let sorted = &mut order[..used];
    sorted.sort_unstable_by(|&a, &b| ctr(a).total_cmp(&ctr(b)).then(a.cmp(&b)));
    sorted
}

/// LightGBM `FindBestThresholdCategoricalInner`, sorted branch: categories with enough data sorted by
/// `sum_g / (sum_h + cat_smooth)`, prefixes of up to `max_cat_threshold` taken from either end.
fn find_sorted_categorical(ctx: &LeafCtx<'_>, feature: usize, hist: &[f64]) -> FeatureResult {
    let cfg = ctx.cfg;
    let mut order = [0u16; MAX_BINS];
    let sorted = sorted_categories(ctx, hist, &mut order);
    let used_bin = sorted.len();
    let max_num_cat = cfg.max_cat_threshold.min(used_bin.div_ceil(2));
    let l2 = cfg.lambda_l2 + cfg.cat_l2;
    let mut splittable = false;
    let mut best: Option<(Best, bool)> = None;
    for forward in [true, false] {
        let order = |i: usize| usize::from(if forward { sorted[i] } else { sorted[used_bin - 1 - i] });
        let scan = SortedScan { ctx, hist, l2 };
        if let Some(candidate) = scan.run(used_bin.min(max_num_cat), order, &mut splittable) {
            if best.is_none_or(|(b, _)| candidate.gain > b.gain) {
                best = Some((candidate, forward));
            }
        }
    }
    let best = best.map(|(b, forward)| candidate(ctx, feature, b, CandidateKind::Sorted { forward }));
    FeatureResult { splittable, best }
}

/// One direction of the sorted categorical scan.
struct SortedScan<'a, 'c> {
    ctx: &'a LeafCtx<'c>,
    hist: &'a [f64],
    l2: f64,
}

impl SortedScan<'_, '_> {
    /// Best prefix of `order(0..steps)`; `threshold` holds the prefix length minus one.
    fn run(&self, steps: usize, order: impl Fn(usize) -> usize, splittable: &mut bool) -> Option<Best> {
        let ctx = self.ctx;
        let cfg = ctx.cfg;
        let mut left = Side::new();
        let mut cnt_cur_group = 0;
        let mut best: Option<Best> = None;
        for i in 0..steps {
            let t = order(i);
            let before = left.count;
            left.add(ctx, &self.hist[2 * t..2 * t + 2]);
            cnt_cur_group += left.count - before;
            let right_count = ctx.num_data - left.count;
            let right_hess = ctx.sum_hess - left.hess;
            match ctx.check(&left, right_count, right_hess) {
                Check::Continue => continue,
                Check::Break => break,
                Check::Evaluate => {}
            }
            if right_count < cfg.min_data_per_group {
                break;
            }
            if cnt_cur_group < cfg.min_data_per_group {
                continue;
            }
            cnt_cur_group = 0;
            let left_grad = left.grad(ctx);
            let gain = split_gain(
                [left_grad, ctx.sum_grad - left_grad],
                [left.hess, right_hess],
                cfg.lambda_l1,
                self.l2,
            );
            if gain.is_nan() || gain <= ctx.min_gain_shift {
                continue;
            }
            *splittable = true;
            if best.is_none_or(|b| gain > b.gain) {
                best = Some(Best {
                    gain,
                    threshold: i as u32,
                    left_grad,
                    left_hess: left.hess,
                    left_count: left.count,
                    left_units: left.units,
                });
            }
        }
        best
    }
}

/// LightGBM `FeatureHistogram::FindBestThreshold` for one feature; `hist` is the feature's `2 * num_bins` slice.
fn find_feature(
    ctx: &LeafCtx<'_>,
    scans: &mut Scans,
    feature: usize,
    meta: &FeatureMeta,
    hist: &[f64],
) -> FeatureResult {
    match &meta.bins {
        BinMapper::Numerical { nan_bin, .. } => find_numerical(ctx, scans, feature, hist, *nan_bin),
        BinMapper::Categorical { .. } if meta.num_bins as usize <= ctx.cfg.max_cat_to_onehot => {
            find_onehot(ctx, feature, hist)
        }
        BinMapper::Categorical { .. } => find_sorted_categorical(ctx, feature, hist),
        BinMapper::Trivial => FeatureResult::NONE,
    }
}

/// `SplitInfo::operator>`: higher gain (NaN as `kMinScore`), then the lower feature index. A total order, so the
/// best of any grouping of candidates is the same.
fn better(a: &Candidate, b: &Candidate) -> bool {
    let gain = |s: &Candidate| if s.gain.is_nan() { MIN_SCORE } else { s.gain };
    match gain(a).partial_cmp(&gain(b)) {
        Some(Ordering::Greater) => true,
        Some(Ordering::Less) => false,
        _ => a.feature < b.feature,
    }
}

fn best_of(a: Option<Candidate>, b: Option<Candidate>) -> Option<Candidate> {
    match (a, b) {
        (Some(a), Some(b)) => Some(if better(&b, &a) { b } else { a }),
        (a, b) => a.or(b),
    }
}

/// Bin positions a feature's scans visit: numerical features with a NaN bin scan twice.
fn scan_work(meta: &FeatureMeta) -> usize {
    let bins = meta.num_bins as usize;
    match meta.bins {
        BinMapper::Numerical { nan_bin: true, .. } => 2 * bins,
        _ => bins,
    }
}

/// The best candidate over `features` (indices from `first`), sequentially, recording which were splittable.
fn search_range(
    ctx: &LeafCtx<'_>,
    hist: &[f64],
    features: &[FeatureMeta],
    evaluate: &[bool],
    first: usize,
    splittable: &mut [bool],
) -> Option<Candidate> {
    let mut best = None;
    let mut scans = Scans::new();
    for (i, meta) in features.iter().enumerate() {
        let f = first + i;
        if !evaluate[f] || meta.is_trivial() {
            continue;
        }
        let begin = 2 * meta.hist_offset as usize;
        let end = begin + 2 * meta.num_bins as usize;
        let result = find_feature(ctx, &mut scans, f, meta, &hist[begin..end]);
        splittable[i] = result.splittable;
        best = best_of(best, result.best);
    }
    best
}

/// The best split of a leaf over the features with `evaluate[f]`. `hist` is the leaf's whole histogram
/// (`2 * total_bins`) and `sums` its totals in histogram units. Runs on rayon over contiguous feature ranges of
/// about `SEARCH_CHUNK_WORK` bin visits each, sequentially when there is only one.
pub fn find_best_split(
    hist: &[f64],
    features: &[FeatureMeta],
    evaluate: &[bool],
    sums: LeafSums,
    scale: HistScale,
    cfg: &SplitConfig,
) -> LeafSplit {
    let mut splittable = vec![false; features.len()];
    let Some(ctx) = LeafCtx::new(cfg, scale, sums) else {
        return LeafSplit { best: None, splittable };
    };
    let mut ranges: Vec<(usize, &[FeatureMeta], &mut [bool])> = Vec::new();
    let (mut start, mut work) = (0, 0);
    let mut rest_features = features;
    let mut rest_flags = splittable.as_mut_slice();
    for (f, meta) in features.iter().enumerate() {
        if evaluate[f] {
            work += scan_work(meta);
        }
        if work >= SEARCH_CHUNK_WORK || f + 1 == features.len() {
            let (chunk, tail) = rest_features.split_at(f + 1 - start);
            let (flags, flags_tail) = std::mem::take(&mut rest_flags).split_at_mut(f + 1 - start);
            ranges.push((start, chunk, flags));
            (rest_features, rest_flags) = (tail, flags_tail);
            (start, work) = (f + 1, 0);
        }
    }
    let search = |(first, chunk, flags): (usize, &[FeatureMeta], &mut [bool])| {
        search_range(&ctx, hist, chunk, evaluate, first, flags)
    };
    let best = if ranges.len() <= 1 {
        ranges.into_iter().map(search).fold(None, best_of)
    } else {
        ranges.into_par_iter().map(search).reduce(|| None, best_of)
    };
    let best = best.map(|c| {
        let meta = &features[c.feature];
        let begin = 2 * meta.hist_offset as usize;
        split_info(&ctx, &hist[begin..begin + 2 * meta.num_bins as usize], &c)
    });
    LeafSplit { best, splittable }
}

/// LightGBM `ColSampler::ResetByTree`: the features one tree may split on. Exactly
/// `max(1, RoundInt(fraction * usable))` of the non-trivial features, those with the lowest
/// `uniform(seed, STREAM_FEATURE_FRACTION, iteration, feature)` (ties by index), so every rank draws the same set.
pub fn feature_mask(features: &[FeatureMeta], fraction: f64, seed: u32, iteration: u32) -> Vec<bool> {
    let usable: Vec<usize> = (0..features.len()).filter(|&f| !features[f].is_trivial()).collect();
    let mut mask = vec![false; features.len()];
    if fraction >= 1.0 {
        for &f in &usable {
            mask[f] = true;
        }
        return mask;
    }
    let count = (round_int(usable.len() as f64 * fraction) as usize).max(usable.len().min(1));
    let mut keyed: Vec<(f32, usize)> = usable
        .iter()
        .map(|&f| (quantize::uniform(seed, STREAM_FEATURE_FRACTION, iteration, f as u64), f))
        .collect();
    keyed.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    for &(_, f) in keyed.iter().take(count) {
        mask[f] = true;
    }
    mask
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A small deterministic generator for test histograms.
    struct Lcg(u64);

    impl Lcg {
        fn next(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (self.0 >> 11) as f64 / (1u64 << 53) as f64
        }

        fn below(&mut self, n: u64) -> u64 {
            (self.next() * n as f64) as u64
        }
    }

    fn numerical(num_bins: u32, nan_bin: bool) -> FeatureMeta {
        let numeric = num_bins - u32::from(nan_bin);
        let mut upper_bounds: Vec<f64> = (0..numeric).map(f64::from).collect();
        *upper_bounds.last_mut().expect("bins") = f64::INFINITY;
        FeatureMeta {
            name: "f".into(),
            bins: BinMapper::Numerical {
                upper_bounds,
                nan_bin,
                min: 0.0,
                max: f64::from(numeric),
            },
            num_bins,
            group: 0,
            column: 0,
            hist_offset: 0,
        }
    }

    fn categorical(num_bins: u32) -> FeatureMeta {
        FeatureMeta {
            name: "c".into(),
            bins: BinMapper::Categorical {
                categories: (1..num_bins as i32).collect(),
            },
            num_bins,
            group: 0,
            column: 0,
            hist_offset: 0,
        }
    }

    fn config() -> SplitConfig {
        SplitConfig::new(&Params {
            min_data_in_leaf: 5,
            ..Params::default()
        })
    }

    /// Bins with `rows[b]` rows of hessian `hess_per_row` each, so count estimates are exact, and random gradients.
    fn random_hist(rng: &mut Lcg, num_bins: usize, hess_per_row: f64) -> (Vec<f64>, Vec<i64>) {
        let rows: Vec<i64> = (0..num_bins).map(|_| rng.below(40) as i64).collect();
        let hist = rows
            .iter()
            .flat_map(|&n| [(rng.next() - 0.45) * n as f64, n as f64 * hess_per_row])
            .collect();
        (hist, rows)
    }

    fn sums(hist: &[f64], rows: &[i64]) -> LeafSums {
        LeafSums::from_feature_hist(hist, rows.iter().sum())
    }

    fn side_sums(hist: &[f64], rows: &[i64], left: impl Fn(usize) -> bool) -> (f64, f64, i64) {
        (0..rows.len())
            .filter(|&b| left(b))
            .fold((0.0, 0.0, 0), |(g, h, n), b| {
                (g + hist[2 * b], h + hist[2 * b + 1], n + rows[b])
            })
    }

    /// The best `gain - min_gain_shift` over every split the rules allow, by direct summation.
    fn brute_force(hist: &[f64], rows: &[i64], cfg: &SplitConfig, candidates: &[Box<dyn Fn(usize) -> bool>]) -> f64 {
        let total = sums(hist, rows);
        let shift = leaf_gain(total.grad, total.hess, cfg.lambda_l1, cfg.lambda_l2) + cfg.min_gain_to_split;
        let mut best = MIN_SCORE;
        for left in candidates {
            let (lg, lh, ln) = side_sums(hist, rows, left);
            let (rg, rh, rn) = (total.grad - lg, total.hess - lh, total.count - ln);
            let ok = |n: i64, h: f64| n >= cfg.min_data_in_leaf && n > 0 && h >= cfg.min_sum_hessian_in_leaf && h > 0.0;
            if !ok(ln, lh) || !ok(rn, rh) {
                continue;
            }
            let gain =
                leaf_gain(lg, lh, cfg.lambda_l1, cfg.lambda_l2) + leaf_gain(rg, rh, cfg.lambda_l1, cfg.lambda_l2);
            if gain > shift {
                best = best.max(gain - shift);
            }
        }
        best
    }

    fn numerical_candidates(num_bins: usize, nan_bin: bool) -> Vec<Box<dyn Fn(usize) -> bool>> {
        let numeric = num_bins - usize::from(nan_bin);
        let mut out: Vec<Box<dyn Fn(usize) -> bool>> = Vec::new();
        let end = if nan_bin { numeric } else { numeric - 1 };
        for t in 0..end {
            for nan_left in [false, true] {
                out.push(Box::new(move |b| if b == numeric { nan_left } else { b <= t }));
            }
        }
        out
    }

    fn assert_close(found: f64, expected: f64, case: &str) {
        assert!(
            (found - expected).abs() <= 1e-9 * expected.abs().max(1.0),
            "{case}: found {found}, brute force {expected}"
        );
    }

    /// Checks the reported child sums and outputs against direct sums of the bins the rule sends left.
    fn check_children(
        split: &SplitInfo,
        meta: &FeatureMeta,
        hist: &[f64],
        rows: &[i64],
        cfg: &SplitConfig,
        case: &str,
    ) {
        let nan_bin = matches!(meta.bins, BinMapper::Numerical { nan_bin: true, .. });
        let num_bins = rows.len();
        let goes_left = |b: usize| match &split.rule {
            SplitRule::Numerical {
                threshold,
                default_left,
            } => {
                if nan_bin && b == num_bins - 1 {
                    *default_left
                } else {
                    b <= *threshold as usize
                }
            }
            SplitRule::Categorical { bins } => bins.contains(&(b as u32)),
        };
        let (lg, lh, ln) = side_sums(hist, rows, goes_left);
        assert_close(split.left.grad, lg, &format!("{case}: left grad"));
        assert_close(split.left.hess, lh, &format!("{case}: left hess"));
        assert_eq!(split.left.count, ln, "{case}: left count");
        let l2 = match &split.rule {
            SplitRule::Categorical { bins } if bins.len() > 1 || rows.len() > cfg.max_cat_to_onehot => {
                cfg.lambda_l2 + cfg.cat_l2
            }
            _ => cfg.lambda_l2,
        };
        assert_close(
            split.left_output,
            leaf_output(lg, lh, cfg.lambda_l1, l2),
            &format!("{case}: left output"),
        );
    }

    #[test]
    fn numerical_matches_brute_force() {
        let mut rng = Lcg(7);
        for case in 0..400 {
            let nan_bin = case % 2 == 1;
            let num_bins = 2 + rng.below(30) as usize;
            let (hist, rows) = random_hist(&mut rng, num_bins, 0.25);
            let meta = numerical(num_bins as u32, nan_bin);
            let mut cfg = config();
            cfg.lambda_l1 = if case % 3 == 0 { 0.5 } else { 0.0 };
            cfg.lambda_l2 = if case % 5 == 0 { 2.0 } else { 0.0 };
            let name = format!("case {case} bins {num_bins} nan {nan_bin}");
            let leaf = find_best_split(
                &hist,
                std::slice::from_ref(&meta),
                &[true],
                sums(&hist, &rows),
                HistScale::UNIT,
                &cfg,
            );
            let expected = brute_force(&hist, &rows, &cfg, &numerical_candidates(num_bins, nan_bin));
            match &leaf.best {
                None => assert_eq!(expected, MIN_SCORE, "{name}: brute force found a split"),
                Some(split) => {
                    assert_close(split.gain, expected, &name);
                    check_children(split, &meta, &hist, &rows, &cfg, &name);
                }
            }
        }
    }

    #[test]
    fn quantized_scale_matches_real_sums() {
        let mut rng = Lcg(11);
        for case in 0..100 {
            let num_bins = 3 + rng.below(20) as usize;
            let rows: Vec<i64> = (0..num_bins).map(|_| rng.below(40) as i64).collect();
            let units: Vec<f64> = rows
                .iter()
                .flat_map(|&n| [rng.below(9) as f64 - 4.0, 2.0 * n as f64])
                .collect();
            let scale = HistScale {
                grad: 0.37,
                hess: 0.125,
            };
            let real: Vec<f64> = units
                .chunks(2)
                .flat_map(|b| [b[0] * scale.grad, b[1] * scale.hess])
                .collect();
            let meta = numerical(num_bins as u32, case % 2 == 0);
            let cfg = config();
            let total = rows.iter().sum();
            let q = find_best_split(
                &units,
                std::slice::from_ref(&meta),
                &[true],
                LeafSums::from_feature_hist(&units, total),
                scale,
                &cfg,
            );
            let f = find_best_split(
                &real,
                &[meta],
                &[true],
                LeafSums::from_feature_hist(&real, total),
                HistScale::UNIT,
                &cfg,
            );
            match (q.best, f.best) {
                (None, None) => {}
                (Some(q), Some(f)) => {
                    assert_eq!(q.rule, f.rule, "case {case}: rule");
                    assert_close(q.gain, f.gain, &format!("case {case}: gain"));
                    assert_close(
                        q.left.grad * scale.grad,
                        f.left.grad,
                        &format!("case {case}: left grad"),
                    );
                }
                (q, f) => panic!("case {case}: quantized {q:?} vs float {f:?}"),
            }
        }
    }

    #[test]
    fn nan_against_rest_is_a_candidate() {
        // Numeric bins share one gradient sign, only the NaN bin differs: the best split isolates NaN.
        let hist = vec![-10.0, 10.0, -10.0, 10.0, -10.0, 10.0, 30.0, 10.0];
        let rows = vec![10, 10, 10, 10];
        let meta = numerical(4, true);
        let leaf = find_best_split(&hist, &[meta], &[true], sums(&hist, &rows), HistScale::UNIT, &config());
        let best = leaf.best.expect("split");
        assert_eq!(
            best.rule,
            SplitRule::Numerical {
                threshold: 2,
                default_left: false
            }
        );
        assert_eq!(best.left.count, 30);
    }

    #[test]
    fn onehot_matches_brute_force() {
        let mut rng = Lcg(3);
        for case in 0..200 {
            let num_bins = 2 + rng.below(3) as usize;
            let (hist, rows) = random_hist(&mut rng, num_bins, 1.0);
            let meta = categorical(num_bins as u32);
            let cfg = config();
            let leaf = find_best_split(
                &hist,
                std::slice::from_ref(&meta),
                &[true],
                sums(&hist, &rows),
                HistScale::UNIT,
                &cfg,
            );
            let candidates: Vec<Box<dyn Fn(usize) -> bool>> = (1..num_bins)
                .map(|t| Box::new(move |b| b == t) as Box<dyn Fn(usize) -> bool>)
                .collect();
            let expected = brute_force(&hist, &rows, &cfg, &candidates);
            let name = format!("onehot case {case}");
            match &leaf.best {
                None => assert_eq!(expected, MIN_SCORE, "{name}: brute force found a split"),
                Some(split) => {
                    assert_close(split.gain, expected, &name);
                    let SplitRule::Categorical { bins } = &split.rule else {
                        panic!("{name}: rule")
                    };
                    assert_eq!(bins.len(), 1, "{name}: one category");
                    check_children(split, &meta, &hist, &rows, &cfg, &name);
                }
            }
        }
    }

    #[test]
    fn sorted_categorical_matches_brute_force() {
        let mut rng = Lcg(5);
        let mut found = 0;
        for case in 0..200 {
            let num_bins = 6 + rng.below(12) as usize;
            let (hist, rows) = random_hist(&mut rng, num_bins, 1.0);
            let mut cfg = config();
            cfg.min_data_per_group = 1;
            cfg.cat_smooth = 1.0;
            cfg.max_cat_threshold = 4;
            let meta = categorical(num_bins as u32);
            let leaf = find_best_split(
                &hist,
                std::slice::from_ref(&meta),
                &[true],
                sums(&hist, &rows),
                HistScale::UNIT,
                &cfg,
            );
            // Brute force over prefixes of the ctr order from both ends, with the categorical l2.
            let ctr = |t: usize| hist[2 * t] / (hist[2 * t + 1] + cfg.cat_smooth);
            let mut sorted: Vec<usize> = (1..num_bins).filter(|&t| rows[t] as f64 >= cfg.cat_smooth).collect();
            sorted.sort_by(|&a, &b| ctr(a).total_cmp(&ctr(b)));
            let used = sorted.len();
            let max_cat = cfg.max_cat_threshold.min(used.div_ceil(2));
            let mut candidates: Vec<Box<dyn Fn(usize) -> bool>> = Vec::new();
            for k in 1..=max_cat.min(used) {
                let low: Vec<usize> = sorted[..k].to_vec();
                let high: Vec<usize> = sorted[used - k..].to_vec();
                candidates.push(Box::new(move |b| low.contains(&b)));
                candidates.push(Box::new(move |b| high.contains(&b)));
            }
            let brute_cfg = SplitConfig {
                lambda_l2: cfg.lambda_l2 + cfg.cat_l2,
                ..cfg
            };
            // The gain shift keeps the plain l2 while candidates use cat_l2, as in LightGBM.
            let total = sums(&hist, &rows);
            let shift_fix = leaf_gain(total.grad, total.hess, 0.0, brute_cfg.lambda_l2)
                - leaf_gain(total.grad, total.hess, 0.0, cfg.lambda_l2);
            let expected = brute_force(&hist, &rows, &brute_cfg, &candidates);
            let expected = if expected == MIN_SCORE {
                expected
            } else {
                expected + shift_fix
            };
            let name = format!("sorted case {case}");
            match &leaf.best {
                None => assert!(expected <= 0.0, "{name}: brute force found {expected}"),
                Some(split) => {
                    found += 1;
                    assert_close(split.gain, expected, &name);
                    check_children(split, &meta, &hist, &rows, &cfg, &name);
                    let SplitRule::Categorical { bins } = &split.rule else {
                        panic!("{name}: rule")
                    };
                    assert!(bins.iter().all(|&b| b >= 1), "{name}: bin 0 never goes left");
                }
            }
        }
        assert!(found > 50, "only {found} sorted cases split");
    }

    #[test]
    fn constraints_reject_splits() {
        let hist = vec![-20.0, 10.0, 20.0, 10.0, -5.0, 5.0];
        let rows = vec![10, 10, 5];
        let meta = numerical(3, false);
        let run = |cfg: &SplitConfig| {
            find_best_split(
                &hist,
                std::slice::from_ref(&meta),
                &[true],
                sums(&hist, &rows),
                HistScale::UNIT,
                cfg,
            )
            .best
        };
        let base = config();
        assert!(run(&base).is_some(), "base config splits");
        let cases = [
            (
                "min_data_in_leaf",
                SplitConfig {
                    min_data_in_leaf: 16,
                    ..base
                },
            ),
            (
                "min_sum_hessian_in_leaf",
                SplitConfig {
                    min_sum_hessian_in_leaf: 16.0,
                    ..base
                },
            ),
            (
                "min_gain_to_split",
                SplitConfig {
                    min_gain_to_split: 1e6,
                    ..base
                },
            ),
            (
                "lambda_l1",
                SplitConfig {
                    lambda_l1: 100.0,
                    ..base
                },
            ),
        ];
        for (name, cfg) in cases {
            assert_eq!(run(&cfg), None, "{name} rejects every split");
        }
        let excluded = find_best_split(
            &hist,
            std::slice::from_ref(&meta),
            &[false],
            sums(&hist, &rows),
            HistScale::UNIT,
            &base,
        );
        assert_eq!(excluded.best, None, "masked feature");
        assert_eq!(excluded.splittable, vec![false]);
    }

    #[test]
    fn zero_hessian_rows_never_make_empty_children() {
        // Half the bins hold rows with g = h = 0 (lambdarank queries without a relevant item), with every minimum
        // relaxed to 0: no candidate may leave a child with no hessian.
        let cfg = SplitConfig {
            min_data_in_leaf: 0,
            min_sum_hessian_in_leaf: 0.0,
            ..config()
        };
        let hist = vec![0.0, 0.0, 0.0, 0.0, -3.0, 2.0, 0.0, 0.0, 4.0, 2.0, 0.0, 0.0];
        let meta = numerical(6, false);
        let leaf = find_best_split(
            &hist,
            &[meta],
            &[true],
            LeafSums::from_feature_hist(&hist, 400),
            HistScale::UNIT,
            &cfg,
        );
        let best = leaf.best.expect("split between the two non-zero bins");
        assert!(best.left.hess > 0.0 && best.right.hess > 0.0, "{best:?}");
        assert!(best.left.count > 0 && best.right.count > 0, "{best:?}");
        let zero = vec![0.0; 12];
        let leaf = find_best_split(
            &zero,
            &[numerical(6, false)],
            &[true],
            LeafSums {
                count: 400,
                ..LeafSums::default()
            },
            HistScale::UNIT,
            &cfg,
        );
        assert_eq!(leaf.best, None, "a leaf without hessian does not split");
        for meta in [categorical(3), categorical(8)] {
            let hist = vec![
                0.0, 0.0, -3.0, 2.0, 0.0, 0.0, 4.0, 2.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0,
            ];
            let hist = &hist[..2 * meta.num_bins as usize];
            let cfg = SplitConfig {
                min_data_per_group: 0,
                cat_smooth: 0.0,
                ..cfg
            };
            let leaf = find_best_split(
                hist,
                &[meta],
                &[true],
                LeafSums::from_feature_hist(hist, 400),
                HistScale::UNIT,
                &cfg,
            );
            if let Some(best) = leaf.best {
                assert!(best.left.hess > 0.0 && best.right.hess > 0.0, "{best:?}");
            }
        }
    }

    #[test]
    fn ties_go_to_the_lower_feature() {
        let hist = vec![-20.0, 10.0, 20.0, 10.0, -20.0, 10.0, 20.0, 10.0];
        let mut second = numerical(2, false);
        second.hist_offset = 2;
        let features = vec![numerical(2, false), second];
        let leaf = find_best_split(
            &hist,
            &features,
            &[true, true],
            LeafSums::from_feature_hist(&hist[..4], 20),
            HistScale::UNIT,
            &config(),
        );
        assert_eq!(leaf.best.expect("split").feature, 0);
        assert_eq!(leaf.splittable, vec![true, true]);
    }

    #[test]
    fn feature_mask_samples_the_lightgbm_count() {
        let mut features: Vec<FeatureMeta> = (0..10).map(|_| numerical(4, false)).collect();
        features[3].num_bins = 0;
        features[3].bins = BinMapper::Trivial;
        for (fraction, expected) in [(1.0, 9), (0.5, 5), (0.8, 7), (0.01, 1)] {
            let mask = feature_mask(&features, fraction, 2, 0);
            assert_eq!(mask.iter().filter(|&&m| m).count(), expected, "fraction {fraction}");
            assert!(!mask[3], "trivial features are never sampled");
        }
        assert_eq!(feature_mask(&features, 0.5, 2, 4), feature_mask(&features, 0.5, 2, 4));
        let distinct = (0..20)
            .map(|i| feature_mask(&features, 0.5, 2, i))
            .collect::<std::collections::HashSet<_>>();
        assert!(distinct.len() > 5, "the sample changes with the iteration");
    }
}
