//! LambdaRank with NDCG: LightGBM 3.3.5 `LambdarankNDCG::GetGradientsForOneQuery` and the `DCGCalculator` pieces it
//! and the NDCG metric use (`CalMaxDCGAtK`, `CalDCG`, `CheckLabel`). LightGBM's `CheckMetadata` cap of
//! `kMaxPosition` rows per query is not applied: queries of any length train, with O(n * truncation_level) pairs.
//!
//! The pair loop is LightGBM's: ranks `i < truncation_level` against every rank `j > i`, each pair's lambda and
//! hessian added to both items in that order, with the pair sigmoid from LightGBM's 1M-entry table
//! ([`SigmoidTable`]). silvic accumulates in f64 where LightGBM accumulates in f32 (`score_t`), then casts to f32
//! after the `log2(1 + sum) / sum` normalization and multiplies by the f32 row weight, as LightGBM's
//! `RankingObjective::GetGradients`. Every query's result depends only on its own rows, so it is deterministic
//! under any thread count.

use std::sync::{Arc, LazyLock, Mutex, PoisonError};

use rayon::prelude::*;

use crate::backend::ObjectiveSetup;
use crate::error::{Result, SilvicError};

use super::K_EPSILON;

/// LightGBM `DCGCalculator::kMaxPosition`: the length of the precomputed discount table. Not a query-size limit:
/// silvic trains on longer queries (LightGBM 3.3.5 rejects them) and computes discounts past the table directly.
pub const MAX_POSITION: usize = 10_000;

/// LightGBM `LambdarankNDCG::min_sigmoid_input_` before scaling by `sigmoid`.
const MIN_SIGMOID_INPUT: f64 = -50.0;

/// LightGBM `LambdarankNDCG::_sigmoid_bins`.
const SIGMOID_BINS: usize = 1024 * 1024;

/// LightGBM's `0.01f` added to `|delta_score|` when normalizing by score distance.
const SCORE_DISTANCE_EPSILON: f64 = 0.01_f32 as f64;

/// LightGBM `DCGCalculator::discount_`.
static DISCOUNTS: LazyLock<Vec<f64>> =
    LazyLock::new(|| (0..MAX_POSITION).map(|i| 1.0 / (2.0 + i as f64).log2()).collect());

/// `DCGCalculator::GetDiscount`: `1 / log2(2 + rank)`, from the same precomputed table LightGBM uses.
pub fn discount(rank: usize) -> f64 {
    DISCOUNTS
        .get(rank)
        .copied()
        .unwrap_or_else(|| 1.0 / (2.0 + rank as f64).log2())
}

/// LightGBM's `sigmoid_table_` (`LambdarankNDCG::ConstructSigmoidTable`): `1 / (1 + exp(sigmoid * x))` at the
/// lower edge of each of `1024 * 1024` bins over `[-50, 50] / sigmoid / 2`.
#[derive(Debug)]
pub struct SigmoidTable {
    sigmoid: f64,
    min_input: f64,
    max_input: f64,
    idx_factor: f64,
    values: Vec<f64>,
}

impl SigmoidTable {
    pub fn new(sigmoid: f64) -> SigmoidTable {
        let min_input = MIN_SIGMOID_INPUT / sigmoid / 2.0;
        let max_input = -min_input;
        let idx_factor = SIGMOID_BINS as f64 / (max_input - min_input);
        let values = (0..SIGMOID_BINS)
            .map(|i| {
                let score = i as f64 / idx_factor + min_input;
                1.0 / (1.0 + (score * sigmoid).exp())
            })
            .collect();
        SigmoidTable {
            sigmoid,
            min_input,
            max_input,
            idx_factor,
            values,
        }
    }

    /// The table for `sigmoid`, built once and shared while `sigmoid` stays the same (8 MiB, ~1M `exp`s to build).
    pub fn shared(sigmoid: f64) -> Arc<SigmoidTable> {
        static LAST: Mutex<Option<Arc<SigmoidTable>>> = Mutex::new(None);
        let mut last = LAST.lock().unwrap_or_else(PoisonError::into_inner);
        match &*last {
            Some(table) if table.sigmoid.to_bits() == sigmoid.to_bits() => Arc::clone(table),
            _ => Arc::clone(last.insert(Arc::new(SigmoidTable::new(sigmoid)))),
        }
    }

    /// The table entries (LightGBM `sigmoid_table_`), `1024 * 1024` of them.
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    /// LightGBM `min_sigmoid_input_`: at or below it, [`get`](Self::get) returns the first entry.
    pub fn min_input(&self) -> f64 {
        self.min_input
    }

    /// LightGBM `max_sigmoid_input_`: at or above it, [`get`](Self::get) returns the last entry.
    pub fn max_input(&self) -> f64 {
        self.max_input
    }

    /// LightGBM `sigmoid_table_idx_factor_`: between the bounds the entry is `(delta - min_input) * idx_factor`,
    /// truncated.
    pub fn idx_factor(&self) -> f64 {
        self.idx_factor
    }

    /// `LambdarankNDCG::GetSigmoid`: the pair probability for a score difference `delta`.
    #[inline]
    pub fn get(&self, delta: f64) -> f64 {
        if delta <= self.min_input {
            self.values[0]
        } else if delta >= self.max_input {
            self.values[SIGMOID_BINS - 1]
        } else {
            // In range, the index is below SIGMOID_BINS; `min` keeps the bound check out of the loop all the same.
            self.values[(((delta - self.min_input) * self.idx_factor) as usize).min(SIGMOID_BINS - 1)]
        }
    }
}

/// `DCGCalculator::CalMaxDCGAtK`: DCG of the best ordering of `labels`, truncated at `k`.
pub fn max_dcg_at_k(k: usize, labels: &[f32], label_gain: &[f64]) -> f64 {
    let mut label_count = vec![0usize; label_gain.len()];
    for &label in labels {
        label_count[label as usize] += 1;
    }
    let mut top_label = label_gain.len() - 1;
    let mut dcg = 0.0;
    for j in 0..k.min(labels.len()) {
        // LightGBM stops the search at label 0 without checking its count.
        while top_label > 0 && label_count[top_label] == 0 {
            top_label -= 1;
        }
        dcg += discount(j) * label_gain[top_label];
        label_count[top_label] = label_count[top_label].wrapping_sub(1);
    }
    dcg
}

/// Row indices of a query sorted by score descending, ties by row (LightGBM's `std::stable_sort`).
pub(crate) fn rank_order(scores: &[f32]) -> Vec<usize> {
    let mut order = Vec::new();
    rank_order_into(scores, &mut Vec::new(), &mut order);
    order
}

/// [`rank_order`] into reused buffers, as one sort of integer keys: the high half orders scores descending, the low
/// half is the row, so ties keep row order as a stable sort would.
fn rank_order_into(scores: &[f32], keys: &mut Vec<u64>, order: &mut Vec<usize>) {
    keys.clear();
    keys.extend(
        scores
            .iter()
            .enumerate()
            .map(|(row, &score)| (u64::from(descending_key(score)) << 32) | row as u64),
    );
    keys.sort_unstable();
    order.clear();
    order.extend(keys.iter().map(|&key| (key & u64::from(u32::MAX)) as usize));
}

/// A key whose ascending order is `score`'s descending order. `+ 0.0` folds `-0.0` into `0.0`, which LightGBM's `>`
/// treats as equal.
fn descending_key(score: f32) -> u32 {
    let bits = (score + 0.0).to_bits();
    // IEEE bits as an ascending unsigned key: flip every bit of negatives, the sign bit of positives.
    let ascending = if bits >> 31 == 1 { !bits } else { bits | 1 << 31 };
    !ascending
}

/// `DCGCalculator::CheckLabel`: non-negative integers below the number of label gains.
pub(crate) fn check_labels(label: &[f32], num_gains: usize) -> Result<()> {
    for &l in label {
        if (f64::from(l) - f64::from(l as i32)).abs() > K_EPSILON {
            return Err(SilvicError::data(format!(
                "label should be int type (met {l}) for ranking task, for the gain of label, please set the \
                 label_gain parameter"
            )));
        }
        if l < 0.0 {
            return Err(SilvicError::data(format!(
                "label should be non-negative (met {l}) for ranking task"
            )));
        }
        if l as usize >= num_gains {
            return Err(SilvicError::data(format!(
                "label {} is not less than the number of label mappings ({num_gains})",
                l as usize
            )));
        }
    }
    Ok(())
}

/// Every query's gradients, parallel over queries.
pub(super) fn gradients(
    setup: &ObjectiveSetup,
    boundaries: &[u32],
    scores: &[f32],
    label: &[f32],
    weight: Option<&[f32]>,
    grad: &mut [f32],
    hess: &mut [f32],
) {
    let ObjectiveSetup::LambdaRank { sigmoid, .. } = setup else {
        panic!("lambdarank gradients need a lambdarank objective setup");
    };
    let table = SigmoidTable::shared(*sigmoid);
    let mut pieces = Vec::with_capacity(boundaries.len().saturating_sub(1));
    let (mut grad_rest, mut hess_rest) = (grad, hess);
    for w in boundaries.windows(2) {
        let len = (w[1] - w[0]) as usize;
        let (g, gr) = std::mem::take(&mut grad_rest).split_at_mut(len);
        let (h, hr) = std::mem::take(&mut hess_rest).split_at_mut(len);
        grad_rest = gr;
        hess_rest = hr;
        pieces.push((w[0] as usize, g, h));
    }
    pieces
        .into_par_iter()
        .enumerate()
        .with_min_len(QUERIES_PER_TASK)
        .for_each_init(
            || Scratch::new(Arc::clone(&table)),
            |scratch, (query, (start, g, h))| {
                let rows = start..start + g.len();
                let query_weight = weight.map(|w| &w[rows.clone()]);
                query_gradients(
                    setup,
                    query,
                    &scores[rows.clone()],
                    &label[rows],
                    query_weight,
                    g,
                    h,
                    scratch,
                );
            },
        );
}

/// Queries per rayon task, at least: per-query work is a few microseconds, so smaller tasks only add scheduling.
const QUERIES_PER_TASK: usize = 16;

/// What every pair of a query shares.
struct QueryContext<'a> {
    sigmoid: f64,
    table: &'a SigmoidTable,
    inverse_max_dcg: f64,
    /// `norm && best_score != worst_score`: divide each pair's delta NDCG by its score distance.
    norm_by_distance: bool,
}

/// What one thread reuses across queries: the sigmoid table and the per-query buffers.
pub(super) struct Scratch {
    table: Arc<SigmoidTable>,
    items: Items,
}

impl Scratch {
    pub(super) fn new(table: Arc<SigmoidTable>) -> Scratch {
        Scratch {
            table,
            items: Items::default(),
        }
    }
}

/// A query's items in rank order, as a struct of arrays so the pair loop streams them.
#[derive(Default)]
struct Items {
    keys: Vec<u64>,
    order: Vec<usize>,
    score: Vec<f64>,
    label: Vec<f32>,
    gain: Vec<f64>,
    discount: Vec<f64>,
    lambda: Vec<f64>,
    hessian: Vec<f64>,
}

/// `LambdarankNDCG::GetGradientsForOneQuery` for one query's rows, weights applied.
#[allow(clippy::too_many_arguments)]
pub(super) fn query_gradients(
    setup: &ObjectiveSetup,
    query: usize,
    scores: &[f32],
    label: &[f32],
    weight: Option<&[f32]>,
    grad: &mut [f32],
    hess: &mut [f32],
    scratch: &mut Scratch,
) {
    let ObjectiveSetup::LambdaRank {
        sigmoid,
        norm,
        truncation_level,
        label_gain,
        inverse_max_dcg,
    } = setup
    else {
        panic!("lambdarank gradients need a lambdarank objective setup");
    };
    if scores.is_empty() {
        return;
    }
    debug_assert_eq!(
        scratch.table.sigmoid.to_bits(),
        sigmoid.to_bits(),
        "sigmoid table of another sigmoid"
    );
    let items = &mut scratch.items;
    load_ranked(items, scores, label, label_gain);
    let context = QueryContext {
        sigmoid: *sigmoid,
        table: &scratch.table,
        inverse_max_dcg: inverse_max_dcg[query],
        norm_by_distance: *norm && items.score[0] != worst_score(&items.score),
    };
    let sum_lambdas = accumulate_pairs(items, *truncation_level as usize, &context);
    let norm_factor = if *norm && sum_lambdas > 0.0 {
        (1.0 + sum_lambdas).log2() / sum_lambdas
    } else {
        1.0
    };
    for ((&lambda, &hessian), &row) in items.lambda.iter().zip(&items.hessian).zip(&items.order) {
        let mut g = (lambda * norm_factor) as f32;
        let mut h = (hessian * norm_factor) as f32;
        if let Some(weight) = weight {
            g *= weight[row];
            h *= weight[row];
        }
        grad[row] = g;
        hess[row] = h;
    }
}

/// Sorts the query and fills the rank-ordered item vectors, lambdas and hessians zeroed.
fn load_ranked(items: &mut Items, scores: &[f32], label: &[f32], label_gain: &[f64]) {
    rank_order_into(scores, &mut items.keys, &mut items.order);
    let n = scores.len();
    let order = &items.order;
    items.score.clear();
    items.score.extend(order.iter().map(|&row| f64::from(scores[row])));
    items.label.clear();
    items.label.extend(order.iter().map(|&row| label[row]));
    items.gain.clear();
    items.gain.extend(items.label.iter().map(|&l| label_gain[l as usize]));
    items.discount.clear();
    items.discount.extend_from_slice(&DISCOUNTS[..n.min(MAX_POSITION)]);
    items.discount.extend((MAX_POSITION..n).map(discount));
    items.lambda.clear();
    items.lambda.resize(n, 0.0);
    items.hessian.clear();
    items.hessian.resize(n, 0.0);
}

/// The lowest score, skipping one trailing `kMinScore` (`-inf`) as LightGBM does.
fn worst_score(ranked_scores: &[f64]) -> f64 {
    let mut worst = ranked_scores.len() - 1;
    if worst > 0 && ranked_scores[worst] == f64::NEG_INFINITY {
        worst -= 1;
    }
    ranked_scores[worst]
}

/// LightGBM's pair loop: ranks `i < truncation_level`, `j > i`, each pair evaluated once and added to both items
/// in LightGBM's order; returns `sum_lambdas`. `kMinScore` items sort last, so skipping them ends both loops early.
fn accumulate_pairs(items: &mut Items, truncation_level: usize, context: &QueryContext) -> f64 {
    let Items {
        score,
        label,
        gain,
        discount,
        lambda,
        hessian,
        ..
    } = items;
    let valid = score
        .iter()
        .position(|&s| s == f64::NEG_INFINITY)
        .unwrap_or(score.len());
    let sigmoid_squared = context.sigmoid * context.sigmoid;
    let mut sum_lambdas = 0.0;
    for i in 0..truncation_level.min(valid.saturating_sub(1)) {
        let (score_i, label_i, gain_i, discount_i) = (score[i], label[i], gain[i], discount[i]);
        // Item i's sums live in registers through its row; they start from what earlier rows added.
        let (mut lambda_i, mut hessian_i) = (lambda[i], hessian[i]);
        for j in i + 1..valid {
            if label[j] == label_i {
                continue;
            }
            // +1 when i is the pair's high (better-labelled) item: negating a difference is exact, so these equal
            // LightGBM's `high - low` bit for bit.
            let sign = if label_i > label[j] { 1.0 } else { -1.0 };
            let delta_score = sign * (score_i - score[j]);
            let dcg_gap = sign * (gain_i - gain[j]);
            let paired_discount = (discount_i - discount[j]).abs();
            let mut delta_pair_ndcg = dcg_gap * paired_discount * context.inverse_max_dcg;
            if context.norm_by_distance {
                delta_pair_ndcg /= SCORE_DISTANCE_EPSILON + delta_score.abs();
            }
            let p = context.table.get(delta_score);
            let p_lambda = p * (-context.sigmoid * delta_pair_ndcg);
            let p_hessian = p * (1.0 - p) * (sigmoid_squared * delta_pair_ndcg);
            lambda_i += sign * p_lambda;
            lambda[j] -= sign * p_lambda;
            hessian_i += p_hessian;
            hessian[j] += p_hessian;
            sum_lambdas -= 2.0 * p_lambda;
        }
        lambda[i] = lambda_i;
        hessian[i] = hessian_i;
    }
    sum_lambdas
}

#[cfg(test)]
mod tests {
    use super::*;

    /// xorshift64*: deterministic test data without a dependency.
    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
        }

        fn below(&mut self, n: u64) -> u64 {
            self.next() % n
        }

        fn unit(&mut self) -> f32 {
            (self.next() >> 40) as f32 / (1u64 << 24) as f32
        }
    }

    /// A line-by-line transcription of LightGBM 3.3.5 `GetGradientsForOneQuery` and the weight loop of
    /// `RankingObjective::GetGradients`, pair loop order and sigmoid table included; accumulates in f64 where
    /// LightGBM uses f32, as silvic does.
    #[allow(clippy::too_many_arguments)]
    fn reference(
        label: &[f32],
        score: &[f32],
        weight: Option<&[f32]>,
        label_gain: &[f64],
        inverse_max_dcg: f64,
        sigmoid: f64,
        norm: bool,
        truncation_level: usize,
    ) -> (Vec<f64>, Vec<f64>) {
        let cnt = label.len();
        let table = SigmoidTable::shared(sigmoid);
        let score: Vec<f64> = score.iter().map(|&s| f64::from(s)).collect();
        let mut lambdas = vec![0.0f64; cnt];
        let mut hessians = vec![0.0f64; cnt];
        let mut sorted_idx: Vec<usize> = (0..cnt).collect();
        sorted_idx.sort_by(|&a, &b| score[b].partial_cmp(&score[a]).unwrap());
        let best_score = score[sorted_idx[0]];
        let mut worst_idx = cnt - 1;
        if worst_idx > 0 && score[sorted_idx[worst_idx]] == f64::NEG_INFINITY {
            worst_idx -= 1;
        }
        let worst_score = score[sorted_idx[worst_idx]];
        let mut sum_lambdas = 0.0;
        let mut i = 0;
        while i + 1 < cnt && i < truncation_level {
            if score[sorted_idx[i]] == f64::NEG_INFINITY {
                i += 1;
                continue;
            }
            for j in i + 1..cnt {
                if score[sorted_idx[j]] == f64::NEG_INFINITY || label[sorted_idx[i]] == label[sorted_idx[j]] {
                    continue;
                }
                let (high_rank, low_rank) = if label[sorted_idx[i]] > label[sorted_idx[j]] {
                    (i, j)
                } else {
                    (j, i)
                };
                let high = sorted_idx[high_rank];
                let low = sorted_idx[low_rank];
                let delta_score = score[high] - score[low];
                let dcg_gap = label_gain[label[high] as usize] - label_gain[label[low] as usize];
                let paired_discount = (discount(high_rank) - discount(low_rank)).abs();
                let mut delta_pair_ndcg = dcg_gap * paired_discount * inverse_max_dcg;
                if norm && best_score != worst_score {
                    delta_pair_ndcg /= 0.01f32 as f64 + delta_score.abs();
                }
                let mut p_lambda = table.get(delta_score);
                let mut p_hessian = p_lambda * (1.0 - p_lambda);
                p_lambda *= -sigmoid * delta_pair_ndcg;
                p_hessian *= sigmoid * sigmoid * delta_pair_ndcg;
                lambdas[low] -= p_lambda;
                hessians[low] += p_hessian;
                lambdas[high] += p_lambda;
                hessians[high] += p_hessian;
                sum_lambdas -= 2.0 * p_lambda;
            }
            i += 1;
        }
        if norm && sum_lambdas > 0.0 {
            let norm_factor = (1.0 + sum_lambdas).log2() / sum_lambdas;
            lambdas.iter_mut().for_each(|l| *l *= norm_factor);
            hessians.iter_mut().for_each(|h| *h *= norm_factor);
        }
        if let Some(weight) = weight {
            for j in 0..cnt {
                lambdas[j] *= f64::from(weight[j]);
                hessians[j] *= f64::from(weight[j]);
            }
        }
        (lambdas, hessians)
    }

    struct Case {
        name: String,
        label: Vec<f32>,
        score: Vec<f32>,
        weight: Option<Vec<f32>>,
        truncation_level: usize,
        norm: bool,
    }

    fn random_case(rng: &mut Rng, index: usize) -> Case {
        let cnt = 1 + rng.below(60) as usize;
        let label = (0..cnt).map(|_| rng.below(5) as f32).collect();
        let score = match index % 4 {
            // Every score tied: the first iteration, where only stable-sort order ranks the items.
            0 => vec![0.0; cnt],
            // Few distinct values: partial ties.
            1 => (0..cnt).map(|_| rng.below(4) as f32 * 0.5).collect(),
            _ => (0..cnt).map(|_| (rng.unit() - 0.5) * 8.0).collect(),
        };
        let weight = index
            .is_multiple_of(3)
            .then(|| (0..cnt).map(|_| 0.5 + rng.unit()).collect());
        let truncation_level = [1, 3, 10, 30, 1000][index % 5];
        Case {
            name: format!("case {index} (n={cnt}, truncation {truncation_level})"),
            label,
            score,
            weight,
            truncation_level,
            norm: index % 7 != 6,
        }
    }

    fn setup_for(case: &Case, label_gain: &[f64]) -> ObjectiveSetup {
        let max_dcg = max_dcg_at_k(case.truncation_level, &case.label, label_gain);
        ObjectiveSetup::LambdaRank {
            sigmoid: 1.3,
            norm: case.norm,
            truncation_level: case.truncation_level as u32,
            label_gain: label_gain.to_vec(),
            inverse_max_dcg: vec![if max_dcg > 0.0 { 1.0 / max_dcg } else { 0.0 }],
        }
    }

    fn assert_close(name: &str, what: &str, got: &[f32], want: &[f64]) {
        let scale = want.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        for (row, (&g, &w)) in got.iter().zip(want).enumerate() {
            // f32 storage rounds at 6e-8 relative; the reference differs only in the f32 cast of the weights.
            let tolerance = 1e-6 * w.abs().max(scale * 1e-3) + 1e-30;
            assert!(
                (f64::from(g) - w).abs() <= tolerance,
                "{name}: {what} of row {row}: {g} vs {w}"
            );
        }
    }

    #[test]
    fn gradients_match_lightgbm_pair_loop() {
        let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
        let label_gain = crate::config::default_label_gain();
        // One scratch for every case: buffers reused across queries of different sizes must not leak values.
        let mut scratch = Scratch::new(SigmoidTable::shared(1.3));
        for index in 0..400 {
            let case = random_case(&mut rng, index);
            let setup = setup_for(&case, &label_gain);
            let ObjectiveSetup::LambdaRank { inverse_max_dcg, .. } = &setup else {
                unreachable!()
            };
            let (want_grad, want_hess) = reference(
                &case.label,
                &case.score,
                case.weight.as_deref(),
                &label_gain,
                inverse_max_dcg[0],
                1.3,
                case.norm,
                case.truncation_level,
            );
            let mut grad = vec![f32::NAN; case.label.len()];
            let mut hess = vec![f32::NAN; case.label.len()];
            query_gradients(
                &setup,
                0,
                &case.score,
                &case.label,
                case.weight.as_deref(),
                &mut grad,
                &mut hess,
                &mut scratch,
            );
            assert_close(&case.name, "gradient", &grad, &want_grad);
            assert_close(&case.name, "hessian", &hess, &want_hess);
        }
    }

    #[test]
    fn huge_query_matches_pair_loop() {
        let mut rng = Rng(0x5eed);
        let cnt = 120_000;
        let label_gain = crate::config::default_label_gain();
        let case = Case {
            name: "120k-row query".into(),
            label: (0..cnt).map(|_| rng.below(5) as f32).collect(),
            score: (0..cnt).map(|_| (rng.unit() - 0.5) * 8.0).collect(),
            weight: None,
            truncation_level: 30,
            norm: true,
        };
        let setup = setup_for(&case, &label_gain);
        let ObjectiveSetup::LambdaRank { inverse_max_dcg, .. } = &setup else {
            unreachable!()
        };
        let (mut grad, mut hess) = (vec![f32::NAN; cnt], vec![f32::NAN; cnt]);
        let started = std::time::Instant::now();
        query_gradients(
            &setup,
            0,
            &case.score,
            &case.label,
            None,
            &mut grad,
            &mut hess,
            &mut Scratch::new(SigmoidTable::shared(1.3)),
        );
        println!("{}: gradients in {:?}", case.name, started.elapsed());
        let (want_grad, want_hess) = reference(
            &case.label,
            &case.score,
            None,
            &label_gain,
            inverse_max_dcg[0],
            1.3,
            true,
            30,
        );
        let sample: Vec<usize> = rank_order(&case.score)
            .into_iter()
            .enumerate()
            .filter(|&(rank, _)| rank < 100 || rank % 997 == 0)
            .map(|(_, row)| row)
            .collect();
        let pick = |values: &[f32]| sample.iter().map(|&row| values[row]).collect::<Vec<_>>();
        let pick64 = |values: &[f64]| sample.iter().map(|&row| values[row]).collect::<Vec<_>>();
        assert_close(&case.name, "gradient", &pick(&grad), &pick64(&want_grad));
        assert_close(&case.name, "hessian", &pick(&hess), &pick64(&want_hess));
    }

    /// `cargo test --release -p silvic-core --lib lambdarank_gradient_speed -- --ignored --nocapture`
    #[test]
    #[ignore = "timing; run in release"]
    fn lambdarank_gradient_speed() {
        let (queries, docs) = (20_000, 100);
        let mut rng = Rng(0xbe9c);
        let rows = queries * docs;
        let label: Vec<f32> = (0..rows).map(|_| rng.below(5) as f32).collect();
        let scores: Vec<f32> = (0..rows).map(|_| (rng.unit() - 0.5) * 4.0).collect();
        let boundaries: Vec<u32> = (0..=queries).map(|q| (q * docs) as u32).collect();
        let label_gain = crate::config::default_label_gain();
        let inverse_max_dcg = boundaries
            .windows(2)
            .map(|w| 1.0 / max_dcg_at_k(30, &label[w[0] as usize..w[1] as usize], &label_gain))
            .collect();
        let setup = ObjectiveSetup::LambdaRank {
            sigmoid: 1.0,
            norm: true,
            truncation_level: 30,
            label_gain,
            inverse_max_dcg,
        };
        let (mut grad, mut hess) = (vec![0.0f32; rows], vec![0.0f32; rows]);
        for round in 0..5 {
            let started = std::time::Instant::now();
            gradients(&setup, &boundaries, &scores, &label, None, &mut grad, &mut hess);
            println!(
                "round {round}: {queries} queries x {docs} docs in {:?}",
                started.elapsed()
            );
        }
    }

    #[test]
    fn min_score_items_get_no_pairs() {
        let label = [2.0, 0.0, 1.0, 0.0];
        let score = [0.5, f32::NEG_INFINITY, -0.2, 0.1];
        let label_gain = crate::config::default_label_gain();
        let case = Case {
            name: "min score".into(),
            label: label.to_vec(),
            score: score.to_vec(),
            weight: None,
            truncation_level: 30,
            norm: true,
        };
        let setup = setup_for(&case, &label_gain);
        let ObjectiveSetup::LambdaRank { inverse_max_dcg, .. } = &setup else {
            unreachable!()
        };
        let (want_grad, want_hess) = reference(&label, &score, None, &label_gain, inverse_max_dcg[0], 1.3, true, 30);
        let mut grad = [0.0f32; 4];
        let mut hess = [0.0f32; 4];
        query_gradients(
            &setup,
            0,
            &score,
            &label,
            None,
            &mut grad,
            &mut hess,
            &mut Scratch::new(SigmoidTable::shared(1.3)),
        );
        assert_eq!((grad[1], hess[1]), (0.0, 0.0), "min score: row 1");
        assert_close("min score", "gradient", &grad, &want_grad);
        assert_close("min score", "hessian", &hess, &want_hess);
    }

    #[test]
    fn sigmoid_table_follows_lightgbm() {
        let sigmoid = 2.0;
        let table = SigmoidTable::new(sigmoid);
        // `ConstructSigmoidTable` and `GetSigmoid`, transcribed.
        let (min, max) = (-50.0 / sigmoid / 2.0, 50.0 / sigmoid / 2.0);
        let factor = (1024.0 * 1024.0) / (max - min);
        let entry = |i: usize| 1.0 / (1.0 + ((i as f64 / factor + min) * sigmoid).exp());
        let want = |x: f64| {
            if x <= min {
                entry(0)
            } else if x >= max {
                entry(1024 * 1024 - 1)
            } else {
                entry(((x - min) * factor) as usize)
            }
        };
        for x in [-100.0, min, -3.7, -1e-9, 0.0, 1e-9, 0.25, 4.0, max - 1e-12, max, 30.0] {
            assert_eq!(table.get(x), want(x), "sigmoid table at {x}");
        }
        assert!((table.get(0.0) - 0.5).abs() < 1e-4, "sigmoid table near 0");
    }

    #[test]
    fn max_dcg_follows_lightgbm() {
        let gain = [0.0, 1.0, 3.0, 7.0];
        let labels = [1.0, 3.0, 0.0, 2.0, 3.0];
        let want = 7.0 * discount(0) + 7.0 * discount(1) + 3.0 * discount(2);
        assert_eq!(max_dcg_at_k(3, &labels, &gain), want);
        assert_eq!(max_dcg_at_k(3, &[0.0, 0.0], &gain), 0.0);
        assert_eq!(max_dcg_at_k(10, &[1.0], &gain), 1.0);
    }

    #[test]
    fn rank_labels_are_checked() {
        assert!(check_labels(&[0.0, 1.0, 4.0], 5).is_ok());
        let err = check_labels(&[0.0, 5.0], 5).unwrap_err().to_string();
        assert!(
            err.contains("label 5 is not less than the number of label mappings (5)"),
            "{err}"
        );
        assert!(check_labels(&[0.5], 5).unwrap_err().to_string().contains("met 0.5"));
        assert!(check_labels(&[-1.0], 5)
            .unwrap_err()
            .to_string()
            .contains("non-negative"));
    }
}
