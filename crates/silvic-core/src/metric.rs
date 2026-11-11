//! Evaluation metrics, as LightGBM 3.3.5 names and computes them: `NDCGMetric` (`ndcg@k`), `RegressionMetric<L2Metric>`
//! (`l2`), `RegressionMetric<RMSEMetric>` (`rmse`), `BinaryMetric<BinaryLoglossMetric>` (`binary_logloss`) and
//! `AUCMetric` (`auc`).
//!
//! Scores go through the objective's `ConvertOutput` first (the sigmoid for `binary`, identity otherwise), as LightGBM
//! passes the objective to pointwise metrics. Across ranks, `auc` gathers every (score, label, weight) and computes on
//! every rank; the others reduce their (sum, total weight) partials with `allreduce_sum_f64`.

use crate::collective::Collective;
use crate::config::{Metric, Objective, Params};
use crate::data::BinnedDataset;
use crate::error::{Result, SilvicError};
use crate::objective::lambdarank::{self, discount, max_dcg_at_k};
use crate::objective::{effective_label_gain, K_EPSILON};

/// LightGBM `DCGCalculator::DefaultEvalAt`, used when `eval_at` is empty.
const DEFAULT_EVAL_AT: [usize; 5] = [1, 2, 3, 4, 5];

/// One metric value on one dataset, named as LightGBM names it (`ndcg@5`, `l2`, `binary_logloss`, `auc`).
#[derive(Debug, Clone, PartialEq)]
pub struct EvalResult {
    pub dataset: String,
    pub metric: String,
    pub value: f64,
    pub higher_is_better: bool,
}

/// Every configured metric of `data` with raw `scores`, reduced over every rank.
pub fn evaluate(
    params: &Params,
    dataset_name: &str,
    scores: &[f32],
    data: &BinnedDataset,
    collective: &mut dyn Collective,
) -> Result<Vec<EvalResult>> {
    let mut results = Vec::new();
    for metric in params.effective_metrics() {
        let mut push = |name: String, value: f64| {
            results.push(EvalResult {
                dataset: dataset_name.to_string(),
                metric: name,
                value,
                higher_is_better: metric.higher_is_better(),
            })
        };
        match metric {
            Metric::Ndcg => {
                let eval_at: &[usize] = if params.eval_at.is_empty() {
                    &DEFAULT_EVAL_AT
                } else {
                    &params.eval_at
                };
                for (k, value) in eval_at.iter().zip(ndcg(params, eval_at, scores, data, collective)?) {
                    push(format!("ndcg@{k}"), value);
                }
            }
            Metric::L2 | Metric::Rmse | Metric::BinaryLogloss => {
                let value = pointwise(params, metric, scores, data, collective)?;
                push(metric.name().to_string(), value);
            }
            Metric::Auc => push(metric.name().to_string(), auc(scores, data, collective)?),
        }
    }
    Ok(results)
}

/// The objective's `ConvertOutput` of a raw score.
fn convert_output(params: &Params, score: f32) -> f64 {
    let score = f64::from(score);
    match params.objective {
        Objective::Binary => 1.0 / (1.0 + (-params.sigmoid * score).exp()),
        Objective::Regression | Objective::LambdaRank => score,
    }
}

/// `BinaryLoglossMetric::LossOnPoint`: the probability of the true class floored at `kEpsilon`. The floor's loss is
/// `-std::log(kEpsilon)` of a float, so LightGBM computes it in f32 (34.538776397705078, not 34.538776391283).
fn logloss(label: f32, prob: f64) -> f64 {
    let p_true = if label <= 0.0 { 1.0 - prob } else { prob };
    if p_true > K_EPSILON {
        -p_true.ln()
    } else {
        f64::from(-(K_EPSILON as f32).ln())
    }
}

/// `RegressionMetric` / `BinaryMetric::Eval`: weighted mean pointwise loss (`rmse` takes its square root).
fn pointwise(
    params: &Params,
    metric: Metric,
    scores: &[f32],
    data: &BinnedDataset,
    collective: &mut dyn Collective,
) -> Result<f64> {
    let loss = |row: usize| {
        let output = convert_output(params, scores[row]);
        let label = data.label[row];
        match metric {
            Metric::BinaryLogloss => logloss(label, output),
            _ => (output - f64::from(label)).powi(2),
        }
    };
    let mut sums = match data.weight.as_deref() {
        Some(weight) => (0..data.num_rows).fold([0.0, 0.0], |[l, w], row| {
            [l + loss(row) * f64::from(weight[row]), w + f64::from(weight[row])]
        }),
        None => [(0..data.num_rows).map(loss).sum(), data.num_rows as f64],
    };
    collective.allreduce_sum_f64(&mut sums)?;
    let mean = sums[0] / sums[1];
    Ok(if metric == Metric::Rmse { mean.sqrt() } else { mean })
}

/// `NDCGMetric::Eval`: per `k`, the mean NDCG@k over queries, weighted by `Metadata::query_weights` (the f32 mean row
/// weight of the query) when the data has weights. A query whose max DCG is 0 counts as 1, unweighted, as in LightGBM.
fn ndcg(
    params: &Params,
    eval_at: &[usize],
    scores: &[f32],
    data: &BinnedDataset,
    collective: &mut dyn Collective,
) -> Result<Vec<f64>> {
    let boundaries = data
        .query_boundaries
        .as_deref()
        .ok_or_else(|| SilvicError::data("the NDCG metric requires query information"))?;
    let label_gain = effective_label_gain(params);
    lambdarank::check_labels(&data.label, label_gain.len())?;
    // Per k the NDCG sum, then the total query weight.
    let mut sums = vec![0.0; eval_at.len() + 1];
    for w in boundaries.windows(2) {
        let rows = w[0] as usize..w[1] as usize;
        let query_weight = data
            .weight
            .as_deref()
            .map_or(1.0, |weight| query_weight(&weight[rows.clone()]));
        sums[eval_at.len()] += query_weight;
        query_ndcg(
            eval_at,
            &data.label[rows.clone()],
            &scores[rows],
            &label_gain,
            query_weight,
            &mut sums,
        );
    }
    collective.allreduce_sum_f64(&mut sums)?;
    let total_weight = sums[eval_at.len()];
    Ok(sums[..eval_at.len()].iter().map(|s| s / total_weight).collect())
}

/// `Metadata::LoadQueryWeights`: the f32 sum of the query's row weights divided by its row count, in f32.
fn query_weight(weight: &[f32]) -> f64 {
    let sum = weight.iter().fold(0.0f32, |acc, &w| acc + w);
    f64::from(sum / weight.len() as f32)
}

/// Adds the query's weighted NDCG@k for every `k` to `sums[..eval_at.len()]`.
fn query_ndcg(eval_at: &[usize], label: &[f32], scores: &[f32], label_gain: &[f64], weight: f64, sums: &mut [f64]) {
    let order = lambdarank::rank_order(scores);
    for (j, &k) in eval_at.iter().enumerate() {
        let max_dcg = max_dcg_at_k(k, label, label_gain);
        if max_dcg <= 0.0 {
            sums[j] += 1.0;
            continue;
        }
        let dcg: f64 = order
            .iter()
            .take(k)
            .enumerate()
            .map(|(rank, &row)| label_gain[label[row] as usize] * discount(rank))
            .sum();
        sums[j] += dcg * (1.0 / max_dcg) * weight;
    }
}

/// `AUCMetric::Eval`: weighted ROC AUC over every rank's rows; tied scores count half.
fn auc(scores: &[f32], data: &BinnedDataset, collective: &mut dyn Collective) -> Result<f64> {
    let (scores, label, weight) = if collective.world_size() > 1 {
        gather_rows(scores, data, collective)?
    } else {
        (scores.to_vec(), data.label.clone(), data.weight.clone())
    };
    Ok(weighted_auc(&scores, &label, weight.as_deref()))
}

/// Every rank's (score, label, weight) rows, concatenated in rank order.
#[allow(clippy::type_complexity)]
fn gather_rows(
    scores: &[f32],
    data: &BinnedDataset,
    collective: &mut dyn Collective,
) -> Result<(Vec<f32>, Vec<f32>, Option<Vec<f32>>)> {
    let local: Vec<u8> = (0..data.num_rows)
        .flat_map(|row| {
            let weight = data.weight.as_deref().map_or(1.0, |w| w[row]);
            [scores[row], data.label[row], weight]
                .into_iter()
                .flat_map(f32::to_le_bytes)
        })
        .collect();
    let mut all = (Vec::new(), Vec::new(), Vec::new());
    for bytes in collective.allgather_bytes(&local)? {
        for row in bytes.as_chunks::<12>().0 {
            let value = |i: usize| f32::from_le_bytes([row[i * 4], row[i * 4 + 1], row[i * 4 + 2], row[i * 4 + 3]]);
            all.0.push(value(0));
            all.1.push(value(1));
            all.2.push(value(2));
        }
    }
    let weight = data.weight.is_some().then_some(all.2);
    Ok((all.0, all.1, weight))
}

fn weighted_auc(scores: &[f32], label: &[f32], weight: Option<&[f32]>) -> f64 {
    if scores.is_empty() {
        return 1.0;
    }
    let order = lambdarank::rank_order(scores);
    let weight_of = |row: usize| weight.map_or(1.0, |w| f64::from(w[row]));
    let sum_weights: f64 = match weight {
        Some(w) => w.iter().map(|&x| f64::from(x)).sum(),
        None => scores.len() as f64,
    };
    let (mut cur_pos, mut cur_neg, mut sum_pos, mut accum) = (0.0, 0.0, 0.0, 0.0);
    let mut threshold = scores[order[0]];
    for &row in &order {
        if scores[row] != threshold {
            threshold = scores[row];
            accum += cur_neg * (cur_pos * 0.5 + sum_pos);
            sum_pos += cur_pos;
            cur_neg = 0.0;
            cur_pos = 0.0;
        }
        if label[row] > 0.0 {
            cur_pos += weight_of(row);
        } else {
            cur_neg += weight_of(row);
        }
    }
    accum += cur_neg * (cur_pos * 0.5 + sum_pos);
    sum_pos += cur_pos;
    if sum_pos > 0.0 && sum_pos != sum_weights {
        accum / (sum_pos * (sum_weights - sum_pos))
    } else {
        1.0
    }
}
