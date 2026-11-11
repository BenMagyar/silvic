//! Gradients of the training objectives on the host: the CPU backend's implementation and the reference the GPU
//! kernels are tested against.
//!
//! Ports of LightGBM 3.3.5 `RegressionL2loss`, `BinaryLogloss` and `LambdarankNDCG`: label checks in `setup`
//! (`ObjectiveFunction::Init`), `BoostFromScore` in [`init_score`], `GetGradients` in [`gradients`]. Binary and
//! lambdarank gradients are computed in f64 and stored as f32, as LightGBM stores `score_t`; L2 gradients use f32
//! arithmetic so every backend (Metal has no f64) produces the same bits.

mod binary;
pub mod lambdarank;
mod regression;

use crate::backend::ObjectiveSetup;
use crate::collective::Collective;
use crate::config::{default_label_gain, Objective, Params};
use crate::data::BinnedDataset;
use crate::error::{Result, SilvicError};

/// LightGBM `kEpsilon` (`const score_t kEpsilon = 1e-15f`): a float, so its double value is not exactly 1e-15.
pub(crate) const K_EPSILON: f64 = 1e-15_f32 as f64;

/// Validates labels for the objective and precomputes what gradients need (lambdarank's inverse max DCGs).
pub fn setup(params: &Params, train: &BinnedDataset) -> Result<ObjectiveSetup> {
    check_finite_labels(&train.label)?;
    match params.objective {
        Objective::Regression => Ok(ObjectiveSetup::Regression),
        // LightGBM's `BinaryLogloss` takes any label > 0 as positive and everything else as negative; so does silvic.
        Objective::Binary => Ok(ObjectiveSetup::Binary {
            sigmoid: params.sigmoid,
        }),
        Objective::LambdaRank => lambdarank_setup(params, train),
    }
}

fn lambdarank_setup(params: &Params, train: &BinnedDataset) -> Result<ObjectiveSetup> {
    let boundaries = train
        .query_boundaries
        .as_deref()
        .ok_or_else(|| SilvicError::data("ranking tasks require query information"))?;
    let label_gain = effective_label_gain(params);
    lambdarank::check_labels(&train.label, label_gain.len())?;
    let truncation_level = u32::try_from(params.lambdarank_truncation_level).unwrap_or(u32::MAX);
    let inverse_max_dcg = boundaries
        .windows(2)
        .map(|w| {
            let max_dcg = lambdarank::max_dcg_at_k(
                truncation_level as usize,
                &train.label[w[0] as usize..w[1] as usize],
                &label_gain,
            );
            if max_dcg > 0.0 {
                1.0 / max_dcg
            } else {
                0.0
            }
        })
        .collect();
    Ok(ObjectiveSetup::LambdaRank {
        sigmoid: params.sigmoid,
        norm: params.lambdarank_norm,
        truncation_level,
        label_gain,
        inverse_max_dcg,
    })
}

/// `label_gain`, or LightGBM's default (`DCGCalculator::DefaultLabelGain`) when empty.
pub(crate) fn effective_label_gain(params: &Params) -> Vec<f64> {
    if params.label_gain.is_empty() {
        default_label_gain()
    } else {
        params.label_gain.clone()
    }
}

fn check_finite_labels(label: &[f32]) -> Result<()> {
    match label.iter().position(|l| !l.is_finite()) {
        Some(row) => Err(SilvicError::data(format!("label of row {row} is {}", label[row]))),
        None => Ok(()),
    }
}

/// The constant every hessian equals, when there is one (L2 regression without weights).
pub fn constant_hessian(setup: &ObjectiveSetup, weight: Option<&[f32]>) -> Option<f64> {
    match (setup, weight) {
        (ObjectiveSetup::Regression, None) => Some(1.0),
        _ => None,
    }
}

/// LightGBM `BoostFromAverage` over every rank's rows; 0 when `boost_from_average` is off or the objective has none.
pub fn init_score(params: &Params, train: &BinnedDataset, collective: &mut dyn Collective) -> Result<f64> {
    if !params.boost_from_average || params.objective == Objective::LambdaRank {
        return Ok(0.0);
    }
    let weight = train.weight.as_deref();
    let mut sums = match params.objective {
        Objective::Binary => binary::label_sums(&train.label, weight),
        _ => regression::label_sums(&train.label, weight),
    };
    collective.allreduce_sum_f64(&mut sums)?;
    let [sum_label, sum_weight] = sums;
    let score = match params.objective {
        Objective::Binary => binary::boost_from_score(sum_label, sum_weight, params.sigmoid),
        _ => sum_label / sum_weight,
    };
    // `GBDT::BoostFromAverage` drops an init score indistinguishable from zero.
    Ok(if score.abs() > K_EPSILON { score } else { 0.0 })
}

/// Gradients and hessians of every row, parallel over rows or queries.
pub fn gradients(
    setup: &ObjectiveSetup,
    scores: &[f32],
    label: &[f32],
    weight: Option<&[f32]>,
    query_boundaries: Option<&[u32]>,
    grad: &mut [f32],
    hess: &mut [f32],
) {
    match setup {
        ObjectiveSetup::Regression => regression::gradients(scores, label, weight, grad, hess),
        ObjectiveSetup::Binary { sigmoid } => binary::gradients(*sigmoid, scores, label, weight, grad, hess),
        ObjectiveSetup::LambdaRank { .. } => {
            let boundaries = query_boundaries.expect("lambdarank gradients need query boundaries");
            lambdarank::gradients(setup, boundaries, scores, label, weight, grad, hess);
        }
    }
}

/// `LambdarankNDCG::GetGradientsForOneQuery` (weights applied) for the query `query` of the training set.
///
/// Every slice holds only the query's rows; the result equals [`gradients`] on those rows.
#[allow(clippy::too_many_arguments)]
pub fn lambdarank_query(
    setup: &ObjectiveSetup,
    query: usize,
    scores: &[f32],
    label: &[f32],
    weight: Option<&[f32]>,
    grad: &mut [f32],
    hess: &mut [f32],
) {
    let ObjectiveSetup::LambdaRank { sigmoid, .. } = setup else {
        panic!("lambdarank gradients need a lambdarank objective setup");
    };
    let mut scratch = lambdarank::Scratch::new(lambdarank::SigmoidTable::shared(*sigmoid));
    lambdarank::query_gradients(setup, query, scores, label, weight, grad, hess, &mut scratch);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::run_ranks;

    fn dataset(label: Vec<f32>, weight: Option<Vec<f32>>, query_boundaries: Option<Vec<u32>>) -> BinnedDataset {
        BinnedDataset {
            num_rows: label.len(),
            row_offset: 0,
            features: Vec::new(),
            groups: Vec::new(),
            total_bins: 0,
            label,
            weight,
            query_boundaries,
        }
    }

    fn params(objective: Objective) -> Params {
        Params {
            objective,
            ..Params::default()
        }
    }

    #[test]
    fn init_score_is_the_global_weighted_mean_over_ranks() {
        let labels = [vec![1.0f32, 2.0, 4.0], vec![0.5f32, 7.0]];
        let weights = [vec![1.0f32, 0.5, 2.0], vec![3.0f32, 0.25]];
        let all_rows = dataset(labels.concat(), Some(weights.concat()), None);
        for objective in [Objective::Regression, Objective::Binary] {
            let params = params(objective);
            let single = init_score(&params, &all_rows, &mut crate::LocalCollective).expect("init score");
            let per_rank = run_ranks(2, |rank, collective| {
                let local = dataset(labels[rank].clone(), Some(weights[rank].clone()), None);
                init_score(&params, &local, collective).expect("init score")
            });
            assert_eq!(per_rank, vec![single; 2], "{objective:?}: ranks agree with one process");
        }
        let regression = init_score(&params(Objective::Regression), &all_rows, &mut crate::LocalCollective).unwrap();
        let want = (1.0 + 1.0 + 8.0 + 1.5 + 1.75) / (1.0 + 0.5 + 2.0 + 3.0 + 0.25);
        assert!((regression - want).abs() < 1e-12, "regression: {regression} vs {want}");
        // Every label is positive: p clamps to 1 - kEpsilon.
        let binary = init_score(&params(Objective::Binary), &all_rows, &mut crate::LocalCollective).unwrap();
        let p = 1.0 - K_EPSILON;
        assert_eq!(binary, (p / (1.0 - p)).ln(), "binary: clamped rate");
    }

    #[test]
    fn init_score_of_binary_is_the_log_odds_over_ranks() {
        let labels = [vec![1.0f32, 0.0, 0.0], vec![1.0f32, 0.0]];
        let mut params = params(Objective::Binary);
        params.sigmoid = 2.0;
        let per_rank = run_ranks(2, |rank, collective| {
            init_score(&params, &dataset(labels[rank].clone(), None, None), collective).expect("init score")
        });
        let want = (0.4f64 / 0.6).ln() / 2.0;
        for score in per_rank {
            assert!((score - want).abs() < 1e-15, "binary: {score} vs {want}");
        }
    }

    #[test]
    fn init_score_is_zero_without_boost_from_average_or_for_ranking() {
        let data = dataset(vec![3.0, 1.0], None, Some(vec![0, 2]));
        let mut off = params(Objective::Regression);
        off.boost_from_average = false;
        assert_eq!(init_score(&off, &data, &mut crate::LocalCollective).unwrap(), 0.0);
        let rank = params(Objective::LambdaRank);
        assert_eq!(init_score(&rank, &data, &mut crate::LocalCollective).unwrap(), 0.0);
    }

    #[test]
    fn constant_hessian_only_for_unweighted_regression() {
        assert_eq!(constant_hessian(&ObjectiveSetup::Regression, None), Some(1.0));
        assert_eq!(constant_hessian(&ObjectiveSetup::Regression, Some(&[1.0, 1.0])), None);
        assert_eq!(constant_hessian(&ObjectiveSetup::Binary { sigmoid: 1.0 }, None), None);
    }

    #[test]
    fn setup_checks_labels() {
        let nan = dataset(vec![1.0, f32::NAN], None, None);
        assert!(setup(&params(Objective::Regression), &nan).is_err());
        let ranking = params(Objective::LambdaRank);
        assert!(
            setup(&ranking, &dataset(vec![1.0, 2.0], None, None)).is_err(),
            "no queries"
        );
        let err = setup(&ranking, &dataset(vec![1.0, 31.0], None, Some(vec![0, 2]))).unwrap_err();
        assert!(err.to_string().contains("label 31"), "{err}");
        let ok = setup(&ranking, &dataset(vec![1.0, 2.0, 0.0, 0.0], None, Some(vec![0, 2, 4]))).expect("setup");
        let ObjectiveSetup::LambdaRank { inverse_max_dcg, .. } = ok else {
            panic!("lambdarank setup")
        };
        let max_dcg = 3.0 * lambdarank::discount(0) + 1.0 * lambdarank::discount(1);
        assert_eq!(inverse_max_dcg, vec![1.0 / max_dcg, 0.0]);
    }

    #[test]
    fn pointwise_gradients_follow_lightgbm() {
        let scores = [0.5f32, -1.0, 2.0];
        let label = [1.0f32, 0.0, 3.0];
        let weight = [2.0f32, 0.5, 1.0];
        let (mut grad, mut hess) = ([0.0f32; 3], [0.0f32; 3]);
        gradients(
            &ObjectiveSetup::Regression,
            &scores,
            &label,
            Some(&weight),
            None,
            &mut grad,
            &mut hess,
        );
        assert_eq!(grad, [-1.0, -0.5, -1.0]);
        assert_eq!(hess, weight);
        let sigmoid = 1.5f64;
        gradients(
            &ObjectiveSetup::Binary { sigmoid },
            &scores,
            &label,
            None,
            None,
            &mut grad,
            &mut hess,
        );
        for row in 0..3 {
            let y = if label[row] > 0.0 { 1.0 } else { -1.0 };
            let response = -y * sigmoid / (1.0 + (y * sigmoid * f64::from(scores[row])).exp());
            assert_eq!(grad[row], response as f32, "binary: gradient of row {row}");
            let abs = response.abs();
            assert_eq!(
                hess[row],
                (abs * (sigmoid - abs)) as f32,
                "binary: hessian of row {row}"
            );
        }
    }

    #[test]
    fn lambdarank_gradients_run_every_query() {
        let label = vec![2.0f32, 0.0, 1.0, 0.0, 3.0, 1.0, 1.0];
        let boundaries = vec![0u32, 3, 3, 7];
        let data = dataset(label.clone(), None, Some(boundaries.clone()));
        let objective = setup(&params(Objective::LambdaRank), &data).expect("setup");
        let scores = [0.1f32, 0.3, -0.2, 0.0, 0.5, 0.5, -1.0];
        let (mut grad, mut hess) = (vec![0.0f32; 7], vec![0.0f32; 7]);
        gradients(
            &objective,
            &scores,
            &label,
            None,
            Some(&boundaries),
            &mut grad,
            &mut hess,
        );
        for (query, w) in boundaries.windows(2).enumerate() {
            let rows = w[0] as usize..w[1] as usize;
            let (mut g, mut h) = (vec![0.0f32; rows.len()], vec![0.0f32; rows.len()]);
            lambdarank_query(
                &objective,
                query,
                &scores[rows.clone()],
                &label[rows.clone()],
                None,
                &mut g,
                &mut h,
            );
            assert_eq!(&grad[rows.clone()], &g[..], "query {query}: gradients");
            assert_eq!(&hess[rows], &h[..], "query {query}: hessians");
        }
        assert!(grad.iter().all(|g| g.is_finite()) && grad[0] < 0.0 && grad[1] > 0.0);
    }
}
