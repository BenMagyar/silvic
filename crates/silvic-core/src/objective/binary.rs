//! Binary log loss: LightGBM 3.3.5 `BinaryLogloss::GetGradients` and `BoostFromScore`, without `is_unbalance` and
//! `scale_pos_weight` (both label weights are 1). A label > 0 is positive, anything else negative, as LightGBM's
//! default `is_pos_`.

use rayon::prelude::*;

use super::regression::ROW_CHUNK;
use super::K_EPSILON;

/// `[sum is_pos * weight, sum weight]` in row order.
pub(super) fn label_sums(label: &[f32], weight: Option<&[f32]>) -> [f64; 2] {
    let positive = |y: f32| if y > 0.0 { 1.0f32 } else { 0.0 };
    match weight {
        Some(weight) => label.iter().zip(weight).fold([0.0, 0.0], |[l, w], (&y, &wi)| {
            [l + f64::from(positive(y) * wi), w + f64::from(wi)]
        }),
        None => [label.iter().map(|&y| f64::from(positive(y))).sum(), label.len() as f64],
    }
}

/// `log(p / (1 - p)) / sigmoid` with the positive rate clamped to `[kEpsilon, 1 - kEpsilon]`.
pub(super) fn boost_from_score(sum_positive: f64, sum_weight: f64, sigmoid: f64) -> f64 {
    let p = (sum_positive / sum_weight).clamp(K_EPSILON, 1.0 - K_EPSILON);
    (p / (1.0 - p)).ln() / sigmoid
}

pub(super) fn gradients(
    sigmoid: f64,
    scores: &[f32],
    label: &[f32],
    weight: Option<&[f32]>,
    grad: &mut [f32],
    hess: &mut [f32],
) {
    grad.par_chunks_mut(ROW_CHUNK)
        .zip(hess.par_chunks_mut(ROW_CHUNK))
        .enumerate()
        .for_each(|(chunk, (grad, hess))| {
            let start = chunk * ROW_CHUNK;
            for (i, (g, h)) in grad.iter_mut().zip(hess.iter_mut()).enumerate() {
                let row = start + i;
                let sign = if label[row] > 0.0 { 1.0 } else { -1.0 };
                let response = -sign * sigmoid / (1.0 + (sign * sigmoid * f64::from(scores[row])).exp());
                let abs_response = response.abs();
                let w = weight.map_or(1.0, |w| f64::from(w[row]));
                *g = (response * w) as f32;
                *h = (abs_response * (sigmoid - abs_response) * w) as f32;
            }
        });
}
