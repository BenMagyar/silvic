//! L2 regression: LightGBM 3.3.5 `RegressionL2loss::GetGradients` and `BoostFromScore`.

use rayon::prelude::*;

/// Rows per rayon task: large enough that scheduling costs nothing next to the arithmetic.
pub(super) const ROW_CHUNK: usize = 1 << 14;

/// `[sum label * weight, sum weight]` in row order. The product is a float product, as `label_t * label_t` is in
/// LightGBM.
pub(super) fn label_sums(label: &[f32], weight: Option<&[f32]>) -> [f64; 2] {
    match weight {
        Some(weight) => label.iter().zip(weight).fold([0.0, 0.0], |[l, w], (&y, &wi)| {
            [l + f64::from(y * wi), w + f64::from(wi)]
        }),
        None => [label.iter().map(|&y| f64::from(y)).sum(), label.len() as f64],
    }
}

/// `g = (s - y) * w`, `h = w`.
pub(super) fn gradients(scores: &[f32], label: &[f32], weight: Option<&[f32]>, grad: &mut [f32], hess: &mut [f32]) {
    grad.par_chunks_mut(ROW_CHUNK)
        .zip(hess.par_chunks_mut(ROW_CHUNK))
        .enumerate()
        .for_each(|(chunk, (grad, hess))| {
            let start = chunk * ROW_CHUNK;
            for (i, (g, h)) in grad.iter_mut().zip(hess.iter_mut()).enumerate() {
                let row = start + i;
                // f32 arithmetic, not LightGBM's double: Metal has no f64, and L2 gradients must be bit-identical on
                // every backend.
                let residual = scores[row] - label[row];
                match weight {
                    Some(weight) => {
                        *g = residual * weight[row];
                        *h = weight[row];
                    }
                    None => {
                        *g = residual;
                        *h = 1.0;
                    }
                }
            }
        });
}
