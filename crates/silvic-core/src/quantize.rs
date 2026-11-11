//! Gradient quantization (LightGBM 4's `GradientDiscretizer`, `use_quantized_grad`) and the counter-based random
//! numbers behind stochastic rounding and bagging.
//!
//! Every random draw is a pure function of (seed, stream, iteration, global row id), so it is the same on every
//! backend and on whichever rank holds the row. The GPU kernels (`silvic-gpu/kernels/kernels.kern`) implement the
//! same functions with the same operation order; [`mix32`], [`uniform`] and [`quantize_row`] are the contract they
//! are tested against.

/// Random-number streams; one per use so the draws for one row never correlate.
pub const STREAM_GRADIENT: u32 = 0;
pub const STREAM_HESSIAN: u32 = 1;
pub const STREAM_BAGGING: u32 = 2;

/// `lowbias32` (Chris Wellons' hash-prospector): a bijective 32-bit mixer with low bias.
#[inline]
pub fn mix32(mut x: u32) -> u32 {
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb_352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846c_a68b);
    x ^= x >> 16;
    x
}

/// A uniform draw in `[0, 1)` with 24 bits of precision, exact in f32.
#[inline]
pub fn uniform(seed: u32, stream: u32, iteration: u32, row: u64) -> f32 {
    let row_hash = mix32((row as u32) ^ mix32((row >> 32) as u32));
    let h = mix32(seed ^ mix32(stream ^ mix32(iteration ^ row_hash)));
    (h >> 8) as f32 * (1.0 / 16_777_216.0)
}

/// What one iteration's quantization needs; mirrored by `QuantParams` in the kernels.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QuantSpec {
    /// `1 / grad_scale`, where `grad_scale = max |g| / (num_grad_quant_bins / 2)` over every rank.
    pub inv_grad_scale: f32,
    /// `1 / hess_scale`, where `hess_scale = max h / num_grad_quant_bins`; unused with a constant hessian.
    pub inv_hess_scale: f32,
    pub grad_scale: f64,
    pub hess_scale: f64,
    /// `num_grad_quant_bins / 2`.
    pub grad_max_q: i32,
    /// `num_grad_quant_bins`.
    pub hess_max_q: i32,
    /// Every hessian equals `hess_scale`, so `h_q` is 1 and hessian sums are exact row counts.
    pub constant_hessian: bool,
    pub stochastic: bool,
    pub seed: u32,
    pub iteration: u32,
}

impl QuantSpec {
    /// LightGBM `GradientDiscretizer::DiscretizeGradients` scales from the global maxima. All-zero gradients get
    /// scale 1 so the inverse stays finite.
    pub fn new(
        max_abs_grad: f64,
        max_hess: f64,
        constant_hessian: Option<f64>,
        num_bins: u32,
        stochastic: bool,
        seed: u32,
        iteration: u32,
    ) -> QuantSpec {
        let grad_max_q = (num_bins / 2) as i32;
        let hess_max_q = num_bins as i32;
        let grad_scale = positive_or_one(max_abs_grad / f64::from(grad_max_q));
        let hess_scale = match constant_hessian {
            Some(value) => positive_or_one(value),
            None => positive_or_one(max_hess / f64::from(hess_max_q)),
        };
        QuantSpec {
            inv_grad_scale: (1.0 / grad_scale) as f32,
            inv_hess_scale: (1.0 / hess_scale) as f32,
            grad_scale,
            hess_scale,
            grad_max_q,
            hess_max_q,
            constant_hessian: constant_hessian.is_some(),
            stochastic,
            seed,
            iteration,
        }
    }

    /// Rows per GPU threadgroup so a packed i32 (two i16 halves) histogram bin never overflows.
    pub fn max_rows_per_packed_block(&self) -> u32 {
        (32_767 / self.grad_max_q.max(self.hess_max_q)) as u32
    }
}

fn positive_or_one(value: f64) -> f64 {
    if value > 0.0 && value.is_finite() {
        value
    } else {
        1.0
    }
}

/// Quantizes one row's gradient and hessian and packs them as `h_q * 65536 + g_q`, the layout the histogram
/// kernels add in one 32-bit atomic.
#[inline]
pub fn quantize_row(spec: &QuantSpec, grad: f32, hess: f32, row: u64) -> i32 {
    let g_q = quantize_grad(spec, grad, row);
    let h_q = quantize_hess(spec, hess, row);
    h_q * 65_536 + g_q
}

#[inline]
fn quantize_grad(spec: &QuantSpec, grad: f32, row: u64) -> i32 {
    let q = if spec.stochastic {
        let u = uniform(spec.seed, STREAM_GRADIENT, spec.iteration, row);
        if grad >= 0.0 {
            grad.mul_add(spec.inv_grad_scale, u).trunc()
        } else {
            grad.mul_add(spec.inv_grad_scale, -u).trunc()
        }
    } else {
        (grad * spec.inv_grad_scale).round_ties_even()
    };
    (q as i32).clamp(-spec.grad_max_q, spec.grad_max_q)
}

#[inline]
fn quantize_hess(spec: &QuantSpec, hess: f32, row: u64) -> i32 {
    if spec.constant_hessian {
        return 1;
    }
    let q = if spec.stochastic {
        let u = uniform(spec.seed, STREAM_HESSIAN, spec.iteration, row);
        hess.mul_add(spec.inv_hess_scale, u).trunc()
    } else {
        (hess * spec.inv_hess_scale).round_ties_even()
    };
    (q as i32).clamp(0, spec.hess_max_q)
}

/// Splits a packed value back into `(g_q, h_q)`.
#[inline]
pub fn unpack(packed: i32) -> (i32, i32) {
    let g = i32::from(packed as u32 as u16 as i16);
    (g, (packed - g) >> 16)
}

/// Whether `row` is in the bag of `bag_round` (LightGBM bags rows independently, also for lambdarank, in 3.3.5).
#[inline]
pub fn in_bag(bagging_seed: u32, bag_round: u32, row: u64, fraction: f64) -> bool {
    f64::from(uniform(bagging_seed, STREAM_BAGGING, bag_round, row)) < fraction
}

/// The power-of-two scale for exact fixed-point leaf sums: `value * scale` is exact in f32 and at most `2^30` in
/// magnitude for every `|value| <= max_abs`, so 2^33 rows still sum inside i64.
pub fn fixed_point_scale(max_abs: f64) -> f32 {
    if !max_abs.is_finite() || max_abs <= 0.0 {
        return 1.0;
    }
    let exponent = 30 - max_abs.log2().ceil() as i32;
    2f32.powi(exponent.clamp(-126, 127))
}

/// `rint(value * scale)` as an i64, matching the kernels' `(i64)F32_RINT(v * scale)`.
#[inline]
pub fn to_fixed(value: f32, scale: f32) -> i64 {
    (value * scale).round_ties_even() as i64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_is_in_unit_interval_and_row_keyed() {
        let draws: Vec<f32> = (0..10_000).map(|row| uniform(7, STREAM_GRADIENT, 3, row)).collect();
        assert!(draws.iter().all(|u| (0.0..1.0).contains(u)));
        let mean = draws.iter().map(|&u| f64::from(u)).sum::<f64>() / draws.len() as f64;
        assert!((mean - 0.5).abs() < 0.02, "mean {mean}");
        assert_ne!(
            uniform(7, STREAM_GRADIENT, 3, 1 << 32),
            uniform(7, STREAM_GRADIENT, 3, 0)
        );
    }

    #[test]
    fn stochastic_rounding_is_unbiased() {
        let spec = QuantSpec::new(2.0, 1.0, None, 4, true, 11, 0);
        let grad = 0.3f32;
        let sum: i64 = (0..200_000u64)
            .map(|row| i64::from(unpack(quantize_row(&spec, grad, 0.5, row)).0))
            .sum();
        let mean = sum as f64 / 200_000.0 * spec.grad_scale;
        assert!((mean - 0.3).abs() < 0.01, "mean {mean}");
    }

    #[test]
    fn pack_round_trips_negative_gradients() {
        for g in -32..=32 {
            for h in 0..=64 {
                assert_eq!(unpack(h * 65_536 + g), (g, h));
            }
        }
    }

    #[test]
    fn constant_hessian_counts_rows() {
        let spec = QuantSpec::new(1.0, 1.0, Some(1.0), 4, true, 1, 0);
        assert_eq!(unpack(quantize_row(&spec, -0.7, 1.0, 5)).1, 1);
    }

    #[test]
    fn fixed_point_scale_bounds_magnitude() {
        let scale = fixed_point_scale(3.7);
        assert!(f64::from(3.7f32 * scale) <= f64::from(1u32 << 30));
        assert!(f64::from(3.7f32 * scale) > f64::from(1u32 << 29));
        assert_eq!(fixed_point_scale(0.0), 1.0);
    }
}
