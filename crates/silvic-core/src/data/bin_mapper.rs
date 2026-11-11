//! Per-feature bin mappers: a port of LightGBM 3.3.5 `BinMapper::FindBin` (`src/io/bin.cpp`) with
//! `FindBinWithZeroAsOneBin` and `GreedyFindBin`, under LightGBM's defaults `use_missing=true`,
//! `zero_as_missing=false` and no forced bins, so the upper bounds, and hence the thresholds in a model, are the ones
//! LightGBM builds from the same sample.
//!
//! As in LightGBM, the sample lists only the non-zero values: values with `|v| <= kZeroThreshold` (and rows absent
//! from the sample) count as zeros, and zero always sits alone in a bin bounded by `±kZeroThreshold`.
//!
//! Where silvic departs from `FindBin`:
//! - A numerical feature with one distinct sampled value and no NaN is trivial. LightGBM gives it a second, empty bin
//!   (zero's) and only drops it through `feature_pre_filter`, which silvic does not otherwise apply.
//! - A categorical feature keeps at most `max_bin` bins: LightGBM keeps adding categories past `max_bin` until they
//!   cover 99% of the sample, which would overflow a byte.

use std::fmt::Write as _;

use crate::config::Params;

/// LightGBM `kZeroThreshold` (`meta.h`): the float literal `1e-35f` widened to double.
pub(crate) const ZERO_THRESHOLD: f64 = 1e-35_f32 as f64;

/// Share of the non-missing sample the kept categories cover (`FindBin`'s `0.99f`, a float).
const CATEGORY_COVERAGE: f32 = 0.99;

#[derive(Debug, Clone, PartialEq)]
pub enum BinMapper {
    Numerical {
        /// Ascending; the last is `+inf`.
        upper_bounds: Vec<f64>,
        /// Whether bin `upper_bounds.len()` holds NaN.
        nan_bin: bool,
        /// Smallest and largest sampled values, for `feature_infos`.
        min: f64,
        max: f64,
    },
    Categorical {
        /// `categories[b - 1]` is the value of bin `b`; bin 0 is "other".
        categories: Vec<i32>,
    },
    Trivial,
}

/// A sample summarised as LightGBM's `FindBin` sees it.
struct Distinct {
    /// Ascending distinct values, zero included when the sample has zeros or straddles zero.
    values: Vec<f64>,
    counts: Vec<i64>,
    nan_count: i64,
    total: i64,
}

impl Distinct {
    /// `FindBin`'s distinct-value pass: values within one ulp merge into the larger, and zero (counted from the rows
    /// the sample leaves out) is inserted where it sorts, with count 0 when the sample straddles zero without one.
    fn new(sample: &[f64], total_rows: usize) -> Distinct {
        let nan_count = sample.iter().filter(|v| v.is_nan()).count();
        let mut sorted: Vec<f64> = sample.iter().copied().filter(|v| v.abs() > ZERO_THRESHOLD).collect();
        sorted.sort_unstable_by(f64::total_cmp);
        let total = total_rows.max(sample.len()) as i64;
        let zero_count = total - sorted.len() as i64 - nan_count as i64;

        let mut values = Vec::new();
        let mut counts = Vec::new();
        if sorted.first().is_none_or(|&first| first > 0.0 && zero_count > 0) {
            values.push(0.0);
            counts.push(zero_count);
        }
        if let Some(&first) = sorted.first() {
            values.push(first);
            counts.push(1);
        }
        for pair in sorted.windows(2) {
            let (prev, value) = (pair[0], pair[1]);
            if equal_ordered(prev, value) {
                *values.last_mut().expect("non-empty") = value;
                *counts.last_mut().expect("non-empty") += 1;
                continue;
            }
            if prev < 0.0 && value > 0.0 {
                values.push(0.0);
                counts.push(zero_count);
            }
            values.push(value);
            counts.push(1);
        }
        if sorted.last().is_some_and(|&last| last < 0.0 && zero_count > 0) {
            values.push(0.0);
            counts.push(zero_count);
        }
        Distinct {
            values,
            counts,
            nan_count: nan_count as i64,
            total,
        }
    }

    fn present_values(&self) -> usize {
        self.counts.iter().filter(|&&c| c > 0).count()
    }
}

/// `Common::CheckDoubleEqualOrdered`: `b` is at most one ulp above `a`.
fn equal_ordered(a: f64, b: f64) -> bool {
    b <= a.next_up()
}

/// Pushes `value` unless it is within one ulp of the last bound, as `GreedyFindBin` does.
fn push_bound(bounds: &mut Vec<f64>, value: f64) -> bool {
    if bounds.last().is_some_and(|&last| equal_ordered(last, value)) {
        return false;
    }
    bounds.push(value);
    true
}

/// `GreedyFindBin`: bin upper bounds for ascending `values` with `counts`, ending with `+inf`.
fn greedy_find_bin(values: &[f64], counts: &[i64], max_bin: i64, total_cnt: i64, min_data_in_bin: i64) -> Vec<f64> {
    let mut bounds = Vec::new();
    if values.len() as i64 <= max_bin {
        let mut cur_cnt_inbin = 0;
        for i in 0..values.len().saturating_sub(1) {
            cur_cnt_inbin += counts[i];
            if cur_cnt_inbin >= min_data_in_bin
                && push_bound(&mut bounds, ((values[i] + values[i + 1]) / 2.0).next_up())
            {
                cur_cnt_inbin = 0;
            }
        }
    } else {
        let (upper, lower) = greedy_bin_edges(values, counts, max_bin, total_cnt, min_data_in_bin);
        for (up, low) in upper.iter().zip(&lower[1..]) {
            push_bound(&mut bounds, ((up + low) / 2.0).next_up());
        }
    }
    bounds.push(f64::INFINITY);
    bounds
}

/// The many-values branch of `GreedyFindBin`: values with at least a mean bin's count get a bin of their own, the
/// rest fill bins of about the mean size of what remains. Returns each bin's largest and smallest value
/// (`upper_bounds`, `lower_bounds`), trimmed to the bins used.
fn greedy_bin_edges(
    values: &[f64],
    counts: &[i64],
    max_bin: i64,
    total_cnt: i64,
    min_data_in_bin: i64,
) -> (Vec<f64>, Vec<f64>) {
    let mut max_bin = max_bin;
    if min_data_in_bin > 0 {
        max_bin = max_bin.min(total_cnt / min_data_in_bin).max(1);
    }
    let mut mean_bin_size = total_cnt as f64 / max_bin as f64;
    let mut rest_bin_cnt = max_bin;
    let mut rest_sample_cnt = total_cnt;
    let is_big: Vec<bool> = counts.iter().map(|&c| c as f64 >= mean_bin_size).collect();
    for (&count, _) in counts.iter().zip(&is_big).filter(|(_, &big)| big) {
        rest_bin_cnt -= 1;
        rest_sample_cnt -= count;
    }
    mean_bin_size = rest_sample_cnt as f64 / rest_bin_cnt as f64;

    // One spare slot: with `max_bin == 1` LightGBM writes `lower_bounds[1]` past its vector and reads it back.
    let slots = max_bin as usize + 1;
    let mut upper = vec![f64::INFINITY; slots];
    let mut lower = vec![f64::INFINITY; slots];
    let mut bin_cnt = 0usize;
    lower[0] = values[0];
    let mut cur_cnt_inbin = 0;
    for i in 0..values.len() - 1 {
        if !is_big[i] {
            rest_sample_cnt -= counts[i];
        }
        cur_cnt_inbin += counts[i];
        let cur = cur_cnt_inbin as f64;
        if is_big[i] || cur >= mean_bin_size || (is_big[i + 1] && cur >= f64::max(1.0, mean_bin_size * 0.5)) {
            upper[bin_cnt] = values[i];
            bin_cnt += 1;
            lower[bin_cnt] = values[i + 1];
            if bin_cnt as i64 >= max_bin - 1 {
                break;
            }
            cur_cnt_inbin = 0;
            if !is_big[i] {
                rest_bin_cnt -= 1;
                mean_bin_size = rest_sample_cnt as f64 / rest_bin_cnt as f64;
            }
        }
    }
    bin_cnt += 1;
    upper.truncate(bin_cnt - 1);
    lower.truncate(bin_cnt);
    (upper, lower)
}

/// `FindBinWithZeroAsOneBin`: negatives and positives binned apart, zero alone between `-kZeroThreshold` and
/// `kZeroThreshold`.
fn find_bin_with_zero_as_one_bin(
    distinct: &Distinct,
    max_bin: i64,
    total_sample_cnt: i64,
    min_data_in_bin: i64,
) -> Vec<f64> {
    let (values, counts) = (&distinct.values, &distinct.counts);
    let (mut left_cnt_data, mut cnt_zero, mut right_cnt_data) = (0i64, 0i64, 0i64);
    for (&value, &count) in values.iter().zip(counts) {
        if value <= -ZERO_THRESHOLD {
            left_cnt_data += count;
        } else if value > ZERO_THRESHOLD {
            right_cnt_data += count;
        } else {
            cnt_zero += count;
        }
    }

    let left_cnt = values.iter().position(|&v| v > -ZERO_THRESHOLD).unwrap_or(values.len());
    let mut bounds = Vec::new();
    if left_cnt > 0 && max_bin > 1 {
        let share = left_cnt_data as f64 / (total_sample_cnt - cnt_zero) as f64 * (max_bin - 1) as f64;
        let left_max_bin = (share as i64).max(1);
        bounds = greedy_find_bin(
            &values[..left_cnt],
            &counts[..left_cnt],
            left_max_bin,
            left_cnt_data,
            min_data_in_bin,
        );
        if let Some(last) = bounds.last_mut() {
            *last = -ZERO_THRESHOLD;
        }
    }

    let right_start = (left_cnt..values.len()).find(|&i| values[i] > ZERO_THRESHOLD);
    let right_max_bin = max_bin - 1 - bounds.len() as i64;
    match right_start {
        Some(start) if right_max_bin > 0 => {
            let right = greedy_find_bin(
                &values[start..],
                &counts[start..],
                right_max_bin,
                right_cnt_data,
                min_data_in_bin,
            );
            bounds.push(ZERO_THRESHOLD);
            bounds.extend(right);
        }
        _ => bounds.push(f64::INFINITY),
    }
    bounds
}

impl BinMapper {
    /// Builds a numerical mapper from sampled values (NaN included) out of `total_rows` sampled rows; sampled rows
    /// missing from `sample` are zeros, as in LightGBM's sparse sample.
    pub fn numerical(sample: &[f64], total_rows: usize, params: &Params) -> BinMapper {
        let distinct = Distinct::new(sample, total_rows);
        let min_data_in_bin = params.min_data_in_bin as i64;
        let max_bin = params.max_bin as i64;
        let nan_bin = distinct.nan_count > 0;
        let upper_bounds = if nan_bin {
            find_bin_with_zero_as_one_bin(
                &distinct,
                max_bin - 1,
                distinct.total - distinct.nan_count,
                min_data_in_bin,
            )
        } else {
            find_bin_with_zero_as_one_bin(&distinct, max_bin, distinct.total, min_data_in_bin)
        };
        let num_bins = upper_bounds.len() + usize::from(nan_bin);
        debug_assert!(num_bins <= params.max_bin, "{num_bins} bins over max_bin");
        if num_bins <= 1 || (!nan_bin && distinct.present_values() == 1) {
            return BinMapper::Trivial;
        }
        BinMapper::Numerical {
            upper_bounds,
            nan_bin,
            min: distinct.values[0],
            max: *distinct.values.last().expect("non-empty"),
        }
    }

    /// Builds a categorical mapper from sampled values out of `total_rows` sampled rows (absent rows are category 0),
    /// following `FindBin`'s categorical branch: negative (after truncation to `int`) and NaN values are "other"; the
    /// rest, by count descending then value ascending, become bins while they cover less than 99% of the sample or
    /// fewer than `max_bin` bins are used, stopping at the third category with fewer than `min_data_in_bin` samples.
    pub fn categorical(sample: &[f64], total_rows: usize, params: &Params) -> BinMapper {
        let distinct = Distinct::new(sample, total_rows);
        let mut other_count = distinct.nan_count;
        let mut by_value: Vec<(i32, i64)> = Vec::new();
        for (&value, &count) in distinct.values.iter().zip(&distinct.counts) {
            let category = value as i32;
            if category < 0 {
                other_count += count;
            } else if let Some(last) = by_value.last_mut().filter(|last| last.0 == category) {
                last.1 += count;
            } else {
                by_value.push((category, count));
            }
        }
        let rest_count = distinct.total - other_count;
        if rest_count <= 0 {
            return BinMapper::Trivial;
        }
        // Stable, so equal counts stay in ascending value order (`SortForPair` with `std::stable_sort`).
        by_value.sort_by_key(|&(_, count)| std::cmp::Reverse(count));
        let cut_cnt = (f64::from(rest_count as f32 * CATEGORY_COVERAGE) + 0.5) as i64;
        let distinct_cnt = by_value.len() + usize::from(other_count > 0);
        let max_bin = distinct_cnt.min(params.max_bin);
        let min_data_in_bin = params.min_data_in_bin as i64;

        let mut categories = Vec::new();
        let mut used_cnt = 0;
        for (index, &(category, count)) in by_value.iter().enumerate() {
            let num_bin = categories.len() + 1;
            if !(used_cnt < cut_cnt || num_bin < max_bin) || num_bin >= params.max_bin {
                break;
            }
            if count < min_data_in_bin && index > 1 {
                break;
            }
            categories.push(category);
            used_cnt += count;
        }
        if categories.is_empty() {
            return BinMapper::Trivial;
        }
        BinMapper::Categorical { categories }
    }

    /// Bins in the histogram, NaN or "other" bin included; 0 when trivial.
    pub fn num_bins(&self) -> u32 {
        match self {
            BinMapper::Numerical {
                upper_bounds, nan_bin, ..
            } => upper_bounds.len() as u32 + u32::from(*nan_bin),
            BinMapper::Categorical { categories } => categories.len() as u32 + 1,
            BinMapper::Trivial => 0,
        }
    }

    /// LightGBM `BinMapper::ValueToBin`.
    pub fn value_to_bin(&self, value: f64) -> u8 {
        match self {
            BinMapper::Numerical {
                upper_bounds, nan_bin, ..
            } => {
                if value.is_nan() {
                    if *nan_bin {
                        return upper_bounds.len() as u8;
                    }
                    return upper_bounds.partition_point(|&bound| bound < 0.0) as u8;
                }
                upper_bounds.partition_point(|&bound| bound < value) as u8
            }
            BinMapper::Categorical { categories } => categorical_bin(categories, value),
            BinMapper::Trivial => 0,
        }
    }

    /// The `feature_infos` entry LightGBM writes: `[min:max]` (`%.17g`), `-1:c1:c2:...` (LightGBM lists its "other"
    /// bin as category -1) or `none`.
    pub fn info_string(&self) -> String {
        match self {
            BinMapper::Numerical { min, max, .. } => format!("[{}:{}]", format_g17(*min), format_g17(*max)),
            BinMapper::Categorical { categories } => {
                let mut out = String::from("-1");
                for category in categories {
                    write!(out, ":{category}").expect("write to String");
                }
                out
            }
            BinMapper::Trivial => "none".to_string(),
        }
    }
}

/// Categorical `ValueToBin`: the value truncates to `int`; negative, NaN and unknown values are bin 0.
pub(crate) fn categorical_bin(categories: &[i32], value: f64) -> u8 {
    if value.is_nan() {
        return 0;
    }
    let category = value as i32;
    if category < 0 {
        return 0;
    }
    categories
        .iter()
        .position(|&c| c == category)
        .map_or(0, |index| index as u8 + 1)
}

/// C's `%.17g` (C++ streams at `digits10 + 2`): 17 significant digits, fixed notation for exponents in `-4..17`,
/// trailing zeros dropped.
pub(crate) fn format_g17(value: f64) -> String {
    if value.is_nan() {
        return "nan".to_string();
    }
    if value.is_infinite() {
        return if value > 0.0 { "inf" } else { "-inf" }.to_string();
    }
    if value == 0.0 {
        return if value.is_sign_negative() { "-0" } else { "0" }.to_string();
    }
    let scientific = format!("{value:.16e}");
    let (mantissa, exponent) = scientific.split_once('e').expect("scientific notation");
    let exponent: i32 = exponent.parse().expect("exponent");
    if (-4..17).contains(&exponent) {
        let decimals = (16 - exponent) as usize;
        let fixed = format!("{value:.decimals$}");
        return trim_fraction(&fixed).to_string();
    }
    let sign = if exponent < 0 { '-' } else { '+' };
    format!("{}e{sign}{:02}", trim_fraction(mantissa), exponent.abs())
}

fn trim_fraction(number: &str) -> &str {
    if number.contains('.') {
        number.trim_end_matches('0').trim_end_matches('.')
    } else {
        number
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params(max_bin: usize, min_data_in_bin: usize) -> Params {
        Params {
            max_bin,
            min_data_in_bin,
            ..Params::default()
        }
    }

    /// A deterministic pseudo-random stream for test values.
    fn values(seed: u32, count: usize) -> Vec<f64> {
        (0..count as u32)
            .map(|i| {
                let h = crate::quantize::mix32(crate::quantize::mix32(i) ^ seed);
                (f64::from(h >> 8) / 16_777_216.0 - 0.5) * 8.0
            })
            .collect()
    }

    fn bounds(mapper: &BinMapper) -> &[f64] {
        match mapper {
            BinMapper::Numerical { upper_bounds, .. } => upper_bounds,
            other => panic!("not numerical: {other:?}"),
        }
    }

    #[test]
    fn value_to_bin_agrees_with_thresholds() {
        let mut sample = values(1, 5000);
        sample.extend([0.0; 300]);
        sample.extend([1.25; 400]);
        for (case, mapper) in [
            (
                "max_bin 255",
                BinMapper::numerical(&sample, sample.len(), &params(255, 3)),
            ),
            ("max_bin 7", BinMapper::numerical(&sample, sample.len(), &params(7, 3))),
            ("with nan", {
                let mut with_nan = sample.clone();
                with_nan.push(f64::NAN);
                BinMapper::numerical(&with_nan, with_nan.len(), &params(63, 3))
            }),
        ] {
            let upper_bounds = bounds(&mapper).to_vec();
            assert_eq!(
                *upper_bounds.last().expect("bounds"),
                f64::INFINITY,
                "{case}: last bound"
            );
            assert!(upper_bounds.windows(2).all(|w| w[0] < w[1]), "{case}: ascending");
            let mut probes = values(2, 2000);
            for &bound in &upper_bounds {
                probes.extend([bound, bound.next_up(), bound.next_down()]);
            }
            probes.extend([
                0.0,
                -0.0,
                f64::INFINITY,
                f64::NEG_INFINITY,
                ZERO_THRESHOLD,
                -ZERO_THRESHOLD,
                1e-300,
            ]);
            for value in probes {
                let bin = usize::from(mapper.value_to_bin(value));
                assert!(bin < upper_bounds.len(), "{case}: {value} bin {bin}");
                for (t, &bound) in upper_bounds.iter().enumerate() {
                    assert_eq!(
                        bin <= t,
                        value <= bound,
                        "{case}: value {value} bin {bin} threshold {t}"
                    );
                }
            }
        }
    }

    #[test]
    fn nan_bin_present_or_absent() {
        let mut sample = values(3, 1000);
        let without = BinMapper::numerical(&sample, sample.len(), &params(255, 3));
        let zero_bin = without.value_to_bin(0.0);
        assert_eq!(without.value_to_bin(f64::NAN), zero_bin, "no nan bin: NaN bins as zero");
        assert_eq!(without.num_bins() as usize, bounds(&without).len());

        sample.push(f64::NAN);
        let with = BinMapper::numerical(&sample, sample.len(), &params(255, 3));
        let BinMapper::Numerical {
            upper_bounds, nan_bin, ..
        } = &with
        else {
            panic!("numerical")
        };
        assert!(*nan_bin);
        assert_eq!(with.num_bins() as usize, upper_bounds.len() + 1);
        assert_eq!(usize::from(with.value_to_bin(f64::NAN)), upper_bounds.len());
        assert!(with.num_bins() <= 255);
    }

    #[test]
    fn trivial_and_all_nan() {
        let p = params(255, 3);
        assert_eq!(BinMapper::numerical(&[2.5; 10], 10, &p), BinMapper::Trivial);
        assert_eq!(BinMapper::numerical(&[], 10, &p), BinMapper::Trivial, "all zero");
        assert_eq!(
            BinMapper::numerical(&[1e-36; 4], 4, &p),
            BinMapper::Trivial,
            "below kZeroThreshold is zero"
        );
        let one_and_nan = BinMapper::numerical(&[2.5, f64::NAN], 2, &p);
        assert_eq!(one_and_nan.num_bins(), 3, "zero bin, value bin, nan bin");
        let zeros_and_value = BinMapper::numerical(&[2.5; 10], 20, &p);
        assert_eq!(bounds(&zeros_and_value), &[ZERO_THRESHOLD, f64::INFINITY]);
        assert_eq!(zeros_and_value.info_string(), "[0:2.5]");
        assert_eq!(BinMapper::Trivial.info_string(), "none");
    }

    #[test]
    fn few_distinct_values_split_at_midpoints() {
        let sample: Vec<f64> = [-2.0, -1.0, 1.0, 2.0, 3.0].iter().flat_map(|&v| [v; 5]).collect();
        let mapper = BinMapper::numerical(&sample, sample.len() + 5, &params(255, 3));
        assert_eq!(
            bounds(&mapper),
            &[
                (-1.5f64).next_up(),
                -ZERO_THRESHOLD,
                ZERO_THRESHOLD,
                1.5f64.next_up(),
                2.5f64.next_up(),
                f64::INFINITY
            ]
        );
        assert_eq!(mapper.info_string(), "[-2:3]");
    }

    #[test]
    fn categorical_orders_by_count_then_value() {
        let mut sample = Vec::new();
        for (value, count) in [
            (3.0, 10),
            (7.0, 30),
            (1.0, 10),
            (2.7, 20),
            (-4.0, 50),
            (f64::NAN, 5),
            (9.0, 1),
        ] {
            sample.extend(std::iter::repeat_n(value, count));
        }
        // 15 rows outside the sample are category 0.
        let total = sample.len() + 15;
        let mapper = BinMapper::categorical(&sample, total, &params(255, 3));
        // 9 has one sample: past the second category, under min_data_in_bin, so it is dropped.
        assert_eq!(
            mapper,
            BinMapper::Categorical {
                categories: vec![7, 2, 0, 1, 3]
            }
        );
        assert_eq!(mapper.num_bins(), 6);
        assert_eq!(mapper.info_string(), "-1:7:2:0:1:3");
        for (value, bin) in [
            (7.0, 1),
            (2.9, 2),
            (0.0, 3),
            (-0.5, 3),
            (3.0, 5),
            (-4.0, 0),
            (f64::NAN, 0),
            (9.0, 0),
        ] {
            assert_eq!(mapper.value_to_bin(value), bin, "category {value}");
        }
    }

    #[test]
    fn categorical_caps_at_max_bin() {
        let sample: Vec<f64> = (1..=40).flat_map(|v| std::iter::repeat_n(f64::from(v), 10)).collect();
        let mapper = BinMapper::categorical(&sample, sample.len(), &params(16, 3));
        let BinMapper::Categorical { categories } = &mapper else {
            panic!("categorical")
        };
        assert_eq!(categories, &(1..=15).collect::<Vec<_>>());
        assert_eq!(mapper.num_bins(), 16);
    }

    #[test]
    fn categorical_trivial_without_categories() {
        let p = params(255, 3);
        assert_eq!(BinMapper::categorical(&[-1.0, f64::NAN], 2, &p), BinMapper::Trivial);
        assert_eq!(
            BinMapper::categorical(&[4.0; 3], 3, &p),
            BinMapper::Categorical { categories: vec![4] }
        );
    }

    #[test]
    fn g17_matches_printf() {
        for (value, expected) in [
            (0.1, "0.10000000000000001"),
            (1.0, "1"),
            (-5.0, "-5"),
            (123_456.0, "123456"),
            (1e16, "10000000000000000"),
            (1e17, "1e+17"),
            (0.0001, "0.0001"),
            (0.000_012_5, "1.2500000000000001e-05"),
            (ZERO_THRESHOLD, "1.0000000180025095e-35"),
        ] {
            assert_eq!(format_g17(value), expected, "{value:e}");
        }
        for printed in [
            "-3.7722751561227339",
            "3.0660367390488967",
            "933.34701911841285",
            "1.2345678901234568e+17",
        ] {
            let value: f64 = printed.parse().expect("number");
            assert_eq!(format_g17(value), printed);
        }
    }
}
