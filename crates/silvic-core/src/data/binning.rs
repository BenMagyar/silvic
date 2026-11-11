//! Bins every row into the feature groups, in parallel over row chunks; the per-value work of LightGBM 3.3.5
//! `Dataset::PushOneRow` with `BinMapper::ValueToBin`.
//!
//! A chunk bins one feature at a time (a strided walk over row-major input, a contiguous one over column-major), so
//! the mapper is resolved once per column and the bin lookup is a branch-free search over the upper bounds padded to
//! 256 with `+inf`.

use rayon::prelude::*;

use crate::data::bin_mapper::categorical_bin;
use crate::data::{BinMapper, DenseData, DenseMatrix, FeatureGroup, FeatureMeta, Layout};

/// Rows per parallel task: small enough that a chunk of row-major input (560 KB at 140 f32 columns) stays in L2
/// while each of its columns is binned in turn.
const CHUNK_ROWS: usize = 1024;

/// Upper bounds padded to a power of two for the branch-free search.
const PADDED_BINS: usize = 256;

enum Binner {
    Numerical {
        bounds: Box<[f64; PADDED_BINS]>,
        /// The bin NaN goes to: the NaN bin, or zero's bin when the sample had no NaN.
        nan: u8,
    },
    Categorical {
        categories: Vec<i32>,
    },
}

impl Binner {
    fn new(mapper: &BinMapper) -> Option<Binner> {
        match mapper {
            BinMapper::Numerical { upper_bounds, .. } => {
                let mut bounds = Box::new([f64::INFINITY; PADDED_BINS]);
                bounds[..upper_bounds.len()].copy_from_slice(upper_bounds);
                Some(Binner::Numerical {
                    bounds,
                    nan: mapper.value_to_bin(f64::NAN),
                })
            }
            BinMapper::Categorical { categories } => Some(Binner::Categorical {
                categories: categories.clone(),
            }),
            BinMapper::Trivial => None,
        }
    }
}

/// The number of bounds below `value` among the first 255: the first bin whose upper bound is `>= value`.
#[inline(always)]
fn numerical_bin(bounds: &[f64; PADDED_BINS], nan: u8, value: f64) -> u8 {
    if value.is_nan() {
        return nan;
    }
    let mut pos = 0usize;
    for step in [128, 64, 32, 16, 8, 4, 2, 1] {
        pos += step * usize::from(bounds[(pos + step - 1) & (PADDED_BINS - 1)] < value);
    }
    pos as u8
}

/// One group's rows of one chunk.
struct GroupChunk<'a> {
    features: &'a [usize],
    stride: usize,
    out: &'a mut [u8],
}

/// Allocates every group's `data` and fills it with the bins of `matrix`'s rows.
pub(crate) fn bin_rows(matrix: &DenseMatrix<'_>, features: &[FeatureMeta], groups: &mut [FeatureGroup]) {
    let binners: Vec<Option<Binner>> = features.iter().map(|f| Binner::new(&f.bins)).collect();
    let rows = matrix.rows;
    let mut chunks: Vec<Vec<GroupChunk<'_>>> = (0..rows.div_ceil(CHUNK_ROWS))
        .map(|_| Vec::with_capacity(groups.len()))
        .collect();
    for group in groups.iter_mut() {
        let stride = group.row_stride as usize;
        // Zeroed allocation, so padding columns are 0 and pages fault in on the threads that fill them.
        group.data = vec![0; rows * stride];
        let FeatureGroup { features, data, .. } = group;
        let features: &[usize] = features;
        for (chunk, out) in data.chunks_mut(CHUNK_ROWS * stride).enumerate() {
            chunks[chunk].push(GroupChunk { features, stride, out });
        }
    }
    chunks.into_par_iter().enumerate().for_each(|(chunk, parts)| {
        let start = chunk * CHUNK_ROWS;
        for part in parts {
            let len = part.out.len() / part.stride;
            for (column, &feature) in part.features.iter().enumerate() {
                let binner = binners[feature].as_ref().expect("grouped features are not trivial");
                let target = &mut part.out[column..];
                bin_column(matrix, binner, feature, start, len, target, part.stride);
            }
        }
    });
}

fn bin_column(
    matrix: &DenseMatrix<'_>,
    binner: &Binner,
    feature: usize,
    start: usize,
    len: usize,
    out: &mut [u8],
    stride: usize,
) {
    let (base, step) = match matrix.layout {
        Layout::RowMajor => (start * matrix.cols + feature, matrix.cols),
        Layout::ColumnMajor => (feature * matrix.rows + start, 1),
    };
    match matrix.data {
        DenseData::F32(data) => fill(binner, &data[base..], step, len, out, stride),
        DenseData::F64(data) => fill(binner, &data[base..], step, len, out, stride),
    }
}

#[inline(always)]
fn fill<T: Copy + Into<f64>>(binner: &Binner, data: &[T], step: usize, len: usize, out: &mut [u8], stride: usize) {
    let values = data.iter().step_by(step).take(len);
    let targets = out.iter_mut().step_by(stride);
    match binner {
        Binner::Numerical { bounds, nan } => {
            for (target, &value) in targets.zip(values) {
                *target = numerical_bin(bounds, *nan, value.into());
            }
        }
        Binner::Categorical { categories } => {
            for (target, &value) in targets.zip(values) {
                *target = categorical_bin(categories, value.into());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quantize::mix32;

    #[test]
    fn padded_search_matches_value_to_bin() {
        for len in [1usize, 2, 3, 7, 64, 127, 128, 129, 200, 254, 255] {
            let mut upper_bounds: Vec<f64> = (0..len - 1).map(|i| i as f64 * 0.5 - 20.0).collect();
            upper_bounds.push(f64::INFINITY);
            for nan_bin in [false, true] {
                let mapper = BinMapper::Numerical {
                    upper_bounds: upper_bounds.clone(),
                    nan_bin,
                    min: 0.0,
                    max: 0.0,
                };
                let Some(Binner::Numerical { bounds, nan }) = Binner::new(&mapper) else {
                    panic!("numerical")
                };
                let mut probes: Vec<f64> = (0..500u32).map(|i| f64::from(mix32(i) >> 20) / 50.0 - 50.0).collect();
                probes.extend(upper_bounds.iter().flat_map(|&b| [b, b.next_up(), b.next_down()]));
                probes.extend([0.0, -0.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY]);
                for value in probes {
                    assert_eq!(
                        numerical_bin(&bounds, nan, value),
                        mapper.value_to_bin(value),
                        "len {len} nan_bin {nan_bin} value {value}"
                    );
                }
            }
        }
    }
}
