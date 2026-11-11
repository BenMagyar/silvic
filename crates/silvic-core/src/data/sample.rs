//! The sample bin mappers are built from, identical on every rank.
//!
//! Each rank draws its proportional share of `bin_construct_sample_cnt` rows with LightGBM 3.3.5's
//! `Random::Sample` (`include/LightGBM/utils/random.h`), the sampler `LGBM_DatasetCreateFromMat` uses, so a
//! single rank picks the rows LightGBM picks for the same `data_random_seed`. As in LightGBM's
//! `ConstructFromSampleData`, only non-zero and NaN values are kept: the rest of the sampled rows count as zeros.
//! The per-feature values are allgathered and concatenated in rank order, so every rank builds the same mappers;
//! when every row is sampled, that is the same multiset of values as one rank holding all rows.

use std::collections::HashSet;

use rayon::prelude::*;

use crate::collective::Collective;
use crate::data::bin_mapper::ZERO_THRESHOLD;
use crate::data::{DenseData, DenseMatrix, Layout};
use crate::error::{Result, SilvicError};
use crate::quantize::mix32;

/// Wire tags for the sampled values' width; f32 input travels as f32, which halves the allgather.
const TAG_F32: u8 = 4;
const TAG_F64: u8 = 8;

/// LightGBM's `Random`: an LCG over `unsigned int` with MSVC's constants.
struct LightGbmRandom {
    x: u32,
}

impl LightGbmRandom {
    fn new(seed: u32) -> Self {
        LightGbmRandom { x: seed }
    }

    fn step(&mut self) -> u32 {
        self.x = self.x.wrapping_mul(214_013).wrapping_add(2_531_011);
        self.x
    }

    /// `NextFloat`: 15 random bits as a float in `[0, 1)`.
    fn next_float(&mut self) -> f32 {
        ((self.step() >> 16) & 0x7FFF) as f32 / 32768.0
    }

    /// `NextInt(0, upper)`.
    fn next_int(&mut self, upper: u64) -> u64 {
        u64::from(self.step() & 0x7FFF_FFFF) % upper
    }
}

/// `Random::Sample(n, k)`: `k` ascending row indices out of `n`.
pub(crate) fn lightgbm_sample(n: usize, k: usize, seed: u32) -> Vec<usize> {
    if k > n || k == 0 {
        return Vec::new();
    }
    if k == n {
        return (0..n).collect();
    }
    let mut random = LightGbmRandom::new(seed);
    if k > 1 && k as f64 > n as f64 / (k as f64).log2() {
        let mut picked = Vec::with_capacity(k);
        for i in 0..n {
            let prob = (k - picked.len()) as f64 / (n - i) as f64;
            if f64::from(random.next_float()) < prob {
                picked.push(i);
            }
        }
        return picked;
    }
    let mut picked = HashSet::with_capacity(k);
    for r in n - k..n {
        let candidate = random.next_int(r as u64 + 1) as usize;
        if !picked.insert(candidate) {
            picked.insert(r);
        }
    }
    let mut picked: Vec<usize> = picked.into_iter().collect();
    picked.sort_unstable();
    picked
}

/// Rank `rank`'s share of a `sample_cnt`-row sample over ranks holding `row_counts` rows: shares follow the
/// cumulative row counts, so they sum to `min(sample_cnt, total)` and a rank's share never exceeds its rows.
pub(crate) fn rank_share(row_counts: &[u64], rank: usize, sample_cnt: usize) -> usize {
    let total: u64 = row_counts.iter().sum();
    if total == 0 {
        return 0;
    }
    let sample = (sample_cnt as u64).min(total) as u128;
    let below: u64 = row_counts[..rank].iter().sum();
    let cut = |rows: u64| sample * u128::from(rows) / u128::from(total);
    (cut(below + row_counts[rank]) - cut(below)) as usize
}

/// Rank 0 samples with `data_random_seed` itself, which is what makes a one-rank sample LightGBM's.
fn rank_seed(seed: u32, rank: usize) -> u32 {
    if rank == 0 {
        seed
    } else {
        mix32(seed ^ mix32(rank as u32))
    }
}

/// The allgathered sample: per feature, every rank's non-zero and NaN sampled values in rank order.
pub(crate) struct GatheredSample {
    pub rows: usize,
    pub values: Vec<Vec<f64>>,
}

/// Samples this rank's share of rows and allgathers every rank's samples.
pub(crate) fn gather_sample(
    features: &DenseMatrix<'_>,
    row_counts: &[u64],
    sample_cnt: usize,
    seed: u32,
    collective: &mut dyn Collective,
) -> Result<GatheredSample> {
    let rank = collective.rank();
    let share = rank_share(row_counts, rank, sample_cnt);
    let rows = lightgbm_sample(features.rows, share, rank_seed(seed, rank));
    let local = encode(features, &rows);
    let gathered = collective.allgather_bytes(&local)?;
    decode(&gathered, features.cols)
}

fn is_sampled(value: f64) -> bool {
    value.is_nan() || value.abs() > ZERO_THRESHOLD
}

/// `[u64 sampled rows][u64 features][u8 width]` then per feature `[u64 count][count values]`, little-endian.
fn encode(features: &DenseMatrix<'_>, rows: &[usize]) -> Vec<u8> {
    let (per_feature, width) = match features.data {
        DenseData::F32(data) => (encode_columns(data, features, rows, f32::to_le_bytes), TAG_F32),
        DenseData::F64(data) => (encode_columns(data, features, rows, f64::to_le_bytes), TAG_F64),
    };
    let mut out = Vec::with_capacity(17 + per_feature.iter().map(Vec::len).sum::<usize>());
    out.extend((rows.len() as u64).to_le_bytes());
    out.extend((features.cols as u64).to_le_bytes());
    out.push(width);
    for column in per_feature {
        out.extend(column);
    }
    out
}

fn encode_columns<T: Copy + Into<f64> + Send + Sync, const N: usize>(
    data: &[T],
    features: &DenseMatrix<'_>,
    rows: &[usize],
    to_bytes: fn(T) -> [u8; N],
) -> Vec<Vec<u8>> {
    let cols = features.cols;
    match features.layout {
        Layout::ColumnMajor => (0..cols)
            .into_par_iter()
            .map(|col| {
                let column = &data[col * features.rows..(col + 1) * features.rows];
                encode_values(rows.iter().map(|&row| column[row]), to_bytes)
            })
            .collect(),
        Layout::RowMajor => {
            // Copying the sampled rows out first reads each row once, contiguously, instead of once per feature
            // from all over a matrix of many gigabytes.
            let mut compact = Vec::with_capacity(rows.len() * cols);
            compact.par_extend(
                rows.par_iter()
                    .flat_map_iter(|&row| data[row * cols..(row + 1) * cols].iter().copied()),
            );
            (0..cols)
                .into_par_iter()
                .map(|col| encode_values(compact.iter().skip(col).step_by(cols).copied(), to_bytes))
                .collect()
        }
    }
}

/// `[u64 count][values]` of the sampled (non-zero or NaN) values.
fn encode_values<T: Copy + Into<f64>, const N: usize>(
    values: impl Iterator<Item = T>,
    to_bytes: fn(T) -> [u8; N],
) -> Vec<u8> {
    let mut out = vec![0; 8];
    let mut count = 0u64;
    for value in values.filter(|&v| is_sampled(v.into())) {
        out.extend(to_bytes(value));
        count += 1;
    }
    out[..8].copy_from_slice(&count.to_le_bytes());
    out
}

fn decode(gathered: &[Vec<u8>], cols: usize) -> Result<GatheredSample> {
    let mut sample = GatheredSample {
        rows: 0,
        values: vec![Vec::new(); cols],
    };
    for (rank, bytes) in gathered.iter().enumerate() {
        let malformed = || SilvicError::collective(format!("malformed bin sample from rank {rank}"));
        let mut reader = Reader { bytes, pos: 0 };
        sample.rows += reader.u64().ok_or_else(malformed)? as usize;
        let rank_cols = reader.u64().ok_or_else(malformed)?;
        if rank_cols != cols as u64 {
            return Err(SilvicError::data(format!(
                "rank {rank} has {rank_cols} features, this rank has {cols}"
            )));
        }
        let width = reader.take(1).ok_or_else(malformed)?[0];
        for values in &mut sample.values {
            let count = reader.u64().ok_or_else(malformed)? as usize;
            let raw = reader
                .take(count.checked_mul(usize::from(width)).ok_or_else(malformed)?)
                .ok_or_else(malformed)?;
            match width {
                TAG_F32 => values.extend(raw.as_chunks::<4>().0.iter().map(|&b| f64::from(f32::from_le_bytes(b)))),
                TAG_F64 => values.extend(raw.as_chunks::<8>().0.iter().map(|&b| f64::from_le_bytes(b))),
                _ => return Err(malformed()),
            }
        }
        if reader.pos != bytes.len() {
            return Err(malformed());
        }
    }
    Ok(sample)
}

struct Reader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, len: usize) -> Option<&'a [u8]> {
        let slice = self.bytes.get(self.pos..self.pos.checked_add(len)?)?;
        self.pos += len;
        Some(slice)
    }

    fn u64(&mut self) -> Option<u64> {
        Some(u64::from_le_bytes(self.take(8)?.try_into().ok()?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sample_is_sorted_distinct_and_sized() {
        for (n, k) in [
            (1000, 1000),
            (1000, 900),
            (100_000, 2000),
            (60_000, 2000),
            (10, 3),
            (5, 0),
            (3, 5),
        ] {
            let picked = lightgbm_sample(n, k, 1);
            let expected = if k > n { 0 } else { k };
            assert_eq!(picked.len(), expected, "n {n} k {k}: size");
            assert!(picked.windows(2).all(|w| w[0] < w[1]), "n {n} k {k}: ascending");
            assert!(picked.iter().all(|&i| i < n), "n {n} k {k}: in range");
        }
    }

    #[test]
    fn lcg_matches_lightgbm_random() {
        // `Random(1)`: x = 214013 * 1 + 2531011 = 2745024, `RandInt16` = (2745024 >> 16) & 0x7FFF = 41.
        let mut random = LightGbmRandom::new(1);
        assert_eq!(random.next_float(), 41.0 / 32768.0);
    }

    #[test]
    fn shares_sum_to_sample() {
        let counts = [10u64, 0, 7, 33];
        for sample_cnt in [0, 1, 13, 49, 50, 1000] {
            let shares: Vec<usize> = (0..counts.len()).map(|r| rank_share(&counts, r, sample_cnt)).collect();
            assert_eq!(shares.iter().sum::<usize>(), sample_cnt.min(50), "sample {sample_cnt}");
            assert!(
                shares.iter().zip(counts).all(|(&s, c)| s as u64 <= c),
                "sample {sample_cnt}: share <= rows"
            );
        }
        assert_eq!(rank_share(&counts, 3, 1000), 33);
    }

    #[test]
    fn encode_decode_round_trip() {
        let data = [1.5f32, 0.0, 7.0, -2.0, 1e-36, 3.0];
        for layout in [Layout::RowMajor, Layout::ColumnMajor] {
            let matrix = DenseMatrix {
                data: DenseData::F32(&data),
                rows: 3,
                cols: 2,
                layout,
            };
            let bytes = encode(&matrix, &[0, 2]);
            let sample = decode(&[bytes.clone(), bytes], 2).expect("decode");
            assert_eq!(sample.rows, 4);
            let expected: [Vec<f64>; 2] = match layout {
                Layout::RowMajor => [vec![1.5], vec![3.0]],
                Layout::ColumnMajor => [vec![1.5, 7.0], vec![-2.0, 3.0]],
            };
            for (got, want) in sample.values.iter().zip(&expected) {
                let doubled: Vec<f64> = want.iter().chain(want).copied().collect();
                assert_eq!(got, &doubled, "{layout:?}");
            }
        }
        assert!(decode(&[vec![1, 2, 3]], 2).is_err());
        assert!(is_sampled(f64::NAN) && !is_sampled(-0.0) && !is_sampled(1e-36));
    }
}
