//! `BinnedDataset` construction from dense matrices: LightGBM 3.3.5 `DatasetLoader::ConstructFromSampleData` (bin
//! mappers from a sample) and `Dataset::Construct` (feature layout), with the `Metadata` checks on labels, weights
//! and query sizes.
//!
//! Where LightGBM's distributed loader builds each rank's mappers from that rank's rows, silvic allgathers one
//! sample over all ranks (see `sample.rs`), so every rank holds identical mappers and a model trained on N ranks can
//! equal the one trained on all rows in one process.

use std::collections::HashSet;

use rayon::prelude::*;

use crate::collective::Collective;
use crate::config::Params;
use crate::data::binning::bin_rows;
use crate::data::sample::gather_sample;
use crate::data::{BinMapper, BinnedDataset, FeatureGroup, FeatureMeta, GROUP_BIN_BUDGET};
use crate::error::{Result, SilvicError};

/// Characters `Common::CheckAllowedJSON` rejects in feature names.
const JSON_SPECIAL: &[char] = &['"', ',', ':', '[', ']', '{', '}'];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Layout {
    RowMajor,
    ColumnMajor,
}

#[derive(Debug, Clone, Copy)]
pub enum DenseData<'a> {
    F32(&'a [f32]),
    F64(&'a [f64]),
}

/// A borrowed dense feature matrix (numpy arrays cross without copies).
#[derive(Debug, Clone, Copy)]
pub struct DenseMatrix<'a> {
    pub data: DenseData<'a>,
    pub rows: usize,
    pub cols: usize,
    pub layout: Layout,
}

impl DenseMatrix<'_> {
    #[inline]
    pub fn get(&self, row: usize, col: usize) -> f64 {
        let index = match self.layout {
            Layout::RowMajor => row * self.cols + col,
            Layout::ColumnMajor => col * self.rows + row,
        };
        match self.data {
            DenseData::F32(values) => f64::from(values[index]),
            DenseData::F64(values) => values[index],
        }
    }

    fn len(&self) -> usize {
        match self.data {
            DenseData::F32(values) => values.len(),
            DenseData::F64(values) => values.len(),
        }
    }
}

/// One rank's rows and their metadata.
#[derive(Debug, Clone)]
pub struct DatasetInput<'a> {
    pub features: DenseMatrix<'a>,
    pub label: &'a [f32],
    pub weight: Option<&'a [f32]>,
    /// Query sizes (LightGBM `group`): consecutive runs of rows.
    pub group: Option<&'a [u32]>,
    /// Defaults to `Column_<i>`, as LightGBM names unnamed columns.
    pub feature_names: Option<Vec<String>>,
    pub categorical_features: Vec<usize>,
}

impl BinnedDataset {
    /// Bins a training set. Bin mappers come from a sample of `bin_construct_sample_cnt` rows over every rank
    /// (allgathered), so every rank builds identical mappers; `row_offset` is the rows of lower ranks.
    pub fn construct(input: &DatasetInput<'_>, params: &Params, collective: &mut dyn Collective) -> Result<Self> {
        let checked = check_max_bin(params)
            .and_then(|()| validate(input))
            .and_then(|()| feature_names(input));
        let row_counts = gather_row_counts(input.features.rows, checked.as_ref().err(), collective)?;
        let names = checked?;
        let row_offset = row_counts[..collective.rank()].iter().sum();
        let sample = gather_sample(
            &input.features,
            &row_counts,
            params.bin_construct_sample_cnt,
            params.data_random_seed,
            collective,
        )?;
        let categorical: HashSet<usize> = input.categorical_features.iter().copied().collect();
        let mappers: Vec<BinMapper> = sample
            .values
            .par_iter()
            .enumerate()
            .map(|(feature, values)| {
                if categorical.contains(&feature) {
                    BinMapper::categorical(values, sample.rows, params)
                } else {
                    BinMapper::numerical(values, sample.rows, params)
                }
            })
            .collect();
        drop(sample);

        let (features, mut groups, total_bins) = layout(names, mappers);
        bin_rows(&input.features, &features, &mut groups);
        tracing::debug!(
            rows = input.features.rows,
            groups = groups.len(),
            total_bins,
            "binned training data"
        );
        Ok(BinnedDataset {
            num_rows: input.features.rows,
            row_offset,
            features,
            groups,
            total_bins,
            label: input.label.to_vec(),
            weight: input.weight.map(<[f32]>::to_vec),
            query_boundaries: query_boundaries(input.group),
        })
    }

    /// Bins a validation set with `reference`'s mappers and group layout; feature names and categorical
    /// declarations come from `reference`.
    pub fn construct_with_reference(
        input: &DatasetInput<'_>,
        reference: &BinnedDataset,
        collective: &mut dyn Collective,
    ) -> Result<Self> {
        let checked = validate(input).and_then(|()| {
            if input.features.cols == reference.features.len() {
                return Ok(());
            }
            Err(SilvicError::data(format!(
                "{} features, the reference dataset has {}",
                input.features.cols,
                reference.features.len()
            )))
        });
        let row_counts = gather_row_counts(input.features.rows, checked.as_ref().err(), collective)?;
        checked?;
        let row_offset = row_counts[..collective.rank()].iter().sum();
        let mut groups: Vec<FeatureGroup> = reference
            .groups
            .iter()
            .map(|group| FeatureGroup {
                features: group.features.clone(),
                row_stride: group.row_stride,
                hist_offset: group.hist_offset,
                total_bins: group.total_bins,
                data: Vec::new(),
            })
            .collect();
        bin_rows(&input.features, &reference.features, &mut groups);
        Ok(BinnedDataset {
            num_rows: input.features.rows,
            row_offset,
            features: reference.features.clone(),
            groups,
            total_bins: reference.total_bins,
            label: input.label.to_vec(),
            weight: input.weight.map(<[f32]>::to_vec),
            query_boundaries: query_boundaries(input.group),
        })
    }
}

fn validate(input: &DatasetInput<'_>) -> Result<()> {
    let matrix = &input.features;
    if matrix.rows.checked_mul(matrix.cols) != Some(matrix.len()) {
        return Err(SilvicError::data(format!(
            "feature matrix holds {} values, not {} rows x {} columns",
            matrix.len(),
            matrix.rows,
            matrix.cols
        )));
    }
    validate_label(input.label, matrix.rows)?;
    if let Some(weight) = input.weight {
        validate_weight(weight, matrix.rows)?;
    }
    if let Some(group) = input.group {
        validate_group(group, matrix.rows)?;
    }
    if let Some(&index) = input.categorical_features.iter().find(|&&index| index >= matrix.cols) {
        return Err(SilvicError::data(format!(
            "categorical feature {index} is out of range for {} features",
            matrix.cols
        )));
    }
    Ok(())
}

fn validate_label(label: &[f32], rows: usize) -> Result<()> {
    if label.len() != rows {
        return Err(SilvicError::data(format!("{} labels for {rows} rows", label.len())));
    }
    if let Some((row, value)) = label.iter().enumerate().find(|(_, v)| !v.is_finite()) {
        return Err(SilvicError::data(format!(
            "label of row {row} is {value}; labels must be finite"
        )));
    }
    Ok(())
}

fn validate_weight(weight: &[f32], rows: usize) -> Result<()> {
    if weight.len() != rows {
        return Err(SilvicError::data(format!("{} weights for {rows} rows", weight.len())));
    }
    if let Some((row, value)) = weight.iter().enumerate().find(|(_, v)| !(v.is_finite() && **v >= 0.0)) {
        return Err(SilvicError::data(format!(
            "weight of row {row} is {value}; weights must be finite and non-negative"
        )));
    }
    Ok(())
}

fn validate_group(group: &[u32], rows: usize) -> Result<()> {
    let total: u64 = group.iter().map(|&size| u64::from(size)).sum();
    if total != rows as u64 {
        return Err(SilvicError::data(format!(
            "query sizes sum to {total}, not the {rows} rows"
        )));
    }
    if total > u64::from(u32::MAX) {
        return Err(SilvicError::data(format!(
            "{rows} rows with queries; query boundaries are 32-bit"
        )));
    }
    Ok(())
}

/// The input's names or `Column_<i>`, with spaces (and `=`, which the model format also splits on) replaced by `_`
/// as `Dataset::set_feature_names` does; JSON special characters and duplicates are errors there and here.
fn feature_names(input: &DatasetInput<'_>) -> Result<Vec<String>> {
    let cols = input.features.cols;
    let Some(names) = &input.feature_names else {
        return Ok((0..cols).map(|i| format!("Column_{i}")).collect());
    };
    if names.len() != cols {
        return Err(SilvicError::data(format!(
            "{} feature names for {cols} features",
            names.len()
        )));
    }
    let mut seen = HashSet::with_capacity(cols);
    let mut out = Vec::with_capacity(cols);
    for name in names {
        if name.contains(JSON_SPECIAL) {
            return Err(SilvicError::data(format!(
                "feature name {name:?} contains a JSON special character"
            )));
        }
        let name = name.replace([' ', '='], "_");
        if !seen.insert(name.clone()) {
            return Err(SilvicError::data(format!(
                "feature name {name:?} appears more than once"
            )));
        }
        out.push(name);
    }
    Ok(out)
}

fn check_max_bin(params: &Params) -> Result<()> {
    if !(2..=255).contains(&params.max_bin) {
        return Err(SilvicError::config(format!(
            "max_bin must be in 2..=255, got {}",
            params.max_bin
        )));
    }
    Ok(())
}

/// Allgathers every rank's row count with its validation outcome (`[u64 rows][error text, empty when valid]`), so
/// a rank with bad input fails every rank at the first collective instead of leaving the others waiting in a later
/// one. All ranks return the lowest failing rank's error.
fn gather_row_counts(
    rows: usize,
    local_error: Option<&SilvicError>,
    collective: &mut dyn Collective,
) -> Result<Vec<u64>> {
    let mut status = (rows as u64).to_le_bytes().to_vec();
    if let Some(error) = local_error {
        let reason = match error {
            SilvicError::Data(message) | SilvicError::Config(message) => message.to_string(),
            other => other.to_string(),
        };
        // An empty text would read as success.
        status.extend(
            if reason.is_empty() {
                "invalid input".to_string()
            } else {
                reason
            }
            .into_bytes(),
        );
    }
    let gathered = collective.allgather_bytes(&status)?;
    let mut counts = Vec::with_capacity(gathered.len());
    for (rank, bytes) in gathered.iter().enumerate() {
        let malformed = || SilvicError::collective(format!("malformed row count from rank {rank}"));
        let (count, reason) = bytes.split_first_chunk::<8>().ok_or_else(malformed)?;
        if !reason.is_empty() {
            return Err(SilvicError::data(format!(
                "rank {rank}: {}",
                String::from_utf8_lossy(reason)
            )));
        }
        counts.push(u64::from_le_bytes(*count));
    }
    Ok(counts)
}

fn query_boundaries(group: Option<&[u32]>) -> Option<Vec<u32>> {
    let group = group?;
    let mut boundaries = Vec::with_capacity(group.len() + 1);
    let mut end = 0u32;
    boundaries.push(end);
    for &size in group {
        end += size;
        boundaries.push(end);
    }
    Some(boundaries)
}

/// Features with their histogram offsets, and the groups they pack into: non-trivial features in input order,
/// a new group whenever the next feature's bins would take the current one over `GROUP_BIN_BUDGET`.
fn layout(names: Vec<String>, mappers: Vec<BinMapper>) -> (Vec<FeatureMeta>, Vec<FeatureGroup>, usize) {
    let mut features = Vec::with_capacity(mappers.len());
    let mut groups: Vec<FeatureGroup> = Vec::new();
    let mut offset = 0u32;
    for (index, (name, bins)) in names.into_iter().zip(mappers).enumerate() {
        let num_bins = bins.num_bins();
        if num_bins == 0 {
            features.push(FeatureMeta {
                name,
                bins,
                num_bins,
                group: 0,
                column: 0,
                hist_offset: offset,
            });
            continue;
        }
        if groups
            .last()
            .is_none_or(|group| group.total_bins + num_bins > GROUP_BIN_BUDGET)
        {
            groups.push(FeatureGroup {
                features: Vec::new(),
                row_stride: 0,
                hist_offset: offset,
                total_bins: 0,
                data: Vec::new(),
            });
        }
        let group_index = groups.len() as u32 - 1;
        let group = groups.last_mut().expect("a group");
        features.push(FeatureMeta {
            name,
            bins,
            num_bins,
            group: group_index,
            column: group.features.len() as u32,
            hist_offset: offset,
        });
        group.features.push(index);
        group.total_bins += num_bins;
        offset += num_bins;
    }
    for group in &mut groups {
        group.row_stride = group.features.len().next_multiple_of(4) as u32;
    }
    (features, groups, offset as usize)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collective::LocalCollective;
    use crate::quantize::mix32;
    use crate::test_util::run_ranks;

    /// A row-major matrix of `rows x cols` with a mix of feature shapes: continuous, with NaN, integer, categorical.
    fn matrix_values(rows: usize, cols: usize, row_offset: usize) -> Vec<f64> {
        let mut out = Vec::with_capacity(rows * cols);
        for row in row_offset..row_offset + rows {
            for col in 0..cols {
                let unit = f64::from(mix32(mix32(row as u32) ^ mix32(col as u32 + 1)) >> 8) / 16_777_216.0;
                out.push(match col % 4 {
                    0 => (unit - 0.3) * 10.0,
                    1 if unit < 0.1 => f64::NAN,
                    1 => unit * unit,
                    2 => (unit * 7.0).floor(),
                    _ => (unit * unit * 9.0).floor() - 1.0,
                });
            }
        }
        out
    }

    fn input<'a>(values: &'a [f64], label: &'a [f32], cols: usize) -> DatasetInput<'a> {
        DatasetInput {
            features: DenseMatrix {
                data: DenseData::F64(values),
                rows: label.len(),
                cols,
                layout: Layout::RowMajor,
            },
            label,
            weight: None,
            group: None,
            feature_names: None,
            categorical_features: (3..cols).step_by(4).collect(),
        }
    }

    fn check_bins(dataset: &BinnedDataset, matrix: &DenseMatrix<'_>, case: &str) {
        for (feature, meta) in dataset.features.iter().enumerate() {
            if meta.is_trivial() {
                continue;
            }
            for row in 0..matrix.rows {
                let expected = meta.bins.value_to_bin(matrix.get(row, feature));
                assert_eq!(
                    dataset.bin(feature, row),
                    expected,
                    "{case}: feature {feature} row {row}"
                );
            }
        }
    }

    #[test]
    fn construct_bins_every_row() {
        let (rows, cols) = (10_000, 9);
        let values = matrix_values(rows, cols, 0);
        let label = vec![0.5f32; rows];
        let group = [4000u32, 0, 6000];
        let mut data = input(&values, &label, cols);
        data.group = Some(&group);
        data.feature_names = Some((0..cols).map(|i| format!("f {i}=x")).collect());
        let dataset = BinnedDataset::construct(&data, &Params::default(), &mut LocalCollective).expect("construct");
        assert_eq!(dataset.row_offset, 0);
        assert_eq!(dataset.query_boundaries, Some(vec![0, 4000, 4000, 10_000]));
        assert_eq!(dataset.features[0].name, "f_0_x");
        assert!(dataset.features[3].is_categorical());
        assert!(dataset.features[1].bins.value_to_bin(f64::NAN) as u32 == dataset.features[1].num_bins - 1);
        check_bins(&dataset, &data.features, "row-major f64");

        let column_major: Vec<f32> = (0..cols)
            .flat_map(|c| (0..rows).map(move |r| (r, c)))
            .map(|(r, c)| values[r * cols + c] as f32)
            .collect();
        let mut data32 = data.clone();
        data32.features = DenseMatrix {
            data: DenseData::F32(&column_major),
            rows,
            cols,
            layout: Layout::ColumnMajor,
        };
        let dataset32 = BinnedDataset::construct(&data32, &Params::default(), &mut LocalCollective).expect("f32");
        check_bins(&dataset32, &data32.features, "column-major f32");
    }

    #[test]
    fn grouping_respects_budget_and_stride() {
        let wide = BinMapper::Numerical {
            upper_bounds: (0..254).map(f64::from).chain([f64::INFINITY]).collect(),
            nan_bin: true,
            min: 0.0,
            max: 1.0,
        };
        let narrow = BinMapper::Categorical { categories: vec![1, 2] };
        let mappers: Vec<BinMapper> = (0..70)
            .map(|i| match i % 7 {
                0 => BinMapper::Trivial,
                1 => narrow.clone(),
                _ => wide.clone(),
            })
            .collect();
        let names = (0..70).map(|i| format!("Column_{i}")).collect();
        let (features, groups, total_bins) = layout(names, mappers);
        assert!(groups.len() > 1);
        let mut offset = 0;
        let mut grouped: Vec<usize> = Vec::new();
        for (index, group) in groups.iter().enumerate() {
            assert!(group.total_bins <= GROUP_BIN_BUDGET, "group {index}: budget");
            assert_eq!(group.row_stride % 4, 0, "group {index}: stride");
            assert!(
                group.row_stride as usize >= group.features.len()
                    && group.row_stride as usize - group.features.len() < 4
            );
            assert_eq!(group.hist_offset, offset, "group {index}: offset");
            for (column, &feature) in group.features.iter().enumerate() {
                let meta = &features[feature];
                assert_eq!(
                    (meta.group, meta.column, meta.hist_offset),
                    (index as u32, column as u32, offset)
                );
                offset += meta.num_bins;
            }
            assert_eq!(offset - group.hist_offset, group.total_bins);
            if let Some(next) = groups.get(index + 1) {
                let next_bins = features[next.features[0]].num_bins;
                assert!(group.total_bins + next_bins > GROUP_BIN_BUDGET, "group {index}: greedy");
            }
            grouped.extend(&group.features);
        }
        assert_eq!(offset as usize, total_bins);
        let expected: Vec<usize> = (0..70).filter(|i| i % 7 != 0).collect();
        assert_eq!(grouped, expected, "non-trivial features in input order");
    }

    #[test]
    fn two_ranks_share_mappers() {
        let cols = 9;
        for (case, rows, sample_cnt) in [
            ("all rows sampled", [3000, 5000], 200_000),
            ("sampled", [3000, 5000], 2000),
        ] {
            let total = rows[0] + rows[1];
            let values = matrix_values(total, cols, 0);
            let label = vec![1.0f32; total];
            let params = Params {
                bin_construct_sample_cnt: sample_cnt,
                ..Params::default()
            };
            let built = run_ranks(2, |rank, collective| {
                let start = if rank == 0 { 0 } else { rows[0] };
                let end = start + rows[rank];
                let local = input(&values[start * cols..end * cols], &label[start..end], cols);
                let dataset = BinnedDataset::construct(&local, &params, collective).expect("construct");
                check_bins(&dataset, &local.features, case);
                dataset
            });
            assert_eq!(built[0].row_offset, 0, "{case}");
            assert_eq!(built[1].row_offset, rows[0] as u64, "{case}");
            assert_eq!(built[0].features, built[1].features, "{case}: mappers");
            if sample_cnt >= total {
                let whole = input(&values, &label, cols);
                let single = BinnedDataset::construct(&whole, &params, &mut LocalCollective).expect("one rank");
                assert_eq!(single.features, built[0].features, "{case}: one rank");
            }
        }
    }

    #[test]
    fn reference_reproduces_layout() {
        let cols = 9;
        let values = matrix_values(4000, cols, 0);
        let label = vec![0.0f32; 4000];
        let train = BinnedDataset::construct(&input(&values, &label, cols), &Params::default(), &mut LocalCollective)
            .expect("train");
        let valid_values = matrix_values(1500, cols, 77_777);
        let valid_label = vec![2.0f32; 1500];
        let valid_input = input(&valid_values, &valid_label, cols);
        let valid =
            BinnedDataset::construct_with_reference(&valid_input, &train, &mut LocalCollective).expect("reference");
        assert_eq!(valid.features, train.features);
        assert_eq!(valid.total_bins, train.total_bins);
        for (a, b) in valid.groups.iter().zip(&train.groups) {
            assert_eq!(
                (&a.features, a.row_stride, a.hist_offset, a.total_bins),
                (&b.features, b.row_stride, b.hist_offset, b.total_bins)
            );
            assert_eq!(a.data.len(), 1500 * a.row_stride as usize);
        }
        check_bins(&valid, &valid_input.features, "reference");

        let narrow = input(&valid_values[..1500 * 3], &valid_label, 3);
        assert!(BinnedDataset::construct_with_reference(&narrow, &train, &mut LocalCollective).is_err());
    }

    #[test]
    fn invalid_rank_fails_every_rank() {
        let cols = 4;
        let values = matrix_values(20, cols, 0);
        let label = vec![1.0f32; 20];
        let train = BinnedDataset::construct(&input(&values, &label, cols), &Params::default(), &mut LocalCollective)
            .expect("train");
        let expected = SilvicError::data("rank 1: 9 labels for 10 rows");
        for reference in [None, Some(&train)] {
            let results = run_ranks(2, |rank, collective| {
                let rows = &values[rank * 10 * cols..(rank + 1) * 10 * cols];
                let mut local = input(rows, &label[..10 - rank], cols);
                local.features.rows = 10;
                match reference {
                    None => BinnedDataset::construct(&local, &Params::default(), collective),
                    Some(reference) => BinnedDataset::construct_with_reference(&local, reference, collective),
                }
            });
            for (rank, result) in results.into_iter().enumerate() {
                let reference = reference.is_some();
                assert_eq!(
                    result.err(),
                    Some(expected.clone()),
                    "rank {rank}, reference {reference}"
                );
            }
        }
    }

    #[test]
    fn rejects_bad_input() {
        let cols = 4;
        let values = matrix_values(10, cols, 0);
        let label = vec![1.0f32; 10];
        let params = Params::default();
        let construct = |data: &DatasetInput<'_>| BinnedDataset::construct(data, &params, &mut LocalCollective);
        assert!(construct(&input(&values, &label, cols)).is_ok());

        let short_label = vec![1.0f32; 9];
        let mut bad = input(&values, &short_label, cols);
        bad.features.rows = 10;
        assert!(construct(&bad).is_err(), "label length");
        let nan_label = [vec![1.0f32; 9], vec![f32::NAN]].concat();
        assert!(construct(&input(&values, &nan_label, cols)).is_err(), "label NaN");
        let negative = [vec![1.0f32; 9], vec![-1.0]].concat();
        let mut bad = input(&values, &label, cols);
        bad.weight = Some(&negative);
        assert!(construct(&bad).is_err(), "negative weight");
        bad.weight = Some(&label[..5]);
        assert!(construct(&bad).is_err(), "weight length");
        let mut bad = input(&values, &label, cols);
        bad.group = Some(&[3, 3]);
        assert!(construct(&bad).is_err(), "group sum");
        let mut bad = input(&values, &label, cols);
        bad.categorical_features = vec![4];
        assert!(construct(&bad).is_err(), "categorical index");
        let mut bad = input(&values, &label, cols);
        bad.feature_names = Some(vec!["a".into(), "b".into()]);
        assert!(construct(&bad).is_err(), "name count");
        bad.feature_names = Some(vec!["a".into(), "b".into(), "c".into(), "a".into()]);
        assert!(construct(&bad).is_err(), "duplicate name");
        bad.feature_names = Some(vec!["a".into(), "b".into(), "c".into(), "d:e".into()]);
        assert!(construct(&bad).is_err(), "json character");
        let mut bad = input(&values[..39], &label, cols);
        bad.features.rows = 10;
        assert!(construct(&bad).is_err(), "matrix length");
    }

    /// Binning throughput on 10M x 140 f32 rows: `cargo test --release -p silvic-core --lib
    /// data::build::tests::binning_throughput -- --ignored --nocapture`.
    #[test]
    #[ignore = "benchmark: allocates 7 GB"]
    fn binning_throughput() {
        let (rows, cols) = (10_000_000usize, 140usize);
        let mut values = vec![0f32; rows * cols];
        values.par_chunks_mut(cols).enumerate().for_each(|(row, out)| {
            for (col, value) in out.iter_mut().enumerate() {
                let hash = mix32(mix32(row as u32) ^ mix32(col as u32 + 1));
                *value = if col % 10 == 9 {
                    (hash % 12) as f32
                } else {
                    (hash >> 8) as f32 / 16_777_216.0 - 0.5
                };
            }
        });
        let label = vec![0f32; rows];
        let data = DatasetInput {
            features: DenseMatrix {
                data: DenseData::F32(&values),
                rows,
                cols,
                layout: Layout::RowMajor,
            },
            label: &label,
            weight: None,
            group: None,
            feature_names: None,
            categorical_features: (9..cols).step_by(10).collect(),
        };
        let start = std::time::Instant::now();
        let dataset = BinnedDataset::construct(&data, &Params::default(), &mut LocalCollective).expect("construct");
        let construct_seconds = start.elapsed().as_secs_f64();
        let start = std::time::Instant::now();
        let rebinned = BinnedDataset::construct_with_reference(&data, &dataset, &mut LocalCollective).expect("rebin");
        let bin_seconds = start.elapsed().as_secs_f64();
        assert_eq!(rebinned.groups, dataset.groups);
        let values_count = (rows * cols) as f64;
        println!(
            "{rows} x {cols} f32 row-major, {} groups, {} bins: construct {construct_seconds:.2} s ({:.3e} values/s), \
             binning alone {bin_seconds:.2} s ({:.3e} values/s)",
            dataset.groups.len(),
            dataset.total_bins,
            values_count / construct_seconds,
            values_count / bin_seconds,
        );
    }
}
