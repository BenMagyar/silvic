//! Compares bin mappers against the bins LightGBM 3.3.5 builds for the same rows, recorded by
//! `fixtures/gen_lightgbm_bins.py` in `fixtures/lightgbm_bins.json`.
//!
//! LightGBM does not expose its bin upper bounds, so the generator recovers them from the thresholds of a one-tree
//! train that splits every leaf down to single bins. A boundary between two bins of which one is empty yields the
//! same partition as its neighbour, so LightGBM never needs to split there: the test asserts that every LightGBM
//! threshold is one of silvic's upper bounds, bit for bit, and that silvic's other bounds all border an empty bin.
//! `feature_infos` (sampled min and max, kept categories in bin order) must match exactly; the sampled min and max
//! differ between the cases, so they also pin the sampled rows to LightGBM's `Random::Sample`.

use serde_json::Value;

use crate::collective::LocalCollective;
use crate::config::Params;
use crate::data::{BinMapper, BinnedDataset, DatasetInput, DenseData, DenseMatrix, Layout};
use crate::quantize::mix32;

const FIXTURE: &str = include_str!("fixtures/lightgbm_bins.json");

/// The generator's `unit(stream, rows)`.
fn unit(stream: u32, row: u32) -> f64 {
    f64::from(mix32(mix32(row) ^ mix32(stream)) >> 8) / 16_777_216.0
}

/// The generator's `feature(name, rows)` at one row, with the same operations in the same order.
fn feature_value(name: &str, row: u32) -> f64 {
    match name {
        "normal" => (((unit(10, row) + unit(11, row)) + unit(12, row)) + unit(13, row) - 2.0) * 1.5,
        "zeros_nan" => match unit(20, row) {
            a if a < 0.3 => 0.0,
            a if a < 0.4 => f64::NAN,
            _ => ((unit(21, row) + unit(22, row)) - 1.0) * 3.0,
        },
        "integers" => (unit(30, row) * 20.0).floor() - 5.0,
        "mass_points" if unit(40, row) < 0.5 => 1.0,
        "mass_points" => ((unit(41, row) * unit(42, row)) * unit(43, row)) * 1000.0,
        "negative" => {
            let u = unit(50, row);
            -(u * u * 10.0) - 0.5
        }
        "tiny" => match unit(60, row) {
            a if a < 0.2 => 1e-36,
            a if a < 0.3 => -1e-36,
            _ => unit(61, row) - 0.5,
        },
        "categorical" if unit(70, row) < 0.05 => f64::NAN,
        "categorical" => {
            let u = unit(71, row);
            ((u * u * 12.0).floor() * 7.0) % 12.0 - 1.0
        }
        "cat_ties" => f64::from(row % 6),
        "cat_rare" if row % 75_000 == 7 => 40.0,
        "cat_rare" => match unit(80, row) {
            a if a < 0.7 => 5.0,
            a if a < 0.99 => 9.0,
            _ => 2.0,
        },
        "constant" => 3.0,
        other => panic!("unknown feature {other}"),
    }
}

fn case_params(case: &Value) -> Params {
    let mut params = Params::default();
    let overrides = &case["params"];
    let get = |key: &str| overrides.get(key).and_then(Value::as_u64).map(|v| v as usize);
    params.max_bin = get("max_bin").unwrap_or(params.max_bin);
    params.min_data_in_bin = get("min_data_in_bin").unwrap_or(params.min_data_in_bin);
    params.bin_construct_sample_cnt = get("bin_construct_sample_cnt").unwrap_or(params.bin_construct_sample_cnt);
    params
}

/// Asserts LightGBM's thresholds are a subset of `upper_bounds` and the rest border an empty bin.
fn check_thresholds(case: &str, mapper: &BinMapper, column: &[f64], thresholds: &[Value]) {
    let BinMapper::Numerical { upper_bounds, .. } = mapper else {
        panic!("{case}: not numerical: {mapper:?}")
    };
    let lightgbm: Vec<f64> = thresholds
        .iter()
        .map(|t| t.as_str().expect("threshold string").parse().expect("threshold"))
        .collect();
    for threshold in &lightgbm {
        assert!(
            upper_bounds.contains(threshold),
            "{case}: LightGBM threshold {threshold:e} not a silvic bound"
        );
    }
    let mut counts = vec![0usize; mapper.num_bins() as usize];
    for &value in column {
        counts[usize::from(mapper.value_to_bin(value))] += 1;
    }
    let finite = &upper_bounds[..upper_bounds.len() - 1];
    for (t, bound) in finite.iter().enumerate() {
        if !lightgbm.contains(bound) {
            assert!(
                counts[t] == 0 || counts[t + 1] == 0,
                "{case}: bound {t} ({bound:e}) unsplit by LightGBM"
            );
        }
    }
}

#[test]
fn bins_match_lightgbm_3_3_5() {
    let fixture: Value = serde_json::from_str(FIXTURE).expect("fixture json");
    assert_eq!(fixture["lightgbm"], "3.3.5");
    let names: Vec<&str> = fixture["features"]
        .as_array()
        .expect("features")
        .iter()
        .map(|n| n.as_str().expect("name"))
        .collect();
    for case in fixture["cases"].as_array().expect("cases") {
        let case_name = case["name"].as_str().expect("case name");
        let rows = case["rows"].as_u64().expect("rows") as usize;
        let params = case_params(case);
        let columns: Vec<Vec<f64>> = names
            .iter()
            .map(|name| (0..rows as u32).map(|row| feature_value(name, row)).collect())
            .collect();
        let values: Vec<f64> = columns.concat();
        let label = vec![0f32; rows];
        let input = DatasetInput {
            features: DenseMatrix {
                data: DenseData::F64(&values),
                rows,
                cols: names.len(),
                layout: Layout::ColumnMajor,
            },
            label: &label,
            weight: None,
            group: None,
            feature_names: None,
            categorical_features: names
                .iter()
                .enumerate()
                .filter(|(_, n)| n.starts_with("cat"))
                .map(|(i, _)| i)
                .collect(),
        };
        let dataset = BinnedDataset::construct(&input, &params, &mut LocalCollective).expect("construct");

        for ((expected, meta), column) in case["features"]
            .as_array()
            .expect("features")
            .iter()
            .zip(&dataset.features)
            .zip(&columns)
        {
            let label = format!("{case_name}/{}", meta.name);
            let feature = expected["name"].as_str().expect("feature name");
            let infos = expected["feature_infos"].as_str().expect("feature_infos");
            if feature == "constant" {
                // LightGBM drops it only through `feature_pre_filter`, which does not fire when
                // `min_data_in_leaf * sample / rows` rounds to 0 (the small-sample case prints `[3:3]`).
                assert_eq!(meta.bins, BinMapper::Trivial, "{label}");
                continue;
            }
            assert_eq!(meta.bins.info_string(), infos, "{label}: feature_infos");
            if let Some(thresholds) = expected.get("thresholds").and_then(Value::as_array) {
                check_thresholds(&label, &meta.bins, column, thresholds);
            }
        }
    }
}
