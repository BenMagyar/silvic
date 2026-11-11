//! Raw scores and prefixes match recorded LightGBM predictions. Transformed scores match the host's core
//! predictor; binary probabilities depend on the host math library, not a cross-platform saved `exp` result.

use std::path::{Path, PathBuf};

use serde_json::Value;
use silvic_core::data::{DenseData, DenseMatrix, Layout};
use silvic_core::Model;
use silvic_serve::{Kernel, Predictor};

struct Case {
    name: String,
    model: Model,
    rows: Vec<Vec<f64>>,
    raw: Vec<f64>,
    prefix_trees: usize,
    raw_prefix: Vec<f64>,
}

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../silvic-core/tests/fixtures/lightgbm")
}

fn floats(value: &Value) -> Vec<f64> {
    value
        .as_array()
        .expect("float list")
        .iter()
        .map(|v| v.as_str().expect("float string").parse().expect("float"))
        .collect()
}

fn load_cases(file: &str) -> Vec<Case> {
    let text = std::fs::read_to_string(fixtures().join(file)).expect("cases file");
    let document: Value = serde_json::from_str(&text).expect("cases json");
    document["cases"]
        .as_array()
        .expect("cases")
        .iter()
        .map(|case| {
            let name = case["name"].as_str().expect("name").to_string();
            let model_file = fixtures().join(case["model_file"].as_str().expect("model_file"));
            let text = std::fs::read_to_string(model_file).expect("model file");
            Case {
                model: Model::from_lightgbm_string(&text).unwrap_or_else(|e| panic!("{name}: parse: {e}")),
                name,
                rows: case["rows"].as_array().expect("rows").iter().map(floats).collect(),
                raw: floats(&case["raw"]),
                prefix_trees: case["prefix_trees"].as_u64().expect("prefix_trees") as usize,
                raw_prefix: floats(&case["raw_prefix"]),
            }
        })
        .collect()
}

fn assert_bits(case: &str, what: &str, got: &[f64], expected: &[f64]) {
    assert_eq!(got.len(), expected.len(), "{case}: {what} count");
    for (row, (g, e)) in got.iter().zip(expected).enumerate() {
        assert_eq!(
            g.to_bits(),
            e.to_bits(),
            "{case}: {what} row {row}: silvic.serve {g:?}, reference {e:?}"
        );
    }
}

#[test]
fn predictions_match_lightgbm_raw_and_host_transform() {
    let cases: Vec<Case> = load_cases("cases.json")
        .into_iter()
        .chain(load_cases("cases_v4.json"))
        .collect();
    for case in &cases {
        let cols = case.rows[0].len();
        let flat: Vec<f64> = case.rows.concat();
        let matrix = DenseMatrix {
            data: DenseData::F64(&flat),
            rows: case.rows.len(),
            cols,
            layout: Layout::RowMajor,
        };
        let expected = case.model.predict(&matrix, false, None);
        for kernel in Kernel::available() {
            let label = format!("{} {kernel}", case.name);
            let predictor = Predictor::with_kernel(&case.model, kernel).expect("compile");
            let mut raw = vec![0.0; case.rows.len()];
            predictor.predict_raw(&matrix, None, &mut raw);
            assert_bits(&label, "raw", &raw, &case.raw);
            let mut prefix = vec![0.0; case.rows.len()];
            predictor.predict_raw(&matrix, Some(case.prefix_trees), &mut prefix);
            assert_bits(&label, "raw prefix", &prefix, &case.raw_prefix);
            for threads in [1, 3] {
                let predicted = predictor.predict(&matrix, false, None, threads);
                assert_bits(&label, &format!("predicted, {threads} threads"), &predicted, &expected);
            }
        }
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_predictions_match_lightgbm_raw_and_host_transform() {
    use silvic_serve::{cuda_is_available, CudaPredictor};

    if !cuda_is_available() {
        eprintln!("skipping CUDA fixtures: driver/NVRTC/device unavailable");
        return;
    }
    for case in load_cases("cases.json").into_iter().chain(load_cases("cases_v4.json")) {
        let flat = case.rows.concat();
        let matrix = DenseMatrix {
            data: DenseData::F64(&flat),
            rows: case.rows.len(),
            cols: case.rows[0].len(),
            layout: Layout::RowMajor,
        };
        let mut predictor = CudaPredictor::new(&case.model, None).expect("CUDA compile");
        let raw = predictor.predict(&matrix, true, None).expect("CUDA raw");
        assert_bits(&case.name, "CUDA raw", &raw, &case.raw);
        let prefix = predictor
            .predict(&matrix, true, Some(case.prefix_trees))
            .expect("CUDA prefix");
        assert_bits(&case.name, "CUDA prefix", &prefix, &case.raw_prefix);
        let predicted = predictor.predict(&matrix, false, None).expect("CUDA predicted");
        assert_bits(
            &case.name,
            "CUDA predicted",
            &predicted,
            &case.model.predict(&matrix, false, None),
        );
    }
}
