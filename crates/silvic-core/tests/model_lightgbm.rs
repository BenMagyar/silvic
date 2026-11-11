//! Compares against models and predictions recorded from LightGBM 3.3.5.
//!
//! `tests/fixtures/lightgbm/gen_models.py` recorded the models (`<case>.txt`) and LightGBM's predictions for their
//! edge-case rows (`cases.json`); `gen_models_v4.py` recorded a subset with LightGBM 4.7 (`cases_v4.json`), which
//! silvic parses and predicts but does not write back byte for byte (it writes `version=v3`). The last test writes
//! models silvic builds by hand and has LightGBM 3.3.5, LightGBM 4.x and lleaves 1.1.0 predict them through
//! `check_load.py`; it is skipped, with the reason printed, when the repo's `.venv-lgb3`/`.venv` are absent.

use std::path::{Path, PathBuf};
use std::process::Command;

use serde_json::{json, Value};
use silvic_core::config::{Objective, Params};
use silvic_core::data::{DenseData, DenseMatrix, Layout};
use silvic_core::model::{ImportanceType, Model, ModelObjective};
use silvic_core::tree::{MissingType, SplitChildren, Tree, CATEGORICAL_MASK};

const V3_CASES: &str = "cases.json";
const V4_CASES: &str = "cases_v4.json";
/// LightGBM `kZeroThreshold` (the float `1e-35f`).
const ZERO_THRESHOLD: f64 = 1e-35_f32 as f64;

struct Case {
    name: String,
    model_text: String,
    rows: Vec<Vec<f64>>,
    raw: Vec<f64>,
    predicted: Vec<f64>,
    prefix_trees: usize,
    raw_prefix: Vec<f64>,
    importance_split: Vec<f64>,
    importance_gain: Vec<f64>,
}

impl Case {
    fn model(&self) -> Model {
        Model::from_lightgbm_string(&self.model_text).unwrap_or_else(|e| panic!("{}: parse: {e}", self.name))
    }
}

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/lightgbm")
}

fn floats(value: &Value) -> Vec<f64> {
    let values = value.as_array().expect("float list");
    values
        .iter()
        .map(|v| v.as_str().expect("float string").parse().expect("float"))
        .collect()
}

fn load_cases(file: &str) -> Vec<Case> {
    let text = std::fs::read_to_string(fixtures().join(file)).expect("cases file");
    let document: Value = serde_json::from_str(&text).expect("cases json");
    let cases = document["cases"].as_array().expect("cases");
    cases
        .iter()
        .map(|case| Case {
            name: case["name"].as_str().expect("name").to_string(),
            model_text: std::fs::read_to_string(fixtures().join(case["model_file"].as_str().expect("model_file")))
                .expect("model file"),
            rows: case["rows"].as_array().expect("rows").iter().map(floats).collect(),
            raw: floats(&case["raw"]),
            predicted: floats(&case["predicted"]),
            prefix_trees: case["prefix_trees"].as_u64().expect("prefix_trees") as usize,
            raw_prefix: floats(&case["raw_prefix"]),
            importance_split: case["importance_split"]
                .as_array()
                .expect("importance_split")
                .iter()
                .map(|v| v.as_f64().expect("split count"))
                .collect(),
            importance_gain: floats(&case["importance_gain"]),
        })
        .collect()
}

fn all_cases() -> Vec<Case> {
    let mut cases = load_cases(V3_CASES);
    cases.extend(load_cases(V4_CASES));
    cases
}

/// The first line where two texts differ, for a readable failure.
fn first_difference(written: &str, expected: &str) -> String {
    let mut written_lines = written.split('\n');
    for (number, line) in expected.split('\n').enumerate() {
        let got = written_lines.next();
        if got != Some(line) {
            return format!("line {}: wrote {got:?}, LightGBM {line:?}", number + 1);
        }
    }
    format!("silvic wrote extra lines: {:?}", written_lines.collect::<Vec<_>>())
}

fn assert_bits(case: &str, what: &str, got: &[f64], expected: &[f64], rows: &[Vec<f64>]) {
    assert_eq!(got.len(), expected.len(), "{case}: {what} count");
    for (i, (g, e)) in got.iter().zip(expected).enumerate() {
        assert_eq!(
            g.to_bits(),
            e.to_bits(),
            "{case}: {what} row {i} {:?}: silvic {g:?}, lightgbm {e:?}",
            rows[i]
        );
    }
}

#[test]
fn lightgbm_files_write_back_byte_for_byte() {
    for case in load_cases(V3_CASES) {
        let written = case.model().to_lightgbm_string();
        assert!(
            written == case.model_text,
            "{}: {}",
            case.name,
            first_difference(&written, &case.model_text)
        );
    }
}

#[test]
fn parsing_written_models_is_lossless() {
    for case in all_cases() {
        let model = case.model();
        let reparsed = Model::from_lightgbm_string(&model.to_lightgbm_string()).expect("reparse");
        assert!(reparsed == model, "{}: reparsed model differs", case.name);
    }
}

#[test]
fn raw_predictions_match_lightgbm_bit_for_bit() {
    for case in all_cases() {
        let model = case.model();
        let raw: Vec<f64> = case.rows.iter().map(|row| model.predict_raw_row(row, None)).collect();
        assert_bits(&case.name, "raw", &raw, &case.raw, &case.rows);
        let prefix: Vec<f64> = case
            .rows
            .iter()
            .map(|row| model.predict_raw_row(row, Some(case.prefix_trees)))
            .collect();
        assert_bits(&case.name, "raw prefix", &prefix, &case.raw_prefix, &case.rows);
    }
}

#[test]
fn predict_matches_lightgbm_in_every_layout() {
    for case in all_cases() {
        let model = case.model();
        let (rows, cols) = (case.rows.len(), case.rows[0].len());
        let row_major: Vec<f64> = case.rows.concat();
        let column_major: Vec<f64> = (0..cols)
            .flat_map(|c| case.rows.iter().map(move |row| row[c]))
            .collect();
        let matrix = DenseMatrix {
            data: DenseData::F64(&row_major),
            rows,
            cols,
            layout: Layout::RowMajor,
        };
        assert_bits(
            &case.name,
            "predicted",
            &model.predict(&matrix, false, None),
            &case.predicted,
            &case.rows,
        );
        let matrix = DenseMatrix {
            data: DenseData::F64(&column_major),
            rows,
            cols,
            layout: Layout::ColumnMajor,
        };
        assert_bits(
            &case.name,
            "column-major raw",
            &model.predict(&matrix, true, None),
            &case.raw,
            &case.rows,
        );
    }
}

#[test]
fn binary_predictions_are_the_sigmoid_of_raw_scores() {
    let case = load_cases(V3_CASES)
        .into_iter()
        .find(|case| case.name == "binary")
        .expect("binary case");
    let model = case.model();
    assert_eq!(model.objective, ModelObjective::Binary { sigmoid: 0.7 });
    assert!(
        case.predicted.iter().zip(&case.raw).any(|(p, r)| p != r),
        "binary: transform recorded"
    );
}

#[test]
fn feature_importance_matches_lightgbm() {
    for case in all_cases() {
        let model = case.model();
        assert_eq!(
            model.feature_importance(ImportanceType::Split),
            case.importance_split,
            "{}: split",
            case.name
        );
        let gain = model.feature_importance(ImportanceType::Gain);
        let bits = |values: &[f64]| values.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&gain), bits(&case.importance_gain), "{}: gain {gain:?}", case.name);
    }
}

fn children(left_value: f64, right_value: f64, left_count: i64, right_count: i64, gain: f32) -> SplitChildren {
    SplitChildren {
        left_value,
        right_value,
        left_count,
        right_count,
        left_weight: left_count as f64 * 0.2375,
        right_weight: right_count as f64 * 0.2375,
        gain,
    }
}

/// Trees with every split kind: NaN, None and Zero missing types in both default directions, multi-word bitsets,
/// thresholds at `+-kZeroThreshold`, a one-leaf tree, and the bias folded into the first tree.
fn hand_built_trees() -> Vec<Tree> {
    let mut first = Tree::new();
    first.split_numerical(
        0,
        0,
        0.5,
        true,
        MissingType::NaN,
        &children(-0.8125, 1.3, 400, 600, 812.5),
    );
    first.split_categorical(
        0,
        3,
        &[1, 3, 33, 40, 65],
        MissingType::None,
        &children(0.37, -1.1, 150, 250, 90.25),
    );
    first.split_numerical(
        1,
        2,
        -1.0,
        false,
        MissingType::Zero,
        &children(0.123456789, 2.2, 200, 400, 45.0),
    );
    first.split_numerical(
        3,
        1,
        ZERO_THRESHOLD,
        true,
        MissingType::None,
        &children(-0.3, 0.71, 300, 100, 12.0),
    );
    first.shrink(0.1);
    first.add_bias(std::f64::consts::LN_2);

    let mut constant = Tree::new();
    constant.as_constant(-0.012345678901234568);

    let mut third = Tree::new();
    third.split_categorical(
        0,
        3,
        &[0, 2, 7],
        MissingType::None,
        &children(0.5, -0.25, 500, 500, 30.5),
    );
    third.split_numerical(
        1,
        0,
        -0.25,
        false,
        MissingType::NaN,
        &children(1.0 / 3.0, -2.0 / 3.0, 200, 300, 7.75),
    );
    third.split_numerical(0, 2, 1.5, true, MissingType::Zero, &children(-0.05, 0.9, 250, 250, 3.0));
    third.split_numerical(
        2,
        1,
        -ZERO_THRESHOLD,
        false,
        MissingType::None,
        &children(0.2, -0.4, 100, 200, 1.5),
    );
    third.shrink(0.1);

    let mut fourth = Tree::new();
    fourth.split_categorical(
        0,
        3,
        &[64, 95],
        MissingType::None,
        &children(-0.6, 0.15, 100, 900, 22.0),
    );
    // Not 0.0: lleaves asserts numerical thresholds are truthy; LightGBM (and silvic) bins put zero between
    // `+-kZeroThreshold`, so no trained split sits at 0.0.
    fourth.split_numerical(
        1,
        4,
        0.25,
        true,
        MissingType::None,
        &children(0.01, 0.02, 450, 450, 0.0),
    );
    fourth.split_numerical(
        1,
        1,
        f64::MAX,
        true,
        MissingType::NaN,
        &children(0.3, -0.7, 400, 50, 2.0),
    );
    fourth.shrink(0.05);
    vec![first, constant, third, fourth]
}

fn hand_built_model(objective: Objective) -> Model {
    let params = Params {
        objective,
        sigmoid: 0.8,
        ..Params::default()
    };
    Model {
        objective: match objective {
            Objective::Regression => ModelObjective::Regression,
            Objective::Binary => ModelObjective::Binary { sigmoid: 0.8 },
            Objective::LambdaRank => ModelObjective::LambdaRank,
        },
        feature_names: ["num_nan", "num_plain", "num_zero", "cat_wide", "num_rare"]
            .map(String::from)
            .to_vec(),
        feature_infos: ["[-3:3]", "[-2.5:2.5]", "[-4:4]", "1:3:33:40:65:0:2:7:64:95", "[-1:1]"]
            .map(String::from)
            .to_vec(),
        trees: hand_built_trees(),
        parameters: params.model_parameters(),
        pandas_categorical: None,
    }
}

/// xorshift64*: reproducible rows without a dependency.
struct Rows(u64);

impl Rows {
    fn next(&mut self) -> f64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        (self.0.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 11) as f64 / (1u64 << 53) as f64
    }

    fn row(&mut self) -> Vec<f64> {
        let nan_or = |draw: f64, value: f64| if draw < 0.1 { f64::NAN } else { value };
        let a = nan_or(self.next(), self.next() * 6.0 - 3.0);
        let b = self.next() * 5.0 - 2.5;
        let c = if self.next() < 0.3 {
            0.0
        } else {
            nan_or(self.next(), self.next() * 8.0 - 4.0)
        };
        let category = nan_or(self.next(), (self.next() * 110.0).floor() - 5.0);
        vec![a, b, c, category, self.next() * 2.0 - 1.0]
    }
}

/// Random rows, every feature set to each edge value, and every numerical threshold with its neighbours.
fn check_rows(model: &Model) -> Vec<Vec<f64>> {
    let mut generator = Rows(0x9e37_79b9_7f4a_7c15);
    let mut rows: Vec<Vec<f64>> = (0..300).map(|_| generator.row()).collect();
    let edges = [
        f64::NAN,
        0.0,
        -0.0,
        1e-36,
        -1e-36,
        ZERO_THRESHOLD,
        -ZERO_THRESHOLD,
        2e-35,
        -2e-35,
        f64::INFINITY,
        f64::NEG_INFINITY,
        1e300,
        -1e300,
        -1.0,
        -0.5,
        2.7,
        33.0,
        64.0,
        95.0,
        96.0,
        2147483648.0,
        1e10,
        -1e10,
    ];
    for feature in 0..model.num_features() {
        for (i, &edge) in edges.iter().enumerate() {
            let mut row = rows[i].clone();
            row[feature] = edge;
            rows.push(row);
        }
    }
    for tree in &model.trees {
        for node in 0..tree.num_leaves - 1 {
            if tree.decision_type[node] & CATEGORICAL_MASK == 0 {
                let threshold = tree.threshold[node];
                for value in [threshold, threshold.next_down(), threshold.next_up()] {
                    let mut row = rows[node].clone();
                    row[tree.split_feature[node] as usize] = value;
                    rows.push(row);
                }
            }
        }
    }
    rows
}

fn float_strings(values: &[f64]) -> Vec<String> {
    values.iter().map(|v| format!("{v:?}")).collect()
}

/// Writes the hand-built models, their rows and silvic's predictions for `check_load.py`; returns the cases file.
fn write_check_cases(dir: &Path) -> PathBuf {
    let mut cases = Vec::new();
    for (name, objective) in [
        ("binary", Objective::Binary),
        ("regression", Objective::Regression),
        ("lambdarank", Objective::LambdaRank),
    ] {
        let model = hand_built_model(objective);
        let text = model.to_lightgbm_string();
        assert_eq!(
            Model::from_lightgbm_string(&text)
                .expect("reparse")
                .to_lightgbm_string(),
            text,
            "{name}"
        );
        let model_file = format!("silvic_{name}.txt");
        std::fs::write(dir.join(&model_file), &text).expect("write model");
        let rows = check_rows(&model);
        let flat = rows.concat();
        let matrix = DenseMatrix {
            data: DenseData::F64(&flat),
            rows: rows.len(),
            cols: 5,
            layout: Layout::RowMajor,
        };
        let tiny = |v: &f64| *v != 0.0 && v.abs() <= ZERO_THRESHOLD;
        let lleaves_skip: Vec<usize> = (0..rows.len()).filter(|&i| rows[i].iter().any(tiny)).collect();
        cases.push(json!({
            "name": format!("silvic_{name}"),
            "model_file": model_file,
            "rows": rows.iter().map(|row| float_strings(row)).collect::<Vec<_>>(),
            "raw": float_strings(&model.predict(&matrix, true, None)),
            "predicted": float_strings(&model.predict(&matrix, false, None)),
            "lleaves_skip": lleaves_skip,
        }));
    }
    let cases_file = dir.join("silvic_cases.json");
    std::fs::write(&cases_file, json!({ "cases": cases }).to_string()).expect("write cases");
    cases_file
}

#[test]
fn silvic_models_load_in_lightgbm_and_lleaves() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let runs = [(".venv-lgb3", true), (".venv", false)];
    let missing: Vec<String> = runs
        .iter()
        .map(|(venv, _)| root.join(venv).join("bin/python"))
        .filter(|python| !python.exists())
        .map(|python| python.display().to_string())
        .collect();
    if !missing.is_empty() {
        eprintln!(
            "SKIPPED silvic_models_load_in_lightgbm_and_lleaves: no {}",
            missing.join(", ")
        );
        return;
    }
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("model_lightgbm");
    std::fs::create_dir_all(&dir).expect("temp dir");
    let cases_file = write_check_cases(&dir);
    for (venv, lleaves) in runs {
        let mut command = Command::new(root.join(venv).join("bin/python"));
        command.arg(fixtures().join("check_load.py")).arg(&cases_file);
        if lleaves {
            command.arg("--lleaves");
        }
        let output = command.output().expect("run check_load.py");
        let stdout = String::from_utf8_lossy(&output.stdout);
        eprintln!("{venv}: {}", stdout.trim_end());
        assert!(
            output.status.success(),
            "{venv}: {stdout}{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
}
