//! Compares `metric::evaluate` against metric values recorded from LightGBM 4.7.0 (`fixtures/metrics/generate.py`).

use serde_json::Value;

use silvic_core::config::{Metric, Objective};
use silvic_core::metric::evaluate;
use silvic_core::test_util::run_ranks;
use silvic_core::{BinnedDataset, LocalCollective, Params};

const CASES: &str = include_str!("fixtures/metrics/cases.json");

struct Case {
    name: String,
    params: Params,
    label: Vec<f32>,
    weight: Option<Vec<f32>>,
    group: Option<Vec<u32>>,
    scores: Vec<f32>,
    expected: Vec<(String, f64)>,
}

fn f32s(value: &Value) -> Vec<f32> {
    value
        .as_array()
        .expect("array")
        .iter()
        .map(|v| v.as_f64().expect("number") as f32)
        .collect()
}

fn parse_case(case: &Value) -> Case {
    let objective = match case["objective"].as_str().expect("objective") {
        "lambdarank" => Objective::LambdaRank,
        "binary" => Objective::Binary,
        _ => Objective::Regression,
    };
    let metric_names: Vec<String> = match &case["metric"] {
        Value::String(name) => vec![name.clone()],
        names => names
            .as_array()
            .expect("metrics")
            .iter()
            .map(|n| n.as_str().expect("name").to_string())
            .collect(),
    };
    let metric = metric_names
        .iter()
        .map(|name| match name.as_str() {
            "ndcg" => Metric::Ndcg,
            "l2" => Metric::L2,
            "rmse" => Metric::Rmse,
            "binary_logloss" => Metric::BinaryLogloss,
            "auc" => Metric::Auc,
            other => panic!("unknown metric {other}"),
        })
        .collect();
    let params = Params {
        objective,
        sigmoid: case["sigmoid"].as_f64().expect("sigmoid"),
        metric,
        eval_at: case["eval_at"]
            .as_array()
            .expect("eval_at")
            .iter()
            .map(|k| k.as_u64().expect("k") as usize)
            .collect(),
        ..Params::default()
    };
    let group = case["group"].as_array().map(|sizes| {
        let mut boundaries = vec![0u32];
        for size in sizes {
            boundaries.push(boundaries.last().expect("boundary") + size.as_u64().expect("size") as u32);
        }
        boundaries
    });
    Case {
        name: case["name"].as_str().expect("name").to_string(),
        params,
        label: f32s(&case["label"]),
        weight: (!case["weight"].is_null()).then(|| f32s(&case["weight"])),
        group,
        scores: f32s(&case["scores"]),
        expected: case["expected"]
            .as_object()
            .expect("expected")
            .iter()
            .map(|(k, v)| (k.clone(), v.as_f64().expect("value")))
            .collect(),
    }
}

fn cases() -> Vec<Case> {
    let fixture: Value = serde_json::from_str(CASES).expect("fixture json");
    fixture["cases"]
        .as_array()
        .expect("cases")
        .iter()
        .map(parse_case)
        .collect()
}

/// Rows `rows` of the case as a dataset without features (metrics only read labels, weights and queries).
fn dataset(case: &Case, rows: std::ops::Range<usize>, group: Option<Vec<u32>>) -> BinnedDataset {
    BinnedDataset {
        num_rows: rows.len(),
        row_offset: rows.start as u64,
        features: Vec::new(),
        groups: Vec::new(),
        total_bins: 0,
        label: case.label[rows.clone()].to_vec(),
        weight: case.weight.as_ref().map(|w| w[rows].to_vec()),
        query_boundaries: group,
    }
}

fn assert_matches(case: &Case, context: &str, results: &[silvic_core::metric::EvalResult]) {
    assert_eq!(
        results.len(),
        case.expected.len(),
        "{}{context}: metric count",
        case.name
    );
    for (name, want) in &case.expected {
        let got = results
            .iter()
            .find(|r| &r.metric == name)
            .unwrap_or_else(|| panic!("{}: {name}", case.name));
        assert!(
            (got.value - want).abs() <= 1e-12 * want.abs(),
            "{}{context}: {name} = {} vs LightGBM {want}",
            case.name,
            got.value
        );
        assert_eq!(got.dataset, "valid_0", "{}: dataset name", case.name);
        assert_eq!(
            got.higher_is_better,
            name.starts_with("ndcg") || name == "auc",
            "{}: {name} direction",
            case.name
        );
    }
}

#[test]
fn metrics_match_lightgbm() {
    for case in cases() {
        let data = dataset(&case, 0..case.label.len(), case.group.clone());
        let results = evaluate(&case.params, "valid_0", &case.scores, &data, &mut LocalCollective).expect("evaluate");
        assert_matches(&case, "", &results);
    }
}

#[test]
fn metrics_reduce_over_two_ranks() {
    for case in cases() {
        // Split on a query boundary, as distributed ranking keeps whole queries on one rank.
        let split = match &case.group {
            Some(boundaries) => boundaries[boundaries.len() / 2] as usize,
            None => case.label.len() / 3,
        };
        let parts = [0..split, split..case.label.len()];
        let results = run_ranks(2, |rank, collective| {
            let rows = parts[rank].clone();
            let group = case.group.as_ref().map(|b| {
                b.iter()
                    .filter(|&&x| rows.contains(&(x as usize)) || x as usize == rows.end)
                    .map(|&x| x - rows.start as u32)
                    .collect()
            });
            let data = dataset(&case, rows.clone(), group);
            evaluate(&case.params, "valid_0", &case.scores[rows], &data, collective).expect("evaluate")
        });
        for (rank, result) in results.iter().enumerate() {
            assert_matches(&case, &format!(" rank {rank}"), result);
        }
    }
}
