//! Continued training (`train_from`): a run checkpointed after some trees and resumed from the checkpoint's model
//! text gives the model string of the uninterrupted run, byte for byte, and the same evaluations on the way.

#[path = "../src/test_util.rs"]
mod test_util;

use silvic_core::collective::Collective;
use silvic_core::data::{DatasetInput, DenseData, DenseMatrix, Layout};
use silvic_core::metric::EvalResult;
use silvic_core::quantize;
use silvic_core::{train, train_from, BinnedDataset, CpuBackend, LocalCollective, Model, Params, Result};
use test_util::run_ranks;

const TOTAL: usize = 30;
const CHECKPOINT: usize = 12;
const COLS: usize = 12;
const CATEGORICAL: usize = 5;

/// Rows of a hashed synthetic set: one categorical column, NaN in column 3, a score the labels come from.
struct Rows {
    x: Vec<f32>,
    score: Vec<f64>,
    group: Vec<u32>,
}

fn generate(first_row: u64, rows: usize, queries: usize) -> Rows {
    let u = |stream: u32, row: u64| f64::from(quantize::uniform(7, stream, 0, row));
    let mut x = Vec::with_capacity(rows * COLS);
    let mut score = Vec::with_capacity(rows);
    for row in first_row..first_row + rows as u64 {
        let values: Vec<f64> = (0..COLS as u32)
            .map(|j| match j as usize {
                CATEGORICAL => (u(j, row) * 6.0).floor(),
                3 if u(100, row) < 0.1 => f64::NAN,
                _ => u(j, row),
            })
            .collect();
        let x3 = if values[3].is_nan() { 0.5 } else { values[3] };
        score.push(
            3.0 * values[0] + 2.0 * values[1] * values[2] - x3
                + [0.0, 0.7, -0.4, 0.2, 0.0, -0.6][values[CATEGORICAL] as usize]
                + u(99, row),
        );
        x.extend(values.iter().map(|&v| v as f32));
    }
    let per_query = (rows / queries) as u32;
    let mut group = vec![per_query; queries];
    group[queries - 1] += rows as u32 - per_query * queries as u32;
    Rows { x, score, group }
}

#[derive(Clone, Copy, PartialEq)]
enum Task {
    Rank,
    Binary,
    Regression,
    /// Regression with labels around 1000: a large initial score folded into small first-tree leaves.
    Offset,
}

impl Rows {
    fn labels(&self, task: Task) -> Vec<f32> {
        self.score
            .iter()
            .map(|&s| match task {
                Task::Rank => (s - 1.0).floor().clamp(0.0, 3.0) as f32,
                Task::Binary => f32::from(u8::from(s > 2.5)),
                Task::Regression => s as f32,
                Task::Offset => (s + 1000.0) as f32,
            })
            .collect()
    }

    /// Whole queries `queries`, as one rank holds them.
    fn queries(&self, queries: std::ops::Range<usize>) -> Rows {
        let start: usize = self.group[..queries.start].iter().map(|&g| g as usize).sum();
        let len: usize = self.group[queries.clone()].iter().map(|&g| g as usize).sum();
        Rows {
            x: self.x[start * COLS..(start + len) * COLS].to_vec(),
            score: self.score[start..start + len].to_vec(),
            group: self.group[queries].to_vec(),
        }
    }
}

fn bin(
    rows: &Rows,
    task: Task,
    params: &Params,
    reference: Option<&BinnedDataset>,
    c: &mut dyn Collective,
) -> BinnedDataset {
    let label = rows.labels(task);
    let input = DatasetInput {
        features: DenseMatrix {
            data: DenseData::F32(&rows.x),
            rows: label.len(),
            cols: COLS,
            layout: Layout::RowMajor,
        },
        label: &label,
        weight: None,
        group: (task == Task::Rank).then_some(&rows.group[..]),
        feature_names: None,
        categorical_features: vec![CATEGORICAL],
    };
    match reference {
        None => BinnedDataset::construct(&input, params, c).expect("train set"),
        Some(reference) => BinnedDataset::construct_with_reference(&input, reference, c).expect("valid set"),
    }
}

fn params(task: Task, extra: &str, iterations: usize) -> Params {
    let objective = match task {
        Task::Rank => r#""objective": "lambdarank", "eval_at": [5]"#,
        Task::Binary => r#""objective": "binary""#,
        Task::Regression | Task::Offset => r#""objective": "regression""#,
    };
    let rate = if extra.contains("learning_rate") {
        ""
    } else {
        r#", "learning_rate": 0.2"#
    };
    Params::from_json(&format!(
        r#"{{{objective}, "num_iterations": {iterations}, "num_leaves": 15, "min_data_in_leaf": 10{rate}{extra}}}"#
    ))
    .expect("params")
}

/// One run: the model string, every iteration's (number, evaluations), and the model of iteration `checkpoint`.
struct Run {
    model: String,
    evals: Vec<(usize, Vec<EvalResult>)>,
    checkpoint: Option<Model>,
    last_report_model: Option<Model>,
}

fn fit(
    params: &Params,
    init: Option<&Model>,
    data: (&BinnedDataset, &BinnedDataset),
    checkpoint: usize,
    collective: &mut dyn Collective,
) -> Run {
    let mut backend = CpuBackend::borrowing(data.0, &[data.1]);
    let mut evals = Vec::new();
    let (mut saved, mut last) = (None, None);
    let model = train_from(
        params,
        init,
        data.0,
        &[("valid", data.1)],
        &mut backend,
        collective,
        &mut |report| {
            evals.push((report.iteration, report.evals.to_vec()));
            if report.iteration == checkpoint {
                saved = Some(report.model());
            }
            last = Some(report.model());
            Ok(())
        },
    )
    .expect("train");
    Run {
        model: model.to_lightgbm_string(),
        evals,
        checkpoint: saved,
        last_report_model: last.filter(|m| *m == model),
    }
}

/// Full run against a run checkpointed at `CHECKPOINT - 1` and resumed from the checkpoint's model text.
fn assert_resume_matches(task: Task, extra: &str, collective: &mut dyn Collective, rows: &Rows, valid_rows: &Rows) {
    let case = format!(
        "{} {extra} rank {}",
        ["rank", "binary", "regression", "offset"][task as usize],
        collective.rank()
    );
    let full_params = params(task, extra, TOTAL);
    let train_set = bin(rows, task, &full_params, None, collective);
    let valid_set = bin(valid_rows, task, &full_params, Some(&train_set), collective);
    let data = (&train_set, &valid_set);
    let full = fit(&full_params, None, data, usize::MAX, collective);
    assert!(
        full.last_report_model.is_some(),
        "{case}: report.model() at the last iteration is the returned model"
    );
    assert_eq!(full.evals.len(), TOTAL, "{case}: trees");
    let first = fit(&params(task, extra, CHECKPOINT), None, data, CHECKPOINT - 1, collective);
    let text = first.checkpoint.expect("checkpoint").to_lightgbm_string();
    let init = Model::from_lightgbm_string(&text).expect("checkpoint parses");
    let resumed = fit(
        &params(task, extra, TOTAL - CHECKPOINT),
        Some(&init),
        data,
        usize::MAX,
        collective,
    );
    assert_eq!(
        resumed.evals.first().map(|e| e.0),
        Some(CHECKPOINT),
        "{case}: iterations continue"
    );
    assert!(
        resumed.last_report_model.is_some(),
        "{case}: report.model() of the resumed run"
    );
    assert_eq!(
        resumed.evals,
        full.evals[CHECKPOINT..],
        "{case}: evaluations after the checkpoint"
    );
    assert!(
        resumed.model == full.model,
        "{case}: resumed model differs from the uninterrupted one"
    );
}

const RANDOMIZED: &str = r#", "bagging_fraction": 0.7, "bagging_freq": 3, "feature_fraction": 0.8, "lambda_l1": 0.1"#;

#[test]
fn resumed_runs_match_uninterrupted_ones() {
    let rows = generate(0, 8_000, 80);
    let valid_rows = generate(1_000_000, 2_000, 20);
    let cases = [
        (Task::Rank, RANDOMIZED),
        (Task::Rank, r#", "use_quantized_grad": false"#),
        (Task::Binary, RANDOMIZED),
        (Task::Binary, r#", "use_quantized_grad": false"#),
        (Task::Regression, RANDOMIZED),
        (Task::Regression, r#", "use_quantized_grad": false, "lambda_l2": 1.0"#),
        (Task::Offset, r#", "learning_rate": 0.001"#),
    ];
    for (task, extra) in cases {
        assert_resume_matches(task, extra, &mut LocalCollective, &rows, &valid_rows);
    }
}

#[test]
fn resumed_runs_match_on_two_ranks() {
    let rows = generate(0, 8_000, 80);
    let valid_rows = generate(1_000_000, 2_000, 20);
    let shards = [rows.queries(0..35), rows.queries(35..80)];
    let valid_shards = [valid_rows.queries(0..9), valid_rows.queries(9..20)];
    for task in [Task::Rank, Task::Binary, Task::Regression] {
        run_ranks(2, |rank, collective| {
            assert_resume_matches(task, RANDOMIZED, collective, &shards[rank], &valid_shards[rank]);
        });
    }
}

#[test]
fn init_model_must_match_objective_and_features() {
    let rows = generate(0, 2_000, 20);
    let valid_rows = generate(1_000_000, 500, 5);
    let rank_params = params(Task::Rank, "", 3);
    let train_set = bin(&rows, Task::Rank, &rank_params, None, &mut LocalCollective);
    let valid_set = bin(
        &valid_rows,
        Task::Rank,
        &rank_params,
        Some(&train_set),
        &mut LocalCollective,
    );
    let mut backend = CpuBackend::borrowing(&train_set, &[]);
    let model = train(
        &rank_params,
        &train_set,
        &[],
        &mut backend,
        &mut LocalCollective,
        &mut |_| Ok(()),
    )
    .expect("train");
    let regression = params(Task::Regression, "", 3);
    let regression_set = bin(&rows, Task::Regression, &regression, None, &mut LocalCollective);
    let mut backend = CpuBackend::borrowing(&regression_set, &[]);
    let error = train_from(
        &regression,
        Some(&model),
        &regression_set,
        &[],
        &mut backend,
        &mut LocalCollective,
        &mut |_| Ok(()),
    )
    .expect_err("objective mismatch");
    assert_eq!(error.kind(), "config", "{error}");
    let mut fewer = model.clone();
    fewer.feature_names.pop();
    fewer.feature_infos.pop();
    let mut backend = CpuBackend::borrowing(&train_set, &[&valid_set]);
    let error = train_from(
        &rank_params,
        Some(&fewer),
        &train_set,
        &[("valid", &valid_set)],
        &mut backend,
        &mut LocalCollective,
        &mut |_| Ok(()),
    )
    .expect_err("feature count mismatch");
    assert_eq!(error.kind(), "config", "{error}");
}
