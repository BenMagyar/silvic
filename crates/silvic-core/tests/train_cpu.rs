//! Trains through `CpuBackend` on synthetic data and compares lambdarank accuracy against LightGBM 4.7's validation
//! NDCG@10 recorded by `fixtures/lambdarank_parity.py`, which generates the same rows from the same hash.

use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use silvic_core::backend::{BagSpec, BinSplit, DeviceTree, GradientStats, LeafRange, ObjectiveSetup};
use silvic_core::collective::Collective;
use silvic_core::data::{DatasetInput, DenseData, DenseMatrix, Layout};
use silvic_core::metric::EvalResult;
use silvic_core::quantize::{self, QuantSpec};
use silvic_core::test_util::run_ranks;
use silvic_core::{train, Backend, BinnedDataset, CpuBackend, LocalCollective, Model, Params, Result};

const SEED: u32 = 20_261_107;
const VALID_ROW_OFFSET: u64 = 10_000_000;
const NOISE_STREAM: u32 = 999;
const NAN_STREAM: u32 = 1000;
const CATEGORICAL: usize = 7;
const CAT_EFFECT: [f64; 8] = [0.0, 0.8, -0.5, 0.3, 0.0, -0.8, 0.6, 0.1];

/// Rows of the generator shared with `fixtures/lambdarank_parity.py`.
struct Synthetic {
    x: Vec<f32>,
    /// The relevance score before labels are cut from it.
    score: Vec<f64>,
    label: Vec<f32>,
    group: Vec<u32>,
    cols: usize,
}

fn uniform(stream: u32, row: u64) -> f64 {
    f64::from(quantize::uniform(SEED, stream, 0, row))
}

/// `generate` of `lambdarank_parity.py`, operation for operation.
fn generate(first_row: u64, rows: usize, cols: usize, queries: usize) -> Synthetic {
    let mut x = Vec::with_capacity(rows * cols);
    let mut score = Vec::with_capacity(rows);
    let mut label = Vec::with_capacity(rows);
    let mut values = vec![0.0f64; cols];
    for row in first_row..first_row + rows as u64 {
        for (j, value) in values.iter_mut().enumerate() {
            let u = uniform(j as u32, row);
            *value = if j == CATEGORICAL {
                (u * 8.0).floor()
            } else if j % 10 == 3 && uniform(NAN_STREAM + j as u32, row) < 0.1 {
                f64::NAN
            } else {
                u
            };
        }
        let x3 = if values[3].is_nan() { 0.5 } else { values[3] };
        let mut s = 3.0 * values[0];
        s += 2.0 * values[1] * values[2];
        s -= 1.5 * (values[4] - 0.5).abs();
        s += x3;
        s += CAT_EFFECT[values[CATEGORICAL] as usize];
        s += 0.6 * values[10] * values[11];
        s += uniform(NOISE_STREAM, row);
        score.push(s);
        label.push((s - 1.0).floor().clamp(0.0, 4.0) as f32);
        x.extend(values.iter().map(|&v| v as f32));
    }
    let per_query = (rows / queries) as u32;
    let mut group = vec![per_query; queries];
    group[queries - 1] += rows as u32 - per_query * queries as u32;
    Synthetic {
        x,
        score,
        label,
        group,
        cols,
    }
}

impl Synthetic {
    fn rows(&self) -> usize {
        self.label.len()
    }

    fn input<'a>(&'a self, label: &'a [f32], ranking: bool) -> DatasetInput<'a> {
        DatasetInput {
            features: DenseMatrix {
                data: DenseData::F32(&self.x),
                rows: self.rows(),
                cols: self.cols,
                layout: Layout::RowMajor,
            },
            label,
            weight: None,
            group: ranking.then_some(&self.group[..]),
            feature_names: None,
            categorical_features: vec![CATEGORICAL],
        }
    }

    /// Rows of whole queries `queries`, as one rank of a distributed run holds them.
    fn queries(&self, queries: std::ops::Range<usize>) -> Synthetic {
        let start: usize = self.group[..queries.start].iter().map(|&g| g as usize).sum();
        let len: usize = self.group[queries.clone()].iter().map(|&g| g as usize).sum();
        Synthetic {
            x: self.x[start * self.cols..(start + len) * self.cols].to_vec(),
            score: self.score[start..start + len].to_vec(),
            label: self.label[start..start + len].to_vec(),
            group: self.group[queries].to_vec(),
            cols: self.cols,
        }
    }
}

fn params(json: &str) -> Params {
    Params::from_json(json).expect("params")
}

/// Binned training and validation sets.
fn bin(
    params: &Params,
    train: &Synthetic,
    valid: &Synthetic,
    labels: [&[f32]; 2],
    ranking: bool,
) -> (BinnedDataset, BinnedDataset) {
    let train_set =
        BinnedDataset::construct(&train.input(labels[0], ranking), params, &mut LocalCollective).expect("train set");
    let valid_set =
        BinnedDataset::construct_with_reference(&valid.input(labels[1], ranking), &train_set, &mut LocalCollective)
            .expect("valid set");
    (train_set, valid_set)
}

/// Trains on one rank and returns the model and every iteration's evaluations.
fn fit(params: &Params, train_set: &BinnedDataset, valid_set: Option<&BinnedDataset>) -> (Model, Vec<Vec<EvalResult>>) {
    fit_on(params, train_set, valid_set, &mut LocalCollective)
}

fn fit_on(
    params: &Params,
    train_set: &BinnedDataset,
    valid_set: Option<&BinnedDataset>,
    collective: &mut dyn Collective,
) -> (Model, Vec<Vec<EvalResult>>) {
    let valid: Vec<(&str, &BinnedDataset)> = valid_set.map(|v| ("valid", v)).into_iter().collect();
    let valid_refs: Vec<&BinnedDataset> = valid.iter().map(|(_, v)| *v).collect();
    let mut backend = CpuBackend::borrowing(train_set, &valid_refs);
    let mut history = Vec::new();
    let model = train(params, train_set, &valid, &mut backend, collective, &mut |report| {
        history.push(report.evals.to_vec());
        Ok(())
    })
    .expect("train");
    (model, history)
}

fn metric(evals: &[EvalResult], name: &str) -> f64 {
    evals
        .iter()
        .find(|e| e.metric == name)
        .unwrap_or_else(|| panic!("no {name} in {evals:?}"))
        .value
}

/// First and last value of a metric over training.
fn first_last(history: &[Vec<EvalResult>], name: &str) -> (f64, f64) {
    let first = metric(history.first().expect("an iteration"), name);
    let last = metric(history.last().expect("an iteration"), name);
    (first, last)
}

const MODES: [(&str, bool); 2] = [("quantized", true), ("float", false)];

#[test]
fn regression_reaches_low_l2() {
    let train_rows = generate(0, 20_000, 12, 1);
    let valid_rows = generate(VALID_ROW_OFFSET, 5_000, 12, 1);
    let target = |d: &Synthetic| d.score.iter().map(|&s| s as f32).collect::<Vec<f32>>();
    let (train_y, valid_y) = (target(&train_rows), target(&valid_rows));
    let mean = valid_y.iter().map(|&y| f64::from(y)).sum::<f64>() / valid_y.len() as f64;
    let variance = valid_y.iter().map(|&y| (f64::from(y) - mean).powi(2)).sum::<f64>() / valid_y.len() as f64;
    for (mode, quantized) in MODES {
        let params = params(&format!(
            r#"{{"objective": "regression", "num_iterations": 150, "use_quantized_grad": {quantized}}}"#
        ));
        let (train_set, valid_set) = bin(&params, &train_rows, &valid_rows, [&train_y, &valid_y], false);
        let (model, history) = fit(&params, &train_set, Some(&valid_set));
        assert_eq!(model.num_trees(), 150, "{mode}: trees");
        let (_, l2) = first_last(&history, "l2");
        // The noise term alone has variance 1/12; the model should explain nearly everything else.
        assert!(
            l2 < 0.12 && l2 < 0.15 * variance,
            "{mode}: l2 {l2}, target variance {variance}"
        );
    }
}

#[test]
fn binary_logloss_improves() {
    let train_rows = generate(0, 20_000, 12, 1);
    let valid_rows = generate(VALID_ROW_OFFSET, 5_000, 12, 1);
    let target = |d: &Synthetic| {
        d.score
            .iter()
            .map(|&s| f32::from(u8::from(s > 3.0)))
            .collect::<Vec<f32>>()
    };
    let (train_y, valid_y) = (target(&train_rows), target(&valid_rows));
    for (mode, quantized) in MODES {
        let params = params(&format!(
            r#"{{"objective": "binary", "metric": "binary_logloss,auc", "num_iterations": 80,
                "use_quantized_grad": {quantized}}}"#
        ));
        let (train_set, valid_set) = bin(&params, &train_rows, &valid_rows, [&train_y, &valid_y], false);
        let (_, history) = fit(&params, &train_set, Some(&valid_set));
        let (first, last) = first_last(&history, "binary_logloss");
        assert!(last < 0.6 * first && last < 0.3, "{mode}: logloss {first} -> {last}");
        let (_, auc) = first_last(&history, "auc");
        assert!(auc > 0.95, "{mode}: auc {auc}");
    }
}

#[test]
fn lambdarank_ndcg_improves() {
    let train_rows = generate(0, 20_000, 20, 200);
    let valid_rows = generate(VALID_ROW_OFFSET, 5_000, 20, 50);
    for (mode, quantized) in MODES {
        let params = params(&format!(
            r#"{{"objective": "lambdarank", "eval_at": [10], "num_iterations": 50, "use_quantized_grad": {quantized}}}"#
        ));
        let (train_set, valid_set) = bin(
            &params,
            &train_rows,
            &valid_rows,
            [&train_rows.label, &valid_rows.label],
            true,
        );
        let (_, history) = fit(&params, &train_set, Some(&valid_set));
        let (first, last) = first_last(&history, "ndcg@10");
        assert!(last > first && last > 0.85, "{mode}: ndcg@10 {first} -> {last}");
    }
}

/// The parameters of the determinism and distributed runs: every source of randomness on.
const RANDOMIZED: &str = r#""objective": "lambdarank", "num_iterations": 30, "bagging_fraction": 0.7,
    "bagging_freq": 2, "feature_fraction": 0.8, "lambda_l1": 0.1, "lambda_l2": 1.0, "max_depth": 6"#;

#[test]
fn identical_runs_give_identical_models() {
    let train_rows = generate(0, 10_000, 20, 100);
    let valid_rows = generate(VALID_ROW_OFFSET, 2_000, 20, 20);
    for (mode, quantized) in MODES {
        let params = params(&format!(r#"{{{RANDOMIZED}, "use_quantized_grad": {quantized}}}"#));
        let (train_set, valid_set) = bin(
            &params,
            &train_rows,
            &valid_rows,
            [&train_rows.label, &valid_rows.label],
            true,
        );
        let (first, _) = fit(&params, &train_set, Some(&valid_set));
        let (second, _) = fit(&params, &train_set, Some(&valid_set));
        assert_eq!(first.to_lightgbm_string(), second.to_lightgbm_string(), "{mode}");
        assert!(first.num_trees() > 10, "{mode}: {} trees", first.num_trees());
    }
}

#[test]
fn two_ranks_train_the_model_of_one() {
    let rows = generate(0, 20_000, 20, 200);
    let params = params(&format!("{{{RANDOMIZED}}}"));
    let full = BinnedDataset::construct(&rows.input(&rows.label, true), &params, &mut LocalCollective).expect("bin");
    let (single, _) = fit(&params, &full, None);
    let expected = single.to_lightgbm_string();
    let shards = [rows.queries(0..90), rows.queries(90..200)];
    let models = run_ranks(2, |rank, collective| {
        let shard = &shards[rank];
        let data = BinnedDataset::construct(&shard.input(&shard.label, true), &params, collective).expect("bin");
        let (model, _) = fit_on(&params, &data, None, collective);
        model.to_lightgbm_string()
    });
    for (rank, model) in models.iter().enumerate() {
        assert!(model == &expected, "rank {rank} model differs from the one-rank model");
    }
}

#[test]
fn queries_without_relevant_items_train() {
    // 30% of queries have only label 0: lambdarank gives their rows g = h = 0.
    let mut rows = generate(0, 20_000, 20, 200);
    let mut start = 0;
    for (query, &size) in rows.group.iter().enumerate() {
        if query % 10 < 3 {
            rows.label[start..start + size as usize].fill(0.0);
        }
        start += size as usize;
    }
    for (mode, quantized) in MODES {
        let params = params(&format!(
            r#"{{"objective": "lambdarank", "num_iterations": 30, "min_data_in_leaf": 0,
                "min_sum_hessian_in_leaf": 0, "num_leaves": 63, "use_quantized_grad": {quantized}}}"#
        ));
        let data =
            BinnedDataset::construct(&rows.input(&rows.label, true), &params, &mut LocalCollective).expect("bin");
        let (model, _) = fit(&params, &data, None);
        assert_eq!(model.num_trees(), 30, "{mode}: trees");
        for (index, tree) in model.trees.iter().enumerate() {
            assert!(
                tree.leaf_count.iter().all(|&c| c > 0),
                "{mode}: tree {index} has an empty leaf"
            );
        }
    }
}

#[test]
fn lambdarank_ndcg_matches_lightgbm() {
    let text = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/lambdarank_parity.json"
    ))
    .expect("fixture");
    let fixture: serde_json::Value = serde_json::from_str(&text).expect("fixture json");
    let size = |key: &str| fixture[key].as_u64().unwrap_or_else(|| panic!("fixture {key}")) as usize;
    let train_rows = generate(0, size("rows"), size("features"), size("queries"));
    let valid_rows = generate(
        VALID_ROW_OFFSET,
        size("valid_rows"),
        size("features"),
        size("valid_queries"),
    );
    for (mode, quantized) in MODES {
        let params = params(&format!(
            r#"{{"objective": "lambdarank", "metric": "ndcg", "eval_at": [10], "learning_rate": 0.1,
                "num_leaves": 31, "min_data_in_leaf": 20, "num_iterations": {}, "max_bin": 255,
                "use_quantized_grad": {quantized}, "num_grad_quant_bins": 4, "quant_train_renew_leaf": true,
                "stochastic_rounding": true}}"#,
            size("iterations")
        ));
        let (train_set, valid_set) = bin(
            &params,
            &train_rows,
            &valid_rows,
            [&train_rows.label, &valid_rows.label],
            true,
        );
        let (_, history) = fit(&params, &train_set, Some(&valid_set));
        let (_, ndcg) = first_last(&history, "ndcg@10");
        let lightgbm = fixture["ndcg@10"][mode].as_f64().expect("fixture ndcg");
        println!("{mode}: silvic ndcg@10 {ndcg:.5}, lightgbm {lightgbm:.5}");
        assert!(
            (ndcg - lightgbm).abs() <= 0.01,
            "{mode}: silvic ndcg@10 {ndcg}, lightgbm {lightgbm}"
        );
    }
}

/// Training speed on a larger copy of the parity data; compare with
/// `lambdarank_parity.py --bench --rows <rows> --features <features> --queries <queries>`.
/// `SILVIC_BENCH_ROWS=2000000 SILVIC_BENCH_FEATURES=140 cargo test --release -p silvic-core --test train_cpu
/// -- --ignored --nocapture cpu_training_speed`
#[test]
#[ignore = "benchmark; run in release"]
fn cpu_training_speed() {
    let env = |name: &str, default: usize| std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default);
    let rows = env("SILVIC_BENCH_ROWS", 1_000_000);
    let features = env("SILVIC_BENCH_FEATURES", 140);
    let queries = env("SILVIC_BENCH_QUERIES", rows / 100);
    let iterations = env("SILVIC_BENCH_ITERATIONS", 100);
    let threads = env("SILVIC_BENCH_THREADS", 14);
    let train_rows = generate(0, rows, features, queries);
    let valid_rows = generate(VALID_ROW_OFFSET, 20_000, features, 200);
    for (mode, quantized) in MODES {
        let params = params(&format!(
            r#"{{"objective": "lambdarank", "metric": "ndcg", "eval_at": [10], "num_leaves": 31,
                "num_iterations": {iterations}, "num_threads": {threads}, "use_quantized_grad": {quantized}}}"#
        ));
        let start = Instant::now();
        let (train_set, valid_set) = bin(
            &params,
            &train_rows,
            &valid_rows,
            [&train_rows.label, &valid_rows.label],
            true,
        );
        let binned = start.elapsed();
        let mut backend = Timed {
            inner: CpuBackend::borrowing(&train_set, &[&valid_set]),
            times: BTreeMap::new(),
        };
        let mut last = Vec::new();
        let start = Instant::now();
        train(
            &params,
            &train_set,
            &[("valid", &valid_set)],
            &mut backend,
            &mut LocalCollective,
            &mut |report| {
                last = report.evals.to_vec();
                Ok(())
            },
        )
        .expect("train");
        let trained = start.elapsed().as_secs_f64();
        let ndcg = metric(&last, "ndcg@10");
        println!(
            "silvic {mode}: {rows} rows x {features} features, {iterations} iterations: bin {:.2}s, train {trained:.2}s, \
             valid ndcg@10 {ndcg:.5}",
            binned.as_secs_f64(),
        );
        let device: f64 = backend.times.values().map(Duration::as_secs_f64).sum();
        for (op, time) in &backend.times {
            println!("  {op:<18} {:>7.2}s", time.as_secs_f64());
        }
        println!("  {:<18} {:>7.2}s", "host (splits, metrics)", trained - device);
    }
}

/// A backend that times each operation of the one it wraps.
struct Timed<B> {
    inner: B,
    times: BTreeMap<&'static str, Duration>,
}

impl<B: Backend> Timed<B> {
    fn time<R>(&mut self, op: &'static str, f: impl FnOnce(&mut B) -> R) -> R {
        let start = Instant::now();
        let result = f(&mut self.inner);
        *self.times.entry(op).or_default() += start.elapsed();
        result
    }
}

impl<B: Backend> Backend for Timed<B> {
    fn name(&self) -> String {
        self.inner.name()
    }

    fn set_thread_pool(&mut self, pool: std::sync::Arc<rayon::ThreadPool>) {
        self.inner.set_thread_pool(pool);
    }

    fn load(&mut self, train: &BinnedDataset, valid: &[&BinnedDataset], objective: &ObjectiveSetup) -> Result<()> {
        self.time("load", |b| b.load(train, valid, objective))
    }

    fn init_scores(&mut self, dataset: usize, value: f64) -> Result<()> {
        self.time("init_scores", |b| b.init_scores(dataset, value))
    }

    fn compute_gradients(&mut self) -> Result<GradientStats> {
        self.time("compute_gradients", |b| b.compute_gradients())
    }

    fn quantize(&mut self, spec: &QuantSpec) -> Result<()> {
        self.time("quantize", |b| b.quantize(spec))
    }

    fn reset_partition(&mut self, bag: &BagSpec) -> Result<u32> {
        self.time("reset_partition", |b| b.reset_partition(bag))
    }

    fn histogram(&mut self, leaf: LeafRange, quantized: bool, out: &mut [f64]) -> Result<()> {
        self.time("histogram", |b| b.histogram(leaf, quantized, out))
    }

    fn partition(&mut self, leaf: LeafRange, split: &BinSplit) -> Result<u32> {
        self.time("partition", |b| b.partition(leaf, split))
    }

    fn leaf_sums(&mut self, leaves: &[LeafRange], grad_scale: f32, hess_scale: f32) -> Result<Vec<[i64; 2]>> {
        self.time("leaf_sums", |b| b.leaf_sums(leaves, grad_scale, hess_scale))
    }

    fn add_tree(&mut self, tree: &DeviceTree) -> Result<()> {
        self.time("add_tree", |b| b.add_tree(tree))
    }

    fn scores(&mut self, dataset: usize) -> Result<Vec<f32>> {
        self.time("scores", |b| b.scores(dataset))
    }
}
