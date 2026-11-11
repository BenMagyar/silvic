//! Single-thread time to score 3000 rows with a random 2000-tree, 53-leaf forest shaped like a learning-to-rank
//! model (136 features, 5 categoricals of 20 categories, NaN and edge values): `Model::predict_raw_row` row by row
//! (the reference walk), and `Predictor::predict_raw` with every kernel this CPU runs.

#[path = "../tests/common/mod.rs"]
mod common;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use silvic_core::data::{DenseData, DenseMatrix, Layout};
use silvic_core::model::ModelObjective;
use silvic_serve::{Kernel, Predictor};

use common::{random_model, random_rows, ForestSpec, Rng};

const ROWS: usize = 3000;

fn spec() -> ForestSpec {
    ForestSpec {
        trees: 2000,
        leaves: 53,
        max_depth: 11,
        features: 136,
        categorical: 5,
        categories: 20,
        category_stride: 1,
        thresholds: 200,
        edge_thresholds: 0.01,
        objective: ModelObjective::LambdaRank,
    }
}

fn bench(c: &mut Criterion) {
    let spec = spec();
    let mut rng = Rng::new(7);
    let (model, grid) = random_model(&mut rng, &spec);
    let rows: Vec<f32> = random_rows(&mut rng, &spec, &grid, ROWS)
        .iter()
        .map(|&v| v as f32)
        .collect();
    let matrix = DenseMatrix {
        data: DenseData::F32(&rows),
        rows: ROWS,
        cols: spec.features,
        layout: Layout::RowMajor,
    };
    let mut group = c.benchmark_group("score_3000_rows");
    group.throughput(Throughput::Elements(ROWS as u64));
    group.sample_size(20);
    let row_values: Vec<Vec<f64>> = rows
        .chunks(spec.features)
        .map(|r| r.iter().map(|&v| f64::from(v)).collect())
        .collect();
    group.bench_function("reference_walk", |b| {
        b.iter(|| {
            row_values
                .iter()
                .map(|row| model.predict_raw_row(row, None))
                .sum::<f64>()
        })
    });
    let mut out = vec![0.0; ROWS];
    for kernel in Kernel::available() {
        let predictor = Predictor::with_kernel(&model, kernel).expect("compile");
        group.bench_with_input(BenchmarkId::new("predictor", kernel), &matrix, |b, matrix| {
            b.iter(|| predictor.predict_raw(matrix, None, &mut out))
        });
    }
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
