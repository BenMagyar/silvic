//! Compares against `silvic_core::Model::predict` (LightGBM 3.3.5's decisions) on regression, binary and lambdarank
//! models silvic trains on rows with NaN, zeros and categorical features (one-hot and many-vs-many splits, bitsets of
//! several words), scored on held-out rows.

mod common;

use silvic_core::data::{DatasetInput, DenseData, DenseMatrix, Layout};
use silvic_core::{train, BinnedDataset, CpuBackend, LocalCollective, Model, Params};
use silvic_serve::{Kernel, Predictor};

use common::Rng;

const FEATURES: usize = 10;
/// Feature 7 has 6 categories (one-hot splits), feature 8 has 45 (many-vs-many, two-word bitsets).
const CATEGORICAL: [(usize, usize); 2] = [(7, 6), (8, 45)];

struct Rows {
    x: Vec<f32>,
    label: Vec<f32>,
    group: Vec<u32>,
}

fn rows(rng: &mut Rng, count: usize, ranking: bool) -> Rows {
    let mut x = Vec::with_capacity(count * FEATURES);
    let mut label = Vec::with_capacity(count);
    for _ in 0..count {
        let mut values = [0.0f64; FEATURES];
        for (feature, value) in values.iter_mut().enumerate() {
            *value = match CATEGORICAL.iter().find(|(f, _)| *f == feature) {
                Some(&(_, categories)) => rng.below(categories) as f64,
                None if feature % 3 == 0 && rng.chance(0.1) => f64::NAN,
                None if feature % 4 == 1 && rng.chance(0.3) => 0.0,
                None => rng.normal(),
            };
        }
        let x0 = if values[0].is_nan() { 0.3 } else { values[0] };
        let score = 2.0 * x0 + values[1] * values[2] - values[4].abs()
            + [0.0, 1.0, -1.0, 0.5, 0.0, -0.5][values[7] as usize]
            + if (values[8] as usize).is_multiple_of(3) {
                0.8
            } else {
                -0.2
            }
            + 0.3 * rng.normal();
        label.push(if ranking {
            (score + 1.0).clamp(0.0, 4.0).floor() as f32
        } else {
            score as f32
        });
        x.extend(values.iter().map(|&v| v as f32));
    }
    let group = if ranking { vec![20; count / 20] } else { Vec::new() };
    Rows { x, label, group }
}

fn fit(params: &str, data: &Rows, binary: bool, categorical: bool) -> Model {
    let params = Params::from_json(params).expect("params");
    let label: Vec<f32> = if binary {
        data.label.iter().map(|&y| f32::from(y > 0.5)).collect()
    } else {
        data.label.clone()
    };
    let input = DatasetInput {
        features: DenseMatrix {
            data: DenseData::F32(&data.x),
            rows: data.label.len(),
            cols: FEATURES,
            layout: Layout::RowMajor,
        },
        label: &label,
        weight: None,
        group: (!data.group.is_empty()).then_some(&data.group[..]),
        feature_names: None,
        categorical_features: if categorical {
            CATEGORICAL.iter().map(|&(f, _)| f).collect()
        } else {
            Vec::new()
        },
    };
    let dataset = BinnedDataset::construct(&input, &params, &mut LocalCollective).expect("bin");
    let mut backend = CpuBackend::borrowing(&dataset, &[]);
    train(&params, &dataset, &[], &mut backend, &mut LocalCollective, &mut |_| {
        Ok(())
    })
    .expect("train")
}

/// The model splits on categories with a multi-word bitset, scores without walking, and matches.
fn assert_categorical_model_matches(case: &str, model: &Model, data: &Rows) {
    let widest_bitset = model
        .trees
        .iter()
        .flat_map(|t| t.cat_boundaries.windows(2).map(|w| w[1] - w[0]))
        .max();
    assert!(
        widest_bitset >= Some(2),
        "{case}: some categorical bitset spans two words"
    );
    assert_eq!(
        Predictor::new(model).expect("compile").num_walked_trees(),
        0,
        "{case}: no tree is walked"
    );
    assert_matches(case, model, data);
}

fn assert_matches(case: &str, model: &Model, data: &Rows) {
    let matrix = DenseMatrix {
        data: DenseData::F32(&data.x),
        rows: data.label.len(),
        cols: FEATURES,
        layout: Layout::RowMajor,
    };
    for kernel in Kernel::available() {
        let predictor = Predictor::with_kernel(model, kernel).expect("compile");
        for (raw_score, num_trees) in [(false, None), (true, None), (true, Some(7))] {
            let expected = model.predict(&matrix, raw_score, num_trees);
            let got = predictor.predict(&matrix, raw_score, num_trees, 1);
            for (row, (g, e)) in got.iter().zip(&expected).enumerate() {
                assert_eq!(
                    g.to_bits(),
                    e.to_bits(),
                    "{case} {kernel} raw {raw_score} trees {num_trees:?}: row {row}"
                );
            }
        }
    }
}

#[test]
fn trained_models_match_model_predict() {
    let mut rng = Rng::new(11);
    let train_rows = rows(&mut rng, 4000, false);
    let test_rows = rows(&mut rng, 1500, false);
    let common = r#""num_iterations": 40, "num_leaves": 53, "max_depth": 11, "min_data_in_leaf": 5,
        "max_cat_to_onehot": 8, "cat_smooth": 1, "min_data_per_group": 5, "verbosity": -1, "num_threads": 2"#;
    let regression = fit(
        &format!(r#"{{"objective": "regression", {common}}}"#),
        &train_rows,
        false,
        true,
    );
    assert_categorical_model_matches("regression", &regression, &test_rows);
    let binary = fit(
        &format!(r#"{{"objective": "binary", {common}}}"#),
        &train_rows,
        true,
        true,
    );
    assert_categorical_model_matches("binary", &binary, &test_rows);

    let train_ranked = rows(&mut rng, 4000, true);
    let test_ranked = rows(&mut rng, 1000, true);
    let lambdarank = fit(
        &format!(r#"{{"objective": "lambdarank", {common}}}"#),
        &train_ranked,
        false,
        true,
    );
    assert_categorical_model_matches("lambdarank", &lambdarank, &test_ranked);
}

#[test]
fn trees_past_64_leaves_from_training_match() {
    let mut rng = Rng::new(12);
    let train_rows = rows(&mut rng, 6000, false);
    let test_rows = rows(&mut rng, 800, false);
    for (leaves, walked) in [(120, false), (300, true)] {
        let params = format!(
            r#"{{"objective": "regression", "num_iterations": 8, "num_leaves": {leaves}, "min_data_in_leaf": 3,
            "verbosity": -1, "num_threads": 2}}"#
        );
        let model = fit(&params, &train_rows, false, false);
        assert!(
            model.trees.iter().any(|t| t.num_leaves > 64),
            "{leaves}: trees past 64 leaves"
        );
        let predictor = Predictor::new(&model).expect("compile");
        assert_eq!(predictor.num_walked_trees() > 0, walked, "{leaves}: walked trees");
        assert_matches(&format!("{leaves} leaves"), &model, &test_rows);
    }
}
