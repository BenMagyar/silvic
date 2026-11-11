#![cfg(feature = "cuda")]

mod common;

use common::{model, random_model, random_rows, ForestSpec, Rng, ZERO_THRESHOLD};
use silvic_core::data::{DenseData, DenseMatrix, Layout};
use silvic_core::model::{Model, ModelObjective};
use silvic_core::tree::{MissingType, SplitChildren, Tree};
use silvic_serve::{cuda_is_available, CudaPredictor, CudaPredictorF32};

fn available() -> bool {
    let available = cuda_is_available();
    if !available {
        eprintln!("skipping CUDA parity test: driver/NVRTC/device unavailable");
    }
    available
}

fn assert_bits(got: &[f64], expected: &[f64]) {
    assert_eq!(got.len(), expected.len());
    for (row, (got, expected)) in got.iter().zip(expected).enumerate() {
        assert_eq!(got.to_bits(), expected.to_bits(), "row {row}: {got:?} != {expected:?}");
    }
}

fn parity(predictor: &mut CudaPredictor, model: &Model, matrix: &DenseMatrix<'_>, prefixes: &[Option<usize>]) {
    for &prefix in prefixes {
        for raw in [true, false] {
            let got = predictor.predict(matrix, raw, prefix).expect("CUDA prediction");
            assert_bits(&got, &model.predict(matrix, raw, prefix));
        }
        let mut out = vec![f64::NAN; matrix.rows];
        predictor
            .predict_raw(matrix, prefix, &mut out)
            .expect("CUDA raw prediction");
        assert_bits(&out, &model.predict(matrix, true, prefix));
    }
}

fn all_layouts(predictor: &mut CudaPredictor, model: &Model, values: &[f64], rows: usize, prefixes: &[Option<usize>]) {
    let cols = values.len().checked_div(rows).unwrap_or_else(|| model.num_features());
    for layout in [Layout::RowMajor, Layout::ColumnMajor] {
        let ordered: Vec<f64> = match layout {
            Layout::RowMajor => values.to_vec(),
            Layout::ColumnMajor => (0..cols)
                .flat_map(|col| (0..rows).map(move |row| values[row * cols + col]))
                .collect(),
        };
        let f32_values: Vec<f32> = ordered.iter().map(|&value| value as f32).collect();
        for data in [DenseData::F64(&ordered), DenseData::F32(&f32_values)] {
            parity(
                predictor,
                model,
                &DenseMatrix {
                    data,
                    rows,
                    cols,
                    layout,
                },
                prefixes,
            );
        }
    }
}

fn children() -> SplitChildren {
    SplitChildren {
        left_value: -0.12345678901234567,
        right_value: 0.7654321098765432,
        left_count: 1,
        right_count: 1,
        left_weight: 1.0,
        right_weight: 1.0,
        gain: 1.0,
    }
}

#[test]
fn every_decision_boundary_matches_core() {
    if !available() {
        return;
    }
    let thresholds = [
        -ZERO_THRESHOLD,
        -1e-300,
        -0.0,
        ZERO_THRESHOLD,
        0.5,
        f64::NEG_INFINITY,
        f64::INFINITY,
        f64::NAN,
    ];
    let mut trees = Vec::new();
    for missing in [MissingType::None, MissingType::Zero, MissingType::NaN] {
        for default_left in [false, true] {
            for threshold in thresholds {
                let mut tree = Tree::new();
                tree.split_numerical(0, 0, threshold, default_left, missing, &children());
                trees.push(tree);
            }
        }
        let mut tree = Tree::new();
        tree.split_categorical(0, 0, &[0, 2, 31, 32, 40, 127], missing, &children());
        trees.push(tree);
    }
    // Missing-type bits 3 map to None, and a categorical flag without num_cat is numerical.
    let mut unusual = Tree::new();
    unusual.split_numerical(0, 0, 0.5, true, MissingType::None, &children());
    unusual.decision_type[0] = 15;
    trees.push(unusual);
    let model = model(trees, 1, ModelObjective::Binary { sigmoid: 1.7 });
    let mut predictor = CudaPredictor::new(&model, None).unwrap();
    assert_eq!(predictor.num_trees(), model.trees.len());
    assert_eq!(predictor.num_features(), 1);
    let mut values = vec![
        f64::NAN,
        f64::NEG_INFINITY,
        f64::INFINITY,
        f64::MIN,
        f64::MAX,
        -2_147_483_649.0,
        -1.0,
        -0.999,
        -0.5,
        -0.0,
        0.0,
        1e-300,
        1.0,
        2.9,
        31.0,
        32.0,
        40.0,
        127.0,
        128.0,
        2_147_483_647.0,
        2_147_483_648.0,
    ];
    for threshold in thresholds {
        values.extend([threshold.next_down(), threshold, threshold.next_up()]);
    }
    let mut prefixes: Vec<Option<usize>> = (0..=model.trees.len()).map(Some).collect();
    prefixes.extend([None, Some(usize::MAX)]);
    // An unused extra feature also exercises the matrix stride rather than the model width.
    let values: Vec<f64> = values.into_iter().flat_map(|value| [value, 999.0]).collect();
    all_layouts(&mut predictor, &model, &values, values.len() / 2, &prefixes);
}

#[test]
fn large_forests_layouts_and_changing_batches_match_core() {
    if !available() {
        return;
    }
    let spec = ForestSpec {
        trees: 9,
        leaves: 513,
        max_depth: 30,
        features: 6,
        categorical: 2,
        categories: 140,
        category_stride: 3,
        objective: ModelObjective::LambdaRank,
        ..ForestSpec::default()
    };
    let mut rng = Rng::new(0x4355_4441);
    let (model, grid) = random_model(&mut rng, &spec);
    assert!(model.trees.iter().all(|tree| tree.num_leaves > 256));
    let mut predictor = CudaPredictor::new(&model, Some(0)).unwrap();
    for rows in [0, 1, 31, 32, 33, 128, 1025, 17, 4097, 2] {
        let values = random_rows(&mut rng, &spec, &grid, rows);
        all_layouts(
            &mut predictor,
            &model,
            &values,
            rows,
            &[Some(0), Some(3), None, Some(100)],
        );
    }
}

#[test]
fn empty_constant_and_order_sensitive_forests_match_core() {
    if !available() {
        return;
    }
    let mut tiled = vec![0.0; 18];
    tiled[7] = 1e16;
    tiled[8] = -1e16;
    tiled[9] = 1.0;
    tiled[15] = 1e16;
    tiled[16] = -1e16;
    tiled[17] = 3.0;
    for values in [
        vec![],
        vec![0.0],
        vec![1e16, 1.0, -1e16, f64::from_bits(1), -0.0],
        tiled,
    ] {
        let trees = values
            .into_iter()
            .map(|value| {
                let mut tree = Tree::new();
                tree.leaf_value[0] = value;
                tree
            })
            .collect();
        let model = model(trees, 0, ModelObjective::Binary { sigmoid: 2.0 });
        let mut predictor = CudaPredictor::new(&model, None).unwrap();
        let prefixes: Vec<_> = (0..=model.trees.len()).map(Some).chain([None]).collect();
        for rows in [0, 1, 31, 32, 33, 257] {
            for layout in [Layout::RowMajor, Layout::ColumnMajor] {
                for data in [DenseData::F32(&[]), DenseData::F64(&[])] {
                    parity(
                        &mut predictor,
                        &model,
                        &DenseMatrix {
                            data,
                            rows,
                            cols: 0,
                            layout,
                        },
                        &prefixes,
                    );
                }
            }
        }
    }
}

#[test]
fn grid_stride_tail_rows_are_scored() {
    if !available() {
        return;
    }
    let mut tree = Tree::new();
    tree.leaf_value[0] = 1.25;
    let model = model(vec![tree], 0, ModelObjective::Regression);
    let mut predictor = CudaPredictor::new(&model, None).unwrap();
    let rows = DenseMatrix {
        data: DenseData::F32(&[]),
        rows: 65_535 * 32 + 1,
        cols: 0,
        layout: Layout::RowMajor,
    };
    let scores = predictor.predict(&rows, true, None).unwrap();
    assert!(scores.iter().all(|score| score.to_bits() == 1.25f64.to_bits()));
}

#[test]
fn malformed_models_return_structural_errors_before_opening_cuda() {
    let mut tree = Tree::new();
    tree.split_numerical(0, 0, 0.5, true, MissingType::None, &children());
    let mut cases = Vec::new();
    let mut cycle = tree.clone();
    cycle.left_child[0] = 0;
    cases.push(cycle);
    let mut child = tree.clone();
    child.right_child[0] = -9;
    cases.push(child);
    let mut feature = tree.clone();
    feature.split_feature[0] = 3;
    cases.push(feature);
    let mut short = tree;
    short.threshold.clear();
    cases.push(short);
    let mut categorical = Tree::new();
    categorical.split_categorical(0, 0, &[1], MissingType::None, &children());
    categorical.cat_boundaries[1] = 9;
    cases.push(categorical);
    let mut empty = Tree::new();
    empty.leaf_value.clear();
    cases.push(empty);
    for tree in cases {
        let model = model(vec![tree], 1, ModelObjective::Regression);
        let error = match CudaPredictor::new(&model, None) {
            Ok(_) => panic!("malformed model accepted"),
            Err(error) => error,
        };
        assert_eq!(error.kind(), "model");
    }
}

#[test]
fn missing_device_and_bad_inputs_are_errors() {
    let model = model(vec![Tree::new()], 1, ModelObjective::Regression);
    assert!(CudaPredictor::new(&model, Some(usize::MAX)).is_err());
    if !available() {
        assert!(CudaPredictor::new(&model, None).is_err());
        return;
    }
    let mut predictor = CudaPredictor::new(&model, None).unwrap();
    for (rows, cols, values) in [(1, 0, &[][..]), (2, 1, &[0.0][..]), (usize::MAX, 2, &[][..])] {
        let matrix = DenseMatrix {
            data: DenseData::F64(values),
            rows,
            cols,
            layout: Layout::RowMajor,
        };
        assert!(predictor.predict(&matrix, true, None).is_err());
    }
    let matrix = DenseMatrix {
        data: DenseData::F64(&[0.0]),
        rows: 1,
        cols: 1,
        layout: Layout::RowMajor,
    };
    assert!(predictor.predict_raw(&matrix, None, &mut []).is_err());
    assert_bits(&predictor.predict(&matrix, true, None).unwrap(), &[0.0]);
}

#[test]
fn deep_trees_validate_without_recursion() {
    let mut tree = Tree::new();
    for _ in 0..20_000 {
        tree.split_numerical(0, 0, 0.5, true, MissingType::None, &children());
    }
    // A deep cycle must be rejected even on a host without CUDA, without exhausting the Rust stack.
    let last = tree.left_child.len() - 1;
    tree.left_child[last] = 0;
    let model = model(vec![tree], 1, ModelObjective::Regression);
    let error = match CudaPredictor::new(&model, None) {
        Ok(_) => panic!("deep cyclic model accepted"),
        Err(error) => error,
    };
    assert_eq!(error.kind(), "model");
}

fn assert_f32_bits(got: &[f32], expected: &[f32]) {
    assert_eq!(got.len(), expected.len());
    for (row, (got, expected)) in got.iter().zip(expected).enumerate() {
        assert_eq!(got.to_bits(), expected.to_bits(), "row {row}: {got:?} != {expected:?}");
    }
}

fn expected_f32(model: &Model, matrix: &DenseMatrix<'_>, prefix: Option<usize>, raw: bool) -> Vec<f32> {
    (0..matrix.rows)
        .map(|index| {
            let row: Vec<f64> = (0..matrix.cols)
                .map(|col| f64::from(matrix.get(index, col) as f32))
                .collect();
            let mut score = model.trees[..prefix.unwrap_or(model.trees.len()).min(model.trees.len())]
                .iter()
                .fold(0.0f32, |score, tree| {
                    score + tree.leaf_value[tree.leaf_index(&row)] as f32
                });
            if !raw {
                if let ModelObjective::Binary { sigmoid } = model.objective {
                    score = 1.0f32 / (1.0f32 + (-(sigmoid as f32) * score).exp());
                }
            }
            score
        })
        .collect()
}

fn all_f32_layouts(
    predictor: &mut CudaPredictorF32,
    model: &Model,
    values: &[f64],
    rows: usize,
    prefixes: &[Option<usize>],
) {
    let cols = values.len().checked_div(rows).unwrap_or_else(|| model.num_features());
    for layout in [Layout::RowMajor, Layout::ColumnMajor] {
        let ordered: Vec<f64> = match layout {
            Layout::RowMajor => values.to_vec(),
            Layout::ColumnMajor => (0..cols)
                .flat_map(|col| (0..rows).map(move |row| values[row * cols + col]))
                .collect(),
        };
        let f32_values: Vec<f32> = ordered.iter().map(|&value| value as f32).collect();
        for data in [DenseData::F32(&f32_values), DenseData::F64(&ordered)] {
            let matrix = DenseMatrix {
                data,
                rows,
                cols,
                layout,
            };
            for &prefix in prefixes {
                for raw in [true, false] {
                    assert_f32_bits(
                        &predictor.predict(&matrix, raw, prefix).unwrap(),
                        &expected_f32(model, &matrix, prefix, raw),
                    );
                }
                let mut out = vec![f32::NAN; rows];
                predictor.predict_raw(&matrix, prefix, &mut out).unwrap();
                assert_f32_bits(&out, &expected_f32(model, &matrix, prefix, true));
            }
        }
    }
}

#[test]
fn f32_rounds_leaves_and_accumulates_in_order_instead_of_casting_f64_scores() {
    if !available() {
        return;
    }
    let mut tiled = vec![0.0; 18];
    tiled[7] = 16_777_216.0;
    tiled[8] = 1.0;
    tiled[9] = -16_777_216.0;
    tiled[17] = 3.0;
    for leaves in [
        vec![],
        vec![-0.0],
        vec![f64::from(f32::from_bits(1))],
        vec![f64::MAX],
        vec![f64::MIN],
        vec![16_777_216.0, 1.0, -16_777_216.0, 0.1000000001],
        tiled,
    ] {
        let trees = leaves
            .iter()
            .map(|&value| {
                let mut tree = Tree::new();
                tree.leaf_value[0] = value;
                tree
            })
            .collect();
        let model = model(trees, 0, ModelObjective::Regression);
        let mut predictor = CudaPredictorF32::new(&model, None).unwrap();
        assert_eq!(predictor.num_trees(), leaves.len());
        assert_eq!(predictor.num_features(), 0);
        let prefixes: Vec<_> = (0..=leaves.len()).map(Some).chain([None, Some(usize::MAX)]).collect();
        for rows in [0, 1, 31, 32, 33, 257, 2] {
            all_f32_layouts(&mut predictor, &model, &[], rows, &prefixes);
        }
        if leaves.len() == 4 {
            let matrix = DenseMatrix {
                data: DenseData::F32(&[]),
                rows: 1,
                cols: 0,
                layout: Layout::RowMajor,
            };
            let score = predictor.predict(&matrix, true, None).unwrap()[0];
            assert_eq!(score.to_bits(), 0.1f32.to_bits());
            assert_ne!(score.to_bits(), (model.predict_raw_row(&[], None) as f32).to_bits());
            assert_eq!(predictor.predict(&matrix, true, Some(3)).unwrap(), [0.0f32]);
        }
    }
}

#[test]
fn f32_thresholds_preserve_adjacent_f32_decisions_and_convert_f64_inputs_on_device() {
    if !available() {
        return;
    }
    for lower in [-100.0f32, -1.0, 1.0, 100.0] {
        let upper = lower.next_up();
        for fraction in [0.25, 0.75] {
            let threshold = f64::from(lower) + (f64::from(upper) - f64::from(lower)) * fraction;
            let mut tree = Tree::new();
            tree.split_numerical(0, 0, threshold, false, MissingType::None, &children());
            let model = model(vec![tree], 1, ModelObjective::Regression);
            let mut predictor = CudaPredictorF32::new(&model, None).unwrap();
            all_f32_layouts(
                &mut predictor,
                &model,
                &[f64::from(lower), f64::from(upper), threshold],
                3,
                &[Some(0), None],
            );
            let values = [lower, upper];
            let matrix = DenseMatrix {
                data: DenseData::F32(&values),
                rows: 2,
                cols: 1,
                layout: Layout::RowMajor,
            };
            assert_f32_bits(
                &predictor.predict(&matrix, true, None).unwrap(),
                &[children().left_value as f32, children().right_value as f32],
            );
        }
    }
}

#[test]
fn f32_missing_categories_and_extreme_thresholds_follow_quantized_inputs() {
    if !available() {
        return;
    }
    let thresholds = [
        f64::NEG_INFINITY,
        f64::MIN,
        -f64::from(f32::MAX),
        -ZERO_THRESHOLD,
        -1e-300,
        -0.0,
        0.0,
        1e-300,
        ZERO_THRESHOLD,
        f64::from(f32::MAX),
        f64::MAX,
        f64::INFINITY,
        f64::NAN,
    ];
    let mut trees = Vec::new();
    for missing in [MissingType::None, MissingType::Zero, MissingType::NaN] {
        for default_left in [false, true] {
            for threshold in thresholds {
                let mut tree = Tree::new();
                tree.split_numerical(0, 0, threshold, default_left, missing, &children());
                trees.push(tree);
            }
        }
        let mut tree = Tree::new();
        tree.split_categorical(0, 0, &[0, 2, 31, 32, 40, 127], missing, &children());
        tree.split_categorical(0, 1, &[1, 33, 255], missing, &children());
        trees.push(tree);
    }
    let model = model(trees, 2, ModelObjective::Binary { sigmoid: 1.7000000001 });
    let mut predictor = CudaPredictorF32::new(&model, None).unwrap();
    let values = [
        f64::NAN,
        f64::NEG_INFINITY,
        f64::INFINITY,
        f64::MIN,
        f64::MAX,
        -1.0,
        -0.99999999,
        -0.5,
        -0.0,
        0.0,
        1e-300,
        -f64::from(f32::from_bits(1)),
        f64::from(f32::from_bits(1)),
        -ZERO_THRESHOLD,
        ZERO_THRESHOLD,
        f64::from((ZERO_THRESHOLD as f32).next_up()),
        1.0,
        2.9,
        31.0,
        32.0,
        40.0,
        127.0,
        128.0,
        255.0,
        2_147_483_520.0,
        2_147_483_647.0,
        2_147_483_648.0,
    ];
    let rows: Vec<f64> = values.into_iter().flat_map(|value| [value, 33.0, 999.0]).collect();
    let prefixes: Vec<_> = (0..=model.trees.len())
        .map(Some)
        .chain([None, Some(usize::MAX)])
        .collect();
    all_f32_layouts(&mut predictor, &model, &rows, values.len(), &prefixes);
}

#[test]
fn f32_large_forests_cover_layouts_prefixes_tiles_and_reused_batch_buffers() {
    if !available() {
        return;
    }
    let spec = ForestSpec {
        trees: 17,
        leaves: 513,
        max_depth: 30,
        features: 6,
        categorical: 2,
        categories: 140,
        category_stride: 3,
        objective: ModelObjective::LambdaRank,
        ..ForestSpec::default()
    };
    let mut rng = Rng::new(0x0046_3332_4355_4441);
    let (model, grid) = random_model(&mut rng, &spec);
    let mut predictor = CudaPredictorF32::new(&model, Some(0)).unwrap();
    for rows in [0, 1, 31, 32, 33, 128, 1025, 17, 4097, 2] {
        let values = random_rows(&mut rng, &spec, &grid, rows);
        all_f32_layouts(
            &mut predictor,
            &model,
            &values,
            rows,
            &[Some(0), Some(3), Some(8), Some(9), None, Some(usize::MAX)],
        );
    }
}

#[test]
fn f32_invalid_models_devices_and_matrices_return_errors_without_cpu_fallback() {
    let mut invalid = Tree::new();
    invalid.split_numerical(0, 0, 0.5, true, MissingType::None, &children());
    invalid.left_child[0] = 0;
    let error = match CudaPredictorF32::new(&model(vec![invalid], 1, ModelObjective::Regression), None) {
        Ok(_) => panic!("cyclic model accepted"),
        Err(error) => error,
    };
    assert_eq!(error.kind(), "model");
    let model = model(vec![Tree::new()], 1, ModelObjective::Regression);
    assert!(CudaPredictorF32::new(&model, Some(usize::MAX)).is_err());
    if !available() {
        assert!(CudaPredictorF32::new(&model, None).is_err());
        return;
    }
    let mut predictor = CudaPredictorF32::new(&model, None).unwrap();
    for (rows, cols, values) in [(1, 0, &[][..]), (2, 1, &[0.0][..]), (usize::MAX, 2, &[][..])] {
        let matrix = DenseMatrix {
            data: DenseData::F64(values),
            rows,
            cols,
            layout: Layout::RowMajor,
        };
        assert!(predictor.predict(&matrix, true, None).is_err());
    }
    let matrix = DenseMatrix {
        data: DenseData::F64(&[0.0]),
        rows: 1,
        cols: 1,
        layout: Layout::RowMajor,
    };
    assert!(predictor.predict_raw(&matrix, None, &mut []).is_err());
    assert_f32_bits(&predictor.predict(&matrix, true, None).unwrap(), &[0.0]);
}
