use silvic_core::data::{DenseData, DenseMatrix, Layout};
use silvic_core::model::{Model, ModelObjective};
use silvic_core::tree::{MissingType, SplitChildren, Tree, CATEGORICAL_MASK};
use silvic_neuron::NeuronPredictor;

fn enabled() -> bool {
    let enabled = std::env::var("SILVIC_TEST_NEURON").as_deref() == Ok("1");
    if !enabled {
        eprintln!("skipping Neuron device regression: set SILVIC_TEST_NEURON=1 on a configured Neuron host");
    }
    enabled
}

fn model(trees: Vec<Tree>, features: usize, objective: ModelObjective) -> Model {
    Model {
        objective,
        feature_names: (0..features).map(|index| format!("f{index}")).collect(),
        feature_infos: vec!["none".into(); features],
        trees,
        parameters: Vec::new(),
        pandas_categorical: None,
    }
}

fn children() -> SplitChildren {
    SplitChildren {
        left_value: -0.123456789012345,
        right_value: 0.76543210987654,
        left_count: 1,
        right_count: 1,
        left_weight: 1.0,
        right_weight: 1.0,
        gain: 1.0,
    }
}

// Independent oracle: let the core predictor choose leaves after quantizing the
// public model and input, then add original-order leaf values in Rust float32.
fn quantized_model(model: &Model) -> Model {
    let mut quantized = model.clone();
    for tree in &mut quantized.trees {
        for node in 0..tree.num_leaves.saturating_sub(1) {
            if tree.num_cat == 0 || tree.decision_type[node] & CATEGORICAL_MASK == 0 {
                let exact = tree.threshold[node];
                let nearest = exact as f32;
                tree.threshold[node] = f64::from(if f64::from(nearest) > exact {
                    nearest.next_down()
                } else {
                    nearest
                });
            }
        }
    }
    quantized
}

fn reference(model: &Model, matrix: &DenseMatrix<'_>, prefix: Option<usize>, raw: bool) -> Vec<f32> {
    let count = prefix.unwrap_or(model.trees.len()).min(model.trees.len());
    (0..matrix.rows)
        .map(|row| {
            let features: Vec<f64> = (0..matrix.cols)
                .map(|col| f64::from(matrix.get(row, col) as f32))
                .collect();
            let mut score = 0.0f32;
            for tree in &model.trees[..count] {
                score += tree.leaf_value[tree.leaf_index(&features)] as f32;
            }
            if !raw {
                if let ModelObjective::Binary { sigmoid } = model.objective {
                    score = 1.0f32 / (1.0f32 + (-(sigmoid as f32) * score).exp());
                }
            }
            score
        })
        .collect()
}

fn compare(predictor: &mut NeuronPredictor, model: &Model, matrix: &DenseMatrix<'_>, prefixes: &[Option<usize>]) {
    let quantized = quantized_model(model);
    for &prefix in prefixes {
        for raw in [true, false] {
            let expected = reference(&quantized, matrix, prefix, raw);
            let got = predictor.predict(matrix, raw, prefix).expect("Neuron prediction");
            for (row, (&got, &expected)) in got.iter().zip(&expected).enumerate() {
                assert!(
                    got.to_bits() == expected.to_bits() || (got.is_nan() && expected.is_nan()),
                    "row {row}, prefix {prefix:?}, raw {raw}: got {got:?}, expected {expected:?}"
                );
            }
            assert_eq!(got.len(), expected.len());
        }
        let mut out = vec![f32::NAN; matrix.rows];
        predictor.predict_raw(matrix, prefix, &mut out).unwrap();
        let expected = reference(&quantized, matrix, prefix, true);
        for (&got, &expected) in out.iter().zip(&expected) {
            assert!(got.to_bits() == expected.to_bits() || (got.is_nan() && expected.is_nan()));
        }
    }
}

#[test]
fn numerical_categories_missing_and_threshold_rounding() {
    if !enabled() {
        return;
    }
    let mut trees = Vec::new();
    let thresholds = [
        -f64::MAX,
        -1.0e-300,
        -0.0,
        0.0,
        1.0e-300,
        0.1,
        f64::from(1.0f32.next_up()).next_down(),
        f64::MAX,
        f64::NEG_INFINITY,
        f64::INFINITY,
        f64::NAN,
    ];
    for missing in [MissingType::None, MissingType::Zero, MissingType::NaN] {
        for default_left in [false, true] {
            for threshold in thresholds {
                let mut tree = Tree::new();
                tree.split_numerical(0, 0, threshold, default_left, missing, &children());
                trees.push(tree);
            }
        }
        let mut tree = Tree::new();
        tree.split_categorical(0, 0, &[0, 2, 31, 32, 40, 127, 16_777_217], missing, &children());
        trees.push(tree);
    }
    let mut unusual = Tree::new();
    unusual.split_numerical(0, 0, 0.5, false, MissingType::None, &children());
    unusual.decision_type[0] = 15;
    trees.push(unusual);
    let model = model(trees, 1, ModelObjective::Binary { sigmoid: 1.7 });
    let mut predictor = NeuronPredictor::new(&model, None).unwrap();
    assert_eq!(predictor.num_features(), 1);
    assert_eq!(predictor.num_trees(), model.trees.len());
    let mut values = vec![
        f64::NAN,
        f64::NEG_INFINITY,
        f64::INFINITY,
        f64::MIN,
        f64::MAX,
        -1.0,
        -0.999,
        -0.0,
        0.0,
        -1.0e-35,
        1.0e-35,
        1.0,
        2.9,
        31.0,
        32.0,
        40.0,
        127.0,
        128.0,
        16_777_216.0,
        16_777_217.0,
        16_777_218.0,
        2_147_483_520.0,
        2_147_483_647.0,
        2_147_483_648.0,
    ];
    for threshold in thresholds {
        values.extend([threshold.next_down(), threshold, threshold.next_up()]);
    }
    let prefixes: Vec<_> = (0..=model.trees.len())
        .map(Some)
        .chain([None, Some(usize::MAX)])
        .collect();
    for rows in [0, 1, 31, 32, 33, 127, 128, 129, 1025, 3] {
        let row_major: Vec<f64> = (0..rows).flat_map(|row| [values[row % values.len()], 999.0]).collect();
        for layout in [Layout::RowMajor, Layout::ColumnMajor] {
            let ordered: Vec<f64> = match layout {
                Layout::RowMajor => row_major.clone(),
                Layout::ColumnMajor => (0..2)
                    .flat_map(|col| row_major.as_chunks::<2>().0.iter().map(move |row| row[col]))
                    .collect(),
            };
            let f32_values: Vec<f32> = ordered.iter().map(|&value| value as f32).collect();
            for data in [DenseData::F64(&ordered), DenseData::F32(&f32_values)] {
                compare(
                    &mut predictor,
                    &model,
                    &DenseMatrix {
                        data,
                        rows,
                        cols: 2,
                        layout,
                    },
                    &prefixes,
                );
            }
        }
    }
}

#[test]
fn deep_nonpreorder_tree_and_empty_order_sensitive_forests() {
    if !enabled() {
        return;
    }
    let mut tree = Tree::new();
    tree.split_numerical(0, 0, 2.0, false, MissingType::None, &children());
    tree.split_numerical(1, 0, 5.0, false, MissingType::None, &children());
    tree.split_numerical(0, 0, 0.0, true, MissingType::NaN, &children());
    for _ in 0..300 {
        tree.split_numerical(0, 0, -1.0, true, MissingType::None, &children());
    }
    let forest = model(vec![tree], 1, ModelObjective::LambdaRank);
    let mut predictor = NeuronPredictor::new(&forest, None).unwrap();
    let values = [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 5.0, 6.0, f64::NAN];
    compare(
        &mut predictor,
        &forest,
        &DenseMatrix {
            data: DenseData::F64(&values),
            rows: values.len(),
            cols: 1,
            layout: Layout::RowMajor,
        },
        &[Some(0), None],
    );
    for leaves in [
        vec![],
        vec![0.0],
        vec![16_777_216.0, 1.0, -16_777_216.0, 3.0],
        vec![f64::from(f32::from_bits(1)), -0.0],
    ] {
        let trees = leaves
            .into_iter()
            .map(|leaf| {
                let mut tree = Tree::new();
                tree.leaf_value[0] = leaf;
                tree
            })
            .collect();
        let forest = model(trees, 0, ModelObjective::Binary { sigmoid: 2.0 });
        let mut predictor = NeuronPredictor::new(&forest, None).unwrap();
        for rows in [0, 1, 31, 32, 33, 128, 129, 257] {
            compare(
                &mut predictor,
                &forest,
                &DenseMatrix {
                    data: DenseData::F32(&[]),
                    rows,
                    cols: 0,
                    layout: Layout::ColumnMajor,
                },
                &[Some(0), Some(2), None, Some(usize::MAX)],
            );
        }
    }
}

#[test]
fn invalid_matrix_and_output_leave_output_untouched() {
    if !enabled() {
        return;
    }
    let mut tree = Tree::new();
    tree.split_numerical(0, 0, 1.0, false, MissingType::None, &children());
    let forest = model(vec![tree], 1, ModelObjective::Regression);
    let mut predictor = NeuronPredictor::new(&forest, None).unwrap();
    let mut output = [123.0f32];
    for matrix in [
        DenseMatrix {
            data: DenseData::F32(&[]),
            rows: 1,
            cols: 0,
            layout: Layout::RowMajor,
        },
        DenseMatrix {
            data: DenseData::F32(&[]),
            rows: 1,
            cols: 1,
            layout: Layout::RowMajor,
        },
        DenseMatrix {
            data: DenseData::F64(&[0.0]),
            rows: usize::MAX,
            cols: 2,
            layout: Layout::ColumnMajor,
        },
        DenseMatrix {
            data: DenseData::F64(&[]),
            rows: isize::MAX as usize,
            cols: 1,
            layout: Layout::RowMajor,
        },
        DenseMatrix {
            data: DenseData::F32(&[0.0, 0.0]),
            rows: 2,
            cols: 1,
            layout: Layout::RowMajor,
        },
    ] {
        assert!(predictor.predict_raw(&matrix, None, &mut output).is_err());
        assert_eq!(output, [123.0]);
    }
    assert!(predictor
        .predict_raw(
            &DenseMatrix {
                data: DenseData::F32(&[]),
                rows: 0,
                cols: 1,
                layout: Layout::RowMajor
            },
            None,
            &mut output
        )
        .is_err());
    assert_eq!(output, [123.0]);
}

#[test]
fn reference_contract_exposes_float32_quantization_and_order() {
    let trees = [16_777_216.0, 1.0, -16_777_216.0]
        .into_iter()
        .map(|value| {
            let mut tree = Tree::new();
            tree.leaf_value[0] = value;
            tree
        })
        .collect();
    let forest = model(trees, 0, ModelObjective::Regression);
    let matrix = DenseMatrix {
        data: DenseData::F32(&[]),
        rows: 1,
        cols: 0,
        layout: Layout::RowMajor,
    };
    assert_eq!(forest.predict_raw_row(&[], None), 1.0);
    assert_eq!(reference(&quantized_model(&forest), &matrix, None, true), [0.0]);
    assert_eq!(reference(&quantized_model(&forest), &matrix, Some(0), true), [0.0]);
    let mut tree = Tree::new();
    tree.split_numerical(0, 0, 0.1, false, MissingType::None, &children());
    let forest = model(vec![tree], 1, ModelObjective::Regression);
    let matrix = DenseMatrix {
        data: DenseData::F64(&[0.1]),
        rows: 1,
        cols: 1,
        layout: Layout::RowMajor,
    };
    assert_eq!(forest.predict_raw_row(&[0.1], None), children().left_value);
    assert_eq!(
        reference(&quantized_model(&forest), &matrix, None, true),
        [children().right_value as f32]
    );
}
