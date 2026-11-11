//! Compares against `silvic_core::Model::predict_raw_row` and `Model::predict` (LightGBM 3.3.5's decisions) on
//! random forests with every missing-value handling, categorical bitsets of one and several words, single-leaf trees,
//! trees past 64 and 256 leaves, NaN thresholds and features with more thresholds than a slot holds, and on rows
//! that hit thresholds, their neighbouring doubles, NaN, ±0, ±`kZeroThreshold` and ±inf.

mod common;

use silvic_core::data::{DenseData, DenseMatrix, Layout};
use silvic_core::model::{Model, ModelObjective};
use silvic_core::tree::{MissingType, SplitChildren, Tree};
use silvic_serve::{Kernel, Predictor};

use common::{model, random_model, random_rows, random_tree, threshold_grid, ForestSpec, Rng, ZERO_THRESHOLD};

/// Every available kernel's raw scores equal `predict_raw_row`'s, for all trees and for prefixes.
fn assert_kernels_match(case: &str, model: &Model, rows: &[f64], cols: usize) {
    let count = rows.len() / cols;
    let matrix = DenseMatrix {
        data: DenseData::F64(rows),
        rows: count,
        cols,
        layout: Layout::RowMajor,
    };
    let prefixes = [
        None,
        Some(0),
        Some(1),
        Some(model.num_trees() / 2),
        Some(model.num_trees() + 5),
    ];
    for kernel in Kernel::available() {
        let predictor = Predictor::with_kernel(model, kernel).expect("compile");
        for num_trees in prefixes {
            let mut out = vec![0.0; count];
            predictor.predict_raw(&matrix, num_trees, &mut out);
            for (index, (row, got)) in rows.chunks(cols).zip(&out).enumerate() {
                let want = model.predict_raw_row(row, num_trees);
                assert_eq!(
                    got.to_bits(),
                    want.to_bits(),
                    "{case} {kernel} trees {num_trees:?}: row {index} {row:?}: got {got}, reference {want}"
                );
            }
        }
    }
}

/// Checks random forests of `spec`; returns how many of their trees were walked.
fn check_spec(case: &str, spec: &ForestSpec, seeds: std::ops::Range<u64>, rows: usize) -> usize {
    let mut walked = 0;
    for seed in seeds {
        let mut rng = Rng::new(seed);
        let (model, grid) = random_model(&mut rng, spec);
        let values = random_rows(&mut rng, spec, &grid, rows);
        assert_kernels_match(&format!("{case} seed {seed}"), &model, &values, spec.features);
        walked += Predictor::new(&model).expect("compile").num_walked_trees();
    }
    walked
}

#[test]
fn small_forests_match() {
    assert_eq!(check_spec("default", &ForestSpec::default(), 0..30, 157), 0);
}

#[test]
fn workload_shaped_forest_matches() {
    let spec = ForestSpec {
        trees: 300,
        leaves: 53,
        features: 40,
        categorical: 5,
        thresholds: 200,
        edge_thresholds: 0.02,
        objective: ModelObjective::LambdaRank,
        ..ForestSpec::default()
    };
    assert_eq!(check_spec("workload", &spec, 100..103, 300), 0);
}

#[test]
fn multi_word_bitsets_match() {
    // Twelve categories spread over 0..=99: bitsets of four words, twelve classes.
    let spec = ForestSpec {
        categories: 12,
        category_stride: 9,
        categorical: 4,
        features: 6,
        ..ForestSpec::default()
    };
    assert_eq!(check_spec("multi-word bitsets", &spec, 200..210, 211), 0);
}

#[test]
fn categoricals_past_64_classes_are_walked_and_match() {
    // 300 categories split at random give far more than 64 distinct behaviours, so these trees are walked.
    let spec = ForestSpec {
        categories: 300,
        categorical: 3,
        features: 5,
        trees: 20,
        ..ForestSpec::default()
    };
    assert!(check_spec("many classes", &spec, 300..305, 200) > 0);
}

#[test]
fn features_past_one_slot_of_thresholds_match() {
    // 700 thresholds per feature spread over three slots of 254.
    let spec = ForestSpec {
        thresholds: 700,
        features: 3,
        categorical: 0,
        trees: 200,
        leaves: 63,
        edge_thresholds: 0.0,
        ..ForestSpec::default()
    };
    assert_eq!(check_spec("700 thresholds", &spec, 400..404, 400), 0);
}

#[test]
fn trees_past_64_and_256_leaves_match() {
    for (leaves, max_depth) in [(64, 40), (65, 40), (100, 40), (256, 300), (300, 300)] {
        let spec = ForestSpec {
            trees: 6,
            leaves,
            max_depth,
            ..ForestSpec::default()
        };
        let walked = check_spec(&format!("{leaves} leaves"), &spec, 500..503, 300);
        assert_eq!(walked > 0, leaves > 256, "{leaves} leaves: {walked} trees walked");
    }
}

#[test]
fn single_leaf_and_mixed_trees_match() {
    let spec = ForestSpec::default();
    let mut rng = Rng::new(600);
    let grid = threshold_grid(&mut rng, &spec);
    let mut trees = Vec::new();
    for index in 0..40 {
        trees.push(match index % 4 {
            0 => {
                let mut tree = Tree::new();
                tree.leaf_value[0] = rng.normal();
                tree
            }
            1 => random_tree(
                &mut rng,
                &ForestSpec {
                    leaves: 2,
                    ..spec.clone()
                },
                &grid,
            ),
            2 => random_tree(
                &mut rng,
                &ForestSpec {
                    leaves: 120,
                    max_depth: 30,
                    ..spec.clone()
                },
                &grid,
            ),
            _ => random_tree(&mut rng, &spec, &grid),
        });
    }
    let model = model(trees, spec.features, ModelObjective::Regression);
    let rows = random_rows(&mut rng, &spec, &grid, 333);
    assert_kernels_match("single leaf and mixed", &model, &rows, spec.features);
}

fn children(left_value: f64, right_value: f64) -> SplitChildren {
    SplitChildren {
        left_value,
        right_value,
        left_count: 1,
        right_count: 1,
        left_weight: 1.0,
        right_weight: 1.0,
        gain: 1.0,
    }
}

/// Values around every threshold of every node: the threshold, its neighbouring doubles, and the edge values.
fn rows_around_thresholds(model: &Model, features: usize) -> Vec<f64> {
    let edges = [
        f64::NAN,
        0.0,
        -0.0,
        ZERO_THRESHOLD,
        -ZERO_THRESHOLD,
        ZERO_THRESHOLD.next_up(),
        -ZERO_THRESHOLD.next_up(),
        ZERO_THRESHOLD.next_down(),
        1e-40,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::MAX,
        f64::MIN,
    ];
    let mut values: Vec<f64> = edges.to_vec();
    for tree in &model.trees {
        for &threshold in &tree.threshold {
            values.extend([threshold, threshold.next_up(), threshold.next_down()]);
        }
    }
    let mut rows = Vec::new();
    for (index, &value) in values.iter().enumerate() {
        for feature in 0..features {
            let mut row: Vec<f64> = (0..features)
                .map(|f| values[(index + 7 * f + 3) % values.len()])
                .collect();
            row[feature] = value;
            rows.extend(row);
        }
    }
    rows
}

#[test]
fn every_missing_handling_on_edge_values_matches() {
    let thresholds = [
        0.0,
        -0.0,
        ZERO_THRESHOLD,
        -ZERO_THRESHOLD,
        1e-300,
        -1e-300,
        0.5,
        -0.5,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::MAX,
    ];
    let mut trees = Vec::new();
    for &threshold in &thresholds {
        for missing in [MissingType::None, MissingType::Zero, MissingType::NaN] {
            for default_left in [false, true] {
                let mut tree = Tree::new();
                let right = tree.split_numerical(0, 0, threshold, default_left, missing, &children(1.0, 2.0));
                tree.split_numerical(right, 1, -threshold, !default_left, missing, &children(4.0, 8.0));
                trees.push(tree);
            }
        }
    }
    let model = model(trees, 2, ModelObjective::Regression);
    assert_kernels_match("edge thresholds", &model, &rows_around_thresholds(&model, 2), 2);
}

#[test]
fn nan_thresholds_are_walked_and_match() {
    let mut trees = Vec::new();
    for missing in [MissingType::None, MissingType::Zero, MissingType::NaN] {
        let mut tree = Tree::new();
        let right = tree.split_numerical(0, 0, f64::NAN, true, missing, &children(1.0, 2.0));
        tree.split_numerical(right, 1, 0.25, false, missing, &children(4.0, 8.0));
        trees.push(tree);
    }
    let model = model(trees, 2, ModelObjective::Regression);
    assert_eq!(Predictor::new(&model).expect("compile").num_walked_trees(), 3);
    assert_kernels_match("NaN thresholds", &model, &rows_around_thresholds(&model, 2), 2);
}

#[test]
fn categorical_edge_values_match() {
    let mut trees = Vec::new();
    for categories in [vec![0], vec![1, 3], vec![0, 31, 32, 33], vec![5, 64, 200], vec![2_000]] {
        let mut tree = Tree::new();
        let right = tree.split_categorical(0, 0, &categories, MissingType::NaN, &children(1.0, 2.0));
        tree.split_numerical(right, 1, 0.5, true, MissingType::NaN, &children(4.0, 8.0));
        trees.push(tree);
    }
    let model = model(trees, 2, ModelObjective::Regression);
    let values = [
        f64::NAN,
        0.0,
        -0.0,
        1e-40,
        0.9,
        -0.5,
        -0.99,
        -1.0,
        1.0,
        1.5,
        3.0,
        31.0,
        32.0,
        32.7,
        33.0,
        63.0,
        64.0,
        200.0,
        1999.0,
        2000.0,
        2000.5,
        2001.0,
        1e10,
        f64::INFINITY,
        f64::NEG_INFINITY,
        2_147_483_648.0,
        -2_147_483_649.0,
    ];
    let rows: Vec<f64> = values.iter().flat_map(|&v| [v, 0.25]).collect();
    assert_kernels_match("categorical edges", &model, &rows, 2);
}

#[test]
fn sixty_four_classes_fill_the_table() {
    // Split `bit` sends left the categories with that bit set, and one split sends all 64 left, so every category in
    // 0..64 behaves differently: 64 classes, the most a table holds.
    let mut trees = Vec::new();
    for bit in 0..=6 {
        let categories: Vec<i32> = (0..64).filter(|c| bit == 6 || c >> bit & 1 == 1).collect();
        let mut tree = Tree::new();
        tree.split_categorical(
            0,
            0,
            &categories,
            MissingType::NaN,
            &children(f64::from(1 << bit), -1.0),
        );
        trees.push(tree);
    }
    let model = model(trees, 1, ModelObjective::Regression);
    assert_eq!(Predictor::new(&model).expect("compile").num_walked_trees(), 0);
    let rows: Vec<f64> = (-2..70).map(f64::from).chain([f64::NAN, 63.5, 1e9]).collect();
    assert_kernels_match("64 classes", &model, &rows, 1);
}

#[test]
fn a_tree_without_categorical_splits_reads_the_categorical_bit_as_numeric() {
    // `num_cat == 0` makes `Tree::GetLeaf` treat every node as numerical, whatever its `decision_type`.
    let mut tree = Tree::new();
    tree.split_numerical(0, 0, 1.5, false, MissingType::NaN, &children(1.0, 2.0));
    tree.decision_type[0] |= silvic_core::tree::CATEGORICAL_MASK;
    let model = model(vec![tree], 1, ModelObjective::Regression);
    let rows = [f64::NAN, 0.0, 1.0, 1.5, 1.6, 2.0, -1.0];
    assert_kernels_match("categorical bit, no bitsets", &model, &rows, 1);
}

#[test]
fn layouts_dtypes_objectives_and_threads_match_model_predict() {
    let spec = ForestSpec {
        objective: ModelObjective::Binary { sigmoid: 1.7 },
        trees: 80,
        ..ForestSpec::default()
    };
    let mut rng = Rng::new(700);
    let (model, grid) = random_model(&mut rng, &spec);
    let rows = 1000;
    let cols = spec.features + 2;
    let mut values = Vec::with_capacity(rows * cols);
    for row in random_rows(&mut rng, &spec, &grid, rows).chunks(spec.features) {
        values.extend_from_slice(row);
        values.extend([rng.normal(), f64::NAN]);
    }
    let narrow: Vec<f32> = values.iter().map(|&v| v as f32).collect();
    let transpose = |values: &[f64]| -> Vec<f64> {
        (0..cols)
            .flat_map(|c| (0..rows).map(move |r| values[r * cols + c]))
            .collect()
    };
    let column_major = transpose(&values);
    let narrow_column_major: Vec<f32> = column_major.iter().map(|&v| v as f32).collect();
    let matrices = [
        ("f64 rows", DenseData::F64(&values), Layout::RowMajor),
        ("f64 columns", DenseData::F64(&column_major), Layout::ColumnMajor),
        ("f32 rows", DenseData::F32(&narrow), Layout::RowMajor),
        ("f32 columns", DenseData::F32(&narrow_column_major), Layout::ColumnMajor),
    ];
    for kernel in Kernel::available() {
        let predictor = Predictor::with_kernel(&model, kernel).expect("compile");
        for (name, data, layout) in matrices {
            let matrix = DenseMatrix {
                data,
                rows,
                cols,
                layout,
            };
            for (raw_score, num_trees) in [(false, None), (true, None), (false, Some(17))] {
                let expected = model.predict(&matrix, raw_score, num_trees);
                for threads in [1, 0, 3, 64] {
                    let got = predictor.predict(&matrix, raw_score, num_trees, threads);
                    assert_eq!(got.len(), rows);
                    for (index, (g, e)) in got.iter().zip(&expected).enumerate() {
                        assert_eq!(
                            g.to_bits(),
                            e.to_bits(),
                            "{kernel} {name} raw {raw_score} trees {num_trees:?} threads {threads}: row {index}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn empty_and_tiny_batches() {
    let spec = ForestSpec::default();
    let mut rng = Rng::new(800);
    let (model, grid) = random_model(&mut rng, &spec);
    for kernel in Kernel::available() {
        let predictor = Predictor::with_kernel(&model, kernel).expect("compile");
        for rows in [0, 1, 2, 31, 32, 33, 63, 64, 65, 129] {
            let values = random_rows(&mut rng, &spec, &grid, rows);
            let matrix = DenseMatrix {
                data: DenseData::F64(&values),
                rows,
                cols: spec.features,
                layout: Layout::RowMajor,
            };
            assert_eq!(
                predictor.predict(&matrix, true, None, 2),
                model.predict(&matrix, true, None),
                "{kernel} {rows}"
            );
        }
    }
}

#[test]
fn malformed_trees_are_errors() {
    let mut cycle = Tree::new();
    cycle.split_numerical(0, 0, 0.5, true, MissingType::NaN, &children(1.0, 2.0));
    cycle.split_numerical(1, 0, 0.7, true, MissingType::NaN, &children(1.0, 2.0));
    cycle.left_child[1] = 0;
    let mut feature = Tree::new();
    feature.split_numerical(0, 5, 0.5, true, MissingType::NaN, &children(1.0, 2.0));
    let mut leaf = Tree::new();
    leaf.split_numerical(0, 0, 0.5, true, MissingType::NaN, &children(1.0, 2.0));
    leaf.right_child[0] = !7;
    let mut bitset = Tree::new();
    bitset.split_categorical(0, 0, &[1], MissingType::NaN, &children(1.0, 2.0));
    bitset.threshold[0] = 3.0;
    for (name, tree) in [
        ("cycle", cycle),
        ("feature", feature),
        ("leaf", leaf),
        ("bitset", bitset),
    ] {
        let error = Predictor::new(&model(vec![tree], 2, ModelObjective::Regression)).expect_err(name);
        assert_eq!(error.kind(), "model", "{name}: {error}");
    }
}

/// `SILVIC_SERVE_EXPECT_KERNELS=portable,avx2` makes the run fail unless exactly those kernels ran, so a run meant to
/// exercise a SIMD kernel (x86-64 under Rosetta needs `ROSETTA_ADVERTISE_AVX=1` to report AVX2) cannot silently
/// test only the portable one.
#[test]
fn kernels_parse_and_report_availability() {
    if let Ok(expected) = std::env::var("SILVIC_SERVE_EXPECT_KERNELS") {
        let available: Vec<&str> = Kernel::available().into_iter().map(Kernel::name).collect();
        assert_eq!(available.join(","), expected, "kernels this CPU runs");
    }
    assert!(Kernel::available().contains(&Kernel::Portable));
    assert!(Kernel::available().contains(&Kernel::detect()));
    for kernel in Kernel::ALL {
        assert_eq!(kernel.name().parse::<Kernel>().expect("name"), kernel);
    }
    assert!("sse".parse::<Kernel>().is_err());
    let model = model(vec![Tree::new()], 1, ModelObjective::Regression);
    for kernel in Kernel::ALL {
        assert_eq!(
            Predictor::with_kernel(&model, kernel).is_ok(),
            kernel.is_available(),
            "{kernel}"
        );
    }
}
