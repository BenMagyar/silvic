//! Prediction as LightGBM 3.3.5 predicts a dense row: the C API's row reader (`RowFunctionFromDenseMatric`) keeps a
//! value only when `|v| > kZeroThreshold` or it is NaN, so tiny values read as 0; each tree then walks
//! `Tree::GetLeaf` with `NumericalDecision` and `CategoricalDecision`, and `GBDT::PredictRaw` sums the leaf values in
//! tree order in f64. `GBDT::Predict` applies the objective's `ConvertOutput` (the sigmoid for binary).

use rayon::prelude::*;

use crate::data::{DenseData, DenseMatrix, Layout};
use crate::model::{ImportanceType, Model, ModelObjective};
use crate::tree::{in_bitset, MissingType, Tree, CATEGORICAL_MASK, DEFAULT_LEFT_MASK};

/// LightGBM `kZeroThreshold`, which is the float `1e-35f` widened to double.
const ZERO_THRESHOLD: f64 = 1e-35_f32 as f64;

impl Model {
    /// Sum of the leaf values `row` reaches in the first `num_trees` trees (all when `None`).
    ///
    /// `row` holds at least `num_features()` values; NaN is a missing value.
    pub fn predict_raw_row(&self, row: &[f64], num_trees: Option<usize>) -> f64 {
        self.used_trees(num_trees)
            .iter()
            .fold(0.0, |score, tree| score + tree.leaf_value[leaf_index(tree, row)])
    }

    /// Predictions for every row, parallel over rows; `raw_score = false` applies the objective's transform.
    ///
    /// # Panics
    /// When `matrix` has fewer columns than the model has features.
    pub fn predict(&self, matrix: &DenseMatrix<'_>, raw_score: bool, num_trees: Option<usize>) -> Vec<f64> {
        assert!(
            matrix.cols >= self.num_features(),
            "the matrix has {} columns, the model reads {} features",
            matrix.cols,
            self.num_features()
        );
        let convert = |raw: f64| if raw_score { raw } else { self.convert_output(raw) };
        if let (DenseData::F64(values), Layout::RowMajor) = (matrix.data, matrix.layout) {
            if matrix.cols > 0 {
                return values[..matrix.rows * matrix.cols]
                    .par_chunks_exact(matrix.cols)
                    .map(|row| convert(self.predict_raw_row(row, num_trees)))
                    .collect();
            }
        }
        (0..matrix.rows)
            .into_par_iter()
            .map_init(
                || vec![0.0; matrix.cols],
                |row, index| {
                    for (col, value) in row.iter_mut().enumerate() {
                        *value = matrix.get(index, col);
                    }
                    convert(self.predict_raw_row(row, num_trees))
                },
            )
            .collect()
    }

    /// LightGBM `GBDT::FeatureImportance`: per feature, the number of splits (`Split`) or their summed gain (`Gain`),
    /// counting only splits with positive gain.
    pub fn feature_importance(&self, kind: ImportanceType) -> Vec<f64> {
        let mut importance = vec![0.0; self.num_features()];
        for tree in &self.trees {
            let nodes = tree.num_leaves.saturating_sub(1);
            for (&feature, &gain) in tree.split_feature[..nodes].iter().zip(&tree.split_gain[..nodes]) {
                if gain > 0.0 {
                    importance[feature as usize] += match kind {
                        ImportanceType::Split => 1.0,
                        ImportanceType::Gain => f64::from(gain),
                    };
                }
            }
        }
        importance
    }

    fn used_trees(&self, num_trees: Option<usize>) -> &[Tree] {
        &self.trees[..num_trees.map_or(self.trees.len(), |n| n.min(self.trees.len()))]
    }

    /// The objective's `ConvertOutput`.
    fn convert_output(&self, raw: f64) -> f64 {
        match self.objective {
            ModelObjective::Binary { sigmoid } => 1.0 / (1.0 + (-sigmoid * raw).exp()),
            ModelObjective::Regression | ModelObjective::LambdaRank => raw,
        }
    }
}

/// `Tree::Predict`'s leaf: a one-leaf tree has no nodes to walk. Like `GetLeaf`, a tree without categorical splits
/// takes every node as numerical.
fn leaf_index(tree: &Tree, row: &[f64]) -> usize {
    if tree.num_leaves <= 1 {
        return 0;
    }
    let mut node = 0;
    while node >= 0 {
        let index = node as usize;
        let value = read_value(row[tree.split_feature[index] as usize]);
        node = if tree.num_cat > 0 && tree.decision_type[index] & CATEGORICAL_MASK != 0 {
            categorical_decision(tree, index, value)
        } else {
            numerical_decision(tree, index, value)
        };
    }
    !node as usize
}

/// The dense-row reader's view of a value: magnitudes up to `kZeroThreshold` are absent, that is 0.
#[inline]
fn read_value(value: f64) -> f64 {
    if value.abs() <= ZERO_THRESHOLD {
        0.0
    } else {
        value
    }
}

/// `Tree::NumericalDecision`: NaN is 0 unless the split has a NaN direction; a missing value (NaN, or `IsZero` for
/// `MissingType::Zero`) takes the default direction.
#[inline]
fn numerical_decision(tree: &Tree, node: usize, value: f64) -> i32 {
    let decision_type = tree.decision_type[node];
    let missing_type = MissingType::from_decision_type(decision_type);
    let value = if value.is_nan() && missing_type != MissingType::NaN {
        0.0
    } else {
        value
    };
    let is_missing = match missing_type {
        MissingType::None => false,
        MissingType::Zero => (-ZERO_THRESHOLD..=ZERO_THRESHOLD).contains(&value),
        MissingType::NaN => value.is_nan(),
    };
    let left = if is_missing {
        decision_type & DEFAULT_LEFT_MASK != 0
    } else {
        value <= tree.threshold[node]
    };
    if left {
        tree.left_child[node]
    } else {
        tree.right_child[node]
    }
}

/// `Tree::CategoricalDecision`: NaN and negative values go right, others left when their bit is set. The cast
/// truncates toward zero like `static_cast<int>`; out of `int` range C++ is undefined, Rust saturates, and the
/// result goes right either way (negative, or past every bitset).
#[inline]
fn categorical_decision(tree: &Tree, node: usize, value: f64) -> i32 {
    if value.is_nan() || (value as i32) < 0 {
        return tree.right_child[node];
    }
    let cat_index = tree.threshold[node] as usize;
    let start = tree.cat_boundaries[cat_index] as usize;
    let end = tree.cat_boundaries[cat_index + 1] as usize;
    if in_bitset(&tree.cat_threshold[start..end], value as i32) {
        tree.left_child[node]
    } else {
        tree.right_child[node]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tree::SplitChildren;

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

    fn model(tree: Tree) -> Model {
        Model {
            objective: ModelObjective::Binary { sigmoid: 2.0 },
            feature_names: vec!["x".into()],
            feature_infos: vec!["[-1:1]".into()],
            trees: vec![tree],
            parameters: Vec::new(),
            pandas_categorical: None,
        }
    }

    #[test]
    fn missing_types_route_like_lightgbm() {
        let cases = [
            (MissingType::None, true, f64::NAN, -1.0),
            (MissingType::None, false, f64::NAN, -1.0),
            (MissingType::Zero, false, f64::NAN, 1.0),
            (MissingType::Zero, true, 1e-36, -1.0),
            (MissingType::Zero, false, 0.0, 1.0),
            (MissingType::NaN, true, f64::NAN, -1.0),
            (MissingType::NaN, false, f64::NAN, 1.0),
            (MissingType::NaN, false, -1e-36, -1.0),
        ];
        for (missing, default_left, value, expected) in cases {
            let mut tree = Tree::new();
            tree.split_numerical(0, 0, 0.0, default_left, missing, &children(-1.0, 1.0));
            let model = model(tree);
            let case = format!("{missing:?} default_left={default_left} value={value}");
            assert_eq!(model.predict_raw_row(&[value], None), expected, "{case}");
        }
    }

    #[test]
    fn categorical_values_outside_the_bitset_go_right() {
        let mut tree = Tree::new();
        tree.split_categorical(0, 0, &[0, 2, 40], MissingType::None, &children(-1.0, 1.0));
        let model = model(tree);
        for (value, expected) in [
            (0.0, -1.0),
            (-0.5, -1.0),
            (2.9, -1.0),
            (40.0, -1.0),
            (1.0, 1.0),
            (-1.0, 1.0),
            (f64::NAN, 1.0),
            (64.0, 1.0),
            (1e10, 1.0),
            (-1e10, 1.0),
        ] {
            assert_eq!(model.predict_raw_row(&[value], None), expected, "category {value}");
        }
    }

    #[test]
    fn predict_matches_rows_in_every_layout() {
        let mut tree = Tree::new();
        tree.split_numerical(0, 0, 0.5, false, MissingType::NaN, &children(-1.0, 1.0));
        let mut model = model(tree);
        model.feature_names.push("y".into());
        model.feature_infos.push("none".into());
        let row_major = [0.0, 9.0, 1.0, 9.0, f64::NAN, 9.0];
        let column_major = [0.0f32, 1.0, f32::NAN, 9.0, 9.0, 9.0];
        let raw = vec![-1.0, 1.0, 1.0];
        let sigmoid: Vec<f64> = raw.iter().map(|r: &f64| 1.0 / (1.0 + (-2.0 * r).exp())).collect();
        let rows = DenseMatrix {
            data: DenseData::F64(&row_major),
            rows: 3,
            cols: 2,
            layout: Layout::RowMajor,
        };
        assert_eq!(model.predict(&rows, true, None), raw);
        assert_eq!(model.predict(&rows, false, None), sigmoid);
        let cols = DenseMatrix {
            data: DenseData::F32(&column_major),
            rows: 3,
            cols: 2,
            layout: Layout::ColumnMajor,
        };
        assert_eq!(model.predict(&cols, true, None), raw);
        assert_eq!(model.predict(&cols, true, Some(0)), vec![0.0; 3]);
    }

    #[test]
    fn importance_counts_positive_gain_splits() {
        let mut tree = Tree::new();
        let mut split = children(0.0, 0.0);
        split.gain = 2.5;
        tree.split_numerical(0, 0, 0.0, false, MissingType::None, &split);
        split.gain = 0.0;
        tree.split_numerical(0, 0, 0.0, false, MissingType::None, &split);
        let model = model(tree);
        assert_eq!(model.feature_importance(ImportanceType::Split), vec![1.0]);
        assert_eq!(model.feature_importance(ImportanceType::Gain), vec![2.5]);
    }
}
