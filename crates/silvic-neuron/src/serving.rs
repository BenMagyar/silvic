//! Explicit float32 serving with resident forest metadata and bounded row tiles.

use silvic_core::data::{DenseData, DenseMatrix, Layout};
use silvic_core::model::{Model, ModelObjective};
use silvic_core::tree::{Tree, CATEGORICAL_MASK};
use silvic_core::{Result, SilvicError};

use crate::runtime::{Buffer, DType, Device, Execution, TensorSpec};

const TILE_ROWS: usize = 32;
const META_FIELDS: usize = 9;
const VALUE_FIELDS: usize = 3;

struct PackedForest {
    metadata: Vec<i32>,
    values: Vec<f32>,
    categories: Vec<u32>,
    records: usize,
}

fn threshold_f32(value: f64) -> f32 {
    let rounded = value as f32;
    if f64::from(rounded) > value {
        rounded.next_down()
    } else {
        rounded
    }
}

fn device_index(value: usize, what: &str) -> Result<i32> {
    i32::try_from(value)
        .map_err(|_| SilvicError::model(format!("Neuron {what} exceeds signed 32-bit device addressing")))
}

impl PackedForest {
    fn new(model: &Model) -> Result<Self> {
        device_index(model.num_features(), "feature count")?;
        device_index(model.trees.len(), "tree count")?;
        let mut packed = Self {
            metadata: Vec::new(),
            values: Vec::new(),
            categories: vec![0],
            records: 0,
        };
        for (tree_index, tree) in model.trees.iter().enumerate() {
            let nodes = checked_preorder(tree, model.num_features())
                .map_err(|message| SilvicError::model(format!("tree {tree_index}: {message}")))?;
            let category_base = packed.categories.len();
            let category_end = category_base
                .checked_add(tree.cat_threshold.len())
                .ok_or_else(|| SilvicError::model("Neuron category storage overflows usize"))?;
            device_index(category_end, "category storage")?;
            packed.categories.extend_from_slice(&tree.cat_threshold);
            if nodes.is_empty() {
                packed.push(
                    [0, -1, -1, 0, 0, 0, 0, tree_index as i32, 1],
                    [f32::NAN, tree.leaf_value[0] as f32, tree.leaf_value[0] as f32],
                )?;
                continue;
            }
            for (position, &node) in nodes.iter().enumerate() {
                let categorical = tree.num_cat > 0 && tree.decision_type[node] & CATEGORICAL_MASK != 0;
                let (category_start, category_len) = if categorical {
                    let category = tree.threshold[node] as usize;
                    let start = tree.cat_boundaries[category] as usize;
                    let end = tree.cat_boundaries[category + 1] as usize;
                    (
                        device_index(category_base + start, "category offset")?,
                        device_index(end - start, "category length")?,
                    )
                } else {
                    (0, 0)
                };
                let leaf = |child: i32| {
                    if child < 0 {
                        tree.leaf_value[!child as usize] as f32
                    } else {
                        0.0
                    }
                };
                packed.push(
                    [
                        node as i32,
                        tree.left_child[node],
                        tree.right_child[node],
                        tree.split_feature[node],
                        i32::from(tree.decision_type[node] as u8 & !(CATEGORICAL_MASK as u8)) | i32::from(categorical),
                        category_start,
                        category_len,
                        tree_index as i32,
                        i32::from(position + 1 == nodes.len()),
                    ],
                    [
                        threshold_f32(tree.threshold[node]),
                        leaf(tree.left_child[node]),
                        leaf(tree.right_child[node]),
                    ],
                )?;
            }
        }
        // An inert tree outside the zero-length prefix keeps all runtime ports live.
        if packed.records == 0 {
            packed.push([0, -1, -1, 0, 0, 0, 0, 0, 1], [f32::NAN, 0.0, 0.0])?;
        }
        Ok(packed)
    }

    fn push(&mut self, metadata: [i32; META_FIELDS], values: [f32; VALUE_FIELDS]) -> Result<()> {
        let records = self
            .records
            .checked_add(1)
            .ok_or_else(|| SilvicError::model("Neuron forest record count overflows usize"))?;
        let elements = records
            .checked_mul(META_FIELDS)
            .ok_or_else(|| SilvicError::model("Neuron forest metadata size overflows usize"))?;
        device_index(elements, "forest metadata storage")?;
        self.metadata.extend_from_slice(&metadata);
        self.values.extend_from_slice(&values);
        self.records = records;
        Ok(())
    }
}

// The same reachable-node validation as the existing serving compiler: ignore unused
// trailing arrays, reject repeated internal nodes, and validate only read bitsets.
fn checked_preorder(tree: &Tree, features: usize) -> std::result::Result<Vec<usize>, String> {
    if tree.leaf_value.is_empty() {
        return Err("no leaf values".into());
    }
    if tree.num_leaves <= 1 {
        return Ok(Vec::new());
    }
    let nodes = tree.num_leaves - 1;
    if nodes > i32::MAX as usize {
        return Err("node count exceeds signed 32-bit device addressing".into());
    }
    if [
        tree.split_feature.len(),
        tree.threshold.len(),
        tree.decision_type.len(),
        tree.left_child.len(),
        tree.right_child.len(),
    ]
    .iter()
    .any(|&len| len < nodes)
        || tree.leaf_value.len() < tree.num_leaves
    {
        return Err(format!("node arrays shorter than num_leaves={}", tree.num_leaves));
    }
    let mut seen = vec![false; nodes];
    let mut stack = vec![0i32];
    let mut preorder = Vec::new();
    while let Some(child) = stack.pop() {
        if child < 0 {
            if !child as usize >= tree.num_leaves {
                return Err(format!("child {child} names an out-of-range leaf"));
            }
            continue;
        }
        let node = child as usize;
        if node >= nodes || seen[node] {
            return Err(format!("node {node} is out of range or reached twice"));
        }
        seen[node] = true;
        let feature = tree.split_feature[node];
        if usize::try_from(feature).map_or(true, |feature| feature >= features) {
            return Err(format!("node {node} splits on feature {feature} of {features}"));
        }
        if tree.num_cat > 0 && tree.decision_type[node] & CATEGORICAL_MASK != 0 {
            let threshold = tree.threshold[node];
            let category = threshold as usize;
            let valid =
                threshold >= 0.0 && category < tree.num_cat && category < tree.cat_boundaries.len().saturating_sub(1);
            if !valid {
                return Err(format!("node {node} names categorical bitset {threshold}"));
            }
            let start = tree.cat_boundaries[category];
            let end = tree.cat_boundaries[category + 1];
            if start < 0 || end < start || end as usize > tree.cat_threshold.len() {
                return Err(format!("node {node} names categorical bitset {threshold}"));
            }
        }
        preorder.push(node);
        stack.push(tree.right_child[node]);
        stack.push(tree.left_child[node]);
    }
    Ok(preorder)
}

fn check_matrix(rows: &DenseMatrix<'_>, features: usize) -> Result<()> {
    if rows.cols < features {
        return Err(SilvicError::config(format!(
            "the matrix has {} columns, the model reads {features} features",
            rows.cols
        )));
    }
    let elements = rows
        .rows
        .checked_mul(rows.cols)
        .ok_or_else(|| SilvicError::config("Neuron prediction matrix dimensions overflow usize"))?;
    let available = match rows.data {
        DenseData::F32(values) => values.len(),
        DenseData::F64(values) => values.len(),
    };
    if elements > available || rows.rows > isize::MAX as usize / std::mem::size_of::<f32>() {
        return Err(SilvicError::config(
            "Neuron prediction matrix storage is too short or output is too large",
        ));
    }
    Ok(())
}

/// A float32 Neuron predictor. Inputs, leaves, and ordered tree sums are float32;
/// numerical thresholds round down. Device failures never select a CPU fallback.
pub struct NeuronPredictor {
    input: Buffer,
    prefix: Buffer,
    output: Buffer,
    execution: Execution,
    input_tile: Vec<f32>,
    output_tile: [f32; TILE_ROWS],
    num_features: usize,
    num_trees: usize,
    objective: ModelObjective,
}

impl NeuronPredictor {
    /// Validates and uploads a forest once and binds reusable, bounded row tiles.
    pub fn new(model: &Model, ordinal: Option<usize>) -> Result<Self> {
        let packed = PackedForest::new(model)?;
        let input_elements = model
            .num_features()
            .max(1)
            .checked_mul(TILE_ROWS)
            .ok_or_else(|| SilvicError::model("Neuron prediction tile size overflows usize"))?;
        device_index(input_elements, "input tile storage")?;
        let device = Device::new(ordinal)?;
        let metadata = device.upload(&packed.metadata)?;
        let values = device.upload(&packed.values)?;
        let categories = device.upload(&packed.categories)?;
        let input_tile = vec![0.0f32; input_elements];
        let input = device.upload(&input_tile)?;
        let prefix = device.upload(&[model.trees.len() as i32])?;
        let output = device.allocate(TILE_ROWS * std::mem::size_of::<f32>())?;
        let inputs = [
            TensorSpec::new("input", DType::F32, vec![1, input_elements]),
            TensorSpec::new("metadata", DType::I32, vec![1, packed.metadata.len()]),
            TensorSpec::new("values", DType::F32, vec![1, packed.values.len()]),
            TensorSpec::new("categories", DType::U32, vec![packed.categories.len(), 1]),
            TensorSpec::new("prefix", DType::I32, vec![1, 1]),
        ];
        let outputs = [TensorSpec::new("scores", DType::F32, vec![1, TILE_ROWS])];
        let kernel = device.compile(
            include_str!("../kernels/serving.py"),
            "score",
            &serde_json::json!({"RECORDS": packed.records, "ROWS": TILE_ROWS}),
            &inputs,
            &outputs,
        )?;
        let execution = kernel.bind(
            &[input.clone(), metadata, values, categories, prefix.clone()],
            std::slice::from_ref(&output),
        )?;
        Ok(Self {
            input,
            prefix,
            output,
            execution,
            input_tile,
            output_tile: [0.0; TILE_ROWS],
            num_features: model.num_features(),
            num_trees: model.trees.len(),
            objective: model.objective.clone(),
        })
    }

    pub fn num_trees(&self) -> usize {
        self.num_trees
    }

    pub fn num_features(&self) -> usize {
        self.num_features
    }

    /// Writes one raw float32 score per row using a clamped tree prefix.
    /// Accepts f32/f64 input in either layout; f64 is quantized before decisions.
    /// Invalid dimensions, storage, or output lengths fail before any output write.
    pub fn predict_raw(&mut self, rows: &DenseMatrix<'_>, num_trees: Option<usize>, out: &mut [f32]) -> Result<()> {
        check_matrix(rows, self.num_features)?;
        if out.len() != rows.rows {
            return Err(SilvicError::config("Neuron prediction requires one output per row"));
        }
        if rows.rows == 0 {
            return Ok(());
        }
        self.prefix
            .write(&[num_trees.unwrap_or(self.num_trees).min(self.num_trees) as i32])?;
        for first in (0..rows.rows).step_by(TILE_ROWS) {
            let count = (rows.rows - first).min(TILE_ROWS);
            for feature in 0..self.num_features {
                let tile = &mut self.input_tile[feature * TILE_ROWS..(feature + 1) * TILE_ROWS];
                for (lane, value) in tile[..count].iter_mut().enumerate() {
                    let row = first + lane;
                    let index = match rows.layout {
                        Layout::RowMajor => row * rows.cols + feature,
                        Layout::ColumnMajor => feature * rows.rows + row,
                    };
                    *value = match rows.data {
                        DenseData::F32(values) => values[index],
                        DenseData::F64(values) => values[index] as f32,
                    };
                }
                tile[count..].fill(0.0);
            }
            self.input.write(&self.input_tile)?;
            self.execution.execute()?;
            self.output.read(&mut self.output_tile)?;
            out[first..first + count].copy_from_slice(&self.output_tile[..count]);
        }
        Ok(())
    }

    /// Scores rows and optionally applies the objective in host float32 arithmetic.
    pub fn predict(&mut self, rows: &DenseMatrix<'_>, raw_score: bool, num_trees: Option<usize>) -> Result<Vec<f32>> {
        check_matrix(rows, self.num_features)?;
        let mut out = vec![0.0f32; rows.rows];
        self.predict_raw(rows, num_trees, &mut out)?;
        if !raw_score {
            if let ModelObjective::Binary { sigmoid } = self.objective {
                let sigmoid = sigmoid as f32;
                for value in &mut out {
                    *value = 1.0f32 / (1.0f32 + (-sigmoid * *value).exp());
                }
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model(trees: Vec<Tree>) -> Model {
        Model {
            objective: ModelObjective::Regression,
            feature_names: vec!["f0".into()],
            feature_infos: vec!["none".into()],
            trees,
            parameters: Vec::new(),
            pandas_categorical: None,
        }
    }

    #[test]
    fn thresholds_round_toward_negative_infinity() {
        for exact in [
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
        ] {
            let rounded = threshold_f32(exact);
            assert!(f64::from(rounded) <= exact);
            if rounded != f32::INFINITY {
                assert!(f64::from(rounded.next_up()) > exact || exact == f64::INFINITY);
            }
        }
        assert!(threshold_f32(f64::NAN).is_nan());
        assert_eq!(threshold_f32(-0.0).to_bits(), (-0.0f32).to_bits());
    }

    #[test]
    fn malformed_models_fail_before_device_creation() {
        let mut tree = Tree::new();
        tree.leaf_value.clear();
        assert!(PackedForest::new(&model(vec![tree])).is_err());
        let mut tree = Tree::new();
        tree.num_leaves = 2;
        assert!(PackedForest::new(&model(vec![tree])).is_err());
        let mut tree = Tree::new();
        tree.num_leaves = 2;
        tree.split_feature = vec![0];
        tree.threshold = vec![0.0];
        tree.decision_type = vec![0];
        tree.left_child = vec![0];
        tree.right_child = vec![-2];
        tree.leaf_value = vec![1.0, 2.0];
        assert!(PackedForest::new(&model(vec![tree.clone()])).is_err());
        tree.left_child[0] = -3;
        assert!(PackedForest::new(&model(vec![tree.clone()])).is_err());
        tree.left_child[0] = -1;
        tree.split_feature[0] = -1;
        assert!(PackedForest::new(&model(vec![tree.clone()])).is_err());
        tree.split_feature[0] = 0;
        tree.num_cat = 1;
        tree.decision_type[0] = 1;
        tree.cat_boundaries = vec![0, 2];
        tree.cat_threshold = vec![u32::MAX];
        assert!(PackedForest::new(&model(vec![tree.clone()])).is_err());
        tree.cat_boundaries[1] = 1;
        assert!(PackedForest::new(&model(vec![tree])).is_ok());
    }

    #[test]
    fn invalid_matrix_boundaries_are_checked_without_device() {
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
                layout: Layout::ColumnMajor,
            },
            DenseMatrix {
                data: DenseData::F64(&[]),
                rows: usize::MAX,
                cols: 2,
                layout: Layout::RowMajor,
            },
            DenseMatrix {
                data: DenseData::F64(&[]),
                rows: isize::MAX as usize,
                cols: 0,
                layout: Layout::RowMajor,
            },
        ] {
            assert!(check_matrix(&matrix, 1).is_err());
        }
        assert!(check_matrix(
            &DenseMatrix {
                data: DenseData::F32(&[]),
                rows: 100,
                cols: 0,
                layout: Layout::RowMajor
            },
            0
        )
        .is_ok());
        assert!(check_matrix(
            &DenseMatrix {
                data: DenseData::F64(&[]),
                rows: 0,
                cols: 1,
                layout: Layout::ColumnMajor
            },
            1
        )
        .is_ok());
    }
}
