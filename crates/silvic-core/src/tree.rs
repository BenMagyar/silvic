//! A regression tree in LightGBM 3.3.5's own array layout (`include/LightGBM/tree.h`), so `Tree::ToString` output
//! is a field-by-field dump.
//!
//! Splits mirror `Tree::Split`: the split leaf keeps its index as the left child, the right child becomes leaf
//! `num_leaves`, and the new internal node is `num_leaves - 1`. A child index `>= 0` is an internal node and `~leaf`
//! a leaf. The root starts with value and weight 0, which is why LightGBM writes `internal_weight=0` for node 0.

/// `decision_type` bit 0: categorical split.
pub const CATEGORICAL_MASK: i8 = 1;
/// `decision_type` bit 1: missing values go left.
pub const DEFAULT_LEFT_MASK: i8 = 2;
/// LightGBM `kZeroThreshold` (`1e-35f`, a float widened to double): smaller magnitudes are written as 0.
const ZERO_THRESHOLD: f64 = 1e-35_f32 as f64;

/// LightGBM `MissingType`, stored in `decision_type` bits 2-3.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MissingType {
    None = 0,
    Zero = 1,
    NaN = 2,
}

impl MissingType {
    pub fn from_decision_type(decision_type: i8) -> MissingType {
        match (decision_type >> 2) & 3 {
            1 => MissingType::Zero,
            2 => MissingType::NaN,
            _ => MissingType::None,
        }
    }
}

/// The statistics of the two children a split creates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SplitChildren {
    pub left_value: f64,
    pub right_value: f64,
    pub left_count: i64,
    pub right_count: i64,
    /// Hessian sums.
    pub left_weight: f64,
    pub right_weight: f64,
    pub gain: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Tree {
    pub num_leaves: usize,
    pub num_cat: usize,
    /// Input (real) feature index of each internal node.
    pub split_feature: Vec<i32>,
    pub split_gain: Vec<f32>,
    /// Numerical: the value `<=` which goes left. Categorical: index into `cat_boundaries`.
    pub threshold: Vec<f64>,
    pub decision_type: Vec<i8>,
    pub left_child: Vec<i32>,
    pub right_child: Vec<i32>,
    pub leaf_value: Vec<f64>,
    pub leaf_weight: Vec<f64>,
    pub leaf_count: Vec<i64>,
    pub internal_value: Vec<f64>,
    pub internal_weight: Vec<f64>,
    pub internal_count: Vec<i64>,
    /// `num_cat + 1` offsets into `cat_threshold`.
    pub cat_boundaries: Vec<i32>,
    /// Bitsets of category values going left, 32 per word.
    pub cat_threshold: Vec<u32>,
    pub shrinkage: f64,
    /// Training only: each leaf's parent node (`-1` for the root).
    pub leaf_parent: Vec<i32>,
    /// Training only: each leaf's depth (root 0).
    pub leaf_depth: Vec<i32>,
}

impl Default for Tree {
    fn default() -> Self {
        Tree::new()
    }
}

impl Tree {
    /// A single leaf of value 0.
    pub fn new() -> Tree {
        Tree {
            num_leaves: 1,
            num_cat: 0,
            split_feature: Vec::new(),
            split_gain: Vec::new(),
            threshold: Vec::new(),
            decision_type: Vec::new(),
            left_child: Vec::new(),
            right_child: Vec::new(),
            leaf_value: vec![0.0],
            leaf_weight: vec![0.0],
            leaf_count: vec![0],
            internal_value: Vec::new(),
            internal_weight: Vec::new(),
            internal_count: Vec::new(),
            cat_boundaries: vec![0],
            cat_threshold: Vec::new(),
            shrinkage: 1.0,
            leaf_parent: vec![-1],
            leaf_depth: vec![0],
        }
    }

    /// `Tree::Split` for a numerical feature; returns the new (right) leaf's index.
    pub fn split_numerical(
        &mut self,
        leaf: usize,
        feature: usize,
        threshold: f64,
        default_left: bool,
        missing: MissingType,
        children: &SplitChildren,
    ) -> usize {
        let mut decision_type = (missing as i8) << 2;
        if default_left {
            decision_type |= DEFAULT_LEFT_MASK;
        }
        self.split(leaf, feature, threshold, decision_type, children)
    }

    /// `Tree::SplitCategorical`: `categories` are the category values going left.
    pub fn split_categorical(
        &mut self,
        leaf: usize,
        feature: usize,
        categories: &[i32],
        missing: MissingType,
        children: &SplitChildren,
    ) -> usize {
        let bitset = to_bitset(categories);
        let cat_index = self.num_cat as f64;
        let last = *self.cat_boundaries.last().expect("cat_boundaries starts with 0");
        self.cat_boundaries.push(last + bitset.len() as i32);
        self.cat_threshold.extend_from_slice(&bitset);
        self.num_cat += 1;
        let decision_type = CATEGORICAL_MASK | ((missing as i8) << 2);
        self.split(leaf, feature, cat_index, decision_type, children)
    }

    fn split(&mut self, leaf: usize, feature: usize, threshold: f64, decision_type: i8, c: &SplitChildren) -> usize {
        let node = self.num_leaves - 1;
        let new_leaf = self.num_leaves;
        let parent = self.leaf_parent[leaf];
        if parent >= 0 {
            let parent = parent as usize;
            if self.left_child[parent] == !(leaf as i32) {
                self.left_child[parent] = node as i32;
            } else {
                self.right_child[parent] = node as i32;
            }
        }
        self.split_feature.push(feature as i32);
        self.split_gain.push(c.gain);
        self.threshold.push(threshold);
        self.decision_type.push(decision_type);
        self.left_child.push(!(leaf as i32));
        self.right_child.push(!(new_leaf as i32));
        self.internal_weight.push(self.leaf_weight[leaf]);
        self.internal_value.push(self.leaf_value[leaf]);
        self.internal_count.push(c.left_count + c.right_count);
        self.leaf_value[leaf] = nan_to_zero(c.left_value);
        self.leaf_weight[leaf] = c.left_weight;
        self.leaf_count[leaf] = c.left_count;
        self.leaf_value.push(nan_to_zero(c.right_value));
        self.leaf_weight.push(c.right_weight);
        self.leaf_count.push(c.right_count);
        self.leaf_parent[leaf] = node as i32;
        self.leaf_parent.push(node as i32);
        let depth = self.leaf_depth[leaf] + 1;
        self.leaf_depth[leaf] = depth;
        self.leaf_depth.push(depth);
        self.num_leaves += 1;
        new_leaf
    }

    /// `Tree::Shrinkage`.
    pub fn shrink(&mut self, rate: f64) {
        for value in self.leaf_value.iter_mut().chain(self.internal_value.iter_mut()) {
            *value = round_to_zero(*value * rate);
        }
        self.shrinkage *= rate;
    }

    /// `Tree::AddBias`: folds the initial score into the first tree, whose shrinkage LightGBM then writes as 1.
    pub fn add_bias(&mut self, bias: f64) {
        for value in self.leaf_value.iter_mut().chain(self.internal_value.iter_mut()) {
            *value = round_to_zero(*value + bias);
        }
        self.shrinkage = 1.0;
    }

    /// `Tree::AsConstantTree`.
    pub fn as_constant(&mut self, value: f64) {
        *self = Tree::new();
        self.leaf_value[0] = value;
    }

    /// `Tree::SetLeafOutput`.
    pub fn set_leaf_output(&mut self, leaf: usize, value: f64) {
        self.leaf_value[leaf] = round_to_zero(value);
    }

    /// Depth of the deepest leaf.
    pub fn max_depth(&self) -> i32 {
        self.leaf_depth.iter().copied().max().unwrap_or(0)
    }
}

fn nan_to_zero(value: f64) -> f64 {
    if value.is_nan() {
        0.0
    } else {
        value
    }
}

/// LightGBM `MaybeRoundToZero`.
fn round_to_zero(value: f64) -> f64 {
    if value.abs() > ZERO_THRESHOLD {
        value
    } else {
        0.0
    }
}

/// LightGBM `Common::ConstructBitset`: one bit per non-negative value, as few words as the largest value needs.
pub fn to_bitset(values: &[i32]) -> Vec<u32> {
    let max = values.iter().copied().filter(|&v| v >= 0).max().unwrap_or(0) as usize;
    let mut words = vec![0u32; max / 32 + 1];
    for &value in values.iter().filter(|&&v| v >= 0) {
        words[value as usize / 32] |= 1 << (value as usize % 32);
    }
    words
}

/// LightGBM `Common::FindInBitset`.
pub fn in_bitset(words: &[u32], value: i32) -> bool {
    if value < 0 {
        return false;
    }
    let word = value as usize / 32;
    word < words.len() && (words[word] >> (value as usize % 32)) & 1 == 1
}

#[cfg(test)]
mod tests {
    use super::*;

    fn children(gain: f32) -> SplitChildren {
        SplitChildren {
            left_value: -1.0,
            right_value: 1.0,
            left_count: 10,
            right_count: 20,
            left_weight: 10.0,
            right_weight: 20.0,
            gain,
        }
    }

    #[test]
    fn splits_follow_lightgbm_node_numbering() {
        let mut tree = Tree::new();
        let right = tree.split_numerical(0, 3, 0.5, true, MissingType::None, &children(5.0));
        assert_eq!(right, 1);
        let right2 = tree.split_numerical(1, 1, 2.0, false, MissingType::NaN, &children(2.0));
        assert_eq!(right2, 2);
        tree.split_numerical(0, 0, -1.0, true, MissingType::None, &children(1.0));
        // Node 0 splits leaf 0 (left) / leaf 1 (right); node 1 replaces leaf 1; node 2 replaces leaf 0.
        assert_eq!(tree.left_child, vec![2, -2, -1]);
        assert_eq!(tree.right_child, vec![1, -3, -4]);
        assert_eq!(tree.decision_type, vec![2, 8, 2]);
        assert_eq!(tree.internal_weight[0], 0.0);
        assert_eq!(tree.leaf_depth, vec![2, 2, 2, 2]);
    }

    #[test]
    fn categorical_split_records_bitset() {
        let mut tree = Tree::new();
        tree.split_categorical(0, 2, &[1, 3, 33], MissingType::None, &children(1.0));
        assert_eq!(tree.cat_boundaries, vec![0, 2]);
        assert_eq!(tree.cat_threshold, vec![0b1010, 0b10]);
        assert_eq!(tree.threshold, vec![0.0]);
        assert_eq!(tree.decision_type, vec![1]);
        assert!(in_bitset(&tree.cat_threshold, 33));
        assert!(!in_bitset(&tree.cat_threshold, 2));
        assert!(!in_bitset(&tree.cat_threshold, -1));
    }

    #[test]
    fn bias_resets_shrinkage() {
        let mut tree = Tree::new();
        tree.split_numerical(0, 0, 0.0, true, MissingType::None, &children(1.0));
        tree.shrink(0.1);
        tree.add_bias(2.0);
        assert_eq!(tree.shrinkage, 1.0);
        assert_eq!(tree.leaf_value, vec![-0.1 + 2.0, 0.1 + 2.0]);
        assert_eq!(tree.internal_value, vec![2.0]);
    }
}
