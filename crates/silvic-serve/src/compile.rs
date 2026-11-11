//! Compiles a [`Model`] into the [`Forest`] the block kernels score.
//!
//! Each tree's leaves are numbered left to right (QuickScorer's order), so that after clearing the left-subtree
//! leaves of every node a row goes right at, the lowest set bit of the leaf bitvector is the leaf `Tree::GetLeaf`
//! reaches. A node becomes one op per leaf-bitvector byte its left subtree covers.
//!
//! Rows reach the kernels as one byte per slot. A numeric slot is a (feature, missing handling, threshold range)
//! triple: a value's code is its rank among the feature's distinct split thresholds (how many lie strictly below it),
//! taken relative to the slot's range of 254 thresholds and clamped to `0..=255`, so `value <= threshold` becomes
//! `code <= local rank` and codes 0 and 255 send a row left and right at every node of the slot. Missing values take
//! those two codes where `Tree::NumericalDecision` sends them a fixed way (`MissingType::NaN` and `Zero`), and the
//! rank of 0 where it reads NaN as 0 (`MissingType::None`). A categorical slot's code is the category's class: the
//! categories every categorical split on the feature treats alike share one, and the classes a split sends left
//! form its 64-byte table. NaN, negative and unseen categories get no class and go right, as
//! `Tree::CategoricalDecision` sends them.
//!
//! Trees with more than 256 leaves, a NaN threshold, or a categorical split on a feature with more than 64 classes
//! keep their [`Tree`] and are walked row by row with `silvic_core`'s own decisions.

use std::collections::HashMap;

use silvic_core::model::Model;
use silvic_core::tree::{in_bitset, MissingType, Tree, CATEGORICAL_MASK, DEFAULT_LEFT_MASK};
use silvic_core::{Result, SilvicError};

/// Leaf-bitvector bytes per tree: 256 leaves, so a leaf's position fits a byte.
pub(crate) const MAX_LEAF_BYTES: usize = 32;
/// Leaf values a fast tree's offset starts in [`Forest::leaf_values`]: one per possible leaf position.
pub(crate) const LEAF_WINDOW: usize = MAX_LEAF_BYTES * 8;
/// Thresholds per numeric slot: local ranks stay below 254, so codes 0 and 255 compare left and right of all.
const SLOT_THRESHOLDS: u32 = 254;
const ALWAYS_LEFT: u8 = 0;
const ALWAYS_RIGHT: u8 = 255;
/// Classes per categorical slot: a split's classes fit one 64-byte table lookup.
pub(crate) const MAX_CLASSES: usize = 64;
/// The code of a category no split sends left; past every table, so lookups return "not left".
const NO_CLASS: u8 = 255;
/// LightGBM `kZeroThreshold` (`1e-35f` widened to double): the dense row reader drops smaller magnitudes.
const ZERO_THRESHOLD: f64 = 1e-35_f32 as f64;

/// In the lanes whose code at `slot` is above `threshold` (the row goes right), clears every bit of leaf byte `byte`
/// outside `keep` (the node's left-subtree leaves there). `threshold` and `keep` are adjacent so NEON broadcasts
/// both with one `ld2r`.
#[derive(Debug, Clone, Copy)]
#[repr(C, align(8))]
pub(crate) struct NumericOp {
    slot: u32,
    pub threshold_keep: [u8; 2],
    byte: u8,
}

impl NumericOp {
    fn new(slot: u32, threshold: u8, byte: u8, keep: u8) -> NumericOp {
        NumericOp {
            slot,
            threshold_keep: [threshold, keep],
            byte,
        }
    }

    #[inline(always)]
    pub(crate) fn slot(self) -> usize {
        self.slot as usize
    }

    /// The threshold's local rank in its slot, below 254.
    #[inline(always)]
    pub(crate) fn threshold(self) -> u8 {
        self.threshold_keep[0]
    }

    /// Below [`MAX_LEAF_BYTES`].
    #[inline(always)]
    pub(crate) fn byte(self) -> usize {
        usize::from(self.byte) & (MAX_LEAF_BYTES - 1)
    }

    #[inline(always)]
    pub(crate) fn keep(self) -> u8 {
        self.threshold_keep[1]
    }
}

/// In the lanes whose class at `slot` table `table` does not send left, clears every bit of leaf byte `byte`
/// outside `keep`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct CategoricalOp {
    pub slot: u32,
    pub table: u32,
    byte: u8,
    pub keep: u8,
}

impl CategoricalOp {
    /// Below [`MAX_LEAF_BYTES`].
    #[inline(always)]
    pub(crate) fn byte(self) -> usize {
        usize::from(self.byte) & (MAX_LEAF_BYTES - 1)
    }
}

/// A tree the kernels score with leaf bitvectors.
#[derive(Debug, Clone, Copy)]
pub(crate) struct FastTree {
    pub numeric_start: u32,
    pub numeric_end: u32,
    pub categorical_start: u32,
    pub categorical_end: u32,
    pub leaf_bytes: u32,
    /// Offset of the tree's leaf values in left-to-right order, `8 * leaf_bytes` of them (zero past the last leaf).
    pub leaves: u32,
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum Plan {
    Fast(FastTree),
    /// Index into [`Forest::walked`].
    Walk(u32),
}

#[derive(Debug, Clone)]
pub(crate) struct NumericSlot {
    pub slot: u32,
    /// Rank of the slot's first threshold among the feature's.
    pub base: u32,
    /// Code of NaN; `None` ranks it as 0.
    pub nan: Option<u8>,
    /// Code of 0 (any magnitude up to `kZeroThreshold`); `None` ranks it.
    pub zero: Option<u8>,
}

#[derive(Debug, Clone)]
pub(crate) struct NumericFeature {
    pub feature: usize,
    /// Distinct split thresholds, ascending, padded with +inf to a power-of-two length.
    pub thresholds: Vec<f64>,
    pub slots: Vec<NumericSlot>,
}

#[derive(Debug, Clone)]
pub(crate) struct CategoricalFeature {
    pub feature: usize,
    pub slot: u32,
    /// Class of each category, `NO_CLASS` for categories no split sends left.
    pub classes: Vec<u8>,
}

/// A model laid out for the block kernels.
#[derive(Debug, Clone)]
pub(crate) struct Forest {
    pub num_features: usize,
    pub plans: Vec<Plan>,
    pub numeric_ops: Vec<NumericOp>,
    pub categorical_ops: Vec<CategoricalOp>,
    /// Per class: 0xFF when the split sends it left.
    pub tables: Vec<[u8; MAX_CLASSES]>,
    pub leaf_values: Vec<f64>,
    pub walked: Vec<Tree>,
    pub numeric_features: Vec<NumericFeature>,
    pub categorical_features: Vec<CategoricalFeature>,
    pub slots: usize,
}

/// How a value reads at a numeric slot before ranking.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ValueKind {
    Nan,
    Zero,
    Other,
}

impl NumericSlot {
    #[inline]
    fn code(&self, rank: u32, kind: ValueKind) -> u8 {
        let special = match kind {
            ValueKind::Nan => self.nan,
            ValueKind::Zero => self.zero,
            ValueKind::Other => None,
        };
        special.unwrap_or_else(|| rank.saturating_sub(self.base).min(u32::from(u8::MAX)) as u8)
    }
}

/// The dense row reader's view of a numeric value, with its kind: magnitudes up to `kZeroThreshold` read as 0, and
/// NaN ranks as 0 where `Tree::NumericalDecision` does not treat it as missing.
#[inline]
fn read_numeric(value: f64) -> (f64, ValueKind) {
    if value.is_nan() {
        (0.0, ValueKind::Nan)
    } else if value.abs() <= ZERO_THRESHOLD {
        (0.0, ValueKind::Zero)
    } else {
        (value, ValueKind::Other)
    }
}

impl NumericFeature {
    /// The ranks of `values` among the thresholds (how many lie strictly below). Every lane runs the same
    /// branchless binary search in lockstep, so the searches overlap instead of waiting on mispredicted branches; the
    /// +inf padding to a power of two counts for no value and keeps every probe inside the slice.
    #[inline]
    fn ranks<const LANES: usize>(&self, values: &[f64; LANES], ranks: &mut [u32; LANES]) {
        let mask = self.thresholds.len() - 1;
        let thresholds = &self.thresholds[..=mask];
        ranks.fill(0);
        let mut half = thresholds.len() / 2;
        while half > 0 {
            for (rank, &value) in ranks.iter_mut().zip(values) {
                *rank += half as u32 * u32::from(thresholds[(*rank as usize + half) & mask] < value);
            }
            half /= 2;
        }
        for (rank, &value) in ranks.iter_mut().zip(values) {
            *rank += u32::from(thresholds[*rank as usize & mask] < value);
        }
    }
}

impl CategoricalFeature {
    /// `Tree::CategoricalDecision`'s category: NaN and negative values (after the truncating cast) have none.
    #[inline]
    fn class(&self, value: f64) -> u8 {
        let value = if value.abs() <= ZERO_THRESHOLD { 0.0 } else { value };
        if value.is_nan() {
            return NO_CLASS;
        }
        usize::try_from(value as i32)
            .ok()
            .and_then(|category| self.classes.get(category).copied())
            .unwrap_or(NO_CLASS)
    }
}

impl Forest {
    pub(crate) fn compile(model: &Model) -> Result<Forest> {
        let num_features = model.num_features();
        let mut shapes = Vec::with_capacity(model.trees.len());
        for (index, tree) in model.trees.iter().enumerate() {
            let shape = Shape::of(tree, num_features).map_err(|m| SilvicError::model(format!("tree {index}: {m}")))?;
            shapes.push(shape);
        }
        let classes = classify_categories(model, &shapes);
        let fast: Vec<bool> = model
            .trees
            .iter()
            .zip(&shapes)
            .map(|(tree, shape)| fits_bitvectors(tree, shape, &classes))
            .collect();
        let mut builder = Builder::new(num_features, thresholds_by_feature(model, &shapes, &fast), classes);
        for ((tree, shape), fast) in model.trees.iter().zip(&shapes).zip(fast) {
            if fast {
                builder.add_fast(tree, shape);
            } else {
                builder.add_walked(tree);
            }
        }
        Ok(builder.finish())
    }

    /// Writes the codes of `LANES` rows of `num_features` values (row after row in `rows`) to `codes`, one array per
    /// slot, each XOR-ed with `bias`.
    pub(crate) fn encode<const LANES: usize>(&self, rows: &[f64], codes: &mut [[u8; LANES]], bias: u8) {
        let width = self.num_features;
        let mut values = [0.0f64; LANES];
        let mut kinds = [ValueKind::Other; LANES];
        let mut ranks = [0u32; LANES];
        for feature in &self.numeric_features {
            for (lane, (value, kind)) in values.iter_mut().zip(&mut kinds).enumerate() {
                (*value, *kind) = read_numeric(rows[lane * width + feature.feature]);
            }
            feature.ranks(&values, &mut ranks);
            for slot in &feature.slots {
                for ((code, &rank), &kind) in codes[slot.slot as usize].iter_mut().zip(&ranks).zip(&kinds) {
                    *code = slot.code(rank, kind) ^ bias;
                }
            }
        }
        for feature in &self.categorical_features {
            let codes = &mut codes[feature.slot as usize];
            for (lane, code) in codes.iter_mut().enumerate() {
                *code = feature.class(rows[lane * width + feature.feature]) ^ bias;
            }
        }
    }
}

fn is_categorical(tree: &Tree, node: usize) -> bool {
    tree.num_cat > 0 && tree.decision_type[node] & CATEGORICAL_MASK != 0
}

/// The categorical bitset of node `node`.
fn bitset(tree: &Tree, node: usize) -> &[u32] {
    let index = tree.threshold[node] as usize;
    let start = tree.cat_boundaries[index] as usize;
    let end = tree.cat_boundaries[index + 1] as usize;
    &tree.cat_threshold[start..end]
}

/// A tree's reachable nodes in left-to-right leaf order, checked so every later index is in bounds.
#[derive(Debug, Clone, Default)]
struct Shape {
    /// Leaf indices, left to right.
    leaves: Vec<usize>,
    /// Reachable internal nodes with the left-to-right positions `[start, end)` of their left subtree's leaves.
    nodes: Vec<(usize, u32, u32)>,
}

impl Shape {
    /// Walks the tree from node 0, failing on anything `Tree::GetLeaf` would index out of bounds or loop on.
    fn of(tree: &Tree, num_features: usize) -> std::result::Result<Shape, String> {
        if tree.leaf_value.is_empty() {
            return Err("no leaf values".into());
        }
        if tree.num_leaves <= 1 {
            return Ok(Shape {
                leaves: vec![0],
                nodes: Vec::new(),
            });
        }
        let nodes = tree.num_leaves - 1;
        let arrays = [
            tree.split_feature.len(),
            tree.threshold.len(),
            tree.decision_type.len(),
            tree.left_child.len(),
            tree.right_child.len(),
        ];
        if arrays.iter().any(|&len| len < nodes) || tree.leaf_value.len() < tree.num_leaves {
            return Err(format!("node arrays shorter than num_leaves={}", tree.num_leaves));
        }
        let mut shape = Shape::default();
        let mut seen = vec![false; nodes];
        shape.visit(tree, num_features, 0, &mut seen)?;
        Ok(shape)
    }

    /// Lays out the subtree under `child` (a node index, or `~leaf`); returns its leaves' positions.
    fn visit(
        &mut self,
        tree: &Tree,
        num_features: usize,
        child: i32,
        seen: &mut [bool],
    ) -> std::result::Result<(u32, u32), String> {
        if child < 0 {
            let leaf = !child as usize;
            if leaf >= tree.num_leaves {
                return Err(format!("child {child} names leaf {leaf} of {}", tree.num_leaves));
            }
            let position = self.leaves.len() as u32;
            self.leaves.push(leaf);
            return Ok((position, position + 1));
        }
        let node = child as usize;
        if node >= seen.len() || seen[node] {
            return Err(format!("node {node} is out of range or reached twice"));
        }
        seen[node] = true;
        check_node(tree, num_features, node)?;
        let slot = self.nodes.len();
        self.nodes.push((node, 0, 0));
        let (start, left_end) = self.visit(tree, num_features, tree.left_child[node], seen)?;
        let (_, end) = self.visit(tree, num_features, tree.right_child[node], seen)?;
        self.nodes[slot] = (node, start, left_end);
        Ok((start, end))
    }
}

fn check_node(tree: &Tree, num_features: usize, node: usize) -> std::result::Result<(), String> {
    let feature = tree.split_feature[node];
    if usize::try_from(feature).map_or(true, |f| f >= num_features) {
        return Err(format!("node {node} splits on feature {feature} of {num_features}"));
    }
    if !is_categorical(tree, node) {
        return Ok(());
    }
    let threshold = tree.threshold[node];
    let index = threshold as usize;
    let bounds = (threshold >= 0.0 && index < tree.num_cat && index + 1 < tree.cat_boundaries.len())
        .then(|| (tree.cat_boundaries[index], tree.cat_boundaries[index + 1]));
    match bounds {
        Some((start, end)) if 0 <= start && start <= end && end as usize <= tree.cat_threshold.len() => Ok(()),
        _ => Err(format!("node {node} names categorical bitset {threshold}")),
    }
}

/// Per feature, the classes of categories its categorical splits tell apart, or `None` past [`MAX_CLASSES`].
/// Categories in one class go the same way at every split; a category no split sends left has no class.
fn classify_categories(model: &Model, shapes: &[Shape]) -> Vec<Option<Classes>> {
    let mut splits: Vec<Vec<&[u32]>> = vec![Vec::new(); model.num_features()];
    for (tree, shape) in model.trees.iter().zip(shapes) {
        if shape.leaves.len() > MAX_LEAF_BYTES * 8 {
            continue;
        }
        for &(node, _, _) in &shape.nodes {
            if is_categorical(tree, node) {
                splits[tree.split_feature[node] as usize].push(bitset(tree, node));
            }
        }
    }
    splits.iter().map(|bitsets| Classes::of(bitsets)).collect()
}

#[derive(Debug, Clone)]
struct Classes {
    /// Class of each category (`NO_CLASS` when no split sends it left).
    of_category: Vec<u8>,
    /// One category per class.
    representative: Vec<i32>,
}

impl Classes {
    fn of(bitsets: &[&[u32]]) -> Option<Classes> {
        let categories = bitsets.iter().map(|words| words.len() * 32).max().unwrap_or(0);
        let mut classes = Classes {
            of_category: vec![NO_CLASS; categories],
            representative: Vec::new(),
        };
        let mut by_signature: HashMap<Vec<u64>, u8> = HashMap::new();
        for category in 0..categories {
            let value = category as i32;
            let mut signature = vec![0u64; bitsets.len().div_ceil(64)];
            for (split, words) in bitsets.iter().enumerate() {
                if in_bitset(words, value) {
                    signature[split / 64] |= 1 << (split % 64);
                }
            }
            if signature.iter().all(|&word| word == 0) {
                continue;
            }
            let next = classes.representative.len();
            let class = *by_signature.entry(signature).or_insert(next as u8);
            if usize::from(class) == next {
                if next == MAX_CLASSES {
                    return None;
                }
                classes.representative.push(value);
            }
            classes.of_category[category] = class;
        }
        Some(classes)
    }
}

/// Whether the leaf-bitvector kernels can score the tree.
fn fits_bitvectors(tree: &Tree, shape: &Shape, classes: &[Option<Classes>]) -> bool {
    shape.leaves.len() <= MAX_LEAF_BYTES * 8
        && shape.nodes.iter().all(|&(node, _, _)| {
            if is_categorical(tree, node) {
                classes[tree.split_feature[node] as usize].is_some()
            } else {
                !tree.threshold[node].is_nan()
            }
        })
}

/// Distinct numeric thresholds per feature over the bitvector trees, ascending (0 and -0 are one threshold: they
/// compare equal).
fn thresholds_by_feature(model: &Model, shapes: &[Shape], fast: &[bool]) -> Vec<Vec<f64>> {
    let mut thresholds: Vec<Vec<f64>> = vec![Vec::new(); model.num_features()];
    for ((tree, shape), _) in model.trees.iter().zip(shapes).zip(fast).filter(|(_, &fast)| fast) {
        for &(node, _, _) in &shape.nodes {
            if !is_categorical(tree, node) {
                thresholds[tree.split_feature[node] as usize].push(tree.threshold[node]);
            }
        }
    }
    for values in &mut thresholds {
        values.sort_by(|a, b| a.partial_cmp(b).expect("NaN thresholds are walked"));
        values.dedup_by(|a, b| a == b);
    }
    thresholds
}

/// `(feature, NaN code, zero code, threshold range)`: one numeric slot.
type NumericKey = (usize, Option<u8>, Option<u8>, u32);

struct Builder {
    forest: Forest,
    thresholds: Vec<Vec<f64>>,
    classes: Vec<Option<Classes>>,
    numeric_slots: HashMap<NumericKey, u32>,
    categorical_slots: HashMap<usize, u32>,
    tables: HashMap<[u8; MAX_CLASSES], u32>,
}

impl Builder {
    fn new(num_features: usize, thresholds: Vec<Vec<f64>>, classes: Vec<Option<Classes>>) -> Builder {
        Builder {
            forest: Forest {
                num_features,
                plans: Vec::new(),
                numeric_ops: Vec::new(),
                categorical_ops: Vec::new(),
                tables: Vec::new(),
                leaf_values: Vec::new(),
                walked: Vec::new(),
                numeric_features: Vec::new(),
                categorical_features: Vec::new(),
                slots: 0,
            },
            thresholds,
            classes,
            numeric_slots: HashMap::new(),
            categorical_slots: HashMap::new(),
            tables: HashMap::new(),
        }
    }

    fn add_walked(&mut self, tree: &Tree) {
        self.forest.plans.push(Plan::Walk(self.forest.walked.len() as u32));
        self.forest.walked.push(tree.clone());
    }

    fn add_fast(&mut self, tree: &Tree, shape: &Shape) {
        let leaf_bytes = shape.leaves.len().div_ceil(8);
        let leaves = self.forest.leaf_values.len() as u32;
        self.forest
            .leaf_values
            .extend(shape.leaves.iter().map(|&leaf| tree.leaf_value[leaf]));
        self.forest.leaf_values.resize(leaves as usize + 8 * leaf_bytes, 0.0);
        let numeric_start = self.forest.numeric_ops.len() as u32;
        let categorical_start = self.forest.categorical_ops.len() as u32;
        for &(node, start, end) in &shape.nodes {
            let feature = tree.split_feature[node] as usize;
            if is_categorical(tree, node) {
                let slot = self.categorical_slot(feature);
                let table = self.table(feature, bitset(tree, node));
                for (byte, clear) in clear_bytes(start, end) {
                    self.forest.categorical_ops.push(CategoricalOp {
                        slot,
                        table,
                        byte,
                        keep: !clear,
                    });
                }
            } else {
                let (slot, threshold) = self.numeric_slot(feature, tree.threshold[node], tree.decision_type[node]);
                for (byte, clear) in clear_bytes(start, end) {
                    self.forest
                        .numeric_ops
                        .push(NumericOp::new(slot, threshold, byte, !clear));
                }
            }
        }
        interleave_bytes(&mut self.forest.numeric_ops[numeric_start as usize..], |op| op.byte());
        interleave_bytes(&mut self.forest.categorical_ops[categorical_start as usize..], |op| {
            op.byte()
        });
        self.forest.plans.push(Plan::Fast(FastTree {
            numeric_start,
            numeric_end: self.forest.numeric_ops.len() as u32,
            categorical_start,
            categorical_end: self.forest.categorical_ops.len() as u32,
            leaf_bytes: leaf_bytes as u32,
            leaves,
        }));
    }

    /// The slot and local rank of a numeric split. `Tree::NumericalDecision` sends missing values (NaN for
    /// `MissingType::NaN`, NaN and 0 for `Zero`) the default way; `MissingType::None` reads NaN as 0.
    fn numeric_slot(&mut self, feature: usize, threshold: f64, decision_type: i8) -> (u32, u8) {
        let rank = self.thresholds[feature].partition_point(|&t| t < threshold) as u32;
        let default = if decision_type & DEFAULT_LEFT_MASK != 0 {
            ALWAYS_LEFT
        } else {
            ALWAYS_RIGHT
        };
        let (nan, zero) = match MissingType::from_decision_type(decision_type) {
            MissingType::None => (None, None),
            MissingType::NaN => (Some(default), None),
            MissingType::Zero => (Some(default), Some(default)),
        };
        let range = rank / SLOT_THRESHOLDS;
        let key = (feature, nan, zero, range);
        let slot = match self.numeric_slots.get(&key) {
            Some(&slot) => slot,
            None => {
                let slot = self.new_slot();
                self.numeric_slots.insert(key, slot);
                self.numeric_feature(feature).slots.push(NumericSlot {
                    slot,
                    base: range * SLOT_THRESHOLDS,
                    nan,
                    zero,
                });
                slot
            }
        };
        (slot, (rank % SLOT_THRESHOLDS) as u8)
    }

    fn numeric_feature(&mut self, feature: usize) -> &mut NumericFeature {
        let features = &mut self.forest.numeric_features;
        let index = match features.iter().position(|f| f.feature == feature) {
            Some(index) => index,
            None => {
                let mut thresholds = self.thresholds[feature].clone();
                thresholds.resize(thresholds.len().next_power_of_two(), f64::INFINITY);
                features.push(NumericFeature {
                    feature,
                    thresholds,
                    slots: Vec::new(),
                });
                features.len() - 1
            }
        };
        &mut features[index]
    }

    fn categorical_slot(&mut self, feature: usize) -> u32 {
        if let Some(&slot) = self.categorical_slots.get(&feature) {
            return slot;
        }
        let slot = self.new_slot();
        self.categorical_slots.insert(feature, slot);
        let classes = self.classes[feature].as_ref().expect("checked by fits_bitvectors");
        self.forest.categorical_features.push(CategoricalFeature {
            feature,
            slot,
            classes: classes.of_category.clone(),
        });
        slot
    }

    /// The table of classes `words` sends left.
    fn table(&mut self, feature: usize, words: &[u32]) -> u32 {
        let classes = self.classes[feature].as_ref().expect("checked by fits_bitvectors");
        let mut table = [0u8; MAX_CLASSES];
        for (entry, &category) in table.iter_mut().zip(&classes.representative) {
            *entry = if in_bitset(words, category) { 0xFF } else { 0 };
        }
        let next = self.forest.tables.len() as u32;
        let index = *self.tables.entry(table).or_insert(next);
        if index == next {
            self.forest.tables.push(table);
        }
        index
    }

    fn new_slot(&mut self) -> u32 {
        self.forest.slots += 1;
        (self.forest.slots - 1) as u32
    }

    fn finish(mut self) -> Forest {
        let padded = self.forest.leaf_values.len() + LEAF_WINDOW;
        self.forest.leaf_values.resize(padded, 0.0);
        self.forest
    }
}

/// Reorders a tree's ops so consecutive ops touch different leaf bytes where possible: the `k`-th op of every byte,
/// then the `k + 1`-th. Ops on one byte form a chain through memory, and interleaving lets the chains overlap.
fn interleave_bytes<T: Copy>(ops: &mut [T], byte: impl Fn(T) -> usize) {
    let mut seen = [0u32; MAX_LEAF_BYTES];
    let mut keyed: Vec<(u32, usize, T)> = ops
        .iter()
        .map(|&op| {
            let b = byte(op);
            seen[b] += 1;
            (seen[b], b, op)
        })
        .collect();
    keyed.sort_by_key(|&(round, b, _)| (round, b));
    for (slot, (_, _, op)) in ops.iter_mut().zip(keyed) {
        *slot = op;
    }
}

/// The leaf-bitvector bytes covering positions `[start, end)`, each with the bits to clear.
fn clear_bytes(start: u32, end: u32) -> impl Iterator<Item = (u8, u8)> {
    (start / 8..end.div_ceil(8)).map(move |byte| {
        let low = start.max(byte * 8) - byte * 8;
        let high = end.min(byte * 8 + 8) - byte * 8;
        let bits = ((1u32 << high) - (1u32 << low)) as u8;
        (byte as u8, bits)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clear_bytes_cover_the_range() {
        assert_eq!(clear_bytes(0, 1).collect::<Vec<_>>(), [(0, 0b1)]);
        assert_eq!(clear_bytes(3, 8).collect::<Vec<_>>(), [(0, 0b1111_1000)]);
        assert_eq!(
            clear_bytes(6, 18).collect::<Vec<_>>(),
            [(0, 0b1100_0000), (1, 0xFF), (2, 0b11)]
        );
        assert_eq!(clear_bytes(248, 256).collect::<Vec<_>>(), [(31, 0xFF)]);
    }

    #[test]
    fn slot_codes_clamp_to_the_slot_range() {
        let slot = NumericSlot {
            slot: 0,
            base: 254,
            nan: None,
            zero: Some(ALWAYS_RIGHT),
        };
        assert_eq!(slot.code(10, ValueKind::Other), 0);
        assert_eq!(slot.code(254 + 253, ValueKind::Other), 253);
        assert_eq!(slot.code(254 + 600, ValueKind::Other), 255);
        assert_eq!(slot.code(300, ValueKind::Nan), 46);
        assert_eq!(slot.code(300, ValueKind::Zero), ALWAYS_RIGHT);
    }

    #[test]
    fn categories_alike_at_every_split_share_a_class() {
        let left_a = [0b0110u32];
        let left_b = [0b0010u32, 0b1];
        let classes = Classes::of(&[&left_a, &left_b]).expect("few classes");
        assert_eq!(classes.of_category[0], NO_CLASS);
        assert_eq!(classes.of_category[1], 0);
        assert_eq!(classes.of_category[2], 1);
        assert_eq!(classes.of_category[32], 2);
        assert_eq!(classes.of_category[3], NO_CLASS);
        assert_eq!(classes.representative, [1, 2, 32]);
    }

    #[test]
    fn too_many_classes_are_refused() {
        let words: Vec<Vec<u32>> = (0..=MAX_CLASSES).map(|c| vec![1u32 << (c % 32); c / 32 + 1]).collect();
        let bitsets: Vec<&[u32]> = words.iter().map(Vec::as_slice).collect();
        assert!(Classes::of(&bitsets).is_none());
        assert!(Classes::of(&bitsets[..MAX_CLASSES]).is_some());
    }
}
