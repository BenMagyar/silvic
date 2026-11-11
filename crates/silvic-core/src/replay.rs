//! Continued training's starting scores: an existing model's trees replayed on binned rows, in the form
//! `Backend::add_tree` applies, so every score is the same chain of f32 adds the original training made.
//!
//! Each node's split becomes the bins of the dataset's bin mapper that LightGBM 3.3.5's `Tree::NumericalDecision`
//! and `Tree::CategoricalDecision` send left. For a model silvic trained on these bin mappers this is exact: a
//! numerical threshold is a bin's upper bound (or `f64::MAX` for NaN against everything else) and categorical
//! bitsets list bin categories. For a model trained elsewhere (LightGBM's own bins) a threshold can fall inside a
//! bin, which then goes wholly to one side, so the replayed scores only approximate the model's predictions; a
//! split on a feature that is trivial in these rows sends every row right.

use crate::backend::{BinSplit, DeviceNode, DeviceTree};
use crate::data::{BinMapper, BinnedDataset, FeatureMeta};
use crate::error::{Result, SilvicError};
use crate::tree::{in_bitset, MissingType, Tree, CATEGORICAL_MASK, DEFAULT_LEFT_MASK};

/// `tree` in device form over `data`'s bins, with `bias` taken off every leaf: the first tree of a model holds the
/// initial score, which the scores already start from.
pub(crate) fn device_tree(tree: &Tree, data: &BinnedDataset, bias: f64) -> Result<DeviceTree> {
    let nodes = (0..tree.num_leaves.saturating_sub(1))
        .map(|node| {
            Ok(DeviceNode {
                split: node_split(tree, node, data)?,
                left: tree.left_child[node],
                right: tree.right_child[node],
            })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(DeviceTree {
        nodes,
        leaf_values: leaf_values(tree, bias),
    })
}

/// The f32 leaves a tree adds to scores that already hold `bias`. Training computes the first tree's leaves this way
/// too, from the model's leaf values with the bias folded in, rather than from the leaves before the bias: the f64
/// round trip `(leaf + bias) - bias` does not always round to the same f32, and this way a replay adds exactly what
/// training added.
pub(crate) fn leaf_values(tree: &Tree, bias: f64) -> Vec<f32> {
    tree.leaf_value.iter().map(|&v| (v - bias) as f32).collect()
}

fn node_split(tree: &Tree, node: usize, data: &BinnedDataset) -> Result<BinSplit> {
    let feature = usize::try_from(tree.split_feature[node])
        .ok()
        .filter(|&f| f < data.features.len())
        .ok_or_else(|| {
            SilvicError::config(format!(
                "init_model splits on feature {}, the training data has {}",
                tree.split_feature[node],
                data.features.len()
            ))
        })?;
    let meta = &data.features[feature];
    let decision_type = tree.decision_type[node];
    let left_bins = match &meta.bins {
        BinMapper::Numerical {
            upper_bounds, nan_bin, ..
        } if decision_type & CATEGORICAL_MASK == 0 => {
            numerical_bins(meta, upper_bounds, *nan_bin, tree.threshold[node], decision_type)
        }
        BinMapper::Categorical { categories } if decision_type & CATEGORICAL_MASK != 0 => {
            categorical_bins(tree, node, categories)?
        }
        BinMapper::Trivial => {
            // No row of a trivial feature has a bin; the split needs some valid column to read.
            if data.groups.is_empty() {
                return Err(SilvicError::config(
                    "init_model splits on features the training data has none of",
                ));
            }
            return Ok(BinSplit {
                group: 0,
                column: 0,
                left_bins: [0; 8],
            });
        }
        _ => {
            return Err(SilvicError::config(format!(
                "init_model splits feature {feature} as {}, the training data bins it as {}",
                if decision_type & CATEGORICAL_MASK != 0 {
                    "categorical"
                } else {
                    "numerical"
                },
                if meta.is_categorical() {
                    "categorical"
                } else {
                    "numerical"
                }
            )))
        }
    };
    Ok(BinSplit {
        group: meta.group,
        column: meta.column,
        left_bins,
    })
}

fn set(bits: &mut [u32; 8], bin: usize, left: bool) {
    let mask = 1 << (bin % 32);
    if left {
        bits[bin / 32] |= mask;
    } else {
        bits[bin / 32] &= !mask;
    }
}

/// `Tree::NumericalDecision`: a value goes left when `value <= threshold`, missing values by `default_left`. A bin
/// goes left when its upper bound does (the last bound, `+inf`, stands for the largest finite value, which a
/// NaN-against-the-rest threshold of `f64::MAX` sends left). NaN is a missing value under `MissingType::NaN`, and
/// 0.0 otherwise; under `MissingType::Zero` the bin holding 0.0 is missing too.
fn numerical_bins(
    meta: &FeatureMeta,
    upper_bounds: &[f64],
    nan_bin: bool,
    threshold: f64,
    decision_type: i8,
) -> [u32; 8] {
    let default_left = decision_type & DEFAULT_LEFT_MASK != 0;
    let mut bits = [0u32; 8];
    for (bin, &bound) in upper_bounds.iter().enumerate() {
        set(&mut bits, bin, bound.min(f64::MAX) <= threshold);
    }
    let nan = upper_bounds.len();
    match MissingType::from_decision_type(decision_type) {
        MissingType::NaN => {
            if nan_bin {
                set(&mut bits, nan, default_left);
            }
        }
        MissingType::Zero => {
            set(&mut bits, usize::from(meta.bins.value_to_bin(0.0)), default_left);
            if nan_bin {
                set(&mut bits, nan, default_left);
            }
        }
        MissingType::None => {
            if nan_bin {
                set(&mut bits, nan, 0.0 <= threshold);
            }
        }
    }
    bits
}

/// `Tree::CategoricalDecision`: categories in the node's bitset go left. Bin 0 (negative, NaN, unseen and rare
/// categories) goes right.
fn categorical_bins(tree: &Tree, node: usize, categories: &[i32]) -> Result<[u32; 8]> {
    let index = tree.threshold[node] as usize;
    let malformed = || SilvicError::config(format!("init_model node {node} has no category bitset"));
    let begin = *tree.cat_boundaries.get(index).ok_or_else(malformed)? as usize;
    let end = *tree.cat_boundaries.get(index + 1).ok_or_else(malformed)? as usize;
    let words = tree.cat_threshold.get(begin..end).ok_or_else(malformed)?;
    let mut bits = [0u32; 8];
    for (b, &category) in categories.iter().enumerate() {
        set(&mut bits, b + 1, in_bitset(words, category));
    }
    Ok(bits)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::FeatureGroup;
    use crate::tree::SplitChildren;

    fn dataset() -> BinnedDataset {
        let meta = |name: &str, bins: BinMapper, num_bins: u32, column: u32, hist_offset: u32| FeatureMeta {
            name: name.into(),
            bins,
            num_bins,
            group: 0,
            column,
            hist_offset,
        };
        let numerical = BinMapper::Numerical {
            upper_bounds: vec![1.5, 2.5, 3.5, f64::INFINITY],
            nan_bin: true,
            min: 0.0,
            max: 5.0,
        };
        let categorical = BinMapper::Categorical {
            categories: vec![7, 3, 9],
        };
        BinnedDataset {
            num_rows: 0,
            row_offset: 0,
            features: vec![meta("n", numerical, 5, 0, 0), meta("c", categorical, 4, 1, 5)],
            groups: vec![FeatureGroup {
                features: vec![0, 1],
                row_stride: 4,
                hist_offset: 0,
                total_bins: 9,
                data: Vec::new(),
            }],
            total_bins: 9,
            label: Vec::new(),
            weight: None,
            query_boundaries: None,
        }
    }

    fn left(split: &BinSplit, bins: u8) -> Vec<u8> {
        (0..bins).filter(|&b| split.goes_left(b)).collect()
    }

    const CHILDREN: SplitChildren = SplitChildren {
        left_value: 1.0,
        right_value: 2.0,
        left_count: 1,
        right_count: 1,
        left_weight: 1.0,
        right_weight: 1.0,
        gain: 1.0,
    };

    #[test]
    fn numerical_thresholds_map_to_the_bins_below_them() {
        let data = dataset();
        let cases = [
            (2.5, true, MissingType::NaN, vec![0, 1, 4]),
            (2.5, false, MissingType::NaN, vec![0, 1]),
            (f64::MAX, false, MissingType::NaN, vec![0, 1, 2, 3]),
            (1.5, false, MissingType::None, vec![0, 4]),
            (-1.0, false, MissingType::None, vec![]),
            // A threshold inside bin 1 (another binning): the bin goes right with its upper bound.
            (2.0, false, MissingType::NaN, vec![0]),
        ];
        for (threshold, default_left, missing, expected) in cases {
            let mut tree = Tree::new();
            tree.split_numerical(0, 0, threshold, default_left, missing, &CHILDREN);
            let device = device_tree(&tree, &data, 0.0).expect("device tree");
            assert_eq!(
                left(&device.nodes[0].split, 5),
                expected,
                "threshold {threshold}, default_left {default_left}, {missing:?}"
            );
        }
    }

    #[test]
    fn categorical_bitsets_map_to_category_bins() {
        let data = dataset();
        let mut tree = Tree::new();
        tree.split_categorical(0, 1, &[9, 7, 4], MissingType::NaN, &CHILDREN);
        let device = device_tree(&tree, &data, 0.5).expect("device tree");
        assert_eq!(left(&device.nodes[0].split, 4), vec![1, 3]);
        assert_eq!(device.leaf_values, vec![0.5, 1.5]);
        let mut wrong = Tree::new();
        wrong.split_numerical(0, 1, 1.0, false, MissingType::None, &CHILDREN);
        assert_eq!(device_tree(&wrong, &data, 0.0).expect_err("kind").kind(), "config");
    }
}
