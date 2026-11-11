//! Exact batch scoring of LightGBM 3.3.5 models (`model.txt`, as `silvic_core::Model` parses it): predictions equal
//! `Model::predict`'s, and so LightGBM's, bit for bit.
//!
//! [`Predictor::new`] compiles a model once. Each tree becomes QuickScorer's leaf bitvector (Lucchese et al., SIGIR
//! 2015): leaves numbered left to right, and per internal node the bits of its left subtree, cleared for rows that
//! go right there; the lowest bit left set is the exit leaf, so a tree is scored with no data-dependent branch. Rows
//! are scored in blocks of 64 (32 on the portable kernel), one SIMD byte lane per row, with each tree's bitvectors
//! byte-sliced across the lanes as in RapidScorer (Ye et al., KDD 2018) and vQS (Lucchese et al., SIGIR 2016). Each
//! block's values are first quantized to one byte per (feature, missing-value handling, 254 thresholds): a value's
//! rank among the feature's split thresholds, so `value <= threshold` is a byte compare. NaN, zero and categories
//! map to codes that reproduce `Tree::NumericalDecision` and `Tree::CategoricalDecision` exactly (see `compile`).
//! Leaf values are summed per row in tree order in f64, as `GBDT::PredictRaw` sums them.
//!
//! Trees with more than 256 leaves, NaN thresholds, or categorical splits on a feature with more than 64 distinct
//! category behaviours are walked node by node with `silvic_core`'s decisions instead, so every model scores.

mod compile;
mod kernel;
mod predictor;

pub use predictor::{Kernel, Predictor};
