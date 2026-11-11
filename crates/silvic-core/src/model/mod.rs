//! A trained model in LightGBM 3.3.5's text format (`GBDT::SaveModelToString`), which LightGBM 3.3.5 and 4.x load
//! and lleaves compiles, and the predictor over it.

mod predict;
mod text;

use crate::tree::Tree;

#[derive(Debug, Clone, PartialEq)]
pub enum ModelObjective {
    Regression,
    /// `binary sigmoid:<sigmoid>`.
    Binary {
        sigmoid: f64,
    },
    LambdaRank,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImportanceType {
    /// Times a feature is split on.
    Split,
    /// Total gain of a feature's splits.
    Gain,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Model {
    pub objective: ModelObjective,
    pub feature_names: Vec<String>,
    /// `BinMapper::info_string` per feature.
    pub feature_infos: Vec<String>,
    pub trees: Vec<Tree>,
    /// `[key: value]` lines of the `parameters:` block.
    pub parameters: Vec<(String, String)>,
    /// The JSON after `pandas_categorical:` (LightGBM's Python package stores pandas category levels there).
    pub pandas_categorical: Option<String>,
}

impl Model {
    pub fn num_features(&self) -> usize {
        self.feature_names.len()
    }

    pub fn num_trees(&self) -> usize {
        self.trees.len()
    }
}
