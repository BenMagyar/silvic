//! Binned training data: every feature value replaced by a bin index (a byte), features packed into groups whose
//! bins fit one GPU threadgroup's shared histogram.
//!
//! Bin mappers follow LightGBM 3.3.5 `BinMapper` semantics where they reach the model file: a numerical value goes
//! to the first bin whose upper bound is `>= value`, and a split "bins `<= t` go left" is written with threshold
//! `upper_bounds[t]`, so `value <= threshold` at prediction time agrees with training for every value. NaN has its
//! own bin when the training sample had NaN (`MissingType::NaN`); otherwise NaN bins as 0.0, as LightGBM predicts
//! with `MissingType::None`. Categorical bin 0 collects negative, NaN, unseen and dropped categories, which LightGBM
//! sends right.

mod bin_mapper;
mod binning;
mod build;
#[cfg(test)]
mod parity_tests;
mod sample;

pub use bin_mapper::BinMapper;
pub use build::{DatasetInput, DenseData, DenseMatrix, Layout};

/// Bins one feature group may hold: a packed i32 per bin fills 32 KiB, Metal's threadgroup memory limit.
pub const GROUP_BIN_BUDGET: u32 = 8192;

#[derive(Debug, Clone, PartialEq)]
pub struct FeatureMeta {
    pub name: String,
    pub bins: BinMapper,
    /// Bins in the histogram, the NaN bin included; 0 for a trivial feature.
    pub num_bins: u32,
    /// The feature's group; meaningless for a trivial feature.
    pub group: u32,
    /// Byte offset of the feature in its group's row.
    pub column: u32,
    /// Index of the feature's bin 0 in the flat histogram.
    pub hist_offset: u32,
}

impl FeatureMeta {
    pub fn is_trivial(&self) -> bool {
        self.num_bins == 0
    }

    pub fn is_categorical(&self) -> bool {
        matches!(self.bins, BinMapper::Categorical { .. })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct FeatureGroup {
    /// Indices into `BinnedDataset::features`, in column order.
    pub features: Vec<usize>,
    /// Bytes per row: the feature count rounded up to a multiple of 4, padding bytes zero.
    pub row_stride: u32,
    /// Index of the group's first bin in the flat histogram; a group's bins are contiguous there.
    pub hist_offset: u32,
    pub total_bins: u32,
    /// Row-major bins: `data[row * row_stride + column]`.
    pub data: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BinnedDataset {
    pub num_rows: usize,
    /// Global id of local row 0: rows of lower ranks come first. Keys the random draws of quantization and bagging.
    pub row_offset: u64,
    /// Every input feature in input order, trivial ones included.
    pub features: Vec<FeatureMeta>,
    pub groups: Vec<FeatureGroup>,
    pub total_bins: usize,
    pub label: Vec<f32>,
    pub weight: Option<Vec<f32>>,
    /// `num_queries + 1` row boundaries; queries are contiguous runs of rows.
    pub query_boundaries: Option<Vec<u32>>,
}

impl BinnedDataset {
    pub fn num_queries(&self) -> usize {
        self.query_boundaries.as_ref().map_or(0, |b| b.len() - 1)
    }

    /// The bin of `feature` at `row`.
    #[inline]
    pub fn bin(&self, feature: usize, row: usize) -> u8 {
        let meta = &self.features[feature];
        let group = &self.groups[meta.group as usize];
        group.data[row * group.row_stride as usize + meta.column as usize]
    }
}
