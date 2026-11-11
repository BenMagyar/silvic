//! LightGBM-compatible gradient boosting: binning, leaf-wise tree growth on quantized integer histograms, and
//! LightGBM 3.3.5 `model.txt` output.
//!
//! [`boosting::train`] grows trees through a [`Backend`] (this crate's [`CpuBackend`], or the Metal and CUDA
//! backends of `silvic-gpu`) and reduces histograms across ranks through a [`Collective`] ([`LocalCollective`] for
//! one process, `silvic-net`'s TCP ring across many). Quantized histograms are integer sums, so training is
//! bit-reproducible and a model trained on N ranks equals the one trained on all rows in one process.

pub mod backend;
pub mod boosting;
pub mod collective;
pub mod config;
pub mod data;
mod error;
mod learner;
pub mod metric;
pub mod model;
pub mod objective;
pub mod quantize;
mod split;
#[cfg(any(test, feature = "test-util"))]
pub mod test_util;
pub mod tree;

pub use backend::{Backend, CpuBackend};
pub use boosting::{train, IterationReport};
pub use collective::{Collective, LocalCollective};
pub use config::Params;
pub use data::BinnedDataset;
pub use error::{Result, SilvicError};
pub use model::Model;
