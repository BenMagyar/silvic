//! A Triton Inference Server backend, `silvic` (`libtriton_silvic.so`), that serves LightGBM `model.txt` models
//! (LightGBM 3.3.5 and 4.x, or silvic) with silvic-serve's `Predictor`: predictions equal `lightgbm.Booster.predict`'s
//! bit for bit. It implements the backend C API of triton-inference-server/core (`tritonbackend.h`) and follows the
//! FIL backend's model-configuration conventions (`input__0`, `output__0`, `model.txt`, `max_batch_size`), so a FIL
//! LightGBM model directory serves by changing `backend: "fil"` to `backend: "silvic"`.
//!
//! The configuration, request checks, batching and output writing are safe Rust (`config`, `batch`), tested without
//! Triton; `triton` declares the C API and wraps its handles, and `backend` exports the entry points. Those two are
//! left out of the crate's unit tests, whose binary has no tritonserver to resolve the C API against.

mod batch;
mod config;
mod error;

#[cfg(not(test))]
mod backend;
#[cfg(not(test))]
mod triton;
