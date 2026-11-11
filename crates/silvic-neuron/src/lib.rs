//! AWS Neuron training and explicit float32 serving.
//!
//! The existing Rust learner drives [`NeuronBackend`]. [`NeuronPredictor`] scores
//! LightGBM models with float32 inputs, thresholds, leaves, ordered sums and output;
//! it does not promise exact float64 predictions. Both require a Neuron runtime
//! and device; device failures never select another backend.

mod compiler;
mod runtime;
mod serving;
mod training;

pub use serving::NeuronPredictor;
pub use training::NeuronBackend;

/// Whether the native runtime exposes NeuronCores, without initializing it or reserving cores.
pub fn is_available() -> bool {
    runtime::is_available()
}
