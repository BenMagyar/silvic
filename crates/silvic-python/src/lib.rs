//! The `silvic._silvic` extension module: numpy-backed datasets, GIL-free training on the CPU, Metal or CUDA, model
//! text in and out, and the TCP collectives distributed training runs on. The `silvic` Python package wraps it in
//! LightGBM's API.

mod booster;
mod collective;
mod dataset;
mod error;
mod training;

use pyo3::prelude::*;

#[pymodule]
fn _silvic(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("__version__", env!("CARGO_PKG_VERSION"))?;
    error::register(module)?;
    module.add_class::<dataset::Dataset>()?;
    module.add_class::<booster::Booster>()?;
    module.add_class::<collective::Listener>()?;
    module.add_class::<collective::Collective>()?;
    module.add_function(wrap_pyfunction!(training::train, module)?)?;
    module.add_function(wrap_pyfunction!(training::available_devices, module)?)?;
    module.add_function(wrap_pyfunction!(collective::connect, module)?)?;
    Ok(())
}
