//! Training from Python: device selection, the GIL-free boosting loop, and the per-iteration callback.
//!
//! `device_type` resolves as LightGBM resolves `device_type=gpu` to its one GPU build, extended to the devices
//! silvic has: `auto` takes CUDA, then Metal, then the CPU; `gpu` takes CUDA, then Metal, and fails without either;
//! `cpu`, `metal` and `cuda` fail when the device is absent instead of falling back silently.

use pyo3::prelude::*;
use silvic_core::config::DeviceType;
use silvic_core::{Backend, BinnedDataset, CpuBackend, IterationReport, Params, SilvicError};

use crate::collective::Collective;
use crate::dataset::{with_collective, Dataset};
use crate::error::IntoPyResult;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Device {
    Cpu,
    /// `None` is the default device.
    Metal(Option<usize>),
    Cuda(Option<usize>),
}

#[cfg(target_os = "macos")]
fn metal_available() -> bool {
    silvic_gpu::metal::is_available()
}

#[cfg(not(target_os = "macos"))]
fn metal_available() -> bool {
    false
}

#[cfg(target_os = "linux")]
fn cuda_available() -> bool {
    silvic_gpu::cuda::is_available()
}

#[cfg(not(target_os = "linux"))]
fn cuda_available() -> bool {
    false
}

/// The devices this process can train on: `cpu`, plus `metal` and `cuda` when present.
#[pyfunction]
pub fn available_devices() -> Vec<&'static str> {
    let mut devices = vec!["cpu"];
    if metal_available() {
        devices.push("metal");
    }
    if cuda_available() {
        devices.push("cuda");
    }
    devices
}

fn choose_device(params: &Params) -> silvic_core::Result<Device> {
    // `gpu_device_id = -1` (LightGBM's default) is the device the driver picks.
    let index = usize::try_from(params.gpu_device_id).ok();
    let gpu = || {
        if cuda_available() {
            Some(Device::Cuda(index))
        } else if metal_available() {
            Some(Device::Metal(index))
        } else {
            None
        }
    };
    match params.device_type {
        DeviceType::Cpu => Ok(Device::Cpu),
        DeviceType::Auto => Ok(gpu().unwrap_or(Device::Cpu)),
        DeviceType::Gpu => {
            gpu().ok_or_else(|| SilvicError::device("device_type=gpu but neither CUDA nor Metal is available"))
        }
        DeviceType::Metal if metal_available() => Ok(Device::Metal(index)),
        DeviceType::Metal => Err(SilvicError::device(
            "device_type=metal but no Metal device is available",
        )),
        DeviceType::Cuda if cuda_available() => Ok(Device::Cuda(index)),
        DeviceType::Cuda => Err(SilvicError::device("device_type=cuda but no CUDA device is available")),
    }
}

impl Device {
    fn backend<'a>(
        self,
        train: &'a BinnedDataset,
        valid: &[&'a BinnedDataset],
    ) -> silvic_core::Result<Box<dyn Backend + 'a>> {
        match self {
            Device::Cpu => Ok(Box::new(CpuBackend::borrowing(train, valid))),
            Device::Metal(index) => metal_backend(index),
            Device::Cuda(index) => cuda_backend(index),
        }
    }
}

#[cfg(target_os = "macos")]
fn metal_backend<'a>(index: Option<usize>) -> silvic_core::Result<Box<dyn Backend + 'a>> {
    Ok(Box::new(silvic_gpu::metal::backend(index)?))
}

#[cfg(not(target_os = "macos"))]
fn metal_backend<'a>(_index: Option<usize>) -> silvic_core::Result<Box<dyn Backend + 'a>> {
    Err(SilvicError::device("Metal is only available on macOS"))
}

#[cfg(target_os = "linux")]
fn cuda_backend<'a>(index: Option<usize>) -> silvic_core::Result<Box<dyn Backend + 'a>> {
    Ok(Box::new(silvic_gpu::cuda::backend(index)?))
}

#[cfg(not(target_os = "linux"))]
fn cuda_backend<'a>(_index: Option<usize>) -> silvic_core::Result<Box<dyn Backend + 'a>> {
    Err(SilvicError::device("CUDA is only available on Linux"))
}

/// Writes `[silvic] [Info] <message>` to Python's `sys.stdout`, where LightGBM's Python package sends its log, so
/// notebooks show it.
fn log_info(message: &str) {
    Python::attach(|py| {
        let line = format!("[silvic] [Info] {message}\n");
        // Logging must never fail training; a closed or replaced stdout drops the line.
        let _ = py
            .import("sys")
            .and_then(|sys| sys.getattr("stdout")?.call_method1("write", (line,)));
    });
}

/// Calls the Python callback with `(iteration, [(dataset, metric, value, higher_is_better), ...])` after checking
/// for signals; `Ok(true)` asks to stop.
fn call_callback(py: Python<'_>, report: &IterationReport<'_>, callback: Option<&Py<PyAny>>) -> PyResult<bool> {
    py.check_signals()?;
    let Some(callback) = callback else {
        return Ok(false);
    };
    let evals: Vec<(&str, &str, f64, bool)> = report
        .evals
        .iter()
        .map(|e| (e.dataset.as_str(), e.metric.as_str(), e.value, e.higher_is_better))
        .collect();
    callback.call1(py, (report.iteration, evals))?.bind(py).is_truthy()
}

/// The core's iteration hook: a Python exception (`KeyboardInterrupt` from Ctrl-C included) is kept in
/// `raised` and stops training, to be re-raised once the GIL is back.
fn on_iteration(
    report: &IterationReport<'_>,
    callback: Option<&Py<PyAny>>,
    raised: &mut Option<PyErr>,
) -> silvic_core::Result<()> {
    Python::attach(|py| match call_callback(py, report, callback) {
        Ok(false) => Ok(()),
        Ok(true) => Err(SilvicError::Interrupted("the iteration callback asked to stop".into())),
        Err(err) => {
            *raised = Some(err);
            Err(SilvicError::Interrupted("the iteration callback raised".into()))
        }
    })
}

/// Trains a model and returns it as LightGBM 3.3.5 model text. Runs without the GIL; `callback` reacquires it after
/// every tree and may raise or return a true value to stop (which raises `KeyboardInterrupt`). `pandas_categorical`
/// is the JSON stored on the model's last line, as LightGBM's Python package stores pandas category levels.
#[pyfunction]
#[pyo3(signature = (params_json, train, valid=Vec::new(), collective=None, callback=None, pandas_categorical=None))]
pub fn train(
    py: Python<'_>,
    params_json: &str,
    train: Bound<'_, Dataset>,
    valid: Vec<(String, Bound<'_, Dataset>)>,
    collective: Option<Bound<'_, Collective>>,
    callback: Option<Py<PyAny>>,
    pandas_categorical: Option<String>,
) -> PyResult<String> {
    let params = Params::from_json(params_json).into_py_result()?;
    let device = choose_device(&params).into_py_result()?;
    let train_set = &train.get().inner;
    let valid_sets: Vec<(&str, &BinnedDataset)> = valid
        .iter()
        .map(|(name, dataset)| (name.as_str(), &dataset.get().inner))
        .collect();
    let collective = collective.as_ref().map(Bound::get);
    let mut raised = None;
    let trained = py.detach(|| {
        let valid_data: Vec<&BinnedDataset> = valid_sets.iter().map(|(_, dataset)| *dataset).collect();
        let mut backend = device.backend(train_set, &valid_data)?;
        if params.verbosity >= 1 {
            log_info(&format!("training on {}", backend.name()));
        }
        let mut hook = |report: &IterationReport<'_>| on_iteration(report, callback.as_ref(), &mut raised);
        with_collective(collective, |c| {
            silvic_core::train(&params, train_set, &valid_sets, backend.as_mut(), c, &mut hook)
        })
    });
    if let Some(err) = raised {
        return Err(err);
    }
    let mut model = trained.into_py_result()?;
    if pandas_categorical.is_some() {
        model.pandas_categorical = pandas_categorical;
    }
    Ok(py.detach(|| model.to_lightgbm_string()))
}
