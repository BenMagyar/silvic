//! Training from Python: device selection, the GIL-free boosting loop, the per-iteration callback, continued
//! training from a model and checkpoints of the model so far.
//!
//! `device_type` resolves as LightGBM resolves `device_type=gpu` to its one GPU build, extended to the devices
//! silvic has: `auto` takes CUDA, then Metal, then the CPU; `gpu` takes CUDA, then Metal, and fails without either;
//! `cpu`, `metal`, `cuda` and `neuron` fail when the requested device is absent instead of falling back silently.

use pyo3::prelude::*;
use silvic_core::boosting::train_from;
use silvic_core::config::DeviceType;
use silvic_core::{Backend, BinnedDataset, CpuBackend, IterationReport, Model, Params, SilvicError};

use crate::collective::Collective;
use crate::dataset::{with_collective, Dataset};
use crate::error::IntoPyResult;

/// Parses `params_json` as training does, raising its `ConfigError` for an unknown key or a bad value, so a
/// distributed job fails before it reads any data.
#[pyfunction]
pub fn validate_params(params_json: &str) -> PyResult<()> {
    Params::from_json(params_json).into_py_result().map(drop)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Device {
    Cpu,
    /// `None` is the default device.
    Metal(Option<usize>),
    Cuda(Option<usize>),
    Neuron(Option<usize>),
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

#[cfg(target_os = "linux")]
fn neuron_available() -> bool {
    silvic_neuron::is_available()
}

#[cfg(not(target_os = "linux"))]
fn neuron_available() -> bool {
    false
}

/// The devices this process can train on: CPU and any available Metal, CUDA or Neuron devices.
#[pyfunction]
pub fn available_devices() -> Vec<&'static str> {
    let mut devices = vec!["cpu"];
    if metal_available() {
        devices.push("metal");
    }
    if cuda_available() {
        devices.push("cuda");
    }
    if neuron_available() {
        devices.push("neuron");
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
        DeviceType::Neuron => Ok(Device::Neuron(index)),
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
            Device::Neuron(index) => neuron_backend(index),
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

#[cfg(target_os = "linux")]
fn neuron_backend<'a>(index: Option<usize>) -> silvic_core::Result<Box<dyn Backend + 'a>> {
    Ok(Box::new(silvic_neuron::NeuronBackend::new(index)?))
}

#[cfg(not(target_os = "linux"))]
fn neuron_backend<'a>(_index: Option<usize>) -> silvic_core::Result<Box<dyn Backend + 'a>> {
    Err(SilvicError::device("Neuron training is only available on Linux"))
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

/// The Python side of one training run: the iteration callback and the checkpoint sink.
struct Hooks<'a> {
    callback: Option<&'a Py<PyAny>>,
    checkpoint: Option<&'a Py<PyAny>>,
    checkpoint_period: usize,
    /// Absolute index of the last tree (init_model's trees included), which is always checkpointed.
    last_iteration: usize,
    pandas_categorical: Option<&'a str>,
    /// A Python exception (`KeyboardInterrupt` from Ctrl-C included) that stopped training, re-raised once the GIL
    /// is back.
    raised: Option<PyErr>,
}

impl Hooks<'_> {
    /// Iterations count from 0 over init_model's trees and the new ones alike, so a resumed run checkpoints at the
    /// same iterations as an uninterrupted one.
    fn checkpoints_at(&self, iteration: usize) -> bool {
        self.checkpoint.is_some()
            && (iteration == self.last_iteration
                || (self.checkpoint_period > 0 && (iteration + 1).is_multiple_of(self.checkpoint_period)))
    }

    /// The model so far as model text, built without the GIL.
    fn checkpoint_text(&self, report: &IterationReport<'_>) -> Option<String> {
        if !self.checkpoints_at(report.iteration) {
            return None;
        }
        let mut model = report.model();
        if let Some(pandas_categorical) = self.pandas_categorical {
            model.pandas_categorical = Some(pandas_categorical.to_owned());
        }
        Some(model.to_lightgbm_string())
    }

    /// The callback runs first, so a checkpoint sink that ships evaluation results has this iteration's.
    fn call(&self, py: Python<'_>, report: &IterationReport<'_>, checkpoint_text: Option<String>) -> PyResult<bool> {
        let stop = call_callback(py, report, self.callback)?;
        if let (Some(checkpoint), Some(text)) = (self.checkpoint, checkpoint_text) {
            checkpoint.call1(py, (report.iteration, text))?;
        }
        Ok(stop)
    }

    /// The core's iteration hook: a Python exception is kept in `raised` and stops training.
    fn on_iteration(&mut self, report: &IterationReport<'_>) -> silvic_core::Result<()> {
        let checkpoint_text = self.checkpoint_text(report);
        let outcome = Python::attach(|py| self.call(py, report, checkpoint_text));
        match outcome {
            Ok(false) => Ok(()),
            Ok(true) => Err(SilvicError::Interrupted("the iteration callback asked to stop".into())),
            Err(err) => {
                self.raised = Some(err);
                Err(SilvicError::Interrupted("the iteration callback raised".into()))
            }
        }
    }
}

/// Trains a model and returns it as LightGBM 3.3.5 model text. Runs without the GIL; `callback` reacquires it after
/// every tree and may raise or return a true value to stop (which raises `KeyboardInterrupt`). `pandas_categorical`
/// is the JSON stored on the model's last line, as LightGBM's Python package stores pandas category levels.
///
/// `init_model` (model text) continues training: its trees are kept, `num_iterations` trees are added, and
/// iterations, bagging and quantization continue where it stopped, so continuing a model trained on the same data
/// equals training every tree in one run. `checkpoint(iteration, model_str)` receives the model so far every
/// `checkpoint_period` iterations (counted over every tree; `0` turns the periodic ones off) and after the last.
#[pyfunction]
#[pyo3(signature = (
    params_json, train, valid=Vec::new(), collective=None, callback=None, pandas_categorical=None, init_model=None,
    checkpoint=None, checkpoint_period=0,
))]
#[allow(clippy::too_many_arguments)]
pub fn train(
    py: Python<'_>,
    params_json: &str,
    train: Bound<'_, Dataset>,
    valid: Vec<(String, Bound<'_, Dataset>)>,
    collective: Option<Bound<'_, Collective>>,
    callback: Option<Py<PyAny>>,
    pandas_categorical: Option<String>,
    init_model: Option<&str>,
    checkpoint: Option<Py<PyAny>>,
    checkpoint_period: usize,
) -> PyResult<String> {
    let params = Params::from_json(params_json).into_py_result()?;
    let device = choose_device(&params).into_py_result()?;
    let init_model = match init_model {
        Some(text) => Some(py.detach(|| Model::from_lightgbm_string(text)).into_py_result()?),
        None => None,
    };
    let init_trees = init_model.as_ref().map_or(0, Model::num_trees);
    let train_set = &train.get().inner;
    let valid_sets: Vec<(&str, &BinnedDataset)> = valid
        .iter()
        .map(|(name, dataset)| (name.as_str(), &dataset.get().inner))
        .collect();
    let collective = collective.as_ref().map(Bound::get);
    let mut hooks = Hooks {
        callback: callback.as_ref(),
        checkpoint: checkpoint.as_ref(),
        checkpoint_period,
        last_iteration: (init_trees + params.num_iterations).saturating_sub(1),
        pandas_categorical: pandas_categorical.as_deref(),
        raised: None,
    };
    let trained = py.detach(|| {
        let valid_data: Vec<&BinnedDataset> = valid_sets.iter().map(|(_, dataset)| *dataset).collect();
        let mut backend = device.backend(train_set, &valid_data)?;
        if params.verbosity >= 1 {
            log_info(&format!("training on {}", backend.name()));
        }
        let mut hook = |report: &IterationReport<'_>| hooks.on_iteration(report);
        with_collective(collective, |c| {
            train_from(
                &params,
                init_model.as_ref(),
                train_set,
                &valid_sets,
                backend.as_mut(),
                c,
                &mut hook,
            )
        })
    });
    if let Some(err) = hooks.raised {
        return Err(err);
    }
    let mut model = trained.into_py_result()?;
    if pandas_categorical.is_some() {
        model.pandas_categorical = pandas_categorical;
    }
    Ok(py.detach(|| model.to_lightgbm_string()))
}
