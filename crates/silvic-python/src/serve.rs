//! Batch scoring from Python over numpy matrices (f32 or f64, C or Fortran order), without the GIL.

#[cfg(target_os = "linux")]
use std::sync::Mutex;

use numpy::IntoPyArray;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use silvic_core::{Model, SilvicError};
use silvic_serve::Kernel;

use crate::dataset::Matrix;
use crate::error::IntoPyResult;

enum PredictorBackend {
    Cpu(silvic_serve::Predictor),
    #[cfg(target_os = "linux")]
    Cuda(Mutex<silvic_serve::CudaPredictor>),
    #[cfg(target_os = "linux")]
    CudaF32(Mutex<silvic_serve::CudaPredictorF32>),
    #[cfg(target_os = "linux")]
    Neuron(Mutex<silvic_neuron::NeuronPredictor>),
}

enum Predictions {
    F64(Vec<f64>),
    #[cfg(target_os = "linux")]
    F32(Vec<f32>),
}

#[pyclass(frozen, module = "silvic._silvic")]
pub struct Predictor {
    inner: PredictorBackend,
    num_trees: usize,
    num_features: usize,
    pandas_categorical: Option<String>,
}

#[pymethods]
impl Predictor {
    /// Prepares a CPU model or an explicitly requested CUDA or Neuron model.
    #[new]
    #[pyo3(signature = (model_str, kernel=None, *, device="cpu", device_id=0, precision="float64"))]
    fn new(
        py: Python<'_>,
        model_str: &str,
        kernel: Option<&str>,
        device: &str,
        device_id: usize,
        precision: &str,
    ) -> PyResult<Predictor> {
        if !matches!(device, "cpu" | "cuda" | "neuron") {
            return Err(PyValueError::new_err(
                "serving device must be `cpu`, `cuda` or `neuron`",
            ));
        }
        if !matches!(precision, "float64" | "float32") {
            return Err(PyValueError::new_err(
                "serving precision must be `float64` or `float32`",
            ));
        }
        if device == "cpu" && precision != "float64" {
            return Err(PyValueError::new_err(
                "float32 serving requires device=\"cuda\" or device=\"neuron\"",
            ));
        }
        if device == "neuron" && precision != "float32" {
            return Err(PyValueError::new_err(
                "Neuron serving requires explicit precision=\"float32\"",
            ));
        }
        if device != "cpu" && kernel.is_some() {
            return Err(PyValueError::new_err(
                "kernel selects a CPU kernel and cannot be used with CUDA or Neuron",
            ));
        }
        if device == "cpu" && device_id != 0 {
            return Err(PyValueError::new_err("device_id applies only to CUDA or Neuron"));
        }
        let kernel = kernel.map(str::parse::<Kernel>).transpose().into_py_result()?;
        let (inner, num_trees, num_features, pandas_categorical) = py
            .detach(|| {
                let model = Model::from_lightgbm_string(model_str)?;
                let (inner, num_trees, num_features) = if device == "cpu" {
                    let predictor = match kernel {
                        Some(kernel) => silvic_serve::Predictor::with_kernel(&model, kernel)?,
                        None => silvic_serve::Predictor::new(&model)?,
                    };
                    let counts = (predictor.num_trees(), predictor.num_features());
                    (PredictorBackend::Cpu(predictor), counts.0, counts.1)
                } else {
                    #[cfg(target_os = "linux")]
                    {
                        if device == "neuron" {
                            let predictor = silvic_neuron::NeuronPredictor::new(&model, Some(device_id))?;
                            let counts = (predictor.num_trees(), predictor.num_features());
                            (PredictorBackend::Neuron(Mutex::new(predictor)), counts.0, counts.1)
                        } else if precision == "float32" {
                            let predictor = silvic_serve::CudaPredictorF32::new(&model, Some(device_id))?;
                            let counts = (predictor.num_trees(), predictor.num_features());
                            (PredictorBackend::CudaF32(Mutex::new(predictor)), counts.0, counts.1)
                        } else {
                            let predictor = silvic_serve::CudaPredictor::new(&model, Some(device_id))?;
                            let counts = (predictor.num_trees(), predictor.num_features());
                            (PredictorBackend::Cuda(Mutex::new(predictor)), counts.0, counts.1)
                        }
                    }
                    #[cfg(not(target_os = "linux"))]
                    {
                        return Err(SilvicError::unsupported(format!(
                            "{device} serving is only available on Linux"
                        )));
                    }
                };
                Ok((inner, num_trees, num_features, model.pandas_categorical))
            })
            .into_py_result()?;
        Ok(Predictor {
            inner,
            num_trees,
            num_features,
            pandas_categorical,
        })
    }

    /// One prediction per row. Default float64 is exact; explicit CUDA or Neuron float32 is approximate.
    /// `num_iteration` `None` or `<= 0` uses every tree. CUDA and Neuron require `num_threads=1`.
    #[pyo3(signature = (data, raw_score=false, num_iteration=None, num_threads=1))]
    fn predict<'py>(
        &self,
        py: Python<'py>,
        data: &Bound<'py, PyAny>,
        raw_score: bool,
        num_iteration: Option<i64>,
        num_threads: usize,
    ) -> PyResult<Bound<'py, PyAny>> {
        #[cfg(target_os = "linux")]
        if matches!(
            &self.inner,
            PredictorBackend::Cuda(_) | PredictorBackend::CudaF32(_) | PredictorBackend::Neuron(_)
        ) && num_threads != 1
        {
            return Err(PyValueError::new_err(
                "num_threads applies only to CPU scoring; use 1 with CUDA or Neuron",
            ));
        }
        let matrix = Matrix::extract(data)?;
        let view = matrix.view()?;
        if view.cols != self.num_features {
            return Err(PyValueError::new_err(format!(
                "the number of features in data ({}) is not the same as it was in training data ({})",
                view.cols, self.num_features
            )));
        }
        let num_trees = num_iteration.and_then(|n| usize::try_from(n).ok()).filter(|&n| n > 0);
        let predictions = py
            .detach(|| match &self.inner {
                PredictorBackend::Cpu(predictor) => Ok(Predictions::F64(predictor.predict(
                    &view,
                    raw_score,
                    num_trees,
                    num_threads,
                ))),
                #[cfg(target_os = "linux")]
                PredictorBackend::Cuda(predictor) => predictor
                    .lock()
                    .map_err(|_| SilvicError::device("CUDA serving predictor lock is poisoned"))?
                    .predict(&view, raw_score, num_trees)
                    .map(Predictions::F64),
                #[cfg(target_os = "linux")]
                PredictorBackend::CudaF32(predictor) => predictor
                    .lock()
                    .map_err(|_| SilvicError::device("CUDA serving predictor lock is poisoned"))?
                    .predict(&view, raw_score, num_trees)
                    .map(Predictions::F32),
                #[cfg(target_os = "linux")]
                PredictorBackend::Neuron(predictor) => predictor
                    .lock()
                    .map_err(|_| SilvicError::device("Neuron serving predictor lock is poisoned"))?
                    .predict(&view, raw_score, num_trees)
                    .map(Predictions::F32),
            })
            .into_py_result()?;
        Ok(match predictions {
            Predictions::F64(values) => values.into_pyarray(py).into_any(),
            #[cfg(target_os = "linux")]
            Predictions::F32(values) => values.into_pyarray(py).into_any(),
        })
    }

    /// The CPU kernel name, `cuda` or `neuron`.
    #[getter]
    fn kernel(&self) -> &'static str {
        match &self.inner {
            PredictorBackend::Cpu(predictor) => predictor.kernel().name(),
            #[cfg(target_os = "linux")]
            PredictorBackend::Cuda(_) | PredictorBackend::CudaF32(_) => "cuda",
            #[cfg(target_os = "linux")]
            PredictorBackend::Neuron(_) => "neuron",
        }
    }

    #[getter]
    fn device(&self) -> &'static str {
        match &self.inner {
            PredictorBackend::Cpu(_) => "cpu",
            #[cfg(target_os = "linux")]
            PredictorBackend::Cuda(_) | PredictorBackend::CudaF32(_) => "cuda",
            #[cfg(target_os = "linux")]
            PredictorBackend::Neuron(_) => "neuron",
        }
    }

    #[getter]
    fn precision(&self) -> &'static str {
        #[cfg(target_os = "linux")]
        if matches!(&self.inner, PredictorBackend::CudaF32(_) | PredictorBackend::Neuron(_)) {
            return "float32";
        }
        "float64"
    }

    #[getter]
    fn num_trees(&self) -> usize {
        self.num_trees
    }

    #[getter]
    fn num_features(&self) -> usize {
        self.num_features
    }

    /// The JSON after the model's `pandas_categorical:` line, if it has one.
    #[getter]
    fn pandas_categorical(&self) -> Option<String> {
        self.pandas_categorical.clone()
    }
}

/// The kernels this CPU runs.
#[pyfunction]
pub fn available_kernels() -> Vec<&'static str> {
    Kernel::available().into_iter().map(Kernel::name).collect()
}

/// Serving devices available in this build and runtime.
#[pyfunction]
pub fn available_serving_devices() -> Vec<&'static str> {
    let devices = vec!["cpu"];
    #[cfg(target_os = "linux")]
    let devices = {
        let mut devices = devices;
        if silvic_serve::cuda_is_available() {
            devices.push("cuda");
        }
        if silvic_neuron::is_available() {
            devices.push("neuron");
        }
        devices
    };
    devices
}
