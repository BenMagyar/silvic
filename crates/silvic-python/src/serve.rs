//! Batch scoring from Python: `silvic_serve::Predictor` over numpy matrices (f32 or f64, C or Fortran order),
//! compiled and scored without the GIL.

use numpy::{IntoPyArray, PyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use silvic_core::Model;
use silvic_serve::Kernel;

use crate::dataset::Matrix;
use crate::error::IntoPyResult;

#[pyclass(frozen, module = "silvic._silvic")]
pub struct Predictor {
    inner: silvic_serve::Predictor,
    pandas_categorical: Option<String>,
}

#[pymethods]
impl Predictor {
    /// Compiles LightGBM model text (written by silvic, or by LightGBM 3.x/4.x) for `kernel`, or for the best kernel
    /// this CPU runs when `None`.
    #[new]
    #[pyo3(signature = (model_str, kernel=None))]
    fn new(py: Python<'_>, model_str: &str, kernel: Option<&str>) -> PyResult<Predictor> {
        let kernel = kernel.map(str::parse::<Kernel>).transpose().into_py_result()?;
        let (inner, pandas_categorical) = py
            .detach(|| {
                let model = Model::from_lightgbm_string(model_str)?;
                let predictor = match kernel {
                    Some(kernel) => silvic_serve::Predictor::with_kernel(&model, kernel)?,
                    None => silvic_serve::Predictor::new(&model)?,
                };
                Ok((predictor, model.pandas_categorical))
            })
            .into_py_result()?;
        Ok(Predictor {
            inner,
            pandas_categorical,
        })
    }

    /// One prediction per row of `data`, equal to `Booster.predict`'s bit for bit. `num_iteration` `None` or `<= 0`
    /// uses every tree; `num_threads` 1 scores on the calling thread and 0 on every core.
    #[pyo3(signature = (data, raw_score=false, num_iteration=None, num_threads=1))]
    fn predict<'py>(
        &self,
        py: Python<'py>,
        data: &Bound<'py, PyAny>,
        raw_score: bool,
        num_iteration: Option<i64>,
        num_threads: usize,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let matrix = Matrix::extract(data)?;
        let view = matrix.view()?;
        if view.cols != self.inner.num_features() {
            return Err(PyValueError::new_err(format!(
                "the number of features in data ({}) is not the same as it was in training data ({})",
                view.cols,
                self.inner.num_features()
            )));
        }
        let num_trees = num_iteration.and_then(|n| usize::try_from(n).ok()).filter(|&n| n > 0);
        let predictions = py.detach(|| self.inner.predict(&view, raw_score, num_trees, num_threads));
        Ok(predictions.into_pyarray(py))
    }

    /// The kernel scoring runs on: `portable`, `neon`, `avx2` or `avx512`.
    #[getter]
    fn kernel(&self) -> &'static str {
        self.inner.kernel().name()
    }

    #[getter]
    fn num_trees(&self) -> usize {
        self.inner.num_trees()
    }

    #[getter]
    fn num_features(&self) -> usize {
        self.inner.num_features()
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
