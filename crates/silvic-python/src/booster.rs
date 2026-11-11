//! A trained model for prediction from Python: LightGBM's `Booster` predicting from a numpy matrix
//! (`LGBM_BoosterPredictForMat`), with model text in and out.

use numpy::{IntoPyArray, PyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use silvic_core::model::ImportanceType;
use silvic_core::Model;

use crate::dataset::Matrix;
use crate::error::IntoPyResult;

#[pyclass(frozen, module = "silvic._silvic")]
pub struct Booster {
    model: Model,
}

#[pymethods]
impl Booster {
    /// Parses LightGBM v3 model text (written by silvic, or by LightGBM 3.x/4.x).
    #[new]
    fn new(py: Python<'_>, model_str: &str) -> PyResult<Booster> {
        let model = py.detach(|| Model::from_lightgbm_string(model_str)).into_py_result()?;
        Ok(Booster { model })
    }

    /// One prediction per row of `data` (f32 or f64, C or Fortran order), parallel over rows without the GIL.
    /// `num_iteration` `None` or `<= 0` uses every tree, as LightGBM does for a model without a best iteration.
    #[pyo3(signature = (data, raw_score=false, num_iteration=None))]
    fn predict<'py>(
        &self,
        py: Python<'py>,
        data: &Bound<'py, PyAny>,
        raw_score: bool,
        num_iteration: Option<i64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let matrix = Matrix::extract(data)?;
        let view = matrix.view()?;
        if view.cols != self.model.num_features() {
            return Err(PyValueError::new_err(format!(
                "the number of features in data ({}) is not the same as it was in training data ({})",
                view.cols,
                self.model.num_features()
            )));
        }
        let num_trees = num_iteration
            .and_then(|n| usize::try_from(n).ok())
            .filter(|&n| n > 0)
            .map(|n| n.min(self.model.num_trees()));
        let predictions = py.detach(|| self.model.predict(&view, raw_score, num_trees));
        Ok(predictions.into_pyarray(py))
    }

    /// The model as LightGBM 3.3.5's Python `Booster.model_to_string()` returns it.
    fn model_to_string(&self, py: Python<'_>) -> String {
        py.detach(|| self.model.to_lightgbm_string())
    }

    /// `split`: times each feature is split on; `gain`: total gain of its splits.
    #[pyo3(signature = (kind="split"))]
    fn feature_importance<'py>(&self, py: Python<'py>, kind: &str) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let kind = match kind {
            "split" => ImportanceType::Split,
            "gain" => ImportanceType::Gain,
            other => return Err(PyValueError::new_err(format!("unknown importance type `{other}`"))),
        };
        Ok(self.model.feature_importance(kind).into_pyarray(py))
    }

    #[getter]
    fn num_trees(&self) -> usize {
        self.model.num_trees()
    }

    #[getter]
    fn num_features(&self) -> usize {
        self.model.num_features()
    }

    #[getter]
    fn feature_names(&self) -> Vec<String> {
        self.model.feature_names.clone()
    }

    /// The JSON after the model's `pandas_categorical:` line, if it has one.
    #[getter]
    fn pandas_categorical(&self) -> Option<String> {
        self.model.pandas_categorical.clone()
    }
}
