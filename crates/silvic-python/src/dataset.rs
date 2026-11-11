//! Binned datasets built from numpy arrays, the counterpart of LightGBM's `LGBM_DatasetCreateFromMat`. The feature
//! matrix (f32 or f64, C or Fortran order) is read in place: binning copies it into bytes, so nothing keeps a
//! reference to the array afterwards.

use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use silvic_core::data::{DatasetInput, DenseData, DenseMatrix, Layout};
use silvic_core::{BinnedDataset, LocalCollective, Params};

use crate::collective::Collective;
use crate::error::IntoPyResult;

/// A borrowed 2-D numpy array of a dtype silvic reads without converting.
pub enum Matrix<'py> {
    F32(PyReadonlyArray2<'py, f32>),
    F64(PyReadonlyArray2<'py, f64>),
}

impl<'py> Matrix<'py> {
    pub fn extract(data: &Bound<'py, PyAny>) -> PyResult<Self> {
        if let Ok(array) = data.extract::<PyReadonlyArray2<'py, f32>>() {
            return Ok(Matrix::F32(array));
        }
        if let Ok(array) = data.extract::<PyReadonlyArray2<'py, f64>>() {
            return Ok(Matrix::F64(array));
        }
        Err(PyTypeError::new_err(
            "data must be a 2-D numpy array of float32 or float64",
        ))
    }

    /// The array as a [`DenseMatrix`]; fails unless it is C- or Fortran-contiguous.
    pub fn view(&self) -> PyResult<DenseMatrix<'_>> {
        let (shape, c_order, f_order) = match self {
            Matrix::F32(a) => (a.shape(), a.is_c_contiguous(), a.is_fortran_contiguous()),
            Matrix::F64(a) => (a.shape(), a.is_c_contiguous(), a.is_fortran_contiguous()),
        };
        let (rows, cols) = (shape[0], shape[1]);
        // A 1-row or 1-column array is both C- and F-contiguous; either layout indexes it the same way.
        let layout = match (c_order, f_order) {
            (true, _) => Layout::RowMajor,
            (false, true) => Layout::ColumnMajor,
            (false, false) => return Err(PyValueError::new_err("data must be C- or Fortran-contiguous")),
        };
        let not_contiguous = |_| PyValueError::new_err("data must be aligned and contiguous");
        let data = match self {
            Matrix::F32(a) => DenseData::F32(a.as_slice().map_err(not_contiguous)?),
            Matrix::F64(a) => DenseData::F64(a.as_slice().map_err(not_contiguous)?),
        };
        Ok(DenseMatrix {
            data,
            rows,
            cols,
            layout,
        })
    }
}

/// Labels, weights and query sizes as numpy borrows, checked against the row count.
struct Metadata<'py> {
    label: PyReadonlyArray1<'py, f32>,
    weight: Option<PyReadonlyArray1<'py, f32>>,
    group: Option<PyReadonlyArray1<'py, u32>>,
}

impl Metadata<'_> {
    fn input<'a>(
        &'a self,
        features: DenseMatrix<'a>,
        feature_names: Option<Vec<String>>,
        categorical_features: Vec<usize>,
    ) -> PyResult<DatasetInput<'a>> {
        let contiguous = |name: &str| PyValueError::new_err(format!("{name} must be a contiguous 1-D array"));
        let label = self.label.as_slice().map_err(|_| contiguous("label"))?;
        let weight = match &self.weight {
            Some(w) => Some(w.as_slice().map_err(|_| contiguous("weight"))?),
            None => None,
        };
        let group = match &self.group {
            Some(g) => Some(g.as_slice().map_err(|_| contiguous("group"))?),
            None => None,
        };
        check_lengths(features.rows, label, weight, group)?;
        Ok(DatasetInput {
            features,
            label,
            weight,
            group,
            feature_names,
            categorical_features,
        })
    }
}

/// Mismatched lengths would index out of bounds in the core, which trusts its input.
fn check_lengths(rows: usize, label: &[f32], weight: Option<&[f32]>, group: Option<&[u32]>) -> PyResult<()> {
    if label.len() != rows {
        return Err(PyValueError::new_err(format!(
            "label has {} values for {rows} rows",
            label.len()
        )));
    }
    if let Some(weight) = weight.filter(|w| w.len() != rows) {
        return Err(PyValueError::new_err(format!(
            "weight has {} values for {rows} rows",
            weight.len()
        )));
    }
    if let Some(group) = group {
        let total: u64 = group.iter().map(|&size| u64::from(size)).sum();
        if total != rows as u64 {
            return Err(PyValueError::new_err(format!(
                "group sizes sum to {total} for {rows} rows"
            )));
        }
    }
    Ok(())
}

pub fn with_collective<T>(
    collective: Option<&Collective>,
    f: impl FnOnce(&mut dyn silvic_core::Collective) -> silvic_core::Result<T>,
) -> silvic_core::Result<T> {
    match collective {
        Some(collective) => collective.with(f),
        None => f(&mut LocalCollective),
    }
}

/// One rank's rows, binned.
#[pyclass(frozen, module = "silvic._silvic")]
pub struct Dataset {
    pub inner: BinnedDataset,
}

#[pymethods]
impl Dataset {
    /// Bins a training set with the `params_json` binning parameters. With a `collective`, bin mappers come from a
    /// sample over every rank, so every rank must construct its training set at the same point.
    #[new]
    #[pyo3(signature = (
        data, label, weight=None, group=None, feature_names=None, categorical_features=Vec::new(),
        params_json="{}", collective=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new<'py>(
        py: Python<'py>,
        data: &Bound<'py, PyAny>,
        label: PyReadonlyArray1<'py, f32>,
        weight: Option<PyReadonlyArray1<'py, f32>>,
        group: Option<PyReadonlyArray1<'py, u32>>,
        feature_names: Option<Vec<String>>,
        categorical_features: Vec<usize>,
        params_json: &str,
        collective: Option<Bound<'py, Collective>>,
    ) -> PyResult<Dataset> {
        let params = Params::from_json(params_json).into_py_result()?;
        let matrix = Matrix::extract(data)?;
        let metadata = Metadata { label, weight, group };
        let input = metadata.input(matrix.view()?, feature_names, categorical_features)?;
        let collective = collective.as_ref().map(Bound::get);
        let inner = py
            .detach(|| with_collective(collective, |c| BinnedDataset::construct(&input, &params, c)))
            .into_py_result()?;
        Ok(Dataset { inner })
    }

    /// Bins a validation set with `reference`'s bin mappers and feature groups (LightGBM `reference=`).
    #[staticmethod]
    #[pyo3(signature = (data, label, reference, weight=None, group=None, collective=None))]
    fn construct_with_reference<'py>(
        py: Python<'py>,
        data: &Bound<'py, PyAny>,
        label: PyReadonlyArray1<'py, f32>,
        reference: Bound<'py, Dataset>,
        weight: Option<PyReadonlyArray1<'py, f32>>,
        group: Option<PyReadonlyArray1<'py, u32>>,
        collective: Option<Bound<'py, Collective>>,
    ) -> PyResult<Dataset> {
        let matrix = Matrix::extract(data)?;
        let metadata = Metadata { label, weight, group };
        let reference = &reference.get().inner;
        let names = reference.features.iter().map(|f| f.name.clone()).collect();
        let categorical = reference
            .features
            .iter()
            .enumerate()
            .filter(|(_, f)| f.is_categorical())
            .map(|(i, _)| i);
        let input = metadata.input(matrix.view()?, Some(names), categorical.collect())?;
        let collective = collective.as_ref().map(Bound::get);
        let inner = py
            .detach(|| {
                with_collective(collective, |c| {
                    BinnedDataset::construct_with_reference(&input, reference, c)
                })
            })
            .into_py_result()?;
        Ok(Dataset { inner })
    }

    #[getter]
    fn num_rows(&self) -> usize {
        self.inner.num_rows
    }

    #[getter]
    fn num_features(&self) -> usize {
        self.inner.features.len()
    }

    #[getter]
    fn feature_names(&self) -> Vec<String> {
        self.inner.features.iter().map(|f| f.name.clone()).collect()
    }
}
