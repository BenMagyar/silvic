//! Python exceptions for [`SilvicError`]: `SilvicError(Exception)` is the base, configuration and data errors also
//! subclass `ValueError` and every other kind `RuntimeError`, the way LightGBM's Python package raises `ValueError`
//! for bad input and `LightGBMError` for the rest. An interrupted run raises `KeyboardInterrupt`.
//!
//! The classes are built at module init with `type(name, bases, namespace)` because PyO3's `create_exception!` takes
//! one base and these need two. Their `__module__` is `silvic`, which re-exports them, so they pickle across Ray.

use pyo3::exceptions::{PyException, PyKeyboardInterrupt, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyDict, PyTuple, PyType};
use silvic_core::SilvicError;

/// The exception classes, in [`KINDS`] order after the base.
struct Classes {
    base: Py<PyType>,
    kinds: Vec<Py<PyType>>,
}

/// `(SilvicError::kind(), class name, whether the class subclasses ValueError)`; `interrupted` is not here because
/// it raises `KeyboardInterrupt`.
const KINDS: &[(&str, &str, bool)] = &[
    ("config", "ConfigError", true),
    ("data", "DataError", true),
    ("device", "DeviceError", false),
    ("collective", "CollectiveError", false),
    ("model", "ModelError", false),
    ("unsupported", "UnsupportedError", false),
];

static CLASSES: PyOnceLock<Classes> = PyOnceLock::new();

fn make_class<'py>(py: Python<'py>, name: &str, bases: Bound<'py, PyTuple>, doc: &str) -> PyResult<Bound<'py, PyType>> {
    let namespace = PyDict::new(py);
    namespace.set_item("__module__", "silvic")?;
    namespace.set_item("__doc__", doc)?;
    let class = py.get_type::<PyType>().call1((name, bases, namespace))?;
    Ok(class.cast_into::<PyType>()?)
}

fn build_classes(py: Python<'_>) -> PyResult<Classes> {
    let base = make_class(
        py,
        "SilvicError",
        PyTuple::new(py, [py.get_type::<PyException>()])?,
        "Base class of every error silvic's training, data and model code raises.",
    )?;
    let mut kinds = Vec::with_capacity(KINDS.len());
    for (kind, name, is_value_error) in KINDS {
        let builtin = if *is_value_error {
            py.get_type::<PyValueError>()
        } else {
            py.get_type::<PyRuntimeError>()
        };
        let doc = format!("A silvic `{kind}` error.");
        kinds.push(make_class(py, name, PyTuple::new(py, [base.clone(), builtin])?, &doc)?.unbind());
    }
    Ok(Classes {
        base: base.unbind(),
        kinds,
    })
}

fn classes(py: Python<'_>) -> PyResult<&Classes> {
    CLASSES.get_or_try_init(py, || build_classes(py))
}

/// Adds the exception classes to the module.
pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = module.py();
    let classes = classes(py)?;
    module.add("SilvicError", classes.base.bind(py))?;
    for ((_, name, _), class) in KINDS.iter().zip(&classes.kinds) {
        module.add(*name, class.bind(py))?;
    }
    Ok(())
}

/// The Python exception for `err`.
pub fn to_py_err(err: &SilvicError) -> PyErr {
    let message = err.to_string();
    if matches!(err, SilvicError::Interrupted(_)) {
        return PyKeyboardInterrupt::new_err(message);
    }
    Python::attach(|py| match classes(py) {
        Ok(classes) => {
            let index = KINDS.iter().position(|(kind, _, _)| *kind == err.kind());
            let class = index.map_or(&classes.base, |i| &classes.kinds[i]);
            PyErr::from_type(class.bind(py).clone(), message)
        }
        Err(init_failure) => init_failure,
    })
}

/// `?`-friendly conversion of a core result.
pub trait IntoPyResult<T> {
    fn into_py_result(self) -> PyResult<T>;
}

impl<T> IntoPyResult<T> for silvic_core::Result<T> {
    fn into_py_result(self) -> PyResult<T> {
        self.map_err(|err| to_py_err(&err))
    }
}
