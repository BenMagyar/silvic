//! `silvic_net`'s TCP ring for Python: every rank binds a [`Listener`], the driver gathers the addresses, and every
//! rank [`connect`]s with the full list and a job token. The resulting [`Collective`] is passed to dataset
//! construction and training, which call the same collectives in the same order on every rank.

use std::sync::Mutex;
use std::time::Duration;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use silvic_core::{Collective as CoreCollective, LocalCollective, SilvicError};
use silvic_net::TcpCollective;

use crate::error::IntoPyResult;

/// A bound TCP listener waiting for its peers; `connect` consumes it.
#[pyclass(frozen, module = "silvic._silvic")]
pub struct Listener {
    inner: Mutex<Option<silvic_net::Listener>>,
    address: String,
}

#[pymethods]
impl Listener {
    /// Binds an ephemeral port on `host` and advertises `host:port`. `advertise` names the address peers connect to
    /// when `host` is a wildcard such as `0.0.0.0`.
    #[staticmethod]
    #[pyo3(signature = (host, advertise=None))]
    fn bind(py: Python<'_>, host: &str, advertise: Option<&str>) -> PyResult<Listener> {
        let listener = py
            .detach(|| match advertise {
                Some(advertised) => TcpCollective::bind_advertised(host, advertised),
                None => TcpCollective::bind(host),
            })
            .into_py_result()?;
        let address = listener.address();
        Ok(Listener {
            inner: Mutex::new(Some(listener)),
            address,
        })
    }

    /// `host:port` the peers connect to.
    #[getter]
    fn address(&self) -> &str {
        &self.address
    }
}

/// The collectives of one rank of a job (`LocalCollective` when built without peers).
#[pyclass(frozen, module = "silvic._silvic")]
pub struct Collective {
    inner: Mutex<Box<dyn CoreCollective>>,
    rank: usize,
    world_size: usize,
}

impl Collective {
    /// Runs `f` on the rank's collective; a poisoned lock means an earlier call panicked mid-collective, after which
    /// the ring is in an unknown state.
    pub fn with<T>(&self, f: impl FnOnce(&mut dyn CoreCollective) -> silvic_core::Result<T>) -> silvic_core::Result<T> {
        let mut guard = self
            .inner
            .lock()
            .map_err(|_| SilvicError::collective("an earlier collective call panicked"))?;
        f(guard.as_mut())
    }
}

#[pymethods]
impl Collective {
    /// A world of one rank.
    #[staticmethod]
    fn local() -> Collective {
        Collective {
            inner: Mutex::new(Box::new(LocalCollective)),
            rank: 0,
            world_size: 1,
        }
    }

    #[getter]
    fn rank(&self) -> usize {
        self.rank
    }

    #[getter]
    fn world_size(&self) -> usize {
        self.world_size
    }
}

/// Connects rank `rank` of a job whose ranks listen on `addresses` (indexed by rank). Every rank passes the same
/// `job_token`, so a stray connection from another job is refused. Blocks until the ring is up or `timeout_s` passes.
#[pyfunction]
#[pyo3(signature = (listener, rank, addresses, job_token, timeout_s=300.0))]
pub fn connect(
    py: Python<'_>,
    listener: &Bound<'_, Listener>,
    rank: usize,
    addresses: Vec<String>,
    job_token: u64,
    timeout_s: f64,
) -> PyResult<Collective> {
    let timeout = Duration::try_from_secs_f64(timeout_s)
        .map_err(|_| PyValueError::new_err(format!("timeout_s must be a non-negative number, got {timeout_s}")))?;
    let world_size = addresses.len();
    let bound = listener
        .get()
        .inner
        .lock()
        .map_err(|_| PyValueError::new_err("listener lock poisoned"))?
        .take()
        .ok_or_else(|| PyValueError::new_err("listener is already connected"))?;
    let collective = py
        .detach(|| TcpCollective::connect(bound, rank, &addresses, job_token, timeout))
        .into_py_result()?;
    Ok(Collective {
        inner: Mutex::new(Box::new(collective)),
        rank,
        world_size,
    })
}
