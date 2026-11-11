//! The backend's entry points, as `tritonbackend.h` declares them and the backend repository's `README.md`
//! describes their lifecycle: `TRITONBACKEND_Initialize` once per process, `TRITONBACKEND_ModelInitialize` and
//! `TRITONBACKEND_ModelFinalize` once per model version (loading and compiling its `Predictor`, which every
//! instance shares: it is immutable and `Sync`), `TRITONBACKEND_ModelInstanceInitialize` and
//! `TRITONBACKEND_ModelInstanceFinalize` per instance, and `TRITONBACKEND_ModelInstanceExecute` per batch of
//! requests, which Triton never calls concurrently for one instance.
//!
//! Every entry point runs under `catch_unwind`: a panic becomes a `TRITONSERVER_Error`, never an unwind into C.

use std::ffi::CString;
use std::panic::{self, AssertUnwindSafe};
use std::slice;
use std::sync::{Arc, Mutex, PoisonError};

use silvic_serve::{Kernel, Predictor};

use crate::batch::{self, RequestInput, Scratch};
use crate::config::{self, Parameters, Settings};
use crate::error::{Error, Result};
use crate::triton::{
    self, Backend, BackendAttribute, BackendAttributes, InstanceHandle, LogLevel, Model, ModelHandle, ModelInstance,
    Request, RequestHandle, Timestamps, TritonError,
};

/// A model version: its compiled predictor and validated configuration.
struct ModelState {
    predictor: Predictor,
    settings: Settings,
    input_name: CString,
    output_name: CString,
}

/// A model instance: the shared model, and the instance's buffer for copied inputs.
struct InstanceState {
    model: Arc<ModelState>,
    name: String,
    scratch: Mutex<Scratch>,
}

/// Runs an entry point's body, turning its error or panic into a Triton error (null on success).
fn guard(body: impl FnOnce() -> Result<()>) -> *mut TritonError {
    let error = match panic::catch_unwind(AssertUnwindSafe(body)) {
        Ok(Ok(())) => return std::ptr::null_mut(),
        Ok(Err(error)) => error,
        Err(payload) => Error::panic(&*payload),
    };
    triton::new_error(&error)
}

/// Checks the server's backend API: major 1, minor at least `API_VERSION_MINOR_REQUIRED`.
///
/// # Safety
/// Called by Triton once, after loading the library.
#[no_mangle]
pub unsafe extern "C" fn TRITONBACKEND_Initialize(_backend: *mut Backend) -> *mut TritonError {
    guard(|| {
        let (major, minor) = triton::api_version()?;
        if major != triton::API_VERSION_MAJOR || minor < triton::API_VERSION_MINOR_REQUIRED {
            return Err(Error::unsupported(format!(
                "Triton backend API {major}.{minor}; the silvic backend needs {}.{} or newer",
                triton::API_VERSION_MAJOR,
                triton::API_VERSION_MINOR_REQUIRED
            )));
        }
        triton::log(
            LogLevel::Info,
            &format!(
                "silvic backend {} on backend API {major}.{minor}; kernels {}, {} detected",
                env!("CARGO_PKG_VERSION"),
                Kernel::available()
                    .iter()
                    .map(|kernel| kernel.name())
                    .collect::<Vec<_>>()
                    .join(", "),
                Kernel::detect()
            ),
        );
        Ok(())
    })
}

/// Prefers CPU instances, so a config without `instance_group` gets CPU instances on a host with GPUs too.
///
/// # Safety
/// Called by Triton with the backend's attributes object.
#[no_mangle]
pub unsafe extern "C" fn TRITONBACKEND_GetBackendAttribute(
    _backend: *mut Backend,
    attributes: *mut BackendAttribute,
) -> *mut TritonError {
    guard(|| {
        // SAFETY: the handle Triton passed to this call.
        let attributes = unsafe { BackendAttributes::new(attributes) }?;
        attributes.prefer_instance_group(triton::KIND_CPU)
    })
}

/// Loads `<model>/<version>/<model_file>`, compiles it, auto-completes and validates the configuration.
///
/// # Safety
/// Called by Triton with a model it is loading.
#[no_mangle]
pub unsafe extern "C" fn TRITONBACKEND_ModelInitialize(model: *mut Model) -> *mut TritonError {
    guard(|| {
        // SAFETY: the handle Triton passed to this call.
        let model = unsafe { ModelHandle::new(model) }?;
        let state = load(&model)?;
        model.set_state(Box::new(Arc::new(state)))
    })
}

fn load(model: &ModelHandle) -> Result<ModelState> {
    let name = model.name()?;
    let version = model.version()?;
    let mut config = model.config()?;
    let parameters = Parameters::from_config(&config)?;
    let silvic_model = config::load_model(&model.repository()?, version, &parameters)?;
    let predictor = match parameters.kernel {
        Some(kernel) => Predictor::with_kernel(&silvic_model, kernel)?,
        None => Predictor::new(&silvic_model)?,
    };
    if model.auto_complete_config()? && config::auto_complete(&mut config, silvic_model.num_features())? {
        model.set_config(&config)?;
        config = model.config()?;
    }
    let settings = Settings::new(&config, &parameters, &silvic_model)?;
    triton::log(
        LogLevel::Info,
        &format!(
            "silvic model {name} version {version}: {} trees ({} walked), {} features, {} kernel, {} in, {} out, \
             max_batch_size {}, raw_score {}, num_threads {}",
            predictor.num_trees(),
            predictor.num_walked_trees(),
            settings.num_features,
            predictor.kernel(),
            settings.input_type.config_name(),
            settings.output_type.config_name(),
            settings.max_batch_size,
            settings.raw_score,
            settings.num_threads,
        ),
    );
    let c_name = |name: &str| CString::new(name).map_err(|_| Error::config(format!("tensor name {name:?} holds NUL")));
    Ok(ModelState {
        input_name: c_name(&settings.input_name)?,
        output_name: c_name(&settings.output_name)?,
        predictor,
        settings,
    })
}

/// Drops the model version's state; Triton has finalized its instances.
///
/// # Safety
/// Called by Triton with a model it is unloading.
#[no_mangle]
pub unsafe extern "C" fn TRITONBACKEND_ModelFinalize(model: *mut Model) -> *mut TritonError {
    guard(|| {
        // SAFETY: the handle Triton passed to this call, whose state `ModelInitialize` stored as `Arc<ModelState>`.
        drop(unsafe { ModelHandle::new(model)?.take_state::<Arc<ModelState>>() }?);
        Ok(())
    })
}

/// Starts a CPU instance of a loaded model. `KIND_AUTO` and `KIND_MODEL` run on the CPU; `KIND_GPU` is an error.
///
/// # Safety
/// Called by Triton with an instance of a model `TRITONBACKEND_ModelInitialize` loaded.
#[no_mangle]
pub unsafe extern "C" fn TRITONBACKEND_ModelInstanceInitialize(instance: *mut ModelInstance) -> *mut TritonError {
    guard(|| {
        // SAFETY: the handle Triton passed to this call.
        let instance = unsafe { InstanceHandle::new(instance) }?;
        let name = instance.name()?;
        match instance.kind()? {
            triton::KIND_CPU | triton::KIND_AUTO | triton::KIND_MODEL => {}
            triton::KIND_GPU => {
                return Err(Error::unsupported(format!(
                    "instance {name} is KIND_GPU; the silvic backend scores on the CPU: use KIND_CPU"
                )))
            }
            kind => {
                return Err(Error::unsupported(format!(
                    "instance {name} has instance group kind {kind}"
                )))
            }
        }
        // SAFETY: `ModelInitialize` stored an `Arc<ModelState>`, which lives until `ModelFinalize`, after this.
        let model = Arc::clone(unsafe { instance.model()?.state::<Arc<ModelState>>() }?);
        instance.set_state(Box::new(InstanceState {
            model,
            name,
            scratch: Mutex::default(),
        }))
    })
}

/// Drops the instance's state.
///
/// # Safety
/// Called by Triton with an instance it is unloading.
#[no_mangle]
pub unsafe extern "C" fn TRITONBACKEND_ModelInstanceFinalize(instance: *mut ModelInstance) -> *mut TritonError {
    guard(|| {
        // SAFETY: the handle Triton passed to this call, whose state `ModelInstanceInitialize` stored.
        drop(unsafe { InstanceHandle::new(instance)?.take_state::<InstanceState>() }?);
        Ok(())
    })
}

/// Answers every request with one response (scores, or the error that request alone caused), reports statistics
/// and releases the requests. It returns an error only before it takes the requests, so Triton keeps them then.
///
/// # Safety
/// Called by Triton with an initialized instance and `request_count` requests.
#[no_mangle]
pub unsafe extern "C" fn TRITONBACKEND_ModelInstanceExecute(
    instance: *mut ModelInstance,
    requests: *mut *mut Request,
    request_count: u32,
) -> *mut TritonError {
    guard(|| {
        let exec_start = triton::now_ns();
        // SAFETY: the handle Triton passed to this call.
        let instance = unsafe { InstanceHandle::new(instance) }?;
        // SAFETY: `ModelInstanceInitialize` stored an `InstanceState`, which lives until `ModelInstanceFinalize`.
        let state = unsafe { instance.state::<InstanceState>() }?;
        let pointers = if request_count == 0 || requests.is_null() {
            &[][..]
        } else {
            // SAFETY: Triton passes `request_count` request pointers; they are copied out, as the array is Triton's.
            unsafe { slice::from_raw_parts(requests, request_count as usize) }
        };
        // SAFETY: each a request of this call, wrapped once; the backend owns them from here.
        let requests: Vec<RequestHandle> = pointers
            .iter()
            .filter_map(|&ptr| unsafe { RequestHandle::new(ptr) })
            .collect();
        execute(&instance, state, requests, exec_start);
        Ok(())
    })
}

fn execute(instance: &InstanceHandle, state: &InstanceState, requests: Vec<RequestHandle>, exec_start: u64) {
    let model = &*state.model;
    let settings = &model.settings;
    let mut times = Timestamps {
        exec_start,
        ..Timestamps::default()
    };
    let (successes, batch_size) = {
        let inputs: Vec<Result<RequestInput<'_>>> = requests
            .iter()
            .map(|request| batch::check_input(settings, request.input(&model.input_name)?))
            .collect();
        times.compute_start = triton::now_ns();
        let scores = {
            // The scratch is rewritten before each use, so a poisoning panic left nothing to distrust in it.
            let mut scratch = state.scratch.lock().unwrap_or_else(PoisonError::into_inner);
            batch::score(&model.predictor, settings, &inputs, &mut scratch)
        };
        times.compute_end = triton::now_ns();
        let successes: Vec<bool> = requests
            .iter()
            .zip(&inputs)
            .enumerate()
            .map(|(index, (request, input))| {
                let scored = input.clone().and_then(|input| Ok((input.rows, scores.request(index)?)));
                respond(state, request, scored)
            })
            .collect();
        let batch_size = inputs
            .iter()
            .zip(&successes)
            .filter(|(_, &success)| success)
            .map(|(input, _)| {
                if settings.batching() {
                    input.as_ref().map_or(0, |input| input.rows)
                } else {
                    1
                }
            })
            .sum::<usize>();
        (successes, batch_size)
    };
    times.exec_end = triton::now_ns();
    for (request, success) in requests.into_iter().zip(successes) {
        if let Err(error) = instance.report(&request, success, &times) {
            log_error(state, "reporting request statistics", &error);
        }
        if let Err(error) = request.release() {
            log_error(state, "releasing a request", &error);
        }
    }
    if batch_size > 0 {
        if let Err(error) = instance.report_batch(batch_size as u64, &times) {
            log_error(state, "reporting batch statistics", &error);
        }
    }
}

/// Sends one request's response; returns whether it carried scores.
fn respond(state: &InstanceState, request: &RequestHandle, scored: Result<(usize, &[f64])>) -> bool {
    let model = &*state.model;
    let sent = match scored {
        Ok((rows, scores)) => request.send_scores(
            &model.output_name,
            model.settings.output_type,
            model.settings.output_shape(rows).as_slice(),
            scores,
        ),
        Err(error) => {
            if let Err(send_error) = request.send_error(&error) {
                log_error(state, "sending an error response", &send_error);
            }
            return false;
        }
    };
    if let Err(error) = &sent {
        log_error(state, "sending a response", error);
    }
    sent.is_ok()
}

fn log_error(state: &InstanceState, action: &str, error: &Error) {
    triton::log(
        LogLevel::Error,
        &format!("silvic instance {}: {action}: {error} ({})", state.name, error.kind()),
    );
}
