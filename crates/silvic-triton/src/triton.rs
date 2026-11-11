//! The part of Triton's C API the backend calls, declared by hand from `tritonserver.h` and `tritonbackend.h` in
//! triton-inference-server/core r24.10 (backend API 1.19, server API 1.34), over opaque handle types, with safe
//! wrappers around handles Triton passed in.
//!
//! Every function declared here has had this signature since backend API 1.0 (core r21.02) except
//! `TRITONBACKEND_BackendAttributeAddPreferredInstanceGroup`, added in 1.10 (core r22.09), so the backend asks
//! for API 1.10 or newer. C enums are passed as `u32`: the headers' enums are `int`-sized with small non-negative
//! values, which the C ABIs of x86-64 and aarch64 pass as they pass a `uint32_t`.

use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::marker::{PhantomData, PhantomPinned};
use std::panic::Location;
use std::path::PathBuf;
use std::ptr::{self, NonNull};
use std::slice;

use serde_json::Value;

use crate::batch::{self, InputTensor};
use crate::config::DataType;
use crate::error::{Error, Result};

macro_rules! opaque {
    ($($name:ident = $c_name:literal),* $(,)?) => {
        $(
            #[doc = concat!("`", $c_name, "`, owned by Triton.")]
            #[repr(C)]
            pub struct $name {
                _data: (),
                _marker: PhantomData<(*mut u8, PhantomPinned)>,
            }
        )*
    };
}

opaque!(
    TritonError = "TRITONSERVER_Error",
    Message = "TRITONSERVER_Message",
    Backend = "TRITONBACKEND_Backend",
    BackendAttribute = "TRITONBACKEND_BackendAttribute",
    Model = "TRITONBACKEND_Model",
    ModelInstance = "TRITONBACKEND_ModelInstance",
    Request = "TRITONBACKEND_Request",
    Response = "TRITONBACKEND_Response",
    Input = "TRITONBACKEND_Input",
    Output = "TRITONBACKEND_Output",
);

/// `TRITONBACKEND_API_VERSION_MAJOR`.
pub const API_VERSION_MAJOR: u32 = 1;
/// The oldest backend API minor version with every call the backend makes (see the module docs).
pub const API_VERSION_MINOR_REQUIRED: u32 = 10;

/// `TRITONSERVER_Error_Code` values.
const ERROR_INTERNAL: u32 = 1;
const ERROR_NOT_FOUND: u32 = 2;
const ERROR_INVALID_ARG: u32 = 3;
const ERROR_UNSUPPORTED: u32 = 5;

/// `TRITONSERVER_DataType` values.
const TYPE_FP32: u32 = 11;
const TYPE_FP64: u32 = 12;

/// `TRITONSERVER_MemoryType` values.
const MEMORY_CPU: u32 = 0;
const MEMORY_GPU: u32 = 2;

/// `TRITONSERVER_InstanceGroupKind` values.
pub const KIND_AUTO: u32 = 0;
pub const KIND_CPU: u32 = 1;
pub const KIND_GPU: u32 = 2;
pub const KIND_MODEL: u32 = 3;

/// `TRITONSERVER_LogLevel` values.
#[derive(Debug, Clone, Copy)]
#[repr(u32)]
pub enum LogLevel {
    Info = 0,
    Error = 2,
}

/// `TRITONBACKEND_ARTIFACT_FILESYSTEM`.
const ARTIFACT_FILESYSTEM: u32 = 0;
/// `TRITONSERVER_REQUEST_RELEASE_ALL`.
const REQUEST_RELEASE_ALL: u32 = 1;
/// `TRITONSERVER_RESPONSE_COMPLETE_FINAL`.
const RESPONSE_COMPLETE_FINAL: u32 = 1;
/// The model configuration format `TRITONBACKEND_ModelConfig` and `TRITONBACKEND_ModelSetConfig` support.
const CONFIG_VERSION: u32 = 1;

/// What a Triton call returns: null for success, else an error the caller owns. Only Triton makes one (it is never
/// built in Rust), which is what makes `check` sound.
#[repr(transparent)]
#[must_use]
struct Status(*mut TritonError);

extern "C" {
    fn TRITONSERVER_ErrorNew(code: u32, msg: *const c_char) -> *mut TritonError;
    fn TRITONSERVER_ErrorDelete(error: *mut TritonError);
    fn TRITONSERVER_ErrorCode(error: *mut TritonError) -> u32;
    fn TRITONSERVER_ErrorMessage(error: *mut TritonError) -> *const c_char;
    fn TRITONSERVER_LogMessage(level: u32, filename: *const c_char, line: c_int, msg: *const c_char) -> Status;
    fn TRITONSERVER_MessageNewFromSerializedJson(
        message: *mut *mut Message,
        base: *const c_char,
        byte_size: usize,
    ) -> Status;
    fn TRITONSERVER_MessageSerializeToJson(
        message: *mut Message,
        base: *mut *const c_char,
        byte_size: *mut usize,
    ) -> Status;
    fn TRITONSERVER_MessageDelete(message: *mut Message) -> Status;

    fn TRITONBACKEND_ApiVersion(major: *mut u32, minor: *mut u32) -> Status;
    fn TRITONBACKEND_BackendAttributeAddPreferredInstanceGroup(
        attributes: *mut BackendAttribute,
        kind: u32,
        count: u64,
        device_ids: *const u64,
        id_count: u64,
    ) -> Status;

    fn TRITONBACKEND_ModelName(model: *mut Model, name: *mut *const c_char) -> Status;
    fn TRITONBACKEND_ModelVersion(model: *mut Model, version: *mut u64) -> Status;
    fn TRITONBACKEND_ModelRepository(
        model: *mut Model,
        artifact_type: *mut u32,
        location: *mut *const c_char,
    ) -> Status;
    fn TRITONBACKEND_ModelConfig(model: *mut Model, config_version: u32, config: *mut *mut Message) -> Status;
    fn TRITONBACKEND_ModelAutoCompleteConfig(model: *mut Model, auto_complete: *mut bool) -> Status;
    fn TRITONBACKEND_ModelSetConfig(model: *mut Model, config_version: u32, config: *mut Message) -> Status;
    fn TRITONBACKEND_ModelState(model: *mut Model, state: *mut *mut c_void) -> Status;
    fn TRITONBACKEND_ModelSetState(model: *mut Model, state: *mut c_void) -> Status;

    fn TRITONBACKEND_ModelInstanceName(instance: *mut ModelInstance, name: *mut *const c_char) -> Status;
    fn TRITONBACKEND_ModelInstanceKind(instance: *mut ModelInstance, kind: *mut u32) -> Status;
    fn TRITONBACKEND_ModelInstanceModel(instance: *mut ModelInstance, model: *mut *mut Model) -> Status;
    fn TRITONBACKEND_ModelInstanceState(instance: *mut ModelInstance, state: *mut *mut c_void) -> Status;
    fn TRITONBACKEND_ModelInstanceSetState(instance: *mut ModelInstance, state: *mut c_void) -> Status;
    fn TRITONBACKEND_ModelInstanceReportStatistics(
        instance: *mut ModelInstance,
        request: *mut Request,
        success: bool,
        exec_start_ns: u64,
        compute_start_ns: u64,
        compute_end_ns: u64,
        exec_end_ns: u64,
    ) -> Status;
    fn TRITONBACKEND_ModelInstanceReportBatchStatistics(
        instance: *mut ModelInstance,
        batch_size: u64,
        exec_start_ns: u64,
        compute_start_ns: u64,
        compute_end_ns: u64,
        exec_end_ns: u64,
    ) -> Status;

    fn TRITONBACKEND_RequestInput(request: *mut Request, name: *const c_char, input: *mut *mut Input) -> Status;
    fn TRITONBACKEND_InputProperties(
        input: *mut Input,
        name: *mut *const c_char,
        datatype: *mut u32,
        shape: *mut *const i64,
        dims_count: *mut u32,
        byte_size: *mut u64,
        buffer_count: *mut u32,
    ) -> Status;
    fn TRITONBACKEND_InputBuffer(
        input: *mut Input,
        index: u32,
        buffer: *mut *const c_void,
        buffer_byte_size: *mut u64,
        memory_type: *mut u32,
        memory_type_id: *mut i64,
    ) -> Status;
    fn TRITONBACKEND_RequestRelease(request: *mut Request, release_flags: u32) -> Status;

    fn TRITONBACKEND_ResponseNew(response: *mut *mut Response, request: *mut Request) -> Status;
    fn TRITONBACKEND_ResponseOutput(
        response: *mut Response,
        output: *mut *mut Output,
        name: *const c_char,
        datatype: u32,
        shape: *const i64,
        dims_count: u32,
    ) -> Status;
    fn TRITONBACKEND_OutputBuffer(
        output: *mut Output,
        buffer: *mut *mut c_void,
        buffer_byte_size: u64,
        memory_type: *mut u32,
        memory_type_id: *mut i64,
    ) -> Status;
    fn TRITONBACKEND_ResponseSend(response: *mut Response, send_flags: u32, error: *mut TritonError) -> Status;
}

/// Turns what a Triton call returned into a `Result`, deleting the error.
fn check(status: Status) -> Result<()> {
    let error = status.0;
    if error.is_null() {
        return Ok(());
    }
    // SAFETY: a non-null `Status` is a live error the backend owns, deleted once here; its message lives until then.
    unsafe {
        let code = TRITONSERVER_ErrorCode(error);
        let message = c_string(TRITONSERVER_ErrorMessage(error));
        TRITONSERVER_ErrorDelete(error);
        Err(Error::Server {
            code,
            message: message.into(),
        })
    }
}

/// Copies a string Triton returned; null reads as empty.
///
/// # Safety
/// `ptr` is null or a NUL-terminated string that stays valid during the call.
unsafe fn c_string(ptr: *const c_char) -> String {
    if ptr.is_null() {
        return String::new();
    }
    // SAFETY: per the function's contract.
    unsafe { CStr::from_ptr(ptr) }.to_string_lossy().into_owned()
}

/// Interior NULs, which a C string cannot hold, become spaces.
pub fn to_c_string(text: &str) -> CString {
    CString::new(text.replace('\0', " ")).expect("no interior NUL is left")
}

/// A new Triton error for `error`, which Triton (or the caller) then owns.
pub fn new_error(error: &Error) -> *mut TritonError {
    let code = match error {
        Error::Config(_) | Error::Model(_) | Error::Request(_) => ERROR_INVALID_ARG,
        Error::Io { .. } => ERROR_NOT_FOUND,
        Error::Unsupported(_) => ERROR_UNSUPPORTED,
        Error::Server { code, .. } => *code,
        Error::Panic(_) => ERROR_INTERNAL,
    };
    let message = to_c_string(&error.to_string());
    // SAFETY: `message` is a valid C string; Triton copies it.
    unsafe { TRITONSERVER_ErrorNew(code, message.as_ptr()) }
}

/// Logs through Triton's logger, attributed to the caller's file and line. A failure to log is dropped.
#[track_caller]
pub fn log(level: LogLevel, message: &str) {
    let location = Location::caller();
    let file = to_c_string(location.file());
    let message = to_c_string(message);
    let line = c_int::try_from(location.line()).unwrap_or(c_int::MAX);
    // SAFETY: both strings are valid C strings for the call.
    let error = unsafe { TRITONSERVER_LogMessage(level as u32, file.as_ptr(), line, message.as_ptr()) };
    let _ = check(error);
}

/// The server's backend API version, `(major, minor)`.
pub fn api_version() -> Result<(u32, u32)> {
    let (mut major, mut minor) = (0, 0);
    // SAFETY: both out-pointers are valid for writes.
    check(unsafe { TRITONBACKEND_ApiVersion(&mut major, &mut minor) })?;
    Ok((major, minor))
}

/// `std::chrono::steady_clock` in nanoseconds, the clock Triton's statistics timestamps use: libstdc++ reads
/// `CLOCK_MONOTONIC` for it.
pub fn now_ns() -> u64 {
    let mut time = libc::timespec { tv_sec: 0, tv_nsec: 0 };
    // SAFETY: `time` is valid for writes; CLOCK_MONOTONIC always exists on Linux and macOS.
    unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut time) };
    (time.tv_sec as u64) * 1_000_000_000 + time.tv_nsec as u64
}

fn data_type(code: u32) -> Option<DataType> {
    match code {
        TYPE_FP32 => Some(DataType::Fp32),
        TYPE_FP64 => Some(DataType::Fp64),
        _ => None,
    }
}

fn type_code(data_type: DataType) -> u32 {
    match data_type {
        DataType::Fp32 => TYPE_FP32,
        DataType::Fp64 => TYPE_FP64,
    }
}

/// The backend attributes Triton asks for in `TRITONBACKEND_GetBackendAttribute`.
pub struct BackendAttributes(NonNull<BackendAttribute>);

impl BackendAttributes {
    /// # Safety
    /// `attributes` is the handle Triton passed to the current `TRITONBACKEND_GetBackendAttribute` call.
    pub unsafe fn new(attributes: *mut BackendAttribute) -> Result<BackendAttributes> {
        NonNull::new(attributes)
            .map(BackendAttributes)
            .ok_or_else(|| Error::unsupported("Triton passed no backend attributes"))
    }

    /// Prefers `kind` instances, Triton's default count, on Triton's default devices.
    pub fn prefer_instance_group(&self, kind: u32) -> Result<()> {
        // SAFETY: a live handle (see `new`); no device ids.
        check(unsafe {
            TRITONBACKEND_BackendAttributeAddPreferredInstanceGroup(self.0.as_ptr(), kind, 0, ptr::null(), 0)
        })
    }
}

/// A model version, between `TRITONBACKEND_ModelInitialize` and `TRITONBACKEND_ModelFinalize`.
pub struct ModelHandle(NonNull<Model>);

impl ModelHandle {
    /// # Safety
    /// `model` is a model handle Triton passed to the current backend call, or got from its instance.
    pub unsafe fn new(model: *mut Model) -> Result<ModelHandle> {
        NonNull::new(model)
            .map(ModelHandle)
            .ok_or_else(|| Error::unsupported("Triton passed no model"))
    }

    pub fn name(&self) -> Result<String> {
        let mut name = ptr::null();
        // SAFETY: a live handle; Triton owns the returned string for the model's lifetime.
        check(unsafe { TRITONBACKEND_ModelName(self.0.as_ptr(), &mut name) })?;
        // SAFETY: as above.
        Ok(unsafe { c_string(name) })
    }

    pub fn version(&self) -> Result<u64> {
        let mut version = 0;
        // SAFETY: a live handle; `version` is valid for writes.
        check(unsafe { TRITONBACKEND_ModelVersion(self.0.as_ptr(), &mut version) })?;
        Ok(version)
    }

    /// The model's directory in the repository (holding `config.pbtxt` and the version directories).
    pub fn repository(&self) -> Result<PathBuf> {
        let (mut artifact_type, mut location) = (0, ptr::null());
        // SAFETY: a live handle; Triton owns the returned string for the model's lifetime.
        check(unsafe { TRITONBACKEND_ModelRepository(self.0.as_ptr(), &mut artifact_type, &mut location) })?;
        if artifact_type != ARTIFACT_FILESYSTEM {
            return Err(Error::unsupported("model repositories other than a filesystem"));
        }
        // SAFETY: as above.
        Ok(PathBuf::from(unsafe { c_string(location) }))
    }

    /// The model configuration, as Triton's `ModelConfig` JSON.
    pub fn config(&self) -> Result<Value> {
        let mut message = ptr::null_mut();
        // SAFETY: a live handle; the message is ours to delete.
        check(unsafe { TRITONBACKEND_ModelConfig(self.0.as_ptr(), CONFIG_VERSION, &mut message) })?;
        // SAFETY: the live message Triton just returned.
        let config = unsafe { message_json(message) };
        // SAFETY: `message` is the live message Triton just returned, deleted once.
        check(unsafe { TRITONSERVER_MessageDelete(message) })?;
        config
    }

    /// Whether Triton lets the backend fill in the configuration.
    pub fn auto_complete_config(&self) -> Result<bool> {
        let mut auto_complete = false;
        // SAFETY: a live handle; `auto_complete` is valid for writes.
        check(unsafe { TRITONBACKEND_ModelAutoCompleteConfig(self.0.as_ptr(), &mut auto_complete) })?;
        Ok(auto_complete)
    }

    /// Replaces the configuration; only valid in `TRITONBACKEND_ModelInitialize`.
    pub fn set_config(&self, config: &Value) -> Result<()> {
        let json = config.to_string();
        let mut message = ptr::null_mut();
        // SAFETY: `json` outlives the call, which copies it into the new message.
        check(unsafe { TRITONSERVER_MessageNewFromSerializedJson(&mut message, json.as_ptr().cast(), json.len()) })?;
        // SAFETY: a live handle and message; Triton does not take the message.
        let set = check(unsafe { TRITONBACKEND_ModelSetConfig(self.0.as_ptr(), CONFIG_VERSION, message) });
        // SAFETY: deleted once, after its last use.
        check(unsafe { TRITONSERVER_MessageDelete(message) })?;
        set
    }

    /// Stores `state` with the model, for `state` and `take_state` to read.
    pub fn set_state<T>(&self, state: Box<T>) -> Result<()> {
        let state = Box::into_raw(state);
        // SAFETY: a live handle; Triton stores the pointer.
        let set = check(unsafe { TRITONBACKEND_ModelSetState(self.0.as_ptr(), state.cast()) });
        if set.is_err() {
            // SAFETY: Triton did not keep the pointer, which came from `Box::into_raw` above.
            drop(unsafe { Box::from_raw(state) });
        }
        set
    }

    /// The state `set_state` stored.
    ///
    /// # Safety
    /// The state was stored by `set_state::<T>` and not yet taken; the reference is used only while the model lives.
    pub unsafe fn state<'a, T>(&self) -> Result<&'a T> {
        let mut state = ptr::null_mut();
        // SAFETY: a live handle; `state` is valid for writes.
        check(unsafe { TRITONBACKEND_ModelState(self.0.as_ptr(), &mut state) })?;
        // SAFETY: per the function's contract, `state` is null or a live `Box<T>`.
        unsafe { state.cast::<T>().as_ref() }.ok_or_else(|| Error::unsupported("the model has no state"))
    }

    /// Takes back the state `set_state` stored, leaving none.
    ///
    /// # Safety
    /// The state was stored by `set_state::<T>`, and no reference from `state` is used afterwards.
    pub unsafe fn take_state<T>(&self) -> Result<Option<Box<T>>> {
        let mut state = ptr::null_mut();
        // SAFETY: a live handle; `state` is valid for writes.
        check(unsafe { TRITONBACKEND_ModelState(self.0.as_ptr(), &mut state) })?;
        // SAFETY: a live handle.
        check(unsafe { TRITONBACKEND_ModelSetState(self.0.as_ptr(), ptr::null_mut()) })?;
        // SAFETY: per the function's contract, `state` came from `Box::<T>::into_raw` and is taken once.
        Ok((!state.is_null()).then(|| unsafe { Box::from_raw(state.cast::<T>()) }))
    }
}

/// # Safety
/// `message` is a live message.
unsafe fn message_json(message: *mut Message) -> Result<Value> {
    let (mut base, mut byte_size) = (ptr::null(), 0);
    // SAFETY: `message` is live; the serialized bytes live as long as it.
    check(unsafe { TRITONSERVER_MessageSerializeToJson(message, &mut base, &mut byte_size) })?;
    let bytes = if byte_size == 0 {
        &[][..]
    } else {
        // SAFETY: Triton returned `byte_size` readable bytes at `base`.
        unsafe { slice::from_raw_parts(base.cast::<u8>(), byte_size) }
    };
    serde_json::from_slice(bytes).map_err(|error| Error::config(format!("Triton's model configuration: {error}")))
}

/// A model instance, between `TRITONBACKEND_ModelInstanceInitialize` and `TRITONBACKEND_ModelInstanceFinalize`.
pub struct InstanceHandle(NonNull<ModelInstance>);

impl InstanceHandle {
    /// # Safety
    /// `instance` is the instance handle Triton passed to the current backend call.
    pub unsafe fn new(instance: *mut ModelInstance) -> Result<InstanceHandle> {
        NonNull::new(instance)
            .map(InstanceHandle)
            .ok_or_else(|| Error::unsupported("Triton passed no model instance"))
    }

    pub fn name(&self) -> Result<String> {
        let mut name = ptr::null();
        // SAFETY: a live handle; Triton owns the returned string for the instance's lifetime.
        check(unsafe { TRITONBACKEND_ModelInstanceName(self.0.as_ptr(), &mut name) })?;
        // SAFETY: as above.
        Ok(unsafe { c_string(name) })
    }

    /// The `TRITONSERVER_InstanceGroupKind` Triton resolved for the instance.
    pub fn kind(&self) -> Result<u32> {
        let mut kind = 0;
        // SAFETY: a live handle; `kind` is valid for writes.
        check(unsafe { TRITONBACKEND_ModelInstanceKind(self.0.as_ptr(), &mut kind) })?;
        Ok(kind)
    }

    pub fn model(&self) -> Result<ModelHandle> {
        let mut model = ptr::null_mut();
        // SAFETY: a live handle; the model outlives its instances.
        check(unsafe { TRITONBACKEND_ModelInstanceModel(self.0.as_ptr(), &mut model) })?;
        // SAFETY: Triton's handle for the instance's model.
        unsafe { ModelHandle::new(model) }
    }

    /// Stores `state` with the instance, for `state` and `take_state` to read.
    pub fn set_state<T>(&self, state: Box<T>) -> Result<()> {
        let state = Box::into_raw(state);
        // SAFETY: a live handle; Triton stores the pointer.
        let set = check(unsafe { TRITONBACKEND_ModelInstanceSetState(self.0.as_ptr(), state.cast()) });
        if set.is_err() {
            // SAFETY: Triton did not keep the pointer, which came from `Box::into_raw` above.
            drop(unsafe { Box::from_raw(state) });
        }
        set
    }

    /// The state `set_state` stored.
    ///
    /// # Safety
    /// As `ModelHandle::state`, for the instance.
    pub unsafe fn state<'a, T>(&self) -> Result<&'a T> {
        let mut state = ptr::null_mut();
        // SAFETY: a live handle; `state` is valid for writes.
        check(unsafe { TRITONBACKEND_ModelInstanceState(self.0.as_ptr(), &mut state) })?;
        // SAFETY: per the function's contract, `state` is null or a live `Box<T>`.
        unsafe { state.cast::<T>().as_ref() }.ok_or_else(|| Error::unsupported("the model instance has no state"))
    }

    /// Takes back the state `set_state` stored, leaving none.
    ///
    /// # Safety
    /// As `ModelHandle::take_state`, for the instance.
    pub unsafe fn take_state<T>(&self) -> Result<Option<Box<T>>> {
        let mut state = ptr::null_mut();
        // SAFETY: a live handle; `state` is valid for writes.
        check(unsafe { TRITONBACKEND_ModelInstanceState(self.0.as_ptr(), &mut state) })?;
        // SAFETY: a live handle.
        check(unsafe { TRITONBACKEND_ModelInstanceSetState(self.0.as_ptr(), ptr::null_mut()) })?;
        // SAFETY: per the function's contract, `state` came from `Box::<T>::into_raw` and is taken once.
        Ok((!state.is_null()).then(|| unsafe { Box::from_raw(state.cast::<T>()) }))
    }

    /// Reports one request's statistics; called before the request is released.
    pub fn report(&self, request: &RequestHandle, success: bool, times: &Timestamps) -> Result<()> {
        // SAFETY: live instance and request handles.
        check(unsafe {
            TRITONBACKEND_ModelInstanceReportStatistics(
                self.0.as_ptr(),
                request.0.as_ptr(),
                success,
                times.exec_start,
                times.compute_start,
                times.compute_end,
                times.exec_end,
            )
        })
    }

    /// Reports the statistics of one execution of `batch_size` rows (or requests, without batching).
    pub fn report_batch(&self, batch_size: u64, times: &Timestamps) -> Result<()> {
        // SAFETY: a live instance handle.
        check(unsafe {
            TRITONBACKEND_ModelInstanceReportBatchStatistics(
                self.0.as_ptr(),
                batch_size,
                times.exec_start,
                times.compute_start,
                times.compute_end,
                times.exec_end,
            )
        })
    }
}

/// An execution's steady-clock timestamps, in nanoseconds.
#[derive(Debug, Clone, Copy, Default)]
pub struct Timestamps {
    pub exec_start: u64,
    pub compute_start: u64,
    pub compute_end: u64,
    pub exec_end: u64,
}

/// A request the backend owns, from `TRITONBACKEND_ModelInstanceExecute` until `release`.
pub struct RequestHandle(NonNull<Request>);

impl RequestHandle {
    /// # Safety
    /// `request` is a request Triton passed to the current `TRITONBACKEND_ModelInstanceExecute`, which the backend
    /// now owns, wrapped once.
    pub unsafe fn new(request: *mut Request) -> Option<RequestHandle> {
        NonNull::new(request).map(RequestHandle)
    }

    /// The input `name`, with its buffers in CPU memory (Triton copies GPU-resident data to the CPU when asked).
    pub fn input<'r>(&'r self, name: &'r CStr) -> Result<InputTensor<'r>> {
        let mut input = ptr::null_mut();
        // SAFETY: a live request; the input lives as long as the request.
        check(unsafe { TRITONBACKEND_RequestInput(self.0.as_ptr(), name.as_ptr(), &mut input) })?;
        let (mut input_name, mut datatype, mut shape, mut dims_count) = (ptr::null(), 0, ptr::null(), 0);
        let (mut byte_size, mut buffer_count) = (0, 0);
        // SAFETY: a live input; every out-pointer is valid for writes.
        check(unsafe {
            TRITONBACKEND_InputProperties(
                input,
                &mut input_name,
                &mut datatype,
                &mut shape,
                &mut dims_count,
                &mut byte_size,
                &mut buffer_count,
            )
        })?;
        let shape = if dims_count == 0 {
            &[][..]
        } else {
            // SAFETY: Triton returned `dims_count` dims at `shape`, owned by the input.
            unsafe { slice::from_raw_parts(shape, dims_count as usize) }
        };
        let buffers = (0..buffer_count)
            // SAFETY: a live input of this request, `index` below its buffer count; the buffers live as long as
            // the request, which `'r` borrows.
            .map(|index| unsafe { input_buffer(input, index) })
            .collect::<Result<Vec<_>>>()?;
        Ok(InputTensor {
            name: name.to_str().unwrap_or("input"),
            data_type: data_type(datatype),
            shape,
            buffers,
        })
    }

    /// Sends the response holding `scores` as the output `name`. When the output cannot be made, the response
    /// carries that error instead, and it is returned.
    pub fn send_scores(&self, name: &CStr, data_type: DataType, shape: &[i64], scores: &[f64]) -> Result<()> {
        let mut response = ptr::null_mut();
        // SAFETY: a live request; the response is ours until sent.
        check(unsafe { TRITONBACKEND_ResponseNew(&mut response, self.0.as_ptr()) })?;
        // SAFETY: the response just made, sent once on either path.
        unsafe {
            match fill_output(response, name, data_type, shape, scores) {
                Ok(()) => send(response, None),
                Err(error) => send(response, Some(&error)).and(Err(error)),
            }
        }
    }

    /// Sends a response carrying `error`.
    pub fn send_error(&self, error: &Error) -> Result<()> {
        let mut response = ptr::null_mut();
        // SAFETY: a live request; the response is ours until sent.
        check(unsafe { TRITONBACKEND_ResponseNew(&mut response, self.0.as_ptr()) })?;
        // SAFETY: the response just made, sent once.
        unsafe { send(response, Some(error)) }
    }

    /// Hands the request back to Triton.
    pub fn release(self) -> Result<()> {
        // SAFETY: a live request the backend owns, released once (`self` is consumed).
        check(unsafe { TRITONBACKEND_RequestRelease(self.0.as_ptr(), REQUEST_RELEASE_ALL) })
    }
}

/// # Safety
/// `input` is a live input, `index` is below its buffer count, and `'r` ends before its request is released.
unsafe fn input_buffer<'r>(input: *mut Input, index: u32) -> Result<&'r [u8]> {
    let (mut buffer, mut byte_size, mut memory_type, mut memory_type_id) = (ptr::null(), 0, MEMORY_CPU, 0);
    // SAFETY: `input` is live and `index` below its buffer count; the memory type asks for CPU memory.
    check(unsafe {
        TRITONBACKEND_InputBuffer(
            input,
            index,
            &mut buffer,
            &mut byte_size,
            &mut memory_type,
            &mut memory_type_id,
        )
    })?;
    if memory_type == MEMORY_GPU {
        return Err(Error::unsupported("an input in GPU memory"));
    }
    if byte_size == 0 {
        return Ok(&[]);
    }
    let len = usize::try_from(byte_size).map_err(|_| Error::request("an input buffer larger than memory"))?;
    // SAFETY: Triton returned `len` readable bytes of CPU memory, owned by the input until the request is released;
    // `'r` is the borrow of the request handle.
    Ok(unsafe { slice::from_raw_parts(buffer.cast::<u8>(), len) })
}

/// # Safety
/// `response` is a live, unsent response.
unsafe fn fill_output(
    response: *mut Response,
    name: &CStr,
    data_type: DataType,
    shape: &[i64],
    scores: &[f64],
) -> Result<()> {
    let mut output = ptr::null_mut();
    let dims_count = u32::try_from(shape.len()).expect("one or two dims");
    // SAFETY: a live response; `shape` holds `dims_count` dims, which Triton copies.
    check(unsafe {
        TRITONBACKEND_ResponseOutput(
            response,
            &mut output,
            name.as_ptr(),
            type_code(data_type),
            shape.as_ptr(),
            dims_count,
        )
    })?;
    let byte_size = scores.len() * data_type.size();
    let (mut buffer, mut memory_type, mut memory_type_id) = (ptr::null_mut(), MEMORY_CPU, 0);
    // SAFETY: a live output; the memory type asks for CPU memory.
    check(unsafe {
        TRITONBACKEND_OutputBuffer(
            output,
            &mut buffer,
            byte_size as u64,
            &mut memory_type,
            &mut memory_type_id,
        )
    })?;
    if memory_type == MEMORY_GPU {
        return Err(Error::unsupported("an output buffer in GPU memory"));
    }
    if byte_size > 0 {
        // SAFETY: Triton returned `byte_size` writable bytes of CPU memory that nothing else accesses until the
        // response is sent.
        let out = unsafe { slice::from_raw_parts_mut(buffer.cast::<u8>(), byte_size) };
        batch::write_scores(scores, data_type, out);
    }
    Ok(())
}

/// Sends `response`, final, with `error` if any; Triton owns the response afterwards.
///
/// # Safety
/// `response` is a live, unsent response.
unsafe fn send(response: *mut Response, error: Option<&Error>) -> Result<()> {
    let error = error.map_or(ptr::null_mut(), new_error);
    // SAFETY: a live response, sent once; the error stays ours.
    let sent = check(unsafe { TRITONBACKEND_ResponseSend(response, RESPONSE_COMPLETE_FINAL, error) });
    if !error.is_null() {
        // SAFETY: the error made above, deleted once.
        unsafe { TRITONSERVER_ErrorDelete(error) };
    }
    sent
}
