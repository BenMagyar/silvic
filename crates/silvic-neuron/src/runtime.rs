use std::ffi::{c_char, c_int, c_void, CString};
use std::ptr::{self, NonNull};
use std::sync::{Arc, Mutex};

use bytemuck::Pod;
use libloading::Library;
use serde::Serialize;
use silvic_core::{Result, SilvicError};

use crate::compiler;

#[derive(Clone, Copy, Debug, Serialize)]
pub(crate) enum DType {
    #[serde(rename = "float32")]
    F32,
    #[serde(rename = "int32")]
    I32,
    #[serde(rename = "uint32")]
    U32,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct TensorSpec {
    pub name: String,
    pub dtype: DType,
    pub shape: Vec<usize>,
}

impl TensorSpec {
    pub fn new(name: impl Into<String>, dtype: DType, shape: impl Into<Vec<usize>>) -> Self {
        Self {
            name: name.into(),
            dtype,
            shape: shape.into(),
        }
    }

    pub(crate) fn bytes(&self) -> Result<usize> {
        let width = 4usize;
        self.shape.iter().try_fold(width, |bytes: usize, &dim| {
            if dim == 0 {
                return Err(SilvicError::config(
                    "Neuron compiled tensor dimensions must be positive",
                ));
            }
            bytes
                .checked_mul(dim)
                .filter(|&n| n <= isize::MAX as usize)
                .ok_or_else(|| SilvicError::config("Neuron tensor size overflows addressable storage"))
        })
    }
}

type Status = c_int;
type Init = unsafe extern "C" fn(c_int, *const c_char, *const c_char) -> Status;
type Count = unsafe extern "C" fn(*mut u32) -> Status;
type Instance = unsafe extern "C" fn(*mut c_void, usize) -> Status;
type Load = unsafe extern "C" fn(*const c_void, usize, i32, i32, *mut *mut c_void) -> Status;
type Unload = unsafe extern "C" fn(*mut c_void) -> Status;
type Allocate = unsafe extern "C" fn(c_int, c_int, usize, *const c_char, *mut *mut c_void) -> Status;
type Free = unsafe extern "C" fn(*mut *mut c_void);
type Write = unsafe extern "C" fn(*mut c_void, *const c_void, usize, usize) -> Status;
type Read = unsafe extern "C" fn(*const c_void, *mut c_void, usize, usize) -> Status;
type Memset = unsafe extern "C" fn(*mut c_void, u64, c_int, usize) -> Status;
type AllocateSet = unsafe extern "C" fn(*mut *mut c_void) -> Status;
type DestroySet = unsafe extern "C" fn(*mut *mut c_void);
type AddTensor = unsafe extern "C" fn(*mut c_void, *const c_char, *mut c_void) -> Status;
type Execute = unsafe extern "C" fn(*mut c_void, *const c_void, *mut c_void) -> Status;

struct Native {
    _library: Library,
    count: Count,
    instance: Instance,
    load: Load,
    unload: Unload,
    allocate: Allocate,
    free: Free,
    write: Write,
    read: Read,
    memset: Memset,
    allocate_set: AllocateSet,
    destroy_set: DestroySet,
    add_tensor: AddTensor,
    execute: Execute,
}

fn check(status: Status, operation: &str) -> Result<()> {
    if status == 0 {
        return Ok(());
    }
    let kind = match status {
        2 => "invalid argument or executable",
        4 => "resource allocation failed",
        5 => "timeout",
        6 => "hardware error",
        9 => "insufficient NeuronCores",
        10 => "unsupported NEFF version",
        1002 => "invalid execution input",
        1003 => "numerical error",
        1005 => "NeuronCore is busy",
        1006 => "device index out of bounds",
        _ => "runtime failure",
    };
    Err(SilvicError::device(format!(
        "Neuron {operation}: {kind} (NRT status {status})"
    )))
}

unsafe fn symbol<T: Copy>(library: &Library, name: &[u8]) -> Result<T> {
    // The signatures match the version-2 public NRT C headers; the library outlives every symbol.
    unsafe { library.get::<T>(name) }.map(|value| *value).map_err(|error| {
        SilvicError::device(format!(
            "Neuron runtime symbol {}: {error}",
            String::from_utf8_lossy(&name[..name.len() - 1])
        ))
    })
}

fn open_library() -> Result<Library> {
    let path = std::env::var_os("SILVIC_NEURON_LIBRARY").unwrap_or_else(|| "libnrt.so".into());
    // Loading the runtime does not initialize it or reserve NeuronCores.
    unsafe { Library::new(&path) }.map_err(|error| {
        SilvicError::device(format!(
            "cannot load AWS Neuron runtime {}: {error}",
            path.to_string_lossy()
        ))
    })
}

pub(crate) fn is_available() -> bool {
    if !cfg!(target_os = "linux") {
        return false;
    }
    let Ok(library) = open_library() else {
        return false;
    };
    let Ok(count) = (unsafe { symbol::<Count>(&library, b"nrt_get_visible_nc_count\0") }) else {
        return false;
    };
    let mut visible = 0;
    // This query is explicitly supported before nrt_init, so a Ray driver does not reserve its workers' cores.
    unsafe { count(&mut visible) == 0 && visible > 0 }
}

impl Native {
    fn open() -> Result<Self> {
        let library = open_library()?;
        unsafe {
            let init: Init = symbol(&library, b"nrt_init\0")?;
            let native = Self {
                count: symbol(&library, b"nrt_get_visible_nc_count\0")?,
                instance: symbol(&library, b"nrt_get_instance_info\0")?,
                load: symbol(&library, b"nrt_load\0")?,
                unload: symbol(&library, b"nrt_unload\0")?,
                allocate: symbol(&library, b"nrt_tensor_allocate\0")?,
                free: symbol(&library, b"nrt_tensor_free\0")?,
                write: symbol(&library, b"nrt_tensor_write\0")?,
                read: symbol(&library, b"nrt_tensor_read\0")?,
                memset: symbol(&library, b"nrt_tensor_memset\0")?,
                allocate_set: symbol(&library, b"nrt_allocate_tensor_set\0")?,
                destroy_set: symbol(&library, b"nrt_destroy_tensor_set\0")?,
                add_tensor: symbol(&library, b"nrt_add_tensor_to_tensor_set\0")?,
                execute: symbol(&library, b"nrt_execute\0")?,
                _library: library,
            };
            check(init(1, c"silvic".as_ptr(), c"0.1".as_ptr()), "initialization")?;
            Ok(native)
        }
    }
}

// NRT initialization and its library are process-scoped; closing one predictor must not close another framework's runtime.
static RUNTIME: Mutex<Option<Arc<Native>>> = Mutex::new(None);

fn native_runtime() -> Result<Arc<Native>> {
    if !cfg!(target_os = "linux") {
        return Err(SilvicError::unsupported("AWS Neuron requires Linux"));
    }
    let mut shared = RUNTIME
        .lock()
        .map_err(|_| SilvicError::device("Neuron runtime initialization lock is poisoned"))?;
    if let Some(runtime) = shared.as_ref() {
        return Ok(Arc::clone(runtime));
    }
    let runtime = Arc::new(Native::open()?);
    *shared = Some(Arc::clone(&runtime));
    Ok(runtime)
}

#[derive(Clone)]
pub(crate) struct Device {
    native: Arc<Native>,
    core: i32,
    target: &'static str,
    name: String,
}

impl Device {
    pub fn new(ordinal: Option<usize>) -> Result<Self> {
        let native = native_runtime()?;
        let mut count = 0;
        unsafe {
            check((native.count)(&mut count), "visible core query")?;
        }
        let ordinal = ordinal.unwrap_or(0);
        if ordinal >= count as usize {
            return Err(SilvicError::device(format!(
                "Neuron core {ordinal} requested, but {count} cores are visible"
            )));
        }
        let core = i32::try_from(ordinal).map_err(|_| SilvicError::config("Neuron core ordinal exceeds i32"))?;
        // The family and size are the stable prefix; newer NRT versions append architecture strings.
        let mut info = [0u32; 2];
        unsafe {
            check(
                (native.instance)(info.as_mut_ptr().cast(), std::mem::size_of_val(&info)),
                "instance query",
            )?;
        }
        let (target, chip) = match info[0] {
            2 | 3 => ("trn1", "Trainium"),
            4 | 7 => ("inf2", "Inferentia2"),
            5 | 6 | 8 | 9 | 10 | 11 => ("trn2", "Trainium2"),
            other => return Err(SilvicError::unsupported(format!("Neuron instance family {other}"))),
        };
        if target == "trn2" && std::env::var("NEURON_LOGICAL_NC_CONFIG").as_deref() != Ok("1") {
            return Err(SilvicError::config(
                "Trainium2 requires NEURON_LOGICAL_NC_CONFIG=1 before process startup",
            ));
        }
        Ok(Self {
            native,
            core,
            target,
            name: format!("AWS {chip} (neuron core {core})"),
        })
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn allocate(&self, bytes: usize) -> Result<Buffer> {
        let bytes = bytes.max(4);
        if bytes > isize::MAX as usize {
            return Err(SilvicError::config("Neuron buffer exceeds addressable storage"));
        }
        let mut raw = ptr::null_mut();
        unsafe {
            check(
                (self.native.allocate)(0, self.core, bytes, ptr::null(), &mut raw),
                "tensor allocation",
            )?;
        }
        let raw = NonNull::new(raw).ok_or_else(|| SilvicError::device("Neuron allocated a null tensor"))?;
        let buffer = Buffer {
            inner: Arc::new(Tensor {
                device: self.clone(),
                raw,
                bytes,
            }),
        };
        unsafe {
            check((self.native.memset)(raw.as_ptr(), 0, 0, bytes), "tensor initialization")?;
        }
        Ok(buffer)
    }

    pub fn upload<T: Pod>(&self, values: &[T]) -> Result<Buffer> {
        let mut buffer = self.allocate(std::mem::size_of_val(values))?;
        buffer.write(values)?;
        Ok(buffer)
    }

    pub fn compile(
        &self,
        source: &'static str,
        entry: &str,
        constants: &serde_json::Value,
        inputs: &[TensorSpec],
        outputs: &[TensorSpec],
    ) -> Result<Kernel> {
        let artifact = compiler::compile(source, entry, constants, inputs, outputs, self.target)?;
        let neff = std::fs::read(&artifact.neff).map_err(|error| {
            SilvicError::device(format!("read Neuron executable {}: {error}", artifact.neff.display()))
        })?;
        let mut raw = ptr::null_mut();
        unsafe {
            check(
                (self.native.load)(neff.as_ptr().cast(), neff.len(), self.core, 1, &mut raw),
                "model load",
            )?;
        }
        let raw = NonNull::new(raw).ok_or_else(|| SilvicError::device("Neuron loaded a null model"))?;
        let kernel = Kernel {
            device: self.clone(),
            raw,
            input_names: artifact.inputs,
            output_names: artifact.outputs,
            inputs: inputs.to_vec(),
            outputs: outputs.to_vec(),
            _neff: neff,
        };
        if kernel.input_names.len() != inputs.len() || kernel.output_names.len() != outputs.len() {
            return Err(SilvicError::device(
                "Neuron executable port counts differ from compiled tensor specifications",
            ));
        }
        Ok(kernel)
    }
}

struct Tensor {
    device: Device,
    raw: NonNull<c_void>,
    bytes: usize,
}

// NRT owns device memory; moving handles does not move storage. Calls retain the owning runtime.
unsafe impl Send for Tensor {}
unsafe impl Sync for Tensor {}

impl Drop for Tensor {
    fn drop(&mut self) {
        let mut raw = self.raw.as_ptr();
        unsafe {
            (self.device.native.free)(&mut raw);
        }
    }
}

#[derive(Clone)]
pub(crate) struct Buffer {
    inner: Arc<Tensor>,
}

impl Buffer {
    pub fn len_bytes(&self) -> usize {
        self.inner.bytes
    }

    fn range(&self, offset: usize, bytes: usize) -> Result<()> {
        if offset.checked_add(bytes).is_none_or(|end| end > self.inner.bytes) {
            return Err(SilvicError::config("Neuron tensor transfer exceeds buffer capacity"));
        }
        Ok(())
    }

    pub fn write<T: Pod>(&mut self, values: &[T]) -> Result<()> {
        self.write_at(0, values)
    }

    pub fn write_at<T: Pod>(&mut self, offset: usize, values: &[T]) -> Result<()> {
        let bytes = std::mem::size_of_val(values);
        self.range(offset, bytes)?;
        if bytes != 0 {
            unsafe {
                check(
                    (self.inner.device.native.write)(self.inner.raw.as_ptr(), values.as_ptr().cast(), offset, bytes),
                    "tensor write",
                )?;
            }
        }
        Ok(())
    }

    pub fn read<T: Pod>(&self, values: &mut [T]) -> Result<()> {
        self.read_at(0, values)
    }

    pub fn read_at<T: Pod>(&self, offset: usize, values: &mut [T]) -> Result<()> {
        let bytes = std::mem::size_of_val(values);
        self.range(offset, bytes)?;
        if bytes != 0 {
            unsafe {
                check(
                    (self.inner.device.native.read)(self.inner.raw.as_ptr(), values.as_mut_ptr().cast(), offset, bytes),
                    "tensor read",
                )?;
            }
        }
        Ok(())
    }
}

pub(crate) struct Kernel {
    device: Device,
    raw: NonNull<c_void>,
    input_names: Vec<String>,
    output_names: Vec<String>,
    inputs: Vec<TensorSpec>,
    outputs: Vec<TensorSpec>,
    _neff: Vec<u8>,
}

// A model stays alive for every bound execution and NRT provides thread-safe native handles.
unsafe impl Send for Kernel {}

impl Drop for Kernel {
    fn drop(&mut self) {
        unsafe {
            (self.device.native.unload)(self.raw.as_ptr());
        }
    }
}

impl Kernel {
    pub fn bind(self, inputs: &[Buffer], outputs: &[Buffer]) -> Result<Execution> {
        let input_set = TensorSet::new(&self.device, &self.input_names, &self.inputs, inputs)?;
        let output_set = TensorSet::new(&self.device, &self.output_names, &self.outputs, outputs)?;
        Ok(Execution {
            input_set,
            output_set,
            kernel: self,
        })
    }
}

struct TensorSet {
    native: Arc<Native>,
    raw: NonNull<c_void>,
    _buffers: Vec<Buffer>,
}

unsafe impl Send for TensorSet {}

impl TensorSet {
    fn new(device: &Device, names: &[String], specs: &[TensorSpec], buffers: &[Buffer]) -> Result<Self> {
        if names.len() != buffers.len() || specs.len() != buffers.len() {
            return Err(SilvicError::config("Neuron tensor binding count mismatch"));
        }
        for (spec, buffer) in specs.iter().zip(buffers) {
            if buffer.len_bytes() < spec.bytes()?
                || buffer.inner.device.core != device.core
                || !Arc::ptr_eq(&buffer.inner.device.native, &device.native)
            {
                return Err(SilvicError::config(format!(
                    "Neuron tensor {} has incompatible storage or core",
                    spec.name
                )));
            }
        }
        let mut raw = ptr::null_mut();
        unsafe {
            check((device.native.allocate_set)(&mut raw), "tensor-set allocation")?;
        }
        let raw = NonNull::new(raw).ok_or_else(|| SilvicError::device("Neuron allocated a null tensor set"))?;
        let set = Self {
            native: Arc::clone(&device.native),
            raw,
            _buffers: buffers.to_vec(),
        };
        for (name, buffer) in names.iter().zip(buffers) {
            let name = CString::new(name.as_str())
                .map_err(|_| SilvicError::device("Neuron executable tensor name contains NUL"))?;
            unsafe {
                check(
                    (device.native.add_tensor)(raw.as_ptr(), name.as_ptr(), buffer.inner.raw.as_ptr()),
                    "tensor binding",
                )?;
            }
        }
        Ok(set)
    }
}

impl Drop for TensorSet {
    fn drop(&mut self) {
        let mut raw = self.raw.as_ptr();
        unsafe {
            (self.native.destroy_set)(&mut raw);
        }
    }
}

pub(crate) struct Execution {
    input_set: TensorSet,
    output_set: TensorSet,
    kernel: Kernel,
}

impl Execution {
    pub fn execute(&mut self) -> Result<()> {
        unsafe {
            check(
                (self.kernel.device.native.execute)(
                    self.kernel.raw.as_ptr(),
                    self.input_set.raw.as_ptr(),
                    self.output_set.raw.as_ptr(),
                ),
                "execution",
            )
        }
    }
}
