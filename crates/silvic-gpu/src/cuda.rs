//! The CUDA device layer: [`GpuDevice`] over the CUDA driver API and NVRTC, both loaded at run time by cudarc, so
//! the crate builds and [`is_available`] answers on machines without the CUDA toolkit or an NVIDIA driver.
//!
//! All work goes to one stream, so launches, uploads and downloads run in call order without events. Kernels are
//! compiled by NVRTC with `--fmad=false` (no implicit multiply-add contraction, matching Metal with fast math off)
//! for the virtual architecture of the device, and the driver JIT-compiles the PTX when it loads the module.

use std::ffi::c_void;
use std::sync::Arc;

use cudarc::driver::sys::{CUdevice_attribute, CUdeviceptr, CUfunction_attribute};
use cudarc::driver::{result, CudaContext, CudaFunction, CudaSlice, CudaStream, DevicePtr, DriverError};
use cudarc::nvrtc::{self, CompileError, CompileOptions};
use silvic_core::{Result, SilvicError};

use crate::backend::GpuBackend;
use crate::device::{Arg, GpuDevice};

/// Whether the CUDA driver and NVRTC libraries load and the driver reports at least one device. Never panics.
pub fn is_available() -> bool {
    libraries_present()
        && std::panic::catch_unwind(CudaContext::device_count).is_ok_and(|count| count.is_ok_and(|count| count > 0))
}

/// The CUDA backend on device `ordinal` (default 0).
pub fn backend(ordinal: Option<usize>) -> Result<GpuBackend<CudaDevice>> {
    Ok(GpuBackend::new(CudaDevice::new(ordinal)?))
}

/// cudarc panics on its first call into a library it cannot load; probing with a plain `dlopen` first turns a
/// missing library into an error.
fn libraries_present() -> bool {
    // SAFETY: both probes only try to open the shared library and drop it again.
    unsafe { cudarc::driver::sys::is_culib_present() && cudarc::nvrtc::sys::is_culib_present() }
}

fn driver_error(context: &'static str) -> impl Fn(DriverError) -> SilvicError {
    move |err| SilvicError::device(format!("cuda {context}: {err}"))
}

/// Runs `body`, turning a panic into an error. cudarc resolves driver and NVRTC symbols on first use and panics
/// when one is missing (a driver or NVRTC older than the bindings), which must surface as an error, not an abort.
fn guarded<T>(what: &str, body: impl FnOnce() -> Result<T>) -> Result<T> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(body)).unwrap_or_else(|payload| {
        let message = payload
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| payload.downcast_ref::<&str>().copied())
            .unwrap_or("unknown panic");
        Err(SilvicError::device(format!(
            "CUDA driver or NVRTC failed in {what}: {message}"
        )))
    })
}

/// One CUDA device with a dedicated stream and the kernels of the last [`GpuDevice::compile`].
pub struct CudaDevice {
    context: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    name: String,
    compute_capability: u32,
    device_max_threads: u32,
    kernel_max_threads: u32,
    functions: Vec<CudaFunction>,
}

impl CudaDevice {
    /// Opens device `ordinal` (default 0) through its primary context.
    pub fn new(ordinal: Option<usize>) -> Result<Self> {
        if !libraries_present() {
            return Err(SilvicError::device(
                "cuda: the CUDA driver (libcuda) or NVRTC (libnvrtc) library is not loadable",
            ));
        }
        guarded("open device", || Self::open(ordinal.unwrap_or(0)))
    }

    fn open(ordinal: usize) -> Result<Self> {
        let context = CudaContext::new(ordinal).map_err(driver_error("open device"))?;
        // SAFETY: every buffer of this device is used on `stream` only, so cudarc's cross-stream events are not
        // needed; skipping them saves two events per buffer.
        unsafe { context.disable_event_tracking() };
        let stream = context.new_stream().map_err(driver_error("create stream"))?;
        let name = context.name().map_err(driver_error("device name"))?;
        let (major, minor) = context
            .compute_capability()
            .map_err(driver_error("compute capability"))?;
        let device_max_threads = context
            .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)
            .map_err(driver_error("max threads per block"))?;
        Ok(CudaDevice {
            context,
            stream,
            name,
            compute_capability: (major * 10 + minor) as u32,
            device_max_threads: device_max_threads as u32,
            kernel_max_threads: device_max_threads as u32,
            functions: Vec::new(),
        })
    }

    /// The virtual architecture NVRTC compiles for: the device's compute capability, or the newest one this NVRTC
    /// knows below it (the driver JIT runs PTX for an older virtual architecture on newer devices).
    fn arch(&self) -> String {
        let supported = nvrtc_supported_archs();
        let arch = supported
            .into_iter()
            .filter(|&arch| arch <= self.compute_capability)
            .max()
            .unwrap_or(self.compute_capability);
        format!("compute_{arch}")
    }

    fn checked_range(buffer: &CudaSlice<u8>, offset: usize, len: usize, what: &str) -> Result<()> {
        match offset.checked_add(len) {
            Some(end) if end <= buffer.len() => Ok(()),
            _ => Err(SilvicError::device(format!(
                "cuda {what}: {len} bytes at offset {offset} exceed the {}-byte buffer",
                buffer.len()
            ))),
        }
    }

    fn device_ptr(&self, buffer: &CudaSlice<u8>, offset: usize) -> CUdeviceptr {
        // The guard records a read event, which is a no-op with event tracking disabled.
        let (ptr, _guard) = buffer.device_ptr(&self.stream);
        ptr + offset as CUdeviceptr
    }
}

/// The real architectures NVRTC can target (e.g. 52, 80, 90), or none if it cannot say.
fn nvrtc_supported_archs() -> Vec<u32> {
    use cudarc::nvrtc::sys::{nvrtcGetNumSupportedArchs, nvrtcGetSupportedArchs, nvrtcResult};

    let mut count = 0;
    // SAFETY: the count is written to a local, then exactly `count` ints into a buffer of that length.
    unsafe {
        if nvrtcGetNumSupportedArchs(&mut count) != nvrtcResult::NVRTC_SUCCESS || count <= 0 {
            return Vec::new();
        }
        let mut archs = vec![0i32; count as usize];
        if nvrtcGetSupportedArchs(archs.as_mut_ptr()) != nvrtcResult::NVRTC_SUCCESS {
            return Vec::new();
        }
        archs.into_iter().map(|arch| arch as u32).collect()
    }
}

fn compile_error(err: CompileError) -> SilvicError {
    match err {
        CompileError::CompileError { nvrtc, options, log } => SilvicError::device(format!(
            "nvrtc compile failed ({nvrtc:?}, options {options:?}):\n{}",
            log.to_string_lossy()
        )),
        other => SilvicError::device(format!("nvrtc: {other:?}")),
    }
}

impl GpuDevice for CudaDevice {
    type Buffer = CudaSlice<u8>;

    const PRELUDE: &'static str = include_str!("../kernels/prelude_cuda.h");

    fn name(&self) -> String {
        self.name.clone()
    }

    /// The device's block size limit before [`GpuDevice::compile`]; after it, the smallest limit of the loaded
    /// kernels (register pressure can lower it).
    fn max_threads_per_group(&self) -> u32 {
        self.kernel_max_threads
    }

    fn alloc(&self, bytes: usize) -> Result<Self::Buffer> {
        guarded("alloc", || {
            self.stream
                .alloc_zeros::<u8>(bytes.max(4))
                .map_err(driver_error("alloc"))
        })
    }

    fn upload(&self, buffer: &Self::Buffer, offset: usize, data: &[u8]) -> Result<()> {
        Self::checked_range(buffer, offset, data.len(), "upload")?;
        if data.is_empty() {
            return Ok(());
        }
        guarded("upload", || {
            self.context.bind_to_thread().map_err(driver_error("bind context"))?;
            let dst = self.device_ptr(buffer, offset);
            // SAFETY: the range was checked above. From pageable memory the copy returns only once `data` has
            // been staged, so the caller may reuse it right away.
            unsafe { result::memcpy_htod_async(dst, data, self.stream.cu_stream()) }.map_err(driver_error("upload"))
        })
    }

    fn download(&self, buffer: &Self::Buffer, offset: usize, out: &mut [u8]) -> Result<()> {
        Self::checked_range(buffer, offset, out.len(), "download")?;
        if out.is_empty() {
            return self.sync();
        }
        guarded("download", || {
            self.context.bind_to_thread().map_err(driver_error("bind context"))?;
            let src = self.device_ptr(buffer, offset);
            // SAFETY: the range was checked above; `out` stays borrowed until the stream is synchronized below.
            unsafe { result::memcpy_dtoh_async(out, src, self.stream.cu_stream()) }
                .map_err(driver_error("download"))?;
            self.stream.synchronize().map_err(driver_error("synchronize"))
        })
    }

    /// NVRTC symbols resolve lazily here, so this is where an old NVRTC shows up.
    fn compile(&mut self, source: &str, kernels: &[&str]) -> Result<()> {
        guarded("compile", || self.compile_kernels(source, kernels))
    }

    fn launch(&self, kernel: usize, grid: [u32; 3], block: [u32; 3], args: &[Arg<'_, Self::Buffer>]) -> Result<()> {
        guarded("launch", || self.launch_kernel(kernel, grid, block, args))
    }

    fn sync(&self) -> Result<()> {
        guarded("synchronize", || {
            self.stream.synchronize().map_err(driver_error("synchronize"))
        })
    }
}

impl CudaDevice {
    fn compile_kernels(&mut self, source: &str, kernels: &[&str]) -> Result<()> {
        let options = CompileOptions {
            fmad: Some(false),
            options: vec!["--std=c++17".to_string(), format!("--gpu-architecture={}", self.arch())],
            ..Default::default()
        };
        let ptx = nvrtc::compile_ptx_with_opts(source, options).map_err(compile_error)?;
        let module = self.context.load_module(ptx).map_err(driver_error("load module"))?;
        let functions = kernels
            .iter()
            .map(|&kernel| {
                module
                    .load_function(kernel)
                    .map_err(|err| SilvicError::device(format!("cuda load kernel {kernel}: {err}")))
            })
            .collect::<Result<Vec<_>>>()?;
        let mut kernel_max_threads = self.device_max_threads;
        for function in &functions {
            let limit = function
                .get_attribute(CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK)
                .map_err(driver_error("kernel max threads"))?;
            kernel_max_threads = kernel_max_threads.min(limit as u32);
        }
        self.functions = functions;
        self.kernel_max_threads = kernel_max_threads;
        Ok(())
    }

    fn launch_kernel(
        &self,
        kernel: usize,
        grid: [u32; 3],
        block: [u32; 3],
        args: &[Arg<'_, CudaSlice<u8>>],
    ) -> Result<()> {
        let function = self.functions.get(kernel).ok_or_else(|| {
            SilvicError::device(format!(
                "cuda launch: kernel {kernel} of {} not compiled",
                self.functions.len()
            ))
        })?;
        self.context.bind_to_thread().map_err(driver_error("bind context"))?;
        // `cuLaunchKernel` takes one pointer per parameter to the parameter's value: the device address for a
        // buffer, the struct bytes for `PARAMS`. The addresses live in `pointers` until the launch has copied them.
        let pointers: Vec<CUdeviceptr> = args
            .iter()
            .map(|arg| match arg {
                Arg::Buffer(buffer) => self.device_ptr(buffer, 0),
                Arg::Bytes(_) => 0,
            })
            .collect();
        let mut params: Vec<*mut c_void> = args
            .iter()
            .zip(&pointers)
            .map(|(arg, pointer)| match arg {
                Arg::Buffer(_) => pointer as *const CUdeviceptr as *mut c_void,
                Arg::Bytes(bytes) => bytes.as_ptr() as *mut c_void,
            })
            .collect();
        // SAFETY: `params` holds one pointer per kernel parameter, each valid for the duration of the call, which
        // copies the values. Dropping a buffer frees it in stream order (or after a stream sync), so a buffer is
        // never freed under a kernel still using it.
        unsafe {
            result::launch_kernel(
                function.cu_function(),
                (grid[0], grid[1], grid[2]),
                (block[0], block[1], block[2]),
                0,
                self.stream.cu_stream(),
                &mut params,
            )
        }
        .map_err(driver_error("launch"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_available_does_not_panic() {
        let available = is_available();
        if cfg!(target_os = "macos") {
            assert!(!available, "macOS has no CUDA driver");
            assert!(CudaDevice::new(None).is_err(), "opening a device without CUDA fails");
        }
    }

    #[test]
    fn guarded_turns_a_panic_into_a_device_error() {
        let missing_symbol = guarded::<()>("open device", || panic!("Missing symbol cuCtxCreate_v4"));
        let err = missing_symbol.expect_err("panic becomes an error");
        assert_eq!(err.kind(), "device", "error kind");
        assert!(
            err.to_string().contains("Missing symbol cuCtxCreate_v4"),
            "panic message kept: {err}"
        );
        assert_eq!(guarded("sync", || Ok(7)), Ok(7), "no panic passes the result through");
    }
}
