//! The Metal device layer: shared-storage buffers (one allocation the CPU and GPU both address, so uploads and
//! downloads are copies with no staging), kernels compiled from source with fast math off, and launches encoded into
//! one lazily opened command buffer that `flush` commits and `sync`, `upload` and `download` commit and wait on.

use std::cell::RefCell;
use std::ffi::c_void;
use std::ptr::NonNull;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue, MTLCompileOptions,
    MTLComputeCommandEncoder, MTLComputePipelineState, MTLCopyAllDevices, MTLCreateSystemDefaultDevice, MTLDevice,
    MTLLibrary, MTLMathMode, MTLResourceOptions, MTLSize,
};
use silvic_core::{Result, SilvicError};

use crate::backend::GpuBackend;
use crate::device::{Arg, GpuDevice};

/// Whether this machine has a Metal device.
pub fn is_available() -> bool {
    MTLCreateSystemDefaultDevice().is_some()
}

/// A compiled [`GpuBackend`] on Metal device `index` (`MTLCopyAllDevices` order), or the system default device.
pub fn backend(index: Option<usize>) -> Result<GpuBackend<MetalDevice>> {
    let mut backend = GpuBackend::new(MetalDevice::new(index)?);
    backend.compile()?;
    Ok(backend)
}

/// A shared-storage `MTLBuffer`.
pub struct MetalBuffer {
    buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    len: usize,
}

// SAFETY: Metal resources may be used from any thread (Apple: "MTLBuffer ... objects are thread safe"); objc2-metal
// only leaves `MTLBuffer` without `Send` because its CPU-visible contents are raw memory, which this crate touches
// only through `&MetalDevice`, after the GPU work using it has completed.
unsafe impl Send for MetalBuffer {}

impl MetalBuffer {
    fn check_range(&self, offset: usize, len: usize) -> Result<()> {
        match offset.checked_add(len) {
            Some(end) if end <= self.len => Ok(()),
            _ => Err(SilvicError::device(format!(
                "buffer access {offset}..{} out of bounds of {} bytes",
                offset.saturating_add(len),
                self.len
            ))),
        }
    }
}

struct Pending {
    command_buffer: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    encoder: Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>,
}

pub struct MetalDevice {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    pipelines: Vec<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    pending: RefCell<Option<Pending>>,
    /// Committed command buffers not yet waited on.
    committed: RefCell<Vec<Retained<ProtocolObject<dyn MTLCommandBuffer>>>>,
}

// SAFETY: the device, queue and pipelines are `Send + Sync`. The open command buffer and encoder are not thread safe,
// but Metal allows using them from another thread as long as no two threads use them at once; `RefCell` keeps
// `MetalDevice` `!Sync`, so only the thread owning it encodes.
unsafe impl Send for MetalDevice {}

impl MetalDevice {
    /// Device `index` of `MTLCopyAllDevices`, or the system default device.
    pub fn new(index: Option<usize>) -> Result<MetalDevice> {
        let device = match index {
            None => MTLCreateSystemDefaultDevice().ok_or_else(|| SilvicError::device("no metal device"))?,
            Some(index) => {
                let devices = MTLCopyAllDevices();
                if index >= devices.count() {
                    return Err(SilvicError::device(format!(
                        "metal device {index} requested, {} present",
                        devices.count()
                    )));
                }
                devices.objectAtIndex(index)
            }
        };
        let queue = device
            .newCommandQueue()
            .ok_or_else(|| SilvicError::device("cannot create a metal command queue"))?;
        Ok(MetalDevice {
            device,
            queue,
            pipelines: Vec::new(),
            pending: RefCell::new(None),
            committed: RefCell::new(Vec::new()),
        })
    }

    fn encoder(&self) -> Result<std::cell::RefMut<'_, Pending>> {
        let mut pending = self.pending.borrow_mut();
        if pending.is_none() {
            let command_buffer = self
                .queue
                .commandBuffer()
                .ok_or_else(|| SilvicError::device("cannot create a metal command buffer"))?;
            let encoder = command_buffer
                .computeCommandEncoder()
                .ok_or_else(|| SilvicError::device("cannot create a metal compute encoder"))?;
            *pending = Some(Pending {
                command_buffer,
                encoder,
            });
        }
        Ok(std::cell::RefMut::map(pending, |pending| {
            pending.as_mut().expect("open command buffer")
        }))
    }

    fn pipeline(&self, kernel: usize) -> Result<&ProtocolObject<dyn MTLComputePipelineState>> {
        self.pipelines
            .get(kernel)
            .map(|pipeline| &**pipeline)
            .ok_or_else(|| SilvicError::device(format!("kernel {kernel} is not compiled")))
    }
}

impl GpuDevice for MetalDevice {
    type Buffer = MetalBuffer;

    const PRELUDE: &'static str = include_str!("../kernels/prelude_metal.h");

    fn name(&self) -> String {
        self.device.name().to_string()
    }

    fn max_threads_per_group(&self) -> u32 {
        let device_max = self.device.maxThreadsPerThreadgroup().width;
        let pipeline_max = self.pipelines.iter().map(|p| p.maxTotalThreadsPerThreadgroup()).min();
        pipeline_max.map_or(device_max, |max| max.min(device_max)) as u32
    }

    fn alloc(&self, bytes: usize) -> Result<MetalBuffer> {
        let len = bytes.max(4);
        if len > self.device.maxBufferLength() {
            return Err(SilvicError::device(format!(
                "metal buffer of {len} bytes exceeds the device limit of {} bytes",
                self.device.maxBufferLength()
            )));
        }
        // `newBufferWithLength:options:` allocates a zero-filled buffer.
        let buffer = self
            .device
            .newBufferWithLength_options(len, MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| SilvicError::device(format!("cannot allocate a metal buffer of {len} bytes")))?;
        Ok(MetalBuffer { buffer, len })
    }

    fn upload(&self, buffer: &MetalBuffer, offset: usize, data: &[u8]) -> Result<()> {
        buffer.check_range(offset, data.len())?;
        // Earlier launches may still read or write this memory.
        self.sync()?;
        // SAFETY: in bounds (checked above), and no GPU work is in flight.
        unsafe {
            let dst = buffer.buffer.contents().as_ptr().cast::<u8>().add(offset);
            std::ptr::copy_nonoverlapping(data.as_ptr(), dst, data.len());
        }
        Ok(())
    }

    fn download(&self, buffer: &MetalBuffer, offset: usize, out: &mut [u8]) -> Result<()> {
        buffer.check_range(offset, out.len())?;
        self.sync()?;
        // SAFETY: in bounds (checked above), and no GPU work is in flight.
        unsafe {
            let src = buffer.buffer.contents().as_ptr().cast::<u8>().add(offset);
            std::ptr::copy_nonoverlapping(src, out.as_mut_ptr(), out.len());
        }
        Ok(())
    }

    fn compile(&mut self, source: &str, kernels: &[&str]) -> Result<()> {
        let options = MTLCompileOptions::new();
        // Fast math would reassociate and contract float expressions, so results would no longer match the host's.
        options.setMathMode(MTLMathMode::Safe);
        let library = self
            .device
            .newLibraryWithSource_options_error(&NSString::from_str(source), Some(&options))
            .map_err(|err| {
                SilvicError::device(format!(
                    "metal kernel compilation failed: {}",
                    err.localizedDescription()
                ))
            })?;
        self.pipelines = kernels
            .iter()
            .map(|&name| compile_pipeline(&self.device, &library, name))
            .collect::<Result<_>>()?;
        Ok(())
    }

    fn launch(&self, kernel: usize, grid: [u32; 3], block: [u32; 3], args: &[Arg<'_, MetalBuffer>]) -> Result<()> {
        let pipeline = self.pipeline(kernel)?;
        let threads = block.iter().map(|&b| b as usize).product::<usize>();
        if threads == 0 || threads > pipeline.maxTotalThreadsPerThreadgroup() {
            return Err(SilvicError::device(format!(
                "kernel {kernel}: {threads} threads per group, at most {} allowed",
                pipeline.maxTotalThreadsPerThreadgroup()
            )));
        }
        if grid.contains(&0) {
            return Ok(());
        }
        let pending = self.encoder()?;
        let encoder = &pending.encoder;
        encoder.setComputePipelineState(pipeline);
        for (index, arg) in args.iter().enumerate() {
            match arg {
                // SAFETY: `commandBuffer` (not `commandBufferWithUnretainedReferences`) retains every bound buffer
                // until the command buffer completes.
                Arg::Buffer(buffer) => unsafe { encoder.setBuffer_offset_atIndex(Some(&buffer.buffer), 0, index) },
                Arg::Bytes(bytes) => {
                    let ptr = NonNull::new(bytes.as_ptr().cast_mut().cast::<c_void>())
                        .ok_or_else(|| SilvicError::device("null parameter bytes"))?;
                    // SAFETY: Metal copies the bytes when encoding.
                    unsafe { encoder.setBytes_length_atIndex(ptr, bytes.len(), index) }
                }
            }
        }
        encoder.dispatchThreadgroups_threadsPerThreadgroup(mtl_size(grid), mtl_size(block));
        Ok(())
    }

    fn sync(&self) -> Result<()> {
        self.flush()?;
        let committed = std::mem::take(&mut *self.committed.borrow_mut());
        // A queue runs its command buffers in commit order; every one is checked for errors.
        for command_buffer in &committed {
            command_buffer.waitUntilCompleted();
        }
        for command_buffer in &committed {
            if command_buffer.status() == MTLCommandBufferStatus::Error {
                let reason = command_buffer.error().map_or_else(
                    || "unknown error".to_string(),
                    |err| err.localizedDescription().to_string(),
                );
                return Err(SilvicError::device(format!("metal command buffer failed: {reason}")));
            }
        }
        Ok(())
    }

    fn flush(&self) -> Result<()> {
        if let Some(pending) = self.pending.borrow_mut().take() {
            pending.encoder.endEncoding();
            pending.command_buffer.commit();
            self.committed.borrow_mut().push(pending.command_buffer);
        }
        Ok(())
    }
}

impl Drop for MetalDevice {
    fn drop(&mut self) {
        // Encoded work references buffers the backend is about to release.
        let _ = self.sync();
    }
}

fn compile_pipeline(
    device: &ProtocolObject<dyn MTLDevice>,
    library: &ProtocolObject<dyn MTLLibrary>,
    name: &str,
) -> Result<Retained<ProtocolObject<dyn MTLComputePipelineState>>> {
    let function = library
        .newFunctionWithName(&NSString::from_str(name))
        .ok_or_else(|| SilvicError::device(format!("metal kernel {name} not found in the library")))?;
    device
        .newComputePipelineStateWithFunction_error(&function)
        .map_err(|err| SilvicError::device(format!("metal pipeline {name}: {}", err.localizedDescription())))
}

fn mtl_size(size: [u32; 3]) -> MTLSize {
    MTLSize {
        width: size[0] as usize,
        height: size[1] as usize,
        depth: size[2] as usize,
    }
}
