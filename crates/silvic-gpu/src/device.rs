//! The few device primitives the GPU backend needs, implemented once per API (Metal, CUDA). Everything above them,
//! kernels included, is shared.
//!
//! Work is ordered: launches, uploads and downloads take effect in call order. `download` waits for every earlier
//! launch; `launch` may return before the kernel runs.

use silvic_core::Result;

/// A kernel argument. Buffer arguments bind at their position (Metal `[[buffer(i)]]`, CUDA positional parameter);
/// `Bytes` is a `#[repr(C)]` parameter struct passed by value (Metal `setBytes`, at most 4 KiB).
pub enum Arg<'a, B> {
    Buffer(&'a B),
    Bytes(&'a [u8]),
}

pub trait GpuDevice {
    type Buffer;

    /// The macro vocabulary (`kernels/prelude_*.h`) that compiles `kernels/common.h` and `kernels/kernels.kern`
    /// for this API.
    const PRELUDE: &'static str;

    /// The device's name, as the driver reports it.
    fn name(&self) -> String;

    /// The largest threadgroup (block) the compiled kernels may be launched with.
    fn max_threads_per_group(&self) -> u32;

    /// A zero-filled buffer of `bytes` bytes (at least 4).
    fn alloc(&self, bytes: usize) -> Result<Self::Buffer>;

    /// Copies `data` into `buffer` at byte `offset`.
    fn upload(&self, buffer: &Self::Buffer, offset: usize, data: &[u8]) -> Result<()>;

    /// Copies `out.len()` bytes from `buffer` at byte `offset`, after every earlier launch completed.
    fn download(&self, buffer: &Self::Buffer, offset: usize, out: &mut [u8]) -> Result<()>;

    /// Compiles `source` and loads `kernels`; later launches name a kernel by its index in `kernels`.
    fn compile(&mut self, source: &str, kernels: &[&str]) -> Result<()>;

    /// Launches kernel `kernel` over `grid` threadgroups of `block` threads.
    fn launch(&self, kernel: usize, grid: [u32; 3], block: [u32; 3], args: &[Arg<'_, Self::Buffer>]) -> Result<()>;

    /// Waits for every earlier launch.
    fn sync(&self) -> Result<()>;

    /// Starts every earlier launch without waiting for it, so the device works while the host does. A device whose
    /// launches start on their own (CUDA streams) needs nothing here.
    fn flush(&self) -> Result<()> {
        Ok(())
    }
}
