//! Metal and CUDA backends for silvic.
//!
//! The kernels are written once (`kernels/kernels.kern`) against a small macro vocabulary that
//! `kernels/prelude_metal.h` and `kernels/prelude_cuda.h` define for Metal Shading Language and CUDA C++, and are
//! compiled at run time (Metal `newLibraryWithSource`, NVRTC). [`GpuBackend`] implements `silvic_core::Backend` once
//! over the [`GpuDevice`] primitives, so the Metal and CUDA backends differ only in their device layer.

mod backend;
#[cfg(feature = "cuda")]
pub mod cuda;
mod device;
#[cfg(all(feature = "metal", target_os = "macos"))]
pub mod metal;
mod params;

pub use backend::GpuBackend;
pub use device::{Arg, GpuDevice};
pub use params::kernel_source;
