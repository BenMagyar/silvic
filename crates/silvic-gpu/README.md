# silvic-gpu

The Metal and CUDA backends of silvic. The kernels are written once (`kernels/kernels.kern`, `kernels/common.h`)
against a small macro vocabulary that `kernels/prelude_metal.h` and `kernels/prelude_cuda.h` define for Metal Shading
Language and CUDA C++, and are compiled at run time: Metal `newLibraryWithSource` with fast math off, NVRTC with
`--fmad=false`. `GpuBackend` implements `silvic_core::Backend` once over the few `GpuDevice` primitives (allocate,
upload, download, compile, launch, sync), so the two backends differ only in their device layer.

| Feature | Device layer | Constructor |
| --- | --- | --- |
| `metal` (macOS) | `metal::MetalDevice`: objc2-metal, shared-storage buffers, one command buffer per sync | `metal::backend(index)` |
| `cuda` | `cuda::CudaDevice`: cudarc, the driver and NVRTC loaded at run time, one stream | `cuda::backend(ordinal)` |

`metal::is_available()` and `cuda::is_available()` probe without panicking, so a process on a machine without the
device (or with a mismatched CUDA driver) gets an error, not an abort.

What runs on the device: gradients (L2, binary, lambdarank with one threadgroup per query; queries over 1024 rows go
to the host's `objective::lambdarank_query`), quantization, bagging, stable partition fused with the child's
histogram (one sync per split), quantized histograms in packed 32-bit threadgroup atomics, float histograms,
fixed-point leaf sums and score updates. Split search stays on the host.

```rust
let mut backend = silvic_gpu::metal::backend(None)?;
let model = silvic_core::train(&params, &train_set, &[], &mut backend, &mut LocalCollective, &mut |_| Ok(()))?;
```

The CUDA backend compiles and its kernel source passes clang's CUDA front end, but it has not run on an NVIDIA GPU.

## Tests

- `tests/metal_backend.rs` checks every `Backend` operation of the Metal backend against `CpuBackend` on random
  data: quantization, quantized histograms, partitions, leaf sums, score updates and L2 gradients bit for bit, binary
  and lambdarank gradients within 1e-5, and quantized L2 training to a byte-identical model. It skips, printing
  why, without a Metal device.
- `tests/cuda_syntax.rs` compiles the CUDA kernel source with clang's CUDA front end (`SILVIC_CUDA_CLANG`, then
  Homebrew's LLVM, then `clang++`) against `kernels/cuda_syntax/shim.h`, and checks that both preludes define the
  same macros. It skips, printing why, without a clang that accepts `-x cuda`.
