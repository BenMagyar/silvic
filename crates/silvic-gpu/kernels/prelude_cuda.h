// CUDA C++ (NVRTC) definitions of the silvic kernel vocabulary. `prelude_metal.h` defines the same macros for Metal
// Shading Language; `common.h` and `kernels.kern` use nothing else, so one kernel source serves both devices.
//
// NVRTC compiles this with `--fmad=false` and without fast math, so `expf`/`logf`/`sqrtf` and friends are the
// IEEE-accurate library versions and no multiply-add contracts implicitly: the quantization formulas then round the
// same way as `quantize.rs` and the Metal kernels.

typedef unsigned int uint;
typedef unsigned char uchar;
typedef unsigned short ushort;
typedef long long i64;
typedef unsigned long long u64;

// Kernels are `extern "C"` so the host loads them by their source name. Every argument is positional: buffer index
// `i` is the parameter position, which `cuLaunchKernel` fills in order. `PARAMS` structs are copied by value into
// the kernel's parameter space, the CUDA counterpart of Metal `setBytes`. Metal declares its thread-position
// builtins as trailing parameters through `KERNEL_ARGS`; CUDA has them as globals, so it expands to nothing.
#define KERNEL(name) extern "C" __global__ void name
#define BUF(T, n, i) T* n
#define CBUF(T, n, i) const T* n
#define ABUF_I32(n, i) int* n
#define ABUF_F32(n, i) float* n
#define PARAMS(T, n, i) const T n
#define KERNEL_ARGS

// Kernels are launched with one-dimensional blocks.
#define LID ((uint)threadIdx.x)
#define GSIZE ((uint)blockDim.x)
#define GID_X ((uint)blockIdx.x)
#define GID_Y ((uint)blockIdx.y)
#define NGROUPS_X ((uint)gridDim.x)
#define NGROUPS_Y ((uint)gridDim.y)

#define DEVICE_FN static __device__ __forceinline__
#define GPTR(T) T*
#define CGPTR(T) const T*
#define SPTR(T) T*

#define SHARED_ARRAY(T, n, N) __shared__ T n[N]
#define SHARED_ATOMIC_I32_ARRAY(n, N) __shared__ int n[N]
#define SHARED_ATOMIC_F32_ARRAY(n, N) __shared__ float n[N]

// Metal's relaxed atomic load/store on threadgroup memory; a volatile access is the CUDA equivalent (never cached
// in a register across other threads' atomics).
#define ATOMIC_ADD_SHARED_I32(a, i, v) atomicAdd(&(a)[i], (int)(v))
#define ATOMIC_STORE_SHARED_I32(a, i, v) (*(volatile int*)&(a)[i] = (int)(v))
#define ATOMIC_LOAD_SHARED_I32(a, i) (*(volatile int*)&(a)[i])
#define ATOMIC_ADD_SHARED_F32(a, i, v) atomicAdd(&(a)[i], (float)(v))
#define ATOMIC_STORE_SHARED_F32(a, i, v) (*(volatile float*)&(a)[i] = (float)(v))
#define ATOMIC_LOAD_SHARED_F32(a, i) (*(volatile float*)&(a)[i])
#define ATOMIC_ADD_GLOBAL_I32(p, i, v) atomicAdd(&(p)[i], (int)(v))
#define ATOMIC_ADD_GLOBAL_F32(p, i, v) atomicAdd(&(p)[i], (float)(v))

// `__syncthreads` is a block barrier and a fence on shared memory, like Metal's
// `threadgroup_barrier(mem_flags::mem_threadgroup)`.
#define BARRIER() __syncthreads()

#define F32_FMA fmaf
#define F32_EXP expf
#define F32_LOG logf
#define F32_LOG2 log2f
#define F32_RINT rintf
#define F32_TRUNC truncf
#define F32_ABS fabsf
#define F32_SQRT sqrtf
