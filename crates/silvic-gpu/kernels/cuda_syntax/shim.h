// What NVRTC declares implicitly for every program, for clang's CUDA frontend run with `-nocudainc -nocudalib`
// (the syntax check in `tests/cuda_syntax.rs`; there is no CUDA toolkit on the machines it runs on). Only the
// declarations NVRTC really provides belong here, so the check rejects anything NVRTC would reject.

#define __global__ __attribute__((global))
#define __device__ __attribute__((device))
#define __shared__ __attribute__((shared))
#define __forceinline__ __inline__ __attribute__((always_inline))
#define __restrict__ __restrict

#include <__clang_cuda_builtin_vars.h>

__device__ void __syncthreads();

__device__ int atomicAdd(int* address, int val);
__device__ unsigned int atomicAdd(unsigned int* address, unsigned int val);
__device__ unsigned long long atomicAdd(unsigned long long* address, unsigned long long val);
__device__ float atomicAdd(float* address, float val);

__device__ float fmaf(float x, float y, float z);
__device__ float expf(float x);
__device__ float logf(float x);
__device__ float log2f(float x);
__device__ float rintf(float x);
__device__ float truncf(float x);
__device__ float fabsf(float x);
__device__ float sqrtf(float x);
__device__ float floorf(float x);
__device__ float ceilf(float x);
__device__ float fminf(float x, float y);
__device__ float fmaxf(float x, float y);

__device__ int min(int a, int b);
__device__ unsigned int min(unsigned int a, unsigned int b);
__device__ float min(float a, float b);
__device__ int max(int a, int b);
__device__ unsigned int max(unsigned int a, unsigned int b);
__device__ float max(float a, float b);

__device__ int __float_as_int(float x);
__device__ unsigned int __float_as_uint(float x);
__device__ float __int_as_float(int x);
__device__ float __uint_as_float(unsigned int x);
