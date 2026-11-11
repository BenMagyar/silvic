// The kernel macro vocabulary for Metal Shading Language. `common.h` and `kernels.kern` use only these names, so the
// same source compiles as CUDA C++ under `prelude_cuda.h`. Compiled with fast math off (`MTLMathModeSafe`), so the
// F32_* functions are the precise variants and nothing contracts into an FMA unless written as F32_FMA.

#include <metal_stdlib>
#include <metal_atomic>
using namespace metal;

typedef long i64;
typedef ulong u64;

#define KERNEL(name) kernel void name
#define BUF(T, n, i) device T *n [[buffer(i)]]
#define CBUF(T, n, i) const device T *n [[buffer(i)]]
#define ABUF_I32(n, i) device atomic_int *n [[buffer(i)]]
#define ABUF_F32(n, i) device atomic_float *n [[buffer(i)]]
#define PARAMS(T, n, i) constant T &n [[buffer(i)]]
// Metal wants every position attribute of a kernel scalar or every one the same vector width, so all are uint3;
// launches use 1-D threadgroups.
#define KERNEL_ARGS                                                                                                \
    , uint3 silvic_lid [[thread_position_in_threadgroup]], uint3 silvic_gsize [[threads_per_threadgroup]],           \
        uint3 silvic_gid [[threadgroup_position_in_grid]], uint3 silvic_ngroups [[threadgroups_per_grid]]

#define LID silvic_lid.x
#define GSIZE silvic_gsize.x
#define GID_X silvic_gid.x
#define GID_Y silvic_gid.y
#define NGROUPS_X silvic_ngroups.x
#define NGROUPS_Y silvic_ngroups.y

#define DEVICE_FN static inline
#define GPTR(T) device T *
#define CGPTR(T) const device T *
#define SPTR(T) threadgroup T *

#define SHARED_ARRAY(T, n, N) threadgroup T n[N]
#define SHARED_ATOMIC_I32_ARRAY(n, N) threadgroup atomic_int n[N]
// Metal has float atomics in device memory only; threadgroup floats are bit patterns in atomic_uint, added by CAS.
#define SHARED_ATOMIC_F32_ARRAY(n, N) threadgroup atomic_uint n[N]

static inline float silvic_atomic_add_shared_f32(threadgroup atomic_uint *slot, float value) {
    uint expected = atomic_load_explicit(slot, memory_order_relaxed);
    while (!atomic_compare_exchange_weak_explicit(slot, &expected, as_type<uint>(as_type<float>(expected) + value),
                                                  memory_order_relaxed, memory_order_relaxed)) {
    }
    return as_type<float>(expected);
}

#define ATOMIC_ADD_SHARED_I32(a, i, v) atomic_fetch_add_explicit(&(a)[i], (v), memory_order_relaxed)
#define ATOMIC_STORE_SHARED_I32(a, i, v) atomic_store_explicit(&(a)[i], (v), memory_order_relaxed)
#define ATOMIC_LOAD_SHARED_I32(a, i) atomic_load_explicit(&(a)[i], memory_order_relaxed)
#define ATOMIC_ADD_SHARED_F32(a, i, v) silvic_atomic_add_shared_f32(&(a)[i], (v))
#define ATOMIC_STORE_SHARED_F32(a, i, v) atomic_store_explicit(&(a)[i], as_type<uint>((float)(v)), memory_order_relaxed)
#define ATOMIC_LOAD_SHARED_F32(a, i) as_type<float>(atomic_load_explicit(&(a)[i], memory_order_relaxed))
#define ATOMIC_ADD_GLOBAL_I32(p, i, v) atomic_fetch_add_explicit(&(p)[i], (v), memory_order_relaxed)
#define ATOMIC_ADD_GLOBAL_F32(p, i, v) atomic_fetch_add_explicit(&(p)[i], (v), memory_order_relaxed)

#define BARRIER() threadgroup_barrier(mem_flags::mem_threadgroup)

#define F32_FMA fma
#define F32_EXP precise::exp
#define F32_LOG precise::log
#define F32_LOG2 precise::log2
#define F32_RINT rint
#define F32_TRUNC trunc
#define F32_ABS fabs
#define F32_SQRT precise::sqrt
