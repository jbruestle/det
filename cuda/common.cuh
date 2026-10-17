// det: shared device primitives. These mirror detref/detmath.py EXACTLY (same op sequence).
// Rules: no FMA (we use __fmul_rn/__fadd_rn which are never contracted, and the Makefile passes -fmad=false),
// no libm transcendentals, IEEE div/sqrt via __fdiv_rn/__fsqrt_rn, RNE conversions.
#pragma once
#include <cuda_runtime.h>
#include <stdint.h>

#define DET_CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); abort(); } } while (0)

namespace det {

__device__ __forceinline__ float mul(float a, float b) { return __fmul_rn(a, b); }
__device__ __forceinline__ float add(float a, float b) { return __fadd_rn(a, b); }
__device__ __forceinline__ float sub(float a, float b) { return __fsub_rn(a, b); }
__device__ __forceinline__ float div(float a, float b) { return __fdiv_rn(a, b); }
__device__ __forceinline__ float sqrt_(float a) { return __fsqrt_rn(a); }
__device__ __forceinline__ float rint_(float a) { return __int2float_rn(__float2int_rn(a)); } // only for |a| < 2^31
__device__ __forceinline__ int   rint_i(float a) { return __float2int_rn(a); }
__device__ __forceinline__ float i64_to_f(long long v) { return __ll2float_rn(v); }

// ---- exp_f (Cephes), see detmath.exp_f
__device__ __forceinline__ float exp_f(float x) {
    x = fminf(x, 88.0f);
    const bool small = x < -87.0f;
    float n = floorf(add(mul(x, 1.44269504088896341f), 0.5f));
    float r = sub(x, mul(n, 0.693359375f));
    r = sub(r, mul(n, -2.12194440e-4f));
    float z = mul(r, r);
    float y = 1.9875691500E-4f;
    y = add(mul(y, r), 1.3981999507E-3f);
    y = add(mul(y, r), 8.3334519073E-3f);
    y = add(mul(y, r), 4.1665795894E-2f);
    y = add(mul(y, r), 1.6666665459E-1f);
    y = add(mul(y, r), 5.0000001201E-1f);
    y = add(mul(y, z), r);
    y = add(y, 1.0f);
    int ni = (int)n;
    float scale = __int_as_float((ni + 127) << 23);
    y = mul(y, scale);
    return small ? 0.0f : y;
}

// ---- log_f (Cephes), x > 0 normal
__device__ __forceinline__ float log_f(float x) {
    unsigned b = __float_as_uint(x);
    int e = (int)((b >> 23) & 0xFFu) - 126;
    float m = __uint_as_float((b & 0x007FFFFFu) | 0x3F000000u);
    const bool lt = m < 0.707106781186547524f;
    if (lt) { e -= 1; m = sub(add(m, m), 1.0f); } else { m = sub(m, 1.0f); }
    float z = mul(m, m);
    float y = 7.0376836292E-2f;
    y = add(mul(y, m), -1.1514610310E-1f);
    y = add(mul(y, m), 1.1676998740E-1f);
    y = add(mul(y, m), -1.2420140846E-1f);
    y = add(mul(y, m), 1.4249322787E-1f);
    y = add(mul(y, m), -1.6668057665E-1f);
    y = add(mul(y, m), 2.0000714765E-1f);
    y = add(mul(y, m), -2.4999993993E-1f);
    y = add(mul(y, m), 3.3333331174E-1f);
    y = mul(y, m);
    y = mul(y, z);
    float ef = (float)e;
    y = add(y, mul(-2.12194440e-4f, ef));
    y = add(y, mul(-0.5f, z));
    float r = add(m, y);
    r = add(r, mul(0.693359375f, ef));
    return r;
}

__device__ __forceinline__ float softplus_f(float x) {
    if (x > 20.0f) return x;
    if (x < -15.0f) return exp_f(x);
    return log_f(add(1.0f, exp_f(x)));
}

__device__ __forceinline__ float sigmoid_f(float x) {
    if (x < -80.0f) return exp_f(x);
    return div(1.0f, add(1.0f, exp_f(-x)));
}

__device__ __forceinline__ float silu_f(float x) { return mul(x, sigmoid_f(x)); }

// ---- treesum helpers.  The spec tree over a power-of-two array: s[i] = s[2i] + s[2i+1] per level.
// A thread holding an aligned contiguous chunk of 2^k values reduces it locally with tree_local<2^k>;
// then a warp butterfly with ASCENDING xor offsets continues the same tree across lanes; then across warps.
template <int N>
__device__ __forceinline__ float tree_local(const float* v) {   // v[N] in registers, N power of 2
    float t[N];
#pragma unroll
    for (int i = 0; i < N; i++) t[i] = v[i];
#pragma unroll
    for (int s = 1; s < N; s <<= 1) {
#pragma unroll
        for (int i = 0; i < N; i += 2 * s) t[i] = add(t[i], t[i + s]);
    }
    return t[0];
}

__device__ __forceinline__ float warp_tree(float v) {
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) v = add(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

// Block tree over NWARPS warp partials (NWARPS power of two <= 32). All threads get the result.
template <int NWARPS>
__device__ __forceinline__ float block_tree(float warp_val, float* smem /*[NWARPS]*/) {
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    __syncthreads();
    if (lane == 0) smem[w] = warp_val;
    __syncthreads();
    float t[NWARPS];
#pragma unroll
    for (int i = 0; i < NWARPS; i++) t[i] = smem[i];
#pragma unroll
    for (int s = 1; s < NWARPS; s <<= 1) {
#pragma unroll
        for (int i = 0; i < NWARPS; i += 2 * s) t[i] = add(t[i], t[i + s]);
    }
    return t[0];
}

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

template <int NWARPS>
__device__ __forceinline__ float block_max(float v, float* smem) {
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    v = warp_max(v);
    __syncthreads();
    if (lane == 0) smem[w] = v;
    __syncthreads();
    float r = smem[0];
#pragma unroll
    for (int i = 1; i < NWARPS; i++) r = fmaxf(r, smem[i]);
    return r;
}

// ---- activation quantization helpers (spec: s = amax/127, s=1 if amax==0; inv = 1/s; q = clamp(rint(x*inv)))
__device__ __forceinline__ float act_scale(float amax) { return amax == 0.0f ? 1.0f : div(amax, 127.0f); }
__device__ __forceinline__ int8_t act_q(float x, float inv) {
    int q = __float2int_rn(mul(x, inv));
    q = q > 127 ? 127 : (q < -127 ? -127 : q);
    return (int8_t)q;
}

} // namespace det
