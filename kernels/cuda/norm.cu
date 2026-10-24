// glue kernels: rmsnorm(+quant), quant, embed, add, silu*mul(+quant), argmax
#include <stdio.h>
#include <stdlib.h>
#include "common.cuh"
#include "det_cuda.h"
using namespace det;

// One row per block of NT threads; each thread owns CHUNK contiguous elements (n_pad = NT*CHUNK, zero padded).
template <int NT, int CHUNK, bool NORM>
__global__ void k_rmsnorm_quant(const float* __restrict__ x, const float* __restrict__ w, float eps, int n,
                                float* __restrict__ y, int8_t* __restrict__ xq, float* __restrict__ sx, uint16_t* __restrict__ mx) {
    __shared__ float smem[32];
    const int row = blockIdx.x;
    const int t = threadIdx.x;
    const float* xr = x + (size_t)row * n;
    float v[CHUNK];
#pragma unroll
    for (int i = 0; i < CHUNK; i++) { int idx = t * CHUNK + i; v[i] = idx < n ? xr[idx] : 0.0f; }
    if (NORM) {
        float sq[CHUNK];
#pragma unroll
        for (int i = 0; i < CHUNK; i++) sq[i] = mul(v[i], v[i]);
        float ss = warp_tree(tree_local<CHUNK>(sq));
        if (NT > 32) ss = block_tree<NT / 32>(ss, smem);
        float mean = div(ss, (float)n);
        float inv = div(1.0f, sqrt_(add(mean, eps)));
#pragma unroll
        for (int i = 0; i < CHUNK; i++) { int idx = t * CHUNK + i; v[i] = idx < n ? mul(mul(v[i], inv), w[idx]) : 0.0f; }
        if (y) {
#pragma unroll
            for (int i = 0; i < CHUNK; i++) { int idx = t * CHUNK + i; if (idx < n) y[(size_t)row * n + idx] = v[i]; }
        }
    }
    if (xq) {
        float am = 0.0f;
#pragma unroll
        for (int i = 0; i < CHUNK; i++) am = fmaxf(am, fabsf(v[i]));
        if (mx) {
            // per-(row, 128-group) multipliers; requires n % 128 == 0 so CHUNK >= 4 (host checks)
            if constexpr (CHUNK >= 4) {
                constexpr int LPG = 128 / CHUNK;              // lanes per group
                float gm = am;
#pragma unroll
                for (int o = 1; o < LPG; o <<= 1) gm = fmaxf(gm, __shfl_xor_sync(0xffffffffu, gm, o));
                float sg = div(gm, 127.0f);
                float smax = (NT > 32) ? block_max<NT / 32>(sg, smem) : warp_max(sg);
                float s = smax == 0.0f ? 1.0f : div(smax, 65535.0f);
                int mxi = __float2int_rn(div(sg, s));
                mxi = mxi < 1 ? 1 : (mxi > 65535 ? 65535 : mxi);
                float inv = div(1.0f, mul(s, (float)mxi));
#pragma unroll
                for (int i = 0; i < CHUNK; i++) { int idx = t * CHUNK + i; if (idx < n) xq[(size_t)row * n + idx] = act_q(v[i], inv); }
                if ((t % LPG) == 0 && (t / LPG) < n / 128) mx[(size_t)row * (n / 128) + t / LPG] = (uint16_t)mxi;
                if (t == 0) sx[row] = s;
            }
        } else {
            am = (NT > 32) ? block_max<NT / 32>(am, smem) : warp_max(am);
            float s = act_scale(am);
            float inv = div(1.0f, s);
#pragma unroll
            for (int i = 0; i < CHUNK; i++) { int idx = t * CHUNK + i; if (idx < n) xq[(size_t)row * n + idx] = act_q(v[i], inv); }
            if (t == 0) sx[row] = s;
        }
    }
}

// Generic per-(row, 128-group) quantization, one warp per group (4 contiguous elements per lane).
// MODE 0: v = a;  MODE 1: v = silu(a) * b.  Optionally stores v to h.
// Each warp owns groups g = w, w+NW, ... (at most MAXG); all loads are issued up front and values stay in registers.
template <int MODE, int NT, int MAXG>
__global__ void __launch_bounds__(NT) k_group_quant(const float* __restrict__ a, const float* __restrict__ b, int n, float* __restrict__ h,
                              int8_t* __restrict__ q, float* __restrict__ sx, uint16_t* __restrict__ mx) {
    __shared__ float sg_s[1024];
    constexpr int NW = NT / 32;
    const int row = blockIdx.x, G = n / 128, w = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const size_t off = (size_t)row * n;
    float v[MAXG][4];
#pragma unroll
    for (int j = 0; j < MAXG; j++) {
        const int g = w + j * NW;
        if (g < G) {
            const int idx = g * 128 + lane * 4;
            float4 av = *reinterpret_cast<const float4*>(a + off + idx);
            if (MODE == 1) {
                float4 bv = *reinterpret_cast<const float4*>(b + off + idx);
                v[j][0] = mul(silu_f(av.x), bv.x); v[j][1] = mul(silu_f(av.y), bv.y); v[j][2] = mul(silu_f(av.z), bv.z); v[j][3] = mul(silu_f(av.w), bv.w);
            } else { v[j][0] = av.x; v[j][1] = av.y; v[j][2] = av.z; v[j][3] = av.w; }
            if (h) *reinterpret_cast<float4*>(h + off + idx) = make_float4(v[j][0], v[j][1], v[j][2], v[j][3]);
            float am = fmaxf(fmaxf(fabsf(v[j][0]), fabsf(v[j][1])), fmaxf(fabsf(v[j][2]), fabsf(v[j][3])));
            am = warp_max(am);
            if (lane == 0) sg_s[g] = div(am, 127.0f);
        }
    }
    __syncthreads();
    float smax = 0.0f;
    for (int g = 0; g < G; g++) smax = fmaxf(smax, sg_s[g]);
    const float s = smax == 0.0f ? 1.0f : div(smax, 65535.0f);
#pragma unroll
    for (int j = 0; j < MAXG; j++) {
        const int g = w + j * NW;
        if (g < G) {
            const int idx = g * 128 + lane * 4;
            int mxi = __float2int_rn(div(sg_s[g], s));
            mxi = mxi < 1 ? 1 : (mxi > 65535 ? 65535 : mxi);
            float inv = div(1.0f, mul(s, (float)mxi));
            char4 c4 = make_char4(act_q(v[j][0], inv), act_q(v[j][1], inv), act_q(v[j][2], inv), act_q(v[j][3], inv));
            *reinterpret_cast<char4*>(q + off + idx) = c4;
            if (lane == 0) mx[(size_t)row * G + g] = (uint16_t)mxi;
        }
    }
    if (threadIdx.x == 0) sx[row] = s;
}

template <int MODE>
static void launch_group_quant(const float* a, const float* b, int M, int n, float* h, int8_t* q, float* sx, uint16_t* mx, cudaStream_t st) {
    const int G = n / 128;
    if (n % 128 || G > 1024) { fprintf(stderr, "det group quant: bad n=%d\n", n); abort(); }
    if (G <= 32)       k_group_quant<MODE, 1024, 1><<<M, 1024, 0, st>>>(a, b, n, h, q, sx, mx);
    else if (G <= 64)  k_group_quant<MODE, 1024, 2><<<M, 1024, 0, st>>>(a, b, n, h, q, sx, mx);
    else if (G <= 128) k_group_quant<MODE, 1024, 4><<<M, 1024, 0, st>>>(a, b, n, h, q, sx, mx);
    else if (G <= 256) k_group_quant<MODE, 1024, 8><<<M, 1024, 0, st>>>(a, b, n, h, q, sx, mx);
    else               k_group_quant<MODE, 1024, 32><<<M, 1024, 0, st>>>(a, b, n, h, q, sx, mx);
}

template <bool NORM>
static void launch_norm(const float* x, const float* w, float eps, int M, int n, float* y, int8_t* xq, float* sx, uint16_t* mx, cudaStream_t st) {
    int np = 32; while (np < n) np <<= 1;
    if (mx && (n % 128)) { fprintf(stderr, "det_rmsnorm/quant: group mode needs n %% 128 == 0 (n=%d)\n", n); abort(); }
    switch (np) {
        case 32:   k_rmsnorm_quant<32, 1, NORM><<<M, 32, 0, st>>>(x, w, eps, n, y, xq, sx, mx); break;
        case 64:   k_rmsnorm_quant<32, 2, NORM><<<M, 32, 0, st>>>(x, w, eps, n, y, xq, sx, mx); break;
        case 128:  k_rmsnorm_quant<32, 4, NORM><<<M, 32, 0, st>>>(x, w, eps, n, y, xq, sx, mx); break;
        case 256:  k_rmsnorm_quant<32, 8, NORM><<<M, 32, 0, st>>>(x, w, eps, n, y, xq, sx, mx); break;
        case 512:  k_rmsnorm_quant<32, 16, NORM><<<M, 32, 0, st>>>(x, w, eps, n, y, xq, sx, mx); break;
        case 1024: k_rmsnorm_quant<256, 4, NORM><<<M, 256, 0, st>>>(x, w, eps, n, y, xq, sx, mx); break;
        case 2048: k_rmsnorm_quant<512, 4, NORM><<<M, 512, 0, st>>>(x, w, eps, n, y, xq, sx, mx); break;
        case 4096: k_rmsnorm_quant<1024, 4, NORM><<<M, 1024, 0, st>>>(x, w, eps, n, y, xq, sx, mx); break;
        case 8192: k_rmsnorm_quant<1024, 8, NORM><<<M, 1024, 0, st>>>(x, w, eps, n, y, xq, sx, mx); break;
        default: fprintf(stderr, "det_rmsnorm: unsupported n=%d\n", n); abort();
    }
}

void det_rmsnorm_quant(const float* x, const float* w, float eps, int M, int n, float* y, int8_t* xq, float* sx, uint16_t* mx, cudaStream_t st) {
    launch_norm<true>(x, w, eps, M, n, y, xq, sx, mx, st);
}

// generic per-row quant for any n (two passes, grid-stride within block)
__global__ void k_quant(const float* __restrict__ x, int n, int8_t* __restrict__ xq, float* __restrict__ sx) {
    __shared__ float smem[32];
    const int row = blockIdx.x;
    const float* xr = x + (size_t)row * n;
    float am = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) am = fmaxf(am, fabsf(xr[i]));
    am = block_max<8>(am, smem);
    float s = act_scale(am);
    float inv = div(1.0f, s);
    for (int i = threadIdx.x; i < n; i += blockDim.x) xq[(size_t)row * n + i] = act_q(xr[i], inv);
    if (threadIdx.x == 0) sx[row] = s;
}
void det_quant(const float* x, int M, int n, int8_t* xq, float* sx, uint16_t* mx, cudaStream_t st) {
    if (n <= 8192) { launch_norm<false>(x, nullptr, 0.0f, M, n, nullptr, xq, sx, mx, st); return; }
    if (mx) { launch_group_quant<0>(x, nullptr, M, n, nullptr, xq, sx, mx, st); return; }
    k_quant<<<M, 256, 0, st>>>(x, n, xq, sx);
}

__global__ void k_embed(const uint16_t* __restrict__ tab, const int* __restrict__ ids, int H, float* __restrict__ x) {
    const int row = blockIdx.x;
    const uint16_t* src = tab + (size_t)ids[row] * H;
    for (int i = threadIdx.x; i < H; i += blockDim.x) x[(size_t)row * H + i] = __uint_as_float(((unsigned)src[i]) << 16);
}
void det_embed(const uint16_t* table_bf16, const int* ids, int M, int H, float* x, cudaStream_t st) {
    k_embed<<<M, 256, 0, st>>>(table_bf16, ids, H, x);
}

__global__ void k_add(float* __restrict__ x, const float* __restrict__ y, long long n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = add(x[i], y[i]);
}
void det_add(float* x, const float* y, long long n, cudaStream_t st) {
    k_add<<<(unsigned)((n + 255) / 256), 256, 0, st>>>(x, y, n);
}

__global__ void k_silu_mul_quant(const float* __restrict__ g, const float* __restrict__ u, int n, float* __restrict__ h,
                                 int8_t* __restrict__ hq, float* __restrict__ sh) {
    __shared__ float smem[32];
    const int row = blockIdx.x;
    const size_t off = (size_t)row * n;
    float am = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float v = mul(silu_f(g[off + i]), u[off + i]);
        if (h) h[off + i] = v;
        am = fmaxf(am, fabsf(v));
    }
    am = block_max<8>(am, smem);
    float s = act_scale(am);
    float inv = div(1.0f, s);
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float v = mul(silu_f(g[off + i]), u[off + i]);
        hq[off + i] = act_q(v, inv);
    }
    if (threadIdx.x == 0) sh[row] = s;
}
void det_silu_mul_quant(const float* g, const float* u, int M, int n, float* h, int8_t* hq, float* sh, uint16_t* mh, cudaStream_t st) {
    if (mh) { launch_group_quant<1>(g, u, M, n, h, hq, sh, mh, st); return; }
    k_silu_mul_quant<<<M, 256, 0, st>>>(g, u, n, h, hq, sh);
}

__global__ void k_argmax(const float* __restrict__ lg, int V, int* __restrict__ out) {
    __shared__ float sv[256]; __shared__ int si[256];
    const int row = blockIdx.x;
    const float* r = lg + (size_t)row * V;
    float best = -INFINITY; int bi = 0x7fffffff;
    for (int i = threadIdx.x; i < V; i += blockDim.x) { float v = r[i]; if (v > best || (v == best && i < bi)) { best = v; bi = i; } }
    sv[threadIdx.x] = best; si[threadIdx.x] = bi;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            float ov = sv[threadIdx.x + s]; int oi = si[threadIdx.x + s];
            if (ov > sv[threadIdx.x] || (ov == sv[threadIdx.x] && oi < si[threadIdx.x])) { sv[threadIdx.x] = ov; si[threadIdx.x] = oi; }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) out[row] = si[0];
}
void det_argmax(const float* logits, int M, int V, int* out, cudaStream_t st) {
    k_argmax<<<M, 256, 0, st>>>(logits, V, out);
}
