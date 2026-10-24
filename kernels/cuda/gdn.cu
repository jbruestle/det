// Gated DeltaNet layer core (bit-exact w.r.t. the spec, crates/det/src/spec).
//
// Three stream-ordered kernels per chunk of tokens:
//   1. k_gdn_prep   : conv1d(4-tap, state = previous 3 raw inputs) + silu, l2norm of q/k heads (q *= QSCALE),
//                     beta = sigmoid(b), decay = exp(-(A * softplus(a + dt_bias)))        -> scratch (and gin)
//   2. k_gdn_recur  : the strictly sequential recurrence over the tokens.  One CTA owns JB columns (v dims) of
//                     one v-head; a column S[:, j] (128 k dims) is split over P lanes of the same warp (R = 128/P
//                     registers each).  The spec tree over i is tree_local<R> in registers followed by a warp
//                     butterfly with ascending xor offsets (== the spec tree).  k/q of the current token are
//                     broadcast through (double buffered) shared memory, one __syncthreads per token.
//   3. k_gdn_out    : on = rmsnorm(o, norm, eps) * silu(z)   (+ conv_state update, in the last chunk)
// The recurrence writes o into the scratch, or straight into gout when the caller exports it.
// All float math goes through det:: helpers (no FMA, no fast math, denormals preserved).
#include <stdio.h>
#include <stdlib.h>
#include "common.cuh"
#include "det_cuda.h"
using namespace det;

namespace {

constexpr int HD = 128;            // dk == dv == 128 (spec); the tree helpers are specialised for this
constexpr int CHUNK = 2048;        // tokens per internal chunk (bounds the scratch size)

__host__ __device__ inline size_t align256(size_t x) { return (x + 255) & ~(size_t)255; }

// QSCALE = fp32(1/sqrt(128)) computed in double on the host side of the compiler (0x3DB504F3)
constexpr float QSCALE = (float)(1.0 / 1.4142135623730951 / 8.0);   // 1/sqrt(128) = 1/(8*sqrt 2) -> 0x3DB504F3
static_assert(sizeof(float) == 4, "fp32");

// ------------------------------------------------------------------------------------------------ prep
// grid = (ceil(conv_dim/512), M_chunk), block = 128 (4 warps; one warp per 128 channels, 4 contiguous per lane)
__global__ void __launch_bounds__(128)
k_gdn_prep(const float* __restrict__ qkv, const float* __restrict__ conv_state, const float* __restrict__ conv_w,
           const float* __restrict__ a, const float* __restrict__ b, const float* __restrict__ A,
           const float* __restrict__ dt_bias, float eps, int t0, int conv_dim, int kd, int n_v, int rep,
           float* __restrict__ y, float* __restrict__ beta, float* __restrict__ decay, float* __restrict__ gin) {
    const int t = blockIdx.y;               // token within chunk
    const int tg = t0 + t;                  // token within the call (indexes qkv / a / b / gin)
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    const int base = (blockIdx.x * 4 + w) * HD;
    const int n_k = kd / HD, gw = 2 * HD + (HD + 2) * rep;
    if (base < conv_dim) {
        const int c0 = base + lane * 4;
        float4 x[4];
#pragma unroll
        for (int m = 0; m < 4; m++) {
            const int tt = tg - 3 + m;
            const float* src = (tt >= 0) ? (qkv + (size_t)tt * conv_dim) : (conv_state + (size_t)(tt + 3) * conv_dim);
            x[m] = *reinterpret_cast<const float4*>(src + c0);
        }
        float yv[4];
#pragma unroll
        for (int i = 0; i < 4; i++) {
            const float4 wc = *reinterpret_cast<const float4*>(conv_w + (size_t)(c0 + i) * 4);
            const float x0 = i == 0 ? x[0].x : i == 1 ? x[0].y : i == 2 ? x[0].z : x[0].w;
            const float x1 = i == 0 ? x[1].x : i == 1 ? x[1].y : i == 2 ? x[1].z : x[1].w;
            const float x2 = i == 0 ? x[2].x : i == 1 ? x[2].y : i == 2 ? x[2].z : x[2].w;
            const float x3 = i == 0 ? x[3].x : i == 1 ? x[3].y : i == 2 ? x[3].z : x[3].w;
            float v = add(mul(wc.x, x0), mul(wc.y, x1));
            v = add(v, mul(wc.z, x2));
            v = add(v, mul(wc.w, x3));
            yv[i] = silu_f(v);
        }
        if (base < 2 * kd) {                // q or k head: l2norm over the 128 channels of this warp
            float sq[4];
#pragma unroll
            for (int i = 0; i < 4; i++) sq[i] = mul(yv[i], yv[i]);
            const float ss = warp_tree(tree_local<4>(sq));
            const float inv = div(1.0f, sqrt_(add(ss, eps)));
#pragma unroll
            for (int i = 0; i < 4; i++) yv[i] = mul(yv[i], inv);
            if (base < kd) {
#pragma unroll
                for (int i = 0; i < 4; i++) yv[i] = mul(yv[i], QSCALE);
            }
        }
        *reinterpret_cast<float4*>(y + (size_t)t * conv_dim + c0) = make_float4(yv[0], yv[1], yv[2], yv[3]);
        if (gin) {
            int kh, off;
            if (base < kd) { kh = base / HD; off = c0 - kh * HD; }
            else if (base < 2 * kd) { kh = (base - kd) / HD; off = HD + (c0 - kd - kh * HD); }
            else { const int h = (base - 2 * kd) / HD; kh = h / rep; off = 2 * HD + (h - kh * rep) * HD + (c0 - 2 * kd - h * HD); }
            float* g = gin + ((size_t)tg * n_k + kh) * gw + off;
#pragma unroll
            for (int i = 0; i < 4; i++) g[i] = yv[i];
        }
    }
    if (blockIdx.x == 0 && (int)threadIdx.x < n_v) {
        const int h = threadIdx.x;
        const float bt = sigmoid_f(b[(size_t)tg * n_v + h]);
        const float sp = softplus_f(add(a[(size_t)tg * n_v + h], dt_bias[h]));
        const float dc = exp_f(-(mul(A[h], sp)));
        beta[(size_t)t * n_v + h] = bt;
        decay[(size_t)t * n_v + h] = dc;
        if (gin) {
            const int kh = h / rep, r = h - kh * rep;
            float* g = gin + ((size_t)tg * n_k + kh) * gw + 2 * HD + rep * HD;
            g[r] = bt;
            g[rep + r] = dc;
        }
    }
}

// ------------------------------------------------------------------------------------------------ recurrence
// JB columns (v dims) of one v-head per CTA, P lanes per column (R = 128/P k dims each), NT = JB*P threads.
// Lane mapping inside a warp: p = lane / (32/P) (which i-slice), column = warp*(32/P) + lane % (32/P).
template <int JB, int P>
__global__ void __launch_bounds__(JB * P)
k_gdn_recur(const float* __restrict__ y, const float* __restrict__ beta, const float* __restrict__ decay,
            float* __restrict__ S, int M, int conv_dim, int kd, int n_v, int rep, float* __restrict__ o) {
    constexpr int NT = JB * P;
    constexpr int R = HD / P;               // k dims per thread
    constexpr int LPP = 32 / P;             // lanes (columns) per part within a warp
    constexpr int KS = R + 4;               // padded smem stride per part (float4 aligned, bank-conflict free)
    constexpr int NLOAD = (2 * HD + NT - 1) / NT;   // k+q floats loaded per thread per token
    static_assert(NT % 32 == 0 && P <= 32 && (P & (P - 1)) == 0 && (JB & (JB - 1)) == 0, "config");
    static_assert(HD % JB == 0, "config");

    __shared__ __align__(16) float skq[2][2][P * KS];   // [buffer][k|q][part*KS + r]

    const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
    const int p = lane / LPP;
    const int j = w * LPP + (lane % LPP);                  // column within the CTA's JB columns
    const int h = blockIdx.x / (HD / JB);
    const int jg = (blockIdx.x % (HD / JB)) * JB + j;      // column within the head
    const int kh = h / rep;

    float* Sp = S + ((size_t)h * HD + (size_t)p * R) * HD + jg;    // S[h][p*R + r][jg] = Sp[r*HD]
    float s[R];
#pragma unroll
    for (int r = 0; r < R; r++) s[r] = Sp[(size_t)r * HD];

    const float* yq = y + (size_t)kh * HD;                 // + t*conv_dim
    const float* yk = y + kd + (size_t)kh * HD;
    const float* yv = y + 2 * kd + (size_t)h * HD + jg;
    const float* bh = beta + h;
    const float* dh = decay + h;

    // token 0 broadcast data
    for (int i = tid; i < 2 * HD; i += NT) {
        const int ii = i & (HD - 1);
        const float val = (i < HD) ? yk[ii] : yq[ii];
        skq[0][i >= HD][(ii / R) * KS + (ii % R)] = val;
    }
    float vcur = yv[0], bcur = bh[0], dcur = dh[0];
    __syncthreads();

    for (int t = 0; t < M; t++) {
        // prefetch token t+1 (latency hidden behind this token's compute)
        float nload[NLOAD];
        float vn = 0.0f, bn = 0.0f, dn = 0.0f;
        if (t + 1 < M) {
            const size_t off = (size_t)(t + 1) * conv_dim;
#pragma unroll
            for (int l = 0; l < NLOAD; l++) {
                const int i = tid + l * NT;
                const int ii = i & (HD - 1);
                nload[l] = (i < HD) ? yk[off + ii] : (i < 2 * HD ? yq[off + ii] : 0.0f);
            }
            vn = yv[off]; bn = bh[(size_t)(t + 1) * n_v]; dn = dh[(size_t)(t + 1) * n_v];
        }
        const float* kb = &skq[t & 1][0][p * KS];
        const float* qb = &skq[t & 1][1][p * KS];

        // S = S * decay
#pragma unroll
        for (int r = 0; r < R; r++) s[r] = mul(s[r], dcur);
        // kv_mem = treesum_i(S[i] * k[i])
        float prod[R];
#pragma unroll
        for (int r = 0; r < R; r += 4) {
            const float4 kk = *reinterpret_cast<const float4*>(kb + r);
            prod[r] = mul(s[r], kk.x); prod[r + 1] = mul(s[r + 1], kk.y);
            prod[r + 2] = mul(s[r + 2], kk.z); prod[r + 3] = mul(s[r + 3], kk.w);
        }
        float kv = tree_local<R>(prod);
#pragma unroll
        for (int off = LPP; off < 32; off <<= 1) kv = add(kv, __shfl_xor_sync(0xffffffffu, kv, off));
        const float delta = mul(sub(vcur, kv), bcur);
        // S = S + k[i] * delta ;  o = treesum_i(S[i] * q[i])
#pragma unroll
        for (int r = 0; r < R; r += 4) {
            const float4 kk = *reinterpret_cast<const float4*>(kb + r);
            const float4 qq = *reinterpret_cast<const float4*>(qb + r);
            s[r] = add(s[r], mul(kk.x, delta));         prod[r] = mul(s[r], qq.x);
            s[r + 1] = add(s[r + 1], mul(kk.y, delta)); prod[r + 1] = mul(s[r + 1], qq.y);
            s[r + 2] = add(s[r + 2], mul(kk.z, delta)); prod[r + 2] = mul(s[r + 2], qq.z);
            s[r + 3] = add(s[r + 3], mul(kk.w, delta)); prod[r + 3] = mul(s[r + 3], qq.w);
        }
        float ov = tree_local<R>(prod);
#pragma unroll
        for (int off = LPP; off < 32; off <<= 1) ov = add(ov, __shfl_xor_sync(0xffffffffu, ov, off));
        if (p == 0) o[((size_t)t * n_v + h) * HD + jg] = ov;

        if (t + 1 < M) {
#pragma unroll
            for (int l = 0; l < NLOAD; l++) {
                const int i = tid + l * NT;
                if (i < 2 * HD) {
                    const int ii = i & (HD - 1);
                    skq[(t + 1) & 1][i >= HD][(ii / R) * KS + (ii % R)] = nload[l];
                }
            }
            vcur = vn; bcur = bn; dcur = dn;
        }
        __syncthreads();
    }
#pragma unroll
    for (int r = 0; r < R; r++) Sp[(size_t)r * HD] = s[r];
}

// ------------------------------------------------------------------------------------------------ output
// blocks [0, nblk_out): 4 rows (of 128) per block, one warp per row: on = rmsnorm(o,norm,eps) * silu(z)
// blocks [nblk_out, ...): conv_state <- last 3 raw inputs (oldest first); each thread owns one channel
__global__ void __launch_bounds__(128)
k_gdn_out(const float* __restrict__ o, const float* __restrict__ z, const float* __restrict__ norm, float eps,
          int nrows, float* __restrict__ on, int nblk_out,
          const float* __restrict__ qkv, int M, int conv_dim, float* __restrict__ conv_state) {
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    if ((int)blockIdx.x < nblk_out) {
        const int row = blockIdx.x * 4 + w;          // row = t*n_v + h within chunk
        if (row >= nrows) return;
        const size_t ro = (size_t)row * HD + lane * 4;   // o / z / on are all [rows][128] (z, on pre-offset by chunk)
        float x[4], sq[4];
#pragma unroll
        for (int i = 0; i < 4; i++) { x[i] = o[ro + i]; sq[i] = mul(x[i], x[i]); }
        const float ss = warp_tree(tree_local<4>(sq));
        const float mean = div(ss, (float)HD);
        const float inv = div(1.0f, sqrt_(add(mean, eps)));
        const float* zr = z + ro;
        float* onr = on + ro;
#pragma unroll
        for (int i = 0; i < 4; i++) {
            const float yv = mul(mul(x[i], inv), norm[lane * 4 + i]);
            onr[i] = mul(yv, silu_f(zr[i]));
        }
    } else {
        const int c = (blockIdx.x - nblk_out) * 128 + threadIdx.x;
        if (c >= conv_dim) return;
        float nv[3];
#pragma unroll
        for (int r = 0; r < 3; r++) {
            const int u = M - 3 + r;
            nv[r] = (u >= 0) ? qkv[(size_t)u * conv_dim + c] : conv_state[(size_t)(u + 3) * conv_dim + c];
        }
#pragma unroll
        for (int r = 0; r < 3; r++) conv_state[(size_t)r * conv_dim + c] = nv[r];
    }
}

// ------------------------------------------------------------------------------------------------ host
// Recurrence configuration: 32 columns x 4 lanes (128 threads, R=32 registers of S per thread).  Measured on the
// RTX 5090 (170 SMs): 27B (192 CTAs) 0.46 us/token, 4B (128 CTAs) 0.32 us/token at M=1024; best of the tested
// configs (<64,4>, <128,2>, <32,8>, <64,2>, <16,8>, <128,1>) for both shapes.  Two CTAs sharing an SM cost the same
// per-token time as one CTA with twice the columns, so oversubscribing the SMs slightly is harmless.
constexpr int RECUR_JB = 32, RECUR_P = 4;
static void launch_recur(const float* y, const float* beta, const float* decay, float* S, int M, int conv_dim, int kd,
                         int n_v, int rep, float* o, cudaStream_t st) {
    k_gdn_recur<RECUR_JB, RECUR_P><<<n_v * (HD / RECUR_JB), RECUR_JB * RECUR_P, 0, st>>>(y, beta, decay, S, M, conv_dim, kd, n_v, rep, o);
}

} // namespace

extern "C" size_t det_gdn_scratch_bytes(int M, int n_k, int n_v, int dk, int dv) {
    const int conv_dim = 2 * n_k * dk + n_v * dv;
    const size_t m = (size_t)(M < CHUNK ? M : CHUNK);
    return align256(m * conv_dim * 4) + 2 * align256(m * n_v * 4) + align256(m * (size_t)n_v * dv * 4) + 256;
}

extern "C" void det_gdn_ex(const det_gdn_layer* L, det_gdn_state* ST, const float* qkv, const float* z, const float* a,
                           const float* b, float eps, int M, int n_k, int n_v, int dk, int dv, float* on, void* scratch,
                           float* gin, float* gout, cudaStream_t st) {
    if (dk != HD || dv != HD || n_v % n_k != 0) { fprintf(stderr, "det_gdn: unsupported shape dk=%d dv=%d n_k=%d n_v=%d\n", dk, dv, n_k, n_v); abort(); }
    if (n_v > 128) { fprintf(stderr, "det_gdn: n_v=%d > 128\n", n_v); abort(); }
    if (M <= 0) return;
    const int conv_dim = 2 * n_k * dk + n_v * dv;
    const int kd = n_k * dk;
    const int rep = n_v / n_k;
    const size_t m = (size_t)(M < CHUNK ? M : CHUNK);
    char* sp = (char*)align256((size_t)scratch);
    float* y = (float*)sp;            sp += align256(m * conv_dim * 4);
    float* beta = (float*)sp;         sp += align256(m * n_v * 4);
    float* decay = (float*)sp;        sp += align256(m * n_v * 4);
    float* o_scratch = (float*)sp;

    for (int t0 = 0; t0 < M; t0 += CHUNK) {
        const int mc = (M - t0 < CHUNK) ? (M - t0) : CHUNK;
        const bool last = (t0 + mc == M);
        float* o = gout ? gout + (size_t)t0 * n_v * dv : o_scratch;
        dim3 gp((conv_dim + 511) / 512, mc);
        k_gdn_prep<<<gp, 128, 0, st>>>(qkv, ST->conv_state, L->conv, a, b, L->A, L->dt_bias, eps, t0, conv_dim, kd, n_v, rep,
                                       y, beta, decay, gin);
        launch_recur(y, beta, decay, ST->S, mc, conv_dim, kd, n_v, rep, o, st);
        const int nrows = mc * n_v;
        const int nblk_out = (nrows + 3) / 4;
        const int nblk_conv = last ? (conv_dim + 127) / 128 : 0;
        k_gdn_out<<<nblk_out + nblk_conv, 128, 0, st>>>(o, z + (size_t)t0 * n_v * dv, L->norm, eps, nrows,
                                                         on + (size_t)t0 * n_v * dv, nblk_out, qkv, M, conv_dim,
                                                         ST->conv_state);
        DET_CHECK(cudaGetLastError());
    }
}

extern "C" void det_gdn(const det_gdn_layer* L, det_gdn_state* ST, const float* qkv, const float* z, const float* a,
                        const float* b, float eps, int M, int n_k, int n_v, int dk, int dv, float* on, void* scratch,
                        cudaStream_t st) {
    det_gdn_ex(L, ST, qkv, z, a, b, eps, M, n_k, n_v, dk, dv, on, scratch, nullptr, nullptr, st);
}
