// Full attention core (docs/SPEC.md 5.1).  Bit-exact w.r.t. the spec.
//
// Pipeline for M new tokens at positions pos0..pos0+M-1:
//   1. k_attn_prep   : one warp per row.  q rows (M*n_h): rmsnorm(q_norm) -> rope -> *qscale -> int8 quant into scratch.
//                      k rows (M*n_kv): rmsnorm(k_norm) -> rope -> int8 quant into the KV cache.  v rows: int16 into the cache.
//   2. k_attn_max    : grid (nsplit, n_kv, M).  Block = (token t, kv head, key range); computes s_j for the rep q heads that
//                      share this kv head from int8 dot products (dp4a) and atomicMax-es the per-head max into scratch.
//   3. k_attn_acc    : same grid; recomputes s_j (deterministic), pq_j = rint(exp_f(s_j - mx) * 2^24), accumulates
//                      den = sum pq_j and num[d] = sum pq_j * vq_j[d] in int64 (order-free).  nsplit == 1: writes the
//                      gated output directly; else atomicAdd-s into int64 scratch and k_attn_fin produces the output.
// Keys are split across blocks only when M <= MSPLIT (then the scratch holds the int64 partials).
// det_attention_dev (graph-capturable decode): the same CUDA-core kernels, but pos0 is read from device memory
// (pos0_dev) and the grid is sized for max_pos; blocks whose key range lies beyond the real pos0 exit at once.
// For M >= tc_min_m() (default 16) the max/accumulate passes instead run on tensor cores (namespace tc below): same
// numerics, with the integer contractions done by int8 mma.sync (scores) and byte-sliced u8/s8 mma.sync (P.V).
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <type_traits>
#include "common.cuh"
#include "det_cuda.h"
using namespace det;

namespace {

#ifndef DET_ATTN_TK
#define DET_ATTN_TK 32
#endif
constexpr int TK = DET_ATTN_TK;            // keys per tile
constexpr int NT = 4 * TK;                 // threads per attention block (4 thread groups over heads)
constexpr int MSPLIT = 256;                // key splitting (int64 scratch) only for M <= MSPLIT
constexpr int SPLIT_TARGET_BLOCKS = 512;   // aim for this many blocks when splitting keys
constexpr float P_FIX = 16777216.0f;       // 2^24

// order-preserving float <-> uint map so that max can be done with atomicMax on unsigned (0 == "less than everything")
__device__ __forceinline__ unsigned f2ord(float f) { unsigned u = __float_as_uint(f); return (u & 0x80000000u) ? ~u : (u | 0x80000000u); }
__device__ __forceinline__ float ord2f(unsigned u) { return __uint_as_float((u & 0x80000000u) ? (u & 0x7fffffffu) : ~u); }

struct Scratch {
    int8_t* qq; float* sq; unsigned* mx; unsigned long long* den; unsigned long long* num;
    unsigned long long* tcnum; unsigned long long* tcden; int tc_slots;   // tensor-core path: tail-split partials
    size_t bytes;
};
constexpr int TC_RB = 48;                  // rows per tensor-core block (tc::RB)
constexpr int TC_SLOTS_MAX = 256;          // tail slots (>= the number of tail blocks, which is < the SM count)
constexpr int TC_SCRATCH_MIN_M = 16;       // tail slots are reserved in the scratch from this M on (= default tc_min_m)
static size_t align_up(size_t x) { return (x + 255) & ~(size_t)255; }
static Scratch layout(void* base, int M, int n_h, int hd) {
    char* b = (char*)base;
    size_t off = 0;
    Scratch s;
    s.qq = (int8_t*)(b + off);              off = align_up(off + (size_t)M * n_h * hd);
    s.sq = (float*)(b + off);               off = align_up(off + (size_t)M * n_h * 4);
    s.mx = (unsigned*)(b + off);            off = align_up(off + (size_t)M * n_h * 4);
    if (M <= MSPLIT) {
        s.den = (unsigned long long*)(b + off); off = align_up(off + (size_t)M * n_h * 8);
        s.num = (unsigned long long*)(b + off); off = align_up(off + (size_t)M * n_h * hd * 8);
    } else { s.den = nullptr; s.num = nullptr; }
    // tensor-core tail slots (M >= TC_SCRATCH_MIN_M only): one per tail block, each RB rows x hd int64 + RB int64
    // (the block count is <= M*n_h).  Without slots the tensor-core path runs unsplit (correct, less balanced).
    s.tc_slots = M < TC_SCRATCH_MIN_M ? 0 : ((long long)M * n_h < TC_SLOTS_MAX ? M * n_h : TC_SLOTS_MAX);
    s.tcnum = (unsigned long long*)(b + off);   off = align_up(off + (size_t)s.tc_slots * TC_RB * hd * 8);
    s.tcden = (unsigned long long*)(b + off);   off = align_up(off + (size_t)s.tc_slots * TC_RB * 8);
    s.bytes = off;
    return s;
}

// ---------------------------------------------------------------------------------------------------------------------
// prep: rows [0, M*n_h) = q, [M*n_h, M*n_h + M*n_kv) = k, then v.  One warp per row, CHUNK contiguous elements per lane
// (spec tree: lane-local tree of CHUNK then warp butterfly with ascending xor offsets, exactly like norm.cu).
template <int HD>
__global__ void __launch_bounds__(256) k_attn_prep(const float* __restrict__ qg, const float* __restrict__ k, const float* __restrict__ v,
                                                   const float* __restrict__ q_norm, const float* __restrict__ k_norm,
                                                   const float* __restrict__ v_inv_scale,
                                                   const float* __restrict__ rope_cos, const float* __restrict__ rope_sin, int rot,
                                                   float eps, float qscale, int M, int pos0, const int* __restrict__ pos0_dev,
                                                   int n_h, int n_kv, int8_t* __restrict__ qq, float* __restrict__ sq,
                                                   int8_t* __restrict__ kq, float* __restrict__ sk, int16_t* __restrict__ vq) {
    constexpr int CHUNK = HD / 32;
    __shared__ float sm[8][HD];
    if (pos0_dev) pos0 = *pos0_dev;
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    const int row = blockIdx.x * 8 + w;
    const int nq = M * n_h, nk = M * n_kv;
    if (row >= nq + 2 * nk) return;
    if (row < nq + nk) {
        int t, hh; const float* src; const float* nw;
        if (row < nq) { t = row / n_h; hh = row - t * n_h; src = qg + ((size_t)t * n_h + hh) * 2 * HD; nw = q_norm; }
        else { int r = row - nq; t = r / n_kv; hh = r - t * n_kv; src = k + ((size_t)t * n_kv + hh) * HD; nw = k_norm; }
        float x[CHUNK], sqv[CHUNK];
#pragma unroll
        for (int i = 0; i < CHUNK; i++) { x[i] = src[lane * CHUNK + i]; sqv[i] = mul(x[i], x[i]); }
        float ss = warp_tree(tree_local<CHUNK>(sqv));
        float mean = div(ss, (float)HD);
        float inv = div(1.0f, sqrt_(add(mean, eps)));
#pragma unroll
        for (int i = 0; i < CHUNK; i++) x[i] = mul(mul(x[i], inv), nw[lane * CHUNK + i]);
        // rope (neox half split on the first rot dims)
        const int pos = pos0 + t, half = rot >> 1;
#pragma unroll
        for (int i = 0; i < CHUNK; i++) { int idx = lane * CHUNK + i; if (idx < rot) sm[w][idx] = x[i]; }
        __syncwarp();
        const float* rc = rope_cos + (size_t)pos * half;
        const float* rs = rope_sin + (size_t)pos * half;
        float y[CHUNK];
#pragma unroll
        for (int i = 0; i < CHUNK; i++) {
            int idx = lane * CHUNK + i;
            if (idx < half)      { float a = x[i], b = sm[w][idx + half]; y[i] = sub(mul(a, rc[idx]), mul(b, rs[idx])); }
            else if (idx < rot)  { float a = sm[w][idx - half], b = x[i]; y[i] = add(mul(b, rc[idx - half]), mul(a, rs[idx - half])); }
            else                 y[i] = x[i];
        }
        if (row < nq) {
#pragma unroll
            for (int i = 0; i < CHUNK; i++) y[i] = mul(y[i], qscale);
        }
        float am = 0.0f;
#pragma unroll
        for (int i = 0; i < CHUNK; i++) am = fmaxf(am, fabsf(y[i]));
        am = warp_max(am);
        float s = act_scale(am);
        float invs = div(1.0f, s);
        int8_t* dst; float* sdst;
        if (row < nq) { dst = qq + (size_t)row * HD; sdst = sq + row; }
        else { dst = kq + ((size_t)pos * n_kv + hh) * HD; sdst = sk + (size_t)pos * n_kv + hh; }
#pragma unroll
        for (int i = 0; i < CHUNK; i++) dst[lane * CHUNK + i] = act_q(y[i], invs);
        if (lane == 0) *sdst = s;
    } else {
        const int r = row - nq - nk, t = r / n_kv, hh = r - t * n_kv, pos = pos0 + t;
        const float* src = v + ((size_t)t * n_kv + hh) * HD;
        const float vinv = v_inv_scale[hh];
        int16_t* dst = vq + ((size_t)pos * n_kv + hh) * HD;
#pragma unroll
        for (int i = 0; i < CHUNK; i++) {
            int q = __float2int_rn(mul(src[lane * CHUNK + i], vinv));
            q = q > 32767 ? 32767 : (q < -32767 ? -32767 : q);
            dst[lane * CHUNK + i] = (int16_t)q;
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Shared tile machinery.  K tile in smem with a padded row stride (HD/16+1 int4) -> conflict-free int4 reads when
// thread kk reads row kk.  Thread (kk = tid % TK, g = tid / TK) computes the dots of key kk for heads r with r % NG == g.
template <int HD>
struct Tile {
    static constexpr int KW = HD / 16;     // int4 words per key
    static constexpr int KS = KW + 1;      // padded stride (int4)
    static constexpr int NG = NT / TK;     // thread groups over heads
};

// Stage nvalid rows of WORDS int4 each (row r at src + r * src_stride_bytes) into smem (row stride DST_STRIDE int4).
// Full tiles issue all loads before any store so the global latency is paid once per tile, not once per word.
template <int WORDS, int DST_STRIDE>
__device__ __forceinline__ void stage_rows(int4* dst, const void* __restrict__ src, size_t src_stride_bytes, int nvalid) {
    constexpr int PER_T = WORDS * TK / NT;
    static_assert(WORDS * TK % NT == 0, "tile not divisible");
    const int tid = threadIdx.x;
    if (nvalid == TK) {
        int4 tmp[PER_T];
#pragma unroll
        for (int r = 0; r < PER_T; r++) {
            const int i = tid + r * NT, key = i / WORDS, w = i - key * WORDS;
            tmp[r] = reinterpret_cast<const int4*>((const char*)src + (size_t)key * src_stride_bytes)[w];
        }
#pragma unroll
        for (int r = 0; r < PER_T; r++) {
            const int i = tid + r * NT, key = i / WORDS, w = i - key * WORDS;
            dst[key * DST_STRIDE + w] = tmp[r];
        }
    } else {
        for (int i = tid; i < nvalid * WORDS; i += NT) {
            const int key = i / WORDS, w = i - key * WORDS;
            dst[key * DST_STRIDE + w] = reinterpret_cast<const int4*>((const char*)src + (size_t)key * src_stride_bytes)[w];
        }
    }
}

template <int HD>
__device__ __forceinline__ void load_k_tile(int4* Ks, float* sks, const int8_t* __restrict__ kq, const float* __restrict__ sk,
                                            int j, int nvalid, int n_kv, int kvh) {
    stage_rows<Tile<HD>::KW, Tile<HD>::KS>(Ks, kq + ((size_t)j * n_kv + kvh) * HD, (size_t)n_kv * HD, nvalid);
    if (threadIdx.x < nvalid) sks[threadIdx.x] = sk[(size_t)(j + threadIdx.x) * n_kv + kvh];
}

template <int HD>
__device__ __forceinline__ void load_v_tile(int16_t* Vs, const int16_t* __restrict__ vq, int j, int nvalid, int n_kv, int kvh) {
    stage_rows<HD / 8, HD / 8>(reinterpret_cast<int4*>(Vs), vq + ((size_t)j * n_kv + kvh) * HD, (size_t)n_kv * HD * 2, nvalid);
}

template <int HD>
__device__ __forceinline__ int dot_key(const int4* Ks, const int4* qs, int kk, int r) {
    constexpr int KW = Tile<HD>::KW, KS = Tile<HD>::KS;
    int dot = 0;
#pragma unroll
    for (int w = 0; w < KW; w++) {
        int4 a = Ks[kk * KS + w];
        int4 b = qs[r * KW + w];
        dot = __dp4a(a.x, b.x, dot); dot = __dp4a(a.y, b.y, dot);
        dot = __dp4a(a.z, b.z, dot); dot = __dp4a(a.w, b.w, dot);
    }
    return dot;
}

__device__ __forceinline__ void key_range(int nkeys, int nsplit, int split, int& j0, int& j1) {
    int chunk = ((nkeys + nsplit - 1) / nsplit + TK - 1) / TK * TK;
    j0 = split * chunk;
    j1 = min(nkeys, j0 + chunk);
}

// ---------------------------------------------------------------------------------------------------------------------
// pass 1: per-head max of s_j over this block's key range -> atomicMax into mx[t][h] (order-free)
template <int HD, int REP>
__global__ void __launch_bounds__(NT) k_attn_max(const int8_t* __restrict__ qq, const float* __restrict__ sq,
                                                 const int8_t* __restrict__ kq, const float* __restrict__ sk,
                                                 int n_h, int n_kv, int pos0, const int* __restrict__ pos0_dev, int nsplit,
                                                 unsigned* __restrict__ mx) {
    constexpr int KW = Tile<HD>::KW, KS = Tile<HD>::KS, NG = Tile<HD>::NG;
    __shared__ __align__(16) int4 Ks[TK * KS];
    __shared__ __align__(16) int4 qs[REP * KW];
    __shared__ float sks[TK];
    __shared__ unsigned bm[REP];
    const int t = blockIdx.z, kvh = blockIdx.y, tid = threadIdx.x;
    if (pos0_dev) pos0 = *pos0_dev;
    const int nkeys = pos0 + t + 1;
    int j0, j1; key_range(nkeys, nsplit, blockIdx.x, j0, j1);
    if (j0 >= j1) return;
    const int h0 = kvh * REP;
    for (int i = tid; i < REP * KW; i += NT) qs[i] = reinterpret_cast<const int4*>(qq + ((size_t)t * n_h + h0) * HD)[i];
    float sqv[REP];
#pragma unroll
    for (int r = 0; r < REP; r++) sqv[r] = sq[(size_t)t * n_h + h0 + r];
    if (tid < REP) bm[tid] = 0u;
    const int kk = tid % TK, g = tid / TK;
    float lm[REP];
#pragma unroll
    for (int r = 0; r < REP; r++) lm[r] = -INFINITY;
    for (int j = j0; j < j1; j += TK) {
        const int nvalid = min(TK, j1 - j);
        __syncthreads();                                  // previous tile fully consumed
        load_k_tile<HD>(Ks, sks, kq, sk, j, nvalid, n_kv, kvh);
        __syncthreads();
        if (kk < nvalid) {
#pragma unroll
            for (int r = 0; r < REP; r++) {
                if ((r % NG) == g) {
                    int dot = dot_key<HD>(Ks, qs, kk, r);
                    float s = mul(mul(__int2float_rn(dot), sqv[r]), sks[kk]);
                    lm[r] = fmaxf(lm[r], s);
                }
            }
        }
    }
    __syncthreads();
#pragma unroll
    for (int r = 0; r < REP; r++) atomicMax(&bm[r], f2ord(lm[r]));
    __syncthreads();
    if (tid < REP) atomicMax(&mx[(size_t)t * n_h + h0 + tid], bm[tid]);
}

// ---------------------------------------------------------------------------------------------------------------------
// pass 2: pq_j = rint(exp_f(s_j - mx) * 2^24); den += pq_j; num[d] += pq_j * vq_j[d]  (int64, order-free)
// One raw smem buffer is used in turn as the K tile (scores), the V tile (PV) and the block accumulators (epilogue).
template <int HD, int REP>
__global__ void __launch_bounds__(NT, 512 / NT) k_attn_acc(const int8_t* __restrict__ qq, const float* __restrict__ sq, const unsigned* __restrict__ mx,
                                                 const int8_t* __restrict__ kq, const float* __restrict__ sk, const int16_t* __restrict__ vq,
                                                 const float* __restrict__ qg, const float* __restrict__ v_scale,
                                                 int n_h, int n_kv, int pos0, const int* __restrict__ pos0_dev, int nsplit,
                                                 float* __restrict__ o, unsigned long long* __restrict__ num, unsigned long long* __restrict__ den) {
    constexpr int KW = Tile<HD>::KW, KS = Tile<HD>::KS, NG = Tile<HD>::NG;
    constexpr int DPT = 4;                 // dims per thread in the PV product (one 8-byte int16x4 smem read)
    constexpr int TPK = HD / DPT;          // threads per key
    constexpr int NKG = NT / TPK;          // key groups
    static_assert(NT % TPK == 0, "bad HD");
    constexpr int RAW_K = TK * KS * 16, RAW_V = TK * HD * 2, RAW_A = REP * HD * 8;
    constexpr int RAW = RAW_K > RAW_V ? (RAW_K > RAW_A ? RAW_K : RAW_A) : (RAW_V > RAW_A ? RAW_V : RAW_A);
    __shared__ __align__(16) unsigned char raw[RAW];
    __shared__ __align__(16) int4 qs[REP * KW];
    __shared__ float sks[TK];
    __shared__ int pqs[REP][TK];
    __shared__ unsigned long long dens[REP];
    int4* Ks = reinterpret_cast<int4*>(raw);
    int16_t* Vs = reinterpret_cast<int16_t*>(raw);
    unsigned long long* accs = reinterpret_cast<unsigned long long*>(raw);   // [REP][HD]
    const int t = blockIdx.z, kvh = blockIdx.y, tid = threadIdx.x;
    if (pos0_dev) pos0 = *pos0_dev;
    const int nkeys = pos0 + t + 1;
    int j0, j1; key_range(nkeys, nsplit, blockIdx.x, j0, j1);
    if (j0 >= j1) return;
    const int h0 = kvh * REP;
    for (int i = tid; i < REP * KW; i += NT) qs[i] = reinterpret_cast<const int4*>(qq + ((size_t)t * n_h + h0) * HD)[i];
    if (tid < REP) dens[tid] = 0ull;
    float sqv[REP], mxv[REP];
#pragma unroll
    for (int r = 0; r < REP; r++) { sqv[r] = sq[(size_t)t * n_h + h0 + r]; mxv[r] = ord2f(mx[(size_t)t * n_h + h0 + r]); }
    const int kk = tid % TK, g = tid / TK;
    const int d0 = (tid % TPK) * DPT, kg = tid / TPK;
    long long acc[REP][DPT];
    long long dl[REP];
#pragma unroll
    for (int r = 0; r < REP; r++) { dl[r] = 0;
#pragma unroll
        for (int d = 0; d < DPT; d++) acc[r][d] = 0; }
    for (int j = j0; j < j1; j += TK) {
        const int nvalid = min(TK, j1 - j);
        __syncthreads();                                  // previous tile fully consumed (Vs, pqs)
        load_k_tile<HD>(Ks, sks, kq, sk, j, nvalid, n_kv, kvh);
        __syncthreads();
#pragma unroll
        for (int r = 0; r < REP; r++) {
            if ((r % NG) == g) {
                int p = 0;
                if (kk < nvalid) {
                    int dot = dot_key<HD>(Ks, qs, kk, r);
                    float s = mul(mul(__int2float_rn(dot), sqv[r]), sks[kk]);
                    p = __float2int_rn(mul(exp_f(sub(s, mxv[r])), P_FIX));
                }
                pqs[r][kk] = p;
                dl[r] += p;
            }
        }
        __syncthreads();                                  // Ks dead, pqs ready
        load_v_tile<HD>(Vs, vq, j, nvalid, n_kv, kvh);
        __syncthreads();
        if (nvalid == TK) {
#pragma unroll 4
            for (int key = kg; key < TK; key += NKG) {
                int2 vv = *reinterpret_cast<const int2*>(Vs + key * HD + d0);
                int v0 = (int)(short)(vv.x & 0xffff), v1 = vv.x >> 16, v2 = (int)(short)(vv.y & 0xffff), v3 = vv.y >> 16;
#pragma unroll
                for (int r = 0; r < REP; r++) {
                    int p = pqs[r][key];
                    acc[r][0] += (long long)p * v0; acc[r][1] += (long long)p * v1;
                    acc[r][2] += (long long)p * v2; acc[r][3] += (long long)p * v3;
                }
            }
        } else {
            for (int key = kg; key < nvalid; key += NKG) {
                int2 vv = *reinterpret_cast<const int2*>(Vs + key * HD + d0);
                int v0 = (int)(short)(vv.x & 0xffff), v1 = vv.x >> 16, v2 = (int)(short)(vv.y & 0xffff), v3 = vv.y >> 16;
#pragma unroll
                for (int r = 0; r < REP; r++) {
                    int p = pqs[r][key];
                    acc[r][0] += (long long)p * v0; acc[r][1] += (long long)p * v1;
                    acc[r][2] += (long long)p * v2; acc[r][3] += (long long)p * v3;
                }
            }
        }
    }
    __syncthreads();                                      // Vs dead -> accs
    for (int i = tid; i < REP * HD; i += NT) accs[i] = 0ull;
    __syncthreads();
#pragma unroll
    for (int r = 0; r < REP; r++) {
#pragma unroll
        for (int d = 0; d < DPT; d++) atomicAdd(&accs[r * HD + d0 + d], (unsigned long long)acc[r][d]);
        atomicAdd(&dens[r], (unsigned long long)dl[r]);
    }
    __syncthreads();
    if (nsplit == 1) {
        const float vs = v_scale[kvh];
        for (int i = tid; i < REP * HD; i += NT) {
            const int r = i / HD, d = i - r * HD;
            const size_t hrow = (size_t)t * n_h + h0 + r;
            float ov = mul(div(i64_to_f((long long)accs[i]), i64_to_f((long long)dens[r])), vs);
            float gate = qg[hrow * 2 * HD + HD + d];
            o[hrow * HD + d] = mul(ov, sigmoid_f(gate));
        }
    } else {
        for (int i = tid; i < REP * HD; i += NT) {
            const int r = i / HD, d = i - r * HD;
            atomicAdd(&num[((size_t)t * n_h + h0 + r) * HD + d], accs[i]);
        }
        if (tid < REP) atomicAdd(&den[(size_t)t * n_h + h0 + tid], dens[tid]);
    }
}

// finalize (split case): o = ((float)num / (float)den) * v_scale * sigmoid(gate).  grid M*n_h, block HD.
__global__ void k_attn_fin(const unsigned long long* __restrict__ num, const unsigned long long* __restrict__ den,
                           const float* __restrict__ qg, const float* __restrict__ v_scale, int n_h, int rep, int hd,
                           float* __restrict__ o) {
    const int row = blockIdx.x, h = row % n_h, d = threadIdx.x;
    if (d >= hd) return;
    float ov = mul(div(i64_to_f((long long)num[(size_t)row * hd + d]), i64_to_f((long long)den[row])), v_scale[h / rep]);
    float gate = qg[(size_t)row * 2 * hd + hd + d];
    o[(size_t)row * hd + d] = mul(ov, sigmoid_f(gate));
}

// ---------------------------------------------------------------------------------------------------------------------
// Tensor-core prefill path (used when M >= tc_min_m(), default 16; env DET_ATTN_TC=0 disables, =N sets the threshold).
// Same numerics as the CUDA-core path above, same two passes; only the integer contractions move to mma.sync:
//   * scores:  s8 x s8 -> s32 m16n8k32 over hd (exact, |dot| < 2^23), then s = ((float)dot * sq) * sk as before.
//   * P.V:     pq (24 bits + the pq == 2^24 bit) and vq (int16) are byte-sliced: pq = p0 + p1<<8 + p2<<16 + p3<<24,
//              vq = vlo (u8) + vhi (s8) << 8.  num = sum_{a,b} 2^(8a+8b) * mma(p_a, v_b), each mma s32-exact within a
//              32-key tile; the s32 partials are widened into int64 per tile.  Slices that are all-zero in a tile
//              (warp-uniform flag) are skipped: their products are exactly zero, so the result is unchanged.
// Block = (RB = 48 flat rows (token t, head r) of one kv head, 12 warps).  Rows are ordered t-major so a block holds
// 48/rep consecutive tokens of all rep heads sharing the kv head: every K/V tile is staged once for 48 rows.
// Warp w = (m-tile mt = w/4, quarter q = w%4).  k_tc_max: warp (mt, q) scores keys q*32..+31 of each 128-key tile.
// k_tc_acc (32-key tiles, software pipelined, one barrier per tile): in the interval of tile i the warp computes the
// 16 x 8 score tile (keys q*8..+7) of tile i+1 for m-tile mt -> pq -> byte-sliced P fragments in smem, and the P.V
// of tile i for m-tile mt and dims [q*HD/4, +HD/4) from the P fragments all four warps of the m-tile wrote.
// Keys j > pos0 + t are masked (pq = 0); rows beyond M*rep (partial last block) are masked entirely.
// Work distribution: 1-D grid, heavy blocks first; the last partial wave is key-split (see decode_item).
namespace tc {
constexpr int MT = 3, RB = 16 * MT, WPM = 4, NW = MT * WPM, NTH = NW * 32;   // 48 rows, 12 warps
constexpr int TK = 32;                      // keys per tile in the accumulate pass (one m16n8k32 k-step of P.V)
constexpr int TKM = 128;                    // keys per tile in the max pass

__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src, bool pred) {
    const int n = pred ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(dst), "l"(src), "r"(n));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
__device__ __forceinline__ void cp_async_wait_all() { asm volatile("cp.async.wait_group 0;\n" ::); }
__device__ __forceinline__ void ldsm_x4(uint32_t addr, uint32_t* r) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}
__device__ __forceinline__ void ldsm_x4_t(uint32_t addr, uint32_t* r) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}
#define DET_TC_MMA(NAME, TA, TB) \
    __device__ __forceinline__ void NAME(int* c, const uint32_t* a, const uint32_t* b) { \
        asm volatile("mma.sync.aligned.m16n8k32.row.col.s32." TA "." TB ".s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n" \
                     : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3]) \
                     : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1])); }
DET_TC_MMA(mma_s8s8, "s8", "s8")
DET_TC_MMA(mma_u8u8, "u8", "u8")
DET_TC_MMA(mma_u8s8, "u8", "s8")
#undef DET_TC_MMA

__device__ __forceinline__ void tile_range(int ntiles, int nsplit, int split, int& t0, int& t1) {
    const int chunk = (ntiles + nsplit - 1) / nsplit;
    t0 = split * chunk; t1 = min(ntiles, t0 + chunk);
}
// Work items (1-D grid): blocks b = (kvh, qb) in heavy-first order (qb descending, same kvh adjacent so that
// concurrently running blocks stream the same K/V from L2).  Items [0, nfull) are whole blocks; the remaining
// r = nblk - nfull "tail" blocks (the last partial wave) are split into s key-range pieces each, which accumulate
// int64 partials into slot tb (one slot per tail block) and are finished by k_tc_fin.
__device__ __forceinline__ void decode_item(int w, int nqb, int nfull, int s, int& qb, int& kvh, int& piece, int& tb) {
    int b;
    if (w < nfull) { b = w; piece = 0; tb = -1; }
    else { const int u = w - nfull; tb = u / s; piece = u - tb * s; b = nfull + tb; }
    kvh = b / nqb; qb = nqb - 1 - (b - kvh * nqb);
}

// Per-thread row bookkeeping for the two rows (gid, gid+8) of its m-tile.  Flat row i of the block = (t, r) with
// t = i / REP, r = i % REP -> q row hrow = t*n_h + kvh*REP + r; lim = last valid key; lim = -1 for rows >= M*REP.
template <int REP>
__device__ __forceinline__ void row_info(int flat, int M, int n_h, int kvh, int pos0, int& hrow, int& lim) {
    if (flat < M * REP) { const int t = flat / REP, r = flat - t * REP; hrow = t * n_h + kvh * REP + r; lim = pos0 + t; }
    else { hrow = 0; lim = -1; }
}

// Q A-fragments of one m16 tile straight from global: a0 = (row gid, k tig*4..+3), a1 = row gid+8, a2/a3 = k + 16.
template <int HD>
__device__ __forceinline__ void load_q_frags(uint32_t (*qa)[4], const int8_t* __restrict__ qq, int hrow0, bool ok0, int hrow1, bool ok1, int tig) {
    const uint32_t* r0 = reinterpret_cast<const uint32_t*>(qq + (size_t)hrow0 * HD) + tig;
    const uint32_t* r1 = reinterpret_cast<const uint32_t*>(qq + (size_t)hrow1 * HD) + tig;
#pragma unroll
    for (int ks = 0; ks < HD / 32; ks++) {
        qa[ks][0] = ok0 ? r0[ks * 8] : 0u;  qa[ks][1] = ok1 ? r1[ks * 8] : 0u;
        qa[ks][2] = ok0 ? r0[ks * 8 + 4] : 0u;  qa[ks][3] = ok1 ? r1[ks * 8 + 4] : 0u;
    }
}

// Stage NK key rows of the int8 K cache into smem (row = HD bytes, 16-byte chunk c of key stored at chunk c ^ (key & 7)
// -> conflict-free ldmatrix over 8 consecutive keys).  Keys >= nkeys_total are zero-filled (and masked anyway).
template <int HD, int NK>
__device__ __forceinline__ void stage_k(unsigned char* Kb, const int8_t* __restrict__ kq, int j0, int nkeys_total, int n_kv, int kvh) {
    constexpr int CH = HD / 16;
    for (int c = threadIdx.x; c < NK * CH; c += NTH) {
        const int key = c / CH, ch = c - key * CH;
        const bool ok = j0 + key < nkeys_total;
        const void* src = ok ? (const void*)(kq + ((size_t)(j0 + key) * n_kv + kvh) * HD + ch * 16) : (const void*)kq;
        cp_async16(smem_u32(Kb + key * HD + ((ch ^ (key & 7)) << 4)), src, ok);
    }
}
template <int HD, int NK>
__device__ __forceinline__ void stage_v(unsigned char* Vb, const int16_t* __restrict__ vq, int j0, int nkeys_total, int n_kv, int kvh) {
    constexpr int CH = HD / 8;
    for (int c = threadIdx.x; c < NK * CH; c += NTH) {
        const int key = c / CH, ch = c - key * CH;
        const bool ok = j0 + key < nkeys_total;
        const void* src = ok ? (const void*)(vq + ((size_t)(j0 + key) * n_kv + kvh) * HD + ch * 8) : (const void*)vq;
        cp_async16(smem_u32(Vb + key * HD * 2 + ((ch ^ (key & 7)) << 4)), src, ok);
    }
}

__device__ __forceinline__ float score_f(int dot, float sq, float sk) { return mul(mul(__int2float_rn(dot), sq), sk); }
__device__ __forceinline__ int score_p(int dot, float sq, float sk, float mx, bool valid) {
    const float s = score_f(dot, sq, sk);
    return valid ? __float2int_rn(mul(exp_f(sub(s, mx)), P_FIX)) : 0;
}

// ---- pass 1: per-row max (atomicMax of the order-preserving code into mx[hrow]).  1-D grid of work items.
template <int HD, int REP>
__global__ void __launch_bounds__(NTH, 1) k_tc_max(const int8_t* __restrict__ qq, const float* __restrict__ sq,
                                                   const int8_t* __restrict__ kq, const float* __restrict__ sk,
                                                   int M, int n_h, int n_kv, int pos0, int nqb, int nfull, int s, unsigned* __restrict__ mx) {
    extern __shared__ __align__(16) unsigned char dsm[];
    unsigned char* Kb = dsm;                                                   // [2][TKM][HD]
    float* sks = reinterpret_cast<float*>(dsm + 2 * TKM * HD);                 // [2][TKM]
    unsigned* mxs = reinterpret_cast<unsigned*>(sks + 2 * TKM);                // [RB]
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, gid = lane >> 2, tig = lane & 3;
    const int mt = warp / WPM, q = warp % WPM;
    int qb, kvh, piece, tb; decode_item(blockIdx.x, nqb, nfull, s, qb, kvh, piece, tb);
    const int rowbase = qb * RB;
    const int t_max = min(M - 1, (rowbase + RB - 1) / REP);
    const int nkeys_total = pos0 + M;
    const int ntiles = (pos0 + t_max + 1 + TKM - 1) / TKM;
    int tile0 = 0, tile1 = ntiles;
    if (tb >= 0) tile_range(ntiles, s, piece, tile0, tile1);
    if (tile0 >= tile1) return;
    int hrow[2], lim[2];
    row_info<REP>(rowbase + mt * 16 + gid, M, n_h, kvh, pos0, hrow[0], lim[0]);
    row_info<REP>(rowbase + mt * 16 + gid + 8, M, n_h, kvh, pos0, hrow[1], lim[1]);
    uint32_t qa[HD / 32][4];
    load_q_frags<HD>(qa, qq, hrow[0], lim[0] >= 0, hrow[1], lim[1] >= 0, tig);
    const float sqv[2] = { lim[0] >= 0 ? sq[hrow[0]] : 0.0f, lim[1] >= 0 ? sq[hrow[1]] : 0.0f };
    if (tid < RB) mxs[tid] = 0u;
    float lm[2] = { -INFINITY, -INFINITY };
    // sk for a tile is loaded into a register one tile ahead and stored to smem just before the tile's barrier
    float skr = 0.0f;
    auto load_sk = [&](int tile) { const int key = tile * TKM + tid; skr = (tid < TKM && key < nkeys_total) ? sk[(size_t)key * n_kv + kvh] : 0.0f; };
    stage_k<HD, TKM>(Kb, kq, tile0 * TKM, nkeys_total, n_kv, kvh);
    cp_async_commit();
    load_sk(tile0);
    for (int tile = tile0; tile < tile1; tile++) {
        const int stage = (tile - tile0) & 1;
        cp_async_wait_all();
        if (tid < TKM) sks[stage * TKM + tid] = skr;
        __syncthreads();
        if (tile + 1 < tile1) { stage_k<HD, TKM>(Kb + (stage ^ 1) * TKM * HD, kq, (tile + 1) * TKM, nkeys_total, n_kv, kvh); load_sk(tile + 1); }
        cp_async_commit();
        const unsigned char* K = Kb + stage * TKM * HD;
        const float* skt = sks + stage * TKM;
        constexpr int NN = TKM / WPM / 8;                  // n8 tiles per warp per tile: independent MMA chains, interleaved
        int cc[NN][4];
#pragma unroll
        for (int n = 0; n < NN; n++) { cc[n][0] = cc[n][1] = cc[n][2] = cc[n][3] = 0; }
#pragma unroll
        for (int p = 0; p < HD / 64; p++) {
#pragma unroll
            for (int n = 0; n < NN; n++) {
                uint32_t r[4];
                const int key = q * (TKM / WPM) + n * 8 + (lane & 7), ch = 4 * p + (lane >> 3);
                ldsm_x4(smem_u32(K + key * HD + ((ch ^ (key & 7)) << 4)), r);
                mma_s8s8(cc[n], qa[2 * p], r);
                mma_s8s8(cc[n], qa[2 * p + 1], r + 2);
            }
        }
#pragma unroll
        for (int n = 0; n < NN; n++) {
            const int kb = q * (TKM / WPM) + n * 8;
            const int* c = cc[n];
            const int k0 = tile * TKM + kb + tig * 2;
            const float sk0 = skt[kb + tig * 2], sk1 = skt[kb + tig * 2 + 1];
            if (k0 <= lim[0])     lm[0] = fmaxf(lm[0], score_f(c[0], sqv[0], sk0));
            if (k0 + 1 <= lim[0]) lm[0] = fmaxf(lm[0], score_f(c[1], sqv[0], sk1));
            if (k0 <= lim[1])     lm[1] = fmaxf(lm[1], score_f(c[2], sqv[1], sk0));
            if (k0 + 1 <= lim[1]) lm[1] = fmaxf(lm[1], score_f(c[3], sqv[1], sk1));
        }
    }
#pragma unroll
    for (int i = 0; i < 2; i++) {
        lm[i] = fmaxf(lm[i], __shfl_xor_sync(0xffffffffu, lm[i], 1));
        lm[i] = fmaxf(lm[i], __shfl_xor_sync(0xffffffffu, lm[i], 2));
    }
    if (tig == 0) { atomicMax(&mxs[mt * 16 + gid], f2ord(lm[0])); atomicMax(&mxs[mt * 16 + gid + 8], f2ord(lm[1])); }
    __syncthreads();
    if (tid < RB) {
        int hr, li; row_info<REP>(rowbase + tid, M, n_h, kvh, pos0, hr, li);
        if (li >= 0) atomicMax(&mx[hr], mxs[tid]);
    }
}

// ---- pass 2: P.V accumulate.  1-D grid of work items: whole blocks write o directly, tail pieces add int64
// partials into their slot (k_tc_fin finishes them).
// Software pipeline, one barrier per TK-key tile: in the interval for tile i every warp computes the scores of tile i+1
// (tensor-light, exp-heavy) and the P.V of tile i (tensor-heavy); warps 4..7 do them in the opposite order so that each
// SM sub-partition always has warps in both phases.  K, V, Pf, flags and sk are double-buffered by tile parity.
// Pf layout: [2][MT][4 slices][32 lanes][4 regs] = the P.V A fragments, written as u16 halves by the score warps
// (warp q owns n8 tile q of its m-tile: regs (q>>1)*2 + rowhalf, u16 half = q & 1).
template <int HD, int REP>
__global__ void __launch_bounds__(NTH, 1) k_tc_acc(const int8_t* __restrict__ qq, const float* __restrict__ sq, const unsigned* __restrict__ mx,
                                                   const int8_t* __restrict__ kq, const float* __restrict__ sk, const int16_t* __restrict__ vq,
                                                   const float* __restrict__ qg, const float* __restrict__ v_scale,
                                                   int M, int n_h, int n_kv, int pos0, int nqb, int nfull, int s,
                                                   float* __restrict__ o, unsigned long long* __restrict__ tcnum, unsigned long long* __restrict__ tcden) {
    static_assert(TK == 32, "one P.V k-step per tile");
    constexpr int NTN = HD / 32;                   // n8 tiles per warp in P.V (HD/4 dims)
    constexpr int PF_U32 = MT * 4 * 32 * 4;        // u32 words of one Pf buffer
    extern __shared__ __align__(16) unsigned char dsm[];
    unsigned char* Qs = dsm;                                                   // [RB][HD]         int8 (swizzled)
    unsigned char* Kb = Qs + RB * HD;                                          // [2][TK][HD]      int8
    unsigned char* Vb = Kb + 2 * TK * HD;                                      // [2][TK][HD]      int16
    uint32_t* Pf = reinterpret_cast<uint32_t*>(Vb + 2 * TK * HD * 2);          // [2][MT][4][32][4]
    float* sks = reinterpret_cast<float*>(Pf + 2 * PF_U32);                    // [2][TK]
    unsigned long long* dens = reinterpret_cast<unsigned long long*>(sks + 2 * TK);   // [RB]
    unsigned* flags = reinterpret_cast<unsigned*>(dens + RB);                  // [2][MT][WPM]
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, gid = lane >> 2, tig = lane & 3;
    const int mt = warp / WPM, q = warp % WPM;
    const bool pv_first = ((warp >> 2) & 1) != 0;
    int qb, kvh, piece, tb; decode_item(blockIdx.x, nqb, nfull, s, qb, kvh, piece, tb);
    const int rowbase = qb * RB;
    const int t_max = min(M - 1, (rowbase + RB - 1) / REP);
    const int nkeys_total = pos0 + M;
    const int ntiles = (pos0 + t_max + 1 + TK - 1) / TK;
    int tile0 = 0, tile1 = ntiles;
    if (tb >= 0) tile_range(ntiles, s, piece, tile0, tile1);
    if (tile0 >= tile1) return;
    int hrow[2], lim[2];
    row_info<REP>(rowbase + mt * 16 + gid, M, n_h, kvh, pos0, hrow[0], lim[0]);
    row_info<REP>(rowbase + mt * 16 + gid + 8, M, n_h, kvh, pos0, hrow[1], lim[1]);
    float sqv[2], mxv[2];
#pragma unroll
    for (int i = 0; i < 2; i++) { sqv[i] = lim[i] >= 0 ? sq[hrow[i]] : 0.0f; mxv[i] = lim[i] >= 0 ? ord2f(mx[hrow[i]]) : 0.0f; }
    if (tid < RB) dens[tid] = 0ull;
    long long dl[2] = { 0, 0 };
    long long acc[NTN][4];
#pragma unroll
    for (int n = 0; n < NTN; n++) { acc[n][0] = acc[n][1] = acc[n][2] = acc[n][3] = 0; }
    // Q tile -> smem (chunk c of row r at c ^ (r & 7)); rows beyond M*REP zero-filled
    {
        constexpr int CH = HD / 16;
        for (int c = tid; c < RB * CH; c += NTH) {
            const int r = c / CH, ch = c - r * CH;
            int hr, li; row_info<REP>(rowbase + r, M, n_h, kvh, pos0, hr, li);
            const bool ok = li >= 0;
            cp_async16(smem_u32(Qs + r * HD + ((ch ^ (r & 7)) << 4)), ok ? (const void*)(qq + (size_t)hr * HD + ch * 16) : (const void*)qq, ok);
        }
    }
    // per-lane ldmatrix addressing of the Q A-fragments: matrix (lane>>3): rows +((lane>>3)&1)*8, k-chunk +(lane>>4)
    const int qrow = mt * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
    const uint32_t qs_base = smem_u32(Qs + qrow * HD);
    const int qchk = lane >> 4;
    // P fragment slots this thread writes (u16 units, within one Pf buffer) and reads (uint4 per slice)
    const int pfw_off = ((mt * 4) * 32 + lane) * 8 + (q >> 1) * 4 + (q & 1);
    const int pfr_off = (mt * 4) * 32 + lane;
    auto sk_of = [&](int tile) -> float { const int key = tile * TK + tid; return (tid < TK && key < nkeys_total) ? sk[(size_t)key * n_kv + kvh] : 0.0f; };
    auto scores = [&](int tile, int buf) {
        const unsigned char* K = Kb + buf * TK * HD;
        const float* skt = sks + buf * TK;
        int c[4] = { 0, 0, 0, 0 }, c2[4] = { 0, 0, 0, 0 };     // two independent chains (integer sums: exact)
        const int key = q * 8 + (lane & 7);
#pragma unroll
        for (int p = 0; p < HD / 64; p++) {
            uint32_t a0[4], a1[4], r[4];
            ldsm_x4(qs_base + (((4 * p + qchk) ^ (qrow & 7)) << 4), a0);
            ldsm_x4(qs_base + (((4 * p + 2 + qchk) ^ (qrow & 7)) << 4), a1);
            const int ch = 4 * p + (lane >> 3);
            ldsm_x4(smem_u32(K + key * HD + ((ch ^ (key & 7)) << 4)), r);
            mma_s8s8(c, a0, r);
            mma_s8s8(c2, a1, r + 2);
        }
#pragma unroll
        for (int e = 0; e < 4; e++) c[e] += c2[e];
        const int k0 = tile * TK + q * 8 + tig * 2;
        const float sk0 = skt[q * 8 + tig * 2], sk1 = skt[q * 8 + tig * 2 + 1];
        const int p0 = score_p(c[0], sqv[0], sk0, mxv[0], k0 <= lim[0]);
        const int p1 = score_p(c[1], sqv[0], sk1, mxv[0], k0 + 1 <= lim[0]);
        const int p2 = score_p(c[2], sqv[1], sk0, mxv[1], k0 <= lim[1]);
        const int p3 = score_p(c[3], sqv[1], sk1, mxv[1], k0 + 1 <= lim[1]);
        dl[0] += p0 + p1; dl[1] += p2 + p3;
        const unsigned wor = __reduce_or_sync(0xffffffffu, (unsigned)(p0 | p1 | p2 | p3));
        if (lane == 0) flags[(buf * MT + mt) * WPM + q] = wor;
        uint16_t* pfw = reinterpret_cast<uint16_t*>(Pf + buf * PF_U32) + pfw_off;
        pfw[0 * 256] = (uint16_t)__byte_perm(p0, p1, 0x40); pfw[0 * 256 + 2] = (uint16_t)__byte_perm(p2, p3, 0x40);
        pfw[1 * 256] = (uint16_t)__byte_perm(p0, p1, 0x51); pfw[1 * 256 + 2] = (uint16_t)__byte_perm(p2, p3, 0x51);
        pfw[2 * 256] = (uint16_t)__byte_perm(p0, p1, 0x62); pfw[2 * 256 + 2] = (uint16_t)__byte_perm(p2, p3, 0x62);
        pfw[3 * 256] = (uint16_t)__byte_perm(p0, p1, 0x73); pfw[3 * 256 + 2] = (uint16_t)__byte_perm(p2, p3, 0x73);
    };
    // V B-fragments of n8 tile n (dims q*HD/4 + n*8 ..): ldmatrix.trans of the 4 key groups, then byte-sliced
    auto load_v = [&](const unsigned char* V, int n, uint32_t* vlo, uint32_t* vhi) {
        uint32_t r[4];
        const int ch = q * NTN + n;
        ldsm_x4_t(smem_u32(V + lane * HD * 2 + ((ch ^ (lane & 7)) << 4)), r);
        vlo[0] = __byte_perm(r[0], r[1], 0x6420); vlo[1] = __byte_perm(r[2], r[3], 0x6420);
        vhi[0] = __byte_perm(r[0], r[1], 0x7531); vhi[1] = __byte_perm(r[2], r[3], 0x7531);
    };
    auto pv = [&](int buf) {
        const unsigned* fl = flags + (buf * MT + mt) * WPM;
        const unsigned f = fl[0] | fl[1] | fl[2] | fl[3];
        if (f == 0u) return;
        const int ns = f < 256u ? 1 : (f < 65536u ? 2 : (f < (1u << 24) ? 3 : 4));   // active byte slices of pq
        const uint4* pfr = reinterpret_cast<const uint4*>(Pf + buf * PF_U32) + pfr_off;
        const unsigned char* V = Vb + buf * TK * HD * 2;
        // NS slices: C_s = sum_{a+b=s} p_a . v_b  (b in {0: lo u8, 1: hi s8}), s = 0..NS; num += sum_s C_s << 8s.
        // The n8 tiles are software pipelined: V fragments of n+1 are loaded before the MMAs of n are issued, and the
        // widening of n is done after the MMAs of n+1 (two C sets live).
        auto run = [&](auto NSC) {
            constexpr int NS = decltype(NSC)::value;
            uint32_t pa[NS][4];
#pragma unroll
            for (int a = 0; a < NS; a++) { const uint4 v = pfr[a * 32]; pa[a][0] = v.x; pa[a][1] = v.y; pa[a][2] = v.z; pa[a][3] = v.w; }
            constexpr int DELAY = NS <= 2 ? 1 : 0;        // delayed widening (two C sets live) only when registers allow
            int C[DELAY + 1][NS + 1][4];
            uint32_t vlo[2], vhi[2];
            load_v(V, 0, vlo, vhi);
            auto widen = [&](const int (*Cp)[4], int n) {
#pragma unroll
                for (int e = 0; e < 4; e++) {
                    long long t = (long long)Cp[0][e];
#pragma unroll
                    for (int sl = 1; sl <= NS; sl++) t += (long long)Cp[sl][e] << (8 * sl);
                    acc[n][e] += t;
                }
            };
#pragma unroll
            for (int n = 0; n < NTN; n++) {
                int (*Cn)[4] = C[n & DELAY];
#pragma unroll
                for (int sl = 0; sl <= NS; sl++) { Cn[sl][0] = Cn[sl][1] = Cn[sl][2] = Cn[sl][3] = 0; }
#pragma unroll
                for (int a = 0; a < NS; a++) { mma_u8u8(Cn[a], pa[a], vlo); mma_u8s8(Cn[a + 1], pa[a], vhi); }
                if (n + 1 < NTN) load_v(V, n + 1, vlo, vhi);
                if (DELAY) { if (n > 0) widen(C[(n - 1) & DELAY], n - 1); }
                else widen(Cn, n);
            }
            if (DELAY) widen(C[(NTN - 1) & DELAY], NTN - 1);
        };
        switch (ns) {
            case 1: run(std::integral_constant<int, 1>()); break;
            case 2: run(std::integral_constant<int, 2>()); break;
            case 3: run(std::integral_constant<int, 3>()); break;
            default: run(std::integral_constant<int, 4>()); break;
        }
    };
    // ---- prologue: Q, K(t0), V(t0) [group A]; K(t0+1) [group B]; sk(t0), sk(t0+1) direct; scores(t0)
    stage_k<HD, TK>(Kb, kq, tile0 * TK, nkeys_total, n_kv, kvh);
    stage_v<HD, TK>(Vb, vq, tile0 * TK, nkeys_total, n_kv, kvh);
    cp_async_commit();
    if (tile0 + 1 < tile1) stage_k<HD, TK>(Kb + TK * HD, kq, (tile0 + 1) * TK, nkeys_total, n_kv, kvh);
    cp_async_commit();
    if (tid < TK) { sks[tid] = sk_of(tile0); sks[TK + tid] = sk_of(tile0 + 1); }
    float skr = sk_of(tile0 + 2);
    asm volatile("cp.async.wait_group 1;\n" ::);
    __syncthreads();
    scores(tile0, 0);
    for (int tile = tile0; tile < tile1; tile++) {
        const int cur = (tile - tile0) & 1;
        cp_async_wait_all();                               // K(tile+1), V(tile) landed (this thread)
        __syncthreads();                                   // ... and visible; Pf[cur] complete; everyone done with tile-1
        if (tile + 2 < tile1) stage_k<HD, TK>(Kb + cur * TK * HD, kq, (tile + 2) * TK, nkeys_total, n_kv, kvh);
        if (tile + 1 < tile1) stage_v<HD, TK>(Vb + (cur ^ 1) * TK * HD * 2, vq, (tile + 1) * TK, nkeys_total, n_kv, kvh);
        cp_async_commit();
        if (tid < TK) sks[cur * TK + tid] = skr;          // sk(tile+2): sks[cur] held sk(tile), consumed in the previous interval
        skr = sk_of(tile + 3);
        if (pv_first) { pv(cur); if (tile + 1 < tile1) scores(tile + 1, cur ^ 1); }
        else          { if (tile + 1 < tile1) scores(tile + 1, cur ^ 1); pv(cur); }
    }
    // ---- epilogue: den per row (smem), then o or int64 partials
#pragma unroll
    for (int i = 0; i < 2; i++) {
        dl[i] += __shfl_xor_sync(0xffffffffu, dl[i], 1);
        dl[i] += __shfl_xor_sync(0xffffffffu, dl[i], 2);
    }
    if (tig == 0) { atomicAdd(&dens[mt * 16 + gid], (unsigned long long)dl[0]); atomicAdd(&dens[mt * 16 + gid + 8], (unsigned long long)dl[1]); }
    __syncthreads();
    if (tb < 0) {
        const float vs = v_scale[kvh];
#pragma unroll
        for (int i = 0; i < 2; i++) {
            if (lim[i] < 0) continue;
            const float denf = i64_to_f((long long)dens[mt * 16 + gid + 8 * i]);
            const float* gate = qg + (size_t)hrow[i] * 2 * HD + HD;
            float* orow = o + (size_t)hrow[i] * HD;
#pragma unroll
            for (int n = 0; n < NTN; n++) {
                const int d = q * (HD / 4) + n * 8 + tig * 2;
                const float2 g = *reinterpret_cast<const float2*>(gate + d);
                float2 ov;
                ov.x = mul(mul(div(i64_to_f(acc[n][2 * i]), denf), vs), sigmoid_f(g.x));
                ov.y = mul(mul(div(i64_to_f(acc[n][2 * i + 1]), denf), vs), sigmoid_f(g.y));
                *reinterpret_cast<float2*>(orow + d) = ov;
            }
        }
    } else {                                               // tail piece: int64 partials into slot tb
#pragma unroll
        for (int i = 0; i < 2; i++) {
            if (lim[i] < 0) continue;
            unsigned long long* nrow = tcnum + ((size_t)tb * RB + mt * 16 + gid + 8 * i) * HD;
#pragma unroll
            for (int n = 0; n < NTN; n++) {
                const int d = q * (HD / 4) + n * 8 + tig * 2;
                atomicAdd(nrow + d, (unsigned long long)acc[n][2 * i]);
                atomicAdd(nrow + d + 1, (unsigned long long)acc[n][2 * i + 1]);
            }
        }
        if (tid < RB) atomicAdd(&tcden[(size_t)tb * RB + tid], dens[tid]);
    }
}

// finalize the tail-split blocks: grid r*RB (one row each), block HD
template <int HD, int REP>
__global__ void __launch_bounds__(HD) k_tc_fin(const unsigned long long* __restrict__ tcnum, const unsigned long long* __restrict__ tcden,
                                              const float* __restrict__ qg, const float* __restrict__ v_scale,
                                              int M, int n_h, int pos0, int nqb, int nfull, float* __restrict__ o) {
    const int tb = blockIdx.x / RB, i = blockIdx.x - tb * RB, d = threadIdx.x;
    int qb, kvh, piece, tbb; decode_item(nfull + tb, nqb, nfull, 1, qb, kvh, piece, tbb);
    int hrow, lim; row_info<REP>(qb * RB + i, M, n_h, kvh, pos0, hrow, lim);
    if (lim < 0) return;
    const float ov = mul(div(i64_to_f((long long)tcnum[((size_t)tb * RB + i) * HD + d]), i64_to_f((long long)tcden[(size_t)tb * RB + i])), v_scale[kvh]);
    const float gate = qg[(size_t)hrow * 2 * HD + HD + d];
    o[(size_t)hrow * HD + d] = mul(ov, sigmoid_f(gate));
}

template <int HD> constexpr size_t smem_max() { return (size_t)2 * TKM * HD + 2 * TKM * 4 + RB * 4; }
template <int HD> constexpr size_t smem_acc() { return (size_t)RB * HD + 2 * TK * HD + 2 * TK * HD * 2 + 2 * MT * 4 * 32 * 16 + 2 * TK * 4 + RB * 8 + 2 * MT * WPM * 4; }

static int num_sms() {
    static int n = 0;
    if (n == 0) { int dev = 0; DET_CHECK(cudaGetDevice(&dev)); DET_CHECK(cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev)); if (n <= 0) n = 1; }
    return n;
}

template <int HD, int REP>
static void run(const det_attn_layer* L, const det_kv_cache* kv, const float* qg, int M, int pos0, int n_h, int n_kv,
                float* o, const Scratch& S, cudaStream_t st) {
    static_assert(RB == TC_RB, "scratch layout");
    static bool attr_done = false;
    if (!attr_done) {
        DET_CHECK(cudaFuncSetAttribute(k_tc_max<HD, REP>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_max<HD>()));
        DET_CHECK(cudaFuncSetAttribute(k_tc_acc<HD, REP>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_acc<HD>()));
        attr_done = true;
    }
    const int nqb = (M * REP + RB - 1) / RB;
    const int nblk = nqb * n_kv;
    // wave quantization: the last partial wave of r blocks is split into s key-range pieces each (see decode_item)
    const int sms = num_sms();
    int nfull = nblk, r = nblk % sms, s = 1;
    if (r > 0 && r <= S.tc_slots) {
        s = (4 * sms + r - 1) / r;
        if (s > 32) s = 32;
        if (s >= 2) nfull = nblk - r; else { s = 1; r = 0; }
    } else r = 0;
    const int items = nfull + r * s;
    DET_CHECK(cudaMemsetAsync(S.mx, 0, (size_t)M * n_h * 4, st));
    if (r > 0) {
        DET_CHECK(cudaMemsetAsync(S.tcnum, 0, (size_t)r * RB * HD * 8, st));
        DET_CHECK(cudaMemsetAsync(S.tcden, 0, (size_t)r * RB * 8, st));
    }
    k_tc_max<HD, REP><<<items, NTH, smem_max<HD>(), st>>>(S.qq, S.sq, kv->kq, kv->sk, M, n_h, n_kv, pos0, nqb, nfull, s, S.mx);
    k_tc_acc<HD, REP><<<items, NTH, smem_acc<HD>(), st>>>(S.qq, S.sq, S.mx, kv->kq, kv->sk, kv->vq, qg, L->v_scale, M, n_h, n_kv, pos0,
                                                           nqb, nfull, s, o, S.tcnum, S.tcden);
    if (r > 0) k_tc_fin<HD, REP><<<r * RB, HD, 0, st>>>(S.tcnum, S.tcden, qg, L->v_scale, M, n_h, pos0, nqb, nfull, o);
}
} // namespace tc

// threshold for the tensor-core path: env DET_ATTN_TC = "0" disables it, "N" uses it for M >= N.  Default 16: measured
// on the RTX 5090 the tensor-core path is never slower than the CUDA-core path for M >= 16 (equal at pos0 = 0, 2-8x
// faster at any nonzero context); decode / short chunks keep the CUDA-core path with its key splitting.
static int tc_min_m() {
    const char* e = getenv("DET_ATTN_TC");
    if (!e || !*e) return 16;
    const int v = atoi(e);
    return v <= 0 ? 0x7fffffff : v;
}

// pos0_dev == nullptr: pos0 is the position (det_attention).  Otherwise the kernels read pos0 from *pos0_dev and the
// launch is sized for pos0 <= max_pos (det_attention_dev; CUDA-core path only).
template <int HD, int REP>
static void run_attention(const det_attn_layer* L, const det_kv_cache* kv, const float* qg, const float* k, const float* v,
                          const float* rope_cos, const float* rope_sin, int rot, float eps, int M, int pos0, const int* pos0_dev,
                          int max_pos, int n_h, int n_kv, float* o, const Scratch& S, cudaStream_t st) {
    const float qscale = (float)(1.0 / sqrt((double)HD));
    const int rows = M * n_h + 2 * M * n_kv;
    k_attn_prep<HD><<<(rows + 7) / 8, 256, 0, st>>>(qg, k, v, L->q_norm, L->k_norm, L->v_inv_scale, rope_cos, rope_sin, rot, eps, qscale,
                                                    M, pos0, pos0_dev, n_h, n_kv, S.qq, S.sq, kv->kq, kv->sk, kv->vq);
    if (!pos0_dev && M >= tc_min_m()) { tc::run<HD, REP>(L, kv, qg, M, pos0, n_h, n_kv, o, S, st); return; }
    const int nkeys_max = (pos0_dev ? max_pos : pos0) + M;
    int nsplit = 1;
    if (M <= MSPLIT) {
        int want = SPLIT_TARGET_BLOCKS / (M * n_kv);
        int maxs = (nkeys_max + TK - 1) / TK;
        nsplit = want < maxs ? want : maxs;
        if (nsplit < 1) nsplit = 1;
    }
    DET_CHECK(cudaMemsetAsync(S.mx, 0, (size_t)M * n_h * 4, st));
    if (nsplit > 1) {
        DET_CHECK(cudaMemsetAsync(S.den, 0, (size_t)M * n_h * 8, st));
        DET_CHECK(cudaMemsetAsync(S.num, 0, (size_t)M * n_h * HD * 8, st));
    }
    dim3 grid(nsplit, n_kv, M);
    k_attn_max<HD, REP><<<grid, NT, 0, st>>>(S.qq, S.sq, kv->kq, kv->sk, n_h, n_kv, pos0, pos0_dev, nsplit, S.mx);
    k_attn_acc<HD, REP><<<grid, NT, 0, st>>>(S.qq, S.sq, S.mx, kv->kq, kv->sk, kv->vq, qg, L->v_scale, n_h, n_kv, pos0, pos0_dev, nsplit,
                                             o, S.num, S.den);
    if (nsplit > 1) k_attn_fin<<<M * n_h, HD, 0, st>>>(S.num, S.den, qg, L->v_scale, n_h, REP, HD, o);
}

template <int HD>
static void dispatch_rep(int rep, const det_attn_layer* L, const det_kv_cache* kv, const float* qg, const float* k, const float* v,
                         const float* rope_cos, const float* rope_sin, int rot, float eps, int M, int pos0, const int* pos0_dev,
                         int max_pos, int n_h, int n_kv, float* o, const Scratch& S, cudaStream_t st) {
#define DET_ATTN_CASE(R) case R: run_attention<HD, R>(L, kv, qg, k, v, rope_cos, rope_sin, rot, eps, M, pos0, pos0_dev, max_pos, n_h, n_kv, o, S, st); break;
    switch (rep) {
        DET_ATTN_CASE(1) DET_ATTN_CASE(2) DET_ATTN_CASE(3) DET_ATTN_CASE(4) DET_ATTN_CASE(6) DET_ATTN_CASE(8)
        default: fprintf(stderr, "det_attention: unsupported n_h/n_kv = %d\n", rep); abort();
    }
#undef DET_ATTN_CASE
}

} // namespace

size_t det_attn_scratch_bytes(int M, int n_h, int hd) { return layout(nullptr, M, n_h, hd).bytes; }

static void attention_impl(const det_attn_layer* L, const det_kv_cache* kv, const float* qg, const float* k, const float* v,
                           const float* rope_cos, const float* rope_sin, int rot, float eps, int M, int pos0, const int* pos0_dev,
                           int max_pos, int n_h, int n_kv, int hd, float* o, void* scratch, cudaStream_t st) {
    const char* who = pos0_dev ? "det_attention_dev" : "det_attention";
    if (M <= 0) return;
    if (n_h % n_kv != 0 || rot < 0 || rot > hd || (rot & 1)) { fprintf(stderr, "%s: bad n_h=%d n_kv=%d rot=%d hd=%d\n", who, n_h, n_kv, rot, hd); abort(); }
    if (M > 65535) { fprintf(stderr, "%s: M=%d too large (grid.z)\n", who, M); abort(); }
    const int pmax = pos0_dev ? max_pos : pos0;
    if (pmax < 0 || pmax + M > kv->capacity) { fprintf(stderr, "%s: KV cache overflow (%s %d + %d > %d)\n", who, pos0_dev ? "max_pos" : "pos0", pmax, M, kv->capacity); abort(); }
    const Scratch S = layout(scratch, M, n_h, hd);
    const int rep = n_h / n_kv;
    switch (hd) {
        case 256: dispatch_rep<256>(rep, L, kv, qg, k, v, rope_cos, rope_sin, rot, eps, M, pos0, pos0_dev, max_pos, n_h, n_kv, o, S, st); break;
        case 128: dispatch_rep<128>(rep, L, kv, qg, k, v, rope_cos, rope_sin, rot, eps, M, pos0, pos0_dev, max_pos, n_h, n_kv, o, S, st); break;
        default: fprintf(stderr, "%s: unsupported hd=%d\n", who, hd); abort();
    }
}

void det_attention(const det_attn_layer* L, const det_kv_cache* kv, const float* qg, const float* k, const float* v,
                   const float* rope_cos, const float* rope_sin, int rot, float eps,
                   int M, int pos0, int n_h, int n_kv, int hd, float* o, void* scratch, cudaStream_t st) {
    attention_impl(L, kv, qg, k, v, rope_cos, rope_sin, rot, eps, M, pos0, nullptr, 0, n_h, n_kv, hd, o, scratch, st);
}

void det_attention_dev(const det_attn_layer* L, const det_kv_cache* kv, const float* qg, const float* k, const float* v,
                       const float* rope_cos, const float* rope_sin, int rot, float eps,
                       int M, const int* pos0_dev, int max_pos, int n_h, int n_kv, int hd, float* o, void* scratch, cudaStream_t st) {
    if (!pos0_dev) { fprintf(stderr, "det_attention_dev: pos0_dev == NULL\n"); abort(); }
    attention_impl(L, kv, qg, k, v, rope_cos, rope_sin, rot, eps, M, 0, pos0_dev, max_pos, n_h, n_kv, hd, o, scratch, st);
}
