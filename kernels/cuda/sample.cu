// Deterministic sampling (spec/sample.rs): temperature, top-k, top-p, multinomial.
//   l_i = mul(logit_i, inv_temp); mx = max_i l_i; w_i = (uint64) rint( mul( exp_f( sub(l_i, mx) ), 2^32 ) )
//   order = sort by (w desc, index asc)   [CUB radix sort is stable; input is in index order]
//   cut = V; if 0 < top_k < V: cut = min(cut, top_k)
//           if top_p < 1:      thresh = (total_all * rint(top_p * 2^24) + 2^24 - 1) >> 24   (total_all = sum of ALL w_i)
//                              cut = min(cut, smallest c with sum_{j<c} w_order[j] >= thresh)
//   kept(i) <=> rank(i) < cut <=> w_i > w* || (w_i == w* && i <= i*)   where (w*, i*) = order[cut-1]
//   total = sum of kept w_i; u = z mod total; out = smallest i with prefix_{kept j<=i} w_j > u   (index order)
// Everything after exp_f is exact integer arithmetic, so the parallel decomposition below is free to differ from
// the reference's sequential loops.
//
// Parallel structure for the top-k/top-p path (per row): the row is split into NB = 32 segments.
//   k_rowmax (NB x M blocks)  segment maxima           -> maxpart[M][NB]
//   k_weights (NB x M blocks) keys = (M-1-row)<<33 | w_i, vals = i  (row id above the 33 weight bits)
//   cub::DeviceRadixSort::SortPairsDescending over all M*V pairs on bits [0, 33 + log2 M): row 0 first, then w desc,
//     then (stable) index asc
//   top-p: k_part (segment sums of sorted w) -> k_cut (scan 32 partials, walk the one segment holding the crossing)
//   k_part (segment sums of kept w in index order) -> k_select (scan 32 partials, u = z mod total, walk one segment)
// Inside the "walk one segment" step each warp owns a contiguous chunk (lanes strided by 32, coalesced), the 32 warp
// sums are scanned, and only the warp whose chunk contains the crossing walks it with warp-level inclusive scans.
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cub/cub.cuh>
#include "common.cuh"
#include "det_cuda.h"
using namespace det;

namespace {

typedef unsigned long long u64;
constexpr int NT = 1024;                              // threads per block (32 warps)
constexpr int NB = 32;                                // segments per row for the parallel passes
constexpr u64 W_MASK = (1ull << 33) - 1;              // w_i <= 2^32 fits in 33 bits; row id lives above bit 33

__device__ __forceinline__ u64 warp_incl_scan(u64 v) {
    const int lane = threadIdx.x & 31;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        u64 y = __shfl_up_sync(0xffffffffu, v, o);
        if (lane >= o) v += y;
    }
    return v;
}
__device__ __forceinline__ u64 warp_sum_u64(u64 v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// Segment b of a row: [b*seg, min(V, (b+1)*seg)), seg a multiple of 32.
__device__ __forceinline__ int seg_len(int V) { return (((V + NB - 1) / NB) + 31) & ~31; }
// Within a block: warp w owns [w*cw, min(n, (w+1)*cw)), cw a multiple of 32.
__device__ __forceinline__ int chunk_width(int n) { return ((n + 31) / 32 + 31) & ~31; }

// Per-warp sums of f(i), i in [0, n), exclusive-scanned into s_warp[0..31]; s_warp[32] = total.  Returns total.
template <class F>
__device__ __forceinline__ u64 block_chunk_total(F f, int n, int cw, u64* s_warp /*[33]*/) {
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    const int b = w * cw, e = min(n, b + cw);
    u64 acc = 0;
    for (int i = b + lane; i < e; i += 32) acc += f(i);
    acc = warp_sum_u64(acc);
    __syncthreads();                                   // previous readers of s_warp are done
    if (lane == 0) s_warp[w] = acc;
    __syncthreads();
    if (w == 0) {
        u64 v = s_warp[lane], inc = warp_incl_scan(v);
        s_warp[lane] = inc - v;
        if (lane == 31) s_warp[32] = inc;
    }
    __syncthreads();
    return s_warp[32];
}

// Smallest j in [0, n) whose inclusive prefix sum of f exceeds u (requires u < total).  Uniform result.
template <class F>
__device__ __forceinline__ int block_chunk_search(F f, int n, int cw, const u64* s_warp, u64 u, int* s_res) {
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    const int b = w * cw, e = min(n, b + cw);
    u64 pre = s_warp[w];
    const u64 end = s_warp[w + 1];                     // s_warp[32] == total for the last warp
    if (pre <= u && u < end) {                         // exactly one warp: its chunk contains the crossing
        for (int i = b; i < e; i += 32) {
            const int j = i + lane;
            u64 v = j < e ? f(j) : 0ull;
            u64 inc = pre + warp_incl_scan(v);
            unsigned m = __ballot_sync(0xffffffffu, inc > u);
            if (m) { if (lane == 0) *s_res = i + __ffs(m) - 1; break; }
            pre = __shfl_sync(0xffffffffu, inc, 31);   // inc already includes pre
        }
    }
    __syncthreads();
    return *s_res;
}

// Block sum of f(i) over [b, e) (all 1024 threads strided).  Result valid in warp 0.
template <class F>
__device__ __forceinline__ u64 block_sum_u64(F f, int b, int e, u64* s_warp /*[32]*/) {
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    u64 acc = 0;
    for (int i = b + threadIdx.x; i < e; i += NT) acc += f(i);
    acc = warp_sum_u64(acc);
    if (lane == 0) s_warp[w] = acc;
    __syncthreads();
    return w == 0 ? warp_sum_u64(s_warp[lane]) : 0ull;
}

// Exclusive scan of the NB segment sums of a row into s_E[0..NB]; returns the row total.
__device__ __forceinline__ u64 row_scan_parts(const u64* part, u64* s_E /*[NB+1]*/) {
    if (threadIdx.x < 32) {
        u64 v = part[threadIdx.x], inc = warp_incl_scan(v);
        s_E[threadIdx.x] = inc - v;
        if (threadIdx.x == 31) s_E[NB] = inc;
    }
    __syncthreads();
    return s_E[NB];
}

// Smallest global j in [0, V) whose inclusive prefix of f exceeds u (u < row total), using the segment scan s_E.
template <class F>
__device__ __forceinline__ int row_search(F f, int V, const u64* s_E, u64 u, u64* s_warp, int* s_res) {
    int bs = 0;
    for (int b = 1; b < NB; b++) if (s_E[b] <= u) bs = b;      // last b with E[b] <= u  <=>  E[b] <= u < E[b+1]
    const int seg = seg_len(V), base = bs * seg, n = min(V, base + seg) - base;
    auto g = [&](int i) { return f(base + i); };
    const int cw = chunk_width(n);
    block_chunk_total(g, n, cw, s_warp);
    return base + block_chunk_search(g, n, cw, s_warp, u - s_E[bs], s_res);
}

__device__ __forceinline__ u64 weight(float lg, float inv_temp, float m) {
    return __float2ull_rn(mul(exp_f(sub(mul(lg, inv_temp), m)), 4294967296.0f));
}

__device__ __forceinline__ int row_cut(int V, int top_k, int topp_on, const int* cut_p, int row) {
    int cut = (top_k > 0 && top_k < V) ? top_k : V;
    if (topp_on) cut = min(cut, cut_p[row]);
    return cut;
}

// ---- temperature-only multinomial (no sort, no scratch): one block per row
__global__ void __launch_bounds__(NT) k_sample(const float* __restrict__ logits, int V, float inv_temp,
                                               const u64* __restrict__ z, int* __restrict__ out) {
    __shared__ float smax[32];
    __shared__ u64 s_warp[33];
    __shared__ int s_res;
    const int row = blockIdx.x;
    const float* lg = logits + (size_t)row * V;
    float m = -INFINITY;
    for (int i = threadIdx.x; i < V; i += NT) m = fmaxf(m, mul(lg[i], inv_temp));
    m = block_max<32>(m, smax);
    const int cw = chunk_width(V);
    auto f = [&](int i) { return weight(lg[i], inv_temp, m); };
    const u64 total = block_chunk_total(f, V, cw, s_warp);
    const int sel = block_chunk_search(f, V, cw, s_warp, z[row] % total, &s_res);
    if (threadIdx.x == 0) out[row] = sel;
}

// ---- top-k / top-p path
__global__ void __launch_bounds__(NT) k_rowmax(const float* __restrict__ logits, int V, float inv_temp, float* __restrict__ maxpart) {
    __shared__ float smax[32];
    const int row = blockIdx.y, b = blockIdx.x, seg = seg_len(V), base = b * seg, e = min(V, base + seg);
    const float* lg = logits + (size_t)row * V;
    float m = -INFINITY;
    for (int i = base + threadIdx.x; i < e; i += NT) m = fmaxf(m, mul(lg[i], inv_temp));
    m = block_max<32>(m, smax);
    if (threadIdx.x == 0) maxpart[row * NB + b] = m;          // -inf for an empty segment (never segment 0)
}

__global__ void __launch_bounds__(NT) k_weights(const float* __restrict__ logits, int M, int V, float inv_temp,
                                                const float* __restrict__ maxpart, u64* __restrict__ keys, int* __restrict__ vals) {
    const int row = blockIdx.y, b = blockIdx.x, seg = seg_len(V), base = b * seg, e = min(V, base + seg);
    const float m = warp_max(maxpart[row * NB + (threadIdx.x & 31)]);
    const float* lg = logits + (size_t)row * V;
    keys += (size_t)row * V; vals += (size_t)row * V;
    const u64 rowkey = (u64)(M - 1 - row) << 33;              // descending sort => row 0 first
    for (int i = base + threadIdx.x; i < e; i += NT) { keys[i] = rowkey | weight(lg[i], inv_temp, m); vals[i] = i; }
}

// Segment sums.  MODE 0: sorted w (for the top-p cut).  MODE 1: kept w in index order (for the selection).
template <int MODE>
__global__ void __launch_bounds__(NT) k_part(const u64* __restrict__ keys_in, const u64* __restrict__ keys_sorted,
                                             const int* __restrict__ idx_sorted, int V, int top_k, int topp_on,
                                             const int* __restrict__ cut_p, u64* __restrict__ part) {
    __shared__ u64 s_warp[32];
    const int row = blockIdx.y, b = blockIdx.x, seg = seg_len(V), base = b * seg, e = min(V, base + seg);
    u64 s;
    if (MODE == 0) {
        const u64* ks = keys_sorted + (size_t)row * V;
        s = block_sum_u64([&](int j) { return ks[j] & W_MASK; }, base, e, s_warp);
    } else {
        const int cut = row_cut(V, top_k, topp_on, cut_p, row);
        const u64 wstar = keys_sorted[(size_t)row * V + cut - 1] & W_MASK;
        const int istar = idx_sorted[(size_t)row * V + cut - 1];
        const u64* kin = keys_in + (size_t)row * V;
        s = block_sum_u64([&](int i) { const u64 w = kin[i] & W_MASK; return (w > wstar || (w == wstar && i <= istar)) ? w : 0ull; },
                          base, e, s_warp);
    }
    if (threadIdx.x == 0) part[row * NB + b] = s;
}

// top-p cut per row: smallest c with sum_{j<c} w_sorted[j] >= thresh
__global__ void __launch_bounds__(NT) k_cut(const u64* __restrict__ keys_sorted, int V, u64 p24, const u64* __restrict__ part,
                                            int* __restrict__ cut_p) {
    __shared__ u64 s_E[NB + 1], s_warp[33];
    __shared__ int s_res;
    const int row = blockIdx.x;
    const u64* ks = keys_sorted + (size_t)row * V;
    const u64 total_all = row_scan_parts(part + row * NB, s_E);
    // thresh = (total_all * p24 + 2^24 - 1) >> 24 in 128-bit arithmetic (total_all < 2^50 for V < 2^18, p24 <= 2^24)
    const u64 lo = total_all * p24; u64 hi = __umul64hi(total_all, p24);
    const u64 lo2 = lo + ((1ull << 24) - 1); hi += (lo2 < lo);
    const u64 thresh = (hi << 40) | (lo2 >> 24);              // <= total_all since p24 <= 2^24
    int cut = 1;
    if (thresh > 0) cut = row_search([&](int j) { return ks[j] & W_MASK; }, V, s_E, thresh - 1, s_warp, &s_res) + 1;
    if (threadIdx.x == 0) cut_p[row] = cut;
}

__global__ void __launch_bounds__(NT) k_select(const u64* __restrict__ keys_in, const u64* __restrict__ keys_sorted,
                                               const int* __restrict__ idx_sorted, int V, int top_k, int topp_on,
                                               const int* __restrict__ cut_p, const u64* __restrict__ part,
                                               const u64* __restrict__ z, int* __restrict__ out) {
    __shared__ u64 s_E[NB + 1], s_warp[33];
    __shared__ int s_res;
    const int row = blockIdx.x;
    const int cut = row_cut(V, top_k, topp_on, cut_p, row);
    const u64 wstar = keys_sorted[(size_t)row * V + cut - 1] & W_MASK;
    const int istar = idx_sorted[(size_t)row * V + cut - 1];
    const u64* kin = keys_in + (size_t)row * V;
    auto fk = [&](int i) { const u64 w = kin[i] & W_MASK; return (w > wstar || (w == wstar && i <= istar)) ? w : 0ull; };
    const u64 total = row_scan_parts(part + row * NB, s_E);   // > 0: the argmax (w = 2^32) is always kept
    const int sel = row_search(fk, V, s_E, z[row] % total, s_warp, &s_res);
    if (threadIdx.x == 0) out[row] = sel;
}

inline size_t align256(size_t x) { return (x + 255) & ~(size_t)255; }
inline int row_bits(int M) { int b = 0; while ((1 << b) < M) b++; return b; }
inline size_t cub_temp_bytes(int M, int V) {
    size_t tmp = 0;
    cub::DeviceRadixSort::SortPairsDescending(nullptr, tmp, (const u64*)nullptr, (u64*)nullptr, (const int*)nullptr, (int*)nullptr,
                                              M * V, 0, 33 + row_bits(M));
    return tmp;
}
struct Scratch { u64 *keys_in, *keys_out, *part; int *vals_in, *vals_out, *cut_p; float* maxpart; void* tmp; size_t tmp_bytes, total; };
inline Scratch carve(void* base, int M, int V) {
    const size_t n = (size_t)M * V; char* p = (char*)base; Scratch s;
    s.keys_in = (u64*)p;  p += align256(n * 8);
    s.keys_out = (u64*)p; p += align256(n * 8);
    s.vals_in = (int*)p;  p += align256(n * 4);
    s.vals_out = (int*)p; p += align256(n * 4);
    s.part = (u64*)p;     p += align256((size_t)M * NB * 8);
    s.maxpart = (float*)p; p += align256((size_t)M * NB * 4);
    s.cut_p = (int*)p;    p += align256((size_t)M * 4);
    s.tmp = p; s.tmp_bytes = cub_temp_bytes(M, V); p += align256(s.tmp_bytes);
    s.total = (size_t)(p - (char*)base);
    return s;
}

} // namespace

extern "C" void det_sample(const float* logits, int M, int V, float inv_temp, const unsigned long long* z, int* out, cudaStream_t st) {
    k_sample<<<M, NT, 0, st>>>(logits, V, inv_temp, z, out);
}

extern "C" size_t det_sample_scratch_bytes(int M, int V) { return carve(nullptr, M, V).total; }

extern "C" void det_sample_topkp(const float* logits, int M, int V, float inv_temp, int top_k, float top_p,
                                 const unsigned long long* z, int* out, void* scratch, cudaStream_t st) {
    const int topk_on = top_k > 0 && top_k < V, topp_on = top_p < 1.0f;
    if (!topk_on && !topp_on) { k_sample<<<M, NT, 0, st>>>(logits, V, inv_temp, z, out); return; }
    if ((size_t)M * V > 0x7fffffffULL) { fprintf(stderr, "det_sample_topkp: M*V too large\n"); abort(); }
    Scratch s = carve(scratch, M, V);
    // p24 = round-half-even(top_p * 2^24): python's round(top_p * 2**24) on the fp32 top_p; top_p <= 0 behaves as 0
    const u64 p24 = top_p <= 0.0f ? 0ull : (u64)llrint((double)top_p * 16777216.0);
    const dim3 grid(NB, M);
    k_rowmax<<<grid, NT, 0, st>>>(logits, V, inv_temp, s.maxpart);
    k_weights<<<grid, NT, 0, st>>>(logits, M, V, inv_temp, s.maxpart, s.keys_in, s.vals_in);
    DET_CHECK(cub::DeviceRadixSort::SortPairsDescending(s.tmp, s.tmp_bytes, s.keys_in, s.keys_out, s.vals_in, s.vals_out,
                                                        M * V, 0, 33 + row_bits(M), st));
    if (topp_on) {
        k_part<0><<<grid, NT, 0, st>>>(s.keys_in, s.keys_out, s.vals_out, V, top_k, topp_on, s.cut_p, s.part);
        k_cut<<<M, NT, 0, st>>>(s.keys_out, V, p24, s.part, s.cut_p);
    }
    k_part<1><<<grid, NT, 0, st>>>(s.keys_in, s.keys_out, s.vals_out, V, top_k, topp_on, s.cut_p, s.part);
    k_select<<<M, NT, 0, st>>>(s.keys_in, s.keys_out, s.vals_out, V, top_k, topp_on, s.cut_p, s.part, z, out);
}
