// det: quantized GEMM
//   acc[t][n] = sum_g ( m[n][g] * mx[t][g] ) * sum_{k in group g} wq[n][k] * xq[t][k]   (exact integer arithmetic)
//   y[t][n]   = ((float)acc[t][n] * c[n]) * sx[t]                                        (int64->fp32 RNE, 2 fp32 muls)
// Bounds: |group partial| < 2^21 (int32), m*mx < 2^32 (uint32), |acc| < 2^61 (int64).  mx == NULL means all ones.
//
// Kernels (dispatch on M in det_gemm):
//   k_decode  (M <= 8): memory-bound matvec. Every lane owns a FIXED 32-wide k-range (its activations stay in
//             registers for the whole kernel), a block of S = ceil(K/1024) warps covers the full K of one row,
//             and the block loops over rows (two rows in flight).  Weights are streamed with 8/16-byte loads and
//             unpacked in registers; the inner group sums use dp4a (int32), widened to int64 before the multiplier.
//   k_prefill (8 < M <= 32): tiled CUDA-core dp4a GEMM, one 128-wide k-group per smem stage, weights unpacked
//             once into shared memory (int8) and reused across all BM rows of the tile.
//   k_mma     (32 < M <= 64): int8 tensor cores (mma.sync m16n8k32, exact), 64x64 block tile, cp.async 3-stage
//             pipeline, weights unpacked packed-smem -> int8-smem once per group, ldmatrix fragments.
//   k_mma2    (M > 64): int8 tensor cores, 128x128 block tile, B fragments built per lane from the packed smem tile
//             (W4/W6 as unsigned codes with a per-(token,group) sum correction from the k_xsum pre-kernel).
//   In every kernel the int32 group partial is scaled by m[n][g]*mx[t][g] (uint32) into int64 accumulators.
// All integer sums may be done in any order.  The fp32 multiplies use det::mul (no FMA).  Every kernel's epilogue goes
// through Out::store: plain y, y += result, or the columns routed into up to 4 part buffers (det_gemm_ex).
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "common.cuh"
#include "det_cuda.h"

namespace {

// ------------------------------------------------------------------ packed format helpers
// Packing per 32 weights (spec §2):  W8: 32 int8.  W4: 16 bytes, byte i = u[i] | u[i+16]<<4 (u = wq+8).
// W6: 16 bytes of low nibbles as W4 (u = wq+32) + 8 bytes: byte i = hi[i] | hi[i+8]<<2 | hi[i+16]<<4 | hi[i+24]<<6
// where hi = u >> 4.
template <int BITS> struct Fmt;
template <> struct Fmt<8> { static constexpr int BPL = 32; static constexpr int OFF = 0;  static constexpr int PSTR = 144; static constexpr int PSTR2 = 128; };
template <> struct Fmt<6> { static constexpr int BPL = 24; static constexpr int OFF = 32; static constexpr int PSTR = 96;  static constexpr int PSTR2 = 112; };
template <> struct Fmt<4> { static constexpr int BPL = 16; static constexpr int OFF = 8;  static constexpr int PSTR = 64;  static constexpr int PSTR2 = 80;  };
// BPL: bytes per 32 weights.  OFF: code offset (u = wq + OFF).  PSTR / PSTR2: smem row strides (bytes) of a packed
// 128-group in k_mma (unpack pass, conflict-free 24/32/16-byte row reads) and k_mma2 (per-lane 4-byte fragment reads;
// W8 rows are XOR-swizzled by 16-byte chunk instead of padded)

// Where the epilogue writes element (t, n): y[M][N], or part i's buffer for n in [offs[i], offs[i+1]) (offs[nparts] == N,
// dst[i] is [M][offs[i+1] - offs[i]]); acc != 0 adds to the existing value.
struct Out {
    float* y;
    int acc, nparts;
    int offs[5];
    float* dst[4];
    __device__ __forceinline__ void store(int N, int t, int n, float v) const {
        float* p;
        if (nparts <= 1) p = y + (size_t)t * N + n;
        else {
            int i = 0;
#pragma unroll
            for (int j = 1; j < 4; j++) if (j < nparts && n >= offs[j]) i = j;
            p = dst[i] + (size_t)t * (offs[i + 1] - offs[i]) + (n - offs[i]);
        }
        *p = acc ? det::add(*p, v) : v;
    }
};

// dot of 4 unsigned bytes with 4 signed bytes (+c)
__device__ __forceinline__ int dp4a_us(unsigned a, int b, int c) {
    int d;
    asm("dp4a.u32.s32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
    return d;
}
__device__ __forceinline__ int dp4a_ss(int a, int b, int c) { return __dp4a(a, b, c); }

// (int64) p * s  for int32 p and uint32 s  (one 32x32->64 unsigned multiply + sign fix)
__device__ __forceinline__ long long mul_s32_u32(int p, uint32_t s) {
    unsigned long long r = (unsigned long long)(uint32_t)p * (unsigned long long)s;
    r -= (unsigned long long)(s & (uint32_t)(p >> 31)) << 32;
    return (long long)r;
}

// acc += (int64) p * s  for int32 p and uint32 s:  mad.wide.u32 on the unsigned view of p, then subtract s<<32 if p<0
// (asm volatile so that the compiler keeps it interleaved with the mma.sync stream in k_mma2)
__device__ __forceinline__ void mad_s32_u32(long long& acc, int p, uint32_t s) {
    unsigned long long a = (unsigned long long)acc;
    asm volatile("{\n\t.reg .u32 lo, hi, t;\n\t"
                 "mad.wide.u32 %0, %1, %2, %0;\n\t"
                 "shr.s32 t, %1, 31;\n\t"                   // p<0 ? -1 : 0
                 "mov.b64 {lo, hi}, %0;\n\t"
                 "mad.lo.u32 hi, t, %2, hi;\n\t"             // hi -= s  when p<0
                 "mov.b64 %0, {lo, hi};\n\t}"
                 : "+l"(a) : "r"((uint32_t)p), "r"(s));
    acc = (long long)a;
}

// Load the BPL packed bytes of one 32-weight block into words q[BPL/4] (p is 8-byte aligned for W6, 16 for others)
template <int BITS> __device__ __forceinline__ void load_block(const uint8_t* p, uint32_t* q);
template <> __device__ __forceinline__ void load_block<8>(const uint8_t* p, uint32_t* q) {
    uint4 a = __ldg(reinterpret_cast<const uint4*>(p)), b = __ldg(reinterpret_cast<const uint4*>(p) + 1);
    q[0] = a.x; q[1] = a.y; q[2] = a.z; q[3] = a.w; q[4] = b.x; q[5] = b.y; q[6] = b.z; q[7] = b.w;
}
template <> __device__ __forceinline__ void load_block<6>(const uint8_t* p, uint32_t* q) {
    const uint2* p2 = reinterpret_cast<const uint2*>(p);
    uint2 a = __ldg(p2), b = __ldg(p2 + 1), c = __ldg(p2 + 2);
    q[0] = a.x; q[1] = a.y; q[2] = b.x; q[3] = b.y; q[4] = c.x; q[5] = c.y;
}
template <> __device__ __forceinline__ void load_block<4>(const uint8_t* p, uint32_t* q) {
    uint4 a = __ldg(reinterpret_cast<const uint4*>(p));
    q[0] = a.x; q[1] = a.y; q[2] = a.z; q[3] = a.w;
}
// same from shared memory
template <int BITS> __device__ __forceinline__ void lds_block(const uint8_t* p, uint32_t* q);
template <> __device__ __forceinline__ void lds_block<8>(const uint8_t* p, uint32_t* q) {
    uint4 a = reinterpret_cast<const uint4*>(p)[0], b = reinterpret_cast<const uint4*>(p)[1];
    q[0] = a.x; q[1] = a.y; q[2] = a.z; q[3] = a.w; q[4] = b.x; q[5] = b.y; q[6] = b.z; q[7] = b.w;
}
template <> __device__ __forceinline__ void lds_block<6>(const uint8_t* p, uint32_t* q) {
    const uint2* p2 = reinterpret_cast<const uint2*>(p);
    uint2 a = p2[0], b = p2[1], c = p2[2];
    q[0] = a.x; q[1] = a.y; q[2] = b.x; q[3] = b.y; q[4] = c.x; q[5] = c.y;
}
template <> __device__ __forceinline__ void lds_block<4>(const uint8_t* p, uint32_t* q) {
    uint4 a = reinterpret_cast<const uint4*>(p)[0];
    q[0] = a.x; q[1] = a.y; q[2] = a.z; q[3] = a.w;
}

// Unpack 32 codes into 8 words of 4 bytes each (consecutive k).  For W4/W6 the bytes are the UNSIGNED codes
// u = wq + OFF; for W8 they are the signed codes.
template <int BITS> __device__ __forceinline__ void unpack32(const uint32_t* q, uint32_t* u);
template <> __device__ __forceinline__ void unpack32<8>(const uint32_t* q, uint32_t* u) {
#pragma unroll
    for (int j = 0; j < 8; j++) u[j] = q[j];
}
template <> __device__ __forceinline__ void unpack32<4>(const uint32_t* q, uint32_t* u) {
#pragma unroll
    for (int j = 0; j < 4; j++) { u[j] = q[j] & 0x0F0F0F0Fu; u[j + 4] = (q[j] >> 4) & 0x0F0F0F0Fu; }
}
template <> __device__ __forceinline__ void unpack32<6>(const uint32_t* q, uint32_t* u) {
    const uint32_t LM = 0x0F0F0F0Fu, HM = 0x30303030u;
    u[0] = (q[0] & LM) | ((q[4] << 4) & HM);
    u[1] = (q[1] & LM) | ((q[5] << 4) & HM);
    u[2] = (q[2] & LM) | ((q[4] << 2) & HM);
    u[3] = (q[3] & LM) | ((q[5] << 2) & HM);
    u[4] = ((q[0] >> 4) & LM) | (q[4] & HM);
    u[5] = ((q[1] >> 4) & LM) | (q[5] & HM);
    u[6] = ((q[2] >> 4) & LM) | ((q[4] >> 2) & HM);
    u[7] = ((q[3] >> 4) & LM) | ((q[5] >> 2) & HM);
}
// unsigned code word -> signed code word (per byte: w = u - OFF, done as (u + 128 - OFF) ^ 0x80, no carries)
template <int BITS> __device__ __forceinline__ uint32_t to_signed(uint32_t u) {
    if (BITS == 8) return u;
    const uint32_t ADD = (uint32_t)(128 - Fmt<BITS>::OFF) * 0x01010101u;
    return (u + ADD) ^ 0x80808080u;
}
// dot of a code word (unsigned for W4/W6, signed for W8) with a signed activation word
template <int BITS> __device__ __forceinline__ int dp4a_w(uint32_t u, int x, int c) {
    return BITS == 8 ? dp4a_ss((int)u, x, c) : dp4a_us(u, x, c);
}

// ------------------------------------------------------------------ PTX wrappers (cp.async / ldmatrix / mma)
__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src, bool pred) {
    const int n = pred ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(dst), "l"(src), "r"(n));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N)); }
__device__ __forceinline__ void ldmatrix_x4(uint32_t addr, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
// A signed int8, B unsigned uint8 (codes u = wq + OFF): acc = sum wq*xq + OFF * sum xq
__device__ __forceinline__ void mma_s8u8(int* c, const uint32_t* a, const uint32_t* b) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.u8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ void mma_s8u8_cd(int* d, const uint32_t* a, const uint32_t* b, const int* c) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.u8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                 : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(c[0]), "r"(c[1]), "r"(c[2]), "r"(c[3]));
}
__device__ __forceinline__ void mma_s8_cd(int* d, const uint32_t* a, const uint32_t* b, const int* c) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                 : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(c[0]), "r"(c[1]), "r"(c[2]), "r"(c[3]));
}
__device__ __forceinline__ void mma_s8(int* c, const uint32_t* a, const uint32_t* b) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// ------------------------------------------------------------------ decode kernel (M <= 8)
// MP: padded token count (1,2,4,8).  SMAX: max warps per block (register budget).  Block = S warps, S = ceil(K/1024);
// warp w, lane l owns k in [w*1024 + l*32, +32) (one 128-group g).  Rows n = blockIdx.x + i*gridDim.x.  Per row every
// lane computes p = sum_k wq*xq over its 32 k (int32, exact), then v = p * (m[n][g]*mx[t][g]) (int64); the warp
// reduces v over its lanes with a token-transposed butterfly, the S warp partials are combined in shared memory, and
// the row's y values are written once per batch of DEC_RB rows.
constexpr int DEC_RB = 8;

template <int MP, int BITS, int SMAX>
struct DecCfg {
    static constexpr bool XSM = (MP == 8 && SMAX == 17);   // tokens 4..7 of x live in (dynamic) shared memory:
    static constexpr int XR = XSM ? 4 : MP;                 // 17-warp blocks are capped at 96 registers/thread
};

template <int MP, int BITS, int SMAX>
__global__ void __launch_bounds__(SMAX * 32)
k_decode(const uint8_t* __restrict__ wq, const float* __restrict__ cc, const uint16_t* __restrict__ mm, int N, int K,
         const int8_t* __restrict__ xq, const float* __restrict__ sx, const uint16_t* __restrict__ mx, int M,
         const Out out) {
    constexpr int BPL = Fmt<BITS>::BPL, NQ = BPL / 4, OFF = Fmt<BITS>::OFF;
    constexpr bool XSM = DecCfg<MP, BITS, SMAX>::XSM;
    constexpr int XR = DecCfg<MP, BITS, SMAX>::XR;
    __shared__ long long red[2][DEC_RB][SMAX][MP];
    extern __shared__ int4 xsm[];                            // [MP-XR][S*32][2] int4 (only if XSM)

    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, S = blockDim.x >> 5;
    const int k0 = warp * 1024 + lane * 32;
    const bool valid = k0 < K;
    const int G = K >> 7;
    const size_t rowbytes = (size_t)K * BITS / 8;
    const uint8_t* wbase = wq + (size_t)warp * (1024 * BITS / 8) + lane * BPL;
    const uint16_t* mbase = mm + (k0 >> 7);

    // activations for this lane's k-range: tokens 0..XR-1 in registers (8 words each), the rest in shared memory;
    // OFF * sum(x) and mx per token in registers
    int xw[XR][8];
    int offx[MP];
    uint32_t mxr[MP];
#pragma unroll
    for (int t = 0; t < MP; t++) {
        int w[8];
        if (t < M && valid) {
            const int4* xp = reinterpret_cast<const int4*>(xq + (size_t)t * K + k0);
            int4 a = __ldg(xp), b = __ldg(xp + 1);
            w[0] = a.x; w[1] = a.y; w[2] = a.z; w[3] = a.w; w[4] = b.x; w[5] = b.y; w[6] = b.z; w[7] = b.w;
            mxr[t] = mx ? (uint32_t)__ldg(mx + (size_t)t * G + (k0 >> 7)) : 1u;
        } else {
#pragma unroll
            for (int j = 0; j < 8; j++) w[j] = 0;
            mxr[t] = 0;
        }
        int sm = 0;
#pragma unroll
        for (int j = 0; j < 8; j++) sm = dp4a_us(0x01010101u, w[j], sm);
        offx[t] = sm * OFF;
        if (t < XR) {
#pragma unroll
            for (int j = 0; j < 8; j++) xw[t < XR ? t : 0][j] = w[j];
        } else if (XSM) {
            int4* d = xsm + ((size_t)(t - XR) * (S * 32) + threadIdx.x) * 2;
            d[0] = make_int4(w[0], w[1], w[2], w[3]);
            d[1] = make_int4(w[4], w[5], w[6], w[7]);
        }
    }
    if (XSM) __syncthreads();

    if ((int)blockIdx.x >= N) return;                                   // block-uniform
    const int nrows = (N - 1 - (int)blockIdx.x) / (int)gridDim.x + 1;   // rows handled by this block

    auto load_row = [&](int i, uint32_t* q, uint32_t& mv) {
        const int n = blockIdx.x + i * gridDim.x;
        if (valid) {
            load_block<BITS>(wbase + (size_t)n * rowbytes, q);
            mv = __ldg(mbase + (size_t)n * G);
        } else {
#pragma unroll
            for (int j = 0; j < NQ; j++) q[j] = 0;
            mv = 0;
        }
    };
    auto compute_row = [&](const uint32_t* q, uint32_t mv, int i) {
        uint32_t u[8];
        unpack32<BITS>(q, u);
        long long v[MP];
#pragma unroll
        for (int t = 0; t < MP; t++) {
            int d = 0;
            if (t < XR) {
#pragma unroll
                for (int j = 0; j < 8; j++) d = dp4a_w<BITS>(u[j], xw[t < XR ? t : 0][j], d);
            } else {
                const int4* sp = xsm + ((size_t)(t - XR) * (S * 32) + threadIdx.x) * 2;
                const int4 a = sp[0], b = sp[1];
                d = dp4a_w<BITS>(u[0], a.x, d); d = dp4a_w<BITS>(u[1], a.y, d);
                d = dp4a_w<BITS>(u[2], a.z, d); d = dp4a_w<BITS>(u[3], a.w, d);
                d = dp4a_w<BITS>(u[4], b.x, d); d = dp4a_w<BITS>(u[5], b.y, d);
                d = dp4a_w<BITS>(u[6], b.z, d); d = dp4a_w<BITS>(u[7], b.w, d);
            }
            v[t] = mul_s32_u32(d - offx[t], mv * mxr[t]);
        }
        // token-transposed butterfly: after it, lane l (< MP) holds the warp total of token tok(l)
        int tok = 0;
#pragma unroll
        for (int d = 1, cnt = MP; d < 32; d <<= 1) {
            if (cnt > 1) {
                const int h = cnt >> 1;
                const bool up = (lane & d) != 0;
#pragma unroll
                for (int j = 0; j < h; j++) {
                    long long send = up ? v[j] : v[j + h];
                    long long keep = up ? v[j + h] : v[j];
                    v[j] = keep + __shfl_xor_sync(0xffffffffu, send, d);
                }
                if (up) tok += h;
                cnt = h;
            } else {
                v[0] += __shfl_xor_sync(0xffffffffu, v[0], d);
            }
        }
        const int phase = i / DEC_RB, r = i % DEC_RB;
        if (lane < MP && tok < M) red[phase & 1][r][warp][tok] = v[0];

        if (r == DEC_RB - 1 || i == nrows - 1) {
            __syncthreads();
            const int cnt_rows = r + 1;
            for (int idx = threadIdx.x; idx < cnt_rows * M; idx += blockDim.x) {
                const int rr = idx / M, t = idx - rr * M;
                const int n = blockIdx.x + (phase * DEC_RB + rr) * gridDim.x;
                long long acc = 0;
                for (int s = 0; s < S; s++) acc += red[phase & 1][rr][s][t];
                out.store(N, t, n, det::mul(det::mul(det::i64_to_f(acc), __ldg(cc + n)), __ldg(sx + t)));
            }
            // no second sync needed: the next phase writes the other buffer, and the phase after that is
            // separated from this read by the next phase's __syncthreads
        }
    };

    // rows are streamed two ahead with two register buffers (A holds even rows, B odd rows)
    uint32_t qA[NQ], qB[NQ];
    uint32_t mA = 0, mB = 0;
    load_row(0, qA, mA);
    if (nrows > 1) load_row(1, qB, mB);
    int i = 0;
    for (; i + 1 < nrows; i += 2) {
        compute_row(qA, mA, i);
        if (i + 2 < nrows) load_row(i + 2, qA, mA);
        compute_row(qB, mB, i + 1);
        if (i + 3 < nrows) load_row(i + 3, qB, mB);
    }
    if (i < nrows) compute_row(qA, mA, i);
}

// ------------------------------------------------------------------ prefill kernel (8 < M <= 32), CUDA-core dp4a
// Block tile BM x 64 (BM = 16*TM), 256 threads as 16 (ty, rows) x 16 (tx, cols), thread tile TM x 4.
// One 128-wide k-group per stage: As[kw][t] (k-word major, padded), Bs[kw][n] (swizzled), Ms[n], Mx[t].
template <int BITS, int TM>
__global__ void __launch_bounds__(256)
k_prefill(const uint8_t* __restrict__ wq, const float* __restrict__ cc, const uint16_t* __restrict__ mm, int N, int K,
          const int8_t* __restrict__ xq, const float* __restrict__ sx, const uint16_t* __restrict__ mx, int M,
          const Out out) {
    constexpr int BM = 16 * TM, BN = 64, TN = 4;
    constexpr int ASTR = BM + 4;                       // A row stride in words (conflict-free transposed stores)
    constexpr int BPL = Fmt<BITS>::BPL, NQ = BPL / 4;
    constexpr int ACH = BM * 8;                        // 16-byte chunks of A per stage
    constexpr int AR = (ACH + 255) / 256;              // chunks per thread
    __shared__ __align__(16) int As[2][32][ASTR];
    __shared__ __align__(16) int Bs[2][32][BN];
    __shared__ uint16_t Ms[2][BN];
    __shared__ uint16_t Mxs[2][BM];

    const int tid = threadIdx.x, tx = tid & 15, ty = tid >> 4;
    const int m0 = blockIdx.x * BM, n0 = blockIdx.y * BN;
    const int G = K >> 7;
    const size_t rowbytes = (size_t)K * BITS / 8;

    // staging assignments
    //   A: chunk e -> t = (q % (BM/16))*16 + (lo & 15), j = (q / (BM/16))*2 + (lo >> 4)  with q = e>>5, lo = e&31
    //      (16 tokens x 2 chunks per warp instruction; stores As[4j+i][t] are bank-conflict free with ASTR = BM+4)
    //   B: thread -> s = tid & 3 (32-block within the group), n = tid >> 2   (8 rows x 96/128/64 B per warp instr)
    const int bs = tid & 3, bn = tid >> 2;
    int4 areg[AR];
    uint32_t breg[NQ];
    uint16_t mreg = 0, mxreg = 1;

    auto gload = [&](int g) {
#pragma unroll
        for (int r = 0; r < AR; r++) {
            const int e = tid + 256 * r;
            if (e < ACH) {
                const int q = e >> 5, lo = e & 31;
                const int t = (q % (BM / 16)) * 16 + (lo & 15), j = (q / (BM / 16)) * 2 + (lo >> 4);
                if (m0 + t < M) areg[r] = __ldg(reinterpret_cast<const int4*>(xq + (size_t)(m0 + t) * K + (size_t)g * 128 + j * 16));
                else areg[r] = make_int4(0, 0, 0, 0);
            }
        }
        if (n0 + bn < N) load_block<BITS>(wq + (size_t)(n0 + bn) * rowbytes + (size_t)g * (128 * BITS / 8) + bs * BPL, breg);
        else {
#pragma unroll
            for (int j = 0; j < NQ; j++) breg[j] = 0;
        }
        if (tid < BN) mreg = (n0 + tid < N) ? __ldg(mm + (size_t)(n0 + tid) * G + g) : (uint16_t)0;
        if (tid < BM) mxreg = (m0 + tid < M) ? (mx ? __ldg(mx + (size_t)(m0 + tid) * G + g) : (uint16_t)1) : (uint16_t)0;
    };
    auto sstore = [&](int buf) {
#pragma unroll
        for (int r = 0; r < AR; r++) {
            const int e = tid + 256 * r;
            if (e < ACH) {
                const int q = e >> 5, lo = e & 31;
                const int t = (q % (BM / 16)) * 16 + (lo & 15), j = (q / (BM / 16)) * 2 + (lo >> 4);
                As[buf][4 * j + 0][t] = areg[r].x;
                As[buf][4 * j + 1][t] = areg[r].y;
                As[buf][4 * j + 2][t] = areg[r].z;
                As[buf][4 * j + 3][t] = areg[r].w;
            }
        }
        uint32_t u[8];
        unpack32<BITS>(breg, u);
#pragma unroll
        for (int j = 0; j < 8; j++) {
            const int kw = 8 * bs + j;
            Bs[buf][kw][bn ^ (kw & 0x18)] = (int)to_signed<BITS>(u[j]);
        }
        if (tid < BN) Ms[buf][tid] = mreg;
        if (tid < BM) Mxs[buf][tid] = mxreg;
    };

    long long acc64[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; i++)
#pragma unroll
        for (int j = 0; j < TN; j++) acc64[i][j] = 0;

    gload(0);
    sstore(0);
    __syncthreads();
    for (int g = 0; g < G; g++) {
        const int buf = g & 1;
        if (g + 1 < G) gload(g + 1);
        int acc[TM][TN];
#pragma unroll
        for (int i = 0; i < TM; i++)
#pragma unroll
            for (int j = 0; j < TN; j++) acc[i][j] = 0;
#pragma unroll 8
        for (int kw = 0; kw < 32; kw++) {
            int a[TM];
            if (TM == 4) { int4 v = *reinterpret_cast<const int4*>(&As[buf][kw][ty * 4]); a[0] = v.x; a[1] = v.y; a[2] = v.z; a[3] = v.w; }
            else if (TM == 2) { int2 v = *reinterpret_cast<const int2*>(&As[buf][kw][ty * 2]); a[0] = v.x; a[1] = v.y; }
            else { a[0] = As[buf][kw][ty]; }
            const int4 bv = *reinterpret_cast<const int4*>(&Bs[buf][kw][(tx * 4) ^ (kw & 0x18)]);
            const int b[4] = {bv.x, bv.y, bv.z, bv.w};
#pragma unroll
            for (int i = 0; i < TM; i++)
#pragma unroll
                for (int j = 0; j < TN; j++) acc[i][j] = dp4a_ss(a[i], b[j], acc[i][j]);
        }
        const uint2 mv2 = *reinterpret_cast<const uint2*>(&Ms[buf][tx * 4]);
        const uint32_t mv[4] = {mv2.x & 0xFFFFu, mv2.x >> 16, mv2.y & 0xFFFFu, mv2.y >> 16};
        uint32_t mxv[TM];
#pragma unroll
        for (int i = 0; i < TM; i++) mxv[i] = Mxs[buf][ty * TM + i];
#pragma unroll
        for (int i = 0; i < TM; i++)
#pragma unroll
            for (int j = 0; j < TN; j++) acc64[i][j] += mul_s32_u32(acc[i][j], mv[j] * mxv[i]);
        if (g + 1 < G) sstore(buf ^ 1);
        __syncthreads();
    }

    // epilogue
    const int nb = n0 + tx * 4;
    float cv[4];
#pragma unroll
    for (int j = 0; j < TN; j++) cv[j] = (nb + j < N) ? __ldg(cc + nb + j) : 0.0f;
#pragma unroll
    for (int i = 0; i < TM; i++) {
        const int t = m0 + ty * TM + i;
        if (t >= M) continue;
        const float s = __ldg(sx + t);
        float o[4];
#pragma unroll
        for (int j = 0; j < TN; j++) o[j] = det::mul(det::mul(det::i64_to_f(acc64[i][j]), cv[j]), s);
        if (out.nparts <= 1 && !out.acc && nb + 4 <= N && (N & 3) == 0) *reinterpret_cast<float4*>(out.y + (size_t)t * N + nb) = make_float4(o[0], o[1], o[2], o[3]);
        else {
#pragma unroll
            for (int j = 0; j < TN; j++) if (nb + j < N) out.store(N, t, nb + j, o[j]);
        }
    }
}

// ------------------------------------------------------------------ prefill kernel (M > 32), int8 tensor cores
// Block tile BM x BN = (32*WM) x (32*WN), WM*WN warps, warp tile 32 x 32 (2 m16 x 4 n8 mma tiles).
// Dynamic smem: A[STAGES][BM][128] int8 (16B chunks XOR-swizzled by row&7), Bp[STAGES][BN][PSTR] packed weights,
// B8[BN][128] unpacked int8 (swizzled), Ms[STAGES][BN], Mxs[STAGES][BM] uint16.
// Per group g: wait(A,Bp of g) ; sync ; store delayed m/mx regs ; issue cp.async for g+STAGES-1 ; load m/mx regs for
// g+STAGES-1 ; unpack Bp[g] -> B8 ; sync ; 4 x (ldmatrix A, ldmatrix B, 8 mma) ; acc64 += acc32 * (m*mx).
template <int BITS, int WM, int WN, int STAGES>
struct MmaCfg {
    static constexpr int BM = 32 * WM, BN = 32 * WN, NT = 32 * WM * WN;
    static constexpr int PSTR = Fmt<BITS>::PSTR, BCH = BITS;            // packed 16B chunks per row per group
    static constexpr int A_BYTES = STAGES * BM * 128;
    static constexpr int BP_BYTES = STAGES * BN * PSTR;
    static constexpr int B8_BYTES = BN * 128;
    static constexpr int MS_BYTES = STAGES * BN * 2;
    static constexpr int MX_BYTES = STAGES * BM * 2;
    static constexpr int SMEM = A_BYTES + BP_BYTES + B8_BYTES + MS_BYTES + MX_BYTES;
};

template <int BITS, int WM, int WN, int STAGES>
__global__ void __launch_bounds__(32 * WM * WN, 1)
k_mma(const uint8_t* __restrict__ wq, const float* __restrict__ cc, const uint16_t* __restrict__ mm, int N, int K,
      const int8_t* __restrict__ xq, const float* __restrict__ sx, const uint16_t* __restrict__ mx, int M,
      const Out out) {
    using C = MmaCfg<BITS, WM, WN, STAGES>;
    constexpr int BM = C::BM, BN = C::BN, NT = C::NT, PSTR = C::PSTR, BCH = C::BCH, BPL = Fmt<BITS>::BPL;
    extern __shared__ __align__(128) uint8_t smem[];
    uint8_t* As = smem;
    uint8_t* Bp = smem + C::A_BYTES;
    uint8_t* B8 = Bp + C::BP_BYTES;
    uint16_t* Ms = reinterpret_cast<uint16_t*>(B8 + C::B8_BYTES);
    uint16_t* Mxs = Ms + STAGES * BN;

    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int wm = warp / WN, wn = warp % WN;
    const int gid = lane >> 2, tig = lane & 3;
    const int m0 = blockIdx.x * BM, n0 = blockIdx.y * BN;
    const int G = K >> 7;
    const size_t rowbytes = (size_t)K * BITS / 8;

    // ---- async staging of group g into stage s
    auto issue = [&](int g, int s) {
        uint8_t* as = As + (size_t)s * BM * 128;
        for (int e = tid; e < BM * 8; e += NT) {
            const int t = e >> 3, c = e & 7;
            const bool ok = (m0 + t) < M;
            const int8_t* src = ok ? xq + (size_t)(m0 + t) * K + (size_t)g * 128 + c * 16 : xq;
            cp_async16(smem_u32(as + t * 128 + ((c ^ (t & 7)) << 4)), src, ok);
        }
        uint8_t* bp = Bp + (size_t)s * BN * PSTR;
        for (int e = tid; e < BN * BCH; e += NT) {
            const int n = e / BCH, c = e - n * BCH;
            const bool ok = (n0 + n) < N;
            const uint8_t* src = ok ? wq + (size_t)(n0 + n) * rowbytes + (size_t)g * (128 * BITS / 8) + c * 16 : wq;
            cp_async16(smem_u32(bp + n * PSTR + c * 16), src, ok);
        }
    };
    uint16_t mreg = 0, mxreg = 0;
    auto load_mregs = [&](int g) {
        if (tid < BN) mreg = (n0 + tid < N) ? __ldg(mm + (size_t)(n0 + tid) * G + g) : (uint16_t)0;
        else if (tid < BN + BM) {
            const int t = tid - BN;
            mxreg = (m0 + t < M) ? (mx ? __ldg(mx + (size_t)(m0 + t) * G + g) : (uint16_t)1) : (uint16_t)0;
        }
    };
    auto store_mregs = [&](int s) {
        if (tid < BN) Ms[s * BN + tid] = mreg;
        else if (tid < BN + BM) Mxs[s * BM + tid - BN] = mxreg;
    };
    static_assert(BN + BM <= NT, "m/mx staging needs BN+BM threads");

    // ---- prologue
#pragma unroll
    for (int s = 0; s < STAGES - 1; s++) {
        if (s < G) issue(s, s);
        cp_async_commit();
    }
#pragma unroll
    for (int s = 0; s < STAGES - 2; s++) { if (s < G) { load_mregs(s); store_mregs(s); } }
    if (STAGES - 2 < G) load_mregs(STAGES - 2);

    int acc[2][4][4];
    long long acc64[2][4][4];
#pragma unroll
    for (int i = 0; i < 2; i++)
#pragma unroll
        for (int j = 0; j < 4; j++)
#pragma unroll
            for (int e = 0; e < 4; e++) acc64[i][j][e] = 0;

    // per-lane fragment addressing (constant parts)
    //   A ldmatrix.x4: matrix i = lane>>3: rows (i&1)*8 + (lane&7), k-chunk (i>>1)   (-> a0,a1,a2,a3)
    //   B ldmatrix.x4: matrix i: n (i>>1)*8 + (lane&7), k-chunk (i&1)                 (-> b0,b1 of tile 2p, 2p+1)
    const int a_row0 = wm * 32 + ((lane >> 3) & 1) * 8 + (lane & 7);
    const int a_chk = (lane >> 4) & 1;
    const int b_row0 = wn * 32 + (lane >> 4) * 8 + (lane & 7);
    const int b_chk = (lane >> 3) & 1;
    const int ub_n = tid / 4, ub_s = tid & 3;          // unpack task: row ub_n (+ NT/4 steps), 32-block ub_s

    for (int g = 0; g < G; g++) {
        cp_async_wait<STAGES - 2>();
        __syncthreads();
        if (g + STAGES - 2 < G) store_mregs((g + STAGES - 2) % STAGES);
        if (g + STAGES - 1 < G) { issue(g + STAGES - 1, (g + STAGES - 1) % STAGES); load_mregs(g + STAGES - 1); }
        cp_async_commit();
        // unpack packed stage -> B8
        {
            const uint8_t* bp = Bp + (size_t)(g % STAGES) * BN * PSTR;
            for (int n = ub_n; n < BN; n += NT / 4) {
                uint32_t q[BPL / 4], u[8];
                lds_block<BITS>(bp + n * PSTR + ub_s * BPL, q);
                unpack32<BITS>(q, u);
                uint4 lo = make_uint4(to_signed<BITS>(u[0]), to_signed<BITS>(u[1]), to_signed<BITS>(u[2]), to_signed<BITS>(u[3]));
                uint4 hi = make_uint4(to_signed<BITS>(u[4]), to_signed<BITS>(u[5]), to_signed<BITS>(u[6]), to_signed<BITS>(u[7]));
                uint8_t* row = B8 + n * 128;
                *reinterpret_cast<uint4*>(row + (((2 * ub_s) ^ (n & 7)) << 4)) = lo;
                *reinterpret_cast<uint4*>(row + (((2 * ub_s + 1) ^ (n & 7)) << 4)) = hi;
            }
        }
        __syncthreads();
        // compute
#pragma unroll
        for (int i = 0; i < 2; i++)
#pragma unroll
            for (int j = 0; j < 4; j++)
#pragma unroll
                for (int e = 0; e < 4; e++) acc[i][j][e] = 0;
        const uint8_t* as = As + (size_t)(g % STAGES) * BM * 128;
#pragma unroll
        for (int ks = 0; ks < 4; ks++) {
            uint32_t a[2][4], b[4][2];
#pragma unroll
            for (int i = 0; i < 2; i++) {
                const int row = a_row0 + i * 16;
                const int chk = (2 * ks + a_chk) ^ (row & 7);
                ldmatrix_x4(smem_u32(as + row * 128 + (chk << 4)), a[i][0], a[i][1], a[i][2], a[i][3]);
            }
#pragma unroll
            for (int p = 0; p < 2; p++) {
                const int row = b_row0 + p * 16;
                const int chk = (2 * ks + b_chk) ^ (row & 7);
                ldmatrix_x4(smem_u32(B8 + row * 128 + (chk << 4)), b[2 * p][0], b[2 * p][1], b[2 * p + 1][0], b[2 * p + 1][1]);
            }
#pragma unroll
            for (int i = 0; i < 2; i++)
#pragma unroll
                for (int j = 0; j < 4; j++) mma_s8(acc[i][j], a[i], b[j]);
        }
        // scale by m*mx and accumulate into int64
        {
            const uint16_t* ms = Ms + (g % STAGES) * BN + wn * 32 + tig * 2;
            const uint16_t* mxs = Mxs + (g % STAGES) * BM + wm * 32 + gid;
            uint32_t mv[4][2], mxv[2][2];
#pragma unroll
            for (int j = 0; j < 4; j++) { const uint32_t w = *reinterpret_cast<const uint32_t*>(ms + j * 8); mv[j][0] = w & 0xFFFFu; mv[j][1] = w >> 16; }
#pragma unroll
            for (int i = 0; i < 2; i++) { mxv[i][0] = mxs[i * 16]; mxv[i][1] = mxs[i * 16 + 8]; }
#pragma unroll
            for (int i = 0; i < 2; i++)
#pragma unroll
                for (int j = 0; j < 4; j++)
#pragma unroll
                    for (int e = 0; e < 4; e++) acc64[i][j][e] += mul_s32_u32(acc[i][j][e], mv[j][e & 1] * mxv[i][e >> 1]);
        }
    }

    // ---- epilogue: acc64[i][j][e] -> row m0 + wm*32 + i*16 + gid + 8*(e>>1), col n0 + wn*32 + j*8 + tig*2 + (e&1)
#pragma unroll
    for (int i = 0; i < 2; i++) {
#pragma unroll
        for (int h = 0; h < 2; h++) {
            const int t = m0 + wm * 32 + i * 16 + gid + 8 * h;
            if (t >= M) continue;
            const float s = __ldg(sx + t);
#pragma unroll
            for (int j = 0; j < 4; j++) {
                const int n = n0 + wn * 32 + j * 8 + tig * 2;
                if (n < N) out.store(N, t, n, det::mul(det::mul(det::i64_to_f(acc64[i][j][2 * h]), __ldg(cc + n)), s));
                if (n + 1 < N) out.store(N, t, n + 1, det::mul(det::mul(det::i64_to_f(acc64[i][j][2 * h + 1]), __ldg(cc + n + 1)), s));
            }
        }
    }
}

// ------------------------------------------------------------------ prefill kernel (M > 64), int8 tensor cores, v2
// Block tile 128 x 128, 8 warps as 2 (m) x 4 (n), warp tile 64 x 32 (4 m16 x 4 n8 mma tiles), 1 block per SM.
// cp.async pipeline: A (activations, L2 resident) 2 stages / prefetch distance 1, packed weights Bp 3 stages /
// distance 2 -- A(g+1) is committed BEFORE Bp(g+2) every iteration so that wait_group(1) covers both A(g) and Bp(g).
// A fragments via ldmatrix.  B fragments are built by every lane straight from the PACKED stage: b0 = k tig*4..+3 and
// b1 = +16 of column n = gid, which for W6 are exactly the low nibbles of packed word tig and the matching 2-bit pairs
// of packed word 4+(tig&1) (W4: the two nibbles of word tig; W8: plain words of the swizzled int8 rows).  W4/W6 codes
// stay UNSIGNED (u = wq+OFF): the MMA runs .s8.u8 with the int32 accumulators initialised to -OFF*sum_k xq[t][k], so
// they end at the exact signed partial.  (An unpack-once-per-group int8 tile + ldmatrix variant was measured slower:
// the extra block barrier costs more than the per-lane unpack ALU.)  After the last k-step the int32 partials are
// scaled by m[n][g]*mx[t][g] (asm-volatile mad.wide interleaved with the mma stream) and accumulated into int64.
template <int BITS>
struct Mma2Cfg {
    static constexpr int BM = 128, BN = 128, NT = 256, SA = 2, SB = 3;
    static constexpr int PSTR = Fmt<BITS>::PSTR2;                       // packed row stride (W8: swizzled, W4/W6: padded)
    static constexpr int BCH = BITS;                                    // packed 16B chunks per row per group
    static constexpr int A_BYTES = SA * BM * 128;
    static constexpr int BP_BYTES = SB * BN * PSTR;
    static constexpr int V_BYTES = SB * (BN + BM + BM) * 4;             // Ms, Mxs (uint32) and Xs (int32) per stage
    static constexpr int SMEM = A_BYTES + BP_BYTES + V_BYTES;
};

template <int BITS>
__global__ void __launch_bounds__(256, 1)
k_mma2(const uint8_t* __restrict__ wq, const float* __restrict__ cc, const uint16_t* __restrict__ mm, int N, int K,
       const int8_t* __restrict__ xq, const float* __restrict__ sx, const uint16_t* __restrict__ mx,
       const int* __restrict__ xsum, int M, const Out out) {
    using C = Mma2Cfg<BITS>;
    constexpr int BM = C::BM, BN = C::BN, NT = C::NT, PSTR = C::PSTR, BCH = C::BCH, OFF = Fmt<BITS>::OFF;
    constexpr int SB = C::SB;
    constexpr int AR = BM * 8 / NT, BR = BN * BCH / NT;                   // cp.async chunks per thread (A, B)
    constexpr int GB = 128 * BITS / 8;                                    // packed bytes per row per group
    extern __shared__ __align__(128) uint8_t smem[];
    uint8_t* As = smem;
    uint8_t* Bp = As + C::A_BYTES;
    uint32_t* Ms = reinterpret_cast<uint32_t*>(Bp + C::BP_BYTES);       // [SB][BN]
    uint32_t* Mxs = Ms + SB * BN;                                         // [SB][BM]
    int* Xs = reinterpret_cast<int*>(Mxs + SB * BM);                      // [SB][BM]  -OFF * sum_k xq[t][k]

    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int wm = warp >> 2, wn = warp & 3;
    const int gid = lane >> 2, tig = lane & 3;
    const int m0 = blockIdx.x * BM, n0 = blockIdx.y * BN;
    const int G = K >> 7;
    const size_t rowbytes = (size_t)K * BITS / 8;

    // ---- cp.async bookkeeping (per-thread constant parts)
    const int a_t0 = tid >> 3, a_c = tid & 7;                            // A chunk r: token a_t0 + 32r, chunk a_c
    const int8_t* a_src0 = xq + (size_t)(m0 + a_t0) * K + a_c * 16;
    const uint32_t a_dst0 = smem_u32(As) + a_t0 * 128 + ((a_c ^ (a_t0 & 7)) << 4);
    const int a_tlim = M - m0 - a_t0;                                     // chunk r valid iff 32r < a_tlim
    uint32_t b_srcoff[BR], b_dstoff[BR];
    bool b_ok[BR];
#pragma unroll
    for (int r = 0; r < BR; r++) {                                        // B chunk r: e = tid + 256r -> (n, c)
        const int e = tid + NT * r, n = e / BCH, c = e - n * BCH;
        b_ok[r] = (n0 + n) < N;
        b_srcoff[r] = (uint32_t)(n * rowbytes) + c * 16;
        b_dstoff[r] = n * PSTR + ((BITS == 8) ? (c ^ (n & 7)) : c) * 16;
    }
    const uint8_t* b_src0 = wq + (size_t)n0 * rowbytes;
    const uint32_t b_dst0 = smem_u32(Bp);

    auto issue_a = [&](int g) {
        const int s = g & 1;
#pragma unroll
        for (int r = 0; r < AR; r++) {
            const bool ok = 32 * r < a_tlim;
            cp_async16(a_dst0 + s * (BM * 128) + r * 4096, ok ? a_src0 + (size_t)r * 32 * K + g * 128 : xq, ok);
        }
    };
    auto issue_b = [&](int g) {
        const int s = g % SB;
#pragma unroll
        for (int r = 0; r < BR; r++)
            cp_async16(b_dst0 + s * (BN * PSTR) + b_dstoff[r], b_ok[r] ? b_src0 + b_srcoff[r] + g * GB : wq, b_ok[r]);
    };
    uint32_t mreg = 0;
    int xreg = 0;
    auto load_mregs = [&](int g) {
        if (tid < BN) mreg = (n0 + tid < N) ? (uint32_t)__ldg(mm + (size_t)(n0 + tid) * G + g) : 0u;
        else {
            const int t = tid - BN;
            mreg = (m0 + t < M) ? (mx ? (uint32_t)__ldg(mx + (size_t)(m0 + t) * G + g) : 1u) : 0u;
            if (BITS != 8) xreg = (m0 + t < M) ? -OFF * __ldg(xsum + (size_t)(m0 + t) * G + g) : 0;
        }
    };
    auto store_mregs = [&](int g) {
        const int s = g % SB;
        if (tid < BN) Ms[s * BN + tid] = mreg;
        else { Mxs[s * BM + tid - BN] = mreg; if (BITS != 8) Xs[s * BM + tid - BN] = xreg; }
    };

    // ---- prologue: [A(0), Bp(0)] [Bp(1)]; m/mx/xsum of group 0 stored, group 1 in registers
    issue_a(0); issue_b(0); cp_async_commit();
    if (G > 1) issue_b(1);
    cp_async_commit();
    load_mregs(0); store_mregs(0);
    if (G > 1) load_mregs(1);

    long long acc64[4][4][4];
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
        for (int j = 0; j < 4; j++)
#pragma unroll
            for (int e = 0; e < 4; e++) acc64[i][j][e] = 0;

    // ldmatrix.x4 lane roles: matrix i = lane>>3 holds rows (i&1)*8 + (lane&7) [A] / (i>>1)*8 + (lane&7) [B],
    // k-chunk (i>>1) [A] / (i&1) [B]
    const int a_row0 = wm * 64 + ((lane >> 3) & 1) * 8 + (lane & 7);
    const int a_chk = (lane >> 4) & 1;
    const int b_n0 = wn * 32 + gid;                                       // B fragment column n = b_n0 + j*8
    const int sh0 = 4 - 2 * (tig >> 1), sh1 = 2 * (tig >> 1);            // W6 hi-bit shifts for this lane

    for (int g = 0; g < G; g++) {
        cp_async_wait<1>();                                               // A(g) and Bp(g) have landed (this thread)
        __syncthreads();                                                  // ... for everybody; stage reuse is safe
        if (g + 1 < G) store_mregs(g + 1);
        if (g + 1 < G) issue_a(g + 1);
        cp_async_commit();
        if (g + 2 < G) { issue_b(g + 2); load_mregs(g + 2); }
        cp_async_commit();

        const uint8_t* as = As + (size_t)(g & 1) * BM * 128;
        const uint8_t* bp = Bp + (size_t)(g % SB) * BN * PSTR;
        int init[4][2];
#pragma unroll
        for (int i = 0; i < 4; i++) {
            if (BITS != 8) {
                const int* xs = Xs + (g % SB) * BM + wm * 64 + gid;
                init[i][0] = xs[i * 16]; init[i][1] = xs[i * 16 + 8];
            } else { init[i][0] = 0; init[i][1] = 0; }
        }
        int acc[4][4][4];
        auto load_a = [&](int ks, int i, uint32_t (&a)[4]) {
            const int row = a_row0 + i * 16;
            const int chk = (2 * ks + a_chk) ^ (row & 7);
            ldmatrix_x4(smem_u32(as + row * 128 + (chk << 4)), a[0], a[1], a[2], a[3]);
        };
        auto load_b = [&](int ks, uint32_t (&b)[4][2]) {
#pragma unroll
            for (int j = 0; j < 4; j++) {
                const int n = b_n0 + j * 8;
                const uint8_t* row = bp + n * PSTR;
                if (BITS == 8) {
                    b[j][0] = *reinterpret_cast<const uint32_t*>(row + (((2 * ks) ^ (n & 7)) << 4) + tig * 4);
                    b[j][1] = *reinterpret_cast<const uint32_t*>(row + (((2 * ks + 1) ^ (n & 7)) << 4) + tig * 4);
                } else if (BITS == 6) {
                    const uint32_t ql = *reinterpret_cast<const uint32_t*>(row + ks * 24 + tig * 4);
                    const uint32_t qh = *reinterpret_cast<const uint32_t*>(row + ks * 24 + 16 + (tig & 1) * 4);
                    b[j][0] = (ql & 0x0F0F0F0Fu) | ((qh << sh0) & 0x30303030u);
                    b[j][1] = ((ql >> 4) & 0x0F0F0F0Fu) | ((qh >> sh1) & 0x30303030u);
                } else {
                    const uint32_t ql = *reinterpret_cast<const uint32_t*>(row + ks * 16 + tig * 4);
                    b[j][0] = ql & 0x0F0F0F0Fu;
                    b[j][1] = (ql >> 4) & 0x0F0F0F0Fu;
                }
            }
        };
        // k-step 0: C operand straight from the init registers
        {
            uint32_t b[4][2];
            load_b(0, b);
#pragma unroll
            for (int i = 0; i < 4; i++) {
                uint32_t a[4];
                load_a(0, i, a);
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    const int c[4] = {init[i][0], init[i][0], init[i][1], init[i][1]};
                    if (BITS == 8) mma_s8_cd(acc[i][j], a, b[j], c); else mma_s8u8_cd(acc[i][j], a, b[j], c);
                }
            }
        }
        // k-steps 1, 2
#pragma unroll
        for (int ks = 1; ks < 3; ks++) {
            uint32_t b[4][2];
            load_b(ks, b);
#pragma unroll
            for (int i = 0; i < 4; i++) {
                uint32_t a[4];
                load_a(ks, i, a);
#pragma unroll
                for (int j = 0; j < 4; j++) { if (BITS == 8) mma_s8(acc[i][j], a, b[j]); else mma_s8u8(acc[i][j], a, b[j]); }
            }
        }
        // k-step 3 with the m*mx scaling of each tile interleaved four tiles behind its last mma
        {
            const uint32_t* ms = Ms + (g % SB) * BN + wn * 32 + tig * 2;
            const uint32_t* mxs = Mxs + (g % SB) * BM + wm * 64 + gid;
            uint32_t mxv[4][2], mv[4][2];
#pragma unroll
            for (int i = 0; i < 4; i++) { mxv[i][0] = mxs[i * 16]; mxv[i][1] = mxs[i * 16 + 8]; }
#pragma unroll
            for (int j = 0; j < 4; j++) { const uint2 w = *reinterpret_cast<const uint2*>(ms + j * 8); mv[j][0] = w.x; mv[j][1] = w.y; }
            auto scale = [&](int t) {
                const int i = t >> 2, j = t & 3;
                mad_s32_u32(acc64[i][j][0], acc[i][j][0], mv[j][0] * mxv[i][0]);
                mad_s32_u32(acc64[i][j][1], acc[i][j][1], mv[j][1] * mxv[i][0]);
                mad_s32_u32(acc64[i][j][2], acc[i][j][2], mv[j][0] * mxv[i][1]);
                mad_s32_u32(acc64[i][j][3], acc[i][j][3], mv[j][1] * mxv[i][1]);
            };
            uint32_t b[4][2];
            load_b(3, b);
            uint32_t a[4];
#pragma unroll
            for (int t = 0; t < 16; t++) {
                const int i = t >> 2, j = t & 3;
                if (j == 0) load_a(3, i, a);
                if (BITS == 8) mma_s8(acc[i][j], a, b[j]); else mma_s8u8(acc[i][j], a, b[j]);
                if (t >= 4) scale(t - 4);
            }
#pragma unroll
            for (int t = 12; t < 16; t++) scale(t);
        }
    }

#pragma unroll
    for (int i = 0; i < 4; i++) {
#pragma unroll
        for (int h = 0; h < 2; h++) {
            const int t = m0 + wm * 64 + i * 16 + gid + 8 * h;
            if (t >= M) continue;
            const float s = __ldg(sx + t);
#pragma unroll
            for (int j = 0; j < 4; j++) {
                const int n = n0 + wn * 32 + j * 8 + tig * 2;
                if (n < N) out.store(N, t, n, det::mul(det::mul(det::i64_to_f(acc64[i][j][2 * h]), __ldg(cc + n)), s));
                if (n + 1 < N) out.store(N, t, n + 1, det::mul(det::mul(det::i64_to_f(acc64[i][j][2 * h + 1]), __ldg(cc + n + 1)), s));
            }
        }
    }
}

// xsum[t][g] = sum of the 128 int8 activations of token t, group g (for the unsigned-code MMA correction)
__global__ void k_xsum(const int8_t* __restrict__ xq, int M, int G, int* __restrict__ xsum) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * G) return;
    const int4* p = reinterpret_cast<const int4*>(xq + (size_t)idx * 128);   // (t*G + g)*128 == t*K + g*128
    int s = 0;
#pragma unroll
    for (int i = 0; i < 8; i++) {
        const int4 v = __ldg(p + i);
        s = dp4a_us(0x01010101u, v.x, s); s = dp4a_us(0x01010101u, v.y, s);
        s = dp4a_us(0x01010101u, v.z, s); s = dp4a_us(0x01010101u, v.w, s);
    }
    xsum[idx] = s;
}

// ------------------------------------------------------------------ host side
int g_num_sms = 0;
// xsum scratch for k_xsum (W4/W6, M > 64 path): normally the host-provided buffer set with det_gemm_set_scratch.
// Fallback when none (or a too small one) is set: a lazily grown global buffer (never freed, one-time stderr note;
// only safe when all det_gemm calls that use it are ordered on one stream).
int* g_xsum_user = nullptr;   size_t g_xsum_user_bytes = 0;
int* g_xsum = nullptr;        size_t g_xsum_cap = 0;          // fallback (elements)

static int* xsum_scratch(int M, int G, cudaStream_t st) {
    const size_t need = (size_t)M * G;
    if (g_xsum_user && need * sizeof(int) <= g_xsum_user_bytes) return g_xsum_user;
    static bool noted = false;
    if (!noted) {
        fprintf(stderr, "det_gemm: note: no scratch (or too small, have %zu bytes, need %zu for M=%d K=%d) set with "
                        "det_gemm_set_scratch; using a lazily allocated global buffer\n", g_xsum_user_bytes, need * sizeof(int), M, G * 128);
        noted = true;
    }
    if (need > g_xsum_cap) {
        DET_CHECK(cudaStreamSynchronize(st));   // the old buffer may still be in use by earlier launches on this stream
        if (g_xsum) DET_CHECK(cudaFree(g_xsum));
        g_xsum_cap = need < (size_t)(4096 * 136) ? (size_t)(4096 * 136) : need;
        DET_CHECK(cudaMalloc(&g_xsum, g_xsum_cap * sizeof(int)));
    }
    return g_xsum;
}

template <int MP, int BITS, int SMAX>
void launch_decode(const det_qlinear& L, const int8_t* xq, const float* sx, const uint16_t* mx, int M, const Out& y, int S, cudaStream_t st) {
    static int occ[33] = {0};  // per S (template instance is unique per <MP,BITS,SMAX>)
    static bool attr_set = false;
    constexpr bool XSM = DecCfg<MP, BITS, SMAX>::XSM;
    constexpr int XTOK = MP - DecCfg<MP, BITS, SMAX>::XR;                 // tokens kept in dynamic shared memory
    const int threads = S * 32;
    const int dsmem = XSM ? S * 32 * XTOK * 32 : 0;
    if (XSM && !attr_set) {   // static red[] + dynamic x exceed the 48 KB default for every S: opt in once (max size)
        DET_CHECK(cudaFuncSetAttribute(k_decode<MP, BITS, SMAX>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMAX * 32 * XTOK * 32));
        attr_set = true;
    }
    if (occ[S] == 0) {
        int nb = 0;
        DET_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, k_decode<MP, BITS, SMAX>, threads, dsmem));
        occ[S] = nb < 1 ? 1 : nb;
    }
    int blocks = occ[S] * g_num_sms;
    if (blocks > L.N) blocks = L.N;
    k_decode<MP, BITS, SMAX><<<blocks, threads, dsmem, st>>>(L.wq, L.c, L.m, L.N, L.K, xq, sx, mx, M, y);
    DET_CHECK(cudaGetLastError());
}

template <int MP, int BITS>
void decode_by_s(const det_qlinear& L, const int8_t* xq, const float* sx, const uint16_t* mx, int M, const Out& y, cudaStream_t st) {
    const int S = (L.K + 1023) / 1024;
    if (S <= 8) launch_decode<MP, BITS, 8>(L, xq, sx, mx, M, y, S, st);
    else if (S <= 17) launch_decode<MP, BITS, 17>(L, xq, sx, mx, M, y, S, st);
    else if (S <= 32) launch_decode<MP, BITS, 32>(L, xq, sx, mx, M, y, S, st);
    else { fprintf(stderr, "det_gemm: K=%d too large for decode kernel\n", L.K); abort(); }
}

template <int BITS>
void decode_by_m(const det_qlinear& L, const int8_t* xq, const float* sx, const uint16_t* mx, int M, const Out& y, cudaStream_t st) {
    if (M <= 1) decode_by_s<1, BITS>(L, xq, sx, mx, M, y, st);
    else if (M <= 2) decode_by_s<2, BITS>(L, xq, sx, mx, M, y, st);
    else if (M <= 4) decode_by_s<4, BITS>(L, xq, sx, mx, M, y, st);
    else decode_by_s<8, BITS>(L, xq, sx, mx, M, y, st);
}

template <int BITS, int TM>
void launch_prefill(const det_qlinear& L, const int8_t* xq, const float* sx, const uint16_t* mx, int M, const Out& y, cudaStream_t st) {
    constexpr int BM = 16 * TM, BN = 64;
    dim3 grid((M + BM - 1) / BM, (L.N + BN - 1) / BN);
    k_prefill<BITS, TM><<<grid, 256, 0, st>>>(L.wq, L.c, L.m, L.N, L.K, xq, sx, mx, M, y);    DET_CHECK(cudaGetLastError());
}

template <int BITS, int WM, int WN, int STAGES>
void launch_mma(const det_qlinear& L, const int8_t* xq, const float* sx, const uint16_t* mx, int M, const Out& y, cudaStream_t st) {
    using C = MmaCfg<BITS, WM, WN, STAGES>;
    static bool configured = false;
    if (!configured) {
        DET_CHECK(cudaFuncSetAttribute(k_mma<BITS, WM, WN, STAGES>, cudaFuncAttributeMaxDynamicSharedMemorySize, C::SMEM));
        configured = true;
    }
    dim3 grid((M + C::BM - 1) / C::BM, (L.N + C::BN - 1) / C::BN);
    k_mma<BITS, WM, WN, STAGES><<<grid, C::NT, C::SMEM, st>>>(L.wq, L.c, L.m, L.N, L.K, xq, sx, mx, M, y);    DET_CHECK(cudaGetLastError());
}

template <int BITS>
void launch_mma2(const det_qlinear& L, const int8_t* xq, const float* sx, const uint16_t* mx, int M, const Out& y, cudaStream_t st) {
    using C = Mma2Cfg<BITS>;
    static bool configured = false;
    if (!configured) {
        DET_CHECK(cudaFuncSetAttribute(k_mma2<BITS>, cudaFuncAttributeMaxDynamicSharedMemorySize, C::SMEM));
        configured = true;
    }
    const int G = L.K / 128;
    int* xs = nullptr;
    if (BITS != 8) {
        xs = xsum_scratch(M, G, st);
        k_xsum<<<(int)(((size_t)M * G + 255) / 256), 256, 0, st>>>(xq, M, G, xs);
        DET_CHECK(cudaGetLastError());
    }
    dim3 grid((M + C::BM - 1) / C::BM, (L.N + C::BN - 1) / C::BN);
    k_mma2<BITS><<<grid, C::NT, C::SMEM, st>>>(L.wq, L.c, L.m, L.N, L.K, xq, sx, mx, xs, M, y);    DET_CHECK(cudaGetLastError());
}

template <int BITS>
void gemm_bits(const det_qlinear& L, const int8_t* xq, const float* sx, const uint16_t* mx, int M, const Out& y, cudaStream_t st) {
    if (M <= 8) decode_by_m<BITS>(L, xq, sx, mx, M, y, st);
    else if (M <= 16) launch_prefill<BITS, 1>(L, xq, sx, mx, M, y, st);
    else if (M <= 32) launch_prefill<BITS, 2>(L, xq, sx, mx, M, y, st);
    else if (M <= 64) launch_mma<BITS, 2, 2, 3>(L, xq, sx, mx, M, y, st);
    else launch_mma2<BITS>(L, xq, sx, mx, M, y, st);
}

} // namespace

extern "C" size_t det_gemm_scratch_bytes(int M, int K) { return (size_t)(M < 1 ? 1 : M) * (size_t)((K + 127) / 128) * sizeof(int); }
extern "C" void det_gemm_set_scratch(void* p, size_t bytes) { g_xsum_user = (int*)p; g_xsum_user_bytes = p ? bytes : 0; }

extern "C" void det_gemm_ex(const det_qlinear* Lp, const int8_t* xq, const float* sx, const uint16_t* mx, int M, float* y, int accumulate,
                            int nparts, const int* offs, float* const* dsts, cudaStream_t st) {
    const det_qlinear L = *Lp;
    if (M < 1) return;
    if (L.K % 128 != 0 || L.K <= 0 || L.N <= 0) { fprintf(stderr, "det_gemm: bad shape N=%d K=%d\n", L.N, L.K); abort(); }
    Out out;
    memset(&out, 0, sizeof out);
    out.y = y;
    out.acc = accumulate ? 1 : 0;
    if (nparts > 1) {
        if (nparts > 4) { fprintf(stderr, "det_gemm: at most 4 output parts\n"); abort(); }
        if (offs[0] != 0) { fprintf(stderr, "det_gemm: the first part must start at column 0\n"); abort(); }
        out.nparts = nparts;
        for (int i = 0; i < nparts; i++) { out.dst[i] = dsts[i]; out.offs[i] = offs[i]; }
        out.offs[nparts] = L.N;
    } else if (!y) { fprintf(stderr, "det_gemm: y == NULL\n"); abort(); }
    if (g_num_sms == 0) {
        int dev = 0;
        DET_CHECK(cudaGetDevice(&dev));
        DET_CHECK(cudaDeviceGetAttribute(&g_num_sms, cudaDevAttrMultiProcessorCount, dev));
    }
    switch (L.bits) {
        case 8: gemm_bits<8>(L, xq, sx, mx, M, out, st); break;
        case 6: gemm_bits<6>(L, xq, sx, mx, M, out, st); break;
        case 4: gemm_bits<4>(L, xq, sx, mx, M, out, st); break;
        default: fprintf(stderr, "det_gemm: unsupported bits=%d\n", L.bits); abort();
    }
}

extern "C" void det_gemm(const det_qlinear* Lp, const int8_t* xq, const float* sx, const uint16_t* mx, int M, float* y, cudaStream_t st) {
    det_gemm_ex(Lp, xq, sx, mx, M, y, 0, 0, nullptr, nullptr, st);
}
