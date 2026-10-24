// det: quantized GEMM
//   acc[t][n] = sum_g ( m[n][g] * mx[t][g] ) * sum_{k in group g} wq[n][k] * xq[t][k]   (exact integer arithmetic)
//   y[t][n]   = ((float)acc[t][n] * c[n]) * sx[t]                                        (int64->fp32 RNE, 2 fp32 muls)
// Weight layout (Metal-specific, built by crates/det/src/metal.rs from the row-major packing): TILED by 64-row blocks
// and 128-wide k-groups, wq[ceil(N/64)][G][64][4*BPL bytes] (rows >= N zero), so that a threadgroup streaming the
// groups of a 64-row block reads one contiguous 4*BPL*64-byte chunk per group (whole cache lines; the row-major
// layout left 33-50% of every line to be re-fetched for W6/W4).
// Kernels:
//   k_gemm_dectc_<BITS>_<BN> (M <= 8): tensor-unit streaming matvec, see below.
//   k_gemm_pre_<BITS>_<BM>_<BN> (M > 8): int8 tensor-unit matmul (mpp::tensor_ops::matmul2d, s8 x s8 -> s32, exact).
//     Per 128-wide k-group: the BN x 128 weight tile is unpacked into threadgroup memory (signed int8), multiplied with
//     the activations straight from device memory into a cooperative (register) tensor, and every element is scaled
//     by m[n][g]*mx[t][g] into a per-thread int64 accumulator.  Epilogue: 2 fp32 multiplies.
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace det;
using namespace mpp::tensor_ops;

// ---- packed formats (spec §2): W8 32 int8; W4 16 bytes (u = wq+8, byte i = u[i] | u[i+16]<<4);
// W6 16 bytes of low nibbles as W4 (u = wq+32) + 8 bytes: byte i = hi[i] | hi[i+8]<<2 | hi[i+16]<<4 | hi[i+24]<<6.
// unpack32: 32 codes -> 8 words of 4 UNSIGNED codes u (consecutive k).  OFF = u - wq.
inline void load_block_8(device const uchar* p, thread uint* q) { device const uint4* p4 = (device const uint4*)p; uint4 a = p4[0], b = p4[1]; q[0]=a.x; q[1]=a.y; q[2]=a.z; q[3]=a.w; q[4]=b.x; q[5]=b.y; q[6]=b.z; q[7]=b.w; }
inline void load_block_6(device const uchar* p, thread uint* q) {
    // 24-byte blocks start 8-byte aligned, alternating 16-aligned / 16-aligned + 8: load the aligned 32-byte window with
    // two 16-byte loads (tile rows are 96 bytes: 16-aligned) and select the 6 words by parity
    const bool odd = (((ulong)p) >> 3) & 1ul;
    device const uint4* p4 = (device const uint4*)(((ulong)p) & ~15ul);
    const uint4 a = p4[0], b = p4[1];
    if (!odd) { q[0] = a.x; q[1] = a.y; q[2] = a.z; q[3] = a.w; q[4] = b.x; q[5] = b.y; }
    else      { q[0] = a.z; q[1] = a.w; q[2] = b.x; q[3] = b.y; q[4] = b.z; q[5] = b.w; }
}
inline void load_block_4(device const uchar* p, thread uint* q) { uint4 a = *(device const uint4*)p; q[0]=a.x; q[1]=a.y; q[2]=a.z; q[3]=a.w; }
// two consecutive blocks whose start is 16-byte aligned (block index even): exact 16-byte loads, no over-fetch
inline void load_pair_8(device const uchar* p, thread uint* q) { device const uint4* p4 = (device const uint4*)p; for (int v = 0; v < 4; v++) { uint4 a = p4[v]; q[4*v]=a.x; q[4*v+1]=a.y; q[4*v+2]=a.z; q[4*v+3]=a.w; } }
inline void load_pair_6(device const uchar* p, thread uint* q) { device const uint4* p4 = (device const uint4*)p; for (int v = 0; v < 3; v++) { uint4 a = p4[v]; q[4*v]=a.x; q[4*v+1]=a.y; q[4*v+2]=a.z; q[4*v+3]=a.w; } }
inline void load_pair_4(device const uchar* p, thread uint* q) { device const uint4* p4 = (device const uint4*)p; for (int v = 0; v < 2; v++) { uint4 a = p4[v]; q[4*v]=a.x; q[4*v+1]=a.y; q[4*v+2]=a.z; q[4*v+3]=a.w; } }
inline void unpack32_8(thread const uint* q, thread uint* u) { DET_U_GEMM for (int j = 0; j < 8; j++) u[j] = q[j] ^ 0x80808080u; }   // signed -> +128
inline void unpack32_4(thread const uint* q, thread uint* u) { DET_U_GEMM for (int j = 0; j < 4; j++) { u[j] = q[j] & 0x0F0F0F0Fu; u[j + 4] = (q[j] >> 4) & 0x0F0F0F0Fu; } }
inline void unpack32_6(thread const uint* q, thread uint* u) {
    const uint LM = 0x0F0F0F0Fu, HM = 0x30303030u;
    u[0] = (q[0] & LM) | ((q[4] << 4) & HM);  u[1] = (q[1] & LM) | ((q[5] << 4) & HM);
    u[2] = (q[2] & LM) | ((q[4] << 2) & HM);  u[3] = (q[3] & LM) | ((q[5] << 2) & HM);
    u[4] = ((q[0] >> 4) & LM) | (q[4] & HM);  u[5] = ((q[1] >> 4) & LM) | (q[5] & HM);
    u[6] = ((q[2] >> 4) & LM) | ((q[4] >> 2) & HM);  u[7] = ((q[3] >> 4) & LM) | ((q[5] >> 2) & HM);
}
// unsigned code word -> signed int8 word (byte-wise u - OFF without borrows; W8: u = wq ^ 0x80, so xor back)
inline uint to_signed_word(uint u, uint off4) { return ((u ^ 0x80808080u) - off4) ^ 0x80808080u; }
inline uint signed_word_8(uint u) { return u ^ 0x80808080u; }
inline uint signed_word_6(uint u) { return to_signed_word(u, 0x20202020u); }
inline uint signed_word_4(uint u) { return to_signed_word(u, 0x08080808u); }
// byte j of u as the float 256 + byte (exact): exponent 2^8, byte in the top mantissa bits
inline float magic(uint u, int j) { return as_type<float>(0x43800000u | ((j == 0 ? (u << 15) : j == 1 ? (u << 7) : j == 2 ? (u >> 1) : (u >> 9)) & 0x007F8000u)); }

#define DET_BPL_8 32
#define DET_BPL_6 24
#define DET_BPL_4 16
#define DET_OFF_8 128
#define DET_TCOFF_8 0
// byte offset of (row n, group g, block blk) in the tiled layout
#define DET_TILE_OFF(n, g, blk, G, BPL) ((((ulong)((n) >> 6) * (G) + (g)) * 64 + ((n) & 63)) * (4 * (BPL)) + (ulong)(blk) * (BPL))
#define DET_TCOFF_6 32
#define DET_TCOFF_4 8
#define DET_OFF_6 32
#define DET_OFF_4 8

// ------------------------------------------------------------------------------------------------ decode (M <= 8), tensor unit
// Threadgroup (4 simdgroups) = (64 rows of N, one of `splits` K-ranges).  It streams its groups: every thread owns 2
// packed blocks per group (64 rows x 4 blocks / 128 threads), prefetches the next group's packed words into registers
// while the current group is multiplied, and unpacks into the alternate threadgroup tile (one barrier per group).  The
// M (<= 8) activation rows of the group are copied into a zero-padded 8 x 128 threadgroup tile, so the 8 x 64 x 128
// int8 matmul on the tensor unit needs no bounds checks.  W4/W6 codes go into the tile UNSIGNED (u = wq + OFF, no
// per-word sign conversion); the matmul result is corrected with OFF * sum_k xq[t][k] over the group (the row sums
// of the next group's activation tile are computed by simdgroup 3 one iteration ahead).  The int32 group result is
// scaled by m[n][g]*mx[t][g] into int64 accumulators (4 elements per thread).  splits == 1 writes y directly; else
// the int64 partials go to part[split][M][N] and k_gemm_dectc_fin sums them (integers: order-free).
// epilogue: y[t][n] = v, or += v (accumulate: the residual add, det::add), optionally routed into up to 4 column
// blocks (a fused projection written straight into its part buffers: part i = columns [yoff[i], yoff[i+1]))
inline void gemm_store(constant det_p_gemm& p, int t, int n, float v) {
    if (p.nparts <= 1) {
        device float* y = (device float*)p.y;
        const ulong i = (ulong)t * p.N + n;
        y[i] = p.accumulate ? add(y[i], v) : v;
        return;
    }
    int part = 0;
    while (part + 1 < p.nparts && n >= p.yoff[part + 1]) part++;
    const int w = (part + 1 < p.nparts ? p.yoff[part + 1] : p.N) - p.yoff[part];
    device float* d = (device float*)p.ydst[part];
    const ulong i = (ulong)t * w + (n - p.yoff[part]);
    d[i] = p.accumulate ? add(d[i], v) : v;
}
inline uint tile_word_8(thread const uint* q, int j) { return q[j]; }          // signed codes as stored
inline int byte_sum(uint w) { const int4 v = int4(as_type<char4>(w)); return (v.x + v.y) + (v.z + v.w); }
// one group step (BN rows, NB = 4*BN/128 blocks per thread): unpack the register set QQ (group g) into the tile,
// stage the activations / multipliers of group g and the row sums of group g+1 (simdgroup 3), prefetch group g+2
// into QQ (the tile is double buffered for BN = 64, single buffered with a trailing barrier for BN = 128), barrier,
// matmul, scale.
#define DET_DECTC_STEP(BITS, BN, QQ)                                                                                   \
    {                                                                                                                  \
        const int cur = (BN == 64) ? (g & 1) : 0;                                                                      \
        DET_U_GEMM for (int b = 0; b < NB; b++) {                                                                                 \
            uint u[8];                                                                                                 \
            unpack32_##BITS(QQ + b * (DET_BPL_##BITS / 4), u);                                                         \
            threadgroup uint* dst = Wt + cur * (BN * 32) + r0 * 32 + (b0 + b) * 8;                                     \
            if (rok) { DET_U_GEMM for (int j = 0; j < 8; j++) dst[j] = (BITS == 8) ? tile_word_8(QQ + b * 8, j) : u[j]; }        \
            else { DET_U_GEMM for (int j = 0; j < 8; j++) dst[j] = 0u; }                                                          \
        }                                                                                                              \
        for (int w = tid; w < 8 * 32; w += 128) {                                                                      \
            const int t = w >> 5, col = w & 31;                                                                        \
            At[cur][w] = (t < M) ? xq[((ulong)t * K + (ulong)g * 128) / 4 + col] : 0u;                                 \
        }                                                                                                              \
        for (int i = tid; i < BN; i += 128) { const int n = n0 + i; ms[cur][i] = n < N ? (uint)m[(ulong)n * G + g] : 0u; } \
        if ((int)tid >= 64 && (int)tid < 72) { const int t = tid - 64; mxs[cur][t] = (t < M) ? (p.mx ? (uint)mx[(ulong)t * G + g] : 1u) : 0u; } \
        else if (tid >= 96 && g + 1 < g1) {                                                                            \
            DET_U_GEMM for (int t = 0; t < 8; t++) {                                                                              \
                const int v = (t < M) ? byte_sum(xq[((ulong)t * K + (ulong)(g + 1) * 128) / 4 + lane]) : 0;            \
                const int sum = simd_sum(v);                                                                           \
                if (lane == 0) xs[(g + 1) % 3][t] = sum;                                                               \
            }                                                                                                          \
        }                                                                                                              \
        if (g + 2 < g1) load_blocks_##BITS(wq + DET_TILE_OFF(wn, g + 2, b0, G, DET_BPL_##BITS), QQ, NB);               \
        threadgroup_barrier(mem_flags::mem_threadgroup);                                                               \
        auto tB = tensor((threadgroup int8_t*)(Wt + cur * (BN * 32)), dextents<int32_t, 2>(128, BN), array<int32_t, 2>{1, 128}); \
        auto tAt = tensor((threadgroup int8_t*)At[cur], dextents<int32_t, 2>(128, 8), array<int32_t, 2>{1, 128});      \
        DET_U_GEMM for (int i = 0; i < CAP; i++) if (i < (int)cT.get_capacity() && cT.is_valid_element(i)) cT[i] = 0;              \
        op.run(tAt, tB, cT);                                                                                           \
        DET_U_GEMM for (int i = 0; i < CAP; i++) {   /* constant trip count (unrollable); CAP >= the probed capacity */           \
            if (i < (int)cT.get_capacity() && cT.is_valid_element(i)) {                                               \
                auto idx = cT.get_multidimensional_index(i);                                                           \
                const int part = cT[i] - DET_TCOFF_##BITS * xs[g % 3][idx[1]];                                         \
                acc[i] += (long)part * (long)(ms[cur][idx[0]] * mxs[cur][idx[1]]);                                     \
            }                                                                                                          \
        }                                                                                                              \
        if (BN != 64) threadgroup_barrier(mem_flags::mem_threadgroup);                                                 \
    }
// NB consecutive blocks (16-byte aligned start): exact 16-byte loads
inline void load_blocks_8(device const uchar* p, thread uint* q, int nb) { device const uint4* p4 = (device const uint4*)p; DET_U_GEMM for (int v = 0; v < 2 * nb; v++) { uint4 a = p4[v]; q[4*v]=a.x; q[4*v+1]=a.y; q[4*v+2]=a.z; q[4*v+3]=a.w; } }
inline void load_blocks_6(device const uchar* p, thread uint* q, int nb) { device const uint4* p4 = (device const uint4*)p; DET_U_GEMM for (int v = 0; v < 3 * nb / 2; v++) { uint4 a = p4[v]; q[4*v]=a.x; q[4*v+1]=a.y; q[4*v+2]=a.z; q[4*v+3]=a.w; } }
inline void load_blocks_4(device const uchar* p, thread uint* q, int nb) { device const uint4* p4 = (device const uint4*)p; DET_U_GEMM for (int v = 0; v < nb; v++) { uint4 a = p4[v]; q[4*v]=a.x; q[4*v+1]=a.y; q[4*v+2]=a.z; q[4*v+3]=a.w; } }
#define DET_GEMM_DECTC(BITS, BN)                                                                                       \
kernel void k_gemm_dectc_##BITS##_##BN(constant det_p_gemm& p [[buffer(0)]], uint2 tgid [[threadgroup_position_in_grid]], \
                                       uint tid [[thread_index_in_threadgroup]]) {                                     \
    constexpr int NB = 4 * BN / 128;             /* blocks per thread per group */                                     \
    constexpr int NBUF = (BN == 64) ? 2 : 1;                                                                           \
    constexpr int CAP = (BN == 64) ? 8 : 16;     /* cooperative tensor capacity per thread (8 x BN tiles; probed) */  \
    threadgroup uint Wt[NBUF * BN * 32];                                                                               \
    threadgroup uint At[2][8 * 32];                                                                                    \
    threadgroup uint ms[2][BN]; threadgroup uint mxs[2][8]; threadgroup int xs[3][8];                                  \
    device const uchar* wq = (device const uchar*)p.wq;                                                                \
    device const float* c = (device const float*)p.c;                                                                  \
    device const ushort* m = (device const ushort*)p.m;                                                                \
    device const uint* xq = (device const uint*)p.xq;                                                                  \
    device const float* sx = (device const float*)p.sx;                                                                \
    device const ushort* mx = (device const ushort*)p.mx;                                                              \
    device float* y = (device float*)p.y;                                                                              \
    const int M = p.M, N = p.N, K = p.K, G = p.G, splits = p.splits;                                                 \
    const int n0 = tgid.x * BN;                                                                                        \
    const int gper = (G + splits - 1) / splits;                                                                        \
    const int g0 = tgid.y * gper, g1 = min(G, g0 + gper);                                                              \
    constexpr auto d = matmul2d_descriptor(8, BN, 128, false, true, false);                                            \
    matmul2d<d, execution_simdgroups<4>> op;                                                                           \
    using TA = tensor<threadgroup int8_t, dextents<int32_t, 2>, tensor_inline>;                                       \
    using TB = tensor<threadgroup int8_t, dextents<int32_t, 2>, tensor_inline>;                                       \
    auto cT = op.get_destination_cooperative_tensor<TA, TB, int32_t>();                                                \
    long acc[CAP];                                                                                                     \
    DET_U_GEMM for (int i = 0; i < CAP; i++) acc[i] = 0;                                                                          \
    /* thread -> (row r0, first block b0): BN=64: 2 blocks of row tid/2; BN=128: all 4 blocks of row tid */            \
    const int r0 = (BN == 64) ? (tid >> 1) : tid, b0 = (BN == 64) ? (tid & 1) * 2 : 0;                                 \
    const int nrow = n0 + r0;                                                                                          \
    const bool rok = nrow < N;                                                                                         \
    const int wn = rok ? nrow : 0;                                                                                     \
    const int lane = tid & 31;                                                                                         \
    uint qA[NB * DET_BPL_##BITS / 4], qB[NB * DET_BPL_##BITS / 4];                                                     \
    if (g0 < g1) {                                                                                                     \
        load_blocks_##BITS(wq + DET_TILE_OFF(wn, g0, b0, G, DET_BPL_##BITS), qA, NB);                                   \
        if (g0 + 1 < g1) load_blocks_##BITS(wq + DET_TILE_OFF(wn, g0 + 1, b0, G, DET_BPL_##BITS), qB, NB);              \
        if (tid >= 96) {                                                                                               \
            DET_U_GEMM for (int t = 0; t < 8; t++) {                                                                              \
                const int v = (t < M) ? byte_sum(xq[((ulong)t * K + (ulong)g0 * 128) / 4 + lane]) : 0;                 \
                const int sum = simd_sum(v);                                                                           \
                if (lane == 0) xs[g0 % 3][t] = sum;                                                                    \
            }                                                                                                          \
        }                                                                                                              \
    }                                                                                                                  \
    for (int g = g0; g < g1; g++) {                                                                                    \
        DET_DECTC_STEP(BITS, BN, qA)                                                                                   \
        g++;                                                                                                           \
        if (g < g1) DET_DECTC_STEP(BITS, BN, qB)                                                                       \
    }                                                                                                                  \
    if (splits == 1) {                                                                                                 \
        DET_U_GEMM for (int i = 0; i < CAP; i++) {                                                                                \
            if (i < (int)cT.get_capacity() && cT.is_valid_element(i)) {                                               \
                auto idx = cT.get_multidimensional_index(i);                                                           \
                const int n = n0 + idx[0], t = idx[1];                                                                 \
                if (n < N && t < M) gemm_store(p, t, n, mul(mul((float)acc[i], c[n]), sx[t]));                         \
            }                                                                                                          \
        }                                                                                                              \
        return;                                                                                                        \
    }                                                                                                                  \
    /* split-K: publish the int64 partials; the last threadgroup of this row tile sums all splits (integers: any order) */ \
    threadgroup int is_last;                                                                                           \
    device long* part = (device long*)p.part;                                                                          \
    DET_U_GEMM for (int i = 0; i < CAP; i++) {                                                                                    \
        if (i < (int)cT.get_capacity() && cT.is_valid_element(i)) {                                                   \
            auto idx = cT.get_multidimensional_index(i);                                                               \
            const int n = n0 + idx[0], t = idx[1];                                                                     \
            if (n < N && t < M) part[((ulong)tgid.y * M + t) * N + n] = acc[i];                                        \
        }                                                                                                              \
    }                                                                                                                  \
    threadgroup_barrier(mem_flags::mem_device);                                                                        \
    if (tid == 0) {                                                                                                    \
        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);                         \
        device atomic_uint* ctr = (device atomic_uint*)p.counters + tgid.x;                                            \
        const uint old = atomic_fetch_add_explicit(ctr, 1u, memory_order_relaxed);                                     \
        is_last = (old == (uint)splits - 1) ? 1 : 0;                                                                   \
        if (is_last) atomic_store_explicit(ctr, 0u, memory_order_relaxed);   /* self-reset for the next call */       \
    }                                                                                                                  \
    threadgroup_barrier(mem_flags::mem_threadgroup);                                                                   \
    if (!is_last) return;                                                                                              \
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);                             \
    DET_U_GEMM for (int i = 0; i < CAP; i++) {                                                                                    \
        if (i < (int)cT.get_capacity() && cT.is_valid_element(i)) {                                                   \
            auto idx = cT.get_multidimensional_index(i);                                                               \
            const int n = n0 + idx[0], t = idx[1];                                                                     \
            if (n < N && t < M) {                                                                                      \
                long v = 0;                                                                                            \
                DET_U_GEMM for (int sp = 0; sp < splits; sp++) v += part[((ulong)sp * M + t) * N + n];                            \
                gemm_store(p, t, n, mul(mul((float)v, c[n]), sx[t]));                                                  \
            }                                                                                                          \
        }                                                                                                              \
    }                                                                                                                  \
}
DET_GEMM_DECTC(8, 64) DET_GEMM_DECTC(6, 64) DET_GEMM_DECTC(4, 64)
DET_GEMM_DECTC(8, 128) DET_GEMM_DECTC(6, 128) DET_GEMM_DECTC(4, 128)

// ------------------------------------------------------------------------------------------------ prefill
#define DET_GEMM_PRE(BITS, BM, BN)                                                                                     \
kernel void k_gemm_pre_##BITS##_##BM##_##BN(constant det_p_gemm& p [[buffer(0)]], uint2 tgid [[threadgroup_position_in_grid]], \
                                            uint tid [[thread_index_in_threadgroup]]) {                                \
    threadgroup uint Wt[BN * 32];                                                                                      \
    threadgroup uint ms[BN]; threadgroup uint mxs[BM];                                                                 \
    device const uchar* wq = (device const uchar*)p.wq;                                                                \
    device const float* c = (device const float*)p.c;                                                                  \
    device const ushort* m = (device const ushort*)p.m;                                                                \
    device int8_t* xq = (device int8_t*)p.xq;                                                                          \
    device const float* sx = (device const float*)p.sx;                                                                \
    device const ushort* mx = (device const ushort*)p.mx;                                                              \
    device float* y = (device float*)p.y;                                                                              \
    const int M = p.M, N = p.N, K = p.K, G = p.G;                                                                    \
    const int n0 = tgid.x * BN, t0 = tgid.y * BM;                                                                      \
    const bool interior = t0 + BM <= M;                                                                                \
    constexpr auto d = matmul2d_descriptor(BM, BN, 128, false, true, false);                                           \
    matmul2d<d, execution_simdgroups<4>> op;                                                                           \
    auto tA = tensor(xq, dextents<int32_t, 2>(K, M), array<int32_t, 2>{1, K});                                         \
    using TB = tensor<threadgroup int8_t, dextents<int32_t, 2>, tensor_inline>;                                       \
    auto cT = op.get_destination_cooperative_tensor<decltype(tA.slice(0, 0)), TB, int32_t>();                          \
    constexpr int CAP = BM * BN / 128;                                                                                 \
    long acc[CAP];                                                                                                     \
    DET_U_GEMM for (int i = 0; i < CAP; i++) acc[i] = 0;                                                                          \
    for (int g = 0; g < G; g++) {                                                                                      \
        threadgroup_barrier(mem_flags::mem_threadgroup);                                                               \
        if ((int)tid < BN) { const int n = n0 + tid; ms[tid] = n < N ? (uint)m[(ulong)n * G + g] : 0u; }               \
        if ((int)tid < BM) { const int t = t0 + tid; mxs[tid] = (t < M) ? (p.mx ? (uint)mx[(ulong)t * G + g] : 1u) : 0u; } \
        for (int b = tid; b < BN * 4; b += 128) {                                                                      \
            const int r = b >> 2, blk = b & 3, n = n0 + r;                                                             \
            threadgroup uint* dst = Wt + r * 32 + blk * 8;                                                             \
            if (n < N) {                                                                                               \
                uint q[DET_BPL_##BITS / 4], u[8];                                                                      \
                load_block_##BITS(wq + DET_TILE_OFF(n, g, blk, G, DET_BPL_##BITS), q);                                   \
                unpack32_##BITS(q, u);                                                                                 \
                DET_U_GEMM for (int j = 0; j < 8; j++) dst[j] = signed_word_##BITS(u[j]);              \
            } else { DET_U_GEMM for (int j = 0; j < 8; j++) dst[j] = 0u; }                                                        \
        }                                                                                                              \
        threadgroup_barrier(mem_flags::mem_threadgroup);                                                               \
        auto tB = tensor((threadgroup int8_t*)Wt, dextents<int32_t, 2>(128, BN), array<int32_t, 2>{1, 128});           \
        DET_U_GEMM for (int i = 0; i < CAP; i++) if (cT.is_valid_element(i)) cT[i] = 0;                                           \
        if (interior) { auto mA = tA.template slice<128, BM>(g * 128, t0); op.run(mA, tB, cT); }                         \
        else { auto mA = tA.slice(g * 128, t0); op.run(mA, tB, cT); }                                                  \
        DET_U_GEMM for (int i = 0; i < CAP; i++) {                                                                                \
            if (cT.is_valid_element(i)) {                                                                              \
                auto idx = cT.get_multidimensional_index(i);                                                           \
                acc[i] += (long)cT[i] * (long)(ms[idx[0]] * mxs[idx[1]]);                                              \
            }                                                                                                          \
        }                                                                                                              \
    }                                                                                                                  \
    DET_U_GEMM for (int i = 0; i < CAP; i++) {                                                                                    \
        if (cT.is_valid_element(i)) {                                                                                  \
            auto idx = cT.get_multidimensional_index(i);                                                               \
            const int n = n0 + idx[0], t = t0 + idx[1];                                                                \
            if (n < N && t < M) gemm_store(p, t, n, mul(mul((float)acc[i], c[n]), sx[t]));                                \
        }                                                                                                              \
    }                                                                                                                  \
}
DET_GEMM_PRE(8, 64, 64) DET_GEMM_PRE(6, 64, 64) DET_GEMM_PRE(4, 64, 64)
DET_GEMM_PRE(8, 32, 64) DET_GEMM_PRE(6, 32, 64) DET_GEMM_PRE(4, 32, 64)
DET_GEMM_PRE(8, 16, 64) DET_GEMM_PRE(6, 16, 64) DET_GEMM_PRE(4, 16, 64)
