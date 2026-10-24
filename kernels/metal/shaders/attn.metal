// Full attention core (docs/SPEC.md 5.1).  Port of the CUDA-core path of cuda/attn.cu:
//   k_attn_prep : one simdgroup per row.  q rows: rmsnorm(q_norm) -> rope -> *qscale -> int8 (scratch); k rows: rmsnorm ->
//                 rope -> int8 (KV cache); v rows: int16 (KV cache).
//   k_attn_max  : grid (nsplit, n_kv, M); block = (token, kv head, key range): s_j for the REP heads of the kv head from
//                 exact int8 dots on the tensor unit (8 x 32 x HD int8 matmul per 32-key tile, K rows read straight
//                 from the cache), atomic max of the order-preserving code into mx[t][h].
//   k_attn_acc  : same grid; pq_j = rint(exp_f(s_j - mx) * 2^24); den = sum pq_j, num[d] = sum pq_j * vq_j[d] in int64
//                 (V rows read straight from the cache).
//                 nsplit == 1 writes the gated output; otherwise the per-split int64 partials go to scratch and
//   k_attn_fin  : sums the splits (integers: order-free) and writes o = ((float)num / (float)den) * v_scale * sigmoid(gate).
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace det;
using namespace mpp::tensor_ops;

#define ATK 32            // keys per tile
#define ANT 128           // threads per block
#define P_FIX 16777216.0f
inline uint f2ord(float f) { uint u = as_type<uint>(f); return (u & 0x80000000u) ? ~u : (u | 0x80000000u); }
inline float ord2f(uint u) { return as_type<float>((u & 0x80000000u) ? (u & 0x7fffffffu) : ~u); }
inline void key_range_t(int nkeys, int nsplit, int split, int tk, thread int& j0, thread int& j1) {
    int chunk = ((nkeys + nsplit - 1) / nsplit + tk - 1) / tk * tk;
    j0 = split * chunk;
    j1 = min(nkeys, j0 + chunk);
}
inline void key_range(int nkeys, int nsplit, int split, thread int& j0, thread int& j1) { key_range_t(nkeys, nsplit, split, ATK, j0, j1); }

// ---- prep: rows [0, M*n_h) = q, [M*n_h, M*n_h + M*n_kv) = k, then v.  CHUNK = HD/32 contiguous elements per lane.
#define DET_ATTN_PREP(HD)                                                                                             \
kernel void k_attn_prep_##HD(constant det_p_attn& p [[buffer(0)]], uint bid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) { \
    constexpr int CHUNK = HD / 32;                                                                                    \
    const int lane = tid & 31, w = tid >> 5;                                                                          \
    const int row = bid * 8 + w;                                                                                      \
    const int M = p.M, n_h = p.n_h, n_kv = p.n_kv, rot = p.rot, pos0 = p.pos0;                                        \
    const int nq = M * n_h, nk = M * n_kv;                                                                            \
    if (row >= nq + 2 * nk) return;                                                                                   \
    device const float* qg = (device const float*)p.qg;                                                               \
    if (row < nq + nk) {                                                                                              \
        int t, hh; device const float* src; device const float* nw;                                                   \
        if (row < nq) { t = row / n_h; hh = row - t * n_h; src = qg + ((ulong)t * n_h + hh) * 2 * HD; nw = (device const float*)p.q_norm; } \
        else { int r = row - nq; t = r / n_kv; hh = r - t * n_kv; src = (device const float*)p.k + ((ulong)t * n_kv + hh) * HD; nw = (device const float*)p.k_norm; } \
        float x[CHUNK], sqv[CHUNK];                                                                                   \
        DET_U_ATTN for (int i = 0; i < CHUNK; i++) { x[i] = src[lane * CHUNK + i]; sqv[i] = mul(x[i], x[i]); }                   \
        float ss = warp_tree(tree_local<CHUNK>(sqv));                                                                 \
        float mean = div(ss, (float)HD);                                                                              \
        float inv = div(1.0f, sqrt_(add(mean, p.eps)));                                                               \
        DET_U_ATTN for (int i = 0; i < CHUNK; i++) x[i] = mul(mul(x[i], inv), nw[lane * CHUNK + i]);                             \
        const int pos = pos0 + t, hf = rot >> 1;                                                                      \
        const ushort xoff = (ushort)(hf / CHUNK);          /* partner lane distance (host checks power of two) */  \
        device const float* rc = (device const float*)p.rope_cos + (ulong)pos * hf;                                 \
        device const float* rs = (device const float*)p.rope_sin + (ulong)pos * hf;                                 \
        float y[CHUNK];                                                                                               \
        DET_U_ATTN for (int i = 0; i < CHUNK; i++) {                                                                             \
            const int idx = lane * CHUNK + i;                                                                         \
            const float other = simd_shuffle_xor(x[i], xoff);                                                         \
            if (idx < hf)        { y[i] = sub(mul(x[i], rc[idx]), mul(other, rs[idx])); }                             \
            else if (idx < rot)  { y[i] = add(mul(x[i], rc[idx - hf]), mul(other, rs[idx - hf])); }               \
            else                 y[i] = x[i];                                                                         \
        }                                                                                                             \
        if (row < nq) DET_U_ATTN for (int i = 0; i < CHUNK; i++) y[i] = mul(y[i], p.qscale);                                     \
        float am = 0.0f;                                                                                              \
        DET_U_ATTN for (int i = 0; i < CHUNK; i++) am = fmax_(am, fabs(y[i]));                                                   \
        am = warp_max(am);                                                                                            \
        const float s = act_scale(am);                                                                                \
        const float invs = div(1.0f, s);                                                                              \
        device char* dst; device float* sdst;                                                                         \
        if (row < nq) { dst = (device char*)p.qq + (ulong)row * HD; sdst = (device float*)p.sq + row; if (lane == 0) ((device uint*)p.mx)[row] = 0u; } \
        else { dst = (device char*)p.kq + ((ulong)pos * n_kv + hh) * HD; sdst = (device float*)p.sk + (ulong)pos * n_kv + hh; } \
        DET_U_ATTN for (int i = 0; i < CHUNK; i++) dst[lane * CHUNK + i] = act_q(y[i], invs);                                    \
        if (lane == 0) *sdst = s;                                                                                     \
    } else {                                                                                                          \
        const int r = row - nq - nk, t = r / n_kv, hh = r - t * n_kv, pos = pos0 + t;                                 \
        device const float* src = (device const float*)p.v + ((ulong)t * n_kv + hh) * HD;                             \
        const float vinv = ((device const float*)p.v_inv_scale)[hh];                                                  \
        device short* dst = (device short*)p.vq + ((ulong)pos * n_kv + hh) * HD;                                      \
        device char* dlo = p.vlo ? (device char*)p.vlo + ((ulong)pos * n_kv + hh) * HD : (device char*)0;             \
        device char* dhi = p.vhi ? (device char*)p.vhi + ((ulong)pos * n_kv + hh) * HD : (device char*)0;                                      \
        DET_U_ATTN for (int i = 0; i < CHUNK; i++) {                                                                             \
            int q = rint_i(mul(src[lane * CHUNK + i], vinv));                                                         \
            q = q > 32767 ? 32767 : (q < -32767 ? -32767 : q);                                                        \
            dst[lane * CHUNK + i] = (short)q;                                                                         \
            if (dlo) { dlo[lane * CHUNK + i] = (char)((q & 255) - 128); dhi[lane * CHUNK + i] = (char)(q >> 8); }                                                                         \
        }                                                                                                             \
    }                                                                                                                 \
}
DET_ATTN_PREP(256) DET_ATTN_PREP(128)

// K tile: ATK keys x HD int8 as uint words, padded row stride KS words; q rows pre-converted to float.
// Thread (kk = tid % ATK, g = tid / ATK) computes the dots of key kk for heads r with r % 4 == g.
#define DET_ATTN_DOT(HD)                                                                                              \
inline float attn_dot_##HD(threadgroup const uint* Ks, threadgroup const float* qs, int kk, int r) {                  \
    constexpr int KW = HD / 4, KS = KW + 1;                                                                           \
    float d = 0.0f;                                                                                                   \
    DET_U_ATTN for (int w = 0; w < KW; w++) {                                                                                    \
        const float4 a = float4(as_type<char4>(Ks[kk * KS + w]));                                                     \
        const float4 b = *(threadgroup const float4*)(qs + r * HD + w * 4);                                           \
        d = fma(a.x, b.x, d); d = fma(a.y, b.y, d); d = fma(a.z, b.z, d); d = fma(a.w, b.w, d);                       \
    }                                                                                                                 \
    return d;                                                                                                         \
}
DET_ATTN_DOT(256) DET_ATTN_DOT(128)

#define DET_ATTN_LOADK(HD)                                                                                            \
inline void attn_load_k_##HD(threadgroup uint* Ks, threadgroup float* sks, device const char* kq, device const float* sk, \
                             int j, int nvalid, int n_kv, int kvh, uint tid) {                                        \
    constexpr int KW = HD / 4, KS = KW + 1;                                                                           \
    for (int i = tid; i < nvalid * KW; i += ANT) {                                                                    \
        const int key = i / KW, w = i - key * KW;                                                                     \
        Ks[key * KS + w] = ((device const uint*)(kq + ((ulong)(j + key) * n_kv + kvh) * HD))[w];                      \
    }                                                                                                                 \
    if ((int)tid < nvalid) sks[tid] = sk[(ulong)(j + tid) * n_kv + kvh];                                              \
}
DET_ATTN_LOADK(256) DET_ATTN_LOADK(128)

// ---- scores on the tensor unit: the REP q rows of the kv head (int8, zero-padded to 8) x the 32 keys of a tile
// (int8 rows straight from the KV cache: no staging) -> 8 x 32 int32 dots, exact.  Each thread owns ~2 (key, head)
// elements of the cooperative result.  Keys beyond the tile's valid count are excluded through the tensor extents.
#define ATT_CAP 8
#define DET_ATTN_SCORES(HD)                                                                                           \
    constexpr auto d = matmul2d_descriptor(8, ATK, HD, false, true, false);                                           \
    matmul2d<d, execution_simdgroups<4>> op;                                                                           \
    using TQ = tensor<threadgroup int8_t, dextents<int32_t, 2>, tensor_inline>;                                       \
    using TK = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>;                                            \
    auto cT = op.get_destination_cooperative_tensor<TQ, TK, int32_t>();                                                \
    auto tQ = tensor((threadgroup int8_t*)qs, dextents<int32_t, 2>(HD, 8), array<int32_t, 2>{1, HD});

// ---- pass 1: per-head max of s_j over this block's key range -> atomic max into mx[t][h]
#define DET_ATTN_MAX(HD, REP)                                                                                         \
kernel void k_attn_max_##HD##_##REP(constant det_p_attn& p [[buffer(0)]], uint3 tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) { \
    threadgroup uint qs[8 * HD / 4];                                                                                  \
    threadgroup atomic_uint bm[REP];                                                                                  \
    const int t = tgid.z, kvh = tgid.y, n_h = p.n_h, n_kv = p.n_kv;                                                   \
    const int nkeys = p.pos0 + t + 1;                                                                                 \
    int j0, j1; key_range(nkeys, p.nsplit, tgid.x, j0, j1);                                                           \
    if (j0 >= j1) return;                                                                                             \
    const int h0 = kvh * REP;                                                                                         \
    device const uint* qq = (device const uint*)((device const char*)p.qq + ((ulong)t * n_h + h0) * HD);              \
    for (int i = tid; i < 8 * HD / 4; i += ANT) qs[i] = (i < REP * HD / 4) ? qq[i] : 0u;                              \
    float sqv[REP];                                                                                                   \
    for (int r = 0; r < REP; r++) sqv[r] = ((device const float*)p.sq)[(ulong)t * n_h + h0 + r];                      \
    if ((int)tid < REP) atomic_store_explicit(&bm[tid], 0u, memory_order_relaxed);                                    \
    threadgroup_barrier(mem_flags::mem_threadgroup);                                                                  \
    DET_ATTN_SCORES(HD)                                                                                               \
    device const char* kq = (device const char*)p.kq;                                                                 \
    device const float* sk = (device const float*)p.sk;                                                               \
    float lm[REP];                                                                                                    \
    for (int r = 0; r < REP; r++) lm[r] = -INFINITY;                                                                  \
    for (int j = j0; j < j1; j += ATK) {                                                                              \
        const int nvalid = min(ATK, j1 - j);                                                                          \
        auto tK = tensor((device int8_t*)(kq + ((ulong)j * n_kv + kvh) * HD), dextents<int32_t, 2>(HD, nvalid), array<int32_t, 2>{1, n_kv * HD}); \
        for (int i = 0; i < ATT_CAP; i++) if (i < (int)cT.get_capacity() && cT.is_valid_element(i)) cT[i] = 0;        \
        op.run(tQ, tK, cT);                                                                                           \
        for (int i = 0; i < ATT_CAP; i++) {                                                                           \
            if (i < (int)cT.get_capacity() && cT.is_valid_element(i)) {                                               \
                auto idx = cT.get_multidimensional_index(i);                                                           \
                const int kk = idx[0], r = idx[1];                                                                    \
                if (kk < nvalid && r < REP) {                                                                         \
                    const float s = mul(mul((float)cT[i], sqv[r]), sk[(ulong)(j + kk) * n_kv + kvh]);                 \
                    lm[r] = fmax_(lm[r], s);                                                                          \
                }                                                                                                     \
            }                                                                                                         \
        }                                                                                                             \
    }                                                                                                                 \
    for (int r = 0; r < REP; r++) atomic_fetch_max_explicit(&bm[r], f2ord(lm[r]), memory_order_relaxed);              \
    threadgroup_barrier(mem_flags::mem_threadgroup);                                                                  \
    if ((int)tid < REP) atomic_fetch_max_explicit((device atomic_uint*)p.mx + (ulong)t * n_h + h0 + tid, atomic_load_explicit(&bm[tid], memory_order_relaxed), memory_order_relaxed); \
}

// ---- pass 2: pq_j = rint(exp_f(s_j - mx) * 2^24); den += pq_j; num[d] += pq_j * vq_j[d]  (int64)
// Scores per tile as in pass 1 (tensor unit) -> pqs[r][key] in threadgroup memory; P.V with V rows read straight
// from the cache: thread (d0 = (tid % TPK) * 4, kg = tid / TPK) accumulates 4 dims over the keys kg, kg + NKG, ...
#define DET_ATTN_ACC(HD, REP)                                                                                         \
kernel void k_attn_acc_##HD##_##REP(constant det_p_attn& p [[buffer(0)]], uint3 tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) { \
    constexpr int TPK = HD / 4, NKG = ANT / TPK;                                                                      \
    threadgroup uint qs[8 * HD / 4];                                                                                  \
    threadgroup int pqs[REP][ATK];                                                                                    \
    threadgroup long accs[REP * HD];                                                                                  \
    threadgroup long dens[4 * REP];                                                                                   \
    const int t = tgid.z, kvh = tgid.y, n_h = p.n_h, n_kv = p.n_kv, nsplit = p.nsplit;                                \
    const int nkeys = p.pos0 + t + 1;                                                                                 \
    int j0, j1; key_range(nkeys, nsplit, tgid.x, j0, j1);                                                             \
    if (j0 >= j1) return;                                                                                             \
    const int h0 = kvh * REP;                                                                                         \
    device const uint* qq = (device const uint*)((device const char*)p.qq + ((ulong)t * n_h + h0) * HD);              \
    for (int i = tid; i < 8 * HD / 4; i += ANT) qs[i] = (i < REP * HD / 4) ? qq[i] : 0u;                              \
    float sqv[REP], mxv[REP];                                                                                         \
    for (int r = 0; r < REP; r++) { sqv[r] = ((device const float*)p.sq)[(ulong)t * n_h + h0 + r]; mxv[r] = ord2f(((device const uint*)p.mx)[(ulong)t * n_h + h0 + r]); } \
    threadgroup_barrier(mem_flags::mem_threadgroup);                                                                  \
    DET_ATTN_SCORES(HD)                                                                                               \
    device const char* kq = (device const char*)p.kq;                                                                 \
    device const float* sk = (device const float*)p.sk;                                                               \
    device const short* vq = (device const short*)p.vq;                                                               \
    const int d0 = (tid % TPK) * 4, kg = tid / TPK;                                                                   \
    long acc[REP][4];                                                                                                 \
    long dl[REP];                                                                                                     \
    for (int r = 0; r < REP; r++) { dl[r] = 0; for (int d = 0; d < 4; d++) acc[r][d] = 0; }                           \
    for (int j = j0; j < j1; j += ATK) {                                                                              \
        const int nvalid = min(ATK, j1 - j);                                                                          \
        auto tK = tensor((device int8_t*)(kq + ((ulong)j * n_kv + kvh) * HD), dextents<int32_t, 2>(HD, nvalid), array<int32_t, 2>{1, n_kv * HD}); \
        for (int i = 0; i < ATT_CAP; i++) if (i < (int)cT.get_capacity() && cT.is_valid_element(i)) cT[i] = 0;        \
        op.run(tQ, tK, cT);                                                                                           \
        threadgroup_barrier(mem_flags::mem_threadgroup);   /* previous tile's pqs consumed */                        \
        for (int i = 0; i < ATT_CAP; i++) {                                                                           \
            if (i < (int)cT.get_capacity() && cT.is_valid_element(i)) {                                               \
                auto idx = cT.get_multidimensional_index(i);                                                           \
                const int kk = idx[0], r = idx[1];                                                                    \
                if (r < REP) {                                                                                        \
                    int pv = 0;                                                                                       \
                    if (kk < nvalid) {                                                                                \
                        const float s = mul(mul((float)cT[i], sqv[r]), sk[(ulong)(j + kk) * n_kv + kvh]);             \
                        pv = rint_i(mul(exp_f(sub(s, mxv[r])), P_FIX));                                               \
                    }                                                                                                 \
                    pqs[r][kk] = pv;                                                                                  \
                    dl[r] += pv;                                                                                      \
                }                                                                                                     \
            }                                                                                                         \
        }                                                                                                             \
        threadgroup_barrier(mem_flags::mem_threadgroup);                                                              \
        for (int key = kg; key < nvalid; key += NKG) {                                                                \
            const int4 vv = int4(*(device const short4*)(vq + ((ulong)(j + key) * n_kv + kvh) * HD + d0));            \
            for (int r = 0; r < REP; r++) {                                                                           \
                const long pv = (long)pqs[r][key];                                                                    \
                acc[r][0] += pv * (long)vv.x; acc[r][1] += pv * (long)vv.y;                                           \
                acc[r][2] += pv * (long)vv.z; acc[r][3] += pv * (long)vv.w;                                           \
            }                                                                                                         \
        }                                                                                                             \
    }                                                                                                                 \
    threadgroup_barrier(mem_flags::mem_threadgroup);                                                                  \
    for (int kgi = 0; kgi < NKG; kgi++) {                                                                             \
        if (kg == kgi) for (int r = 0; r < REP; r++) for (int d = 0; d < 4; d++) {                                    \
            if (kgi == 0) accs[r * HD + d0 + d] = acc[r][d]; else accs[r * HD + d0 + d] += acc[r][d];                 \
        }                                                                                                              \
        threadgroup_barrier(mem_flags::mem_threadgroup);                                                              \
    }                                                                                                                 \
    for (int r = 0; r < REP; r++) {                                                                                   \
        long v = dl[r];                                                                                               \
        for (int o = 1; o < 32; o <<= 1) {                                                                            \
            uint lo = (uint)v, hi = (uint)((ulong)v >> 32);                                                           \
            lo = simd_shuffle_xor(lo, (ushort)o); hi = simd_shuffle_xor(hi, (ushort)o);                               \
            v += (long)(((ulong)hi << 32) | lo);                                                                      \
        }                                                                                                             \
        if ((tid & 31) == 0) dens[(tid >> 5) * REP + r] = v;                                                          \
    }                                                                                                                 \
    threadgroup_barrier(mem_flags::mem_threadgroup);                                                                  \
    if (nsplit == 1) {                                                                                                \
        const float vs = ((device const float*)p.v_scale)[kvh];                                                       \
        device float* o = (device float*)p.o;                                                                         \
        device const float* qg = (device const float*)p.qg;                                                           \
        for (int i = tid; i < REP * HD; i += ANT) {                                                                   \
            const int r = i / HD, dd = i - r * HD;                                                                    \
            const long den = dens[r] + dens[REP + r] + dens[2 * REP + r] + dens[3 * REP + r];                         \
            const ulong hrow = (ulong)t * n_h + h0 + r;                                                               \
            float ov = mul(div(i64_to_f(accs[i]), i64_to_f(den)), vs);                                                      \
            const float gate = qg[hrow * 2 * HD + HD + dd];                                                           \
            o[hrow * HD + dd] = mul(ov, sigmoid_f(gate));                                                             \
        }                                                                                                             \
    } else {                                                                                                          \
        device long* num = (device long*)p.num + (ulong)tgid.x * ((ulong)p.M * n_h * HD);                             \
        device long* den = (device long*)p.den + (ulong)tgid.x * ((ulong)p.M * n_h);                                  \
        for (int i = tid; i < REP * HD; i += ANT) {                                                                   \
            const int r = i / HD, dd = i - r * HD;                                                                    \
            num[((ulong)t * n_h + h0 + r) * HD + dd] = accs[i];                                                       \
        }                                                                                                             \
        if ((int)tid < REP) den[(ulong)t * n_h + h0 + tid] = dens[tid] + dens[REP + tid] + dens[2 * REP + tid] + dens[3 * REP + tid]; \
    }                                                                                                                 \
}

#define DET_ATTN_KERNELS(HD, REP) DET_ATTN_MAX(HD, REP) DET_ATTN_ACC(HD, REP)
DET_ATTN_KERNELS(256, 1) DET_ATTN_KERNELS(256, 2) DET_ATTN_KERNELS(256, 3) DET_ATTN_KERNELS(256, 4) DET_ATTN_KERNELS(256, 6) DET_ATTN_KERNELS(256, 8)
DET_ATTN_KERNELS(128, 1) DET_ATTN_KERNELS(128, 2) DET_ATTN_KERNELS(128, 4) DET_ATTN_KERNELS(128, 8)

// ---- prefill path (M >= 16): row-batched blocks, 128-key tiles, P.V on the tensor unit.  A threadgroup owns one
// kv head, a tile of PR = 16 rows = NTOK tokens x REP heads (t-major), one key split and (k_attn_pacc) one 128-dim
// half of the output; it streams its keys in tiles of 128:  scores = one 16 x 128 x HD int8 matmul (q tile in
// threadgroup memory, K rows straight from the cache); P.V: pq (<= 2^24) split into 4 slices of 7 bits (int8-safe,
// slices that are all zero in a tile are skipped: their products are zero), V as the (vl - 128, vh) int8 planes of the
// cache, 8 int8 matmuls (16 x 128 x 128) per tile whose int32 results are widened into per-thread int64 accumulators
// with the slice shifts; the vl offset is undone with 128 * den.  Every contraction is exact.  Blocks run heavy-first;
// the dim halves keep the accumulator registers low enough for several threadgroups per core (measured 1.15x).
#define PR 16
#define PTK 128
#define PCAP_S 16
#define PCAP_V 16
#define PDC 128      /* dims per P.V op */
#define DET_ATTN_PRE_SETUP(HD, REP)                                                                                   \
    const int n_h = p.n_h, n_kv = p.n_kv, M = p.M, pos0 = p.pos0;                                                     \
    const int kvh = (int)tgid.x % n_kv, dh = (int)tgid.x / n_kv;   /* dh: dim half (pacc only; pmax grid.x == n_kv) */ \
    constexpr int NTOK = PR / REP;                                                                                    \
    const int ntiles = (M + NTOK - 1) / NTOK;                                                                         \
    const int tile = ntiles - 1 - (int)tgid.y;            /* heavy first */                                           \
    const int t0 = tile * NTOK;                                                                                       \
    const int ntok = min(NTOK, M - t0);                                                                               \
    const int h0 = kvh * REP;                                                                                         \
    const int kmax = pos0 + t0 + ntok - 1;                /* last key of the block */                                 \
    int j0, j1; key_range_t(kmax + 1, p.nsplit, tgid.z, PTK, j0, j1);                                                 \
    for (int w = tid; w < PR * HD / 4; w += ANT) {                                                                    \
        const int i = w / (HD / 4), col = w % (HD / 4);                                                               \
        const int tok = i / REP, r = i % REP;                                                                         \
        uint v = 0u;                                                                                                  \
        if (i < NTOK * REP && tok < ntok) v = ((device const uint*)((device const char*)p.qq + ((ulong)(t0 + tok) * n_h + h0 + r) * HD))[col]; \
        qs[w] = v;                                                                                                    \
    }                                                                                                                 \
    if ((int)tid < PR) {   /* per-row scale, key limit (-1: padding row) and max, in threadgroup memory */           \
        const int i = tid, tok = i / REP, r = i % REP;                                                                \
        const bool ok = i < NTOK * REP && tok < ntok;                                                                 \
        sqv[i] = ok ? ((device const float*)p.sq)[(ulong)(t0 + tok) * n_h + h0 + r] : 0.0f;                           \
        lim[i] = ok ? (pos0 + t0 + tok) : -1;                                                                         \
        mxv[i] = ok ? ord2f(((device const uint*)p.mx)[(ulong)(t0 + tok) * n_h + h0 + r]) : 0.0f;                     \
    }                                                                                                                 \
    constexpr auto dS = matmul2d_descriptor(PR, PTK, HD, false, true, false);                                         \
    matmul2d<dS, execution_simdgroups<4>> opS;                                                                        \
    using TQ = tensor<threadgroup int8_t, dextents<int32_t, 2>, tensor_inline>;                                       \
    using TK = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>;                                            \
    auto cS = opS.get_destination_cooperative_tensor<TQ, TK, int32_t>();                                              \
    auto tQ = tensor((threadgroup int8_t*)qs, dextents<int32_t, 2>(HD, PR), array<int32_t, 2>{1, HD});               \
    device const char* kq = (device const char*)p.kq;                                                                 \
    device const float* sk = (device const float*)p.sk;                                                               \
    threadgroup_barrier(mem_flags::mem_threadgroup);

#define DET_ATTN_PMAX(HD, REP)                                                                                        \
kernel void k_attn_pmax_##HD##_##REP(constant det_p_attn& p [[buffer(0)]], uint3 tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) { \
    threadgroup uint qs[PR * HD / 4];                                                                                 \
    threadgroup float sqv[PR]; threadgroup int lim[PR]; threadgroup float mxv[PR];                                    \
    threadgroup atomic_uint bm[PR];                                                                                   \
    if ((int)tid < PR) atomic_store_explicit(&bm[tid], 0u, memory_order_relaxed);                                     \
    DET_ATTN_PRE_SETUP(HD, REP)                                                                                       \
    float lm[PR];                                                                                                     \
    for (int i = 0; i < PR; i++) lm[i] = -INFINITY;                                                                   \
    for (int j = j0; j < j1; j += PTK) {                                                                              \
        const int nvalid = min(PTK, j1 - j);                                                                          \
        auto tK = tensor((device int8_t*)(kq + ((ulong)j * n_kv + kvh) * HD), dextents<int32_t, 2>(HD, nvalid), array<int32_t, 2>{1, n_kv * HD}); \
        for (int e = 0; e < PCAP_S; e++) if (e < (int)cS.get_capacity() && cS.is_valid_element(e)) cS[e] = 0;        \
        opS.run(tQ, tK, cS);                                                                                          \
        for (int e = 0; e < PCAP_S; e++) {                                                                            \
            if (e < (int)cS.get_capacity() && cS.is_valid_element(e)) {                                               \
                auto idx = cS.get_multidimensional_index(e);                                                          \
                const int kk = idx[0], i = idx[1];                                                                    \
                if (kk < nvalid && j + kk <= lim[i]) {                                                                \
                    const float sc = mul(mul((float)cS[e], sqv[i]), sk[(ulong)(j + kk) * n_kv + kvh]);               \
                    lm[i] = fmax_(lm[i], sc);                                                                         \
                }                                                                                                     \
            }                                                                                                         \
        }                                                                                                             \
    }                                                                                                                 \
    for (int i = 0; i < PR; i++) atomic_fetch_max_explicit(&bm[i], f2ord(lm[i]), memory_order_relaxed);               \
    threadgroup_barrier(mem_flags::mem_threadgroup);                                                                  \
    if ((int)tid < PR && lim[tid] >= 0) {                                                                             \
        const int tok = tid / REP, r = tid % REP;                                                                     \
        atomic_fetch_max_explicit((device atomic_uint*)p.mx + (ulong)(t0 + tok) * n_h + h0 + r, atomic_load_explicit(&bm[tid], memory_order_relaxed), memory_order_relaxed); \
    }                                                                                                                 \
}

#define DET_ATTN_PACC(HD, REP)                                                                                        \
kernel void k_attn_pacc_##HD##_##REP(constant det_p_attn& p [[buffer(0)]], uint3 tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) { \
    threadgroup uint qs[PR * HD / 4];                                                                                 \
    threadgroup float sqv[PR]; threadgroup int lim[PR]; threadgroup float mxv[PR];                                    \
    threadgroup int8_t Ps[4][PR * PTK];                                                                               \
    threadgroup long dens[PR];                                                                                        \
    DET_ATTN_PRE_SETUP(HD, REP)                                                                                       \
    constexpr auto dV = matmul2d_descriptor(PR, PDC, PTK, false, false, false);                                        \
    matmul2d<dV, execution_simdgroups<4>> opV;                                                                        \
    using TP = tensor<threadgroup int8_t, dextents<int32_t, 2>, tensor_inline>;                                       \
    using TV = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>;                                            \
    auto cV = opV.get_destination_cooperative_tensor<TP, TV, int32_t>();                                              \
    device int8_t* vlo = (device int8_t*)p.vlo;                                                                       \
    device int8_t* vhi = (device int8_t*)p.vhi;                                                                       \
    threadgroup int nz[4];                                                                                            \
    long acc[1][PCAP_V];                                            /* this threadgroup's PDC dims only */          \
    for (int c = 0; c < 1; c++) for (int e = 0; e < PCAP_V; e++) acc[c][e] = 0;                                 \
    long den_part = 0;                                                                                                \
    const int drow = tid / 8, dpart = tid % 8;                                                                        \
    for (int j = j0; j < j1; j += PTK) {                                                                              \
        const int nvalid = min(PTK, j1 - j);                                                                          \
        auto tK = tensor((device int8_t*)(kq + ((ulong)j * n_kv + kvh) * HD), dextents<int32_t, 2>(HD, nvalid), array<int32_t, 2>{1, n_kv * HD}); \
        for (int e = 0; e < PCAP_S; e++) if (e < (int)cS.get_capacity() && cS.is_valid_element(e)) cS[e] = 0;        \
        opS.run(tQ, tK, cS);                                                                                          \
        threadgroup_barrier(mem_flags::mem_threadgroup);                                                              \
        if (tid < 4) nz[tid] = 0;                                                                                     \
        threadgroup_barrier(mem_flags::mem_threadgroup);                                                              \
        int pmax_t = 0;                                                                                               \
        for (int e = 0; e < PCAP_S; e++) {                                                                            \
            if (e < (int)cS.get_capacity() && cS.is_valid_element(e)) {                                               \
                auto idx = cS.get_multidimensional_index(e);                                                          \
                const int kk = idx[0], i = idx[1];                                                                    \
                int pv = 0;                                                                                           \
                if (kk < nvalid && j + kk <= lim[i]) {                                                                \
                    const float sc = mul(mul((float)cS[e], sqv[i]), sk[(ulong)(j + kk) * n_kv + kvh]);               \
                    pv = rint_i(mul(exp_f(sub(sc, mxv[i])), P_FIX));                                                 \
                }                                                                                                     \
                Ps[0][i * PTK + kk] = (int8_t)(pv & 127); Ps[1][i * PTK + kk] = (int8_t)((pv >> 7) & 127);            \
                Ps[2][i * PTK + kk] = (int8_t)((pv >> 14) & 127); Ps[3][i * PTK + kk] = (int8_t)((pv >> 21) & 127);   \
                pmax_t = max(pmax_t, pv);                                                                             \
            }                                                                                                         \
        }                                                                                                             \
        pmax_t = simd_max(pmax_t);                                                                                    \
        if ((tid & 31) == 0) { if (pmax_t >= (1 << 7)) nz[1] = 1; if (pmax_t >= (1 << 14)) nz[2] = 1; if (pmax_t >= (1 << 21)) nz[3] = 1; if (pmax_t > 0) nz[0] = 1; } \
        threadgroup_barrier(mem_flags::mem_threadgroup);                                                              \
        {                                                                                                             \
            long d = 0;                                                                                               \
            for (int kk = dpart * 16; kk < dpart * 16 + 16; kk++) {                                                   \
                const int o = drow * PTK + kk;                                                                        \
                d += (long)Ps[0][o] + ((long)Ps[1][o] << 7) + ((long)Ps[2][o] << 14) + ((long)Ps[3][o] << 21);        \
            }                                                                                                         \
            den_part += d;                                                                                            \
        }                                                                                                             \
        for (int c = 0; c < 1; c++) {                                                                                 \
            for (int a = 0; a < 4; a++) {                                                                             \
                if (!nz[a]) continue;              /* slice all zero in this tile: its products are zero */          \
                auto tP = tensor((threadgroup int8_t*)Ps[a], dextents<int32_t, 2>(PTK, PR), array<int32_t, 2>{1, PTK}); \
                for (int pl = 0; pl < 2; pl++) {                                                                      \
                    device int8_t* vb = (pl ? vhi : vlo) + ((ulong)j * n_kv + kvh) * HD + dh * PDC;                   \
                    auto tV = tensor(vb, dextents<int32_t, 2>(PDC, nvalid), array<int32_t, 2>{1, n_kv * HD});         \
                    for (int e = 0; e < PCAP_V; e++) if (e < (int)cV.get_capacity() && cV.is_valid_element(e)) cV[e] = 0; \
                    opV.run(tP, tV, cV);                                                                              \
                    const int sh = 7 * a + 8 * pl;                                                                    \
                    for (int e = 0; e < PCAP_V; e++) if (e < (int)cV.get_capacity() && cV.is_valid_element(e)) acc[c][e] += ((long)cV[e]) << sh; \
                }                                                                                                     \
            }                                                                                                         \
        }                                                                                                             \
    }                                                                                                                 \
    for (int o = 1; o < 8; o <<= 1) {                                                                                 \
        uint lo = (uint)den_part, hi = (uint)((ulong)den_part >> 32);                                                 \
        lo = simd_shuffle_xor(lo, (ushort)o); hi = simd_shuffle_xor(hi, (ushort)o);                                   \
        den_part += (long)(((ulong)hi << 32) | lo);                                                                   \
    }                                                                                                                 \
    if (dpart == 0) dens[drow] = den_part;                                                                            \
    threadgroup_barrier(mem_flags::mem_threadgroup);                                                                  \
    device float* o = (device float*)p.o;                                                                             \
    device const float* qg = (device const float*)p.qg;                                                               \
    const float vs = ((device const float*)p.v_scale)[kvh];                                                           \
    device long* num_p = (device long*)p.num + (ulong)tgid.z * ((ulong)M * n_h * HD);                                 \
    device long* den_p = (device long*)p.den + (ulong)tgid.z * ((ulong)M * n_h);                                      \
    for (int c = 0; c < 1; c++) {                                                                                     \
        for (int e = 0; e < PCAP_V; e++) {                                                                            \
            if (e < (int)cV.get_capacity() && cV.is_valid_element(e)) {                                               \
                auto idx = cV.get_multidimensional_index(e);                                                          \
                const int d = dh * PDC + idx[0], i = idx[1];                                                            \
                if (lim[i] < 0) continue;                                                                             \
                const int tok = i / REP, r = i % REP;                                                                 \
                const long den = dens[i];                                                                             \
                const long num = acc[c][e] + 128 * den;                                                               \
                const ulong hrow = (ulong)(t0 + tok) * n_h + h0 + r;                                                  \
                if (p.nsplit == 1) {                                                                                  \
                    const float ov = mul(div(i64_to_f(num), i64_to_f(den)), vs);                                            \
                    const float gate = qg[hrow * 2 * HD + HD + d];                                                    \
                    o[hrow * HD + d] = mul(ov, sigmoid_f(gate));                                                      \
                } else {                                                                                              \
                    num_p[hrow * HD + d] = num;                                                                       \
                    if (d == 0) den_p[hrow] = den;                                                                    \
                }                                                                                                     \
            }                                                                                                         \
        }                                                                                                             \
    }                                                                                                                 \
}
#define DET_ATTN_PRE_KERNELS(HD, REP) DET_ATTN_PMAX(HD, REP) DET_ATTN_PACC(HD, REP)
DET_ATTN_PRE_KERNELS(256, 1) DET_ATTN_PRE_KERNELS(256, 2) DET_ATTN_PRE_KERNELS(256, 3) DET_ATTN_PRE_KERNELS(256, 4) DET_ATTN_PRE_KERNELS(256, 6) DET_ATTN_PRE_KERNELS(256, 8)
DET_ATTN_PRE_KERNELS(128, 1) DET_ATTN_PRE_KERNELS(128, 2) DET_ATTN_PRE_KERNELS(128, 4) DET_ATTN_PRE_KERNELS(128, 8)

// ---- finalize (split case): sum the split partials, o = ((float)num / (float)den) * v_scale * sigmoid(gate).
// grid M*n_h threadgroups, HD threads.  Splits beyond the key count wrote nothing; only splits with j0 < nkeys count.
#define DET_ATTN_FIN(HD)                                                                                              \
kernel void k_attn_fin_##HD(constant det_p_attn& p [[buffer(0)]], uint row [[threadgroup_position_in_grid]], uint d [[thread_index_in_threadgroup]]) { \
    const int n_h = p.n_h, rep = n_h / p.n_kv, h = row % n_h, t = row / n_h;                                          \
    const int nkeys = p.pos0 + t + 1;                                                                                 \
    const ulong stride_num = (ulong)p.M * n_h * HD, stride_den = (ulong)p.M * n_h;                                    \
    long num = 0, den = 0;                                                                                            \
    for (int s = 0; s < p.nsplit; s++) {                                                                              \
        int j0, j1; key_range_t(nkeys, p.nsplit, s, p.pre ? PTK : ATK, j0, j1);                                       \
        if (j0 >= j1) break;                                                                                          \
        num += ((device const long*)p.num)[s * stride_num + (ulong)row * HD + d];                                     \
        if (d == 0) den += ((device const long*)p.den)[s * stride_den + row];                                         \
    }                                                                                                                 \
    threadgroup long dsh;                                                                                             \
    if (d == 0) dsh = den;                                                                                            \
    threadgroup_barrier(mem_flags::mem_threadgroup);                                                                  \
    den = dsh;                                                                                                        \
    float ov = mul(div(i64_to_f(num), i64_to_f(den)), ((device const float*)p.v_scale)[h / rep]);                           \
    const float gate = ((device const float*)p.qg)[(ulong)row * 2 * HD + HD + d];                                     \
    ((device float*)p.o)[(ulong)row * HD + d] = mul(ov, sigmoid_f(gate));                                             \
}
DET_ATTN_FIN(256) DET_ATTN_FIN(128)
