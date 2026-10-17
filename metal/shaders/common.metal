// det: shared device primitives for the Metal kernels.  These mirror detref/detmath.py EXACTLY (same op sequence).
// Rules: the library is compiled with MTLMathModeSafe (no fast math, no FMA contraction, precise div/sqrt), no
// library transcendentals, denormals preserved (the Apple GPU keeps fp32 denormals; never write denormal literals,
// the compiler folds those to zero).  This file is prepended to every other shader file (one translation unit).
#include <metal_stdlib>
using namespace metal;

#define DET_SIMD 32   // simdgroup width on Apple GPUs (threadExecutionWidth)
// loop-unrolling switches (experiments via DET_METAL_DEFINES, e.g. "DET_U_TREE_ON=1"): full unrolling of the small
// register-array loops; on Apple GPUs blanket unrolling increased code size and made several kernels slower
#define DET_UNROLL_FULL _Pragma("clang loop unroll(full)")
#ifndef DET_U_TREE_ON
#define DET_U_TREE_ON 1    // measured: faster norms / GDN trees
#endif
#ifndef DET_U_GLUE_ON
#define DET_U_GLUE_ON 0
#endif
#ifndef DET_U_GDN_ON
#define DET_U_GDN_ON 0
#endif
#ifndef DET_U_ATTN_ON
#define DET_U_ATTN_ON 0
#endif
#ifndef DET_U_GEMM_ON
#define DET_U_GEMM_ON 0
#endif
#if DET_U_TREE_ON
#define DET_U_TREE DET_UNROLL_FULL
#else
#define DET_U_TREE
#endif
#if DET_U_GLUE_ON
#define DET_U_GLUE DET_UNROLL_FULL
#else
#define DET_U_GLUE
#endif
#if DET_U_GDN_ON
#define DET_U_GDN DET_UNROLL_FULL
#else
#define DET_U_GDN
#endif
#if DET_U_ATTN_ON
#define DET_U_ATTN DET_UNROLL_FULL
#else
#define DET_U_ATTN
#endif
#if DET_U_GEMM_ON
#define DET_U_GEMM DET_UNROLL_FULL
#else
#define DET_U_GEMM
#endif

namespace det {

// ---- IEEE fp32 arithmetic with denormals.  The Apple GPU flushes denormal inputs and outputs to zero in every fp32
// arithmetic op (add/sub/mul/div/sqrt/fma, and compares them as zero), which the spec forbids.  Each primitive below
// takes the hardware result when it is provably identical to the IEEE one (no denormal operand, result not in the
// zero / denormal / minimum-normal range) and otherwise recomputes it exactly with integer arithmetic.
// DET_SOFT_DENORM=0 (benchmarking only) uses the raw hardware ops.
#ifndef DET_SOFT_DENORM
#define DET_SOFT_DENORM 1
#endif

#define DET_EXP_MASK 0x7f800000u
#define DET_ABS_MASK 0x7fffffffu

// value = P * 2^E (P != 0) rounded to nearest even into fp32 (denormals, overflow to inf)
inline float make_float(bool neg, ulong P, int E) {
    if (P == 0ul) return as_type<float>(neg ? 0x80000000u : 0u);
    const uint hi = (uint)(P >> 32), lo = (uint)P;
    const int msb = hi ? (63 - clz(hi)) : (31 - clz(lo));
    const int L = msb + E;
    const int g = max(L - 23, -149);
    const int sh = g - E;
    ulong q;
    if (sh <= 0) q = P << (uint)(-sh);
    else if (sh > 64) q = 0ul;
    else if (sh == 64) q = (P > 0x8000000000000000ul) ? 1ul : 0ul;
    else {
        q = P >> (uint)sh;
        const ulong rem = P & ((1ul << (uint)sh) - 1ul), hlf = 1ul << (uint)(sh - 1);
        if (rem > hlf || (rem == hlf && (q & 1ul))) q += 1ul;
    }
    int gg = g;
    if (q >= (1ul << 24)) { q >>= 1; gg += 1; }
    uint bits;
    if (q >= (1ul << 23)) {
        const int be = gg + 23 + 127;
        bits = be >= 255 ? DET_EXP_MASK : (((uint)be << 23) | ((uint)q & 0x7fffffu));
    } else bits = (uint)q;                      // denormal (gg == -149)
    return as_type<float>(bits | (neg ? 0x80000000u : 0u));
}
// x = (-1)^neg * m * 2^e exactly (x finite)
inline void decompose(float x, thread bool& neg, thread ulong& m, thread int& e) {
    const uint u = as_type<uint>(x);
    neg = (u >> 31) != 0u;
    const uint ex = (u >> 23) & 0xFFu;
    if (ex == 0u) { m = u & 0x7fffffu; e = -149; } else { m = (u & 0x7fffffu) | 0x800000u; e = (int)ex - 150; }
}
inline bool is_zero_bits(uint u) { return (u & DET_ABS_MASK) == 0u; }

inline float mul_slow(float a, float b, float hw) {
    const uint ua = as_type<uint>(a), ub = as_type<uint>(b);
    if (is_zero_bits(ua) || is_zero_bits(ub) || (ua & DET_EXP_MASK) == DET_EXP_MASK || (ub & DET_EXP_MASK) == DET_EXP_MASK) return hw;
    bool na, nb; ulong ma, mb; int ea, eb;
    decompose(a, na, ma, ea); decompose(b, nb, mb, eb);
    return make_float(na != nb, ma * mb, ea + eb);
}
inline float add_slow(float a, float b, float hw) {
    const uint ua = as_type<uint>(a), ub = as_type<uint>(b);
    if ((ua & DET_EXP_MASK) == DET_EXP_MASK || (ub & DET_EXP_MASK) == DET_EXP_MASK) return hw;
    if (is_zero_bits(ua)) return is_zero_bits(ub) ? hw : b;
    if (is_zero_bits(ub)) return a;
    // a tiny addend (|x| < 2^-126) cannot change a value >= 2^-90 under round-to-nearest
    const uint aa = ua & DET_ABS_MASK, ab = ub & DET_ABS_MASK;
    const uint lim = (uint)(-90 + 127) << 23;        // 2^-90
    if (aa >= lim) return ((ab & DET_EXP_MASK) == 0u) ? a : hw;
    if (ab >= lim) return ((aa & DET_EXP_MASK) == 0u) ? b : hw;
    // both |x| < 2^-90: exact fixed point in units of 2^-149 (|m << (e + 149)| < 2^24 * 2^36)
    bool na, nb; ulong ma, mb; int ea, eb;
    decompose(a, na, ma, ea); decompose(b, nb, mb, eb);
    long va = (long)(ma << (uint)(ea + 149)), vb = (long)(mb << (uint)(eb + 149));
    if (na) va = -va;
    if (nb) vb = -vb;
    const long v = va + vb;
    if (v == 0) return 0.0f;                          // exact cancellation: +0 under round-to-nearest
    const bool neg = v < 0;
    return make_float(neg, (ulong)(neg ? -v : v), -149);
}
inline float div_slow(float a, float b, float hw) {
    const uint ua = as_type<uint>(a), ub = as_type<uint>(b);
    if (is_zero_bits(ua) || (ua & DET_EXP_MASK) == DET_EXP_MASK || (ub & DET_EXP_MASK) == DET_EXP_MASK) return hw;
    if (is_zero_bits(ub)) return as_type<float>(DET_EXP_MASK | ((ua ^ ub) & 0x80000000u));
    bool na, nb; ulong ma, mb; int ea, eb;
    decompose(a, na, ma, ea); decompose(b, nb, mb, eb);
    while (ma < 0x800000ul) { ma <<= 1; ea -= 1; }     // normalise to 24 bits
    const ulong num = ma << 38;                        // < 2^62
    const ulong q = num / mb, rem = num - q * mb;      // q in [2^37, 2^39)
    return make_float(na != nb, (q << 1) | (rem != 0ul ? 1ul : 0ul), ea - eb - 39);
}
inline float sqrt_slow(float a) {                      // a denormal (> 0): sqrt(m * 2^-149) = sqrt(2m) * 2^-75
    const uint m = as_type<uint>(a) & 0x7fffffu;
    return sqrt((float)(2u * m)) * as_type<float>((uint)(-75 + 127) << 23);
}

#if DET_SOFT_DENORM
inline float mul(float a, float b) {
    const float r = a * b;
    const uint ua = as_type<uint>(a), ub = as_type<uint>(b), ur = as_type<uint>(r);
    if (((ua & DET_EXP_MASK) != 0u) & ((ub & DET_EXP_MASK) != 0u) & ((ur & 0x7f000000u) != 0u)) return r;
    return mul_slow(a, b, r);
}
inline float add(float a, float b) {
    const float r = a + b;
    const uint ua = as_type<uint>(a), ub = as_type<uint>(b), ur = as_type<uint>(r);
    if (((ua & DET_EXP_MASK) != 0u) & ((ub & DET_EXP_MASK) != 0u) & ((ur & 0x7f000000u) != 0u)) return r;
    return add_slow(a, b, r);
}
inline float sub(float a, float b) { return add(a, as_type<float>(as_type<uint>(b) ^ 0x80000000u)); }
inline float div(float a, float b) {
    const float r = a / b;
    const uint ua = as_type<uint>(a), ub = as_type<uint>(b), ur = as_type<uint>(r);
    if (((ua & DET_EXP_MASK) != 0u) & ((ub & DET_EXP_MASK) != 0u) & ((ur & 0x7f000000u) != 0u)) return r;
    return div_slow(a, b, r);
}
inline float sqrt_(float a) {
    const uint ua = as_type<uint>(a);
    if (((ua & DET_EXP_MASK) != 0u) | is_zero_bits(ua) | ((ua >> 31) != 0u)) return sqrt(a);
    return sqrt_slow(a);
}
#else
inline float mul(float a, float b) { return a * b; }
inline float add(float a, float b) { return a + b; }
inline float sub(float a, float b) { return a - b; }
inline float div(float a, float b) { return a / b; }
inline float sqrt_(float a) { return sqrt(a); }
#endif
// exact comparisons / max (the hardware compares denormals as zero); -0 < +0 here (never relevant: used on |x| and scores)
inline int ord_key(float a) { const int i = as_type<int>(a); return i < 0 ? (i ^ 0x7fffffff) : i; }
inline bool lt(float a, float b) { return ord_key(a) < ord_key(b); }
inline float fmax_(float a, float b) { return lt(a, b) ? b : a; }
inline bool is_zero(float a) { return is_zero_bits(as_type<uint>(a)); }
inline float rint_(float a) { return rint(a); }
inline int   rint_i(float a) { return (int)rint(a); }

// int64 -> fp32, round to nearest even (explicit, independent of the hardware conversion)
inline float i64_to_f(long v) {
    if (v == 0) return 0.0f;
    const bool neg = v < 0;
    ulong a = neg ? (ulong)(-v) : (ulong)v;
    if (a < (1ul << 24)) { float f = (float)(uint)a; return neg ? -f : f; }
    const uint hi = (uint)(a >> 32), lo = (uint)a;
    const int msb = hi ? (63 - clz(hi)) : (31 - clz(lo));
    const int shift = msb - 23;
    ulong mant = a >> shift;
    const ulong rem = a & ((1ul << shift) - 1ul), hlf = 1ul << (shift - 1);
    if (rem > hlf || (rem == hlf && (mant & 1ul))) mant += 1ul;
    float f = (float)(uint)mant * as_type<float>((uint)(shift + 127) << 23);
    return neg ? -f : f;
}

// Raw hardware ops (flush denormals).  Used only inside exp_f / log_f / softplus_f / sigmoid_f, whose results are
// provably flush-invariant: every intermediate that can be denormal is added to a quantity >= 2^-30 (the polynomial
// constants, r, 1.0) or squared to zero, so flushing it changes nothing; the final results are normal or 0.
inline float hmul(float a, float b) { return a * b; }
inline float hadd(float a, float b) { return a + b; }
inline float hsub(float a, float b) { return a - b; }

// ---- exp_f (Cephes), see detmath.exp_f
inline float exp_f(float x) {
    x = min(x, 88.0f);
    const bool small = x < -87.0f;
    float n = floor(hadd(hmul(x, 1.44269504088896341f), 0.5f));
    float r = hsub(x, hmul(n, 0.693359375f));
    r = hsub(r, hmul(n, -2.12194440e-4f));
    float z = hmul(r, r);
    float y = 1.9875691500E-4f;
    y = hadd(hmul(y, r), 1.3981999507E-3f);
    y = hadd(hmul(y, r), 8.3334519073E-3f);
    y = hadd(hmul(y, r), 4.1665795894E-2f);
    y = hadd(hmul(y, r), 1.6666665459E-1f);
    y = hadd(hmul(y, r), 5.0000001201E-1f);
    y = hadd(hmul(y, z), r);
    y = hadd(y, 1.0f);
    int ni = (int)n;
    float scale = as_type<float>((uint)((ni + 127) << 23));
    y = hmul(y, scale);
    return small ? 0.0f : y;
}

// ---- log_f (Cephes), x > 0 normal
inline float log_f(float x) {
    uint b = as_type<uint>(x);
    int e = (int)((b >> 23) & 0xFFu) - 126;
    float m = as_type<float>((b & 0x007FFFFFu) | 0x3F000000u);
    const bool lt = m < 0.707106781186547524f;
    if (lt) { e -= 1; m = hsub(hadd(m, m), 1.0f); } else { m = hsub(m, 1.0f); }
    float z = hmul(m, m);
    float y = 7.0376836292E-2f;
    y = hadd(hmul(y, m), -1.1514610310E-1f);
    y = hadd(hmul(y, m), 1.1676998740E-1f);
    y = hadd(hmul(y, m), -1.2420140846E-1f);
    y = hadd(hmul(y, m), 1.4249322787E-1f);
    y = hadd(hmul(y, m), -1.6668057665E-1f);
    y = hadd(hmul(y, m), 2.0000714765E-1f);
    y = hadd(hmul(y, m), -2.4999993993E-1f);
    y = hadd(hmul(y, m), 3.3333331174E-1f);
    y = hmul(y, m);
    y = hmul(y, z);
    float ef = (float)e;
    y = hadd(y, hmul(-2.12194440e-4f, ef));
    y = hadd(y, hmul(-0.5f, z));
    float r = hadd(m, y);
    r = hadd(r, hmul(0.693359375f, ef));
    return r;
}

inline float softplus_f(float x) {
    if (x > 20.0f) return x;
    if (x < -15.0f) return exp_f(x);
    return log_f(hadd(1.0f, exp_f(x)));
}
inline float sigmoid_f(float x) {
    if (x < -80.0f) return exp_f(x);
    return 1.0f / hadd(1.0f, exp_f(-x));
}
inline float silu_f(float x) { return mul(x, sigmoid_f(x)); }

// ---- treesum helpers.  The spec tree over a power-of-two array: s[i] = s[2i] + s[2i+1] per level.
// A thread holding an aligned contiguous chunk of 2^k values reduces it locally with tree_local<2^k>; then a simdgroup
// butterfly with ASCENDING xor offsets continues the same tree across lanes; then across simdgroups.
template <int N>
inline float tree_local(thread const float* v) {
    float t[N];
    DET_U_TREE for (int i = 0; i < N; i++) t[i] = v[i];
    DET_U_TREE for (int s = 1; s < N; s <<= 1)
        for (int i = 0; i < N; i += 2 * s) t[i] = add(t[i], t[i + s]);
    return t[0];
}
inline float warp_tree(float v) {
    DET_U_TREE for (int o = 1; o < DET_SIMD; o <<= 1) v = add(v, simd_shuffle_xor(v, (ushort)o));
    return v;
}
// Block tree over NWARPS simdgroup partials (NWARPS power of two <= 32).  All threads get the result.
template <int NWARPS>
inline float block_tree(float warp_val, threadgroup float* smem, uint tid) {
    const uint lane = tid & 31, w = tid >> 5;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) smem[w] = warp_val;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float t[NWARPS];
    DET_U_TREE for (int i = 0; i < NWARPS; i++) t[i] = smem[i];
    DET_U_TREE for (int s = 1; s < NWARPS; s <<= 1)
        for (int i = 0; i < NWARPS; i += 2 * s) t[i] = add(t[i], t[i + s]);
    return t[0];
}
inline float warp_max(float v) {
    DET_U_TREE for (int o = 16; o > 0; o >>= 1) v = fmax_(v, simd_shuffle_xor(v, (ushort)o));
    return v;
}
template <int NWARPS>
inline float block_max(float v, threadgroup float* smem, uint tid) {
    const uint lane = tid & 31, w = tid >> 5;
    v = warp_max(v);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) smem[w] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float r = smem[0];
    DET_U_TREE for (int i = 1; i < NWARPS; i++) r = fmax_(r, smem[i]);
    return r;
}

// ---- activation quantization helpers (spec: s = amax/127, s=1 if amax==0; inv = 1/s; q = clamp(rint(x*inv)))
inline float act_scale(float amax) { return is_zero(amax) ? 1.0f : div(amax, 127.0f); }
inline char act_q(float x, float inv) {
    int q = rint_i(mul(x, inv));
    q = q > 127 ? 127 : (q < -127 ? -127 : q);
    return (char)q;
}

} // namespace det
