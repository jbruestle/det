// glue kernels: rmsnorm(+quant), quant, embed, add, silu*mul(+quant), argmax, fill/copy, primitive test hooks
using namespace det;

// One row per threadgroup of NT threads; each thread owns CHUNK contiguous elements (n_pad = NT*CHUNK, zero padded).
#define DET_RMSNORM_KERNEL(NAME, NT, CHUNK, NORM)                                                                     \
kernel void NAME(constant det_p_norm& p [[buffer(0)]], uint row [[threadgroup_position_in_grid]],                    \
                 uint t [[thread_index_in_threadgroup]]) {                                                            \
    threadgroup float smem[32];                                                                                       \
    device const float* x = (device const float*)p.x;                                                                 \
    device const float* w = (device const float*)p.w;                                                                 \
    device float* y = (device float*)p.y;                                                                             \
    device char* xq = (device char*)p.xq;                                                                             \
    device float* sx = (device float*)p.sx;                                                                           \
    device ushort* mx = (device ushort*)p.mx;                                                                         \
    const int n = p.n;                                                                                                \
    device const float* xr = x + (ulong)row * n;                                                                      \
    float v[CHUNK];                                                                                                   \
    DET_U_GLUE for (int i = 0; i < CHUNK; i++) { int idx = t * CHUNK + i; v[i] = idx < n ? xr[idx] : 0.0f; }                     \
    if (NORM) {                                                                                                       \
        float sq[CHUNK];                                                                                              \
        DET_U_GLUE for (int i = 0; i < CHUNK; i++) sq[i] = mul(v[i], v[i]);                                                      \
        float ss = warp_tree(tree_local<CHUNK>(sq));                                                                  \
        if (NT > 32) ss = block_tree<(NT > 32 ? NT / 32 : 1)>(ss, smem, t);                                           \
        float mean = div(ss, (float)n);                                                                               \
        float inv = div(1.0f, sqrt_(add(mean, p.eps)));                                                               \
        DET_U_GLUE for (int i = 0; i < CHUNK; i++) { int idx = t * CHUNK + i; v[i] = idx < n ? mul(mul(v[i], inv), w[idx]) : 0.0f; } \
        if (p.y) DET_U_GLUE for (int i = 0; i < CHUNK; i++) { int idx = t * CHUNK + i; if (idx < n) y[(ulong)row * n + idx] = v[i]; } \
    }                                                                                                                 \
    if (p.xq) {                                                                                                       \
        float am = 0.0f;                                                                                              \
        DET_U_GLUE for (int i = 0; i < CHUNK; i++) am = fmax_(am, fabs(v[i]));                                                     \
        if (p.mx) {                                                                                                   \
            if (CHUNK >= 4) {                                                                                         \
                const int LPG = 128 / (CHUNK >= 4 ? CHUNK : 4);                                                       \
                float gm = am;                                                                                        \
                DET_U_GLUE for (int o = 1; o < LPG; o <<= 1) gm = fmax_(gm, simd_shuffle_xor(gm, (ushort)o));                      \
                float sg = div(gm, 127.0f);                                                                           \
                float smax = (NT > 32) ? block_max<(NT > 32 ? NT / 32 : 1)>(sg, smem, t) : warp_max(sg);              \
                float s = is_zero(smax) ? 1.0f : div(smax, 65535.0f);                                                  \
                int mxi = rint_i(div(sg, s));                                                                         \
                mxi = mxi < 1 ? 1 : (mxi > 65535 ? 65535 : mxi);                                                      \
                float inv = div(1.0f, mul(s, (float)mxi));                                                            \
                DET_U_GLUE for (int i = 0; i < CHUNK; i++) { int idx = t * CHUNK + i; if (idx < n) xq[(ulong)row * n + idx] = act_q(v[i], inv); } \
                if ((t % LPG) == 0 && (int)(t / LPG) < n / 128) mx[(ulong)row * (n / 128) + t / LPG] = (ushort)mxi;  \
                if (t == 0) sx[row] = s;                                                                              \
            }                                                                                                         \
        } else {                                                                                                      \
            am = (NT > 32) ? block_max<(NT > 32 ? NT / 32 : 1)>(am, smem, t) : warp_max(am);                          \
            float s = act_scale(am);                                                                                  \
            float inv = div(1.0f, s);                                                                                 \
            DET_U_GLUE for (int i = 0; i < CHUNK; i++) { int idx = t * CHUNK + i; if (idx < n) xq[(ulong)row * n + idx] = act_q(v[i], inv); } \
            if (t == 0) sx[row] = s;                                                                                  \
        }                                                                                                             \
    }                                                                                                                 \
}
DET_RMSNORM_KERNEL(k_norm_32_1_1, 32, 1, 1)     DET_RMSNORM_KERNEL(k_norm_32_1_0, 32, 1, 0)
DET_RMSNORM_KERNEL(k_norm_32_2_1, 32, 2, 1)     DET_RMSNORM_KERNEL(k_norm_32_2_0, 32, 2, 0)
DET_RMSNORM_KERNEL(k_norm_32_4_1, 32, 4, 1)     DET_RMSNORM_KERNEL(k_norm_32_4_0, 32, 4, 0)
DET_RMSNORM_KERNEL(k_norm_32_8_1, 32, 8, 1)     DET_RMSNORM_KERNEL(k_norm_32_8_0, 32, 8, 0)
DET_RMSNORM_KERNEL(k_norm_32_16_1, 32, 16, 1)   DET_RMSNORM_KERNEL(k_norm_32_16_0, 32, 16, 0)
DET_RMSNORM_KERNEL(k_norm_256_4_1, 256, 4, 1)   DET_RMSNORM_KERNEL(k_norm_256_4_0, 256, 4, 0)
DET_RMSNORM_KERNEL(k_norm_512_4_1, 512, 4, 1)   DET_RMSNORM_KERNEL(k_norm_512_4_0, 512, 4, 0)
DET_RMSNORM_KERNEL(k_norm_1024_4_1, 1024, 4, 1) DET_RMSNORM_KERNEL(k_norm_1024_4_0, 1024, 4, 0)
DET_RMSNORM_KERNEL(k_norm_1024_8_1, 1024, 8, 1) DET_RMSNORM_KERNEL(k_norm_1024_8_0, 1024, 8, 0)
DET_RMSNORM_KERNEL(k_norm_128_16_1, 128, 16, 1)   DET_RMSNORM_KERNEL(k_norm_128_16_0, 128, 16, 0)
DET_RMSNORM_KERNEL(k_norm_256_16_1, 256, 16, 1)   DET_RMSNORM_KERNEL(k_norm_256_16_0, 256, 16, 0)
DET_RMSNORM_KERNEL(k_norm_256_32_1, 256, 32, 1)   DET_RMSNORM_KERNEL(k_norm_256_32_0, 256, 32, 0)

// Generic per-(row, 128-group) quantization, one simdgroup per group (4 contiguous elements per lane), NT = 1024.
// MODE 0: v = a;  MODE 1: v = silu(a) * b.  Optionally stores v to h.  Each simdgroup owns groups g = w, w+32, ... (<= MAXG).
#define DET_GQUANT_KERNEL(NAME, MODE, MAXG)                                                                           \
kernel void NAME(constant det_p_gquant& p [[buffer(0)]], uint row [[threadgroup_position_in_grid]],                  \
                 uint tid [[thread_index_in_threadgroup]]) {                                                          \
    threadgroup float sg_s[1024];                                                                                     \
    device const float* a = (device const float*)p.a;                                                                 \
    device const float* b = (device const float*)p.b;                                                                 \
    device float* h = (device float*)p.h;                                                                             \
    device char* q = (device char*)p.q;                                                                               \
    device float* sx = (device float*)p.sx;                                                                           \
    device ushort* mx = (device ushort*)p.mx;                                                                         \
    const int n = p.n, G = n / 128, w = tid >> 5, lane = tid & 31;                                                    \
    const ulong off = (ulong)row * n;                                                                                 \
    float4 v[MAXG];                                                                                                   \
    DET_U_GLUE for (int j = 0; j < MAXG; j++) {                                                                                  \
        const int g = w + j * 32;                                                                                     \
        if (g < G) {                                                                                                  \
            const int idx = g * 128 + lane * 4;                                                                       \
            float4 av = *(device const float4*)(a + off + idx);                                                       \
            if (MODE == 1) {                                                                                          \
                float4 bv = *(device const float4*)(b + off + idx);                                                   \
                v[j] = float4(mul(silu_f(av.x), bv.x), mul(silu_f(av.y), bv.y), mul(silu_f(av.z), bv.z), mul(silu_f(av.w), bv.w)); \
            } else v[j] = av;                                                                                         \
            if (p.h) *(device float4*)(h + off + idx) = v[j];                                                         \
            float am = fmax_(fmax_(fabs(v[j].x), fabs(v[j].y)), fmax_(fabs(v[j].z), fabs(v[j].w)));                         \
            am = warp_max(am);                                                                                        \
            if (lane == 0) sg_s[g] = div(am, 127.0f);                                                                 \
        }                                                                                                             \
    }                                                                                                                 \
    threadgroup_barrier(mem_flags::mem_threadgroup);                                                                  \
    float smax = 0.0f;                                                                                                \
    for (int g = 0; g < G; g++) smax = fmax_(smax, sg_s[g]);                                                            \
    const float s = is_zero(smax) ? 1.0f : div(smax, 65535.0f);                                                        \
    DET_U_GLUE for (int j = 0; j < MAXG; j++) {                                                                                  \
        const int g = w + j * 32;                                                                                     \
        if (g < G) {                                                                                                  \
            const int idx = g * 128 + lane * 4;                                                                       \
            int mxi = rint_i(div(sg_s[g], s));                                                                        \
            mxi = mxi < 1 ? 1 : (mxi > 65535 ? 65535 : mxi);                                                          \
            float inv = div(1.0f, mul(s, (float)mxi));                                                                \
            char4 c4 = char4(act_q(v[j].x, inv), act_q(v[j].y, inv), act_q(v[j].z, inv), act_q(v[j].w, inv));         \
            *(device char4*)(q + off + idx) = c4;                                                                     \
            if (lane == 0) mx[(ulong)row * G + g] = (ushort)mxi;                                                      \
        }                                                                                                             \
    }                                                                                                                 \
    if (tid == 0) sx[row] = s;                                                                                        \
}
DET_GQUANT_KERNEL(k_gquant_0_1, 0, 1)  DET_GQUANT_KERNEL(k_gquant_0_2, 0, 2)  DET_GQUANT_KERNEL(k_gquant_0_4, 0, 4)
DET_GQUANT_KERNEL(k_gquant_0_8, 0, 8)  DET_GQUANT_KERNEL(k_gquant_0_32, 0, 32)
DET_GQUANT_KERNEL(k_gquant_1_1, 1, 1)  DET_GQUANT_KERNEL(k_gquant_1_2, 1, 2)  DET_GQUANT_KERNEL(k_gquant_1_4, 1, 4)
DET_GQUANT_KERNEL(k_gquant_1_8, 1, 8)  DET_GQUANT_KERNEL(k_gquant_1_32, 1, 32)

// generic per-row quant for any n (two passes), 256 threads
kernel void k_quant(constant det_p_norm& p [[buffer(0)]], uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float smem[32];
    device const float* xr = (device const float*)p.x + (ulong)row * p.n;
    device char* xq = (device char*)p.xq;
    device float* sx = (device float*)p.sx;
    const int n = p.n;
    float am = 0.0f;
    for (int i = tid; i < n; i += 256) am = fmax_(am, fabs(xr[i]));
    am = block_max<8>(am, smem, tid);
    float s = act_scale(am);
    float inv = div(1.0f, s);
    for (int i = tid; i < n; i += 256) xq[(ulong)row * n + i] = act_q(xr[i], inv);
    if (tid == 0) sx[row] = s;
}

kernel void k_embed(constant det_p_embed& p [[buffer(0)]], uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    device const ushort* tab = (device const ushort*)p.tab;
    device const int* ids = (device const int*)p.ids;
    device float* x = (device float*)p.x;
    device const ushort* src = tab + (ulong)ids[row] * p.H;
    for (int i = tid; i < p.H; i += 256) x[(ulong)row * p.H + i] = as_type<float>(((uint)src[i]) << 16);
}

kernel void k_add(constant det_p_add& p [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    device float* x = (device float*)p.x;
    device const float* y = (device const float*)p.y;
    if ((ulong)i < p.n) x[i] = add(x[i], y[i]);
}

kernel void k_silu_mul_quant(constant det_p_gquant& p [[buffer(0)]], uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float smem[32];
    device const float* g = (device const float*)p.a;
    device const float* u = (device const float*)p.b;
    device float* h = (device float*)p.h;
    device char* hq = (device char*)p.q;
    device float* sh = (device float*)p.sx;
    const int n = p.n;
    const ulong off = (ulong)row * n;
    float am = 0.0f;
    for (int i = tid; i < n; i += 256) {
        float v = mul(silu_f(g[off + i]), u[off + i]);
        if (p.h) h[off + i] = v;
        am = fmax_(am, fabs(v));
    }
    am = block_max<8>(am, smem, tid);
    float s = act_scale(am);
    float inv = div(1.0f, s);
    for (int i = tid; i < n; i += 256) {
        float v = mul(silu_f(g[off + i]), u[off + i]);
        hq[off + i] = act_q(v, inv);
    }
    if (tid == 0) sh[row] = s;
}

kernel void k_argmax(constant det_p_argmax& p [[buffer(0)]], uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float sv[256]; threadgroup int si[256];
    device const float* r = (device const float*)p.lg + (ulong)row * p.V;
    device int* out = (device int*)p.out;
    float best = -INFINITY; int bi = 0x7fffffff;
    for (int i = tid; i < p.V; i += 256) { float v = r[i]; if (lt(best, v) || (v == best && i < bi)) { best = v; bi = i; } }
    sv[tid] = best; si[tid] = bi;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int s = 128; s > 0; s >>= 1) {
        if ((int)tid < s) {
            float ov = sv[tid + s]; int oi = si[tid + s];
            if (lt(sv[tid], ov) || (ov == sv[tid] && oi < si[tid])) { sv[tid] = ov; si[tid] = oi; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) out[row] = si[0];
}

kernel void k_fill_u32(constant det_p_fill& p [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    device uint* d = (device uint*)p.dst;
    for (ulong j = (ulong)i * 8; j < (ulong)i * 8 + 8 && j < p.n; j++) d[j] = p.value;
}
kernel void k_copy_u32(constant det_p_copy& p [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    device uint* d = (device uint*)p.dst;
    device const uint* s = (device const uint*)p.src;
    for (ulong j = (ulong)i * 4; j < (ulong)i * 4 + 4 && j < p.n; j++) d[j] = s[j];
}

// ---- primitive test hooks (det_test_prim)
kernel void k_prim(constant det_p_prim& p [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    device const float* x = (device const float*)p.x;
    device float* out = (device float*)p.out;
    if ((int)i >= p.n) return;
    float v = x[i], r = 0.0f;
    switch (p.op) {
        case 0: r = exp_f(v); break;
        case 1: r = log_f(v); break;
        case 2: r = softplus_f(v); break;
        case 3: r = sigmoid_f(v); break;
        case 4: r = silu_f(v); break;
        case 5: { if (i & 1) return; ulong lo = as_type<uint>(x[i]), hi = as_type<uint>(x[i + 1]); r = i64_to_f((long)((hi << 32) | lo)); break; }
        case 6: r = rint_(v); break;
        case 7: r = div(1.0f, v); break;
        case 8: r = sqrt_(v); break;
        case 9: r = mul(v, x[p.n - 1 - i]); break;
        case 10: r = add(v, x[p.n - 1 - i]); break;
        case 12: { if (i & 1) return; ulong lo = as_type<uint>(x[i]), hi = as_type<uint>(x[i + 1]); r = (float)((long)((hi << 32) | lo)); break; }   // native conversion
        case 13: r = (float)rint_i(v); break;
        case 14: r = div(v, x[p.n - 1 - i]); break;
        case 15: r = sub(v, x[p.n - 1 - i]); break;
        case 16: r = fmax_(v, x[p.n - 1 - i]); break;
        default: r = v;
    }
    out[i] = r;
}
// treesum of rows of n (n <= 8192): 1024 threads, 8 per thread, zero padded
kernel void k_prim_treesum(constant det_p_prim& p [[buffer(0)]], uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float smem[32];
    device const float* x = (device const float*)p.x + (ulong)row * p.n;
    float v[8];
    DET_U_GLUE for (int i = 0; i < 8; i++) { int idx = tid * 8 + i; v[i] = idx < p.n ? x[idx] : 0.0f; }
    // general n: choose the padded size by how many threads are busy -> use the full 8192 tree (same result: zero padding)
    float s = warp_tree(tree_local<8>(v));
    s = block_tree<32>(s, smem, tid);
    if (tid == 0) ((device float*)p.out)[row] = s;
}

// dst[t][0..n) = src[t][off..off+n)  (src rows of N floats); grid (ceil(n/256), T)
kernel void k_copy_cols(constant det_p_copycols& p [[buffer(0)]], uint2 id [[thread_position_in_grid]]) {
    if ((int)id.x >= p.n) return;
    ((device float*)p.dst)[(ulong)id.y * p.n + id.x] = ((device const float*)p.src)[(ulong)id.y * p.N + p.off + id.x];
}

// split the column blocks of a fused GEMM output into up to 4 part buffers in one dispatch:
// dst[i][t][0..n_i) = src[t][off_i .. off_i + n_i); grid (ceil(sum n_i / 256), T)
kernel void k_split_cols(constant det_p_splitcols& p [[buffer(0)]], uint2 id [[thread_position_in_grid]]) {
    int x = id.x;
    for (int i = 0; i < p.nparts; i++) {
        if (x < p.n[i]) {
            ((device float*)p.dst[i])[(ulong)id.y * p.n[i] + x] = ((device const float*)p.src)[(ulong)id.y * p.N + p.off[i] + x];
            return;
        }
        x -= p.n[i];
    }
}
