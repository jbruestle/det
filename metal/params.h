// Kernel parameter structs shared between the host (.mm) and the shaders (.metal): plain C, uint64_t GPU addresses.
#ifndef __METAL_VERSION__
#pragma once
#endif
#ifndef __METAL_VERSION__
#include <stdint.h>
#endif
typedef struct { uint64_t x, w, y, xq, sx, mx; float eps; int n; } det_p_norm;
typedef struct { uint64_t a, b, h, q, sx, mx; int n; } det_p_gquant;
typedef struct { uint64_t tab, ids, x; int H; } det_p_embed;
typedef struct { uint64_t x, y; uint64_t n; } det_p_add;
typedef struct { uint64_t lg, out; int V; } det_p_argmax;
typedef struct { uint64_t dst; uint32_t value; uint64_t n; } det_p_fill;
typedef struct { uint64_t dst, src; uint64_t n; } det_p_copy;
typedef struct { uint64_t x, out; int op, n; } det_p_prim;
typedef struct { uint64_t wq, c, m, xq, sx, mx, y, part, counters; uint64_t ydst[4]; int yoff[4]; int M, N, K, G, bpr, splits, nparts, accumulate; } det_p_gemm;
typedef struct { uint64_t qkv, conv_state, conv_w, a, b, A, dt_bias, y, beta, decay; float eps; int t0, conv_dim, kd, n_v; } det_p_gdn_prep;
typedef struct { uint64_t y, beta, decay, S, o; int M, conv_dim, kd, n_v, rep; } det_p_gdn_recur;
typedef struct { uint64_t o, z, norm, on, qkv, conv_state; float eps; int nrows, nblk_out, M, conv_dim; } det_p_gdn_out;
typedef struct { uint64_t qg, k, v, q_norm, k_norm, v_scale, v_inv_scale, rope_cos, rope_sin, qq, sq, mx, kq, sk, vq, o, num, den;
                 float eps, qscale; int rot, M, pos0, n_h, n_kv, nsplit; } det_p_attn;
typedef struct { uint64_t logits, keys, mx, hist, part, state, z, out; float inv_temp; int V, top_k, digit, nblk; uint64_t p24; int use_k, use_p; } det_p_smp;
typedef struct { uint64_t src, dst; int N, off, n, T; } det_p_copycols;
typedef struct { uint64_t src, dst[4]; int off[4], n[4]; int N, T, nparts; } det_p_splitcols;
