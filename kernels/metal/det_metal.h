// det: C API of the Metal implementation (bit-exact w.r.t. the det spec, crates/det/src/spec).
//   * every pointer is a HOST pointer into a buffer obtained from det_malloc (unified memory, storage mode shared):
//     the host reads/writes it directly (after det_sync), the kernels get the matching GPU address;
//   * det_stream_t is accepted for signature compatibility and ignored: there is one in-order command queue.
//     Work is encoded into the current command buffer; det_sync() commits it and waits, det_flush() commits only.
// The shaders are compiled from source at first use (embedded at build time; DET_METAL_SHADERS=<dir> overrides).
#pragma once
#include <stdint.h>
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef void* det_stream_t;

// ---- runtime
int         det_init(void);                 // loads the device + shaders; returns 0 on success (called lazily otherwise)
const char* det_device_name(void);
void*       det_malloc(size_t bytes);       // zero-initialised shared buffer (host pointer); NULL on failure
void        det_free(void* p);
size_t      det_allocated_bytes(void);
void        det_sync(void);                 // commit pending GPU work and wait for it
void        det_flush(void);                // commit pending GPU work (no wait)
void        det_memset(void* p, int value, size_t bytes, det_stream_t st);   // GPU-side fill (stream ordered)
void        det_memcpy(void* dst, const void* src, size_t bytes, det_stream_t st);   // GPU-side copy (stream ordered)
// Handles for code that enqueues its own kernels after det's (the id<MTLDevice> and id<MTLCommandQueue>; det's pending
// work must be flushed first so the queue order holds), and the MTLBuffer + byte offset behind a det_malloc pointer.
void*       det_metal_device(void);
void*       det_metal_queue(void);
void*       det_metal_buffer(const void* p, size_t* offset);

// ---- weights: a quantized linear y = W x, W[N][K] (spec §2): codes packed per 32 weights, in the Metal TILED layout
// [ceil(N/64)][K/128][64][4*bytes_per_32] (rows >= N zero; crates/det/src/metal.rs builds it from the row-major rows),
// c[N] fp32 per row, m[N][K/128] uint16 per group.
typedef struct {
    const uint8_t* wq;
    const float*   c;
    const uint16_t* m;
    int N, K, bits;      // bits in {4,6,8}
} det_qlinear;

// ---- glue (glue.metal)
void det_rmsnorm_quant(const float* x, const float* w, float eps, int M, int n, float* y, int8_t* xq, float* sx, uint16_t* mx, det_stream_t st);
void det_quant(const float* x, int M, int n, int8_t* xq, float* sx, uint16_t* mx, det_stream_t st);
void det_embed(const uint16_t* table_bf16, const int* ids, int M, int H, float* x, det_stream_t st);
void det_add(float* x, const float* y, long long n, det_stream_t st);
void det_silu_mul_quant(const float* g, const float* u, int M, int n, float* h, int8_t* hq, float* sh, uint16_t* mh, det_stream_t st);
void det_argmax(const float* logits, int M, int V, int* out, det_stream_t st);
void det_copy_cols(const float* src, int N, int off, int n, int T, float* dst, det_stream_t st);
void det_split_cols(const float* src, int N, int T, int nparts, const int* offs, const int* ns, float* const* dsts, det_stream_t st);

// ---- GEMM (gemm.metal):  y[M][N] = ((float)acc * c[n]) * sx[m],  acc = sum_g (m[n,g]*mx[m,g]) * sum_{k in g} wq[n,k]*xq[m,k]
//      mx[M][K/128] uint16 (NULL = all ones).  acc exact in int64.  accumulate != 0: y += result.  nparts > 1: the
//      output columns are routed into nparts buffers (part i = columns [offs[i], offs[i+1]), offs[0] == 0, dsts[i] is
//      [M][width_i]; y unused).
void det_gemm(const det_qlinear* L, const int8_t* xq, const float* sx, const uint16_t* mx, int M, float* y, det_stream_t st);
void det_gemm_ex(const det_qlinear* L, const int8_t* xq, const float* sx, const uint16_t* mx, int M, float* y, int accumulate,
                 int nparts, const int* offs, float* const* dsts, det_stream_t st);
// Scratch of det_gemm_scratch_bytes(M <= 8, largest N) bytes enables split-K in the decode kernel; set once (NULL unsets).
size_t det_gemm_scratch_bytes(int M, int N);
void det_gemm_set_scratch(void* p, size_t bytes);

// ---- full attention (attn.metal).  KV cache layout: kq[pos][n_kv][hd] int8, sk[pos][n_kv] f32, vq[pos][n_kv][hd] int16,
// plus V as two int8 planes vlo = (vq & 255) - 128 and vhi = vq >> 8 for the prefill path.
typedef struct {
    const float* q_norm; const float* k_norm; const float* v_scale; const float* v_inv_scale;
} det_attn_layer;
typedef struct { int8_t* kq; float* sk; int16_t* vq; int capacity; int8_t* vlo; int8_t* vhi; } det_kv_cache;
void det_attention(const det_attn_layer* L, const det_kv_cache* kv, const float* qg, const float* k, const float* v,
                   const float* rope_cos, const float* rope_sin, int rot, float eps,
                   int M, int pos0, int n_h, int n_kv, int hd, float* o, void* scratch, det_stream_t st);
size_t det_attn_scratch_bytes(int M, int n_h, int hd);

// ---- gated delta net (gdn.metal).  v head h uses k head h / (n_v / n_k).
typedef struct { const float* conv; const float* A; const float* dt_bias; const float* norm; } det_gdn_layer;
typedef struct { float* conv_state; float* S; } det_gdn_state;
void det_gdn(const det_gdn_layer* L, det_gdn_state* S, const float* qkv, const float* z, const float* a, const float* b,
             float eps, int M, int n_k, int n_v, int dk, int dv, float* on, void* scratch, det_stream_t st);
size_t det_gdn_scratch_bytes(int M, int n_k, int n_v, int dk, int dv);
// det_gdn that also exports the mixer's per-token inputs and outputs:
//   gin  [M][n_k][256 + 130*rep] fp32 per k head: q[128] | k[128] | v[rep][128] | beta[rep] | decay[rep]  (rep = n_v/n_k;
//        q, k after conv/silu/l2norm, q scaled by QSCALE)
//   gout [M][n_v][dv] fp32: the mixer output o before the gated norm
// Either pointer may be NULL.
void det_gdn_ex(const det_gdn_layer* L, det_gdn_state* S, const float* qkv, const float* z, const float* a, const float* b,
                float eps, int M, int n_k, int n_v, int dk, int dv, float* on, void* scratch, float* gin, float* gout, det_stream_t st);

// ---- deterministic sampling (sample.metal): see the spec.  z[M] = raw 64-bit RNG draws (splitmix64 on the host)
void det_sample(const float* logits, int M, int V, float inv_temp, const unsigned long long* z, int* out, det_stream_t st);
void det_sample_topkp(const float* logits, int M, int V, float inv_temp, int top_k, float top_p, const unsigned long long* z, int* out, void* scratch, det_stream_t st);
size_t det_sample_scratch_bytes(int M, int V);

// ---- test hooks (glue.metal): elementwise fp32 primitives on arrays, for bit-exactness tests vs the spec
// op: 0 exp_f, 1 log_f, 2 softplus_f, 3 sigmoid_f, 4 silu_f, 5 i64->f32 of (long long) bits of x pairs (x[2i],x[2i+1]),
//     6 rint, 7 1/x, 8 sqrt, 9 x*y (y = x reversed), 10 x+y, 11 treesum of each row of n (out[M]), 12 native i64->f32,
//     13 rint as int, 14 x/y, 15 x-y, 16 max(x,y)
void det_test_prim(int op, const float* x, int M, int n, float* out, det_stream_t st);
#ifdef __cplusplus
}
#endif
