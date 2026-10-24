// det: C API of the CUDA implementation (bit-exact w.r.t. the det spec, crates/det/src/spec).
//   * every pointer is a DEVICE pointer (det_malloc); the host moves data with det_memcpy;
//   * cudaStream_t: 0 (the legacy default stream) is the one in-order stream the Rust backend uses; det_sync waits
//     for the whole device.
// Row-major layouts, fp32 unless noted.
#pragma once
#include <stdint.h>
#include <stddef.h>
#include <cuda_runtime.h>
#ifdef __cplusplus
extern "C" {
#endif

// ---- runtime (runtime.cu)
int         det_init(void);                 // selects the current device; returns 0 on success (called lazily otherwise)
const char* det_device_name(void);
void*       det_malloc(size_t bytes);       // zero-initialised device buffer; NULL on failure
void        det_free(void* p);              // waits for pending work first
size_t      det_allocated_bytes(void);
void        det_sync(void);                 // wait for all pending GPU work (aborts on a kernel error)
void        det_flush(void);                // no-op (launches are already submitted)
void        det_memset(void* p, int value, size_t bytes, cudaStream_t st);
void        det_memcpy(void* dst, const void* src, size_t bytes, cudaStream_t st);   // any direction (cudaMemcpyDefault)
void*       det_cuda_stream(void);          // the stream det's work runs on, for code that enqueues its own kernels after it

// ---- weights: a quantized linear y = W x, W[N][K] (spec §2)
typedef struct {
    const uint8_t* wq;   // packed codes, row-major [N][K/32*bytes_per_32]
    const float*   c;    // [N]
    const uint16_t* m;   // [N][K/128]
    int N, K, bits;      // bits in {4,6,8}
} det_qlinear;

// ---- glue (norm.cu)
// Activation quantization (spec quant_act_i8g): per row sx[M] fp32 and, if mx != NULL, per-(row, 128-group)
// multipliers mx[M][n/128] uint16 (n must be a multiple of 128); if mx == NULL, plain per-row int8 (quant_act_i8).
// x[M][n] -> y[M][n] = rmsnorm(x,w,eps) (y may be NULL) and, if xq != NULL, quantize y into xq[M][n], sx, mx
void det_rmsnorm_quant(const float* x, const float* w, float eps, int M, int n, float* y, int8_t* xq, float* sx, uint16_t* mx, cudaStream_t st);
void det_quant(const float* x, int M, int n, int8_t* xq, float* sx, uint16_t* mx, cudaStream_t st);
void det_embed(const uint16_t* table_bf16, const int* ids, int M, int H, float* x, cudaStream_t st);
void det_add(float* x, const float* y, long long n, cudaStream_t st);                          // x += y
void det_silu_mul_quant(const float* g, const float* u, int M, int n, float* h, int8_t* hq, float* sh, uint16_t* mh, cudaStream_t st); // h = silu(g)*u (h may be NULL) + quant
void det_argmax(const float* logits, int M, int V, int* out, cudaStream_t st);                 // lowest index on ties

// ---- GEMM (gemm.cu):  y[M][N] = ((float)acc * c[n]) * sx[m],  acc = sum_g (m[n,g]*mx[m,g]) * sum_{k in g} wq[n,k]*xq[m,k]
//      mx[M][K/128] uint16 (NULL = all ones).  acc exact in int64.  accumulate != 0: y += result.  nparts > 1: the
//      output columns are routed into nparts buffers (part i = columns [offs[i], offs[i+1]), offs[0] == 0, dsts[i] is
//      [M][width_i]; y unused).
void det_gemm(const det_qlinear* L, const int8_t* xq, const float* sx, const uint16_t* mx, int M, float* y, cudaStream_t st);
void det_gemm_ex(const det_qlinear* L, const int8_t* xq, const float* sx, const uint16_t* mx, int M, float* y, int accumulate,
                 int nparts, const int* offs, float* const* dsts, cudaStream_t st);
// Scratch for the W4/W6 M > 64 path (per-(token, 128-group) activation sums): a device buffer of at least
// det_gemm_scratch_bytes(M, K) bytes for the largest M and K in use, set once with det_gemm_set_scratch (the host keeps
// it alive; NULL unsets).  Without one (or with a too small one) a lazily allocated global buffer is used instead
// (one-time note on stderr).
size_t det_gemm_scratch_bytes(int M, int K);
void det_gemm_set_scratch(void* p, size_t bytes);

// ---- full attention (attn.cu).  KV cache layout: kq[pos][n_kv][hd] int8, sk[pos][n_kv] f32, vq[pos][n_kv][hd] int16
typedef struct {
    const float* q_norm;      // [hd]  (1+w)
    const float* k_norm;      // [hd]
    const float* v_scale;     // [n_kv]
    const float* v_inv_scale; // [n_kv]
} det_attn_layer;
// vlo / vhi exist for signature compatibility with the Metal runtime (which keeps V as two int8 planes); unused here.
typedef struct { int8_t* kq; float* sk; int16_t* vq; int capacity; int8_t* vlo; int8_t* vhi; } det_kv_cache;
// Inputs for M new tokens at positions pos0..pos0+M-1:
//   qg[M][n_h*2*hd] (q_lin output: per head [q(hd) | gate(hd)]), k[M][n_kv*hd], v[M][n_kv*hd],
//   rope_cos/sin[pos][rot/2].  Writes KV cache rows, computes attention for every new token against all
//   positions <= its own, multiplies by sigmoid(gate), and writes o[M][n_h*hd].
//   scratch: device buffer of at least det_attn_scratch_bytes(M, n_h, hd)
void det_attention(const det_attn_layer* L, const det_kv_cache* kv, const float* qg, const float* k, const float* v,
                   const float* rope_cos, const float* rope_sin, int rot, float eps,
                   int M, int pos0, int n_h, int n_kv, int hd, float* o, void* scratch, cudaStream_t st);
size_t det_attn_scratch_bytes(int M, int n_h, int hd);
// Graph-capturable variant (decode): same as det_attention, but the kernels read pos0 from device memory (*pos0_dev:
// cache row index, key count and rope row) and the launch is sized for any pos0 <= max_pos (max_pos + M <= capacity);
// blocks whose key range lies beyond the real pos0 exit immediately.  Bit-identical to det_attention (integer sums
// and the max are order-free).  Always the CUDA-core path (intended for M == 1); scratch as for det_attention(M).
void det_attention_dev(const det_attn_layer* L, const det_kv_cache* kv, const float* qg, const float* k, const float* v,
                       const float* rope_cos, const float* rope_sin, int rot, float eps,
                       int M, const int* pos0_dev, int max_pos, int n_h, int n_kv, int hd, float* o, void* scratch, cudaStream_t st);

// ---- gated delta net (gdn.cu)
typedef struct {
    const float* conv;     // [conv_dim][4]
    const float* A;        // [n_v]  (= exp(A_log))
    const float* dt_bias;  // [n_v]
    const float* norm;     // [dv]  (used as-is)
} det_gdn_layer;
typedef struct { float* conv_state; /* [3][conv_dim] oldest first */ float* S; /* [n_v][dk][dv] */ } det_gdn_state;
// qkv[M][conv_dim] (conv_dim = 2*n_k*dk + n_v*dv), z[M][n_v*dv], a[M][n_v], b[M][n_v]
// Runs conv+silu+l2norm (sequential conv state), the recurrence over the M tokens in order (updating S),
// gated rmsnorm with silu(z), and writes on[M][n_v*dv] (fp32).  scratch >= det_gdn_scratch_bytes(M, ...)
void det_gdn(const det_gdn_layer* L, det_gdn_state* S, const float* qkv, const float* z, const float* a, const float* b,
             float eps, int M, int n_k, int n_v, int dk, int dv, float* on, void* scratch, cudaStream_t st);
size_t det_gdn_scratch_bytes(int M, int n_k, int n_v, int dk, int dv);
// det_gdn that also exports the mixer's per-token inputs and outputs:
//   gin  [M][n_k][256 + 130*rep] fp32 per k head: q[128] | k[128] | v[rep][128] | beta[rep] | decay[rep]  (rep = n_v/n_k;
//        q, k after conv/silu/l2norm, q scaled by QSCALE)
//   gout [M][n_v][dv] fp32: the mixer output o before the gated norm
// Either pointer may be NULL.
void det_gdn_ex(const det_gdn_layer* L, det_gdn_state* S, const float* qkv, const float* z, const float* a, const float* b,
                float eps, int M, int n_k, int n_v, int dk, int dv, float* on, void* scratch, float* gin, float* gout, cudaStream_t st);

// ---- deterministic sampling (sample.cu): see spec/sample.rs.  z[M] = raw 64-bit RNG draws (splitmix64 on host)
void det_sample(const float* logits, int M, int V, float inv_temp, const unsigned long long* z, int* out, cudaStream_t st);
// ---- top-k / top-p sampling (sample.cu, same spec): 0 < top_k < V and/or top_p < 1 enable the filters (else == det_sample).
//      scratch: device buffer of at least det_sample_scratch_bytes(M, V) (sort buffers + CUB temp storage)
void det_sample_topkp(const float* logits, int M, int V, float inv_temp, int top_k, float top_p, const unsigned long long* z, int* out, void* scratch, cudaStream_t st);
size_t det_sample_scratch_bytes(int M, int V);

// ---- test hooks (runtime.cu): elementwise fp32 primitives on arrays, for bit-exactness tests vs the spec
// op: 0 exp_f, 1 log_f, 2 softplus_f, 3 sigmoid_f, 4 silu_f, 5 i64->f32 of (long long) bits of x pairs (x[2i],x[2i+1]),
//     6 rint, 7 1/x, 8 sqrt, 9 x*y (y = x reversed), 10 x+y, 11 treesum of each row of n (out[M]), 12 native i64->f32,
//     13 rint as int, 14 x/y, 15 x-y, 16 max(x,y)
void det_test_prim(int op, const float* x, int M, int n, float* out, cudaStream_t st);

#ifdef __cplusplus
}
#endif
