// det: C API for the CUDA implementation.  All pointers are DEVICE pointers unless noted.
// Row-major layouts, fp32 unless noted.  Every function is bit-exact w.r.t. detref/ (numpy spec).
#pragma once
#include <stdint.h>
#include <cuda_runtime.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    const uint8_t* wq;   // packed codes (see detref/quant.py pack), row-major [N][K/32*bytes_per_32]
    const float*   c;    // [N]
    const uint16_t* m;   // [N][K/128]
    int N, K, bits;      // bits in {4,6,8}
} det_qlinear;

// ---- glue (norm.cu)
// Activation quantization (detmath.quant_act_i8g): per row sx[M] fp32 and, if mx != NULL, per-(row, 128-group)
// multipliers mx[M][n/128] uint16 (n must be a multiple of 128); if mx == NULL, plain per-row int8 (quant_act_i8).
// x[M][n] -> y[M][n] = rmsnorm(x,w,eps) (y may be NULL) and, if xq != NULL, quantize y into xq[M][n], sx, mx
void det_rmsnorm_quant(const float* x, const float* w, float eps, int M, int n, float* y, int8_t* xq, float* sx, uint16_t* mx, cudaStream_t st);
void det_quant(const float* x, int M, int n, int8_t* xq, float* sx, uint16_t* mx, cudaStream_t st);
void det_embed(const uint16_t* table_bf16, const int* ids, int M, int H, float* x, cudaStream_t st);
void det_add(float* x, const float* y, long long n, cudaStream_t st);                          // x += y
void det_silu_mul_quant(const float* g, const float* u, int M, int n, float* h, int8_t* hq, float* sh, uint16_t* mh, cudaStream_t st); // h = silu(g)*u (h may be NULL) + quant
void det_argmax(const float* logits, int M, int V, int* out, cudaStream_t st);                 // lowest index on ties

// ---- GEMM (gemm.cu):  y[M][N] = ((float)acc * c[n]) * sx[m],  acc = sum_g (m[n,g]*mx[m,g]) * sum_{k in g} wq[n,k]*xq[m,k]
//      mx[M][K/128] uint16 (NULL = all ones).  acc exact in int64.
void det_gemm(const det_qlinear* L, const int8_t* xq, const float* sx, const uint16_t* mx, int M, float* y, cudaStream_t st);
// Scratch for the W4/W6 M > 64 path (per-(token, 128-group) activation sums): a device buffer of at least
// det_gemm_scratch_bytes(M, K) bytes for the largest M and K that will be used, set once with det_gemm_set_scratch
// (the host keeps it alive; NULL unsets).  Subsequent det_gemm calls use it (stream-ordered, like any scratch).
// If none is set (or it is too small), a lazily allocated global buffer is used instead (one-time note on stderr).
size_t det_gemm_scratch_bytes(int M, int K);
void det_gemm_set_scratch(void* p, size_t bytes);

// ---- full attention (attn.cu).  KV cache layout: kq[pos][n_kv][hd] int8, sk[pos][n_kv] f32, vq[pos][n_kv][hd] int16
typedef struct {
    const float* q_norm;      // [hd]  (1+w)
    const float* k_norm;      // [hd]
    const float* v_scale;     // [n_kv]
    const float* v_inv_scale; // [n_kv]
} det_attn_layer;
typedef struct { int8_t* kq; float* sk; int16_t* vq; int capacity; } det_kv_cache;
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

// ---- deterministic sampling (sample.cu): see detref/sample.py.  z[M] = raw 64-bit RNG draws (splitmix64 on host)
void det_sample(const float* logits, int M, int V, float inv_temp, const unsigned long long* z, int* out, cudaStream_t st);
// ---- top-k / top-p sampling (sample.cu, same spec): 0 < top_k < V and/or top_p < 1 enable the filters (else == det_sample).
//      scratch: device buffer of at least det_sample_scratch_bytes(M, V) (sort buffers + CUB temp storage)
void det_sample_topkp(const float* logits, int M, int V, float inv_temp, int top_k, float top_p, const unsigned long long* z, int* out, void* scratch, cudaStream_t st);
size_t det_sample_scratch_bytes(int M, int V);

#ifdef __cplusplus
}
#endif
