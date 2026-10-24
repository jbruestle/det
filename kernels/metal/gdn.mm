// host side of det_gdn (see shaders/gdn.metal)
#include <stdio.h>
#include <stdlib.h>
#include "det_metal.h"
#include "runtime.h"
#include "params.h"
using namespace det;

namespace {
constexpr int HD = 128;
constexpr int CHUNK = 2048;
inline size_t align256(size_t x) { return (x + 255) & ~(size_t)255; }
}

extern "C" {

size_t det_gdn_scratch_bytes(int M, int n_k, int n_v, int dk, int dv) {
    const int conv_dim = 2 * n_k * dk + n_v * dv;
    const size_t m = (size_t)(M < CHUNK ? M : CHUNK);
    return align256(m * conv_dim * 4) + 2 * align256(m * n_v * 4) + align256(m * (size_t)n_v * dv * 4) + 256;
}

void det_gdn(const det_gdn_layer* L, det_gdn_state* ST, const float* qkv, const float* z, const float* a, const float* b,
             float eps, int M, int n_k, int n_v, int dk, int dv, float* on, void* scratch, det_stream_t st) {
    det_gdn_ex(L, ST, qkv, z, a, b, eps, M, n_k, n_v, dk, dv, on, scratch, nullptr, nullptr, st);
}

void det_gdn_ex(const det_gdn_layer* L, det_gdn_state* ST, const float* qkv, const float* z, const float* a, const float* b,
                float eps, int M, int n_k, int n_v, int dk, int dv, float* on, void* scratch, float* gin, float* gout, det_stream_t) {
    if (dk != HD || dv != HD || n_v % n_k != 0) fail("det_gdn: unsupported shape dk=%d dv=%d n_k=%d n_v=%d", dk, dv, n_k, n_v);
    if (n_v > 128) fail("det_gdn: n_v=%d > 128", n_v);
    if (M <= 0) return;
    const int conv_dim = 2 * n_k * dk + n_v * dv;
    const int kd = n_k * dk;
    const int rep = n_v / n_k;
    const size_t m = (size_t)(M < CHUNK ? M : CHUNK);
    char* sp = (char*)align256((size_t)scratch);
    float* y = (float*)sp;            sp += align256(m * conv_dim * 4);
    float* beta = (float*)sp;         sp += align256(m * n_v * 4);
    float* decay = (float*)sp;        sp += align256(m * n_v * 4);
    float* o_scratch = (float*)sp;
    for (int t0 = 0; t0 < M; t0 += CHUNK) {
        const int mc = (M - t0 < CHUNK) ? (M - t0) : CHUNK;
        const bool last = (t0 + mc == M);
        float* o = gout ? gout + (size_t)t0 * n_v * dv : o_scratch;
        det_p_gdn_prep pp = { gpu(qkv), gpu(ST->conv_state), gpu(L->conv), gpu(a), gpu(b), gpu(L->A), gpu(L->dt_bias),
                              gpu(y), gpu(beta), gpu(decay), gpu_or_zero(gin), eps, t0, conv_dim, kd, n_v, rep };
        dispatch("k_gdn_prep", &pp, sizeof pp, MTLSizeMake((conv_dim + 511) / 512, mc, 1), MTLSizeMake(128, 1, 1));
        det_p_gdn_recur pr = { gpu(y), gpu(beta), gpu(decay), gpu(ST->S), gpu(o), mc, conv_dim, kd, n_v, rep };
        dispatch("k_gdn_recur", &pr, sizeof pr, MTLSizeMake(n_v * (HD / DET_GDN_JB), 1, 1), MTLSizeMake(DET_GDN_JB * DET_GDN_P, 1, 1));
        const int nrows = mc * n_v;
        const int nblk_out = (nrows + 3) / 4;
        const int nblk_conv = last ? (conv_dim + 127) / 128 : 0;
        det_p_gdn_out po = { gpu(o), gpu(z + (size_t)t0 * n_v * dv), gpu(L->norm), gpu(on + (size_t)t0 * n_v * dv), gpu(qkv), gpu(ST->conv_state),
                             eps, nrows, nblk_out, M, conv_dim };
        dispatch("k_gdn_out", &po, sizeof po, MTLSizeMake(nblk_out + nblk_conv, 1, 1), MTLSizeMake(128, 1, 1));
    }
}

}
