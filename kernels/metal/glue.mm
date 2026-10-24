// host side of the glue kernels
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include "det_metal.h"
#include "runtime.h"
#include "params.h"
using namespace det;

static void launch_norm(bool norm, const float* x, const float* w, float eps, int M, int n, float* y, int8_t* xq, float* sx, uint16_t* mx) {
    if (M <= 0) return;
    int np = 32; while (np < n) np <<= 1;
    if (mx && (n % 128)) fail("det_rmsnorm/quant: group mode needs n %% 128 == 0 (n=%d)", n);
    int NT, CHUNK;
    // M <= 8 (decode): 512/1024-thread threadgroups (lowest latency); larger M: 128/256 threads x 16/32 elements (throughput)
    switch (np) {
        case 32: NT = 32; CHUNK = 1; break;   case 64: NT = 32; CHUNK = 2; break;    case 128: NT = 32; CHUNK = 4; break;
        case 256: NT = 32; CHUNK = 8; break;  case 512: NT = 32; CHUNK = 16; break;  case 1024: NT = 256; CHUNK = 4; break;
        case 2048: NT = 128; CHUNK = 16; break; case 4096: NT = 256; CHUNK = 16; break; case 8192: NT = 256; CHUNK = 32; break;
        default: fail("det_rmsnorm: unsupported n=%d", n);
    }
    if (M <= 8) { if (np == 2048) { NT = 512; CHUNK = 4; } else if (np == 4096) { NT = 1024; CHUNK = 4; } else if (np == 8192) { NT = 1024; CHUNK = 8; } }
    if (mx && CHUNK < 4) fail("det_rmsnorm/quant: group mode needs n >= 128 (n=%d)", n);
    std::string name = "k_norm_" + std::to_string(NT) + "_" + std::to_string(CHUNK) + "_" + (norm ? "1" : "0");
    det_p_norm p = { gpu(x), gpu_or_zero(w), gpu_or_zero(y), gpu_or_zero(xq), gpu_or_zero(sx), gpu_or_zero(mx), eps, n };
    dispatch(name.c_str(), &p, sizeof p, MTLSizeMake(M, 1, 1), MTLSizeMake(NT, 1, 1));
}

static void launch_group_quant(int mode, const float* a, const float* b, int M, int n, float* h, int8_t* q, float* sx, uint16_t* mx) {
    if (M <= 0) return;
    const int G = n / 128;
    if (n % 128 || G > 1024) fail("det group quant: bad n=%d", n);
    int maxg = G <= 32 ? 1 : G <= 64 ? 2 : G <= 128 ? 4 : G <= 256 ? 8 : 32;
    std::string name = "k_gquant_" + std::to_string(mode) + "_" + std::to_string(maxg);
    det_p_gquant p = { gpu(a), gpu_or_zero(b), gpu_or_zero(h), gpu(q), gpu(sx), gpu(mx), n };
    dispatch(name.c_str(), &p, sizeof p, MTLSizeMake(M, 1, 1), MTLSizeMake(1024, 1, 1));
}

extern "C" {

void det_rmsnorm_quant(const float* x, const float* w, float eps, int M, int n, float* y, int8_t* xq, float* sx, uint16_t* mx, det_stream_t) {
    launch_norm(true, x, w, eps, M, n, y, xq, sx, mx);
}

void det_quant(const float* x, int M, int n, int8_t* xq, float* sx, uint16_t* mx, det_stream_t) {
    if (M <= 0) return;
    if (n <= 8192) { launch_norm(false, x, nullptr, 0.0f, M, n, nullptr, xq, sx, mx); return; }
    if (mx) { launch_group_quant(0, x, nullptr, M, n, nullptr, xq, sx, mx); return; }
    det_p_norm p = { gpu(x), 0, 0, gpu(xq), gpu(sx), 0, 0.0f, n };
    dispatch("k_quant", &p, sizeof p, MTLSizeMake(M, 1, 1), MTLSizeMake(256, 1, 1));
}

void det_embed(const uint16_t* table_bf16, const int* ids, int M, int H, float* x, det_stream_t) {
    if (M <= 0) return;
    det_p_embed p = { gpu(table_bf16), gpu(ids), gpu(x), H };
    dispatch("k_embed", &p, sizeof p, MTLSizeMake(M, 1, 1), MTLSizeMake(256, 1, 1));
}

void det_add(float* x, const float* y, long long n, det_stream_t) {
    if (n <= 0) return;
    det_p_add p = { gpu(x), gpu(y), (uint64_t)n };
    dispatch("k_add", &p, sizeof p, MTLSizeMake((n + 255) / 256, 1, 1), MTLSizeMake(256, 1, 1));
}

void det_silu_mul_quant(const float* g, const float* u, int M, int n, float* h, int8_t* hq, float* sh, uint16_t* mh, det_stream_t) {
    if (M <= 0) return;
    if (mh) { launch_group_quant(1, g, u, M, n, h, hq, sh, mh); return; }
    det_p_gquant p = { gpu(g), gpu(u), gpu_or_zero(h), gpu(hq), gpu(sh), 0, n };
    dispatch("k_silu_mul_quant", &p, sizeof p, MTLSizeMake(M, 1, 1), MTLSizeMake(256, 1, 1));
}

void det_argmax(const float* logits, int M, int V, int* out, det_stream_t) {
    if (M <= 0) return;
    det_p_argmax p = { gpu(logits), gpu(out), V };
    dispatch("k_argmax", &p, sizeof p, MTLSizeMake(M, 1, 1), MTLSizeMake(256, 1, 1));
}

}
extern "C" void det_copy_cols(const float* src, int N, int off, int n, int T, float* dst, det_stream_t) {
    if (T <= 0 || n <= 0) return;
    det_p_copycols p = { gpu(src), gpu(dst), N, off, n, T };
    dispatch("k_copy_cols", &p, sizeof p, MTLSizeMake((n + 255) / 256, T, 1), MTLSizeMake(256, 1, 1));
}
// one dispatch for all column blocks of a fused projection (see k_split_cols)
extern "C" void det_split_cols(const float* src, int N, int T, int nparts, const int* offs, const int* ns, float* const* dsts, det_stream_t) {
    if (T <= 0 || nparts <= 0) return;
    if (nparts > 4) fail("det_split_cols: at most 4 parts");
    det_p_splitcols p; p.src = gpu(src); p.N = N; p.T = T; p.nparts = nparts;
    int total = 0;
    for (int i = 0; i < 4; i++) { p.dst[i] = i < nparts ? gpu(dsts[i]) : 0; p.off[i] = i < nparts ? offs[i] : 0; p.n[i] = i < nparts ? ns[i] : 0; if (i < nparts) total += ns[i]; }
    dispatch("k_split_cols", &p, sizeof p, MTLSizeMake((total + 255) / 256, T, 1), MTLSizeMake(256, 1, 1));
}
