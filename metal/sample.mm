// host side of det_sample / det_sample_topkp (see shaders/sample.metal)
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <fenv.h>
#include "det_metal.h"
#include "runtime.h"
#include "params.h"
using namespace det;

namespace {
constexpr int SNT = 1024, SNB = 32, NBLK = 64;
inline size_t align_up(size_t x) { return (x + 255) & ~(size_t)255; }
struct Carve { char* keys; char* mx; char* hist; char* part; char* state; size_t total; };
Carve carve(void* base, int M, int V) {
    char* b = (char*)base; size_t off = 0; Carve c;
    c.keys = b + off;  off = align_up(off + (size_t)M * V * 8);
    c.mx = b + off;    off = align_up(off + (size_t)M * 4);
    c.hist = b + off;  off = align_up(off + (size_t)M * NBLK * 768 * 4);
    c.part = b + off;  off = align_up(off + (size_t)M * SNB * 8);
    c.state = b + off; off = align_up(off + (size_t)M * 48);
    c.total = off;
    return c;
}
void* g_scratch = nullptr; size_t g_scratch_bytes = 0;

void run(const float* logits, int M, int V, float inv_temp, int top_k, float top_p, const unsigned long long* z, int* out, void* scratch) {
    if (M <= 0) return;
    if (V > (1 << 20) || V < 1) fail("det_sample: V=%d out of range", V);
    const Carve c = carve(scratch, M, V);
    const int use_k = (top_k > 0 && top_k < V) ? 1 : 0;
    const int use_p = (top_p < 1.0f) ? 1 : 0;
    unsigned long long p24 = 0;
    if (use_p) { const int rm = fegetround(); fesetround(FE_TONEAREST); p24 = (unsigned long long)llrint((double)top_p * 16777216.0); fesetround(rm); }
    det_p_smp p = { gpu(logits), gpu(c.keys), gpu(c.mx), gpu(c.hist), gpu(c.part), gpu(c.state), gpu(z), gpu(out), inv_temp, V, top_k, 0, NBLK, p24, use_k, use_p };
    dispatch("k_smp_max", &p, sizeof p, MTLSizeMake(M, 1, 1), MTLSizeMake(SNT, 1, 1));
    dispatch("k_smp_weights", &p, sizeof p, MTLSizeMake(NBLK, M, 1), MTLSizeMake(SNT, 1, 1));
    dispatch("k_smp_init", &p, sizeof p, MTLSizeMake(M, 1, 1), MTLSizeMake(1, 1, 1));   // prefix 0, remk = top_k, cutkey 0
    if (use_k || use_p) {
        for (int d = 6; d >= 0; d--) {
            p.digit = d;
            dispatch("k_smp_hist", &p, sizeof p, MTLSizeMake(NBLK, M, 1), MTLSizeMake(SNT, 1, 1));
            dispatch("k_smp_scan", &p, sizeof p, MTLSizeMake(M, 1, 1), MTLSizeMake(256, 1, 1));
        }
    }
    dispatch("k_smp_part", &p, sizeof p, MTLSizeMake(SNB, M, 1), MTLSizeMake(SNT, 1, 1));
    dispatch("k_smp_select", &p, sizeof p, MTLSizeMake(M, 1, 1), MTLSizeMake(SNT, 1, 1));
}
}

extern "C" {

size_t det_sample_scratch_bytes(int M, int V) { return carve(nullptr, M, V).total; }

void det_sample(const float* logits, int M, int V, float inv_temp, const unsigned long long* z, int* out, det_stream_t) {
    const size_t need = det_sample_scratch_bytes(M, V);
    if (g_scratch_bytes < need) { if (g_scratch) det_free(g_scratch); g_scratch = det_malloc(need); g_scratch_bytes = need; }
    run(logits, M, V, inv_temp, 0, 1.0f, z, out, g_scratch);
}

void det_sample_topkp(const float* logits, int M, int V, float inv_temp, int top_k, float top_p, const unsigned long long* z, int* out, void* scratch, det_stream_t) {
    run(logits, M, V, inv_temp, top_k, top_p, z, out, scratch);
}

}
