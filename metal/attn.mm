// host side of det_attention (see shaders/attn.metal)
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string>
#include "det_metal.h"
#include "runtime.h"
#include "params.h"
using namespace det;

namespace {
constexpr int ATK = 32;
constexpr int MSPLIT = 256;                // key splitting (int64 scratch) only for M <= MSPLIT
constexpr int SPLIT_TARGET_BLOCKS = 512;
constexpr int NSPLIT_CAP = 128;
inline size_t align_up(size_t x) { return (x + 255) & ~(size_t)255; }
inline int nsplit_max(int M) { int s = SPLIT_TARGET_BLOCKS / (M < 1 ? 1 : M); return s < 1 ? 1 : (s > NSPLIT_CAP ? NSPLIT_CAP : s); }
struct Scratch { int8_t* qq; float* sq; uint32_t* mx; long long* num; long long* den; size_t bytes; };
Scratch layout(void* base, int M, int n_h, int hd) {
    char* b = (char*)base; size_t off = 0; Scratch s;
    s.qq = (int8_t*)(b + off);      off = align_up(off + (size_t)M * n_h * hd);
    s.sq = (float*)(b + off);       off = align_up(off + (size_t)M * n_h * 4);
    s.mx = (uint32_t*)(b + off);    off = align_up(off + (size_t)M * n_h * 4);
    if (M <= MSPLIT) {
        const size_t ns = nsplit_max(M);
        s.num = (long long*)(b + off); off = align_up(off + ns * (size_t)M * n_h * hd * 8);
        s.den = (long long*)(b + off); off = align_up(off + ns * (size_t)M * n_h * 8);
    } else { s.num = nullptr; s.den = nullptr; }
    s.bytes = off;
    return s;
}
}

extern "C" {

size_t det_attn_scratch_bytes(int M, int n_h, int hd) { return layout(nullptr, M, n_h, hd).bytes; }

void det_attention(const det_attn_layer* L, const det_kv_cache* kv, const float* qg, const float* k, const float* v,
                   const float* rope_cos, const float* rope_sin, int rot, float eps,
                   int M, int pos0, int n_h, int n_kv, int hd, float* o, void* scratch, det_stream_t) {
    if (M <= 0) return;
    if (n_h % n_kv != 0 || rot < 0 || rot > hd || (rot & 1)) fail("det_attention: bad n_h=%d n_kv=%d rot=%d hd=%d", n_h, n_kv, rot, hd);
    if (hd != 256 && hd != 128) fail("det_attention: unsupported hd=%d", hd);
    const int rep = n_h / n_kv;
    const bool rep_ok = hd == 256 ? (rep == 1 || rep == 2 || rep == 3 || rep == 4 || rep == 6 || rep == 8) : (rep == 1 || rep == 2 || rep == 4 || rep == 8);
    if (!rep_ok) fail("det_attention: unsupported n_h/n_kv = %d for hd=%d", rep, hd);
    const int chunk = hd / 32, half = rot / 2;
    if (half % chunk != 0 || ((half / chunk) & (half / chunk - 1)) != 0) fail("det_attention: rot=%d not supported for hd=%d", rot, hd);
    if (pos0 < 0 || pos0 + M > kv->capacity) fail("det_attention: KV cache overflow (pos0 %d + %d > %d)", pos0, M, kv->capacity);
    const Scratch S = layout(scratch, M, n_h, hd);
    int nsplit = 1;
    if (M <= MSPLIT) {
        int want = SPLIT_TARGET_BLOCKS / (M * n_kv);
        int maxs = (pos0 + M + ATK - 1) / ATK;
        nsplit = want < maxs ? want : maxs;
        if (nsplit > nsplit_max(M)) nsplit = nsplit_max(M);
        if (nsplit < 1) nsplit = 1;
    }
    det_p_attn p = { gpu(qg), gpu(k), gpu(v), gpu(L->q_norm), gpu(L->k_norm), gpu(L->v_scale), gpu(L->v_inv_scale), gpu(rope_cos), gpu(rope_sin),
                     gpu(S.qq), gpu(S.sq), gpu(S.mx), gpu(kv->kq), gpu(kv->sk), gpu(kv->vq), gpu(o), gpu_or_zero(S.num), gpu_or_zero(S.den), gpu_or_zero(kv->vlo), gpu_or_zero(kv->vhi),
                     eps, (float)(1.0 / sqrt((double)hd)), rot, M, pos0, n_h, n_kv, nsplit, 0 };
    const std::string hds = std::to_string(hd), sfx = hds + "_" + std::to_string(rep);
    const int rows = M * n_h + 2 * M * n_kv;
    dispatch(("k_attn_prep_" + hds).c_str(), &p, sizeof p, MTLSizeMake((rows + 7) / 8, 1, 1), MTLSizeMake(256, 1, 1));
    // prefill path (row-batched blocks, 128-key tiles, P.V on the tensor unit): M >= DET_ATTN_PRE_MIN (env; 0 = off)
    static const int pre_min = getenv("DET_ATTN_PRE") ? atoi(getenv("DET_ATTN_PRE")) : 16;   // DET_ATTN_PRE=0 disables
    if (pre_min > 0 && M >= pre_min && kv->vlo && kv->vhi) {
        const int ntok = 16 / rep, ntiles = (M + ntok - 1) / ntok;
        // key splits so that there are enough threadgroups for the cores (int64 partials need the split scratch: M <= MSPLIT)
        int ps = 1;
        if (M <= MSPLIT) {
            ps = (512 + n_kv * ntiles - 1) / (n_kv * ntiles);
            const int maxs = (pos0 + M + 127) / 128;
            if (ps > maxs) ps = maxs;
            if (ps > nsplit_max(M)) ps = nsplit_max(M);
            if (ps < 1) ps = 1;
        }
        p.nsplit = ps; p.pre = 1;
        const MTLSize pgrid = MTLSizeMake(n_kv, ntiles, ps);
        dispatch(("k_attn_pmax_" + sfx).c_str(), &p, sizeof p, pgrid, MTLSizeMake(128, 1, 1));
        // P.V: one threadgroup per 128-dim half (half the int64 accumulators per thread: more threadgroups per core)
        dispatch(("k_attn_pacc_" + sfx).c_str(), &p, sizeof p, MTLSizeMake(n_kv * (hd / 128), ntiles, ps), MTLSizeMake(128, 1, 1));
        if (ps > 1) dispatch(("k_attn_fin_" + hds).c_str(), &p, sizeof p, MTLSizeMake((size_t)M * n_h, 1, 1), MTLSizeMake(hd, 1, 1));
        return;
    }
    const MTLSize grid = MTLSizeMake(nsplit, n_kv, M);
    dispatch(("k_attn_max_" + sfx).c_str(), &p, sizeof p, grid, MTLSizeMake(128, 1, 1));
    dispatch(("k_attn_acc_" + sfx).c_str(), &p, sizeof p, grid, MTLSizeMake(128, 1, 1));
    if (nsplit > 1) dispatch(("k_attn_fin_" + hds).c_str(), &p, sizeof p, MTLSizeMake((size_t)M * n_h, 1, 1), MTLSizeMake(hd, 1, 1));
}

}
