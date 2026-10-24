// host side of det_gemm (see shaders/gemm.metal)
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include "det_metal.h"
#include "runtime.h"
#include "params.h"
using namespace det;

extern "C" {

// scratch: split-K int64 partials of the decode kernel, up to 8 splits x M x N (N <= the largest N in use; the runner
// sizes it with the largest N it has, see det_gemm_scratch_bytes(M, N_or_K))
static void* g_scratch = nullptr; static size_t g_scratch_bytes = 0;
constexpr size_t CTR_BYTES = 4096 * 4;   // split-K tile counters (self-resetting), then the partials
size_t det_gemm_scratch_bytes(int M, int N) { return CTR_BYTES + (size_t)8 * (M < 8 ? M : 8) * (size_t)N * 8; }
void det_gemm_set_scratch(void* ptr, size_t bytes) { g_scratch = ptr; g_scratch_bytes = ptr ? bytes : 0; }

void det_gemm_ex(const det_qlinear* Lp, const int8_t* xq, const float* sx, const uint16_t* mx, int M, float* y, int accumulate,
                 int nparts, const int* offs, float* const* dsts, det_stream_t) {
    const det_qlinear L = *Lp;
    if (M < 1) return;
    if (L.K % 128 != 0 || L.K <= 0 || L.N <= 0) fail("det_gemm: bad shape N=%d K=%d", L.N, L.K);
    if (L.bits != 4 && L.bits != 6 && L.bits != 8) fail("det_gemm: unsupported bits=%d", L.bits);
    const int bpl = L.bits == 8 ? 32 : L.bits == 6 ? 24 : 16;
    det_p_gemm p;
    memset(&p, 0, sizeof p);
    p.wq = gpu(L.wq); p.c = gpu(L.c); p.m = gpu(L.m); p.xq = gpu(xq); p.sx = gpu(sx); p.mx = gpu_or_zero(mx); p.y = gpu_or_zero(y);
    p.M = M; p.N = L.N; p.K = L.K; p.G = L.K / 128; p.bpr = L.K / 32 * bpl; p.splits = 1; p.accumulate = accumulate ? 1 : 0;
    if (nparts > 1) {
        if (nparts > 4) fail("det_gemm: at most 4 output parts");
        if (offs[0] != 0) fail("det_gemm: the first part must start at column 0");
        p.nparts = nparts;
        for (int i = 0; i < nparts; i++) { p.ydst[i] = gpu(dsts[i]); p.yoff[i] = offs[i]; }
    } else if (!y) fail("det_gemm: y == NULL");
    const char* force = getenv("DET_GEMM_FORCE");   // "dec" / "pre" (testing)
    const bool dec = force ? (force[0] == 'd') : (M <= 8);
    if (dec) {
        if (M > 8) fail("det_gemm: decode kernel forced with M=%d > 8", M);
        // row tile: 128 rows amortise the per-op cost of the tensor unit better; 64 rows for small N (more threadgroups)
        static const int BN_env = getenv("DET_GEMM_BN") ? atoi(getenv("DET_GEMM_BN")) : 0;
        const int BN = BN_env ? BN_env : (L.N >= 8192 ? 128 : 64);
        const int ntg = (L.N + BN - 1) / BN;
        // split K when there are too few threadgroups to hide memory latency (needs the scratch)
        int splits = 1;
        if (ntg < 256) {
            splits = (256 + ntg - 1) / ntg;
            if (splits > 8) splits = 8;
            if (splits > p.G) splits = p.G;
            const size_t need = CTR_BYTES + (size_t)splits * M * L.N * 8;
            if (ntg > 4096) splits = 1;
            if (splits > 1 && (!g_scratch || g_scratch_bytes < need)) {
                static bool warned = false;
                if (!warned) { fprintf(stderr, "det_gemm: no scratch for split-K (need %zu bytes; see det_gemm_set_scratch); running unsplit\n", need); warned = true; }
                splits = 1;
            }
        }
        p.splits = splits;
        p.counters = splits > 1 ? gpu(g_scratch) : 0;
        p.part = splits > 1 ? gpu((char*)g_scratch + CTR_BYTES) : 0;
        std::string name = "k_gemm_dectc_" + std::to_string(L.bits) + "_" + std::to_string(BN);
        dispatch(name.c_str(), &p, sizeof p, MTLSizeMake(ntg, splits, 1), MTLSizeMake(128, 1, 1));
    } else {
        static const int bm_env = getenv("DET_GEMM_PRE_BM") ? atoi(getenv("DET_GEMM_PRE_BM")) : 0;
        // measured: 32-row tiles run the int8 op at ~55 TOPS vs ~34 for 64-row tiles (register pressure of the 64x64 result)
        const int BM = bm_env ? bm_env : (M <= 16 ? 16 : 32), BN = 64;
        std::string name = "k_gemm_pre_" + std::to_string(L.bits) + "_" + std::to_string(BM) + "_" + std::to_string(BN);
        dispatch(name.c_str(), &p, sizeof p, MTLSizeMake((L.N + BN - 1) / BN, (M + BM - 1) / BM, 1), MTLSizeMake(128, 1, 1));
    }
}

}

extern "C" void det_gemm(const det_qlinear* Lp, const int8_t* xq, const float* sx, const uint16_t* mx, int M, float* y, det_stream_t st) {
    det_gemm_ex(Lp, xq, sx, mx, M, y, 0, 0, nullptr, nullptr, st);
}
