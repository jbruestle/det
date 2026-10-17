"""Bit-exact tests + benchmark for det_gemm (cuda/gemm.cu).

usage:  DET_LIB=cuda/libdet_gemm.so python tests/test_gemm.py            # correctness (all shapes/bits/M), both prefill paths
        DET_LIB=cuda/libdet_gemm.so python tests/test_gemm.py quick      # smaller subset
        DET_LIB=cuda/libdet_gemm.so python tests/test_gemm.py bench      # GB/s (decode) and TOPS (prefill)
        DET_GEMM_DP4A=1 ... python tests/test_gemm.py dp4a               # force the CUDA-core prefill kernel for M > 32

Activations: (xq, sx, mx) from detref.detmath.quant_act_i8g (per-(row,128-group) uint16 multipliers mx); every case is
also run with random mx in [1, 65535] and with mx=None (all ones).

Reference: detref.model.Linear (numpy, the spec).  For the very large cases (where the numpy reference would take
minutes / tens of GB) an exact torch-GPU reference `ref_torch` is used instead; it does the same integer math
(fp32 matmul of small integers is exact, int64 group scaling, fp32 RNE conversion, two fp32 multiplies) and is
itself checked bit-exactly against detref.model.Linear on every case where both are run.
"""
import sys, os, time
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "detref"))
import detcuda as D
import detmath as DM
import subprocess
from quant import quantize_linear, pack, qmax_for, GROUP
from model import Linear

f32 = np.float32
rng = np.random.RandomState(0)
torch.backends.cuda.matmul.allow_tf32 = False

SHAPES = [(48, 2560), (2560, 2560), (9216, 2560), (2560, 9216), (8192, 2560), (4096, 2560),
          (17408, 5120), (5120, 17408), (248320, 5120)]
MS = [1, 2, 3, 8, 17, 64, 129, 1024]
NUMPY_REF_MAX = 1024 * 17408 * 5120    # M*N*K budget for the numpy reference
SYNTH_MIN = 17408 * 5120 + 1           # shapes with N*K above this get synthesized random codes (no float W)


def make_weights(N, K, bits, synth=False):
    """returns (packed u8 [N, K/32*bpr], c f32 [N], m u16 [N, G], wq int8 [N,K] unpacked)"""
    qmax = qmax_for(bits)
    if not synth:
        W = rng.randn(N, K) * 0.02
        q = quantize_linear(W, bits, search=(N * K <= 2560 * 9216))
        wq, c, m = q["wq"], q["c"], q["m"]
    else:  # random codes with realistic c/m statistics (full use of the code range)
        wq = rng.randint(-qmax, qmax + 1, size=(N, K)).astype(np.int8)
        c = (rng.rand(N) * 1e-6 + 1e-8).astype(f32)
        m = rng.randint(1, 65536, size=(N, K // GROUP)).astype(np.uint16)
    return pack(wq, bits), c, m, wq


def make_acts(M, K, scale=1.0, mx_mode="real"):
    """mx_mode: 'real' (quant_act_i8g), 'rand' (random uint16 in [1,65535]), 'none' (NULL -> all ones)"""
    # per-group varying magnitudes so the group multipliers mx are non-trivial
    x = (rng.randn(M, K) * scale * (rng.rand(M, 1) * 3 + 0.1)).astype(f32)
    x = (x.reshape(M, K // GROUP, GROUP) * (rng.rand(M, K // GROUP, 1) * 4 + 0.05)).reshape(M, K).astype(f32)
    xq, sx, mx = DM.quant_act_i8g(x)
    if mx_mode == "rand":
        mx = rng.randint(1, 65536, size=(M, K // GROUP)).astype(np.uint16)
    elif mx_mode == "none":
        mx = None
    return xq, sx, mx


def ref_numpy(packed, c, m, N, K, bits, xq, sx, mx):
    L = Linear({"w.wq": packed, "w.c": c, "w.m": m}, "w", dict(N=N, K=K, bits=bits))
    return L(xq, sx, mx)


def ref_torch(wq, c, m, xq, sx, mx):
    """exact reference on the GPU with the same integer semantics (see module docstring)"""
    N, K = wq.shape; M = xq.shape[0]; G = K // GROUP
    w = torch.from_numpy(wq).cuda().float().reshape(N, G, GROUP)
    x = torch.from_numpy(xq).cuda().float().reshape(M, G, GROUP)
    mm = torch.from_numpy(m.astype(np.int64)).cuda()                      # [N,G]
    mxl = torch.ones(M, G, dtype=torch.int64, device="cuda") if mx is None else torch.from_numpy(mx.astype(np.int64)).cuda()
    acc = torch.zeros(M, N, dtype=torch.int64, device="cuda")
    for g in range(G):
        part = x[:, g] @ w[:, g].T                                        # exact integers < 2^21
        acc += part.to(torch.int64) * (mm[:, g][None, :] * mxl[:, g][:, None])
    y = acc.float() * torch.from_numpy(c).cuda()[None, :]                 # int64 -> f32 RNE, then one fp32 mul
    y = y * torch.from_numpy(sx).cuda()[:, None]                           # second fp32 mul
    return y.cpu().numpy()


def run_gemm(packed, c, m, N, K, bits, xq, sx, mx):
    L = D.make_qlinear(D.to_dev(packed), D.to_dev(c), D.to_dev(m), N, K, bits)
    M = xq.shape[0]
    y = torch.empty(M, N, device="cuda", dtype=torch.float32)
    D.gemm(L, D.to_dev(xq), D.to_dev(sx), None if mx is None else D.to_dev(mx), y)
    torch.cuda.synchronize()
    return y.cpu().numpy()


def same(a, b):
    return a.shape == b.shape and np.array_equal(a.view(np.uint32), b.view(np.uint32))


def report(name, a, b):
    ok = same(a, b)
    if ok:
        print("PASS", name)
    else:
        d = np.abs(a.astype(np.float64) - b.astype(np.float64))
        bad = np.argwhere(a.view(np.uint32) != b.view(np.uint32))
        print("FAIL", name, f"max abs diff {d.max():.3e}, n_diff {len(bad)} / {a.size}, first {bad[:3].tolist()}",
              "got", a[tuple(bad[0])] if len(bad) else None, "want", b[tuple(bad[0])] if len(bad) else None)
    return ok


def correctness(quick=False, dp4a_only=False):
    allok = True
    shapes = SHAPES if not quick else [(48, 2560), (2560, 2560), (17408, 5120), (5120, 17408)]
    ms = MS if not quick else [1, 3, 8, 17, 64, 1024]
    if dp4a_only:
        shapes = [(48, 2560), (2560, 2560), (17408, 5120), (5120, 17408)]; ms = [33, 64, 129, 1024]
    for bits in (4, 6, 8):
        for (N, K) in shapes:
            if N == 248320 and bits != 8:
                continue
            synth = N * K >= SYNTH_MIN
            t0 = time.time()
            packed, c, m, wq = make_weights(N, K, bits, synth)
            tw = time.time() - t0
            for M in ms:
                for mx_mode in (("real", "rand", "none") if (M <= 129 or quick) else ("real", "rand")):
                    xq, sx, mx = make_acts(M, K, mx_mode=mx_mode)
                    y = run_gemm(packed, c, m, N, K, bits, xq, sx, mx)
                    name = f"gemm W{bits} N={N} K={K} M={M} mx={mx_mode}"
                    if M * N * K <= NUMPY_REF_MAX and not (N == 248320 and M > 64):
                        yr = ref_numpy(packed, c, m, N, K, bits, xq, sx, mx)
                        if N * K <= 2560 * 9216 and M <= 64:   # cross-check the torch reference against the spec reference
                            allok &= report(name + " [torch-ref vs numpy-ref]", ref_torch(wq, c, m, xq, sx, mx), yr)
                    else:
                        yr = ref_torch(wq, c, m, xq, sx, mx)
                        name += " [torch-ref]"
                    allok &= report(name, y, yr)
            del packed, c, m, wq
            print(f"   (weights {N}x{K} W{bits} built in {tw:.1f}s)")
    if dp4a_only:
        print("ALL PASS (dp4a prefill)" if allok else "SOME FAILED (dp4a prefill)")
        return allok
    # ---- partially filled decode token tiles (M < padded MP) across the warp-count variants of the decode kernel
    # (S = K/1024 warps: 3, 9 and 17; the 9-warp / M=5..8 combination once launched with too much shared memory)
    for bits in (4, 6, 8):
        for K in (2560, 9216, 10240, 17408):
            N = 256
            packed, c, m, wq = make_weights(N, K, bits)
            for M in (5, 6, 7, 8):
                xq, sx, mx = make_acts(M, K)
                y = run_gemm(packed, c, m, N, K, bits, xq, sx, mx)
                allok &= report(f"decode partial-MP W{bits} N={N} K={K} M={M}", y, ref_numpy(packed, c, m, N, K, bits, xq, sx, mx))
    # ---- adversarial: all-max codes, all-max activations, max m: inner group sums hit 128*127*qmax (no int32 overflow)
    for bits in (4, 6, 8):
        for (N, K) in [(64, 17408), (300, 5120)]:
            qmax = qmax_for(bits)
            wq = np.where((np.arange(N) % 2 == 0)[:, None], qmax, -qmax).astype(np.int8) * np.ones((N, K), np.int8)
            wq[N // 2:] = np.where((np.arange(K) % 2 == 0)[None, :], qmax, -qmax).astype(np.int8)
            m = np.full((N, K // GROUP), 65535, np.uint16)
            c = (rng.rand(N) * 1e-5 + 1e-7).astype(f32)
            packed = pack(wq, bits)
            for M in [1, 3, 8, 17, 64, 129]:
                xq = np.full((M, K), 127, np.int8)
                if M > 1: xq[1] = -127
                if M > 2: xq[2] = np.where(np.arange(K) % 2 == 0, 127, -127)
                if M > 3: xq[3:] = rng.randint(-127, 128, size=(M - 3, K))
                sx = (rng.rand(M) + 0.5).astype(f32)
                mx = np.full((M, K // GROUP), 65535, np.uint16)        # m*mx = 65535^2 < 2^32, |acc| < 2^61
                y = run_gemm(packed, c, m, N, K, bits, xq, sx, mx)
                yr = ref_numpy(packed, c, m, N, K, bits, xq, sx, mx)
                allok &= report(f"adversarial W{bits} N={N} K={K} M={M} (|acc| up to {np.abs(yr/ (c[None,:]*sx[:,None])).max():.2e})", y, yr)
    if not quick:   # the CUDA-core prefill kernel is also used for M > 32 when DET_GEMM_DP4A=1: test it in a subprocess
        env = dict(os.environ, DET_GEMM_DP4A="1")
        r = subprocess.run([sys.executable, os.path.abspath(__file__), "dp4a"], env=env, capture_output=True, text=True)
        tail = r.stdout.strip().splitlines()
        fails = [l for l in tail if l.startswith("FAIL")]
        print(*fails, sep="\n")
        print(tail[-1] if tail else "(no output)", "" if r.returncode == 0 else r.stderr[-2000:])
        allok &= (r.returncode == 0)
    print("ALL PASS" if allok else "SOME FAILED")
    return allok


def bench():
    HBM = 1792e9
    print("=== decode (M small) : GB/s of packed weights (+m) streamed; several weight copies > L2 to defeat caching")
    for (N, K, bits) in [(17408, 5120, 6), (5120, 17408, 6), (10240, 5120, 6), (248320, 5120, 8), (48, 5120, 8), (5120, 5120, 6), (17408, 5120, 4), (17408, 5120, 8)]:
        wbytes = N * K * bits // 8 + N * (K // 128) * 2
        ncopy = max(1, min(8, int(400e6 // wbytes) + 1))
        Ls = []
        for i in range(ncopy):
            packed = torch.randint(0, 256, (N, K * bits // 8), dtype=torch.uint8, device="cuda")
            c = torch.rand(N, device="cuda") * 1e-6
            m = torch.randint(1, 65536, (N, K // 128), dtype=torch.int32, device="cuda").to(torch.int16)
            Ls.append(D.make_qlinear(packed, c, m, N, K, bits))
        for M in [1, 2, 4, 8]:
            xq = torch.randint(-127, 128, (M, K), dtype=torch.int8, device="cuda")
            sx = torch.rand(M, device="cuda")
            mx = torch.randint(1, 65536, (M, K // 128), dtype=torch.int32, device="cuda").to(torch.int16)
            y = torch.empty(M, N, device="cuda")
            for i in range(3): D.gemm(Ls[i % ncopy], xq, sx, mx, y)
            torch.cuda.synchronize()
            iters = 20 if wbytes > 500e6 else 100
            st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            st.record()
            for i in range(iters): D.gemm(Ls[i % ncopy], xq, sx, mx, y)
            en.record(); torch.cuda.synchronize()
            dt = st.elapsed_time(en) / iters * 1e-3
            gbs = wbytes / dt / 1e9
            print(f"  W{bits} N={N:6d} K={K:5d} M={M}: {dt*1e6:8.1f} us  {gbs:7.1f} GB/s  ({gbs/HBM*1e9*100:5.1f}% of 1792)  "
                  f"({2*M*N*K/dt/1e12:.2f} TOPS)")
        del Ls
    print("=== prefill : TOPS (2*M*N*K / time)")
    for (N, K, bits) in [(17408, 5120, 6), (17408, 5120, 8), (17408, 5120, 4), (5120, 17408, 6), (248320, 5120, 8)]:
        packed = torch.randint(0, 256, (N, K * bits // 8), dtype=torch.uint8, device="cuda")
        c = torch.rand(N, device="cuda") * 1e-6
        m = torch.randint(1, 65536, (N, K // 128), dtype=torch.int32, device="cuda").to(torch.int16)
        L = D.make_qlinear(packed, c, m, N, K, bits)
        for M in ([16, 32, 64, 129, 256, 1024, 4096] if N < 100000 else [1024]):
            xq = torch.randint(-127, 128, (M, K), dtype=torch.int8, device="cuda")
            sx = torch.rand(M, device="cuda")
            mx = torch.randint(1, 65536, (M, K // 128), dtype=torch.int32, device="cuda").to(torch.int16)
            y = torch.empty(M, N, device="cuda")
            for i in range(2): D.gemm(L, xq, sx, mx, y)
            torch.cuda.synchronize()
            iters = 10 if M >= 1024 else 30
            st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            st.record()
            for i in range(iters): D.gemm(L, xq, sx, mx, y)
            en.record(); torch.cuda.synchronize()
            dt = st.elapsed_time(en) / iters * 1e-3
            print(f"  W{bits} N={N:6d} K={K:5d} M={M:5d}: {dt*1e6:9.1f} us  {2*M*N*K/dt/1e12:7.2f} TOPS   "
                  f"(weights {N*K*bits/8/dt/1e9:.0f} GB/s-equivalent)")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "full"
    if mode == "bench":
        bench()
    elif mode == "dp4a":
        sys.exit(0 if correctness(dp4a_only=True) else 1)
    else:
        ok = correctness(quick=(mode == "quick"))
        sys.exit(0 if ok else 1)
