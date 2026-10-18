"""Bit-exact tests + benchmark for the Metal det_gemm (port of tests/test_gemm.py; numpy references only).
usage: python tests/metal/test_gemm.py [quick|full|bench|dec|pre]"""
import sys, os, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "detref"))
import detmetal as D
import detmath as DM
try:
    import safetensors.numpy
except ImportError:   # the spec reference imports safetensors at module level; it is not needed here
    import types; sys.modules['safetensors'] = types.ModuleType('safetensors'); sys.modules['safetensors.numpy'] = types.ModuleType('safetensors.numpy'); sys.modules['safetensors.numpy'].load_file = None
from quant import quantize_linear, pack, qmax_for, GROUP
from model import Linear
f32 = np.float32
rng = np.random.RandomState(0)
_scratch = D.empty((D.gemm_scratch_bytes(8, 262144),), np.uint8); D.gemm_set_scratch(_scratch)
SHAPES = [(48, 2560), (96, 5120), (2560, 2560), (9216, 2560), (2560, 9216), (8192, 2560), (4096, 2560), (17408, 5120), (5120, 17408), (1000, 1152)]
MS = [1, 2, 3, 5, 8, 9, 17, 32, 33, 64, 129, 256, 1024]

def make_weights(N, K, bits, synth=False):
    qmax = qmax_for(bits)
    if not synth:
        W = rng.randn(N, K) * 0.02
        q = quantize_linear(W, bits, search=(N * K <= 2560 * 9216))
        wq, c, m = q["wq"], q["c"], q["m"]
    else:
        wq = rng.randint(-qmax, qmax + 1, size=(N, K)).astype(np.int8)
        c = (rng.rand(N) * 1e-6 + 1e-8).astype(f32)
        m = rng.randint(1, 65536, size=(N, K // GROUP)).astype(np.uint16)
    return pack(wq, bits), c, m, wq

def make_acts(M, K, scale=1.0, mx_mode="real"):
    x = (rng.randn(M, K) * scale * (rng.rand(M, 1) * 3 + 0.1)).astype(f32)
    x = (x.reshape(M, K // GROUP, GROUP) * (rng.rand(M, K // GROUP, 1) * 4 + 0.05)).reshape(M, K).astype(f32)
    xq, sx, mx = DM.quant_act_i8g(x)
    if mx_mode == "rand": mx = rng.randint(1, 65536, size=(M, K // GROUP)).astype(np.uint16)
    elif mx_mode == "none": mx = None
    return xq, sx, mx

def ref_numpy(packed, c, m, N, K, bits, xq, sx, mx):
    L = Linear({"w.wq": packed, "w.c": c, "w.m": m, }, "w", dict(N=N, K=K, bits=bits))
    return L(xq, sx, mx)

def ref_fast(wq, c, m, xq, sx, mx):
    """same integer semantics via int64 numpy (faster than model.Linear for big cases)"""
    N, K = wq.shape; M = xq.shape[0]; G = K // GROUP
    acc = np.zeros((M, N), np.int64)
    mxl = np.ones((M, G), np.int64) if mx is None else mx.astype(np.int64)
    w = wq.astype(np.float64); x = xq.astype(np.float64)     # float64 BLAS: exact for these integers (< 2^21 per group)
    for g in range(G):
        part = x[:, g * GROUP:(g + 1) * GROUP] @ w[:, g * GROUP:(g + 1) * GROUP].T
        acc += part.astype(np.int64) * (m[None, :, g].astype(np.int64) * mxl[:, g, None])
    y = acc.astype(f32) * c[None, :]
    return (y * sx[:, None]).astype(f32)

def run_gemm(packed, c, m, N, K, bits, xq, sx, mx):
    L = D.make_qlinear(packed, c, m, N, K, bits)
    M = xq.shape[0]
    y = D.empty((M, N))
    D.gemm(L, D.to_dev(xq), D.to_dev(sx), None if mx is None else D.to_dev(mx), y)
    D.sync()
    return y.np.copy()

def report(name, a, b):
    ok = a.shape == b.shape and np.array_equal(a.view(np.uint32), b.view(np.uint32))
    if ok: print("PASS", name)
    else:
        bad = np.argwhere(a.view(np.uint32) != b.view(np.uint32))
        print("FAIL", name, f"n_diff {len(bad)} / {a.size}, first {bad[:3].tolist()}", "got", a[tuple(bad[0])] if len(bad) else None, "want", b[tuple(bad[0])] if len(bad) else None)
    return ok

def correctness(quick=False, ms=None, shapes=None):
    allok = True
    shapes = shapes or (SHAPES if not quick else [(48, 2560), (96, 5120), (2560, 2560), (1000, 1152), (17408, 5120), (5120, 17408)])
    ms = ms or (MS if not quick else [1, 3, 8, 9, 17, 64, 1024])
    for bits in (4, 6, 8):
        for (N, K) in shapes:
            synth = N * K > 2560 * 9216
            packed, c, m, wq = make_weights(N, K, bits, synth)
            for M in ms:
                for mx_mode in (("real", "rand", "none") if M <= 129 else ("real",)):
                    xq, sx, mx = make_acts(M, K, mx_mode=mx_mode)
                    y = run_gemm(packed, c, m, N, K, bits, xq, sx, mx)
                    yr = ref_fast(wq, c, m, xq, sx, mx)
                    if M * N * K <= 64 * 2560 * 2560 and mx_mode == "real":   # cross-check the fast reference vs the spec reference
                        allok &= report(f"[ref_fast vs numpy-ref] W{bits} N={N} K={K} M={M}", yr, ref_numpy(packed, c, m, N, K, bits, xq, sx, mx))
                    allok &= report(f"gemm W{bits} N={N} K={K} M={M} mx={mx_mode}", y, yr)
            del packed, c, m, wq
    # ---- fused epilogue: accumulate (y += result) and column routing into parts
    for bits in (6, 8):
        N, K = 2560 + 96, 2560
        packed, c, m, wq = make_weights(N, K, bits)
        for M in ([1, 3, 8, 17, 64] if not os.environ.get("DET_GEMM_FORCE", "").startswith("d") else [1, 3, 8]):
            xq, sx, mx = make_acts(M, K)
            yr = ref_fast(wq, c, m, xq, sx, mx)
            L = D.make_qlinear(packed, c, m, N, K, bits)
            y0 = (rng.randn(M, N) * 3).astype(f32); y0[:, :5] *= f32(1e-39)
            y = D.to_dev(y0); D.gemm(L, D.to_dev(xq), D.to_dev(sx), D.to_dev(mx), y, accumulate=True); D.sync()
            allok &= report(f"gemm accumulate W{bits} N={N} K={K} M={M}", y.np.copy(), (y0 + yr).astype(f32))
            widths = [2560, 64, 32]; offs = [0, 2560, 2624]
            dsts = [D.empty((M, w)) for w in widths]
            D.gemm(L, D.to_dev(xq), D.to_dev(sx), D.to_dev(mx), None, parts=(offs, dsts)); D.sync()
            got = np.concatenate([d.np.copy() for d in dsts], axis=1)
            allok &= report(f"gemm parts W{bits} N={N} K={K} M={M}", got, yr)
    # ---- adversarial: all-max codes / activations / multipliers (no int32 overflow, |acc| up to 2^61)
    for bits in (4, 6, 8):
        for (N, K) in [(64, 17408), (300, 5120)]:
            qmax = qmax_for(bits)
            wq = np.where((np.arange(N) % 2 == 0)[:, None], qmax, -qmax).astype(np.int8) * np.ones((N, K), np.int8)
            wq[N // 2:] = np.where((np.arange(K) % 2 == 0)[None, :], qmax, -qmax).astype(np.int8)
            m = np.full((N, K // GROUP), 65535, np.uint16)
            c = (rng.rand(N) * 1e-5 + 1e-7).astype(f32)
            packed = pack(wq, bits)
            for M in [mm for mm in [1, 3, 8, 17, 64, 129] if not (os.environ.get('DET_GEMM_FORCE', '').startswith('d') and mm > 8)]:
                xq = np.full((M, K), 127, np.int8)
                if M > 1: xq[1] = -127
                if M > 2: xq[2] = np.where(np.arange(K) % 2 == 0, 127, -127)
                if M > 3: xq[3:] = rng.randint(-127, 128, size=(M - 3, K))
                sx = (rng.rand(M) + 0.5).astype(f32)
                mx = np.full((M, K // GROUP), 65535, np.uint16)
                y = run_gemm(packed, c, m, N, K, bits, xq, sx, mx)
                allok &= report(f"adversarial W{bits} N={N} K={K} M={M}", y, ref_fast(wq, c, m, xq, sx, mx))
    print("ALL PASS" if allok else "SOME FAILED")
    return allok

def bench():
    print("=== decode (M small): GB/s of packed weights (+m) streamed")
    for (N, K, bits) in [(17408, 5120, 6), (5120, 17408, 6), (10240, 5120, 6), (248320, 5120, 8), (96, 5120, 8), (5120, 5120, 6), (17408, 5120, 4), (17408, 5120, 8)]:
        wbytes = N * K * bits // 8 + N * (K // 128) * 2
        ncopy = max(1, min(6, int(300e6 // wbytes) + 1))
        Ls = []
        for i in range(ncopy):
            packed = rng.randint(0, 256, (N, K * bits // 8)).astype(np.uint8)
            c = (rng.rand(N) * 1e-6).astype(f32); m = rng.randint(1, 65536, (N, K // 128)).astype(np.uint16)
            Ls.append(D.make_qlinear(packed, c, m, N, K, bits))
        for M in [1, 2, 4, 8]:
            xq = D.to_dev(rng.randint(-127, 128, (M, K)).astype(np.int8)); sx = D.to_dev(rng.rand(M).astype(f32))
            mx = D.to_dev(rng.randint(1, 65536, (M, K // 128)).astype(np.uint16)); y = D.empty((M, N))
            for i in range(30): D.gemm(Ls[i % ncopy], xq, sx, mx, y)   # long warm-up: GPU clocks ramp
            D.sync()
            iters = 20 if wbytes > 500e6 else 100
            t0 = time.time()
            for i in range(iters): D.gemm(Ls[i % ncopy], xq, sx, mx, y)
            D.sync(); dt = (time.time() - t0) / iters
            print(f"  W{bits} N={N:6d} K={K:5d} M={M}: {dt*1e6:8.1f} us  {wbytes/dt/1e9:7.1f} GB/s  ({2*M*N*K/dt/1e12:.2f} TOPS)")
        del Ls
    print("=== prefill: TOPS (2*M*N*K / time)")
    for (N, K, bits) in [(17408, 5120, 6), (17408, 5120, 8), (17408, 5120, 4), (5120, 17408, 6), (248320, 5120, 8)]:
        packed = rng.randint(0, 256, (N, K * bits // 8)).astype(np.uint8)
        c = (rng.rand(N) * 1e-6).astype(f32); m = rng.randint(1, 65536, (N, K // 128)).astype(np.uint16)
        L = D.make_qlinear(packed, c, m, N, K, bits)
        for M in ([16, 32, 64, 129, 256, 1024, 4096] if N < 100000 else [1024]):
            xq = D.to_dev(rng.randint(-127, 128, (M, K)).astype(np.int8)); sx = D.to_dev(rng.rand(M).astype(f32))
            mx = D.to_dev(rng.randint(1, 65536, (M, K // 128)).astype(np.uint16)); y = D.empty((M, N))
            for i in range(2): D.gemm(L, xq, sx, mx, y)
            D.sync()
            iters = 5 if M >= 1024 else 20
            t0 = time.time()
            for i in range(iters): D.gemm(L, xq, sx, mx, y)
            D.sync(); dt = (time.time() - t0) / iters
            print(f"  W{bits} N={N:6d} K={K:5d} M={M:5d}: {dt*1e6:9.1f} us  {2*M*N*K/dt/1e12:7.2f} TOPS   (weights {N*K*bits/8/dt/1e9:.0f} GB/s-equivalent)")

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "full"
    if mode == "bench": bench()
    elif mode == "dec": os.environ["DET_GEMM_FORCE"] = "dec"; sys.exit(0 if correctness(ms=[1, 2, 3, 4, 5, 7, 8]) else 1)
    elif mode == "pre": os.environ["DET_GEMM_FORCE"] = "pre"; sys.exit(0 if correctness(ms=[1, 2, 3, 8, 31, 32, 33, 63, 64, 65, 200]) else 1)
    else: sys.exit(0 if correctness(quick=(mode == "quick")) else 1)
