"""Bit-exact test of the Metal sampling kernels vs detref/sample.py (port of tests/test_sample.py)."""
import sys, os, time, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python")); sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "detref"))
import detmetal as D, sample as S
from detmath import f32
rng = np.random.RandomState(0); ok = True; n = 0

def draws(M, st=987654321):
    pre, zs = [], []
    for _ in range(M):
        pre.append(st); st, z = S.splitmix64(st); zs.append(z)
    return pre, D.to_dev(np.array(zs, dtype=np.uint64))

for V in [1000, 248320, 151936]:
    for temp in [1.0, 0.7, 1.5, 0.05]:
        M = 6
        lg = (rng.randn(M, V) * (rng.rand(M, 1) * 6 + 1)).astype(f32)
        lg[1, 17] = 500.0
        pre, zt = draws(M)
        ref = [S.sample(lg[i], pre[i], temperature=temp)[0] for i in range(M)]
        out = D.empty((M,), np.int32)
        D.sample(D.to_dev(lg), f32(1.0 / temp), zt, out); D.sync()
        got = out.np.tolist()
        same = got == ref; ok &= same; n += 1
        print("PASS" if same else "FAIL", f"V={V} temp={temp}: metal={got} ref={ref}")

def make_rows(V):
    M = 6
    lg = (rng.randn(M, V) * (rng.rand(M, 1) * 6 + 1)).astype(f32)
    lg[1, 17] = 500.0
    lg[2] = rng.randint(0, 6, V).astype(f32)
    lg[3] = 0.0
    top = rng.choice(V, 200, replace=False); lg[5, top] = 30.0
    return lg
for V in [1000, 248320]:
    lg = make_rows(V); lgd = D.to_dev(lg); M = lg.shape[0]
    scratch = D.empty((D.sample_scratch_bytes(M, V),), np.uint8)
    out = D.empty((M,), np.int32)
    for temp in [1.0, 0.7, 0.05]:
        for top_k in [0, 1, 10, 50, 1000]:
            for top_p in [1.0, 0.95, 0.5, 0.1]:
                pre, zt = draws(M, st=1234567 + top_k * 7 + int(top_p * 100))
                t0 = time.time()
                ref = [S.sample(lg[i], pre[i], temperature=temp, top_k=top_k, top_p=top_p)[0] for i in range(M)]
                tr = time.time() - t0
                D.sample_topkp(lgd, f32(1.0 / temp), top_k, top_p, zt, out, scratch); D.sync()
                got = out.np.tolist()
                same = got == ref; ok &= same; n += 1
                print("PASS" if same else "FAIL", f"V={V} temp={temp} top_k={top_k} top_p={top_p}: metal={got} ref={ref} (ref {tr:.2f}s)")
V = 248320; lg = make_rows(V)[:1]; lgd = D.to_dev(lg)
scratch = D.empty((D.sample_scratch_bytes(1, V),), np.uint8); out = D.empty((1,), np.int32)
_, zt = draws(1)
for name, fn in [("temperature only  (det_sample)", lambda: D.sample(lgd, f32(1.0), zt, out)),
                 ("top_k=50          (det_sample_topkp)", lambda: D.sample_topkp(lgd, f32(1.0), 50, 1.0, zt, out, scratch)),
                 ("top_p=0.95        (det_sample_topkp)", lambda: D.sample_topkp(lgd, f32(1.0), 0, 0.95, zt, out, scratch)),
                 ("top_k=50,top_p=.95(det_sample_topkp)", lambda: D.sample_topkp(lgd, f32(1.0), 50, 0.95, zt, out, scratch))]:
    for _ in range(5): fn()
    D.sync(); t0 = time.time(); it = 100
    for _ in range(it): fn()
    D.sync(); print(f"timing V={V} M=1 {name}: {(time.time() - t0) / it * 1e6:.1f} us/sample")
print(f"{n} checks;", "ALL PASS" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
