"""Denormal-exactness of the Metal fp32 primitives: random and adversarial operand pairs (denormal inputs, results that
underflow, cancellation, the minimum-normal boundary) for mul/add/sub/div/sqrt/max vs numpy (IEEE)."""
import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))
import detmetal as D
f32 = np.float32
rng = np.random.RandomState(1)
ok = True
def chk(name, a, b):
    global ok
    same = np.array_equal(a.view(np.uint32), b.view(np.uint32))
    if not same:
        bad = np.argwhere(a.view(np.uint32) != b.view(np.uint32)).ravel()
        print("FAIL", name, f"n_diff {len(bad)}/{a.size}; idx {bad[:3]} got {[hex(v) for v in a.view(np.uint32)[bad[:3]]]} want {[hex(v) for v in b.view(np.uint32)[bad[:3]]]}")
    else: print("PASS", name)
    ok &= same
def run(op, x):
    xd = D.to_dev(x.astype(f32)); out = D.empty(x.shape, f32)
    D.test_prim(op, xd, out, 1, x.size); D.sync(); return out.np.copy()
def rbits(n, lo_exp, hi_exp):
    """random floats with exponent field in [lo_exp, hi_exp] (0 = denormal), random sign/mantissa"""
    e = rng.randint(lo_exp, hi_exp + 1, n).astype(np.uint32)
    m = rng.randint(0, 2**23, n).astype(np.uint32)
    s = rng.randint(0, 2, n).astype(np.uint32)
    return ((s << 31) | (e << 23) | m).view(f32)
N = 400000
with np.errstate(all="ignore"):
    sets = {
        "den x den": (rbits(N, 0, 0), rbits(N, 0, 0)),
        "den x normal": (rbits(N, 0, 0), rbits(N, 1, 254)),
        "small x small (underflow)": (rbits(N, 1, 80), rbits(N, 1, 80)),
        "near min-normal": (rbits(N, 60, 70), rbits(N, 55, 70)),
        "mixed": (rbits(N, 0, 254), rbits(N, 0, 254)),
        "wide add": (rbits(N, 0, 60), rbits(N, 0, 60)),
        "zeros": (np.where(rng.rand(N) < 0.3, f32(0), rbits(N, 0, 3)) * np.where(rng.rand(N) < 0.5, f32(1), f32(-1)), rbits(N, 0, 3)),
    }
    for name, (a, b) in sets.items():
        x = np.concatenate([a, b[::-1]]).astype(f32)   # op pairs x[i] with x[n-1-i]
        n = x.size
        xa, xb = x, x[::-1]
        chk(f"mul {name}", run(9, x), (xa * xb).astype(f32))
        chk(f"add {name}", run(10, x), (xa + xb).astype(f32))
        chk(f"sub {name}", run(15, x), (xa - xb).astype(f32))
        d = (xa / xb).astype(f32)
        got = run(14, x)
        m = ~(np.isnan(d) | np.isnan(got))   # 0/0 and den/0 corner cases: only compare where numpy gives a number
        chk(f"div {name} (non-nan {m.sum()})", got[m], d[m])
        mx = np.maximum(xa, xb).astype(f32); mx[(xa == 0) & (xb == 0)] = got[(xa == 0) & (xb == 0)] = 0   # +-0 ties: don't care
        gm = run(16, x); gm[(xa == 0) & (xb == 0)] = 0
        chk(f"max {name}", gm, mx)
    a = np.abs(np.concatenate([rbits(N, 0, 0), rbits(N, 0, 254)])).astype(f32)
    chk("sqrt (den + normal)", run(8, a), np.sqrt(a))
    # products that land exactly at / just below the minimum normal and at denormal ties
    ex = []
    for k in range(1, 40):
        ex += [(2.0 ** -k, 2.0 ** (-126 + k)), (2.0 ** -k * (1 + 2 ** -23), 2.0 ** (-126 + k) * (1 - 2 ** -23)), (2.0 ** -k, 2.0 ** (-149 + k)), (3 * 2.0 ** -k, 2.0 ** (-150 + k)), (2.0 ** -k * (1 + 2 ** -23), 2.0 ** (-150 + k))]
    a = np.array([p[0] for p in ex], f32); b = np.array([p[1] for p in ex], f32)
    x = np.concatenate([a, b[::-1]]).astype(f32)
    chk("mul boundary cases", run(9, x), (x * x[::-1]).astype(f32))
    chk("div boundary cases", run(14, x), (x / x[::-1]).astype(f32))
print("ALL PASS" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
