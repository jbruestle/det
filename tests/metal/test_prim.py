"""Bit-exactness of the Metal fp32 primitives (exp_f, log_f, softplus, sigmoid, silu, int64->f32, rint, div, sqrt,
mul/add incl. denormals, treesum) against detref/detmath.py.  Run: python tests/metal/test_prim.py"""
import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "detref"))
import detmetal as D
import detmath as DM
f32 = np.float32
rng = np.random.RandomState(0)
print("device:", D.device_name())
ok = True
def chk(name, a, b):
    global ok
    a = np.asarray(a); b = np.asarray(b)
    same = a.shape == b.shape and np.array_equal(a.view(np.uint32), b.view(np.uint32))
    if not same:
        bad = np.argwhere(a.view(np.uint32) != b.view(np.uint32)).ravel()
        print("FAIL", name, f"n_diff {len(bad)}/{a.size}; first idx {bad[:4]} got {a.ravel()[bad[:4]]} want {b.ravel()[bad[:4]]}")
    else: print("PASS", name)
    ok &= same
def run(op, x):
    xd = D.to_dev(x.astype(f32)); out = D.empty(x.shape, f32)
    D.test_prim(op, xd, out, 1, x.size); D.sync(); return out.np.copy()

# dense sweeps + random + special values
xs = np.concatenate([np.linspace(-100, 100, 400001), rng.randn(200000) * 30, rng.uniform(-1, 1, 100000),
                     np.array([0, -0, 1e-40, -1e-40, 87.9, -86.9, -87, -87.0000001, 88, 88.1, 1e30, -1e30, 20, 20.0001, -15, -15.0001, -80, -80.0001])]).astype(f32)
chk("exp_f", run(0, xs), DM.exp_f(xs))
chk("softplus_f", run(2, xs), DM.softplus_f(xs))
chk("sigmoid_f", run(3, xs), DM.sigmoid_f(xs))
chk("silu_f", run(4, xs), DM.silu_f(xs))
xl = np.concatenate([np.linspace(1e-30, 1, 200001), np.linspace(1, 1e30, 200001), 10 ** rng.uniform(-37, 37, 200000)]).astype(f32)
xl = xl[xl > 0]
chk("log_f", run(1, xl), DM.log_f(xl))
# int64 -> f32 (manual RNE) and native conversion
vals = np.concatenate([rng.randint(-2**62, 2**62, 100000), rng.randint(-2**40, 2**40, 100000), rng.randint(-2**25, 2**25, 100000),
                       np.array([0, 1, -1, 2**24, 2**24 + 1, 2**24 + 3, 2**25 - 1, 2**31 - 1, 2**31, 2**31 + 1, 2**33 + 5, -(2**40 + 7), 2**61 - 1, -(2**61), 16777219, 16777221, 33554435, 33554437])]).astype(np.int64)
bits = vals.view(np.uint32).astype(np.uint32)   # pairs (lo, hi) little endian
got = run(5, bits.view(f32))[0::2]
chk("i64_to_f (manual RNE)", got, vals.astype(f32))
got12 = run(12, bits.view(f32))[0::2]
print("INFO native long->float RNE:", np.array_equal(got12.view(np.uint32), vals.astype(f32).view(np.uint32)))
xr = (rng.randn(300000) * 1000).astype(f32); xr[:10] = [0.5, 1.5, 2.5, -0.5, -1.5, 2**23 + 1, 1e9, -2.5, 3.5, 0.49999997]
chk("rint", run(6, xr), np.rint(xr))
xd = np.concatenate([rng.randn(300000) * 1e3, 10 ** rng.uniform(-38, 38, 200000) * np.sign(rng.randn(200000))]).astype(f32)
xd = xd[xd != 0]
chk("1/x", run(7, xd), (f32(1) / xd).astype(f32))
xq = np.abs(xd)
chk("sqrt", run(8, xq), np.sqrt(xq))
xm = np.concatenate([rng.randn(200000), 10 ** rng.uniform(-38, 38, 200000) * np.sign(rng.randn(200000)), np.array([1e-39, 1.5e-38, 3e-39, 1e-45, 2.5e-45])]).astype(f32)
chk("mul (incl. denormals)", run(9, xm), (xm * xm[::-1]).astype(f32))
chk("add (incl. denormals)", run(10, xm), (xm + xm[::-1]).astype(f32))
chk("rint_i", run(13, xr[np.abs(xr) < 1e9]).astype(np.int32), np.rint(xr[np.abs(xr) < 1e9]).astype(np.int32))
# treesum
for n in [1, 5, 32, 100, 128, 256, 1000, 2560, 4096, 5120, 8192]:
    a = (rng.randn(7, n) * (10 ** rng.uniform(-3, 3, (7, 1)))).astype(f32)
    ad = D.to_dev(a); out = D.empty((7,), f32)
    D.test_prim(11, ad, out, 7, n); D.sync()
    chk(f"treesum n={n}", out.np.copy(), DM.treesum(a))
print("ALL PASS" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
