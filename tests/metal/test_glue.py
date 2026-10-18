"""Bit-exact tests of the Metal glue kernels (rmsnorm/quant/embed/add/silu_mul/argmax) vs detref (port of tests/test_glue.py)."""
import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "detref"))
import detmetal as D
import detmath as DM
rng = np.random.RandomState(0)
f32 = np.float32
def chk(name, a, b):
    a = np.asarray(a); b = np.asarray(b)
    ok = a.shape == b.shape and (np.array_equal(a, b) or (a.dtype.kind == 'f' and np.array_equal(a.view(np.uint32), b.view(np.uint32))))
    print(("PASS" if ok else "FAIL"), name, "" if ok else f"(max diff {np.abs(a.astype(np.float64)-b.astype(np.float64)).max() if a.shape == b.shape else 'shape'}, n_diff {np.sum(a!=b) if a.shape == b.shape else '?'})")
    return ok
allok = True
for n in [128, 256, 2560, 5120, 6144, 9216, 17408]:
    M = 7
    x = (rng.randn(M, n) * (rng.rand(M, 1) * 10)).astype(f32)
    x[2, :64] *= f32(1e-36)                                  # a few denormal / underflowing squares
    w = (1 + 0.5 * rng.randn(n)).astype(f32)
    xd, wd = D.to_dev(x), D.to_dev(w)
    yd = D.empty((M, n)); xq = D.empty((M, n), np.int8); sx = D.empty((M,))
    mx = D.empty((M, n // 128), np.uint16)
    if n <= 8192:
        D.rmsnorm_quant(xd, wd, 1e-6, yd, xq, sx); D.sync()
        yr = DM.rmsnorm(x, w, 1e-6); qr, sr = DM.quant_act_i8(yr)
        allok &= chk(f"rmsnorm n={n}", yd.np, yr)
        allok &= chk(f"rmsnorm quant n={n}", xq.np, qr) and chk(f"rmsnorm sx n={n}", sx.np, sr)
        D.rmsnorm_quant(xd, wd, 1e-6, None, xq, sx, mx); D.sync()
        qr, sr, mr = DM.quant_act_i8g(yr)
        allok &= chk(f"rmsnorm gquant n={n}", xq.np, qr) and chk(f"rmsnorm gsx n={n}", sx.np, sr) and chk(f"rmsnorm gmx n={n}", mx.np, mr)
    D.quant(xd, xq, sx); D.sync()
    qr, sr = DM.quant_act_i8(x)
    allok &= chk(f"quant n={n}", xq.np, qr) and chk(f"quant sx n={n}", sx.np, sr)
    D.quant(xd, xq, sx, mx); D.sync()
    qr, sr, mr = DM.quant_act_i8g(x)
    allok &= chk(f"gquant n={n}", xq.np, qr) and chk(f"gquant sx n={n}", sx.np, sr) and chk(f"gquant mx n={n}", mx.np, mr)
    g = (rng.randn(M, n) * 5).astype(f32); u = (rng.randn(M, n) * 3).astype(f32)
    g[3, :32] = -87.3; u[4, :32] *= f32(1e-38)
    hd = D.empty((M, n))
    D.silu_mul_quant(D.to_dev(g), D.to_dev(u), hd, xq, sx); D.sync()
    hr = (DM.silu_f(g) * u).astype(f32); qr, sr = DM.quant_act_i8(hr)
    allok &= chk(f"silu_mul n={n}", hd.np, hr) and chk(f"silu_mul quant n={n}", xq.np, qr)
    hd.zero_(); D.silu_mul_quant(D.to_dev(g), D.to_dev(u), hd, xq, sx, mx); D.sync()
    qr, sr, mr = DM.quant_act_i8g(hr)
    allok &= chk(f"silu_mul g h n={n}", hd.np, hr) and chk(f"silu_mul gquant n={n}", xq.np, qr) and chk(f"silu_mul gmx n={n}", mx.np, mr)
    x0 = x.copy(); x0[0] = 0; x0[1, :128] = 0
    D.quant(D.to_dev(x0), xq, sx, mx); D.sync(); qr, sr, mr = DM.quant_act_i8g(x0)
    allok &= chk(f"gquant zeros n={n}", xq.np, qr) and chk(f"gquant zeros mx n={n}", mx.np, mr) and chk(f"gquant zeros sx n={n}", sx.np, sr)
tab = rng.randint(0, 65536, size=(100, 64)).astype(np.uint16); ids = rng.randint(0, 100, size=5).astype(np.int32)
xd = D.empty((5, 64)); D.embed(D.to_dev(tab), D.to_dev(ids), xd); D.sync()
allok &= chk("embed", xd.np, (tab[ids].astype(np.uint32) << 16).view(f32))
a = rng.randn(1000).astype(f32); b = rng.randn(1000).astype(f32); a[:5] *= f32(1e-39); b[:5] *= f32(1e-39)
ad = D.to_dev(a); D.add(ad, D.to_dev(b)); D.sync()
allok &= chk("add", ad.np, (a + b).astype(f32))
lg = rng.randn(3, 248320).astype(f32); lg[1, 5] = lg[1].max() + 1; lg[1, 777] = lg[1, 5]
od = D.empty((3,), np.int32); D.argmax(D.to_dev(lg), od); D.sync()
allok &= chk("argmax", od.np, lg.argmax(1).astype(np.int32))
# memset / memcpy
m1 = D.to_dev(np.arange(1000, dtype=np.int32)); m2 = D.empty((1000,), np.int32); D.memcpy(m2, m1); D.memset(m1, 0); D.sync()
allok &= chk("memcpy", m2.np, np.arange(1000, dtype=np.int32)) and chk("memset", m1.np, np.zeros(1000, np.int32))
print("ALL PASS" if allok else "SOME FAILED")
sys.exit(0 if allok else 1)
