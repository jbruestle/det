import sys, os, numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "detref"))
import detcuda as D
import detmath as DM
rng = np.random.RandomState(0)
f32 = np.float32
def chk(name, a, b):
    a = np.asarray(a); b = np.asarray(b)
    ok = np.array_equal(a, b) or (a.dtype.kind == 'f' and np.array_equal(a.view(np.uint32), b.view(np.uint32)))
    print(("PASS" if ok else "FAIL"), name, "" if ok else f"(max diff {np.abs(a.astype(np.float64)-b.astype(np.float64)).max()}, n_diff {np.sum(a!=b)})")
    return ok
allok = True
for n in [128, 256, 2560, 5120, 6144, 9216, 17408]:
    M = 7
    x = (rng.randn(M, n) * (rng.rand(M, 1) * 10)).astype(f32)
    w = (1 + 0.5 * rng.randn(n)).astype(f32)
    xd, wd = D.to_dev(x), D.to_dev(w)
    yd = torch.empty(M, n, device="cuda"); xq = torch.empty(M, n, dtype=torch.int8, device="cuda"); sx = torch.empty(M, device="cuda")
    mx = torch.empty(M, n // 128, dtype=torch.int16, device="cuda")
    if n <= 8192:
        D.rmsnorm_quant(xd, wd, 1e-6, yd, xq, sx); torch.cuda.synchronize()
        yr = DM.rmsnorm(x, w, 1e-6); qr, sr = DM.quant_act_i8(yr)
        allok &= chk(f"rmsnorm n={n}", yd.cpu().numpy(), yr)
        allok &= chk(f"rmsnorm quant n={n}", xq.cpu().numpy(), qr) and chk(f"rmsnorm sx n={n}", sx.cpu().numpy(), sr)
        D.rmsnorm_quant(xd, wd, 1e-6, None, xq, sx, mx); torch.cuda.synchronize()
        qr, sr, mr = DM.quant_act_i8g(yr)
        allok &= chk(f"rmsnorm gquant n={n}", xq.cpu().numpy(), qr) and chk(f"rmsnorm gsx n={n}", sx.cpu().numpy(), sr) and chk(f"rmsnorm gmx n={n}", mx.cpu().numpy().view(np.uint16), mr)
    D.quant(xd, xq, sx); torch.cuda.synchronize()
    qr, sr = DM.quant_act_i8(x)
    allok &= chk(f"quant n={n}", xq.cpu().numpy(), qr) and chk(f"quant sx n={n}", sx.cpu().numpy(), sr)
    D.quant(xd, xq, sx, mx); torch.cuda.synchronize()
    qr, sr, mr = DM.quant_act_i8g(x)
    allok &= chk(f"gquant n={n}", xq.cpu().numpy(), qr) and chk(f"gquant sx n={n}", sx.cpu().numpy(), sr) and chk(f"gquant mx n={n}", mx.cpu().numpy().view(np.uint16), mr)
    g = (rng.randn(M, n) * 5).astype(f32); u = (rng.randn(M, n) * 3).astype(f32)
    hd = torch.empty(M, n, device="cuda")
    D.silu_mul_quant(D.to_dev(g), D.to_dev(u), hd, xq, sx); torch.cuda.synchronize()
    hr = (DM.silu_f(g) * u).astype(f32); qr, sr = DM.quant_act_i8(hr)
    allok &= chk(f"silu_mul n={n}", hd.cpu().numpy(), hr) and chk(f"silu_mul quant n={n}", xq.cpu().numpy(), qr)
    hd.zero_(); D.silu_mul_quant(D.to_dev(g), D.to_dev(u), hd, xq, sx, mx); torch.cuda.synchronize()
    qr, sr, mr = DM.quant_act_i8g(hr)
    allok &= chk(f"silu_mul g h n={n}", hd.cpu().numpy(), hr) and chk(f"silu_mul gquant n={n}", xq.cpu().numpy(), qr) and chk(f"silu_mul gmx n={n}", mx.cpu().numpy().view(np.uint16), mr)
    # zero rows / zero groups
    x0 = x.copy(); x0[0] = 0; x0[1, :128] = 0
    D.quant(D.to_dev(x0), xq, sx, mx); torch.cuda.synchronize(); qr, sr, mr = DM.quant_act_i8g(x0)
    allok &= chk(f"gquant zeros n={n}", xq.cpu().numpy(), qr) and chk(f"gquant zeros mx n={n}", mx.cpu().numpy().view(np.uint16), mr) and chk(f"gquant zeros sx n={n}", sx.cpu().numpy(), sr)
# embed / add / argmax
tab = rng.randint(0, 65536, size=(100, 64)).astype(np.uint16); ids = rng.randint(0, 100, size=5).astype(np.int32)
xd = torch.empty(5, 64, device="cuda"); D.embed(D.to_dev(tab), D.to_dev(ids), xd); torch.cuda.synchronize()
allok &= chk("embed", xd.cpu().numpy(), (tab[ids].astype(np.uint32) << 16).view(f32))
a = rng.randn(1000).astype(f32); b = rng.randn(1000).astype(f32); ad = D.to_dev(a); D.add(ad, D.to_dev(b)); torch.cuda.synchronize()
allok &= chk("add", ad.cpu().numpy(), (a + b).astype(f32))
lg = rng.randn(3, 248320).astype(f32); lg[1, 5] = lg[1].max() + 1; lg[1, 777] = lg[1, 5]
od = torch.empty(3, dtype=torch.int32, device="cuda"); D.argmax(D.to_dev(lg), od); torch.cuda.synchronize()
allok &= chk("argmax", od.cpu().numpy(), lg.argmax(1).astype(np.int32))
print("ALL PASS" if allok else "SOME FAILED")
