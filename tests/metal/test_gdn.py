"""Bit-exact test of the Metal det_gdn against detref/model.py: gdn_core (port of tests/test_gdn.py).
Run: python tests/metal/test_gdn.py [--bench]"""
import sys, os, time, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "detref"))
import detmetal as D
from model import gdn_core, QSCALE_GDN
f32 = np.float32
EPS = 1e-6
assert QSCALE_GDN.view(np.uint32) == 0x3DB504F3

def chk(name, a, b):
    a = np.asarray(a); b = np.asarray(b)
    ok = a.shape == b.shape and np.array_equal(a.view(np.uint32), b.view(np.uint32))
    if not ok:
        d = np.abs(a.astype(np.float64) - b.astype(np.float64))
        print("FAIL", name, f"(max diff {d.max()}, n_diff {np.sum(a.view(np.uint32) != b.view(np.uint32))} of {a.size})")
    else: print("PASS", name)
    return ok

class Case:
    def __init__(self, n_k, n_v, dk, dv, seed):
        self.n_k, self.n_v, self.dk, self.dv = n_k, n_v, dk, dv
        self.conv_dim = 2 * n_k * dk + n_v * dv
        rng = np.random.RandomState(seed); self.rng = rng
        self.conv = (0.3 * rng.randn(self.conv_dim, 4)).astype(f32)
        self.A = rng.uniform(0.05, 16.0, size=n_v).astype(f32)
        self.dt_bias = rng.randn(n_v).astype(f32)
        self.norm = (1 + 0.1 * rng.randn(dv)).astype(f32)
        self.dev = dict(conv=D.to_dev(self.conv), A=D.to_dev(self.A), dt_bias=D.to_dev(self.dt_bias), norm=D.to_dev(self.norm))
        self.L = D.GdnLayer(D.P(self.dev["conv"]), D.P(self.dev["A"]), D.P(self.dev["dt_bias"]), D.P(self.dev["norm"]))
        self.reset()
    def reset(self):
        self.st = dict(conv=np.zeros((3, self.conv_dim), dtype=f32), S=np.zeros((self.n_v, self.dk, self.dv), dtype=f32))
        self.conv_state_d = D.zeros((3, self.conv_dim)); self.S_d = D.zeros((self.n_v, self.dk, self.dv))
        self.S = D.GdnState(D.P(self.conv_state_d), D.P(self.S_d))
    def inputs(self, M, scale=2.0):
        r = self.rng
        return ((r.randn(M, self.conv_dim) * scale).astype(f32), (r.randn(M, self.n_v * self.dv) * scale).astype(f32),
                (r.randn(M, self.n_v) * scale).astype(f32), (r.randn(M, self.n_v) * scale).astype(f32))
    def run_dev(self, qkv, z, a, b):
        M = qkv.shape[0]
        on = D.empty((M, self.n_v * self.dv))
        scratch = D.empty((D.gdn_scratch_bytes(M, self.n_k, self.n_v, self.dk, self.dv),), np.uint8)
        D.gdn(self.L, self.S, D.to_dev(qkv), D.to_dev(z), D.to_dev(a), D.to_dev(b), EPS, self.n_k, self.n_v, self.dk, self.dv, on, scratch)
        D.sync()
        return on.np.copy()
    def run_ref(self, qkv, z, a, b):
        return gdn_core(qkv, z, a, b, self.st, self.conv, self.A, self.dt_bias, self.norm, EPS, self.n_k, self.n_v, self.dk, self.dv)
    def step(self, M, tag, scale=2.0):
        qkv, z, a, b = self.inputs(M, scale)
        on_d = self.run_dev(qkv, z, a, b); on_r = self.run_ref(qkv, z, a, b)
        ok = chk(f"{tag} M={M} on", on_d, on_r)
        ok &= chk(f"{tag} M={M} S", self.S_d.np, self.st["S"])
        ok &= chk(f"{tag} M={M} conv_state", self.conv_state_d.np, self.st["conv"])
        return ok

def main():
    allok = True
    for name, n_k, n_v in [("4B", 16, 32), ("27B", 16, 48)]:
        c = Case(n_k, n_v, 128, 128, seed=1 if name == "4B" else 2)
        for M in [1, 4, 37, 256]:
            c.reset(); allok &= c.step(M, f"{name} fresh")
            for i in range(3): allok &= c.step(1, f"{name} after{M} step{i}")
        c.reset(); allok &= c.step(2000, f"{name} long"); allok &= c.step(1, f"{name} long+1")
        S = c.S_d.np; nz = S[S != 0]
        print(f"   {name} long-run |S| min nonzero {np.abs(nz).min():.3e}, denormals {np.sum((np.abs(nz) < 1.1754944e-38))}, max {np.abs(S).max():.3e}")
        c.reset(); allok &= c.step(64, f"{name} predecay")
        qkv, z, a, b = c.inputs(300)
        b[:] = -100.0
        a[:] = np.linspace(-3.0, 4.0, n_v, dtype=f32)[None, :] - c.dt_bias[None, :]
        on_d = c.run_dev(qkv, z, a, b); on_r = c.run_ref(qkv, z, a, b)
        allok &= chk(f"{name} decay M=300 on", on_d, on_r)
        allok &= chk(f"{name} decay M=300 S", c.S_d.np, c.st["S"])
        S = c.S_d.np; nz = S[S != 0]
        print(f"   {name} decay-run |S| min nonzero {np.abs(nz).min():.3e}, denormals {np.sum((np.abs(nz) < 1.1754944e-38))}, zeros {np.sum(S == 0)} of {S.size}")
        allok &= c.step(1, f"{name} decay+1")
        if name == "4B":
            c.reset(); allok &= c.step(2100, f"{name} chunked"); allok &= c.step(1, f"{name} chunked+1")
    print("ALL PASS" if allok else "SOME FAILED")
    return allok

def bench():
    for name, n_k, n_v in [("4B", 16, 32), ("27B", 16, 48)]:
        c = Case(n_k, n_v, 128, 128, seed=3)
        for M in [1, 64, 1024]:
            qkv, z, a, b = c.inputs(M)
            qd, zd, ad, bd = D.to_dev(qkv), D.to_dev(z), D.to_dev(a), D.to_dev(b)
            on = D.empty((M, n_v * 128)); scratch = D.empty((D.gdn_scratch_bytes(M, n_k, n_v, 128, 128),), np.uint8)
            def go(): D.gdn(c.L, c.S, qd, zd, ad, bd, EPS, n_k, n_v, 128, 128, on, scratch)
            for _ in range(3): go()
            D.sync()
            iters = 200 if M == 1 else (50 if M == 64 else 10)
            t0 = time.perf_counter()
            for _ in range(iters): go()
            D.sync(); dt = (time.perf_counter() - t0) / iters
            print(f"{name} M={M}: {dt*1e6:9.1f} us per call, {dt/M*1e6:8.3f} us per token")

if __name__ == "__main__":
    if "--bench" in sys.argv: bench()
    else: sys.exit(0 if main() else 1)
