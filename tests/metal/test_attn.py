"""Bit-exact tests for the Metal det_attention against detref/model.py attention_core (port of tests/test_attn.py).
Run: python tests/metal/test_attn.py [--bench]"""
import sys, os, time, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "detref"))
import detmetal as D
from model import attention_core
f32 = np.float32
rng = np.random.RandomState(0)
ROT, THETA, EPS = 64, 1e7, 1e-6

def chk(name, a, b):
    a = np.asarray(a); b = np.asarray(b)
    ok = a.shape == b.shape and (np.array_equal(a.view(np.uint32), b.view(np.uint32)) if a.dtype.kind == 'f' else np.array_equal(a, b))
    msg = ""
    if not ok and a.shape == b.shape:
        d = np.abs(a.astype(np.float64) - b.astype(np.float64)); msg = f"(max diff {d.max()}, n_diff {np.sum(a != b)} / {a.size})"
    print(("PASS" if ok else "FAIL"), name, msg, flush=True)
    return ok

def rope_tables(P, rot=ROT, theta=THETA):
    inv_freq = 1.0 / (theta ** (np.arange(0, rot, 2, dtype=np.float64) / rot))
    ang = np.arange(P, dtype=np.float64)[:, None] * inv_freq[None, :]
    return np.cos(ang).astype(f32), np.sin(ang).astype(f32)

class Setup:
    def __init__(self, n_h, n_kv, hd, capacity, qnorm_scale=1.0, seed=0):
        r = np.random.RandomState(seed)
        self.n_h, self.n_kv, self.hd, self.cap = n_h, n_kv, hd, capacity
        self.q_norm = (qnorm_scale * (1 + 0.3 * r.randn(hd))).astype(f32)
        self.k_norm = (1 + 0.3 * r.randn(hd)).astype(f32)
        self.v_scale = (10 ** r.uniform(-4, -2, size=n_kv)).astype(f32)
        self.v_inv_scale = (f32(1) / self.v_scale).astype(f32)
        self.cos, self.sin = rope_tables(capacity)
        self.d = dict(q_norm=D.to_dev(self.q_norm), k_norm=D.to_dev(self.k_norm), v_scale=D.to_dev(self.v_scale), v_inv_scale=D.to_dev(self.v_inv_scale),
                      cos=D.to_dev(self.cos), sin=D.to_dev(self.sin))
        self.L = D.AttnLayer(D.P(self.d["q_norm"]), D.P(self.d["k_norm"]), D.P(self.d["v_scale"]), D.P(self.d["v_inv_scale"]))
        self.kq = D.zeros((capacity, n_kv, hd), np.int8); self.sk = D.zeros((capacity, n_kv)); self.vq = D.zeros((capacity, n_kv, hd), np.int16)
        self.kv = D.KVCache(D.P(self.kq), D.P(self.sk), D.P(self.vq), capacity)
        self.st = dict(kq=[], sk=[], vq=[]); self.pos = 0; self.scratch = {}
    def random_inputs(self, M, r=rng, vbig=False):
        qg = (r.randn(M, self.n_h * 2 * self.hd) * 2).astype(f32)
        k = (r.randn(M, self.n_kv * self.hd) * 2).astype(f32)
        v = (r.randn(M, self.n_kv * self.hd) * 0.5).astype(f32)
        if vbig:
            idx = r.rand(*v.shape) < 0.01; v[idx] = (r.randn(idx.sum()) * 1e3).astype(f32)
        return qg, k, v
    def prefill_cache(self, n, r=rng):
        kq = r.randint(-127, 128, size=(n, self.n_kv, self.hd)).astype(np.int8)
        sk = (10 ** r.uniform(-3, -1, size=(n, self.n_kv))).astype(f32)
        vq = r.randint(-32767, 32768, size=(n, self.n_kv, self.hd)).astype(np.int16)
        D.sync(); self.kq.np[:n] = kq; self.sk.np[:n] = sk; self.vq.np[:n] = vq
        for i in range(n): self.st["kq"].append(kq[i]); self.st["sk"].append(sk[i]); self.st["vq"].append(vq[i])
        self.pos = n
    def step(self, qg, k, v, check=True, label=""):
        M = qg.shape[0]; pos0 = self.pos
        if M not in self.scratch: self.scratch[M] = D.empty((D.attn_scratch_bytes(M, self.n_h, self.hd),), np.uint8)
        o = D.empty((M, self.n_h * self.hd))
        D.attention(self.L, self.kv, D.to_dev(qg), D.to_dev(k), D.to_dev(v), self.d["cos"], self.d["sin"], ROT, EPS, pos0, self.n_h, self.n_kv, self.hd, o, self.scratch[M])
        D.sync()
        ok = True
        if check:
            pos = np.arange(pos0, pos0 + M)
            oref = attention_core(qg, k, v, pos, self.st, self.q_norm, self.k_norm, self.v_scale, self.v_inv_scale, self.cos, self.sin, ROT, EPS, self.n_h, self.n_kv, self.hd)
            tag = f"{label} M={M} pos0={pos0} n_h={self.n_h}"
            ok &= chk(f"o   {tag}", o.np, oref)
            ok &= chk(f"kq  {tag}", self.kq.np[pos0:pos0 + M], np.stack(self.st["kq"][pos0:pos0 + M]))
            ok &= chk(f"sk  {tag}", self.sk.np[pos0:pos0 + M], np.stack(self.st["sk"][pos0:pos0 + M]))
            ok &= chk(f"vq  {tag}", self.vq.np[pos0:pos0 + M], np.stack(self.st["vq"][pos0:pos0 + M]))
            n = pos0 + M
            ok &= chk(f"kq* {tag}", self.kq.np[:n], np.stack(self.st["kq"]))
        self.pos += M
        return ok

def run_tests():
    allok = True
    for n_h in [24, 16]:
        for M in [1, 5, 64, 300]:
            S = Setup(n_h, 4, 256, 1024, seed=n_h * 100 + M)
            allok &= S.step(*S.random_inputs(M, vbig=True), label="prefill")
            allok &= S.step(*S.random_inputs(1), label="decode")
            allok &= S.step(*S.random_inputs(1), label="decode")
            allok &= S.step(*S.random_inputs(17), label="chunk")
    S = Setup(24, 4, 256, 8192 + 64, seed=7)
    S.prefill_cache(8191)
    allok &= S.step(*S.random_inputs(1), label="long-decode")
    allok &= S.step(*S.random_inputs(3), label="long-chunk")
    S = Setup(24, 4, 256, 1024, qnorm_scale=60.0, seed=11)
    allok &= S.step(*S.random_inputs(40), label="peaked-prefill")
    allok &= S.step(*S.random_inputs(1), label="peaked-decode")
    S = Setup(24, 4, 256, 8192 + 64, qnorm_scale=200.0, seed=12)
    S.prefill_cache(5000)
    allok &= S.step(*S.random_inputs(2), label="peaked-long")
    S = Setup(24, 4, 256, 64, seed=13)
    qg, k, v = S.random_inputs(3)
    qg[1, :256] = 0; k[2, :256] = 0; v[0] = 0
    allok &= S.step(qg, k, v, label="zeros")
    S = Setup(8, 4, 128, 256, seed=15)      # hd=128 variant
    allok &= S.step(*S.random_inputs(33, vbig=True), label="hd128")
    allok &= S.step(*S.random_inputs(1), label="hd128-decode")
    print("ALL PASS" if allok else "SOME FAILED")
    return allok

def bench():
    for n_h, n_kv in [(24, 4)]:
        for ctx in [1024, 8192, 32768, 131072]:
            S = Setup(n_h, n_kv, 256, ctx + 64, seed=1)
            S.pos = ctx
            for M in [1, 64]:
                qg, k, v = S.random_inputs(M)
                qd, kd, vd = D.to_dev(qg), D.to_dev(k), D.to_dev(v)
                scratch = D.empty((D.attn_scratch_bytes(M, n_h, 256),), np.uint8); o = D.empty((M, n_h * 256))
                def go(): D.attention(S.L, S.kv, qd, kd, vd, S.d["cos"], S.d["sin"], ROT, EPS, ctx, n_h, n_kv, 256, o, scratch)
                for _ in range(2): go()
                D.sync(); iters = 20 if ctx < 32768 else 5; t0 = time.perf_counter()
                for _ in range(iters): go()
                D.sync(); dt = (time.perf_counter() - t0) / iters
                print(f"ctx={ctx:6d} M={M:3d}: {dt*1e3:8.3f} ms  ({dt/M*1e6:8.1f} us/token)")

if __name__ == "__main__":
    if "--bench" in sys.argv: bench()
    else: sys.exit(0 if run_tests() else 1)
