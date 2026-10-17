"""Bit-exact tests for cuda/attn.cu (det_attention) against detref/model.py attention_core.
Run: DET_LIB=cuda/libdet_attn.so .venv/bin/python tests/test_attn.py [--bench [--bench-long]]
run_tests() covers the default dispatch (CUDA-core path for M < 16, tensor-core path above); run_tc_tests() forces the
tensor-core path (env DET_ATTN_TC=1) for every size; DET_ATTN_TC=0 forces the CUDA-core path."""
import sys, os, time, numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "detref"))
import detcuda as D
import detmath as DM
from model import attention_core

f32 = np.float32
rng = np.random.RandomState(0)
ROT, THETA, EPS = 64, 1e7, 1e-6

def chk(name, a, b):
    a = np.asarray(a); b = np.asarray(b)
    if a.dtype.kind == 'f':
        ok = a.shape == b.shape and np.array_equal(a.view(np.uint32), b.view(np.uint32))
    else:
        ok = np.array_equal(a, b)
    msg = ""
    if not ok and a.shape == b.shape:
        d = np.abs(a.astype(np.float64) - b.astype(np.float64))
        msg = f"(max diff {d.max()}, n_diff {np.sum(a != b)} / {a.size})"
    print(("PASS" if ok else "FAIL"), name, msg, flush=True)
    return ok

def rope_tables(P, rot=ROT, theta=THETA):
    inv_freq = 1.0 / (theta ** (np.arange(0, rot, 2, dtype=np.float64) / rot))
    ang = np.arange(P, dtype=np.float64)[:, None] * inv_freq[None, :]
    return np.cos(ang).astype(f32), np.sin(ang).astype(f32)

class Setup:
    """random layer params + KV cache on both sides"""
    def __init__(self, n_h, n_kv, hd, capacity, qnorm_scale=1.0, seed=0):
        r = np.random.RandomState(seed)
        self.n_h, self.n_kv, self.hd, self.cap = n_h, n_kv, hd, capacity
        self.q_norm = (qnorm_scale * (1 + 0.3 * r.randn(hd))).astype(f32)
        self.k_norm = (1 + 0.3 * r.randn(hd)).astype(f32)
        self.v_scale = (10 ** r.uniform(-4, -2, size=n_kv)).astype(f32)
        self.v_inv_scale = (f32(1) / self.v_scale).astype(f32)
        self.cos, self.sin = rope_tables(capacity)
        dev = lambda a: D.to_dev(a)
        self.d = dict(q_norm=dev(self.q_norm), k_norm=dev(self.k_norm), v_scale=dev(self.v_scale), v_inv_scale=dev(self.v_inv_scale),
                      cos=dev(self.cos), sin=dev(self.sin))
        self.L = D.AttnLayer(D.P(self.d["q_norm"]), D.P(self.d["k_norm"]), D.P(self.d["v_scale"]), D.P(self.d["v_inv_scale"]))
        self.kq = torch.zeros(capacity, n_kv, hd, dtype=torch.int8, device="cuda")
        self.sk = torch.zeros(capacity, n_kv, device="cuda")
        self.vq = torch.zeros(capacity, n_kv, hd, dtype=torch.int16, device="cuda")
        self.kv = D.KVCache(D.P(self.kq), D.P(self.sk), D.P(self.vq), capacity)
        self.st = dict(kq=[], sk=[], vq=[])
        self.pos = 0
        self.scratch = {}

    def random_inputs(self, M, r=rng, vbig=False):
        qg = (r.randn(M, self.n_h * 2 * self.hd) * 2).astype(f32)
        k = (r.randn(M, self.n_kv * self.hd) * 2).astype(f32)
        v = (r.randn(M, self.n_kv * self.hd) * 0.5).astype(f32)
        if vbig:  # exercise int16 clipping
            idx = r.rand(*v.shape) < 0.01
            v[idx] = (r.randn(idx.sum()) * 1e3).astype(f32)
        return qg, k, v

    def prefill_cache(self, n, r=rng):
        """fill positions [0, n) with random valid cache contents on both sides"""
        kq = r.randint(-127, 128, size=(n, self.n_kv, self.hd)).astype(np.int8)
        sk = (10 ** r.uniform(-3, -1, size=(n, self.n_kv))).astype(f32)
        vq = r.randint(-32767, 32768, size=(n, self.n_kv, self.hd)).astype(np.int16)
        self.kq[:n] = D.to_dev(kq); self.sk[:n] = D.to_dev(sk); self.vq[:n] = D.to_dev(vq)
        for i in range(n):
            self.st["kq"].append(kq[i]); self.st["sk"].append(sk[i]); self.st["vq"].append(vq[i])
        self.pos = n

    def step(self, qg, k, v, check=True, label=""):
        M = qg.shape[0]
        pos0 = self.pos
        if M not in self.scratch:
            self.scratch[M] = torch.empty(D.attn_scratch_bytes(M, self.n_h, self.hd), dtype=torch.uint8, device="cuda")
        o = torch.empty(M, self.n_h * self.hd, device="cuda")
        D.attention(self.L, self.kv, D.to_dev(qg), D.to_dev(k), D.to_dev(v), self.d["cos"], self.d["sin"], ROT, EPS,
                    pos0, self.n_h, self.n_kv, self.hd, o, self.scratch[M])
        torch.cuda.synchronize()
        ok = True
        if check:
            pos = np.arange(pos0, pos0 + M)
            oref = attention_core(qg, k, v, pos, self.st, self.q_norm, self.k_norm, self.v_scale, self.v_inv_scale,
                                  self.cos, self.sin, ROT, EPS, self.n_h, self.n_kv, self.hd)
            tag = f"{label} M={M} pos0={pos0} n_h={self.n_h}"
            ok &= chk(f"o   {tag}", o.cpu().numpy(), oref)
            ok &= chk(f"kq  {tag}", self.kq[pos0:pos0 + M].cpu().numpy(), np.stack(self.st["kq"][pos0:pos0 + M]))
            ok &= chk(f"sk  {tag}", self.sk[pos0:pos0 + M].cpu().numpy(), np.stack(self.st["sk"][pos0:pos0 + M]))
            ok &= chk(f"vq  {tag}", self.vq[pos0:pos0 + M].cpu().numpy(), np.stack(self.st["vq"][pos0:pos0 + M]))
            # whole cache must be untouched elsewhere
            n = pos0 + M
            ok &= chk(f"kq* {tag}", self.kq[:n].cpu().numpy(), np.stack(self.st["kq"]))
        self.pos += M
        return ok

def cross_check(S, M, label=""):
    """same inputs through the CUDA-core path (DET_ATTN_TC=0) and the tensor-core path (DET_ATTN_TC=1); both are
    bit-exact w.r.t. the numpy reference, so they must agree bitwise (used where numpy is too slow)"""
    qg, k, v = S.random_inputs(M, vbig=True)
    pos0 = S.pos
    qgd, kd, vd = D.to_dev(qg), D.to_dev(k), D.to_dev(v)
    scratch = torch.empty(D.attn_scratch_bytes(M, S.n_h, S.hd), dtype=torch.uint8, device="cuda")
    outs = []
    for mode in ["0", "1"]:
        os.environ["DET_ATTN_TC"] = mode
        o = torch.empty(M, S.n_h * S.hd, device="cuda")
        D.attention(S.L, S.kv, qgd, kd, vd, S.d["cos"], S.d["sin"], ROT, EPS, pos0, S.n_h, S.n_kv, S.hd, o, scratch)
        torch.cuda.synchronize()
        outs.append(o.cpu().numpy())
    os.environ["DET_ATTN_TC"] = "1"
    ok = chk(f"o   {label} M={M} pos0={pos0} n_h={S.n_h} (tensor-core vs CUDA-core)", outs[1], outs[0])
    S.pos += M
    return ok

def run_tc_tests():
    """tensor-core prefill path (forced for every M via DET_ATTN_TC=1)"""
    allok = True
    os.environ["DET_ATTN_TC"] = "1"
    for n_h in [24, 16]:
        for M in [64, 300, 1024]:
            S = Setup(n_h, 4, 256, 2048, seed=n_h * 1000 + M)
            allok &= S.step(*S.random_inputs(M, vbig=True), label="tc-prefill")
            allok &= S.step(*S.random_inputs(1), label="tc-decode")      # M=1 through the tensor-core path too
        S = Setup(n_h, 4, 256, 2048, seed=n_h * 1000 + 2048)
        allok &= cross_check(S, 2048, label="tc-prefill")
        # long context: random pre-filled cache (same contents on both sides)
        S = Setup(n_h, 4, 256, 32768 + 2048 + 64, seed=n_h * 1000 + 7)
        S.prefill_cache(32768, r=np.random.RandomState(n_h))
        allok &= S.step(*S.random_inputs(64, vbig=True), label="tc-long")   # numpy (~32k keys x 64 queries)
        allok &= cross_check(S, 2048, label="tc-long")
    # peaked scores (most pq underflow to 0; p2/p3 slices only around the max)
    for M in [64, 300]:
        S = Setup(24, 4, 256, 1024, qnorm_scale=60.0, seed=11 + M)
        allok &= S.step(*S.random_inputs(M), label="tc-peaked")
    S = Setup(24, 4, 256, 8192 + 64, qnorm_scale=200.0, seed=12)
    S.prefill_cache(5000)
    allok &= S.step(*S.random_inputs(64), label="tc-peaked-long")
    # degenerate rows: zero q rows (amax == 0 -> s = 1), zero k rows, zero v rows, and 8 identical keys (ties at the
    # max -> several pq == 2^24 in one tile, exercising the p3 byte slice with multiple hits)
    S = Setup(24, 4, 256, 128, seed=13)
    qg, k, v = S.random_inputs(64)
    qg[1, :256] = 0; qg[40, 256 * 2 * 3:256 * 2 * 3 + 256] = 0; k[2, :256] = 0; k[50] = 0; v[0] = 0; v[33] = 0
    k[8:16] = k[8]
    allok &= S.step(qg, k, v, label="tc-zeros-ties")
    S = Setup(24, 4, 256, 64, seed=14)
    qg, k, v = S.random_inputs(17)
    qg[:, :] = 0
    allok &= S.step(qg, k, v, label="tc-allzero-q")
    del os.environ["DET_ATTN_TC"]
    print("TC ALL PASS" if allok else "TC SOME FAILED")
    return allok

def run_tests():
    allok = True
    for n_h in [24, 16]:
        for M in [1, 5, 64, 300]:
            S = Setup(n_h, 4, 256, 1024, seed=n_h * 100 + M)
            allok &= S.step(*S.random_inputs(M, vbig=True), label="prefill")
            allok &= S.step(*S.random_inputs(1), label="decode")
            allok &= S.step(*S.random_inputs(1), label="decode")
            allok &= S.step(*S.random_inputs(17), label="chunk")
    # long context: cache filled with random valid contents, then decode + a small chunk
    S = Setup(24, 4, 256, 8192 + 64, seed=7)
    S.prefill_cache(8191)
    allok &= S.step(*S.random_inputs(1), label="long-decode")
    allok &= S.step(*S.random_inputs(3), label="long-chunk")
    # very peaked scores (q_norm scaled up -> exp underflows to 0 for most keys)
    S = Setup(24, 4, 256, 1024, qnorm_scale=60.0, seed=11)
    allok &= S.step(*S.random_inputs(40), label="peaked-prefill")
    allok &= S.step(*S.random_inputs(1), label="peaked-decode")
    S = Setup(24, 4, 256, 8192 + 64, qnorm_scale=200.0, seed=12)
    S.prefill_cache(5000)
    allok &= S.step(*S.random_inputs(2), label="peaked-long")
    # degenerate inputs: all-zero q row (amax == 0 -> s = 1), zero v
    S = Setup(24, 4, 256, 64, seed=13)
    qg, k, v = S.random_inputs(3)
    qg[1, :256] = 0; k[2, :256] = 0; v[0] = 0
    allok &= S.step(qg, k, v, label="zeros")
    print("ALL PASS" if allok else "SOME FAILED")
    return allok

def run_dev_tests():
    """det_attention_dev (pos0 from device memory, grid sized for max_pos) must be bit-identical to det_attention and
    must be CUDA-graph capturable with the position changed between replays"""
    allok = True
    def one(S, M, max_pos, label, graph=None):
        qg, k, v = S.random_inputs(M, vbig=True)
        pos0 = S.pos
        qgd, kd, vd = D.to_dev(qg), D.to_dev(k), D.to_dev(v)
        scratch = torch.empty(D.attn_scratch_bytes(M, S.n_h, S.hd), dtype=torch.uint8, device="cuda")
        o_ref = torch.empty(M, S.n_h * S.hd, device="cuda")
        D.attention(S.L, S.kv, qgd, kd, vd, S.d["cos"], S.d["sin"], ROT, EPS, pos0, S.n_h, S.n_kv, S.hd, o_ref, scratch)
        torch.cuda.synchronize()
        rows_ref = [t[pos0:pos0 + M].clone() for t in (S.kq, S.sk, S.vq)]
        for t in (S.kq, S.sk, S.vq): t[pos0:pos0 + M].zero_()
        pos_dev = torch.full((1,), pos0, dtype=torch.int32, device="cuda")
        o_dev = torch.empty(M, S.n_h * S.hd, device="cuda")
        D.attention_dev(S.L, S.kv, qgd, kd, vd, S.d["cos"], S.d["sin"], ROT, EPS, pos_dev, max_pos, S.n_h, S.n_kv, S.hd, o_dev, scratch)
        torch.cuda.synchronize()
        tag = f"{label} M={M} pos0={pos0} max_pos={max_pos} n_h={S.n_h}"
        ok = chk(f"o   dev {tag}", o_dev.cpu().numpy(), o_ref.cpu().numpy())
        for nm, t, rr in zip(("kq", "sk", "vq"), (S.kq, S.sk, S.vq), rows_ref):
            ok &= chk(f"{nm}  dev {tag}", t[pos0:pos0 + M].cpu().numpy(), rr.cpu().numpy())
        S.pos += M
        return ok
    for n_h in [24, 16]:
        S = Setup(n_h, 4, 256, 4096 + 64, seed=n_h * 10 + 1)
        allok &= one(S, 1, 4095, "dev-first")                 # pos0 = 0: a single key, grid sized for 4096
        allok &= one(S, 1, 4095, "dev-second")
        allok &= one(S, 5, 4096 + 64 - 5, "dev-chunk")
        S.prefill_cache(2000, r=np.random.RandomState(n_h))
        allok &= one(S, 1, 4095, "dev-long")
        allok &= one(S, 1, 2001, "dev-tight")                  # max_pos == pos0: no idle blocks
    # peaked scores (exp underflow) and a zero q row through the dev path
    S = Setup(24, 4, 256, 1024, qnorm_scale=60.0, seed=21)
    S.prefill_cache(500, r=np.random.RandomState(21))
    allok &= one(S, 1, 1023, "dev-peaked")
    # CUDA graph: capture once with pos0 read from device memory, replay at several positions
    S = Setup(24, 4, 256, 2048, seed=22)
    S.prefill_cache(300, r=np.random.RandomState(22))
    M = 1
    qg, k, v = S.random_inputs(M, vbig=True)
    qgd, kd, vd = D.to_dev(qg), D.to_dev(k), D.to_dev(v)
    scratch = torch.empty(D.attn_scratch_bytes(M, S.n_h, S.hd), dtype=torch.uint8, device="cuda")
    pos_pin = torch.zeros(1, dtype=torch.int32, pin_memory=True)
    pos_dev = torch.zeros(1, dtype=torch.int32, device="cuda")
    o_g = torch.empty(M, S.n_h * S.hd, device="cuda")
    st = torch.cuda.Stream()
    with torch.cuda.stream(st):
        D.attention_dev(S.L, S.kv, qgd, kd, vd, S.d["cos"], S.d["sin"], ROT, EPS, pos_dev, 2047, S.n_h, S.n_kv, S.hd, o_g, scratch, st)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=st):
            pos_dev.copy_(pos_pin, non_blocking=True)
            D.attention_dev(S.L, S.kv, qgd, kd, vd, S.d["cos"], S.d["sin"], ROT, EPS, pos_dev, 2047, S.n_h, S.n_kv, S.hd, o_g, scratch, st)
    torch.cuda.synchronize()
    for pos0 in [300, 301, 302, 1000, 2047]:
        o_ref = torch.empty(M, S.n_h * S.hd, device="cuda")
        D.attention(S.L, S.kv, qgd, kd, vd, S.d["cos"], S.d["sin"], ROT, EPS, pos0, S.n_h, S.n_kv, S.hd, o_ref, scratch)
        torch.cuda.synchronize()
        rows_ref = [t[pos0:pos0 + M].clone() for t in (S.kq, S.sk, S.vq)]
        for t in (S.kq, S.sk, S.vq): t[pos0:pos0 + M].zero_()
        pos_pin[0] = pos0
        g.replay(); torch.cuda.synchronize()
        tag = f"graph M={M} pos0={pos0} n_h={S.n_h}"
        allok &= chk(f"o   dev {tag}", o_g.cpu().numpy(), o_ref.cpu().numpy())
        for nm, t, rr in zip(("kq", "sk", "vq"), (S.kq, S.sk, S.vq), rows_ref):
            allok &= chk(f"{nm}  dev {tag}", t[pos0:pos0 + M].cpu().numpy(), rr.cpu().numpy())
    print("DEV ALL PASS" if allok else "DEV SOME FAILED")
    return allok

def bench():
    def timeit(S, M, pos0, iters=10):
        qg, k, v = S.random_inputs(M)
        qgd, kd, vd = D.to_dev(qg), D.to_dev(k), D.to_dev(v)
        o = torch.empty(M, S.n_h * S.hd, device="cuda")
        scratch = torch.empty(D.attn_scratch_bytes(M, S.n_h, S.hd), dtype=torch.uint8, device="cuda")
        run = lambda: D.attention(S.L, S.kv, qgd, kd, vd, S.d["cos"], S.d["sin"], ROT, EPS, pos0, S.n_h, S.n_kv, S.hd, o, scratch)
        run(); torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(iters): run()
        e1.record(); torch.cuda.synchronize()
        return e0.elapsed_time(e1) / iters
    S = Setup(24, 4, 256, 32768 + 2048, seed=1)
    S.prefill_cache(32768, r=np.random.RandomState(1))
    print(f"decode  M=1    pos0=32768 : {timeit(S, 1, 32768):.3f} ms/layer")
    print(f"decode  M=1    pos0=4096  : {timeit(S, 1, 4096):.3f} ms/layer")
    print(f"prefill M=2048 pos0=0     : {timeit(S, 2048, 0, iters=3):.3f} ms/layer")
    print(f"prefill M=2048 pos0=32768 : {timeit(S, 2048, 32768, iters=3):.3f} ms/layer")
    print(f"chunk   M=128  pos0=32768 : {timeit(S, 128, 32768):.3f} ms/layer")
    print(f"chunk   M=64   pos0=32768 : {timeit(S, 64, 32768):.3f} ms/layer")
    print(f"prefill M=1024 pos0=32768 : {timeit(S, 1024, 32768, iters=3):.3f} ms/layer")
    if "--bench-long" in sys.argv:
        # 131072-token context: the cache is filled on the GPU (no host copy); timing only
        S = Setup(24, 4, 256, 131072 + 2048, seed=2)
        g = torch.Generator(device="cuda"); g.manual_seed(2)
        S.kq.random_(-127, 128, generator=g)
        S.sk.uniform_(-3, -1, generator=g); S.sk.copy_(torch.pow(10.0, S.sk))
        S.vq.random_(-32767, 32768, generator=g)
        print(f"chunk   M=128  pos0=131072: {timeit(S, 128, 131072):.3f} ms/layer")
        print(f"prefill M=2048 pos0=131072: {timeit(S, 2048, 131072, iters=3):.3f} ms/layer")

if __name__ == "__main__":
    if "--bench" in sys.argv:
        bench()
    else:
        ok = run_tests()
        if "--no-tc" not in sys.argv: ok &= run_tc_tests()
        ok &= run_dev_tests()
        if "--bench-after" in sys.argv: bench()
        sys.exit(0 if ok else 1)
