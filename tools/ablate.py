"""Perplexity ablations (NOT bit-exact; float32 BLAS) to attribute quantization loss.
Usage: ablate.py <det_model_dir> <hf_ppl.npz> --act {i8,i8g,none} --attn {int,float} [--n 1024] [--chunk 128]"""
import sys, os, time, argparse, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "detref"))
import model as M
from detmath import f32, exp_f, sigmoid_f, rmsnorm, quant_act_i8
from quant import dequant
ap = argparse.ArgumentParser(); ap.add_argument("d"); ap.add_argument("npz"); ap.add_argument("--act", default="i8"); ap.add_argument("--attn", default="int")
ap.add_argument("--requant", default="", help="re-quantize dequantized weights: q6k | w6g16 | w6g32 | w6g128 | w5g16 | w4g16 ...")
ap.add_argument("--keep8", default="", help="comma list of linear name suffixes kept at W8 (not re-quantized), e.g. k,v,down")
ap.add_argument("--n", type=int, default=1024); ap.add_argument("--chunk", type=int, default=128); ap.add_argument("--tag", default="")
args = ap.parse_args()

def act(x):
    x = np.asarray(x, dtype=f32)
    if args.act == "none": return x
    if args.act == "i8":
        q, s = quant_act_i8(x); return (q.astype(f32) * s[..., None]).astype(f32)
    if args.act == "i8g":
        sh = x.shape; xg = x.reshape(-1, sh[-1] // 128, 128)
        q, s = quant_act_i8(xg); return (q.astype(f32) * s[..., None]).reshape(sh).astype(f32)
    raise ValueError(args.act)

def requant(W, scheme):
    """Q6_K-like: superblock 256, sub-block `g`, int8 sub-scales under a per-superblock float scale (q6k),
    or wNgG: N-bit symmetric codes with exact float scales per group G (no sub-scale quantization)."""
    W = W.astype(np.float64); N, K = W.shape
    if scheme == "q6k":
        bits, g, sub8 = 6, 16, True
    else:
        bits, g, sub8 = int(scheme[1]), int(scheme.split("g")[1]), False
    qmax = (1 << (bits - 1)) - 1
    Wg = W.reshape(N, K // g, g)
    amax = np.abs(Wg).max(-1); amax = np.where(amax == 0, 1e-12, amax)
    best_err = None; best_s = None
    for f in np.linspace(0.80, 1.0, 11):
        s_ = amax / qmax * f
        q = np.clip(np.rint(Wg / s_[..., None]), -qmax, qmax)
        err = ((q * s_[..., None] - Wg) ** 2).sum(-1)
        if best_err is None: best_err, best_s = err, s_
        else:
            b = err < best_err; best_s = np.where(b, s_, best_s); best_err = np.where(b, err, best_err)
    s_ = best_s
    if sub8:   # quantize sub-scales to int8 relative to the superblock (256) max scale, like Q6_K
        ss = s_.reshape(N, K // 256, 256 // g)
        d = ss.max(-1, keepdims=True) / 127.0
        sc = np.clip(np.rint(ss / d), 1, 127)
        s_ = (d * sc).reshape(N, K // g)
    q = np.clip(np.rint(Wg / s_[..., None]), -qmax, qmax)
    return (q * s_[..., None]).reshape(N, K).astype(f32)

class LinearF(M.Linear):
    def __init__(self, t, name, info):
        super().__init__(t, name, info)
        self.Wd = dequant(dict(wq=self.wq, c=self.c, m=self.m)).astype(f32); del self.wq
        keep = [k for k in args.keep8.split(",") if k]
        if args.requant and info["N"] > 64 and not any(name.endswith("." + k) for k in keep):
            self.Wd = requant(self.Wd, args.requant)
    def __call__(self, x, sx=None, mx=None):   # x: dequantized float activations
        return (x @ self.Wd.T).astype(f32)
M.Linear = LinearF
M.quant_act_i8g = lambda x: (act(x), None, None)   # DetModel methods call quant_act_i8g then Linear(xq, sx, mx)

def attention_core_f(qg, k, v, pos, st, q_norm, k_norm, v_scale, v_inv_scale, rope_cos, rope_sin, rot, eps, n_h, n_kv, hd):
    T = qg.shape[0]; qg = qg.reshape(T, n_h, 2, hd); q, gate = qg[:, :, 0], qg[:, :, 1]
    k = k.reshape(T, n_kv, hd); v = v.reshape(T, n_kv, hd)
    q = rmsnorm(q, q_norm, eps); k = rmsnorm(k, k_norm, eps)
    q = M.rope(q, pos, rope_cos, rope_sin, rot); k = M.rope(k, pos, rope_cos, rope_sin, rot)
    q = q * f32(1.0 / np.sqrt(hd))
    if args.attn == "int":
        qq, sq = quant_act_i8(q); q = qq.astype(f32) * sq[..., None]
        kq, sk = quant_act_i8(k); k = kq.astype(f32) * sk[..., None]
        vq = np.clip(np.rint(v * v_inv_scale[None, :, None]), -32767, 32767); v = vq.astype(f32) * v_scale[None, :, None]
    rep = n_h // n_kv
    st.setdefault("k", []); st.setdefault("v", [])
    out = np.zeros((T, n_h, hd), dtype=f32)
    for t in range(T):
        st["k"].append(k[t]); st["v"].append(v[t])
        K_all = np.stack(st["k"]); V_all = np.stack(st["v"])
        for h in range(n_h):
            kh = h // rep
            s = K_all[:, kh] @ q[t, h]
            p = exp_f(s - s.max())
            if args.attn == "int": p = np.rint(p * M.P_FIX)
            out[t, h] = (p @ V_all[:, kh]) / p.sum()
    out = out * sigmoid_f(gate)
    return out.reshape(T, n_h * hd)
M.attention_core = attention_core_f

base = np.load(args.npz); ids = base["ids"][:args.n]; n = len(ids); has_ref = "nll" in base.files
m = M.DetModel(args.d)
t0 = time.time(); logits = []
for i in range(0, n, args.chunk):
    logits.append(m.forward(ids[i:i + args.chunk]))
logits = np.concatenate(logits)
nll = np.empty(n - 1)
for i in range(0, n - 1, 256):   # chunked log-softmax to bound memory
    lg = logits[i:min(i + 256, n - 1)].astype(np.float64); lg -= lg.max(-1, keepdims=True)
    lp = lg - np.log(np.exp(lg).sum(-1, keepdims=True)); j = np.arange(lg.shape[0])
    nll[i:i + lg.shape[0]] = -lp[j, ids[i + 1:i + 1 + lg.shape[0]]]
ref = f"(HF {np.exp(base['nll'][:n-1].mean()):.4f}), argmax agree {np.mean(logits.argmax(-1) == base['logits'][:n].argmax(-1)):.3f}" if has_ref else ""
print(f"ABLATION {args.tag} n={n} model={args.d} requant={args.requant or '-'} keep8={args.keep8 or '-'} act={args.act} attn={args.attn}: ppl {np.exp(nll.mean()):.4f} {ref}, {time.time()-t0:.0f}s", flush=True)
