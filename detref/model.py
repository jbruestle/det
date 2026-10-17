"""Deterministic reference forward pass for the Qwen3.5 architecture (the SPEC, executable).

Everything not a GEMM is explicit fp32 from detmath; GEMMs/attention are exact integer arithmetic.
Token mixers (attention, gated delta net) are evaluated strictly one token at a time; the linears
may be batched over tokens because their integer semantics are per-token anyway.
"""
import os, sys, json
import numpy as np
from safetensors.numpy import load_file
sys.path.insert(0, os.path.dirname(__file__))
from detmath import (f32, exp_f, softplus_f, sigmoid_f, silu_f, treesum, rsqrt_f, rmsnorm, l2norm,
                     quant_act_i8, quant_act_i8g, ONE, ZERO)
from quant import unpack, GROUP

QSCALE_GDN = f32(1.0 / np.sqrt(128.0))     # 0x3DB504F3
P_FIX = f32(16777216.0)                     # 2^24 fixed point for attention probabilities
I64 = np.int64

class Linear:
    def __init__(self, t, name, info):
        self.N, self.K, self.bits = info["N"], info["K"], info["bits"]
        self.wq = unpack(t[name + ".wq"], self.bits, self.K)          # int8 [N,K]
        self.c = t[name + ".c"].astype(f32)                           # [N]
        self.m = t[name + ".m"].astype(I64)                           # [N,G]
        self.G = self.K // GROUP
        self._wf = None

    def __call__(self, xq, sx, mx=None):
        """xq int8 [M,K], sx f32 [M], mx uint16 [M,G] (None = all ones) -> f32 [M,N]"""
        M = xq.shape[0]
        acc = np.zeros((M, self.N), dtype=I64)
        xf = xq.astype(f32)
        mxl = np.ones((M, self.G), dtype=I64) if mx is None else mx.astype(I64)
        for g in range(self.G):
            wg = self.wq[:, g * GROUP:(g + 1) * GROUP].astype(f32)      # [N,128]
            part = xf[:, g * GROUP:(g + 1) * GROUP] @ wg.T               # exact integers < 2^24
            acc += part.astype(I64) * (self.m[None, :, g] * mxl[:, g, None])
        y = (acc.astype(f32) * self.c[None, :]) * sx[:, None]
        return y.astype(f32)


def rope(x, pos, rope_cos, rope_sin, rot):
    """x [T, heads, hd]; rotate first `rot` dims (neox half-split) with table rows pos[T]."""
    half = rot // 2
    c = rope_cos[pos][:, None, :]; s = rope_sin[pos][:, None, :]
    a = x[..., :half]; b = x[..., half:rot]
    out = x.copy()
    out[..., :half] = (a * c) - (b * s)
    out[..., half:rot] = (b * c) + (a * s)
    return out


def attention_core(qg, k, v, pos, st, q_norm, k_norm, v_scale, v_inv_scale, rope_cos, rope_sin, rot, eps, n_h, n_kv, hd):
    """Full attention for T new tokens.  qg [T, n_h*2*hd] (per head: q | gate), k/v [T, n_kv*hd], pos [T].
    st = dict(kq=[], sk=[], vq=[]) KV cache lists (appended).  Returns gated o [T, n_h*hd] fp32."""
    T = qg.shape[0]
    qg = qg.reshape(T, n_h, 2, hd)
    q, gate = qg[:, :, 0], qg[:, :, 1]
    k = k.reshape(T, n_kv, hd); v = v.reshape(T, n_kv, hd)
    q = rmsnorm(q, q_norm, eps)
    k = rmsnorm(k, k_norm, eps)
    q = rope(q, pos, rope_cos, rope_sin, rot); k = rope(k, pos, rope_cos, rope_sin, rot)
    q = q * f32(1.0 / np.sqrt(hd))             # 0.0625 for hd=256 (exact power of two)
    qq, sq = quant_act_i8(q)                   # [T,n_h,hd], [T,n_h]
    kq, sk = quant_act_i8(k)                   # [T,n_kv,hd], [T,n_kv]
    vq = np.clip(np.rint(v * v_inv_scale[None, :, None]), -32767, 32767).astype(np.int16)
    rep = n_h // n_kv
    out = np.zeros((T, n_h, hd), dtype=f32)
    for t in range(T):
        st["kq"].append(kq[t]); st["sk"].append(sk[t]); st["vq"].append(vq[t])
        K_all = np.stack(st["kq"]); SK = np.stack(st["sk"]); V_all = np.stack(st["vq"])   # [n, n_kv, ...]
        for h in range(n_h):
            kh = h // rep
            dot = K_all[:, kh].astype(np.int32) @ qq[t, h].astype(np.int32)        # [n] exact
            s = (dot.astype(f32) * sq[t, h]) * SK[:, kh]
            mx = s.max()
            p = exp_f(s - mx)
            pq = np.rint(p * P_FIX).astype(I64)
            den = pq.sum()
            num = pq @ V_all[:, kh].astype(I64)                                      # [hd] exact
            out[t, h] = (num.astype(f32) / f32(den)) * v_scale[kh]
    out = out * sigmoid_f(gate)
    return out.reshape(T, n_h * hd)


def gdn_core(qkv, z, a, b, st, conv_w, A, dt_bias, norm_w, eps, n_k, n_v, dk, dv):
    """Gated DeltaNet for T tokens in order.  qkv [T, conv_dim], z [T, n_v*dv], a/b [T, n_v].
    st = dict(conv=[3, conv_dim] previous inputs oldest first, S=[n_v, dk, dv]) updated in place.
    Returns on [T, n_v*dv] fp32 (after gated rmsnorm * silu(z))."""
    T = qkv.shape[0]
    z = z.reshape(T, n_v, dv)
    w = conv_w                                   # [conv_dim, 4]
    beta = sigmoid_f(b)
    decay = exp_f(-(A[None, :] * softplus_f(a + dt_bias[None, :])))
    kd = n_k * dk
    rep = n_v // n_k
    out = np.zeros((T, n_v, dv), dtype=f32)
    for t in range(T):
        win = np.concatenate([st["conv"], qkv[t][None, :]], axis=0)   # [4, conv_dim]
        y = (w[:, 0] * win[0]) + (w[:, 1] * win[1])
        y = y + (w[:, 2] * win[2])
        y = y + (w[:, 3] * win[3])
        st["conv"] = win[1:].copy()
        y = silu_f(y)
        q = l2norm(y[:kd].reshape(n_k, dk), eps) * QSCALE_GDN
        k = l2norm(y[kd:2 * kd].reshape(n_k, dk), eps)
        v = y[2 * kd:].reshape(n_v, dv)
        S = st["S"]
        for h in range(n_v):
            kh = h // rep
            Sh = S[h] * decay[t, h]
            kv_mem = treesum(Sh * k[kh][:, None], axis=0)           # [dv]
            delta = (v[h] - kv_mem) * beta[t, h]
            Sh = Sh + (k[kh][:, None] * delta[None, :])
            S[h] = Sh
            out[t, h] = treesum(Sh * q[kh][:, None], axis=0)
    on = rmsnorm(out, norm_w, eps) * silu_f(z)
    return on.reshape(T, n_v * dv)


class DetModel:
    def __init__(self, d):
        self.meta = json.load(open(os.path.join(d, "model.json")))
        files = self.meta.get("files")
        if files:
            t = {}
            for f in files: t.update(load_file(os.path.join(d, "parts", f)))
        else:
            t = load_file(os.path.join(d, "tensors.safetensors"))
        vs_override = os.path.join(d, "v_scales.npz")
        if os.path.exists(vs_override):
            t.update(dict(np.load(vs_override)))
        m = self.meta
        self.H, self.L, self.V = m["hidden"], m["layers"], m["vocab"]
        self.eps = m["eps"]
        self.n_h, self.n_kv, self.hd, self.rot = m["n_heads"], m["n_kv"], m["head_dim"], m["rot_dim"]
        self.n_k, self.n_v, self.dk, self.dv, self.conv_k = m["gdn_k_heads"], m["gdn_v_heads"], m["gdn_dk"], m["gdn_dv"], m["conv_k"]
        self.embed = t["embed_bf16"]
        self.final_norm = t["final_norm"]
        self.rope_cos, self.rope_sin = t["rope_cos"], t["rope_sin"]
        lin = lambda n: Linear(t, n, m["linears"][n])
        self.lm_head = lin("lm_head")
        self.layers = []
        for i in range(self.L):
            n = f"L{i}."
            ly = dict(type=m["layer_types"][i], attn_norm=t[n + "attn_norm"], ffn_norm=t[n + "ffn_norm"],
                      gate=lin(n + "gate"), up=lin(n + "up"), down=lin(n + "down"))
            if ly["type"] == "full_attention":
                ly.update(q=lin(n + "q"), k=lin(n + "k"), v=lin(n + "v"), o=lin(n + "o"),
                          q_norm=t[n + "q_norm"], k_norm=t[n + "k_norm"],
                          v_scale=t[n + "v_scale"], v_inv_scale=t[n + "v_inv_scale"])
            else:
                ly.update(qkv=lin(n + "qkv"), z=lin(n + "z"), a=lin(n + "a"), b=lin(n + "b"), out=lin(n + "out"),
                          conv=t[n + "conv"], A=t[n + "A"], dt_bias=t[n + "dt_bias"], gdn_norm=t[n + "gdn_norm"])
            self.layers.append(ly)
        self.reset()

    def reset(self):
        self.pos = 0
        self.state = []
        for ly in self.layers:
            if ly["type"] == "full_attention":
                self.state.append(dict(kq=[], sk=[], vq=[]))
            else:
                self.state.append(dict(conv=np.zeros((self.conv_k - 1, ly["conv"].shape[0]), dtype=f32),
                                       S=np.zeros((self.n_v, self.dk, self.dv), dtype=f32)))

    # ------------------------------------------------------------ pieces
    def embed_tokens(self, ids):
        return (self.embed[ids].astype(np.uint32) << 16).view(f32)

    def attention(self, li, ly, xn, pos):
        xq, sx, mx = quant_act_i8g(xn)
        qg = ly["q"](xq, sx, mx); k = ly["k"](xq, sx, mx); v = ly["v"](xq, sx, mx)
        o = attention_core(qg, k, v, pos, self.state[li], ly["q_norm"], ly["k_norm"], ly["v_scale"], ly["v_inv_scale"],
                           self.rope_cos, self.rope_sin, self.rot, self.eps, self.n_h, self.n_kv, self.hd)
        oq, so, mo = quant_act_i8g(o)
        return ly["o"](oq, so, mo)

    def gdn(self, li, ly, xn):
        xq, sx, mx = quant_act_i8g(xn)
        qkv = ly["qkv"](xq, sx, mx); z = ly["z"](xq, sx, mx); b = ly["b"](xq, sx, mx); a = ly["a"](xq, sx, mx)
        on = gdn_core(qkv, z, a, b, self.state[li], ly["conv"], ly["A"], ly["dt_bias"], ly["gdn_norm"], self.eps,
                      self.n_k, self.n_v, self.dk, self.dv)
        oq, so, mo = quant_act_i8g(on)
        return ly["out"](oq, so, mo)

    def mlp(self, ly, xn):
        xq, sx, mx = quant_act_i8g(xn)
        h = silu_f(ly["gate"](xq, sx, mx)) * ly["up"](xq, sx, mx)
        hq, sh, mh = quant_act_i8g(h)
        return ly["down"](hq, sh, mh)

    # ------------------------------------------------------------ forward
    def forward(self, ids, return_all=True, verbose=False):
        """ids: list of token ids appended at current position. Returns logits [T,V] (or last only)."""
        ids = np.asarray(ids, dtype=np.int64)
        T = len(ids)
        pos = np.arange(self.pos, self.pos + T)
        x = self.embed_tokens(ids)
        for li, ly in enumerate(self.layers):
            xn = rmsnorm(x, ly["attn_norm"], self.eps)
            if ly["type"] == "full_attention":
                x = x + self.attention(li, ly, xn, pos)
            else:
                x = x + self.gdn(li, ly, xn)
            xn = rmsnorm(x, ly["ffn_norm"], self.eps)
            x = x + self.mlp(ly, xn)
            if verbose: print(f"layer {li} done", flush=True)
        self.pos += T
        xn = rmsnorm(x, self.final_norm, self.eps)
        if not return_all:
            xn = xn[-1:]
        xq, sx, mx = quant_act_i8g(xn)
        return self.lm_head(xq, sx, mx)
