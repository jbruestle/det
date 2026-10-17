"""Metal implementation of the det model (orchestrates the kernels in metal/libdet_metal.dylib via detmetal.py).
Must produce logits bit-identical to detref/model.py (and to the CUDA runner).  Port of detmodel_cuda.py: the same
buffer plan and kernel sequence; no CUDA graphs (decode steps are encoded into one command buffer per forward)."""
import os, sys, json, time
import numpy as np
from safetensors import safe_open
sys.path.insert(0, os.path.dirname(__file__))
import detmetal as D
_gemm_scratch = None   # split-K partials for the decode GEMM (the library keeps the pointer)

class MetalModel:
    def __init__(self, d, max_ctx=8192, max_batch=256, verbose=False):
        self.meta = m = json.load(open(os.path.join(d, "model.json")))
        self.H, self.L, self.V = m["hidden"], m["layers"], m["vocab"]
        self.eps = float(m["eps"])
        self.n_h, self.n_kv, self.hd, self.rot = m["n_heads"], m["n_kv"], m["head_dim"], m["rot_dim"]
        self.n_k, self.n_v, self.dk, self.dv, self.conv_k = m["gdn_k_heads"], m["gdn_v_heads"], m["gdn_dk"], m["gdn_dv"], m["conv_k"]
        self.I = m["intermediate"]
        self.conv_dim = 2 * self.n_k * self.dk + self.n_v * self.dv
        self.max_ctx, self.max_batch = max_ctx, max_batch
        t0 = time.time()
        files = m.get("files")
        paths = [os.path.join(d, "parts", f) if os.path.exists(os.path.join(d, "parts", f)) else os.path.join(d, f) for f in files] if files else [os.path.join(d, "tensors.safetensors")]
        handles = [safe_open(pth, "np") for pth in paths]
        where = {}
        for h in handles:
            for k in h.keys(): where[k] = h
        def up(name): return D.to_dev(where[name].get_tensor(name))
        def lin(name):
            info = m["linears"][name]
            return D.make_qlinear(where[name + ".wq"].get_tensor(name + ".wq"), up(name + ".c"), up(name + ".m"), info["N"], info["K"], info["bits"])
        def lin_fused(names):
            infos = [m["linears"][n] for n in names]
            assert len(set(i["K"] for i in infos)) == 1 and len(set(i["bits"] for i in infos)) == 1, names
            wq = np.concatenate([where[n + ".wq"].get_tensor(n + ".wq") for n in names])
            c = np.concatenate([where[n + ".c"].get_tensor(n + ".c") for n in names])
            mm = np.concatenate([where[n + ".m"].get_tensor(n + ".m") for n in names])
            L = D.make_qlinear(wq, D.to_dev(c), D.to_dev(mm), int(sum(i["N"] for i in infos)), infos[0]["K"], infos[0]["bits"])
            L.widths = [i["N"] for i in infos]; return L
        vs_override = os.path.join(d, "v_scales.npz")
        self.v_override = dict(np.load(vs_override)) if os.path.exists(vs_override) else None
        self.v_hook = None   # optional callable(layer_idx, v numpy [M, n_kv*hd]) for calibration
        self.embed_host = where["embed_bf16"].get_tensor("embed_bf16")      # uint16 bits, stays in RAM
        self.final_norm = up("final_norm")
        self.rope_cos, self.rope_sin = up("rope_cos"), up("rope_sin")
        assert self.rope_cos.shape[0] >= max_ctx, "rope table too short for max_ctx"
        self.lm_head = lin("lm_head")
        self.layers = []
        for i in range(self.L):
            n = f"L{i}."
            ly = dict(type=m["layer_types"][i], attn_norm=up(n + "attn_norm"), ffn_norm=up(n + "ffn_norm"),
                      gu_f=lin_fused([n + "gate", n + "up"]), down=lin(n + "down"))
            if ly["type"] == "full_attention":
                ly.update(qkv_f=lin_fused([n + "q", n + "k", n + "v"]), o=lin(n + "o"))
                if self.v_override is not None:
                    vs = D.to_dev(self.v_override[f"L{i}.v_scale"]); vis = D.to_dev(self.v_override[f"L{i}.v_inv_scale"])
                else:
                    vs, vis = up(n + "v_scale"), up(n + "v_inv_scale")
                keep = [up(n + "q_norm"), up(n + "k_norm"), vs, vis]
                ly["al"] = D.AttnLayer(*[D.P(x) for x in keep]); ly["keep"] = keep
                kq = D.zeros((max_ctx, self.n_kv, self.hd), np.int8); sk = D.zeros((max_ctx, self.n_kv)); vq = D.zeros((max_ctx, self.n_kv, self.hd), np.int16)
                ly["kv"] = D.KVCache(D.P(kq), D.P(sk), D.P(vq), max_ctx); ly["kv_t"] = (kq, sk, vq)
            else:
                ly.update(qkvz_f=lin_fused([n + "qkv", n + "z"]), ab_f=lin_fused([n + "a", n + "b"]), out=lin(n + "out"))
                keep = [up(n + "conv"), up(n + "A"), up(n + "dt_bias"), up(n + "gdn_norm")]
                ly["gl"] = D.GdnLayer(*[D.P(x) for x in keep]); ly["keep"] = keep
                cs = D.zeros((self.conv_k - 1, self.conv_dim)); S = D.zeros((self.n_v, self.dk, self.dv))
                ly["gs"] = D.GdnState(D.P(cs), D.P(S)); ly["gs_t"] = (cs, S)
            self.layers.append(ly)
            if verbose: print(f"loaded layer {i}", flush=True)
        if verbose: print(f"model loaded in {time.time() - t0:.1f}s, {D.allocated_bytes() / 1e9:.2f} GB", flush=True)
        self._alloc(max_batch)
        global _gemm_scratch
        _gemm_scratch = D.empty((D.gemm_scratch_bytes(8, max(self.V, max(int(v["N"]) for v in m["linears"].values()) * 2)),), np.uint8)
        D.gemm_set_scratch(_gemm_scratch)
        self.reset()

    def _alloc(self, M):
        f = lambda *s: D.empty(s, np.float32)
        i8 = lambda *s: D.empty(s, np.int8)
        u16 = lambda *s: D.empty(s, np.uint16)
        self.b = dict(
            x=f(M, self.H), xq=i8(M, self.H), sx=f(M), mx=u16(M, self.H // 128), r=f(M, self.H),
            qg=f(M, self.n_h * 2 * self.hd), k=f(M, self.n_kv * self.hd), v=f(M, self.n_kv * self.hd),
            o=f(M, self.n_h * self.hd), oq=i8(M, self.n_h * self.hd), so=f(M), mo=u16(M, self.n_h * self.hd // 128),
            onq=i8(M, self.n_v * self.dv), son=f(M), mon=u16(M, self.n_v * self.dv // 128),
            qkv=f(M, self.conv_dim), z=f(M, self.n_v * self.dv), a=f(M, self.n_v), bb=f(M, self.n_v), on=f(M, self.n_v * self.dv),
            g=f(M, self.I), u=f(M, self.I), hq=i8(M, self.I), sh=f(M), mh=u16(M, self.I // 128),
            f_qkv=f(M, self.n_h * 2 * self.hd + 2 * self.n_kv * self.hd), f_qkvz=f(M, self.conv_dim + self.n_v * self.dv),
            f_ab=f(M, 2 * self.n_v), f_gu=f(M, 2 * self.I),
            logits=f(1, self.V), amax=D.empty((M,), np.int32),
            attn_scratch=D.empty((max(D.attn_scratch_bytes(mm, self.n_h, self.hd) for mm in range(1, M + 1)),), np.uint8),
            gdn_scratch=D.empty((max(D.gdn_scratch_bytes(mm, self.n_k, self.n_v, self.dk, self.dv) for mm in range(1, M + 1)),), np.uint8),
        )

    def reset(self):
        D.sync()
        self.pos = 0
        for ly in self.layers:
            if ly["type"] != "full_attention":
                ly["gs_t"][0].zero_(); ly["gs_t"][1].zero_()

    def _fused(self, L, fused_buf, T, parts):
        """fused projection written straight into its part buffers (GEMM epilogue column routing)"""
        b = self.b
        res = [buf[:T] for buf in parts]
        offs = [sum(L.widths[:i]) for i in range(len(L.widths))]
        D.gemm(L, b["xq"][:T], b["sx"][:T], b["mx"][:T], None, parts=(offs, res))
        return res

    def _embed(self, ids, T):
        rows = (self.embed_host[np.asarray(ids, dtype=np.int64)].astype(np.uint32) << 16).view(np.float32)   # exact bf16->fp32
        self.b["x"].np[:T] = rows

    def _norm_in(self, ly, T):
        b = self.b; D.rmsnorm_quant(b["x"][:T], ly["attn_norm"], self.eps, None, b["xq"][:T], b["sx"][:T], b["mx"][:T])
    def _attn_qkv(self, ly, T):
        b = self.b
        self._qkv_views = self._fused(ly["qkv_f"], b["f_qkv"], T, [b["qg"], b["k"], b["v"]])
    def _attn_core(self, li, ly, T):
        b = self.b; qg, k, v = self._qkv_views
        if self.v_hook is not None: D.sync(); self.v_hook(li, v.np)
        D.attention(ly["al"], ly["kv"], qg, k, v, self.rope_cos, self.rope_sin, self.rot, self.eps,
                    self.pos, self.n_h, self.n_kv, self.hd, b["o"][:T], b["attn_scratch"])
    def _attn_out(self, ly, T):
        b = self.b; oq, so, mo = b["oq"][:T], b["so"][:T], b["mo"][:T]
        D.quant(b["o"][:T], oq, so, mo); D.gemm(ly["o"], oq, so, mo, b["x"][:T], accumulate=True)
    def _gdn_layer(self, ly, T):
        b = self.b; on = b["on"][:T]
        qkv, z = self._fused(ly["qkvz_f"], b["f_qkvz"], T, [b["qkv"], b["z"]])
        a, bb = self._fused(ly["ab_f"], b["f_ab"], T, [b["a"], b["bb"]])
        D.gdn(ly["gl"], ly["gs"], qkv, z, a, bb, self.eps, self.n_k, self.n_v, self.dk, self.dv, on, b["gdn_scratch"])
        oq, so, mo = b["onq"][:T], b["son"][:T], b["mon"][:T]
        D.quant(on, oq, so, mo); D.gemm(ly["out"], oq, so, mo, b["x"][:T], accumulate=True)
    def _ffn(self, ly, T):
        b = self.b; x, xq, sx, mx = b["x"][:T], b["xq"][:T], b["sx"][:T], b["mx"][:T]
        D.rmsnorm_quant(x, ly["ffn_norm"], self.eps, None, xq, sx, mx)
        hq, sh, mh = b["hq"][:T], b["sh"][:T], b["mh"][:T]
        g, u = self._fused(ly["gu_f"], b["f_gu"], T, [b["g"], b["u"]])
        D.silu_mul_quant(g, u, None, hq, sh, mh)
        D.gemm(ly["down"], hq, sh, mh, x, accumulate=True)
    def _head(self, T, all_logits):
        b = self.b; x, xq, sx, mx = b["x"][:T], b["xq"][:T], b["sx"][:T], b["mx"][:T]
        if all_logits:
            if b["logits"].shape[0] < T: b["logits"] = D.empty((self.max_batch, self.V), np.float32)
            D.rmsnorm_quant(x, self.final_norm, self.eps, None, xq, sx, mx)
            lg = b["logits"][:T]; D.gemm(self.lm_head, xq, sx, mx, lg); return lg
        D.rmsnorm_quant(x[T - 1:T], self.final_norm, self.eps, None, xq[:1], sx[:1], mx[:1])
        lg = b["logits"][:1]; D.gemm(self.lm_head, xq[:1], sx[:1], mx[:1], lg); return lg

    def forward(self, ids, all_logits=False):
        """ids: list[int] of new tokens.  Returns the logits Buf [T,V] (or [1,V] of the last token); call D.sync() (or
        .numpy()) before reading it.  The embedding rows are written to the shared buffer by the host, so all GPU work
        of the previous forward must be complete (any read of its logits guarantees that)."""
        T = len(ids); b = self.b
        assert T <= self.max_batch and self.pos + T <= self.max_ctx
        D.sync()
        self._embed(ids, T)
        for li, ly in enumerate(self.layers):
            self._norm_in(ly, T)
            if ly["type"] == "full_attention":
                self._attn_qkv(ly, T); self._attn_core(li, ly, T); self._attn_out(ly, T)
            else:
                self._gdn_layer(ly, T)
            self._ffn(ly, T)
        lg = self._head(T, all_logits)
        self.pos += T
        D.flush()
        return lg

    def generate(self, ids, n_new, chunk=None):
        """greedy generation; returns list of new token ids"""
        chunk = chunk or self.max_batch
        for i in range(0, len(ids), chunk):
            lg = self.forward(ids[i:i + chunk])
        out = []
        am = self.b["amax"][:1]
        for step in range(n_new):
            D.argmax(lg, am); out.append(int(am.item()))
            lg = self.forward([out[-1]])
        return out
