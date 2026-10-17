"""CUDA implementation of the det model (orchestrates the kernels in cuda/libdet.so).
Must produce logits bit-identical to detref/model.py."""
import os, sys, json, ctypes, time
import numpy as np
import torch
from safetensors import safe_open
sys.path.insert(0, os.path.dirname(__file__))
import detcuda as D

_gemm_scratch = None   # process-wide det_gemm scratch (the library keeps one pointer): grown to the largest request, never shrunk
def _ensure_gemm_scratch(nbytes):
    global _gemm_scratch
    if _gemm_scratch is None or _gemm_scratch.numel() < nbytes:
        _gemm_scratch = torch.empty(nbytes, dtype=torch.uint8, device="cuda")
        D.gemm_set_scratch(_gemm_scratch)

class CudaModel:
    def __init__(self, d, max_ctx=8192, max_batch=256, verbose=False, embed_on_host=True, graph_mode="full"):
        """graph_mode: decode (T=1) CUDA graphs -- "full" (one graph per step, attention via det_attention_dev with the
        position read from device memory), "segments" (17 graphs with eager attention in between), or None (eager)."""
        assert graph_mode in ("full", "segments", None), graph_mode
        self.graph_mode = graph_mode
        self.meta = m = json.load(open(os.path.join(d, "model.json")))
        self.H, self.L, self.V = m["hidden"], m["layers"], m["vocab"]
        self.eps = float(m["eps"])
        self.n_h, self.n_kv, self.hd, self.rot = m["n_heads"], m["n_kv"], m["head_dim"], m["rot_dim"]
        self.n_k, self.n_v, self.dk, self.dv, self.conv_k = m["gdn_k_heads"], m["gdn_v_heads"], m["gdn_dk"], m["gdn_dv"], m["conv_k"]
        self.I = m["intermediate"]
        self.conv_dim = 2 * self.n_k * self.dk + self.n_v * self.dv
        self.max_ctx, self.max_batch = max_ctx, max_batch
        self.dev = torch.device("cuda")
        t0 = time.time()
        files = m.get("files")
        paths = [os.path.join(d, "parts", f) if os.path.exists(os.path.join(d, "parts", f)) else os.path.join(d, f) for f in files] if files else [os.path.join(d, "tensors.safetensors")]
        handles = [safe_open(pth, "np") for pth in paths]
        where = {}
        for h in handles:
            for k in h.keys(): where[k] = h
        def up(name):
            a = where[name].get_tensor(name)
            return D.to_dev(a)
        def lin(name):
            info = m["linears"][name]
            return D.make_qlinear(up(name + ".wq"), up(name + ".c"), up(name + ".m"), info["N"], info["K"], info["bits"])
        def lin_fused(names):
            """concatenate several linears with equal K and bits along N (rows are independent => bit-exact)"""
            infos = [m["linears"][n] for n in names]
            assert len(set(i["K"] for i in infos)) == 1 and len(set(i["bits"] for i in infos)) == 1, names
            wq = np.concatenate([where[n + ".wq"].get_tensor(n + ".wq") for n in names])
            c = np.concatenate([where[n + ".c"].get_tensor(n + ".c") for n in names])
            mm = np.concatenate([where[n + ".m"].get_tensor(n + ".m") for n in names])
            L = D.make_qlinear(D.to_dev(wq), D.to_dev(c), D.to_dev(mm), int(sum(i["N"] for i in infos)), infos[0]["K"], infos[0]["bits"])
            L.widths = [i["N"] for i in infos]; return L
        vs_override = os.path.join(d, "v_scales.npz")
        self.v_override = dict(np.load(vs_override)) if os.path.exists(vs_override) else None
        self.v_hook = None   # optional callable(layer_idx, v_tensor[M, n_kv*hd]) for calibration
        self.embed_on_host = embed_on_host
        if embed_on_host:
            self.embed_host = where["embed_bf16"].get_tensor("embed_bf16")      # uint16 bits, stays in RAM
            self.embed_pin = torch.empty(max_batch, self.H, dtype=torch.float32, pin_memory=True)
        else:
            self.embed = up("embed_bf16")
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
                kq = torch.zeros(max_ctx, self.n_kv, self.hd, dtype=torch.int8, device=self.dev)
                sk = torch.zeros(max_ctx, self.n_kv, dtype=torch.float32, device=self.dev)
                vq = torch.zeros(max_ctx, self.n_kv, self.hd, dtype=torch.int16, device=self.dev)
                ly["kv"] = D.KVCache(D.P(kq), D.P(sk), D.P(vq), max_ctx); ly["kv_t"] = (kq, sk, vq)
            else:
                ly.update(qkvz_f=lin_fused([n + "qkv", n + "z"]), ab_f=lin_fused([n + "a", n + "b"]), out=lin(n + "out"))
                keep = [up(n + "conv"), up(n + "A"), up(n + "dt_bias"), up(n + "gdn_norm")]
                ly["gl"] = D.GdnLayer(*[D.P(x) for x in keep]); ly["keep"] = keep
                cs = torch.zeros(self.conv_k - 1, self.conv_dim, dtype=torch.float32, device=self.dev)
                S = torch.zeros(self.n_v, self.dk, self.dv, dtype=torch.float32, device=self.dev)
                ly["gs"] = D.GdnState(D.P(cs), D.P(S)); ly["gs_t"] = (cs, S)
            self.layers.append(ly)
            if verbose: print(f"loaded layer {i}", flush=True)
        if verbose: print(f"model loaded in {time.time() - t0:.1f}s, {torch.cuda.memory_allocated() / 1e9:.2f} GB", flush=True)
        self._alloc(max_batch)
        _ensure_gemm_scratch(D.gemm_scratch_bytes(max_batch, max(int(v["K"]) for v in m["linears"].values())))
        self.pos_pin = torch.zeros(1, dtype=torch.int32, pin_memory=True)   # host copy of pos, read by the full decode graph
        self.pos_dev = torch.zeros(1, dtype=torch.int32, device=self.dev)   # device copy (det_attention_dev)
        self._capture_full = False
        self.reset()

    def _alloc(self, M):
        f = lambda *s: torch.empty(*s, dtype=torch.float32, device=self.dev)
        i8 = lambda *s: torch.empty(*s, dtype=torch.int8, device=self.dev)
        u16 = lambda *s: torch.empty(*s, dtype=torch.int16, device=self.dev)
        self.b = dict(
            ids=torch.empty(M, dtype=torch.int32, device=self.dev),
            x=f(M, self.H), xn=f(M, self.H), xq=i8(M, self.H), sx=f(M), mx=u16(M, self.H // 128), r=f(M, self.H),
            qg=f(M, self.n_h * 2 * self.hd), k=f(M, self.n_kv * self.hd), v=f(M, self.n_kv * self.hd),
            o=f(M, self.n_h * self.hd), oq=i8(M, self.n_h * self.hd), so=f(M), mo=u16(M, self.n_h * self.hd // 128),
            onq=i8(M, self.n_v * self.dv), son=f(M), mon=u16(M, self.n_v * self.dv // 128),
            qkv=f(M, self.conv_dim), z=f(M, self.n_v * self.dv), a=f(M, self.n_v), bb=f(M, self.n_v), on=f(M, self.n_v * self.dv),
            g=f(M, self.I), u=f(M, self.I), hq=i8(M, self.I), sh=f(M), mh=u16(M, self.I // 128),
            f_qkv=f(M, self.n_h * 2 * self.hd + 2 * self.n_kv * self.hd), f_qkvz=f(M, self.conv_dim + self.n_v * self.dv),
            f_ab=f(M, 2 * self.n_v), f_gu=f(M, 2 * self.I),
            logits=f(1, self.V), amax=torch.empty(M, dtype=torch.int32, device=self.dev),   # full [M,V] allocated on demand
            # scratch requirements are not monotonic in M (attention key-splitting only for M <= 256): take the max over M
            attn_scratch=torch.empty(max(D.attn_scratch_bytes(mm, self.n_h, self.hd) for mm in range(1, M + 1)), dtype=torch.uint8, device=self.dev),
            gdn_scratch=torch.empty(max(D.gdn_scratch_bytes(mm, self.n_k, self.n_v, self.dk, self.dv) for mm in range(1, M + 1)), dtype=torch.uint8, device=self.dev),
        )

    def reset(self):
        self.pos = 0
        if hasattr(self, "graphs"): pass   # graphs stay valid: they only reference fixed buffers
        for ly in self.layers:
            if ly["type"] != "full_attention":
                ly["gs_t"][0].zero_(); ly["gs_t"][1].zero_()

    def _fused(self, L, fused_buf, T, parts):
        """gemm into fused_buf[:T], return list of [T, n_i] tensors for the parts (buffers given for T > 1)."""
        b = self.b; out = fused_buf[:T]
        D.gemm(L, b["xq"][:T], b["sx"][:T], b["mx"][:T], out)
        res = []; off = 0
        for buf, n in zip(parts, L.widths):
            if T == 1: res.append(out[:, off:off + n])
            else: buf[:T].copy_(out[:, off:off + n]); res.append(buf[:T])
            off += n
        return res

    def _embed(self, ids, T):
        b = self.b
        if self.embed_on_host:
            rows = (self.embed_host[np.asarray(ids, dtype=np.int64)].astype(np.uint32) << 16).view(np.float32)   # exact bf16->fp32
            self.embed_pin[:T].copy_(torch.from_numpy(rows))
            b["x"][:T].copy_(self.embed_pin[:T], non_blocking=True)
        else:
            b["ids"][:T].copy_(torch.tensor(ids, dtype=torch.int32))
            D.embed(self.embed, b["ids"][:T], b["x"][:T])

    # ---------------------------------------------------------------- per-layer pieces (T tokens, on self.b)
    def _norm_in(self, ly, T):
        b = self.b; D.rmsnorm_quant(b["x"][:T], ly["attn_norm"], self.eps, None, b["xq"][:T], b["sx"][:T], b["mx"][:T])
    def _attn_qkv(self, ly, T):
        b = self.b
        self._qkv_views = self._fused(ly["qkv_f"], b["f_qkv"], T, [b["qg"], b["k"], b["v"]])
    def _attn_core(self, li, ly, T):
        b = self.b; qg, k, v = self._qkv_views
        if self._capture_full:   # position from device memory, launch sized for the whole cache (graph-capturable)
            D.attention_dev(ly["al"], ly["kv"], qg, k, v, self.rope_cos, self.rope_sin, self.rot, self.eps,
                            self.pos_dev, self.max_ctx - T, self.n_h, self.n_kv, self.hd, b["o"][:T], b["attn_scratch"])
            return
        if self.v_hook is not None: self.v_hook(li, v)
        D.attention(ly["al"], ly["kv"], qg, k, v, self.rope_cos, self.rope_sin, self.rot, self.eps,
                    self.pos, self.n_h, self.n_kv, self.hd, b["o"][:T], b["attn_scratch"])
    def _attn_out(self, ly, T):
        b = self.b; oq, so, mo = b["oq"][:T], b["so"][:T], b["mo"][:T]
        D.quant(b["o"][:T], oq, so, mo); D.gemm(ly["o"], oq, so, mo, b["r"][:T]); D.add(b["x"][:T], b["r"][:T])
    def _gdn_layer(self, ly, T):
        b = self.b; on = b["on"][:T]
        qkv, z = self._fused(ly["qkvz_f"], b["f_qkvz"], T, [b["qkv"], b["z"]])
        a, bb = self._fused(ly["ab_f"], b["f_ab"], T, [b["a"], b["bb"]])
        D.gdn(ly["gl"], ly["gs"], qkv, z, a, bb, self.eps, self.n_k, self.n_v, self.dk, self.dv, on, b["gdn_scratch"])
        oq, so, mo = b["onq"][:T], b["son"][:T], b["mon"][:T]
        D.quant(on, oq, so, mo); D.gemm(ly["out"], oq, so, mo, b["r"][:T]); D.add(b["x"][:T], b["r"][:T])
    def _ffn(self, ly, T):
        b = self.b; x, xq, sx, mx = b["x"][:T], b["xq"][:T], b["sx"][:T], b["mx"][:T]
        D.rmsnorm_quant(x, ly["ffn_norm"], self.eps, None, xq, sx, mx)
        hq, sh, mh = b["hq"][:T], b["sh"][:T], b["mh"][:T]
        g, u = self._fused(ly["gu_f"], b["f_gu"], T, [b["g"], b["u"]])
        D.silu_mul_quant(g, u, None, hq, sh, mh)
        D.gemm(ly["down"], hq, sh, mh, b["r"][:T]); D.add(x, b["r"][:T])
    def _head(self, T, all_logits):
        b = self.b; x, xq, sx, mx = b["x"][:T], b["xq"][:T], b["sx"][:T], b["mx"][:T]
        if all_logits:
            if b["logits"].shape[0] < T: b["logits"] = torch.empty(self.max_batch, self.V, dtype=torch.float32, device=self.dev)
            D.rmsnorm_quant(x, self.final_norm, self.eps, None, xq, sx, mx)
            lg = b["logits"][:T]; D.gemm(self.lm_head, xq, sx, mx, lg); return lg
        D.rmsnorm_quant(x[T - 1:T], self.final_norm, self.eps, None, xq[:1], sx[:1], mx[:1])
        lg = b["logits"][:1]; D.gemm(self.lm_head, xq[:1], sx[:1], mx[:1], lg); return lg

    def forward(self, ids, all_logits=False):
        """ids: list[int] of new tokens. Returns logits [T,V] fp32 cuda tensor (or [1,V] of last token)."""
        T = len(ids); b = self.b
        assert T <= self.max_batch and self.pos + T <= self.max_ctx
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
        return lg

    # ---------------------------------------------------------------- CUDA-graph decode (T = 1)
    # "full": the whole step is one graph.  Its first node copies the position from the pinned host int pos_pin into
    # pos_dev (updated before each replay, like the embedding row in embed_pin); attention runs via det_attention_dev,
    # which reads the position from pos_dev and is launched for the full cache capacity.
    # "segments": det_attention takes the position as a kernel argument, so it stays eager; everything between
    # attention layers is captured once as a graph segment (17 graphs for 16 attention layers).
    # Call after at least one eager T=1 forward (kernel one-time setup).
    def build_decode_graphs(self, mode=None):
        mode = mode or self.graph_mode or "segments"
        assert mode in ("full", "segments"), mode
        T = 1; segs = []; cur = []   # cur: list of callables
        if mode == "full": cur.append(lambda: self.pos_dev.copy_(self.pos_pin, non_blocking=True))
        if self.embed_on_host: cur.append(lambda: self.b["x"][:1].copy_(self.embed_pin[:1], non_blocking=True))
        else: cur.append(lambda: D.embed(self.embed, self.b["ids"][:1], self.b["x"][:1]))
        attn_layers = []
        for li, ly in enumerate(self.layers):
            cur.append(lambda ly=ly: self._norm_in(ly, T))
            if ly["type"] == "full_attention":
                cur.append(lambda ly=ly: self._attn_qkv(ly, T))
                if mode == "full":
                    cur.append(lambda li=li, ly=ly: self._attn_core(li, ly, T))
                else:
                    segs.append(cur); cur = []; attn_layers.append((li, ly))
                cur.append(lambda ly=ly: self._attn_out(ly, T))
            else:
                cur.append(lambda ly=ly: self._gdn_layer(ly, T))
            cur.append(lambda ly=ly: self._ffn(ly, T))
        cur.append(lambda: self._head(1, False))
        segs.append(cur)
        self.graphs = []
        torch.cuda.synchronize()
        self._capture_full = mode == "full"
        try:
            for seg in segs:
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    for f in seg: f()
                self.graphs.append(g)
        finally:
            self._capture_full = False
        self.graph_attn = attn_layers
        self.graphs_mode = mode
        torch.cuda.synchronize()

    def decode_step(self, tok):
        """One T=1 step via the captured graphs for token id `tok` (python int).  The pinned inputs (pos_pin, embed_pin)
        are rewritten here, so the previous step must have been consumed (any sync, e.g. reading its logits)."""
        self.pos_pin[0] = self.pos
        if self.embed_on_host:
            row = (self.embed_host[int(tok)].astype(np.uint32) << 16).view(np.float32)
            self.embed_pin[0].copy_(torch.from_numpy(row))
        else:
            self.b["ids"][:1].fill_(int(tok))
        for i, g in enumerate(self.graphs):
            g.replay()
            if i < len(self.graph_attn):
                li, ly = self.graph_attn[i]; self._attn_core(li, ly, 1)
        self.pos += 1
        return self.b["logits"][:1]

    def generate(self, ids, n_new, chunk=None, use_graphs=True):
        """greedy generation; returns list of new token ids"""
        use_graphs = use_graphs and self.graph_mode is not None
        chunk = chunk or self.max_batch
        for i in range(0, len(ids), chunk):
            lg = self.forward(ids[i:i + chunk])
        out = []
        am = self.b["amax"][:1]
        for step in range(n_new):
            D.argmax(lg, am); out.append(int(am.item()))
            if use_graphs and step == 1 and not hasattr(self, "graphs"): self.build_decode_graphs()
            lg = self.decode_step(out[-1]) if (use_graphs and hasattr(self, "graphs")) else self.forward([out[-1]])
        return out
