"""Streaming HF -> det converter: downloads safetensors shards on demand from the Hub, quantizes layer by layer
with a process pool, deletes shards once no later layer needs them.  Output format identical to detref/convert.py.

Usage: convert_stream.py <repo_id> <out_dir> [--bits 6] [--workers 8] [--v-cover 1000] [--calib calib.json]
"""
import sys, os, json, argparse, time, threading
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from safetensors import safe_open
from safetensors.numpy import save_file
from huggingface_hub import hf_hub_download
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "detref"))
from quant import quantize_linear, pack

def bf16_to_f32(bits_u16):
    return (bits_u16.astype(np.uint32) << 16).view(np.float32)

def qjob(args):
    name, W, bits = args
    q = quantize_linear(W, bits)
    return name, pack(q["wq"], bits), q["c"], q["m"], dict(N=int(W.shape[0]), K=int(W.shape[1]), bits=bits)

class ShardSource:
    def __init__(self, repo, local, keep=False):
        """keep=True: `local` already holds the full checkpoint (e.g. a HF cache snapshot); never delete shards."""
        self.repo, self.local, self.keep = repo, local, keep
        os.makedirs(local, exist_ok=True)
        idx_path = os.path.join(local, "model.safetensors.index.json")
        if not (keep and os.path.exists(idx_path)):
            idx_path = hf_hub_download(repo, "model.safetensors.index.json", local_dir=local)
        idx = json.load(open(idx_path))
        self.map = idx["weight_map"]
        self.shards = sorted(set(self.map.values()))
        self.handles = {}
        self.lock = threading.Lock()
        self.prefetch = None
    def ensure(self, shard):
        p = os.path.join(self.local, shard)
        if not os.path.exists(p):
            t0 = time.time(); hf_hub_download(self.repo, shard, local_dir=self.local)
            print(f"  downloaded {shard} in {time.time()-t0:.0f}s", flush=True)
        return p
    def prefetch_shard(self, shard):
        if shard is None or os.path.exists(os.path.join(self.local, shard)): return
        if self.prefetch is not None: self.prefetch.join()
        self.prefetch = threading.Thread(target=self.ensure, args=(shard,)); self.prefetch.start()
    def raw(self, name):
        shard = self.map[name]
        if self.prefetch is not None and not os.path.exists(os.path.join(self.local, shard)):
            self.prefetch.join()
        p = self.ensure(shard)
        if p not in self.handles: self.handles[p] = safe_open(p, "pt")
        t = self.handles[p].get_tensor(name)
        import torch
        if t.dtype == torch.bfloat16: return t.view(torch.int16).numpy().view(np.uint16)
        return t.numpy()
    def f32(self, name):
        a = self.raw(name)
        return bf16_to_f32(a) if a.dtype == np.uint16 else a.astype(np.float32)
    def release(self, shard):
        p = os.path.join(self.local, shard)
        if p in self.handles: del self.handles[p]
        if self.keep: return
        if os.path.exists(p): os.remove(p); print(f"  deleted {shard}", flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repo"); ap.add_argument("out")
    ap.add_argument("--bits", type=int, default=6); ap.add_argument("--head-bits", type=int, default=8)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--v-cover", type=float, default=1000.0, help="placeholder: int16 V range covers |v| <= this")
    ap.add_argument("--calib", default=None); ap.add_argument("--v-headroom", type=float, default=8.0)
    ap.add_argument("--rope-positions", type=int, default=65536)
    ap.add_argument("--local", default=None, help="directory holding the checkpoint (e.g. a HF cache snapshot); shards are then kept")
    args = ap.parse_args()
    local = args.local or os.path.join(os.path.dirname(__file__), "..", "dl", args.repo.split("/")[-1].lower())
    src = ShardSource(args.repo, local, keep=args.local is not None)
    cfg_path = os.path.join(local, "config.json")
    if not (args.local and os.path.exists(cfg_path)): cfg_path = hf_hub_download(args.repo, "config.json", local_dir=local)
    cfg = json.load(open(cfg_path))
    tc = cfg.get("text_config", cfg)
    H, L, V = tc["hidden_size"], tc["num_hidden_layers"], tc["vocab_size"]
    layer_types = tc["layer_types"][:L]
    n_h, n_kv, hd = tc["num_attention_heads"], tc["num_key_value_heads"], tc["head_dim"]
    n_k, n_v, dk, dv = tc["linear_num_key_heads"], tc["linear_num_value_heads"], tc["linear_key_head_dim"], tc["linear_value_head_dim"]
    conv_k = tc["linear_conv_kernel_dim"]; rot = int(hd * tc["rope_parameters"]["partial_rotary_factor"])
    theta = tc["rope_parameters"]["rope_theta"]; eps = tc["rms_norm_eps"]; tied = tc.get("tie_word_embeddings", False)
    calib = json.load(open(args.calib)) if args.calib else None
    pre = "model.language_model." if any(k.startswith("model.language_model.") for k in src.map) else "model."
    meta = dict(arch="qwen3_5", hidden=H, layers=L, vocab=V, layer_types=layer_types, n_heads=n_h, n_kv=n_kv, head_dim=hd,
                rot_dim=rot, rope_theta=theta, eps=eps, gdn_k_heads=n_k, gdn_v_heads=n_v, gdn_dk=dk, gdn_dv=dv, conv_k=conv_k,
                intermediate=tc["intermediate_size"], group=128, linears={}, v_scale_calibrated=calib is not None)
    out = {}
    # which shards does each layer need; release a shard after the last layer that uses it
    def shard_of(name): return src.map[name]
    layer_shards = []
    for i in range(L):
        names = [k for k in src.map if k.startswith(f"{pre}layers.{i}.")]
        layer_shards.append(sorted(set(shard_of(k) for k in names)))
    last_use = {}
    for i, ss in enumerate(layer_shards):
        for s in ss: last_use[s] = i
    head_names = [pre + "embed_tokens.weight", pre + "norm.weight"] + (["lm_head.weight"] if "lm_head.weight" in src.map else [])
    for k in head_names: last_use[shard_of(k)] = L   # keep until the end
    pool = ProcessPoolExecutor(args.workers)
    parts_dir = os.path.join(args.out, "parts"); os.makedirs(parts_dir, exist_ok=True)
    meta_path = os.path.join(args.out, "model.json")
    if os.path.exists(meta_path):
        old = json.load(open(meta_path)); meta["linears"].update(old.get("linears", {}))
    def submit_all(jobs):
        futs = [pool.submit(qjob, j) for j in jobs]
        for f in futs:
            name, wq, c, m, info = f.result()
            out[name + ".wq"] = wq; out[name + ".c"] = c; out[name + ".m"] = m; meta["linears"][name] = info
    def flush_part(name):
        """write accumulated tensors to parts/<name>.safetensors and clear"""
        save_file(out, os.path.join(parts_dir, name + ".safetensors")); out.clear()
        meta["files"] = sorted(os.listdir(parts_dir)); json.dump(meta, open(meta_path, "w"), indent=1)
    t_start = time.time()
    for i in range(L):
        p = f"{pre}layers.{i}."; n = f"L{i}."
        if os.path.exists(os.path.join(parts_dir, f"L{i}.safetensors")):
            print(f"layer {i}: already converted, skipping", flush=True)
            for s_ in layer_shards[i]:
                if last_use[s_] == i: src.release(s_)
            continue
        # prefetch the next layer's first new shard
        nxt = [s for s in (layer_shards[i + 1] if i + 1 < L else []) if s not in layer_shards[i]]
        for s in layer_shards[i]: src.ensure(s)
        if nxt: src.prefetch_shard(nxt[0])
        print(f"layer {i} ({layer_types[i]}) shards {layer_shards[i]}  [{time.time()-t_start:.0f}s]", flush=True)
        out[n + "attn_norm"] = (1.0 + src.f32(p + "input_layernorm.weight")).astype(np.float32)
        out[n + "ffn_norm"] = (1.0 + src.f32(p + "post_attention_layernorm.weight")).astype(np.float32)
        jobs = [(n + "gate", src.f32(p + "mlp.gate_proj.weight"), args.bits), (n + "up", src.f32(p + "mlp.up_proj.weight"), args.bits),
                (n + "down", src.f32(p + "mlp.down_proj.weight"), args.bits)]
        if layer_types[i] == "full_attention":
            a = p + "self_attn."
            jobs += [(n + "q", src.f32(a + "q_proj.weight"), args.bits), (n + "k", src.f32(a + "k_proj.weight"), args.bits),
                     (n + "v", src.f32(a + "v_proj.weight"), args.bits), (n + "o", src.f32(a + "o_proj.weight"), args.bits)]
            out[n + "q_norm"] = (1.0 + src.f32(a + "q_norm.weight")).astype(np.float32)
            out[n + "k_norm"] = (1.0 + src.f32(a + "k_norm.weight")).astype(np.float32)
            if calib is not None: vs = (np.array(calib[str(i)], dtype=np.float64) * args.v_headroom / 32767.0).astype(np.float32)
            else: vs = np.full(n_kv, args.v_cover / 32767.0, dtype=np.float32)
            out[n + "v_scale"] = vs; out[n + "v_inv_scale"] = (np.float32(1.0) / vs).astype(np.float32)
        else:
            a = p + "linear_attn."
            jobs += [(n + "qkv", src.f32(a + "in_proj_qkv.weight"), args.bits), (n + "z", src.f32(a + "in_proj_z.weight"), args.bits),
                     (n + "a", src.f32(a + "in_proj_a.weight"), 8), (n + "b", src.f32(a + "in_proj_b.weight"), 8),
                     (n + "out", src.f32(a + "out_proj.weight"), args.bits)]
            out[n + "conv"] = src.f32(a + "conv1d.weight").reshape(-1, conv_k).astype(np.float32)
            out[n + "A"] = np.exp(src.f32(a + "A_log").astype(np.float64)).astype(np.float32)
            out[n + "dt_bias"] = src.f32(a + "dt_bias").astype(np.float32)
            out[n + "gdn_norm"] = src.f32(a + "norm.weight").astype(np.float32)
        submit_all(jobs)
        flush_part(f"L{i}")
        for s in layer_shards[i]:
            if last_use[s] == i: src.release(s)
    print("head / embeddings", flush=True)
    emb = src.raw(pre + "embed_tokens.weight"); assert emb.dtype == np.uint16
    out["embed_bf16"] = emb
    out["final_norm"] = (1.0 + src.f32(pre + "norm.weight")).astype(np.float32)
    head = src.f32("lm_head.weight") if (not tied and "lm_head.weight" in src.map) else bf16_to_f32(emb)
    submit_all([("lm_head", head, args.head_bits)])
    P = args.rope_positions
    inv_freq = 1.0 / (theta ** (np.arange(0, rot, 2, dtype=np.float64) / rot))
    ang = np.arange(P, dtype=np.float64)[:, None] * inv_freq[None, :]
    out["rope_cos"] = np.cos(ang).astype(np.float32); out["rope_sin"] = np.sin(ang).astype(np.float32)
    flush_part("head")
    for s in src.shards: src.release(s)
    total = sum(os.path.getsize(os.path.join(parts_dir, f)) for f in meta["files"])
    print(f"done in {time.time()-t_start:.0f}s: {total/1e9:.2f} GB in {len(meta['files'])} part files", flush=True)
    pool.shutdown()

if __name__ == "__main__":
    main()
