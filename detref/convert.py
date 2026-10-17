"""Convert a HF Qwen3.5-architecture checkpoint into the det model format.

Usage: convert.py <hf_snapshot_dir> <out_dir> [--bits 6] [--calib calib_v.json] [--rope-positions 32768]

Output: <out_dir>/model.json (config + tensor list) and <out_dir>/tensors.safetensors
Tensor naming (see model.py for semantics):
  embed_bf16        uint16 [V, H]      raw bf16 bits of the embedding table
  final_norm        f32 [H]            (1 + w)
  lm_head.{wq,c,m}  quantized linear   (tied: quantized copy of embed)
  rope_cos/rope_sin f32 [P, 32]
  L{i}.attn_norm, L{i}.ffn_norm   f32 [H]  (1 + w)
  L{i}.gate/up/down.{wq,c,m}
  full attention layers:  L{i}.q/k/v/o.{wq,c,m}, L{i}.q_norm, L{i}.k_norm (f32 [256], 1+w), L{i}.v_scale f32 [n_kv], L{i}.v_inv_scale f32 [n_kv]
  linear attention layers: L{i}.qkv/z/a/b/out.{wq,c,m}, L{i}.conv f32 [conv_dim,4], L{i}.A f32 [n_v] (=exp(A_log)),
                           L{i}.dt_bias f32 [n_v], L{i}.gdn_norm f32 [128] (used as-is)
"""
import sys, os, json, glob, argparse, math
import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file
sys.path.insert(0, os.path.dirname(__file__))
from quant import quantize_linear, pack

def bf16_bits_to_f32(bits_u16):
    return (bits_u16.astype(np.uint32) << 16).view(np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("out")
    ap.add_argument("--bits", type=int, default=6)
    ap.add_argument("--head-bits", type=int, default=8)
    ap.add_argument("--calib", default=None)
    ap.add_argument("--rope-positions", type=int, default=32768)
    ap.add_argument("--v-headroom", type=float, default=8.0, help="calibrated absmax maps to 32767/headroom")
    ap.add_argument("--layers", type=int, default=None, help="truncate to first N layers (debug)")
    args = ap.parse_args()
    cfg = json.load(open(os.path.join(args.src, "config.json")))
    tc = cfg.get("text_config", cfg)
    H = tc["hidden_size"]; L = tc["num_hidden_layers"]; V = tc["vocab_size"]
    if args.layers: L = args.layers
    layer_types = tc["layer_types"][:L]
    n_h, n_kv, hd = tc["num_attention_heads"], tc["num_key_value_heads"], tc["head_dim"]
    n_k, n_v, dk, dv = tc["linear_num_key_heads"], tc["linear_num_value_heads"], tc["linear_key_head_dim"], tc["linear_value_head_dim"]
    conv_k = tc["linear_conv_kernel_dim"]
    rot = int(hd * tc["rope_parameters"]["partial_rotary_factor"])
    theta = tc["rope_parameters"]["rope_theta"]
    eps = tc["rms_norm_eps"]
    tied = tc.get("tie_word_embeddings", False)
    calib = json.load(open(args.calib)) if args.calib else None

    # index all tensors across shards (raw)
    files = sorted(glob.glob(os.path.join(args.src, "*.safetensors")))
    where = {}
    for f in files:
        with safe_open(f, "np") as st:
            for k in st.keys(): where[k] = f
    handles = {}
    def get_raw(name):
        """returns np array; bf16 returned as uint16 bits"""
        f = where[name]
        if f not in handles: handles[f] = safe_open(f, "np")
        st = handles[f]
        try:
            return st.get_tensor(name)
        except Exception:
            # bf16 not supported by numpy framework: read via torch
            import torch
            from safetensors import safe_open as so
            with so(f, "pt") as st2:
                t = st2.get_tensor(name)
            return t.view(torch.int16).numpy().view(np.uint16)
    def get_f32(name):
        a = get_raw(name)
        if a.dtype == np.uint16: return bf16_bits_to_f32(a)
        return a.astype(np.float32)
    pre = "model.language_model." if any(k.startswith("model.language_model.") for k in where) else "model."

    out = {}
    meta = dict(arch="qwen3_5", hidden=H, layers=L, vocab=V, layer_types=layer_types,
                n_heads=n_h, n_kv=n_kv, head_dim=hd, rot_dim=rot, rope_theta=theta, eps=eps,
                gdn_k_heads=n_k, gdn_v_heads=n_v, gdn_dk=dk, gdn_dv=dv, conv_k=conv_k,
                intermediate=tc["intermediate_size"], group=128, linears={})
    def put_linear(name, W, bits):
        q = quantize_linear(W, bits)
        out[name + ".wq"] = pack(q["wq"], bits)
        out[name + ".c"] = q["c"]; out[name + ".m"] = q["m"]
        meta["linears"][name] = dict(N=int(W.shape[0]), K=int(W.shape[1]), bits=bits)
        print(f"  {name:14s} {W.shape} W{bits}", flush=True)

    emb_bits = get_raw(pre + "embed_tokens.weight")
    assert emb_bits.dtype == np.uint16
    out["embed_bf16"] = emb_bits
    out["final_norm"] = (1.0 + get_f32(pre + "norm.weight")).astype(np.float32)
    head_name = "lm_head.weight" if (not tied and "lm_head.weight" in where) else pre + "embed_tokens.weight"
    put_linear("lm_head", get_f32(head_name), args.head_bits)
    # rope tables (float64 math, rounded once)
    P = args.rope_positions
    inv_freq = 1.0 / (theta ** (np.arange(0, rot, 2, dtype=np.float64) / rot))
    ang = np.arange(P, dtype=np.float64)[:, None] * inv_freq[None, :]
    out["rope_cos"] = np.cos(ang).astype(np.float32); out["rope_sin"] = np.sin(ang).astype(np.float32)

    for i in range(L):
        p = f"{pre}layers.{i}."
        n = f"L{i}."
        print(f"layer {i} ({layer_types[i]})", flush=True)
        out[n + "attn_norm"] = (1.0 + get_f32(p + "input_layernorm.weight")).astype(np.float32)
        out[n + "ffn_norm"] = (1.0 + get_f32(p + "post_attention_layernorm.weight")).astype(np.float32)
        put_linear(n + "gate", get_f32(p + "mlp.gate_proj.weight"), args.bits)
        put_linear(n + "up", get_f32(p + "mlp.up_proj.weight"), args.bits)
        put_linear(n + "down", get_f32(p + "mlp.down_proj.weight"), args.bits)
        if layer_types[i] == "full_attention":
            a = p + "self_attn."
            put_linear(n + "q", get_f32(a + "q_proj.weight"), args.bits)
            put_linear(n + "k", get_f32(a + "k_proj.weight"), args.bits)
            put_linear(n + "v", get_f32(a + "v_proj.weight"), args.bits)
            put_linear(n + "o", get_f32(a + "o_proj.weight"), args.bits)
            out[n + "q_norm"] = (1.0 + get_f32(a + "q_norm.weight")).astype(np.float32)
            out[n + "k_norm"] = (1.0 + get_f32(a + "k_norm.weight")).astype(np.float32)
            if calib is not None:
                amax = np.array(calib[str(i)], dtype=np.float64)
            else:
                amax = np.full(n_kv, 64.0)   # crude fallback
            vs = (amax * args.v_headroom / 32767.0).astype(np.float32)
            out[n + "v_scale"] = vs
            out[n + "v_inv_scale"] = (np.float32(1.0) / vs).astype(np.float32)
        else:
            a = p + "linear_attn."
            put_linear(n + "qkv", get_f32(a + "in_proj_qkv.weight"), args.bits)
            put_linear(n + "z", get_f32(a + "in_proj_z.weight"), args.bits)
            put_linear(n + "a", get_f32(a + "in_proj_a.weight"), 8)
            put_linear(n + "b", get_f32(a + "in_proj_b.weight"), 8)
            put_linear(n + "out", get_f32(a + "out_proj.weight"), args.bits)
            out[n + "conv"] = get_f32(a + "conv1d.weight").reshape(-1, conv_k).astype(np.float32)
            out[n + "A"] = np.exp(get_f32(a + "A_log").astype(np.float64)).astype(np.float32)
            out[n + "dt_bias"] = get_f32(a + "dt_bias").astype(np.float32)
            out[n + "gdn_norm"] = get_f32(a + "norm.weight").astype(np.float32)
    os.makedirs(args.out, exist_ok=True)
    save_file(out, os.path.join(args.out, "tensors.safetensors"))
    json.dump(meta, open(os.path.join(args.out, "model.json"), "w"), indent=1)
    print("done:", sum(v.nbytes for v in out.values()) / 1e9, "GB")

if __name__ == "__main__":
    main()
