"""Re-quantize selected linears of an existing det model (per-layer parts) at a different bit width, streaming the
bf16 source shards from the Hub one at a time.  Usage: requant_tensors.py <model_dir> <repo_id> --suffixes out,down --bits 8"""
import sys, os, json, argparse, time, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "detref")); sys.path.insert(0, os.path.dirname(__file__))
from safetensors.numpy import load_file, save_file
from quant import quantize_linear, pack
from convert_stream import ShardSource
HF = {"out": "linear_attn.out_proj.weight", "down": "mlp.down_proj.weight", "gate": "mlp.gate_proj.weight", "up": "mlp.up_proj.weight",
      "qkv": "linear_attn.in_proj_qkv.weight", "z": "linear_attn.in_proj_z.weight", "q": "self_attn.q_proj.weight",
      "k": "self_attn.k_proj.weight", "v": "self_attn.v_proj.weight", "o": "self_attn.o_proj.weight"}
ap = argparse.ArgumentParser(); ap.add_argument("d"); ap.add_argument("repo"); ap.add_argument("--suffixes", default="out,down"); ap.add_argument("--bits", type=int, default=8)
ap.add_argument("--local", default=None, help="directory holding the full checkpoint (shards kept)")
args = ap.parse_args()
meta = json.load(open(os.path.join(args.d, "model.json"))); L = meta["layers"]
local = args.local or os.path.join(os.path.dirname(__file__), "..", "dl", args.repo.split("/")[-1].lower())
src = ShardSource(args.repo, local, keep=args.local is not None)
pre = "model.language_model." if any(k.startswith("model.language_model.") for k in src.map) else "model."
sfx = args.suffixes.split(",")
# shard release schedule: last layer that uses each shard (for the tensors we touch)
need = {i: [pre + f"layers.{i}." + HF[s] for s in sfx if (pre + f"layers.{i}." + HF[s]) in src.map] for i in range(L)}
last = {}
for i in range(L):
    for nme in need[i]: last[src.map[nme]] = i
t0 = time.time()
for i in range(L):
    names = need[i]
    if not names: continue
    done = all(meta["linears"][f"L{i}.{s}"]["bits"] == args.bits for s in sfx if f"L{i}.{s}" in meta["linears"])
    shards = sorted(set(src.map[n] for n in names))
    if not done:
        part_path = os.path.join(args.d, "parts", f"L{i}.safetensors"); part = load_file(part_path)
        for s in sfx:
            key = f"L{i}.{s}"
            if key not in meta["linears"] or meta["linears"][key]["bits"] == args.bits: continue
            W = src.f32(pre + f"layers.{i}." + HF[s]); q = quantize_linear(W, args.bits)
            part[key + ".wq"] = pack(q["wq"], args.bits); part[key + ".c"] = q["c"]; part[key + ".m"] = q["m"]
            meta["linears"][key]["bits"] = args.bits
            print(f"  L{i}.{s} {W.shape} -> W{args.bits}  [{time.time()-t0:.0f}s]", flush=True)
        save_file(part, part_path); json.dump(meta, open(os.path.join(args.d, "model.json"), "w"), indent=1)
    for sh in shards:
        if last[sh] == i: src.release(sh)
print("done", flush=True)
