"""Write a larger RoPE table part file for a det model (same float64 formula as the converter).
Usage: extend_rope.py <model_dir> <positions>"""
import sys, os, json, numpy as np
from safetensors.numpy import save_file
d, P = sys.argv[1], int(sys.argv[2])
meta = json.load(open(os.path.join(d, "model.json")))
rot, theta = meta["rot_dim"], meta["rope_theta"]
inv_freq = 1.0 / (theta ** (np.arange(0, rot, 2, dtype=np.float64) / rot))
ang = np.arange(P, dtype=np.float64)[:, None] * inv_freq[None, :]
os.makedirs(os.path.join(d, "parts"), exist_ok=True)
save_file({"rope_cos": np.cos(ang).astype(np.float32), "rope_sin": np.sin(ang).astype(np.float32)}, os.path.join(d, "parts", "rope_ext.safetensors"))
files = meta.get("files") or ["tensors.safetensors"]
if "rope_ext.safetensors" not in files: files.append("rope_ext.safetensors")
meta["files"] = sorted(files); json.dump(meta, open(os.path.join(d, "model.json"), "w"), indent=1)
print("rope table extended to", P, "positions")
