"""Calibrate static V scales with the Metal det model (port of calib_v_cuda.py): runs calibration text with the current
scales, records absmax of fp32 V per (layer, kv head), writes <model_dir>/v_scales.npz (headroom x over absmax).
Usage: calib_v_metal.py <det_model_dir> <hf_tokenizer_id> [--headroom 8] [--rounds 2] [--text file]"""
import sys, os, argparse, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
from detmodel_metal import MetalModel
import detmetal as D
from transformers import AutoTokenizer
ap = argparse.ArgumentParser(); ap.add_argument("d"); ap.add_argument("tok"); ap.add_argument("--headroom", type=float, default=8.0)
ap.add_argument("--rounds", type=int, default=2); ap.add_argument("--text", default=os.path.join(os.path.dirname(__file__), "..", "dl", "wikitext", "test.txt"))
args = ap.parse_args()
tok = AutoTokenizer.from_pretrained(args.tok)
longtext = open(args.text).read()[:20000] if os.path.exists(args.text) else " ".join(["The history of science is the study of the development of science and scientific knowledge."] * 300)
texts = [longtext,
 "def fibonacci(n):\n    if n < 2:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n\nSELECT name, COUNT(*) FROM users GROUP BY name;\n",
 "La Révolution française est une période de bouleversements. 日本語のテキストもあります。 Ελληνικά, Русский, العربية. ∑ 1/n^2 = π^2/6.",
 tok.apply_chat_template([{"role": "user", "content": "Explain quantum entanglement to a five year old."}], tokenize=False, add_generation_prompt=True),
 " ".join(["hello world"] * 300)]
seqs = [tok(t).input_ids[:2048] for t in texts]
g = np.random.RandomState(0)
seqs += [g.randint(0, 200000, size=512).tolist() for _ in range(3)]
m = MetalModel(args.d, max_ctx=4096, max_batch=512)
n_kv, hd = m.n_kv, m.hd
for r in range(args.rounds):
    stats = {}
    def hook(li, v):
        a = np.abs(v.reshape(-1, n_kv, hd)).max(axis=(0, 2))
        stats[li] = np.maximum(stats.get(li, 0), a)
    m.v_hook = hook
    for ids in seqs:
        m.reset()
        for i in range(0, len(ids), 512): m.forward(ids[i:i + 512])
    D.sync()
    out = {}
    for li, a in stats.items():
        vs = (a.astype(np.float64) * args.headroom / 32767.0).astype(np.float32)
        out[f"L{li}.v_scale"] = vs; out[f"L{li}.v_inv_scale"] = (np.float32(1.0) / vs).astype(np.float32)
        print(f"round {r} layer {li}: absmax {np.round(a, 2)}")
    np.savez(os.path.join(args.d, "v_scales.npz"), **out)
    for li, ly in enumerate(m.layers):
        if ly["type"] == "full_attention":
            ly["keep"][2].copy_(out[f"L{li}.v_scale"]); ly["keep"][3].copy_(out[f"L{li}.v_inv_scale"])
print("wrote", os.path.join(args.d, "v_scales.npz"))
