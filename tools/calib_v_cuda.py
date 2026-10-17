"""Calibrate static V scales with the CUDA det model itself: runs calibration text with the current (placeholder)
scales, records absmax of fp32 V per (layer, kv head), and writes <model_dir>/v_scales.npz (headroom x over absmax).
Usage: calib_v_cuda.py <det_model_dir> <hf_tokenizer_id> [--headroom 8] [--rounds 2]"""
import sys, os, json, argparse, numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
from detmodel_cuda import CudaModel
from transformers import AutoTokenizer
ap = argparse.ArgumentParser(); ap.add_argument("d"); ap.add_argument("tok"); ap.add_argument("--headroom", type=float, default=8.0)
ap.add_argument("--rounds", type=int, default=2); ap.add_argument("--text", default=os.path.join(os.path.dirname(__file__), "..", "dl", "wikitext", "test.txt"))
args = ap.parse_args()
tok = AutoTokenizer.from_pretrained(args.tok)
texts = [open(args.text).read()[:20000],
 "def fibonacci(n):\n    if n < 2:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n\nSELECT name, COUNT(*) FROM users GROUP BY name;\n",
 "La Révolution française est une période de bouleversements. 日本語のテキストもあります。 Ελληνικά, Русский, العربية. ∑ 1/n^2 = π^2/6.",
 tok.apply_chat_template([{"role": "user", "content": "Explain quantum entanglement to a five year old."}], tokenize=False, add_generation_prompt=True),
 " ".join(["hello world"] * 300)]
seqs = [tok(t).input_ids[:2048] for t in texts]
g = torch.Generator().manual_seed(0)
seqs += [torch.randint(0, 200000, (512,), generator=g).tolist() for _ in range(3)]
m = CudaModel(args.d, max_ctx=4096, max_batch=512)
n_kv, hd = m.n_kv, m.hd
for r in range(args.rounds):
    stats = {}
    def hook(li, v):
        a = v.view(-1, n_kv, hd).abs().amax(dim=(0, 2)).cpu().numpy()
        stats[li] = np.maximum(stats.get(li, 0), a)
    m.v_hook = hook
    for ids in seqs:
        m.reset()
        for i in range(0, len(ids), 512): m.forward(ids[i:i + 512])
    out = {}
    for li, a in stats.items():
        vs = (a.astype(np.float64) * args.headroom / 32767.0).astype(np.float32)
        out[f"L{li}.v_scale"] = vs; out[f"L{li}.v_inv_scale"] = (np.float32(1.0) / vs).astype(np.float32)
        print(f"round {r} layer {li}: absmax {np.round(a, 2)}")
    np.savez(os.path.join(args.d, "v_scales.npz"), **out)
    # reload with the new scales for the next round
    for li, ly in enumerate(m.layers):
        if ly["type"] == "full_attention":
            ly["keep"][2].copy_(torch.from_numpy(out[f"L{li}.v_scale"])); ly["keep"][3].copy_(torch.from_numpy(out[f"L{li}.v_inv_scale"]))
print("wrote", os.path.join(args.d, "v_scales.npz"))
