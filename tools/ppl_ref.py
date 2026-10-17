"""Perplexity of the det reference on the token ids saved by ppl_hf.py (same tokens -> comparable).
Usage: ppl_ref.py <det_model_dir> <hf_ppl.npz> [chunk]"""
import sys, os, time, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "detref"))
from model import DetModel
d = sys.argv[1]; base = np.load(sys.argv[2]); chunk = int(sys.argv[3]) if len(sys.argv) > 3 else 128
ids = base["ids"]; n = len(ids)
m = DetModel(d)
t0 = time.time(); logits = []
for i in range(0, n, chunk):
    logits.append(m.forward(ids[i:i + chunk].tolist()))
    print(f"  {i + chunk}/{n} tokens, {time.time() - t0:.0f}s", flush=True)
logits = np.concatenate(logits)
lg = logits[:-1].astype(np.float64); lg -= lg.max(-1, keepdims=True)
lp = lg - np.log(np.exp(lg).sum(-1, keepdims=True))
nll = -lp[np.arange(n - 1), ids[1:]]
print(f"det ppl over {n-1} tokens: {np.exp(nll.mean()):.4f}   (HF bf16: {np.exp(base['nll'].mean()):.4f})")
# token-level agreement
print("argmax agreement with HF: %.3f" % np.mean(logits.argmax(-1) == base["logits"].argmax(-1)))
np.savez(os.path.join(d, "ppl.npz"), nll=nll, logits=logits)
