"""Run the det reference on the HF baseline prompt and compare logits / greedy continuation.
Usage: ref_compare.py <det_model_dir> <hf_baseline.npz> [n_gen]"""
import sys, os, time, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "detref"))
from model import DetModel
d, base = sys.argv[1], np.load(sys.argv[2])
n_gen = int(sys.argv[3]) if len(sys.argv) > 3 else 8
ids = base["ids"].tolist()
t0 = time.time(); m = DetModel(d); print("load %.1fs" % (time.time() - t0), flush=True)
t0 = time.time(); lg = m.forward(ids, verbose=False); print("prefill %d tokens %.1fs" % (len(ids), time.time() - t0), flush=True)
hf = base["logits"]
for t in [0, len(ids) // 2, len(ids) - 1]:
    a, b = lg[t], hf[t]
    pa = np.exp(a - a.max()); pa /= pa.sum(); pb = np.exp(b - b.max()); pb /= pb.sum()
    kl = float((pb * (np.log(pb + 1e-30) - np.log(pa + 1e-30))).sum())
    print(f"pos {t}: argmax det={a.argmax()} hf={b.argmax()}  top5 det={np.argsort(-a)[:5]} hf={np.argsort(-b)[:5]}  KL(hf||det)={kl:.4f}")
gen = []
nxt = int(lg[-1].argmax())
for i in range(n_gen):
    gen.append(nxt)
    lg = m.forward([nxt], return_all=False)
    nxt = int(lg[-1].argmax())
print("det greedy:", gen)
print("hf  greedy:", base["gen_ids"][:n_gen].tolist())
np.save(os.path.join(d, "ref_logits_prompt.npy"), lg)
