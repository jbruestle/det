"""Bit-exact check of the CUDA model against the numpy reference on a prompt + greedy continuation.
Usage: cuda_compare.py <det_model_dir> <hf_baseline.npz> [n_gen]"""
import sys, os, time, numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "detref"))
from detmodel_cuda import CudaModel
from model import DetModel
d, base = sys.argv[1], np.load(sys.argv[2]); n_gen = int(sys.argv[3]) if len(sys.argv) > 3 else 8
ids = base["ids"].tolist()
cm = CudaModel(d, max_ctx=4096, max_batch=64, verbose=False)
t0 = time.time(); lg_c = cm.forward(ids, all_logits=True).cpu().numpy().copy(); torch.cuda.synchronize(); print(f"cuda prefill {len(ids)} tok: {time.time()-t0:.3f}s", flush=True)
rm = DetModel(d)
t0 = time.time(); lg_r = rm.forward(ids); print(f"ref prefill: {time.time()-t0:.1f}s", flush=True)
def same(a, b): return np.array_equal(a.view(np.uint32), b.view(np.uint32))
ok = same(lg_c, lg_r)
print("prefill logits bit-exact:", ok, "" if ok else f"max abs diff {np.abs(lg_c - lg_r).max()}, n_diff {np.sum(lg_c != lg_r)} / {lg_c.size}")
nxt_c = int(lg_c[-1].argmax()); nxt_r = int(lg_r[-1].argmax())
gen_c, gen_r = [], []
allok = ok
for i in range(n_gen):
    gen_c.append(nxt_c); gen_r.append(nxt_r)
    t0 = time.time(); lc = cm.forward([nxt_c]).cpu().numpy().copy(); torch.cuda.synchronize(); tc = time.time() - t0
    lr = rm.forward([nxt_r], return_all=False)
    s = same(lc, lr); allok &= s
    print(f"step {i}: tok cuda={nxt_c} ref={nxt_r} exact={s} ({tc*1000:.1f} ms/token cuda)", flush=True)
    nxt_c = int(lc[-1].argmax()); nxt_r = int(lr[-1].argmax())
print("cuda greedy:", gen_c); print("ref  greedy:", gen_r); print("hf   greedy:", base["gen_ids"][:n_gen].tolist())
print("ALL BIT-EXACT" if allok else "MISMATCH")
