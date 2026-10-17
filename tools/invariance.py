"""Tiling/batching invariance + long-sequence exactness of the CUDA model.
Usage: invariance.py <det_model_dir> <hf_ppl.npz> (uses <det_model_dir>/ppl.npz logits from ppl_ref.py as reference)"""
import sys, os, time, hashlib, numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
from detmodel_cuda import CudaModel
d = sys.argv[1]; ids = np.load(sys.argv[2])["ids"].tolist(); n = len(ids)
ref = np.load(os.path.join(d, "ppl.npz"))["logits"]
m = CudaModel(d, max_ctx=4096, max_batch=1024)
def run(chunks, label):
    m.reset(); out = []; torch.cuda.synchronize(); t0 = time.time()
    i = 0
    for c in chunks:
        out.append(m.forward(ids[i:i + c], all_logits=True).cpu().numpy().copy()); i += c
    torch.cuda.synchronize(); dt = time.time() - t0
    lg = np.concatenate(out); h = hashlib.sha256(lg.tobytes()).hexdigest()[:16]
    same = np.array_equal(lg.view(np.uint32), ref.view(np.uint32))
    print(f"{label:28s} {dt*1000:8.1f} ms  sha={h}  bit-exact vs reference: {same}", flush=True)
    return same
ok = True
ok &= run([n], "prefill 1x1024")
ok &= run([128] * (n // 128), "prefill 8x128")
ok &= run([1000, 24], "prefill 1000+24")
ok &= run([1] * n, "token by token (1024 steps)")
ok &= run([n], "prefill 1x1024 (again)")
print("INVARIANCE OK" if ok else "INVARIANCE FAILED")
