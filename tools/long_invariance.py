"""4096-token exactness: CUDA (one shot, chunked, mixed) vs numpy reference (chunked) on wikitext tokens."""
import sys, os, time, hashlib, numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python")); sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "detref"))
from detmodel_cuda import CudaModel
from model import DetModel
from transformers import AutoTokenizer
d, n = sys.argv[1], int(sys.argv[2])
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B")
ids = tok(open(os.path.join(os.path.dirname(__file__), "..", "dl", "wikitext", "test.txt")).read()[:n * 8]).input_ids[:n]
m = CudaModel(d, max_ctx=n + 64, max_batch=n)
res = {}
for label, chunks in [("one-shot", [n]), ("chunks of 512", [512] * (n // 512)), ("mixed", [3000, 1, 1, 1094])]:
    m.reset(); out = []; i = 0; torch.cuda.synchronize(); t0 = time.time()
    for c in chunks: out.append(m.forward(ids[i:i + c], all_logits=True).cpu().numpy().copy()); i += c
    torch.cuda.synchronize(); lg = np.concatenate(out); res[label] = lg
    print(f"cuda {label:14s} {time.time()-t0:.2f}s sha={hashlib.sha256(lg.tobytes()).hexdigest()[:16]}", flush=True)
del m; torch.cuda.empty_cache()
r = DetModel(d); t0 = time.time(); out = []
for i in range(0, n, 256): out.append(r.forward(ids[i:i + 256])); print(f"  ref {i+256}/{n} {time.time()-t0:.0f}s", flush=True)
ref = np.concatenate(out); print(f"ref sha={hashlib.sha256(ref.tobytes()).hexdigest()[:16]}")
ok = all(np.array_equal(v.view(np.uint32), ref.view(np.uint32)) for v in res.values())
print("LONG INVARIANCE OK" if ok else "LONG INVARIANCE FAILED")
