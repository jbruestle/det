"""Speed benchmark of the CUDA det model, matching llama-bench's pp/tg tests.
Usage: bench.py <model_dir> [--pp 512,2048,4096] [--tg 128] [--depth 0,4096] [--reps 3]"""
import sys, os, time, argparse, numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
from detmodel_cuda import CudaModel
import detcuda as D
ap = argparse.ArgumentParser(); ap.add_argument("model"); ap.add_argument("--pp", default="512,2048,4096"); ap.add_argument("--tg", type=int, default=128)
ap.add_argument("--depth", default="0,4096"); ap.add_argument("--reps", type=int, default=3); ap.add_argument("--batch", type=int, default=2048)
ap.add_argument("--graph", default="full", choices=["full", "segments"], help="decode CUDA-graph mode (see CudaModel)")
args = ap.parse_args()
pps = [int(x) for x in args.pp.split(",")]; depths = [int(x) for x in args.depth.split(",")]
ap2 = None
maxb = args.batch; m = CudaModel(args.model, max_ctx=max(depths) + max(pps) + args.tg + 16, max_batch=maxb, graph_mode=args.graph)
print(f"model loaded: {torch.cuda.memory_allocated()/1e9:.2f} GB allocated, {torch.cuda.max_memory_reserved()/1e9:.2f} GB reserved", flush=True)
rng = np.random.RandomState(0)
def prefill(n):
    toks = rng.randint(1000, 100000, size=n).tolist()
    for i in range(0, n, maxb): lg = m.forward(toks[i:i + maxb])
    return lg
rows = []
for d in depths:
    for n in pps:
        ts = []
        for r in range(args.reps):
            m.reset()
            if d: prefill(d)
            torch.cuda.synchronize(); t0 = time.time(); prefill(n); torch.cuda.synchronize(); ts.append(time.time() - t0)
        rows.append((f"pp{n}", d, n / np.mean(ts), np.std([n / t for t in ts])))
    ts = []
    ids_buf = m.b["ids"][:1]
    for r in range(args.reps):
        m.reset(); lg = prefill(d) if d else m.forward([1000])
        lg = m.forward([1000])
        if not hasattr(m, "graphs"): m.build_decode_graphs()
        torch.cuda.synchronize(); t0 = time.time()
        am = m.b["amax"][:1]
        for _ in range(args.tg):
            D.argmax(lg, am); lg = m.decode_step(int(am.item()))
        torch.cuda.synchronize(); ts.append(time.time() - t0)
    rows.append((f"tg{args.tg}", d, args.tg / np.mean(ts), np.std([args.tg / t for t in ts])))
print(f"\n{'test':8s} {'depth':>6s} {'tok/s':>10s}")
for name, d, v, s in rows: print(f"{name:8s} {d:6d} {v:10.1f} +/- {s:.1f}")
print(f"peak reserved {torch.cuda.max_memory_reserved()/1e9:.2f} GB")
