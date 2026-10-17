"""Bit-exact check of the Metal model against the numpy reference on a prompt + greedy continuation.
Usage: metal_compare.py <det_model_dir> <hf_tokenizer_id_or_baseline.npz> [n_gen] [--chunk N] [--ref-logits out.npz]
The prompt ids come from the HF baseline .npz if given (then the HF greedy continuation is printed too), else from the
tokenizer on a fixed prompt."""
import sys, os, time, argparse, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "detref"))
from detmodel_metal import MetalModel
import detmetal as D
from model import DetModel
ap = argparse.ArgumentParser(); ap.add_argument("d"); ap.add_argument("src"); ap.add_argument("n_gen", type=int, nargs="?", default=8)
ap.add_argument("--chunk", type=int, default=0, help="also prefill in chunks of this size (chunking invariance)")
ap.add_argument("--save-ref", default=None, help="save the reference logits (reuse with --load-ref)")
ap.add_argument("--load-ref", default=None)
args = ap.parse_args()
base = None
if args.src.endswith(".npz"):
    base = np.load(args.src); ids = base["ids"].tolist()
else:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.src)
    ids = tok("The capital of France is Paris. The capital of Germany is Berlin. The capital of Italy is").input_ids
print("prompt ids:", ids, flush=True)
mm = MetalModel(args.d, max_ctx=4096, max_batch=64)
t0 = time.time(); lg_m = mm.forward(ids, all_logits=True).numpy().copy(); print(f"metal prefill {len(ids)} tok: {time.time()-t0:.3f}s", flush=True)
def same(a, b): return np.array_equal(a.view(np.uint32), b.view(np.uint32))
if args.load_ref:
    rf = np.load(args.load_ref); lg_r = rf["prefill"]; ref_steps = rf["steps"]; rm = None
else:
    rm = DetModel(args.d)
    t0 = time.time(); lg_r = rm.forward(ids); print(f"ref prefill: {time.time()-t0:.1f}s", flush=True)
ok = same(lg_m, lg_r)
print("prefill logits bit-exact:", ok, "" if ok else f"max abs diff {np.abs(lg_m - lg_r).max()}, n_diff {np.sum(lg_m != lg_r)} / {lg_m.size}", flush=True)
allok = ok
if args.chunk:
    mm.reset(); parts = []
    for i in range(0, len(ids), args.chunk): parts.append(mm.forward(ids[i:i + args.chunk], all_logits=True).numpy().copy())
    lg_c = np.concatenate(parts); s = same(lg_c, lg_r); allok &= s
    print(f"chunked prefill (chunk={args.chunk}) bit-exact:", s, flush=True)
    mm.reset(); lg_m2 = mm.forward(ids, all_logits=True).numpy().copy(); assert same(lg_m2, lg_m)
nxt_m = int(lg_m[-1].argmax()); nxt_r = int(lg_r[-1].argmax())
gen_m, gen_r, steps = [], [], []
for i in range(args.n_gen):
    gen_m.append(nxt_m); gen_r.append(nxt_r)
    t0 = time.time(); lm = mm.forward([nxt_m]).numpy().copy(); tm = time.time() - t0
    lr = rm.forward([nxt_r], return_all=False) if rm is not None else ref_steps[i]
    steps.append(lr)
    s = same(lm, lr); allok &= s
    print(f"step {i}: tok metal={nxt_m} ref={nxt_r} exact={s} ({tm*1000:.1f} ms/token metal)", flush=True)
    nxt_m = int(lm[-1].argmax()); nxt_r = int(lr[-1].argmax())
if args.save_ref: np.savez(args.save_ref, prefill=lg_r, steps=np.stack(steps))
print("metal greedy:", gen_m); print("ref   greedy:", gen_r)
if base is not None: print("hf    greedy:", base["gen_ids"][:args.n_gen].tolist())
print("ALL BIT-EXACT" if allok else "MISMATCH")
sys.exit(0 if allok else 1)
