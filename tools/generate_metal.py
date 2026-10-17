"""Greedy / sampled generation with the Metal det model.
Usage: generate_metal.py <det_model_dir> <hf_tokenizer_id> "<prompt>" [n_new] [--chat] [--ctx N] [--temp T --seed S --top-k K --top-p P]"""
import sys, os, time, argparse, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
from detmodel_metal import MetalModel
import detmetal as D
from transformers import AutoTokenizer
ap = argparse.ArgumentParser()
ap.add_argument("model"); ap.add_argument("tokenizer"); ap.add_argument("prompt"); ap.add_argument("n_new", type=int, nargs="?", default=64)
ap.add_argument("--chat", action="store_true"); ap.add_argument("--ctx", type=int, default=8192); ap.add_argument("--batch", type=int, default=256)
ap.add_argument("--think", action="store_true")
ap.add_argument("--temp", type=float, default=0.0, help="0 = greedy; else deterministic multinomial sampling")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--top-k", type=int, default=0); ap.add_argument("--top-p", type=float, default=1.0)
args = ap.parse_args()
tok = AutoTokenizer.from_pretrained(args.tokenizer)
text = tok.apply_chat_template([{"role": "user", "content": args.prompt}], tokenize=False, add_generation_prompt=True, enable_thinking=args.think) if args.chat else args.prompt
ids = tok(text).input_ids
t0 = time.time(); m = MetalModel(args.model, max_ctx=args.ctx, max_batch=args.batch); print(f"load {time.time()-t0:.1f}s, {D.allocated_bytes()/1e9:.2f} GB", file=sys.stderr)
t0 = time.time()
for i in range(0, len(ids), args.batch): lg = m.forward(ids[i:i + args.batch])
D.sync(); tp = time.time() - t0
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "detref")); import sample as S
out = []; am = m.b["amax"][:1]; t0 = time.time(); rng = args.seed; zt = D.empty((1,), np.uint64)
V = lg.shape[1]; topkp = (0 < args.top_k < V) or args.top_p < 1.0
scratch = D.empty((D.sample_scratch_bytes(1, V),), np.uint8) if (args.temp > 0 and topkp) else None
for _ in range(args.n_new):
    if args.temp > 0:
        rng, z = S.splitmix64(rng); D.sync(); zt.np[0] = np.uint64(z)
        if topkp: D.sample_topkp(lg, np.float32(1.0 / args.temp), args.top_k, args.top_p, zt, am, scratch)
        else: D.sample(lg, np.float32(1.0 / args.temp), zt, am)
    else:
        D.argmax(lg, am)
    nxt = int(am.item()); out.append(nxt)
    if nxt == tok.eos_token_id or nxt in (248044, 248045): break
    lg = m.forward([nxt])
D.sync(); td = time.time() - t0
print(tok.decode(out))
print(f"\n[prefill {len(ids)} tok in {tp:.3f}s = {len(ids)/tp:.0f} tok/s; decode {len(out)} tok in {td:.3f}s = {len(out)/td:.1f} tok/s]", file=sys.stderr)
