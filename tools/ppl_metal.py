"""Perplexity of the Metal det model under llama.cpp's `llama-perplexity` protocol (port of ppl_protocol.py `det`):
chunks of n_ctx tokens, each from a fresh state, scoring tokens [start+n_ctx/2+1, start+n_ctx) from the logits at
positions n_ctx/2 .. n_ctx-2.  No BOS (Qwen).
Usage: ppl_metal.py <model_dir> <ids.txt|tokenizer_id> [--ctx 2048] [--chunks 8] [--text dl/wikitext/test.txt] [--batch 512]"""
import sys, os, time, argparse, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
from detmodel_metal import MetalModel
import detmetal as D
ap = argparse.ArgumentParser(); ap.add_argument("model"); ap.add_argument("tokens")
ap.add_argument("--ctx", type=int, default=2048); ap.add_argument("--chunks", type=int, default=8); ap.add_argument("--batch", type=int, default=512)
ap.add_argument("--text", default=os.path.join(os.path.dirname(__file__), "..", "dl", "wikitext", "test.txt"))
args = ap.parse_args()
if os.path.isfile(args.tokens):
    ids = [int(x) for x in open(args.tokens).read().replace("[", " ").replace("]", " ").replace(",", " ").split()]
else:
    from transformers import AutoTokenizer
    ids = AutoTokenizer.from_pretrained(args.tokens)(open(args.text).read()).input_ids
n_ctx = args.ctx; first = n_ctx // 2; n_chunk = min(args.chunks, len(ids) // n_ctx)
print(f"{len(ids)} tokens, {n_chunk} chunks of {n_ctx}, scoring {n_chunk * (n_ctx - 1 - first)} tokens", flush=True)
m = MetalModel(args.model, max_ctx=n_ctx + 8, max_batch=args.batch)
def log_softmax(x):
    x = x.astype(np.float64); x = x - x.max(-1, keepdims=True)
    return x - np.log(np.exp(x).sum(-1, keepdims=True))
nll = 0.0; count = 0; t0 = time.time()
for i in range(n_chunk):
    tok = ids[i * n_ctx:(i + 1) * n_ctx]
    m.reset()
    parts = []
    for j in range(0, n_ctx, args.batch):
        parts.append(m.forward(tok[j:j + args.batch], all_logits=True).numpy().copy())
    lg = np.concatenate(parts)
    lp = log_softmax(lg[first:n_ctx - 1])
    tgt = np.array(tok[first + 1:n_ctx])
    nll += float(-lp[np.arange(len(tgt)), tgt].sum()); count += len(tgt)
    print(f"[{i+1}]{np.exp(nll / count):.4f},", end="", flush=True)
print(f"\nFinal estimate: PPL = {np.exp(nll / count):.4f}  ({time.time()-t0:.1f}s)")
