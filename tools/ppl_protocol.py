"""Perplexity under llama.cpp's `llama-perplexity` protocol, for the det CUDA model (or HF torch model):
chunks of n_ctx tokens, each evaluated from a fresh state, scoring predictions of tokens
[start+n_ctx/2+1, start+n_ctx) from the logits at positions n_ctx/2 .. n_ctx-2.  No BOS (Qwen).
Usage: ppl_protocol.py det <model_dir> <ids.txt|tokenizer_id> [--ctx 2048] [--chunks 8]
       ppl_protocol.py hf  <model_id>  <ids.txt|tokenizer_id> [--ctx 2048] [--chunks 8]
ids.txt: whitespace-separated token ids (e.g. from `llama-tokenize --ids`); otherwise the text is tokenized
with the given HF tokenizer."""
import sys, os, time, argparse, numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
ap = argparse.ArgumentParser(); ap.add_argument("kind"); ap.add_argument("model"); ap.add_argument("tokens")
ap.add_argument("--ctx", type=int, default=2048); ap.add_argument("--chunks", type=int, default=8)
ap.add_argument("--text", default=os.path.join(os.path.dirname(__file__), "..", "dl", "wikitext", "test.txt"))
args = ap.parse_args()
if os.path.exists(args.tokens):
    ids = [int(x) for x in open(args.tokens).read().replace("[", " ").replace("]", " ").replace(",", " ").split()]
else:
    from transformers import AutoTokenizer
    ids = AutoTokenizer.from_pretrained(args.tokens)(open(args.text).read()).input_ids
n_ctx = args.ctx; first = n_ctx // 2; n_chunk = min(args.chunks, len(ids) // n_ctx)
print(f"{len(ids)} tokens, {n_chunk} chunks of {n_ctx}, scoring {n_chunk * (n_ctx - 1 - first)} tokens", flush=True)
if args.kind == "det":
    from detmodel_cuda import CudaModel
    m = CudaModel(args.model, max_ctx=n_ctx + 8, max_batch=n_ctx)
    def chunk_logits(tok):
        m.reset(); return m.forward(tok, all_logits=True).float()
else:
    from transformers import AutoModelForCausalLM
    hm = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda").eval()
    def chunk_logits(tok):
        with torch.no_grad(): return hm(input_ids=torch.tensor([tok], device="cuda")).logits[0].float()
nll = 0.0; count = 0; t0 = time.time()
for i in range(n_chunk):
    tok = ids[i * n_ctx:(i + 1) * n_ctx]
    lg = chunk_logits(tok)
    lp = torch.log_softmax(lg[first:n_ctx - 1].double(), -1)
    tgt = torch.tensor(tok[first + 1:n_ctx], device="cuda")
    nll += float(-lp[torch.arange(len(tgt)), tgt].sum()); count += len(tgt)
    print(f"[{i+1}]{np.exp(nll / count):.4f},", end="", flush=True)
print(f"\nFinal estimate: PPL = {np.exp(nll / count):.4f}  ({time.time()-t0:.1f}s)")
