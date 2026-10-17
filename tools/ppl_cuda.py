"""Perplexity of the CUDA det model on wikitext-2 test tokens (tokenized with the given tokenizer).
Usage: ppl_cuda.py <det_model_dir> <hf_tokenizer_id> [n_tokens] [chunk]"""
import sys, os, time, numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
from detmodel_cuda import CudaModel
from transformers import AutoTokenizer
d, tk = sys.argv[1], sys.argv[2]; n = int(sys.argv[3]) if len(sys.argv) > 3 else 2048; chunk = int(sys.argv[4]) if len(sys.argv) > 4 else 512
tok = AutoTokenizer.from_pretrained(tk)
ids = tok(open(os.path.join(os.path.dirname(__file__), "..", "dl", "wikitext", "test.txt")).read()[:n * 8]).input_ids[:n]
m = CudaModel(d, max_ctx=n + 8, max_batch=chunk); print(f"loaded, {torch.cuda.memory_allocated()/1e9:.2f} GB", flush=True)
nll = []; torch.cuda.synchronize(); t0 = time.time()
for i in range(0, n, chunk):
    lg = m.forward(ids[i:i + chunk], all_logits=True).double()
    lp = torch.log_softmax(lg, -1)
    tgt = torch.tensor(ids[i + 1:i + chunk + 1], device="cuda")
    k = min(len(tgt), lg.shape[0] - (1 if i + chunk >= n else 0))
    nll.append(-lp[torch.arange(k), tgt[:k]].cpu().numpy())
torch.cuda.synchronize(); dt = time.time() - t0
nll = np.concatenate(nll)
print(f"ppl over {len(nll)} tokens: {np.exp(nll.mean()):.4f}  ({n} tokens in {dt:.1f}s = {n/dt:.0f} tok/s prefill)")
