"""Perplexity of the HF bf16 model on the first N tokens of a text file; saves per-token NLL.
Usage: ppl_hf.py <model_id> <text.txt> <n_tokens> <out.npz>"""
import sys, numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
mid, txt, n, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]; dt = getattr(torch, sys.argv[5]) if len(sys.argv) > 5 else torch.bfloat16
tok = AutoTokenizer.from_pretrained(mid)
ids = tok(open(txt).read()[:n * 8], return_tensors="pt").input_ids[0, :n]
model = AutoModelForCausalLM.from_pretrained(mid, dtype=dt, device_map="cuda").eval()
with torch.no_grad():
    logits = model(input_ids=ids[None].cuda()).logits[0].float()
lp = torch.log_softmax(logits[:-1], -1)
nll = -lp[torch.arange(n - 1), ids[1:].cuda()].cpu().numpy()
print(f"HF {dt} ppl over {n-1} tokens: {np.exp(nll.mean()):.4f}")
np.savez(out, ids=ids.numpy(), nll=nll, logits=logits.cpu().numpy())
