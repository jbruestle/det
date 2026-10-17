"""Run the HF bf16 model on a fixed prompt; save token ids, full-sequence logits, and greedy continuation.
Usage: hf_baseline.py <model_dir_or_id> <out.npz> [n_gen]"""
import sys, os, numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
mid, out = sys.argv[1], sys.argv[2]
dev = os.environ.get("HF_DEVICE", "cuda")
n_gen = int(sys.argv[3]) if len(sys.argv) > 3 else 32
tok = AutoTokenizer.from_pretrained(mid)
model = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.bfloat16, device_map=dev)
model.eval()
prompt = "The capital of France is Paris. The capital of Germany is Berlin. The capital of Italy is"
ids = tok(prompt, return_tensors="pt").input_ids.to(dev)
with torch.no_grad():
    logits = model(input_ids=ids).logits[0].float().cpu().numpy()
    gen = model.generate(ids, max_new_tokens=n_gen, do_sample=False)
gen_ids = gen[0, ids.shape[1]:].cpu().numpy()
print("prompt ids:", ids[0].tolist())
print("greedy:", repr(tok.decode(gen_ids)))
print("top5 last:", np.argsort(-logits[-1])[:5], logits[-1][np.argsort(-logits[-1])[:5]])
np.savez(out, ids=ids[0].cpu().numpy(), logits=logits, gen_ids=gen_ids)
