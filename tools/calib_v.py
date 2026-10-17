"""Collect per-layer, per-kv-head absmax of v_proj outputs on the HF model (for static V scales).
Usage: calib_v.py <model_id> <out.json>"""
import sys, json, numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
mid, out = sys.argv[1], sys.argv[2]
tok = AutoTokenizer.from_pretrained(mid)
model = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.bfloat16, device_map="cuda")
model.eval()
cfg = model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config
n_kv, hd = cfg.num_key_value_heads, cfg.head_dim
stats = {}
def mk(i):
    def hook(mod, inp, outp):
        v = outp.detach().float().reshape(-1, n_kv, hd).abs().amax(dim=(0, 2)).cpu().numpy()
        stats[i] = np.maximum(stats.get(i, 0), v)
    return hook
for i, layer in enumerate(model.model.language_model.layers if hasattr(model.model, "language_model") else model.model.layers):
    if hasattr(layer, "self_attn"):
        layer.self_attn.v_proj.register_forward_hook(mk(i))
texts = [
 "The quick brown fox jumps over the lazy dog. In 1492, Columbus sailed the ocean blue. "
 "def fibonacci(n):\n    if n < 2:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n",
 "<|im_start|>user\nExplain quantum entanglement to a five year old.<|im_end|>\n<|im_start|>assistant\n<think>\nOkay, the user wants a simple explanation.\n</think>\n\nImagine two magic coins that always land the same way, even when far apart!",
 "La Révolution française est une période de bouleversements politiques et sociaux en France. 日本語のテキストもあります。 Ελληνικά, Русский, العربية.",
 "SELECT name, COUNT(*) FROM users WHERE created_at > '2024-01-01' GROUP BY name ORDER BY 2 DESC;  \n\n| a | b |\n|---|---|\n| 1 | 2 |",
 "∞ ∑_{n=1}^{∞} 1/n^2 = π^2/6. The eigenvalues of a Hermitian matrix are real. E = mc^2.",
]
with torch.no_grad():
    for t in texts:
        ids = tok(t, return_tensors="pt").input_ids.cuda()
        model(input_ids=ids)
    g = torch.Generator().manual_seed(0)
    for _ in range(4):
        ids = torch.randint(0, cfg.vocab_size, (1, 256), generator=g).cuda()
        model(input_ids=ids)
    # a long-ish repeated structure
    ids = tok(" ".join(["hello world"] * 400), return_tensors="pt").input_ids.cuda()
    model(input_ids=ids)
res = {str(k): v.tolist() for k, v in sorted(stats.items())}
json.dump(res, open(out, "w"), indent=1)
for k, v in res.items():
    print(k, ["%.2f" % x for x in v])
