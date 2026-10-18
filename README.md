# det — bit-exact deterministic inference for Qwen3.8-27B

`det` runs Qwen3.8-27B (the Qwen3.5 hybrid architecture: 48 Gated DeltaNet layers + 16 gated-attention
layers) with logits that are identical bit for bit across GPUs, batch sizes and prefill chunkings, and that
match a pure-numpy CPU reference.  Weights are 6-bit (group 128), activations int8 with per-group
multipliers, every GEMM and attention sum is exact integer arithmetic, and everything else is a fixed
sequence of IEEE fp32 operations.  `docs/SPEC.md` is the normative definition; `detref/` executes it.

## Results (RTX 5090, Qwen3.8-27B, wikitext-2, llama-perplexity protocol, identical token ids)

| | perplexity | VRAM (weights + buffers) | max context in 32 GB |
|---|---|---|---|
| llama.cpp, unsloth UD-Q6_K GGUF | 5.982 | 21 GB + KV | — |
| det W6 (shipped: W6 everywhere, lm_head and GDN a/b at W8) | 6.004 (+0.38%) | 20 GB + 1.2 GB | ~230k (model max 262k) |
| det W6 with out_proj and down_proj at W8 | 6.000 (+0.30%) | 21.8 GB + 1.2 GB | ~190k |

Speed, tokens/s (llama-bench tests; prefill of N tokens, then 128-token decode), det = shipped W6 model:

| test | llama.cpp UD-Q6_K | det |
|---|---|---|
| prefill 2048 / 4096 | 3511 / 3456 | 4351 / 4487 |
| prefill 16k / 32k / 64k / 128k | 3464 / 3218 / 2742 / 2017 | 4333 / 3867 / 3185 / 2364 |
| decode @ 0 / 4k | 64.3 / 63.5 | 67.9 / 67.0 |
| decode @ 32k / 64k / 128k / 192k | — / 55.3 / 48.3 / — | 63.1 / 58.1 / 50.0 / 44.4 |

KV cache is 49.4 KB per token (int8 K, int16 V); 128k context runs in 28.3 GB, 192k in 30.8 GB.  Decode is one
CUDA graph per token; prefill attention runs on int8 tensor cores with exact byte-sliced P.V.  Logits are
identical for one-shot, chunked and token-by-token prefill and match the numpy CPU reference bit for bit.

## Layout
- `docs/SPEC.md`        numerics specification (normative)
- `detref/`             numpy golden reference: `detmath.py` (primitives), `quant.py` (weight format),
                        `model.py` (forward pass), `convert.py` (HF -> det, small models), `sample.py` (sampling)
- `cuda/`               CUDA implementation: `common.cuh` (device primitives), `norm.cu` (glue), `gemm.cu`,
                        `attn.cu`, `gdn.cu`, `sample.cu`, `det.h` (C API).  `make -C cuda` builds `libdet.so`.
- `python/`             `detcuda.py` (ctypes bindings), `detmodel_cuda.py` (model runner: prefill, graphed decode)
- `tests/`              per-kernel bit-exact tests vs the reference; `run_tests.sh` runs everything
- `tools/`              `convert_stream.py` (HF -> det for the 27B, resumable), `requant_tensors.py` (change the bit
                        width of selected tensors), `calib_v_cuda.py` (V scales), `extend_rope.py`, `generate.py`
                        (chat generation), `bench.py`, `ppl_protocol.py` / `ppl_cuda.py` / `ppl_hf.py` (perplexity),
                        `cuda_compare.py` / `invariance.py` / `long_invariance.py` (bit-exactness), `ablate.py`
                        (quantization ablations), `hf_baseline.py`, `calib_v.py` (HF-side calibration)

## Reproduce
```
python3 -m venv .venv && .venv/bin/pip install numpy safetensors huggingface_hub transformers torch
make -C cuda
# 4B dev model (same architecture), reference baseline and full bit-exact check:
.venv/bin/python tools/hf_baseline.py Qwen/Qwen3.5-4B out/hf_4b_baseline.npz
.venv/bin/python tools/calib_v.py Qwen/Qwen3.5-4B out/calib_v_4b.json
.venv/bin/python detref/convert.py <hf_snapshot_dir> models/qwen35-4b-w6 --bits 6 --calib out/calib_v_4b.json
./run_tests.sh
# 27B: convert from the Hub (or a local snapshot with --local <dir>), calibrate V scales, generate:
.venv/bin/python tools/convert_stream.py Qwen/Qwen3.8-27B models/qwen38-27b-w6 --bits 6
.venv/bin/python tools/calib_v_cuda.py models/qwen38-27b-w6 Qwen/Qwen3.8-27B
.venv/bin/python tools/extend_rope.py models/qwen38-27b-w6 262144
.venv/bin/python tools/generate.py models/qwen38-27b-w6 Qwen/Qwen3.8-27B "prompt" 200 --chat [--temp 0.8 --seed 1]
```

## Status / next
NVIDIA path complete (sm_80/89/120 builds).  Determinism has been demonstrated on one GPU against the CPU
reference; cross-generation runs on other NVIDIA cards are the next validation.  A Metal backend behind the same
`det.h` API is planned; the first step there is a bit-exactness test of the fp32 primitives (and denormals).
