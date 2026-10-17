# det — bit-exact deterministic inference for Qwen3.8-27B

`det` runs Qwen3.8-27B (the Qwen3.5 hybrid architecture: 48 Gated DeltaNet layers + 16 gated-attention
layers) with logits that are identical bit for bit across GPUs, batch sizes and prefill chunkings, and that
match a pure-numpy CPU reference.  Weights are 6-bit (group 128), activations int8 with per-group
multipliers, every GEMM and attention sum is exact integer arithmetic, and everything else is a fixed
sequence of IEEE fp32 operations.  `docs/SPEC.md` is the normative definition; `detref/` executes it.

## Results (RTX 5090, 27B, wikitext-2, llama-perplexity protocol)

| | perplexity | decode tok/s | prefill 4k tok/s | prefill 128k tok/s | VRAM |
|---|---|---|---|---|---|
| llama.cpp, unsloth UD-Q6_K GGUF | 5.982 | 64 | 3456 | 2017 | 21 GB + KV |
| det W6 (option A) | 6.004 | ~67 | ~4300 | see below | 20 GB + KV |
| det W6 + out/down at W8 | 6.000 | 63 | 4951 | 2420 | 21.8 GB + KV |

KV cache: 49.4 KB per token; 128k context fits in ~30 GB.  Decode uses CUDA graphs; prefill attention uses
int8 tensor cores (exact byte-sliced P.V).  See the git history / `out/` logs for the full benchmark tables.

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
