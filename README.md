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
- `metal/`              Metal implementation (Apple silicon), same C API as `cuda/det.h` in `det_metal.h`:
                        `det_metal.mm` (runtime: unified-memory buffers, residency set, runtime shader compilation,
                        dispatch), `shaders/*.metal` (`common.metal` primitives + software denormals, `glue`, `gemm`,
                        `attn`, `gdn`, `sample`), `make -C metal` builds `libdet_metal.dylib` (no Metal toolchain needed)
- `python/detmetal.py`  ctypes bindings (shared buffers with numpy views), `detmodel_metal.py` (model runner)
- `tests/metal/`        Metal bit-exactness suites (`run_tests_metal.sh` runs everything)
- `tools/*_metal.py`    `metal_compare.py` (vs the numpy reference), `generate_metal.py`, `bench_metal.py`,
                        `calib_v_metal.py`, `ppl_metal.py`
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

## Metal (Apple silicon)
`metal/` implements the same `det.h` API (`metal/det_metal.h`: host pointers into unified-memory buffers from
`det_malloc`, one in-order queue, `det_sync`).  Every kernel suite and the end-to-end 4B check are bit-exact against the
numpy reference (`./run_tests_metal.sh`), so the logits are identical to the CUDA path's.  Notes:
* Apple GPUs flush fp32 denormals in every arithmetic op; the Metal primitives emulate IEEE denormals exactly
  (fast-path check + integer slow path, see `docs/SPEC.md` 8).  All fp32 math goes through them.
* GEMMs and attention/GDN contractions use the M5 tensor unit (`mpp::tensor_ops::matmul2d`, int8 -> int32, exact);
  decode GEMMs stream the weights in a tiled layout (`[N/64][K/128][64 rows][group bytes]`, built at load time).
* Shaders are compiled from source at first use (a few seconds; no Metal toolchain required).

```
make -C metal && ./run_tests_metal.sh                       # kernel suites + 4B end-to-end bit-exactness
.venv/bin/python tools/generate_metal.py models/qwen38-27b-w6 Qwen/Qwen3.8-27B "prompt" 200 --chat
.venv/bin/python tools/bench_metal.py models/qwen38-27b-w6 --pp 512,2048 --tg 128 --depth 0
.venv/bin/python tools/ppl_metal.py models/qwen38-27b-w6 Qwen/Qwen3.8-27B --ctx 2048 --chunks 8
```

### Results (M5 Max, 128 GB, macOS 26)

| model | pp512 | pp2048 | tg128 | pp512 @ d4096 | pp2048 @ d4096 | tg128 @ d4096 |
|---|---|---|---|---|---|---|
| det Qwen3.8-27B W6 (22.5 GB) | 689 | 738 | 17.9 | 536 | 540 | 17.4 |
| llama.cpp Qwen3.8-27B Q8_0 (27.1 GB, fa=1) | 778 | 718 | 18.4 | 637 | 587 | 17.7 |
| det Qwen3.5-4B W6 (dev model) | ~1150 | ~1170 | ~58 | | | |

(tok/s, llama-bench protocol; `tools/bench_metal.py` / `llama-bench -fa 1`.)  Decode is within ~3% of llama.cpp with
a smaller model file and prefill is at parity (pp4096: 675 tok/s at depth 0).  The int8 tensor-unit GEMM reaches ~55
TOPS with 32-row tiles; prefill attention keeps its P.V product on the ALUs (int64), because on this chip the matrix unit
needs 128-deep contractions to be efficient and the exact 24-bit x 16-bit product would need 8-12x operand slicing.
The 4B decode is dispatch-latency bound (~390 dependent dispatches per token at ~15 us each).  Perplexity (wikitext-2 test, llama-perplexity
protocol, ctx 2048, HF tokenization of the raw text, V scales from `tools/calib_v_metal.py`): 6.193 over the full set
(`tools/ppl_metal.py`).  The numerics are bit-identical to the CUDA path for identical inputs and V scales; the gap to the
6.004 above comes from the tokenization / chunking (llama-tokenize ids there) and the calibration run, not the kernels.

Kernel level (tests/metal/test_gemm.py bench, bench_kernels): decode GEMM streams W6 weights at 450-490 GB/s and W8 at
~585 GB/s (614 GB/s peak); prefill GEMM ~55 TOPS (int8 tensor unit, 32x64 tiles); GDN recurrence 7 us/token/layer (27B shape, M >= 64);
attention (scores on the tensor unit) 77 us (1k context) / 200 us (8k) / 450 us (32k) per layer at M=1, and 8 / 55 / 235 us per
token at M=64.  Logits are bit-identical to the numpy
reference for every configuration tested (one-shot vs chunked prefill, decode), hence to the CUDA path.

## Status / next
NVIDIA path complete (sm_80/89/120 builds); Metal path complete and bit-identical to it (same numpy reference).
Determinism has been demonstrated on one GPU per vendor against the CPU reference; cross-generation runs on other
NVIDIA cards and other Apple chips are the next validation.
