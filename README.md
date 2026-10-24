# det — bit-exact deterministic LLM inference

`det` runs Qwen3.5-architecture models (Gated DeltaNet + gated attention; the 27B is the target, the 4B
the dev model) with logits that are identical bit for bit across GPUs, batch sizes and prefill
chunkings, and that match a CPU reference. Weights are 6-bit (group 128) in det's own integer-scale
format inside a GGUF container, activations int8 with per-group multipliers; every GEMM and attention
sum is exact integer arithmetic, and everything else is a fixed sequence of IEEE fp32 operations.
`docs/SPEC.md` is the normative definition; `crates/det/src/spec` executes it.

```
cargo build --release                        # the det binary with the GPU backend of the platform (Metal on macOS, CUDA on Linux)
det convert Qwen3.5-4B-BF16.gguf             # bf16/f16 GGUF (as llama.cpp's converter writes) -> Qwen3.5-4B-det-w6.gguf, calibrated
det generate model.gguf --prompt "The capital of France is" --n 32          # the GPU backend (default), --backend cpu (AVX2 / AVX-VNNI) or --backend ref
det generate model.gguf --chat --prompt "Write a haiku." --temp 0.7 --top-p 0.95   # the file's chat template; top-k/top-p
det bench model.gguf --pp 512 --tg 128      det bench model.gguf --gemm     # throughput; per-linear GEMM rates
det ppl model.gguf wikitext-test.txt --ctx 2048 --chunks 8                  # llama-perplexity protocol
det info model.gguf  |  det tensor model.gguf blk.0.attn_norm.weight  |  det tokens model.gguf 248040 248060
```

`det convert` quantizes every linear (`--bits` sets the width per linear; the default is W6 with W8 for
the output head and the GDN a/b projections),
writes the rotary tables and the output head, and calibrates the int16 value cache (one static scale
per attention layer and kv head, `det.v_scale` in the metadata; `--no-calibrate` leaves the default
scale, `det calibrate` writes it later in place). The tokenizer, the chat template and the
hyperparameters come from the GGUF metadata, so nothing else is needed. Split source files are read as
one.

Width presets measured on the 4B (wikitext-2, ctx 2048, 8 chunks; bf16 through llama.cpp is 7.46):

| `--bits` | size | perplexity |
|---|---|---|
| `8` | 5.6 GB | 7.44 |
| `6,ffn_down=8,ssm_out=8,attn_output=8` | 4.8 GB | 7.46 (27B: 6.27, +8% size, −4% decode) |
| `6` (the default) | 4.7 GB | 7.57 (27B: 6.28) |
| `6,ffn_gate=4,ffn_up=4,ffn_down=4` | 4.0 GB | 7.77 |
| `4,ffn_down=6,ssm_out=6,attn_output=6` | 3.8 GB | 7.87 |
| `4` | 3.5 GB | 8.04 |

## Layout

- `docs/SPEC.md`: the numerics (normative).
- `crates/det`: the library.
  - `spec/`: the numerics as plain scalar Rust (primitives, the weight row and GEMM element, activation
    quantization, the attention and GDN steps, sampling). Normative.
  - `quantize.rs`, `convert.rs`: the weight quantizer and `det convert`.
  - `backend.rs`: the op set a backend implements (`Backend`) and its buffer type `Buf`.
  - `cpu.rs`: the reference backend (`--backend ref`), the spec applied row by row (parallel only across
    independent rows and heads).
  - `x86/`: the x86-64 backend (`--backend cpu`): `gemm.rs` repacks the weights at load into 8-row strips
    and runs the integer GEMM on `vpdpbusd` (AVX-VNNI) or `vpmaddubsw` (AVX2 only); `attn.rs`, `gdn.rs`,
    `glue.rs` (norms and activation quantization) and `simd.rs` (the fp32 primitives on 8 lanes) cover the
    rest. Threads come from rayon (`RAYON_NUM_THREADS`); `DET_X86_VNNI=0` forces the AVX2 dot products.
  - `metal.rs`: the Metal backend over `det-sys`; builds the tiled weight layout the kernels read.
  - `cuda.rs`: the CUDA backend over `det-sys`; row-major weights, the KV cache without Metal's V planes.
  - `qwen35.rs`: the network's forward pass against the op set; `model.rs`: loading, buffers, generation.
  - `hooks.rs`: observation taps (residual stream, attention inputs/outputs, GDN inputs/outputs/state,
    logits, sampled tokens) for calibration, debugging and external observers; `calibrate.rs` uses them.
  - `gguf.rs`, `config.rs`, `tokenizer.rs`: the file, its hyperparameters, its tokenizer.
  - `tests/backends.rs`: every op of the GPU and x86 backends against the reference backend, bit for bit;
    `tests/tokenizer.rs`.
- `crates/det-sys`: FFI to the GPU runtime, one C API for both; the `metal` feature compiles
  `kernels/metal` with the shaders embedded (cargo tracks shader edits; `DET_METAL_SHADERS=<dir>` reads
  them from disk instead), the `cuda` feature runs nvcc from `CUDA_HOME` (default `/usr/local/cuda`) on
  `kernels/cuda` for the SM versions in `DET_CUDA_ARCHS` (default `120,89,80`), linking `cudart_static`.
- `crates/det-cli`: the `det` command.
- `kernels/metal`: the Metal runtime (`det_metal.mm`: unified-memory buffers, one in-order queue,
  runtime shader compilation) and the kernels (`shaders/common.metal`: the fp32 primitives with software
  denormals; `gemm`, `attn`, `gdn`, `glue`, `sample`). `det_metal.h` is the C API.
- `kernels/cuda`: the CUDA runtime (`runtime.cu`: device buffers, the default stream) and the kernels
  (`common.cuh`: the fp32 primitives; `gemm`, `attn`, `gdn`, `norm`, `sample`). `det_cuda.h` is the C API,
  the same functions as `det_metal.h`. Knobs: `DET_ATTN_TC` (tensor-core attention from this many tokens
  on, default 16; 0 disables), `DET_GEMM_DP4A` (1 forces the dp4a GEMM for M > 8, 2 the 32x32-warp-tile
  MMA kernel for M > 64).

## Status

Measured on an M5 Max (tok/s): Qwen3.5-4B W6 prefill 512 at 3091, decode 69.5 (W8: 3083 / 70.0);
Qwen3.8-27B W6 prefill 2048 at 671, prefill 65536 at 278, decode 17.3 (14.8 at 64k depth).
Wikitext-2 perplexity (ctx 2048, llama-perplexity protocol, calibrated): 4B W6 7.57, W8 7.44, llama.cpp
on the bf16 file 7.46 (8 chunks); 27B W6 6.28, llama.cpp bf16 6.22 (4 chunks). CPU and Metal agree bit
for bit on every path (`cargo test --release -p det`, and `det generate --logits` on each backend).

On an RTX 5090 (CUDA), tok/s; det prefills N tokens from an empty context in batches of 1024 (the average
rate over the whole fill; between 64k and 128k it runs at 1915) and then decodes at that depth, llama.cpp
prefills 2048 tokens at that depth:

| | 4B W6 | 27B W6 | llama.cpp Q6_K 27B (pp2048 at that depth / decode) |
|---|---|---|---|
| prefill 512 (batch 256) / decode | 12931 / 243.6 | 2333 / 60.1 | 2938 / 65.7 |
| 8k | | 4592 / 58.8 | 3299 / 62.8 |
| 32k | | 3866 / 56.3 | 2611 / 59.3 |
| 64k | | 3197 / 51.7 | 1756 / 55.1 |
| 128k | | 2395 / 45.7 | 1111 / 48.2 |

CPU and CUDA logits are identical for both models on a 5-token prompt with 16 decoded tokens and on a
prefill (300 tokens for the 4B, 512 for the 27B); the 27B logits after 8k, 64k and 128k tokens are
identical for prefill batches of 256 and 1024. Perplexity on CUDA: 4B W6 7.57, 27B W6 5.08 over 8
chunks (llama.cpp Q6_K 5.07).

On a Core Ultra 9 285K (8 P + 16 E cores, AVX2 + AVX-VNNI, DDR5 reading at ~70 GB/s), the x86 backend
prefills the 4B W6 at 495 tok/s (512 tokens, batch 256) and decodes at 17.1 tok/s; the 27B W6 prefills at
91 tok/s and decodes at 3.0 tok/s, the DRAM rate of its 22.6 GB. Its logits match the reference and CUDA
backends bit for bit on the same prompts, with and without VNNI.

Next: CUDA graphs for decode, AVX-512 / AMX paths, more architectures.
