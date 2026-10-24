# det

Bit-exact deterministic LLM inference: identical logits on every GPU, batch size and prefill chunking,
matching a CPU reference. `docs/SPEC.md` is the normative numerics; `crates/det/src/spec` is its
executable form, and every backend must reproduce it bit for bit. `README.md` has the layout, the
commands and the measurements.

Conventions:

- Do not overcomment code: the code should be readable enough to be its own reference. Why something
  exists belongs in `docs/SPEC.md` or the README.
- Never put change logs or references to how old versions worked in code or docs; history lives in
  commit messages. Docs and code describe current behavior only.
- Nothing is sacred: until users depend on the library, huge refactors are fine.
- Always do the simplest thing unless the complexity is required for performance or bit-exactness.
  Less code, fewer docs, fewer new terms and abstractions; orthogonal pieces over coupled ones.
- The library is model inference only. Consumers that need observation (calibration, debugging, external
  verification) go through `hooks.rs`; nothing specific to any consumer lives here.
- Weights are det's own integer-scale quantization inside a GGUF container (`det convert` from a bf16
  GGUF). Standard ggml quantizations were tried and rejected for performance (Q6_K needs two int8 MMAs
  per K slice); do not reintroduce them.

Orientation:

- Workspace: `crates/det` (library; `spec/` normative, `cpu.rs` reference, `x86/`, `metal.rs`, `cuda.rs`
  backends behind one `Backend` trait), `crates/det-sys` (FFI and the GPU runtime build), `crates/det-cli`
  (the `det` binary). `kernels/metal` and `kernels/cuda` hold the runtimes and shaders, one C API for both.
- `crates/det-sys/build.rs` compiles the GPU runtime: the Metal runtime with the shaders embedded
  (feature `metal`, macOS only, cargo tracks shader edits) or the CUDA kernels with nvcc (feature `cuda`;
  `det-cli` turns it on for Linux; `CUDA_HOME`, `DET_CUDA_ARCHS`). `DET_METAL_SHADERS=<dir>` reads
  shaders from disk instead; `DET_METAL_DEFINES=A=1,B=2` injects defines for experiments.
- Tests: `cargo test --release -p det` on a Mac, `cargo test --release -p det --features cuda` on a CUDA
  box, `--no-default-features` on any x86-64 box (spec unit tests; every GPU and x86 op against the
  reference CPU backend bit for bit). End to end: `det generate <gguf> --backend ref --logits a.bin` vs
  the x86 (`--backend cpu`) and GPU runs and `cmp`.
- Each GPU runtime is one global in-order queue: GPU tests take a mutex, and benchmarks warm up first
  (`det bench --gemm` reports its second pass).
- Ground truth for a model's numerics: llama.cpp on the bf16 GGUF (`llama-completion -no-cnv
  --verbose-prompt` for greedy output and token ids, `llama-eval-callback` for every intermediate tensor,
  `llama-perplexity -c 2048 --chunks 8`, which scores the second half of each chunk as `det ppl` does).
  The GDN v heads are stored for `h mod n_k` in llama.cpp files; `det convert` reorders them to `h / rep`.
- `models/` holds the bf16 sources and the converted `*-det-w6.gguf` files (4B dev, 27B target); `dl/`
  the downloaded HF snapshots; `out/` scratch logs. All three are ignored. Never un-ignore `dl/` or
  `models/`: HF snapshot directories carry a `.gitattributes` that routes their files to git LFS, and
  `git add -A` would commit tens of GB.
