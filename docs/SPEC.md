# det: bit-exact deterministic inference

Goal: identical logits, bit for bit, on every GPU, vendor and generation, for every batching and prefill
chunking, with ~6-bit weights and competitive speed. This document
defines the numerics. `crates/det/src/spec` is its executable form, written as plainly as possible, and
every backend (the CPU backend's batched loops, the Metal and CUDA kernels) must reproduce it bit for
bit; `crates/det/tests` checks that they do.

## 1. Principles

1. **Integer where the hardware chooses the reduction order.** All GEMMs (projections, lm_head) and
   attention (QK^T, PV) accumulate in integers. Integer addition is associative, so tiling, split-K,
   atomics, MMA shapes and warp sizes are all free. Tensor cores are used only for integer MMA
   (s8/u8 → s32), which is exact.
2. **Controlled fp32 everywhere else.** Elementwise math uses only IEEE-754 binary32 add, sub, mul, div,
   sqrt, floor, rint, comparisons, int ↔ float conversions (round to nearest even) and bit casts. No FMA
   (compile with `-fmad=false` / `-ffp-contract=off`, no fast math), no library transcendentals
   (`exp_f`, `log_f`, `softplus_f`, `sigmoid_f`, `silu_f` are explicit op sequences derived from Cephes,
   `spec/prim.rs`), denormals preserved (no FTZ).
3. **One floating-point reduction primitive: `treesum`.** Balanced binary tree over the input padded with
   +0.0 to the next power of two: `s[i] = s[2i] + s[2i+1]` per level. Used for RMSNorm / l2norm sums
   of squares and the Gated DeltaNet contractions. (A warp butterfly with *ascending* xor offsets over
   aligned contiguous chunks reproduces it exactly.)
4. Max is order-free and may be computed any way.

## 2. Tensor formats

* Residual stream, norms, gates, GDN state: fp32.
* Quantized linear `y = W x`, W[N,K], group size 128 along K, bit width chosen per linear:
  * `wq[n,k]`: signed integer codes, |wq| ≤ 2^(bits−1) − 1 (W6: ±31, W8: ±127, W4: ±7)
  * `c[n]` fp32 per output row, `m[n,g]` uint16 ≥ 1 per (row, 128-group)
  * meaning: `W[n,k] ≈ c[n] · m[n,k/128] · wq[n,k]`
  * packing per 32 weights (`spec/quant.rs: unpack_row`, `quantize.rs: pack`): W8 = int8; W4 = 16 bytes
    of nibbles (u = wq+8, byte i = u[i] | u[i+16]<<4); W6 = 16 bytes low nibbles as W4 (u = wq+32) + 8
    bytes of high 2-bit pairs (byte i = hi[i] | hi[i+8]<<2 | hi[i+16]<<4 | hi[i+24]<<6).
  * a det row (§6) is `[codes (K/32 · bytes_per_32)] [c fp32 LE] [m u16 LE × K/128]`.
  * Conversion (`det convert`, `quantize.rs`): per 128-group the scale with the least squared error over
    a short search around `amax / qmax`, then `c = max_g s_g / 65535`, `m_g = rint(s_g / c)`,
    `wq = rint(w / (c · m_g))`. Bit widths are chosen per linear at conversion (`det convert --bits`);
    the default is W6 with W8 for the output head and the GDN a/b projections.
* Vectors (norm weights, biases, the GDN conv taps, `A`) are `F32`, `F16` or `BF16`, converted to fp32
  exactly. The embedding table is `BF16` (or `F16`/`F32`), looked up exactly; the output head is a
  separate quantized linear (a quantized copy of the embedding when the model ties them).
* Activation quantization for GEMM inputs (`quant_act_i8g`): per row (token) fp32 scale `sx` plus a
  per-(row, 128-group) uint16 multiplier `mx[g] ≥ 1`:
  `amax_g = max|x_g|; s_g = amax_g / 127; smax = max_g s_g; sx = smax / 65535 (sx = 1 if smax == 0);
   mx_g = clamp(rint(s_g / sx), 1, 65535); inv_g = 1 / (sx · mx_g); xq = clamp(rint(x · inv_g), -127, 127)`.
* `quant_act_i8` (per row only: `s = amax / 127`, `s = 1 if amax == 0`) is used inside attention for q
  and k per head.
* KV cache: `kq[pos][n_kv][hd]` int8 + `sk[pos][n_kv]` fp32 (per token, per kv head), `vq[pos][n_kv][hd]`
  int16 with a static per-(layer, kv head) scale `v_scale` (§7).
* Rotary tables `rope_cos[pos][i]`, `rope_sin[pos][i]` of `pos · theta^(-2i / rot)` for `i < rot / 2` are
  model data: `det convert` computes them in binary64 and rounds once.

## 3. GEMM semantics

```
acc[t,n] = sum_g ( m[n,g] · mx[t,g] ) · ( sum_{k in group g} wq[n,k] · xq[t,k] )      exact integer (fits int64)
y[t,n]   = ( float32(acc[t,n]) · c[n] ) · sx[t]                                        int64→fp32 RNE, then 2 muls in this order
```
Bounds: inner group sum ≤ 128·127·127 < 2^21; times m·mx < 2^32; times ≤ 136 groups < 2^61.
Kernels may compute the inner sum with dp4a / IMMA int32, then widen. Concatenating linears along N
(fused q/k/v, qkv+z, a+b, gate+up) is exact because every output row is independent.

## 4. Elementwise / norm definitions (see `spec/prim.rs` for exact op order)

* `rmsnorm(x, w, eps)`: `ss = treesum(x·x); mean = ss / n; inv = 1 / sqrt(mean + eps); y = (x · inv) · w`.
  The GGUF norm weights of zero-centered norms already contain the `+1`.
* `l2norm(x, eps)`: `inv = 1 / sqrt(treesum(x·x) + eps); y = x · inv`.
* `exp_f`: clamp x to ≤ 88; x < −87 → 0; Cephes polynomial, 2^n by exponent-bit construction.
* `log_f`: Cephes; `softplus_f(x) = x (x > 20) | exp_f(x) (x < −15) | log_f(1 + exp_f(x))`.
* `sigmoid_f(x) = exp_f(x) (x < −80) | 1 / (1 + exp_f(−x))`; `silu_f(x) = x · sigmoid_f(x)`.
* RoPE (neox half-split on the first `rot` of `hd` dims): `x[i] = (a·c) − (b·s); x[i + rot/2] = (b·c) + (a·s)`.

## 5. Layer semantics (Qwen3.5; per layer `xn = rmsnorm(x, attn_norm)`; residual adds in fp32)

The hyperparameters come from the GGUF metadata (`crates/det/src/config.rs`); layer `i` is full
attention when `(i + 1) mod full_attention_interval == 0`, Gated DeltaNet otherwise.

### 5.1 Full attention (`attn_q`, `attn_k`, `attn_v`, `attn_output`, `attn_q_norm`, `attn_k_norm`)
```
qg = q_lin(xq)  -> per head h: q_h = qg[h·2hd : h·2hd + hd], gate_h = qg[h·2hd + hd : (h+1)·2hd]
k = k_lin(xq), v = v_lin(xq)              (same xq, sx, mx as q)
q_h = rmsnorm(q_h, q_norm, eps); k_h = rmsnorm(k_h, k_norm, eps); rope both; q_h *= fp32(1/sqrt(hd)) (exact)
qq, sq = quant_act_i8(q_h)                                   per (token, head)
kq, sk = quant_act_i8(k_h)                                   per (token, kv head)     -> KV cache
vq      = clamp(rint(v · v_inv_scale[kvh]), -32767, 32767)  int16                     -> KV cache
s_j   = ( f32(dot_int(qq, kq_j)) · sq ) · sk_j               dot < 2^23 so the conversion is exact
mx    = max_j s_j
pq_j  = rint( exp_f(s_j − mx) · 2^24 )                       integer in [0, 2^24]
den   = Σ_j pq_j                 (int64)
num_i = Σ_j pq_j · vq_j[i]       (int64)
o_i   = ( f32(num_i) / f32(den) ) · v_scale[kvh]
o    *= sigmoid_f(gate_h);  out = o_lin(quant_act_i8g(o))
```
Head `h` reads kv head `h / (n_h / n_kv)`. Two passes over keys (max, then accumulate). No online
softmax rescaling: it is not tiling-invariant. Implementations: integer accumulation on CUDA cores
(decode), or int8 tensor cores (prefill): QK^T as s8×s8 MMA; PV by byte-slicing `pq` into u8 slices and
`vq` into (u8 low, s8 high) and summing the shifted s32 MMA results into int64 — exact.

### 5.2 Gated DeltaNet (`attn_qkv`, `attn_gate`, `ssm_alpha`, `ssm_beta`, `ssm_conv1d`, `ssm_a`, `ssm_dt`, `ssm_norm`, `ssm_out`)
```
qkv = qkv_lin(xq) [conv_dim = 2·n_k·dk + n_v·dv]; z = z_lin(xq) [n_v·dv]; a = alpha_lin(xq) [n_v]; b = beta_lin(xq) [n_v]
conv (per channel, state = previous 3 inputs):  y = ((w0·x[t−3] + w1·x[t−2]) + w2·x[t−1]) + w3·x[t];  y = silu_f(y)
q = l2norm(y[0 : n_k·dk] per head) · QSCALE (fp32(1/sqrt(dk)));  k = l2norm(y[n_k·dk : 2·n_k·dk] per head);  v = y[2·n_k·dk :]
beta  = sigmoid_f(b);  decay = exp_f( −( A · softplus_f(a + dt_bias) ) )        (A = −ssm_a, the file stores −exp(A_log))
per v head h, k head kh = h / (n_v / n_k), state S[dk][dv] fp32:
   S      = S · decay
   kv_mem = treesum_i( S[i,:] · k[i] )
   delta  = (v − kv_mem) · beta
   S      = S + k[i] · delta[j]
   o      = treesum_i( S[i,:] · q[i] )
on = rmsnorm(o, ssm_norm, eps) · silu_f(z);  out = out_lin(quant_act_i8g(on))
```
The conv taps are `ssm_conv1d` (`[conv_dim][4]`, oldest input first). `det convert` reorders the v
heads of the file it reads so that this mapping holds (llama.cpp's converter stores them so that v head
h reads k head h mod n_k). Strictly sequential over tokens (the chunked parallel form rounds differently
and is not allowed).

### 5.3 MLP (`ffn_gate`, `ffn_up`, `ffn_down`, pre-norm `post_attention_norm`)
`h = silu_f(gate_lin(xq)) · up_lin(xq); out = down_lin(quant_act_i8g(h))`.

### 5.4 Head
`logits = lm_head(quant_act_i8g(rmsnorm(x, output_norm)))`. Greedy = argmax, lowest index on ties.

### 5.5 Sampling (`spec/sample.rs`)
`l = logits · inv_temp; w_i = rint(exp_f(l_i − max) · 2^32)` as integers. Candidates are ordered by
(w descending, index ascending); optional top-k keeps the first k, optional top-p keeps the shortest
prefix whose integer sum reaches `ceil(p24 · Σw / 2^24)` with `p24 = rint(top_p · 2^24)`; a token is
kept when both keep it. `u = rng mod Σ kept w`; pick the smallest kept index whose inclusive prefix sum
of kept weights exceeds `u`. RNG = splitmix64, one draw per token.

## 6. Model file

A GGUF file written by `det convert` from a float (BF16/F16/F32) GGUF: the source metadata and vectors,
every linear as a det tensor (type ids `0xDE04`, `0xDE06`, `0xDE08` for W4/W6/W8; rows as in §2, `N`
rows of `K`), `output.weight` always present, `rope_cos` / `rope_sin` (`[rot/2][positions]`), and the
`det.v_scale` metadata of §7. Tensor names follow llama.cpp's converter (`blk.{i}.attn_norm.weight`,
`attn_qkv`, `attn_gate`, `ssm_alpha`, …). The file's hash identifies the model.

## 7. Value scale calibration
Calibration (run by `det convert`, or `det calibrate` on an existing file) runs mixed text and random
tokens through the model, records per-(attention layer, kv head) absmax of fp32 V, and writes
`v_scale = absmax · 8 / 32767` (8× headroom; values beyond it clip) into the file's `det.v_scale`
metadata, for which the converter reserves the room. `v_inv_scale = 1 / v_scale` in fp32. A file without the
key runs with `v_scale = 2^-10`.

## 8. Compiler / platform requirements
* Rust (spec and CPU backend): plain `f32` arithmetic; no `mul_add`, no `f32::exp`/`ln`/`powf`;
  `round_ties_even`; `as` for int ↔ float (round to nearest even); never a target with flush-to-zero.
* Metal (`kernels/metal`): shaders compiled with `MTLMathModeSafe` (no fast math, no FMA contraction;
  div and sqrt correctly rounded). Apple GPUs flush fp32 denormals, which the spec forbids:
  `shaders/common.metal` wraps every fp32 add/sub/mul/div/sqrt in a check that takes the hardware result
  only when it is provably the IEEE one and otherwise recomputes it in integer arithmetic; comparisons
  and max use an order-preserving integer key; int64 → fp32 is a software RNE conversion. The integer
  contractions run on the tensor unit (int8 × int8 → int32, exact).
* CUDA (`kernels/cuda`, pending the port to the GGUF formats): `-fmad=false`, no `--use_fast_math`,
  `-ftz=false`; `__fmul_rn/__fadd_rn/__fdiv_rn/__fsqrt_rn`, `__float2int_rn`, `__ll2float_rn`.

## 9. Open risks
* Static V scale clipping on pathological inputs (8× headroom over calibration).
* Softmax probability fixed point 2^-24 absolute resolution (worse than fp32 for p < 2^-14; negligible).
* Cross-generation determinism has been demonstrated on one Apple GPU against the CPU reference;
  other Apple chips and the CUDA port are the next validation.
