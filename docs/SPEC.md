# det: bit-exact deterministic inference for Qwen3.8-27B (Qwen3.5 architecture)

Goal: identical logits, bit for bit, on every GPU/vendor/generation, for every batching and prefill chunking,
with ~6-bit weights and competitive speed.  The numerics below are THE definition of the model; `detref/`
(numpy) is the executable golden reference.  Every kernel must match it exactly.

## 1. Principles

1. **Integer where the hardware chooses the reduction order.**  All GEMMs (projections, lm_head) and attention
   (QK^T, PV) accumulate in integers.  Integer addition is associative, so tiling, split-K, atomics, MMA shapes
   and warp sizes are all free.  Tensor cores are used only for integer MMA (s8/u8 -> s32), which is exact.
2. **Controlled fp32 everywhere else.**  Elementwise math uses only IEEE-754 binary32 add/sub/mul/div/sqrt/
   floor/rint, comparisons, int<->float conversions (round-to-nearest-even) and bit casts.  No FMA (compile
   with `-fmad=false` / `-ffp-contract=off` / `-fno-fast-math`), no library transcendentals (`exp_f`, `log_f`,
   `softplus_f`, `sigmoid_f`, `silu_f` are defined in `detref/detmath.py` as explicit op sequences, derived from
   Cephes), denormals preserved (no FTZ).
3. **One floating-point reduction primitive: `treesum`.**  Balanced binary tree over the input padded with +0.0
   to the next power of two: `s[i] = s[2i] + s[2i+1]` per level.  Used for RMSNorm / l2norm sums of squares and
   the Gated DeltaNet contractions.  (A warp butterfly with *ascending* xor offsets 1,2,4,... over aligned
   contiguous chunks reproduces it exactly.)
4. Max is order-free and may be computed any way.

## 2. Tensor formats

* Residual stream, norms, gates, GDN state: fp32.
* Embedding table: bf16 bits, converted to fp32 exactly on lookup (the lookup may run on the host).
* Quantized linear `y = W x`, W[N,K], group size 128 along K, bit width chosen per linear (manifest `bits`):
  * `wq[n,k]`: signed integer codes, |wq| <= 2^(bits-1)-1  (W6: +-31, W8: +-127, W4: +-7)
  * `c[n]` fp32 per output row, `m[n,g]` uint16 >= 1 per (row, 128-group)
  * meaning: `W[n,k] ~= c[n] * m[n,k/128] * wq[n,k]`
  * packing per 32 weights (`detref/quant.py: pack/unpack`): W8 = int8; W4 = 16 bytes of nibbles (u=wq+8,
    byte i = u[i] | u[i+16]<<4); W6 = 16 bytes low nibbles as W4 (u=wq+32) + 8 bytes of high 2-bit pairs
    (byte i = hi[i] | hi[i+8]<<2 | hi[i+16]<<4 | hi[i+24]<<6).
  * Conversion: round-to-nearest with a per-group MSE scale search (`quant.quantize_linear`), then `c = max_g s_g
    / 65535`, `m_g = rint(s_g / c)`.  Bit widths used for the 27B: W6 everywhere except lm_head and the GDN
    a/b projections (W8); optionally out_proj / down_proj at W8 (see README for the measured trade-offs).
* Activation quantization for GEMM inputs (`quant_act_i8g`): per row (token) fp32 scale `sx` plus a
  per-(row, 128-group) uint16 multiplier `mx[g] >= 1`, mirroring the weight scheme:
  `amax_g = max|x_g|; s_g = amax_g/127; smax = max_g s_g; sx = smax/65535 (sx = 1 if smax == 0);
   mx_g = clamp(rint(s_g / sx), 1, 65535); inv_g = 1/(sx*mx_g); xq = clamp(rint(x*inv_g), -127, 127)`.
  (Per-token-only int8 cost 1.2% perplexity on the 4B; per-group is lossless.)
* `quant_act_i8` (per row only: `s = amax/127`, s = 1 if amax == 0) is used inside attention for q and k per head.
* KV cache: `kq[pos][n_kv][hd]` int8 + `sk[pos][n_kv]` fp32 (per token, per kv head), `vq[pos][n_kv][hd]` int16
  with a STATIC per-(layer, kv head) scale `v_scale` (calibrated: `absmax * 8 / 32767`, see 7).

## 3. GEMM semantics

```
acc[t,n] = sum_g ( m[n,g] * mx[t,g] ) * ( sum_{k in group g} wq[n,k] * xq[t,k] )      exact integer (fits int64)
y[t,n]   = ( float32(acc[t,n]) * c[n] ) * sx[t]                                        int64->fp32 RNE, then 2 muls in this order
```
Bounds: inner group sum <= 128*127*127 < 2^21; times m*mx < 2^32; times <= 136 groups < 2^61.
Kernels may compute the inner sum with dp4a / IMMA int32, then widen.  Concatenating linears along N (fused
q/k/v, qkv+z, a+b, gate+up) is exact because every output row is independent.

## 4. Elementwise / norm definitions (see detmath.py for exact op order)

* `rmsnorm(x, w, eps)`: `ss = treesum(x*x); mean = ss / n; inv = 1 / sqrt(mean + eps); y = (x*inv) * w`.
  For Qwen3.5 zero-centered norms the stored `w` is already `1 + weight` (exact in fp32).
* `l2norm(x, eps)`: `inv = 1 / sqrt(treesum(x*x) + eps); y = x * inv`.
* `exp_f`: clamp x to <= 88; x < -87 -> 0; Cephes polynomial, 2^n by exponent-bit construction.
* `log_f`: Cephes; `softplus_f(x) = x (x>20) | exp_f(x) (x<-15) | log_f(1 + exp_f(x))`.
* `sigmoid_f(x) = exp_f(x) (x<-80) | 1 / (1 + exp_f(-x))`; `silu_f(x) = x * sigmoid_f(x)`.
* RoPE (neox half-split on the first 64 of 256 dims; table `rope_cos/sin[pos][32]` is model data computed in
  float64 once, `tools/extend_rope.py` regenerates it for more positions): `x[i] = (a*c) - (b*s); x[i+32] = (b*c) + (a*s)`.

## 5. Layer semantics (per layer; `xn = rmsnorm(x, attn_norm)`; residual adds in fp32)

### 5.1 Full attention (16 of 64 layers; 24 q heads, 4 kv heads, head_dim 256)
```
qg = q_lin(xq)  -> per head h: q_h = qg[h*512 : h*512+256], gate_h = qg[h*512+256 : h*512+512]
k = k_lin(xq), v = v_lin(xq)              (same xq, sx, mx as q)
q_h = rmsnorm(q_h, q_norm, eps); k_h = rmsnorm(k_h, k_norm, eps); rope both; q_h *= 0.0625 (exact)
qq, sq = quant_act_i8(q_h)                                   per (token, head)
kq, sk = quant_act_i8(k_h)                                   per (token, kv head)     -> KV cache
vq      = clamp(rint(v * v_inv_scale[kvh]), -32767, 32767)  int16                     -> KV cache
s_j   = ( float32(dot_int(qq, kq_j)) * sq ) * sk_j          dot < 2^23 so the conversion is exact
mx    = max_j s_j
pq_j  = rint( exp_f(s_j - mx) * 2^24 )                       integer in [0, 2^24]
den   = sum_j pq_j                 (int64)
num_i = sum_j pq_j * vq_j[i]       (int64, < 2^57 for 262k context)
o_i   = ( float32(num_i) / float32(den) ) * v_scale[kvh]
o    *= sigmoid_f(gate_h);  out = o_lin(quant_act_i8g(o))
```
Two passes over keys (max, then accumulate).  No online-softmax rescaling: it is not tiling-invariant.
Implementations: CUDA cores with int64 accumulation (decode), or int8 tensor cores (prefill): QK^T as s8xs8
MMA; PV by byte-slicing pq into u8 slices and vq into (u8 low, s8 high) and summing the shifted s32 MMA
results into int64 — exact.

### 5.2 Gated DeltaNet (48 of 64 layers; 16 k heads x 128, 48 v heads x 128, conv kernel 4)
```
qkv = qkv_lin(xq) [conv_dim = 2*2048 + 6144]; z = z_lin(xq) [6144]; a = a_lin(xq) [48]; b = b_lin(xq) [48]
conv (per channel, state = previous 3 inputs):  y = ((w0*x[t-3] + w1*x[t-2]) + w2*x[t-1]) + w3*x[t];  y = silu_f(y)
q = l2norm(y[0:2048] per head) * QSCALE (fp32(1/sqrt(128)) = 0x3DB504F3);  k = l2norm(y[2048:4096] per head);  v = y[4096:]
beta  = sigmoid_f(b);  decay = exp_f( -( A * softplus_f(a + dt_bias) ) )        (A = exp(A_log) is model data)
per v-head h (k-head kh = h / 3), state S[128 k][128 v] fp32:
   S      = S * decay
   kv_mem = treesum_i( S[i,:] * k[i] )
   delta  = (v - kv_mem) * beta
   S      = S + k[i] * delta[j]
   o      = treesum_i( S[i,:] * q[i] )
on = rmsnorm(o, gdn_norm, eps) * silu_f(z);  out = out_lin(quant_act_i8g(on))
```
Strictly sequential over tokens (the chunked parallel form rounds differently and is NOT allowed).

### 5.3 MLP
`h = silu_f(gate_lin(xq)) * up_lin(xq); out = down_lin(quant_act_i8g(h))`.

### 5.4 Head
`logits = lm_head(quant_act_i8g(rmsnorm(x, final_norm)))`.  Greedy = argmax, lowest index on ties.

### 5.5 Sampling (`detref/sample.py`, `cuda/sample.cu`)
`l = logits * inv_temp; w_i = rint(exp_f(l_i - max) * 2^32)` as integers; optional top-k (k largest w, ties to
the lower index) and top-p (smallest prefix in (w desc, index asc) order whose integer sum reaches
`ceil(top_p * total)`); `u = rng mod sum(kept w)`; pick the smallest kept index whose inclusive prefix sum
exceeds `u`.  RNG = splitmix64 on the host, one draw per token.

## 6. Model file format

A directory with `model.json` (architecture, per-linear `{N, K, bits}`, `files`) and safetensors part files
(`parts/L{i}.safetensors` per layer, `parts/head.safetensors`, optional `parts/rope_ext.safetensors`; or a single
`tensors.safetensors`).  Tensor names: `L{i}.{gate,up,down,q,k,v,o,qkv,z,a,b,out}.{wq,c,m}`, `L{i}.attn_norm`,
`L{i}.ffn_norm`, `L{i}.q_norm`, `L{i}.k_norm`, `L{i}.v_scale`, `L{i}.v_inv_scale`, `L{i}.conv`, `L{i}.A`,
`L{i}.dt_bias`, `L{i}.gdn_norm`, `embed_bf16`, `final_norm`, `lm_head.*`, `rope_cos`, `rope_sin`.
A `v_scales.npz` next to `model.json` overrides the V scales (used during calibration).

## 7. V scale calibration
`tools/calib_v_cuda.py` runs mixed text and random tokens through the model, records per-(layer, kv head)
absmax of fp32 V, and sets `v_scale = absmax * 8 / 32767` (8x headroom; values beyond it clip).

## 8. Compiler / platform requirements
* CUDA: `-fmad=false`, no `--use_fast_math`, default `-prec-div=true -prec-sqrt=true -ftz=false`;
  never call `__expf`, `expf`, `rsqrtf`, `__fdividef`, etc.  Use `__float2int_rn`, `__ll2float_rn`.
  The kernels use `__fmul_rn/__fadd_rn/__fdiv_rn/__fsqrt_rn` explicitly (never contracted).
* Metal (`metal/`): shaders are compiled at runtime with `MTLMathModeSafe` (no fast math, no FMA contraction; div
  and sqrt are correctly rounded; int<->float conversions are RNE).  Apple GPUs flush fp32 denormals (inputs and
  outputs, in every arithmetic op and in comparisons), which the spec forbids: `metal/shaders/common.metal` wraps
  every fp32 add/sub/mul/div/sqrt in a check that takes the hardware result only when it is provably the IEEE one
  (no denormal operand, result outside the zero / denormal / minimum-normal range) and otherwise recomputes it
  exactly in integer arithmetic; comparisons and max use an order-preserving integer key.  `tests/metal/test_denorm.py`
  and `test_prim.py` verify the primitives (including denormal operands and results) against numpy bit for bit.
  The integer GEMM / attention contractions run on the tensor unit (`mpp::tensor_ops::matmul2d`, int8 x int8 ->
  int32, exact) or as fp32 FMAs on integers < 2^24 (exact).
* CPU reference: numpy float32 (each ufunc is one correctly rounded op); integer GEMMs via fp32 BLAS on
  128-wide groups (all partial sums are integers < 2^24, hence exact).

## 9. Open risks
* ~~Denormal handling on Apple GPUs~~: resolved by the software denormal layer of the Metal backend (verified on
  the GDN decay runs of `tests/metal/test_gdn.py`, which drive the state through the denormal range).
* Static V scale clipping on pathological inputs (8x headroom over calibration).
* Softmax probability fixed point 2^-24 absolute resolution (worse than fp32 for p < 2^-14; negligible).
* Determinism has been demonstrated on one GPU (RTX 5090) against the CPU reference; cross-generation runs
  on other NVIDIA cards are still to be done (the library builds for sm_80/sm_89/sm_120).
