//! The token mixers for one head (SPEC §4–5): rope, full attention, Gated DeltaNet.

use crate::spec::prim::{exp_f, fmax, rint, sigmoid_f, treesum};

/// Neox half-split rotation of the first `2 * cos.len()` dims of `x` in place.
pub fn rope(x: &mut [f32], cos: &[f32], sin: &[f32]) {
    let half = cos.len();
    for i in 0..half {
        let (a, b) = (x[i], x[i + half]);
        x[i] = (a * cos[i]) - (b * sin[i]);
        x[i + half] = (b * cos[i]) + (a * sin[i]);
    }
}

/// Attention probability fixed point: `pq = rint(exp(s - mx) * 2^24)`.
pub const P_FIX: f32 = 16777216.0;

/// One quantized key/value row of the KV cache for one kv head.
pub struct KvRow<'a> {
    pub kq: &'a [i8],
    pub sk: f32,
    pub vq: &'a [i16],
}

/// Score of one key: `(f32(dot_i8(qq, kq)) * sq) * sk`.
pub fn attn_score(qq: &[i8], sq: f32, key: &KvRow) -> f32 {
    let dot: i32 = qq.iter().zip(key.kq).map(|(a, b)| (*a as i32) * (*b as i32)).sum();
    ((dot as f32) * sq) * key.sk
}

/// Exact integer partial sums of one key chunk given the global max `mx`:
/// `den = Σ pq_j`, `num[i] = Σ pq_j * vq_j[i]`.
pub fn attn_chunk(qq: &[i8], sq: f32, keys: &[KvRow], mx: f32, num: &mut [i64]) -> i64 {
    let mut den: i64 = 0;
    for key in keys {
        let s = attn_score(qq, sq, key);
        let pq = rint(exp_f(s - mx) * P_FIX) as i64;
        den += pq;
        for i in 0..num.len() {
            num[i] += pq * (key.vq[i] as i64);
        }
    }
    den
}

/// `o[i] = (f32(num[i]) / f32(den)) * v_scale`
pub fn attn_output(num: &[i64], den: i64, v_scale: f32, o: &mut [f32]) {
    for i in 0..o.len() {
        o[i] = ((num[i] as f32) / (den as f32)) * v_scale;
    }
}

/// Full attention of one quantized query against all `keys`, before the output gate.
pub fn attn_head(qq: &[i8], sq: f32, keys: &[KvRow], v_scale: f32, o: &mut [f32]) {
    let mx = keys.iter().fold(f32::NEG_INFINITY, |m, k| fmax(m, attn_score(qq, sq, k)));
    let mut num = vec![0i64; o.len()];
    let den = attn_chunk(qq, sq, keys, mx, &mut num);
    attn_output(&num, den, v_scale, o);
}

/// `o *= sigmoid(gate)` elementwise.
pub fn attn_gate(o: &mut [f32], gate: &[f32]) {
    for i in 0..o.len() {
        o[i] = o[i] * sigmoid_f(gate[i]);
    }
}

/// Quantize a value row to the KV cache's int16 format: `clamp(rint(v * v_inv_scale), -32767, 32767)`.
pub fn quant_v(v: &[f32], v_inv_scale: f32, vq: &mut [i16]) {
    for i in 0..v.len() {
        let r = rint(v[i] * v_inv_scale);
        vq[i] = if r < -32767.0 { -32767 } else if r > 32767.0 { 32767 } else { r as i16 };
    }
}

/// Gated DeltaNet state update of one column `S[:, j]` (length dk) for one token:
/// `S = S*decay; kv = treesum_i(S[i]*k[i]); delta = (v_j - kv)*beta; S[i] += k[i]*delta`.
pub fn gdn_column_step(col: &mut [f32], k: &[f32], v_j: f32, beta: f32, decay: f32) {
    for i in 0..col.len() {
        col[i] = col[i] * decay;
    }
    let prod: Vec<f32> = col.iter().zip(k).map(|(s, kk)| s * kk).collect();
    let kv_mem = treesum(&prod);
    let delta = (v_j - kv_mem) * beta;
    for i in 0..col.len() {
        col[i] = col[i] + (k[i] * delta);
    }
}

/// Mixer output element `o_j = treesum_i(S[i, j] * q[i])`.
pub fn gdn_column_out(col: &[f32], q: &[f32]) -> f32 {
    let prod: Vec<f32> = col.iter().zip(q).map(|(s, qq)| s * qq).collect();
    treesum(&prod)
}

/// Full-state convenience over all columns: `s` is `[dk][dv]` row-major, `o` has length dv.
pub fn gdn_head_step(s: &mut [f32], dk: usize, dv: usize, k: &[f32], q: &[f32], v: &[f32], beta: f32, decay: f32, o: &mut [f32]) {
    let mut col = vec![0f32; dk];
    for j in 0..dv {
        for i in 0..dk {
            col[i] = s[i * dv + j];
        }
        gdn_column_step(&mut col, k, v[j], beta, decay);
        for i in 0..dk {
            s[i * dv + j] = col[i];
        }
        o[j] = gdn_column_out(&col, q);
    }
}

/// Causal conv (kernel 4) + silu over one channel: `win` is the 4 inputs oldest first, `w` the 4 taps.
pub fn gdn_conv(win: &[f32; 4], w: &[f32; 4]) -> f32 {
    let mut y = (w[0] * win[0]) + (w[1] * win[1]);
    y = y + (w[2] * win[2]);
    y = y + (w[3] * win[3]);
    crate::spec::prim::silu_f(y)
}

/// `decay = exp(-(A * softplus(a + dt_bias)))`
pub fn gdn_decay(a: f32, dt_bias: f32, a_coef: f32) -> f32 {
    exp_f(-(a_coef * crate::spec::prim::softplus_f(a + dt_bias)))
}

/// `QSCALE = fp32(1/sqrt(128))`, applied to the l2-normed q of a GDN layer.
pub const QSCALE_GDN: f32 = f32::from_bits(0x3DB504F3);
