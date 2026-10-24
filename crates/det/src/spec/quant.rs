//! Activation quantization, weight code unpacking and the integer GEMM row (SPEC §2–3).

use crate::spec::prim::{absmax, rint};

pub const GROUP: usize = 128;

/// Per-row symmetric int8: `s = amax/127` (1 if amax == 0), `q = clamp(rint(x / s), -127, 127)`.
pub fn quant_act_i8(x: &[f32], q: &mut [i8]) -> f32 {
    let amax = absmax(x);
    let s = if amax == 0.0 { 1.0 } else { amax / 127.0 };
    let inv = 1.0 / s;
    for i in 0..x.len() {
        q[i] = clamp_i8(rint(x[i] * inv));
    }
    s
}

/// Per-row scale `sx` with per-128-group integer multipliers `mx[g] >= 1`:
/// effective scale of group g is `sx * mx[g]`; `q = clamp(rint(x / (sx*mx[g])), -127, 127)`.
/// Returns `sx`; `mx` must have `x.len() / 128` entries.
pub fn quant_act_i8g(x: &[f32], q: &mut [i8], mx: &mut [u16]) -> f32 {
    let g = x.len() / GROUP;
    assert_eq!(g * GROUP, x.len());
    assert_eq!(mx.len(), g);
    let s_g: Vec<f32> = (0..g).map(|i| absmax(&x[i * GROUP..(i + 1) * GROUP]) / 127.0).collect();
    let smax = s_g.iter().fold(0.0f32, |m, v| crate::spec::prim::fmax(m, *v));
    let sx = if smax == 0.0 { 1.0 } else { smax / 65535.0 };
    for i in 0..g {
        let m = rint(s_g[i] / sx);
        let m = if m < 1.0 { 1.0 } else if m > 65535.0 { 65535.0 } else { m };
        mx[i] = m as u16;
        let eff = sx * (mx[i] as f32);
        let inv = 1.0 / eff;
        for k in 0..GROUP {
            q[i * GROUP + k] = clamp_i8(rint(x[i * GROUP + k] * inv));
        }
    }
    sx
}

#[inline]
fn clamp_i8(v: f32) -> i8 {
    if v < -127.0 { -127 } else if v > 127.0 { 127 } else { v as i8 }
}

/// Unpack one row of packed weight codes (`bits` in {4, 6, 8}) into `K` signed codes.
pub fn unpack_row(packed: &[u8], bits: u32, k: usize, out: &mut [i8]) {
    assert_eq!(out.len(), k);
    assert_eq!(k % 32, 0);
    match bits {
        8 => {
            assert_eq!(packed.len(), k);
            for i in 0..k {
                out[i] = packed[i] as i8;
            }
        }
        4 => {
            assert_eq!(packed.len(), k / 2);
            for blk in 0..k / 32 {
                let b = &packed[blk * 16..blk * 16 + 16];
                for i in 0..16 {
                    out[blk * 32 + i] = (b[i] & 0xF) as i8 - 8;
                    out[blk * 32 + 16 + i] = (b[i] >> 4) as i8 - 8;
                }
            }
        }
        6 => {
            assert_eq!(packed.len(), k / 32 * 24);
            for blk in 0..k / 32 {
                let b = &packed[blk * 24..blk * 24 + 24];
                let (ql, qh) = (&b[..16], &b[16..]);
                for i in 0..32 {
                    let lo = if i < 16 { ql[i] & 0xF } else { ql[i - 16] >> 4 };
                    let hi = (qh[i % 8] >> (2 * (i / 8))) & 3;
                    out[blk * 32 + i] = (lo | (hi << 4)) as i8 - 32;
                }
            }
        }
        _ => panic!("unsupported bit width {bits}"),
    }
}

/// One output element of a quantized linear:
/// `acc = Σ_g (m[g] * mx[g]) * Σ_{k in g} wq[k] * xq[k]` (exact int64), `y = (f32(acc) * c) * sx`.
/// `mx == None` means all ones (plain per-row int8 input).
pub fn gemm_row(wq: &[i8], c: f32, m: &[u16], xq: &[i8], sx: f32, mx: Option<&[u16]>) -> f32 {
    let k = wq.len();
    assert_eq!(xq.len(), k);
    let g = k / GROUP;
    assert_eq!(m.len(), g);
    let mut acc: i64 = 0;
    for gi in 0..g {
        let mut part: i64 = 0;
        for i in gi * GROUP..(gi + 1) * GROUP {
            part += (wq[i] as i64) * (xq[i] as i64);
        }
        let mult = (m[gi] as i64) * mx.map_or(1, |v| v[gi] as i64);
        acc += mult * part;
    }
    ((acc as f32) * c) * sx
}
