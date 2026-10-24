//! Weight quantization into det rows (SPEC §2): each 128-group gets the scale that minimizes its
//! squared error over a short search around `amax / qmax`, and the row's scales are stored as one fp32
//! scale `c` times a u16 multiplier per group.

use crate::spec::{bytes_per_32, det_row_bytes, GROUP};
use rayon::prelude::*;

/// Packs `codes` (a multiple of 32, each within the signed range of `bits`) per 32 weights.
pub fn pack(codes: &[i8], bits: u32, out: &mut Vec<u8>) {
    assert_eq!(codes.len() % 32, 0);
    for blk in codes.chunks(32) {
        match bits {
            8 => out.extend(blk.iter().map(|&q| q as u8)),
            4 => {
                let u: Vec<u8> = blk.iter().map(|&q| (q + 8) as u8).collect();
                out.extend((0..16).map(|i| u[i] | (u[i + 16] << 4)));
            }
            6 => {
                let u: Vec<u8> = blk.iter().map(|&q| (q + 32) as u8).collect();
                out.extend((0..16).map(|i| (u[i] & 0xF) | ((u[i + 16] & 0xF) << 4)));
                out.extend((0..8).map(|i| {
                    (u[i] >> 4)
                        | ((u[i + 8] >> 4) << 2)
                        | ((u[i + 16] >> 4) << 4)
                        | ((u[i + 24] >> 4) << 6)
                }));
            }
            _ => panic!("unsupported bit width {bits}"),
        }
    }
}

fn group_scale(w: &[f64], qmax: f64) -> f64 {
    let amax = w.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    if amax == 0.0 {
        return 1.0;
    }
    let base = amax / qmax;
    let mut best = (f64::INFINITY, base);
    for step in 0..=10 {
        let s = base * (0.80 + 0.02 * step as f64);
        let err: f64 = w
            .iter()
            .map(|&v| {
                let q = (v / s).round_ties_even().clamp(-qmax, qmax);
                let d = q * s - v;
                d * d
            })
            .sum();
        if err < best.0 {
            best = (err, s);
        }
    }
    best.1
}

/// One det row from `k` weights.
pub fn quantize_row(w: &[f32], bits: u32, out: &mut Vec<u8>) {
    let k = w.len();
    assert!(k.is_multiple_of(GROUP));
    let qmax = ((1i32 << (bits - 1)) - 1) as f64;
    let wd: Vec<f64> = w.iter().map(|&v| v as f64).collect();
    let scales: Vec<f64> = wd.chunks(GROUP).map(|g| group_scale(g, qmax)).collect();
    let smax = scales.iter().cloned().fold(0.0f64, f64::max);
    let c = (smax / 65535.0) as f32;
    let m: Vec<u16> = scales
        .iter()
        .map(|&s| (s / c as f64).round_ties_even().clamp(1.0, 65535.0) as u16)
        .collect();
    let mut codes = vec![0i8; k];
    for (g, &mg) in m.iter().enumerate() {
        let eff = c as f64 * mg as f64;
        for i in g * GROUP..(g + 1) * GROUP {
            codes[i] = (wd[i] / eff).round_ties_even().clamp(-qmax, qmax) as i8;
        }
    }
    let start = out.len();
    pack(&codes, bits, out);
    out.extend_from_slice(&c.to_le_bytes());
    for &v in &m {
        out.extend_from_slice(&v.to_le_bytes());
    }
    debug_assert_eq!(out.len() - start, det_row_bytes(bits, k));
    let _ = bytes_per_32(bits);
}

/// `n` det rows from a row-major `n × k` matrix.
pub fn quantize_rows(w: &[f32], n: usize, k: usize, bits: u32) -> Vec<u8> {
    assert_eq!(w.len(), n * k);
    let rb = det_row_bytes(bits, k);
    let mut out = vec![0u8; n * rb];
    out.par_chunks_mut(rb)
        .zip(w.par_chunks(k))
        .for_each(|(dst, row)| {
            let mut v = Vec::with_capacity(rb);
            quantize_row(row, bits, &mut v);
            dst.copy_from_slice(&v);
        });
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::GgmlType;
    use crate::spec::{dequant_row, unpack_row};

    #[test]
    fn pack_roundtrip_and_error() {
        let k = 512;
        let w: Vec<f32> = (0..k)
            .map(|i| (((i * 7919) % 1000) as f32 / 500.0 - 1.0) * 0.05 * (1.0 + (i / 128) as f32))
            .collect();
        for bits in [4, 6, 8] {
            let mut row = Vec::new();
            quantize_row(&w, bits, &mut row);
            let r = crate::spec::det_row(bits, k, &row);
            let mut q = vec![0i8; k];
            unpack_row(r.codes, bits, k, &mut q);
            let mut re = Vec::new();
            pack(&q, bits, &mut re);
            assert_eq!(re, r.codes, "pack/unpack roundtrip W{bits}");
            let mut back = vec![0f32; k];
            dequant_row(GgmlType::det(bits), &row, k, &mut back);
            let qmax = ((1 << (bits - 1)) - 1) as f32;
            for g in 0..k / 128 {
                let amax = w[g * 128..(g + 1) * 128]
                    .iter()
                    .fold(0f32, |m, v| m.max(v.abs()));
                for i in g * 128..(g + 1) * 128 {
                    assert!(
                        (back[i] - w[i]).abs() <= amax / qmax,
                        "W{bits} element {i}: {} vs {}",
                        back[i],
                        w[i]
                    );
                }
            }
        }
    }
}
