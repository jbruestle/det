//! Tensor storage (SPEC §2, §6): the det weight row and the exact dequantization of vectors and
//! embedding rows.

use crate::gguf::GgmlType;
use crate::spec::quant::{unpack_row, GROUP};

/// IEEE binary16 bits to fp32 (exact).
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h as u32) & 0x8000) << 16;
    let exp = ((h >> 10) & 0x1F) as u32;
    let mant = (h & 0x3FF) as u32;
    let bits = if exp == 0 {
        if mant == 0 {
            sign
        } else {
            let mut m = mant;
            let mut e: i32 = 127 - 15 + 1;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            sign | ((e as u32) << 23) | ((m & 0x3FF) << 13)
        }
    } else if exp == 31 {
        sign | 0x7F80_0000 | (mant << 13)
    } else {
        sign | ((exp + 127 - 15) << 23) | (mant << 13)
    };
    f32::from_bits(bits)
}

/// Packed bytes per 32 weight codes.
pub fn bytes_per_32(bits: u32) -> usize {
    match bits {
        8 => 32,
        6 => 24,
        4 => 16,
        _ => panic!("unsupported bit width {bits}"),
    }
}

/// A det weight row of `k` codes: `[codes (k/32 · bytes_per_32)] [c fp32 LE] [m u16 LE × k/128]`.
pub struct DetRow<'a> {
    pub codes: &'a [u8],
    pub c: f32,
    pub m: Vec<u16>,
}

pub fn det_row_bytes(bits: u32, k: usize) -> usize {
    assert!(
        k.is_multiple_of(GROUP),
        "K = {k} is not a multiple of {GROUP}"
    );
    k / 32 * bytes_per_32(bits) + 4 + k / GROUP * 2
}

pub fn det_row(bits: u32, k: usize, row: &[u8]) -> DetRow<'_> {
    let nc = k / 32 * bytes_per_32(bits);
    assert_eq!(row.len(), det_row_bytes(bits, k));
    let c = f32::from_le_bytes(row[nc..nc + 4].try_into().unwrap());
    let m = row[nc + 4..]
        .chunks(2)
        .map(|b| u16::from_le_bytes([b[0], b[1]]))
        .collect();
    DetRow {
        codes: &row[..nc],
        c,
        m,
    }
}

/// Dequantizes one row of `k` elements: float types exactly; det rows as `(c · f32(m[g])) · f32(q)`.
pub fn dequant_row(ty: GgmlType, row: &[u8], k: usize, out: &mut [f32]) {
    assert_eq!(out.len(), k);
    match ty {
        GgmlType::F32 => {
            for i in 0..k {
                out[i] = f32::from_le_bytes(row[4 * i..4 * i + 4].try_into().unwrap());
            }
        }
        GgmlType::F16 => {
            for i in 0..k {
                out[i] = f16_to_f32(u16::from_le_bytes([row[2 * i], row[2 * i + 1]]));
            }
        }
        GgmlType::BF16 => {
            for i in 0..k {
                out[i] = crate::spec::prim::bf16_to_f32(u16::from_le_bytes([
                    row[2 * i],
                    row[2 * i + 1],
                ]));
            }
        }
        GgmlType::Det4 | GgmlType::Det6 | GgmlType::Det8 => {
            let bits = ty.bits().unwrap();
            let r = det_row(bits, k, row);
            let mut q = vec![0i8; k];
            unpack_row(r.codes, bits, k, &mut q);
            for i in 0..k {
                out[i] = (r.c * (r.m[i / GROUP] as f32)) * (q[i] as f32);
            }
        }
    }
}
