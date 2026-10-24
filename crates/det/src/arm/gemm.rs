//! The quantized GEMM (SPEC §3). Weights are repacked at load into strips of 4 rows; a strip's codes
//! come in blocks of 16 k holding the codes of its 4 rows as `[row][4 k]` bytes (signed for W8,
//! offset-binary nibbles for W4 and W6) so that the four `sdot` operands of a block come out of one to
//! three loads and a few mask and shift ops. The strip's `m[g][row]` and `c[row]` follow its codes.
//!
//! A work item is 4 strips (16 rows) times a chunk of tokens. It unpacks the strips' codes of a K chunk
//! once into an L1 scratch, then runs four tokens at a time with one 4-lane s32 accumulator per
//! (token, strip): `sdot` against a broadcast of the token's 4 codes, or, with i8mm, `smmla` of a
//! 2-row × 8-k operand (the scratch zipped at unpack time) against a 2-token × 8-k operand, which
//! yields the same lanes after an unzip. At the end of every 128-group the lanes are multiplied by
//! `m · mx` and added into int64 accumulators, which the epilogue converts.

use super::simd::*;
use crate::backend::{Buf, GemmOut};
use crate::cpu::CpuLinear;
use crate::spec::{self, GROUP};
use rayon::prelude::*;
use std::arch::aarch64::*;

const ROWS: usize = 4;
const STRIPS: usize = 4;
const ITEM: usize = STRIPS * ROWS;
const BLOCK: usize = 16;
const KC: usize = 1024;
const TC: usize = 64;

pub struct ArmLinear {
    data: Buf<u8>,
    pub widths: Vec<usize>,
    pub n: usize,
    pub k: usize,
    pub bits: u32,
    strip_bytes: usize,
    m_off: usize,
    c_off: usize,
}

fn block_bytes(bits: u32) -> usize {
    2 * spec::bytes_per_32(bits)
}

/// The nibble offset: W4 and W6 store `u = wq + OFF`.
fn off(bits: u32) -> i8 {
    1 << (bits - 1)
}

impl ArmLinear {
    fn strip(&self, s: usize) -> *const u8 {
        unsafe { self.data.as_ptr().add(s * self.strip_bytes) }
    }
    fn m(&self, s: usize, g: usize) -> *const u16 {
        unsafe { self.strip(s).add(self.m_off + g * ROWS * 2) as *const u16 }
    }
    fn c(&self, n: usize) -> f32 {
        unsafe { *(self.strip(n / ROWS).add(self.c_off + (n % ROWS) * 4) as *const f32) }
    }
    pub fn bytes(&self) -> usize {
        self.data.len()
    }
}

pub fn repack(src: &CpuLinear) -> ArmLinear {
    let (k, bits) = (src.k, src.bits);
    let n: usize = src.widths.iter().sum();
    assert!(k.is_multiple_of(GROUP));
    let strips = n.div_ceil(ITEM) * STRIPS;
    let bb = block_bytes(bits);
    let m_off = k / BLOCK * bb;
    let c_off = m_off + k / GROUP * ROWS * 2;
    let strip_bytes = (c_off + ROWS * 4).next_multiple_of(64);
    let data = Buf::<u8>::new_host(strips * strip_bytes);
    let o = off(bits);
    data.host_mut()
        .par_chunks_mut(strip_bytes)
        .enumerate()
        .for_each(|(s, d)| {
            let mut u = vec![0u8; ROWS * k];
            let mut wq = vec![0i8; k];
            for r in 0..ROWS {
                let row = s * ROWS + r;
                let (c, m) = if row < n {
                    let dr = spec::det_row(bits, k, src.row(row));
                    spec::unpack_row(dr.codes, bits, k, &mut wq);
                    for i in 0..k {
                        u[r * k + i] = if bits == 8 {
                            wq[i] as u8
                        } else {
                            (wq[i] + o) as u8
                        };
                    }
                    (dr.c, dr.m)
                } else {
                    u[r * k..(r + 1) * k].fill(if bits == 8 { 0 } else { o as u8 });
                    (0.0, vec![1u16; k / GROUP])
                };
                d[c_off + 4 * r..c_off + 4 * r + 4].copy_from_slice(&c.to_le_bytes());
                for (g, mg) in m.iter().enumerate() {
                    let p = m_off + (g * ROWS + r) * 2;
                    d[p..p + 2].copy_from_slice(&mg.to_le_bytes());
                }
            }
            for b in 0..k / BLOCK {
                let blk = &mut d[b * bb..(b + 1) * bb];
                for r in 0..ROWS {
                    for kk in 0..4 {
                        let q = |i: usize| u[r * k + b * BLOCK + i * 4 + kk];
                        let i = r * 4 + kk;
                        match bits {
                            8 => {
                                for qi in 0..4 {
                                    blk[qi * 16 + i] = q(qi);
                                }
                            }
                            4 => {
                                blk[i] = q(0) | q(1) << 4;
                                blk[16 + i] = q(2) | q(3) << 4;
                            }
                            _ => {
                                blk[i] = (q(0) & 15) | (q(1) & 15) << 4;
                                blk[16 + i] = (q(2) & 15) | (q(3) & 15) << 4;
                                blk[32 + i] = (q(0) >> 4)
                                    | (q(1) >> 4) << 2
                                    | (q(2) >> 4) << 4
                                    | (q(3) >> 4) << 6;
                            }
                        }
                    }
                }
            }
        });
    ArmLinear {
        data,
        widths: src.widths.clone(),
        n,
        k,
        bits,
        strip_bytes,
        m_off,
        c_off,
    }
}

/// The 4 signed code operands (`[row][4 k]` bytes, k-quads 0..4) of one 16-k block.
#[inline(always)]
unsafe fn unpack_block<const BITS: u32>(p: *const u8) -> [int8x16_t; 4] {
    let ld = |i: usize| vld1q_u8(p.add(16 * i));
    let signed = |v: uint8x16_t| vsubq_s8(vreinterpretq_s8_u8(v), vdupq_n_s8(off(BITS)));
    match BITS {
        8 => [
            vreinterpretq_s8_u8(ld(0)),
            vreinterpretq_s8_u8(ld(1)),
            vreinterpretq_s8_u8(ld(2)),
            vreinterpretq_s8_u8(ld(3)),
        ],
        4 => {
            let nib = vdupq_n_u8(0x0F);
            let (l0, l1) = (ld(0), ld(1));
            [
                signed(vandq_u8(l0, nib)),
                signed(vshrq_n_u8::<4>(l0)),
                signed(vandq_u8(l1, nib)),
                signed(vshrq_n_u8::<4>(l1)),
            ]
        }
        _ => {
            let nib = vdupq_n_u8(0x0F);
            let two = vdupq_n_u8(0x30);
            let (l0, l1, h) = (ld(0), ld(1), ld(2));
            [
                signed(vorrq_u8(
                    vandq_u8(l0, nib),
                    vandq_u8(vshlq_n_u8::<4>(h), two),
                )),
                signed(vorrq_u8(
                    vshrq_n_u8::<4>(l0),
                    vandq_u8(vshlq_n_u8::<2>(h), two),
                )),
                signed(vorrq_u8(vandq_u8(l1, nib), vandq_u8(h, two))),
                signed(vorrq_u8(
                    vshrq_n_u8::<4>(l1),
                    vandq_u8(vshrq_n_u8::<2>(h), two),
                )),
            ]
        }
    }
}

/// `p · mm` for 4 lanes of s32 `p` (|p| < 2^22) and u32 `mm`, as int64 (lanes 0, 1 and 2, 3): a signed
/// 32×32 multiply, plus `p << 32` where `mm` has its top bit set.
#[inline(always)]
unsafe fn mul_s32_u32(p: int32x4_t, mm: uint32x4_t) -> (int64x2_t, int64x2_t) {
    let mms = vreinterpretq_s32_u32(mm);
    let top = vshrq_n_s32::<31>(mms);
    let fix = |p: int32x2_t, t: int32x2_t| vandq_s64(vshlq_n_s64::<32>(vmovl_s32(p)), vmovl_s32(t));
    let (pl, ph) = (vget_low_s32(p), vget_high_s32(p));
    (
        vaddq_s64(vmull_s32(pl, vget_low_s32(mms)), fix(pl, vget_low_s32(top))),
        vaddq_s64(
            vmull_s32(ph, vget_high_s32(mms)),
            fix(ph, vget_high_s32(top)),
        ),
    )
}

struct Item<'a> {
    lin: &'a ArmLinear,
    quad: usize,
    xq: &'a [i8],
    mx: &'a [u16],
    t0: usize,
    t1: usize,
}

/// `T` tokens from `t` over the unpacked chunk: the group dot products and the int64 accumulation.
/// `acc[token][strip · 4 + row]`. With i8mm `T` is even and a token past `t1` repeats the last one.
#[inline(always)]
unsafe fn tile<const I8MM: bool, const BITS: u32, const T: usize>(
    it: &Item,
    scratch: *const i8,
    kc0: usize,
    groups: usize,
    t: usize,
    acc: &mut [[i64; ITEM]],
) {
    let (k, g_all) = (it.lin.k, it.lin.k / GROUP);
    let xrow = |tt: usize| it.xq.as_ptr().add((t + tt).min(it.t1 - 1) * k + kc0);
    for gi in 0..groups {
        let g = kc0 / GROUP + gi;
        let mut a = [[vdupq_n_s32(0); STRIPS]; T];
        if I8MM {
            for k8 in 0..GROUP / 8 {
                let w = scratch.add((gi * (GROUP / 8) + k8) * STRIPS * 32);
                let mut tt = 0;
                while tt < T {
                    let x = vcombine_s8(
                        vld1_s8(xrow(tt).add(gi * GROUP + k8 * 8)),
                        vld1_s8(xrow(tt + 1).add(gi * GROUP + k8 * 8)),
                    );
                    for s in 0..STRIPS {
                        a[tt][s] = smmla(a[tt][s], vld1q_s8(w.add(s * 32)), x);
                        a[tt + 1][s] = smmla(a[tt + 1][s], vld1q_s8(w.add(s * 32 + 16)), x);
                    }
                    tt += 2;
                }
            }
            let mut tt = 0;
            while tt < T {
                for s in 0..STRIPS {
                    let (r01, r23) = (a[tt][s], a[tt + 1][s]);
                    a[tt][s] = vuzp1q_s32(r01, r23);
                    a[tt + 1][s] = vuzp2q_s32(r01, r23);
                }
                tt += 2;
            }
        } else {
            for kq in 0..GROUP / 4 {
                let w = scratch.add((gi * (GROUP / 4) + kq) * STRIPS * 16);
                let ws: [int8x16_t; STRIPS] = std::array::from_fn(|s| vld1q_s8(w.add(16 * s)));
                for tt in 0..T {
                    let x = vreinterpretq_s8_s32(vld1q_dup_s32(
                        xrow(tt).add(gi * GROUP + kq * 4) as *const i32
                    ));
                    for s in 0..STRIPS {
                        a[tt][s] = sdot(a[tt][s], ws[s], x);
                    }
                }
            }
        }
        let m: [uint32x4_t; STRIPS] =
            std::array::from_fn(|s| vmovl_u16(vld1_u16(it.lin.m(STRIPS * it.quad + s, g))));
        for tt in 0..T {
            if t + tt >= it.t1 {
                break;
            }
            let mxv = vdupq_n_u32(it.mx[(t + tt) * g_all + g] as u32);
            for s in 0..STRIPS {
                let (lo, hi) = mul_s32_u32(a[tt][s], vmulq_u32(m[s], mxv));
                let dst = acc[t + tt - it.t0].as_mut_ptr().add(s * ROWS);
                vst1q_s64(dst, vaddq_s64(vld1q_s64(dst), lo));
                vst1q_s64(dst.add(2), vaddq_s64(vld1q_s64(dst.add(2)), hi));
            }
        }
    }
}

#[repr(align(16))]
#[derive(Clone, Copy)]
struct Line(#[allow(dead_code)] [u8; 16]);

#[inline(always)]
unsafe fn item<const I8MM: bool, const BITS: u32>(it: &Item, acc: &mut [[i64; ITEM]]) {
    let k = it.lin.k;
    let bb = block_bytes(BITS);
    let mut scratch = vec![Line([0; 16]); KC / 4 * STRIPS];
    let sp = scratch.as_mut_ptr() as *mut i8;
    let mut kc0 = 0;
    while kc0 < k {
        let kc = KC.min(k - kc0);
        for s in 0..STRIPS {
            let codes = it.lin.strip(STRIPS * it.quad + s).add(kc0 / BLOCK * bb);
            for b in 0..kc / BLOCK {
                let q = unpack_block::<BITS>(codes.add(b * bb));
                if I8MM {
                    for j in 0..2 {
                        let (lo, hi) = (
                            vreinterpretq_s32_s8(q[2 * j]),
                            vreinterpretq_s32_s8(q[2 * j + 1]),
                        );
                        let dst = sp.add(((b * 2 + j) * STRIPS + s) * 32);
                        vst1q_s8(dst, vreinterpretq_s8_s32(vzip1q_s32(lo, hi)));
                        vst1q_s8(dst.add(16), vreinterpretq_s8_s32(vzip2q_s32(lo, hi)));
                    }
                } else {
                    for (qi, v) in q.iter().enumerate() {
                        vst1q_s8(sp.add(((b * 4 + qi) * STRIPS + s) * 16), *v);
                    }
                }
            }
        }
        let groups = kc / GROUP;
        let mut t = it.t0;
        while t + 4 <= it.t1 {
            tile::<I8MM, BITS, 4>(it, sp, kc0, groups, t, acc);
            t += 4;
        }
        while t < it.t1 {
            if I8MM {
                tile::<true, BITS, 2>(it, sp, kc0, groups, t, acc);
                t += 2;
            } else {
                tile::<false, BITS, 1>(it, sp, kc0, groups, t, acc);
                t += 1;
            }
        }
        kc0 += kc;
    }
}

#[target_feature(enable = "dotprod,i8mm")]
unsafe fn item_i8mm<const BITS: u32>(it: &Item, acc: &mut [[i64; ITEM]]) {
    item::<true, BITS>(it, acc)
}

#[target_feature(enable = "dotprod")]
unsafe fn item_dotprod<const BITS: u32>(it: &Item, acc: &mut [[i64; ITEM]]) {
    item::<false, BITS>(it, acc)
}

pub fn gemm(
    lin: &ArmLinear,
    i8mm: bool,
    xq: &Buf<i8>,
    sx: &Buf<f32>,
    mx: &Buf<u16>,
    m: usize,
    out: &GemmOut,
) {
    let (k, n) = (lin.k, lin.n);
    let g = k / GROUP;
    let xq = &xq.host()[..m * k];
    let (sx, mx) = (&sx.host()[..m], &mx.host()[..m * g]);
    let mut offs = vec![0usize];
    for w in &lin.widths {
        offs.push(offs.last().unwrap() + w);
    }
    let quads = n.div_ceil(ITEM);
    let tchunks = m.div_ceil(TC);
    (0..quads * tchunks).into_par_iter().for_each(|i| {
        let (quad, tc) = (i % quads, i / quads);
        let (t0, t1) = (tc * TC, ((tc + 1) * TC).min(m));
        let it = Item {
            lin,
            quad,
            xq,
            mx,
            t0,
            t1,
        };
        let mut acc = vec![[0i64; ITEM]; t1 - t0];
        unsafe {
            match (i8mm, lin.bits) {
                (true, 8) => item_i8mm::<8>(&it, &mut acc),
                (true, 6) => item_i8mm::<6>(&it, &mut acc),
                (true, _) => item_i8mm::<4>(&it, &mut acc),
                (false, 8) => item_dotprod::<8>(&it, &mut acc),
                (false, 6) => item_dotprod::<6>(&it, &mut acc),
                (false, _) => item_dotprod::<4>(&it, &mut acc),
            }
        }
        for r in 0..ITEM {
            let row = quad * ITEM + r;
            if row >= n {
                break;
            }
            let c = lin.c(row);
            let (dst, stride, col) = match out {
                GemmOut::Write(d) | GemmOut::Accumulate(d) => (d.as_mut_ptr(), n, row),
                GemmOut::Parts(ds) => {
                    let p = offs.iter().rposition(|o| *o <= row).unwrap();
                    (ds[p].as_mut_ptr(), lin.widths[p], row - offs[p])
                }
            };
            for t in t0..t1 {
                let y = ((acc[t - t0][r] as f32) * c) * sx[t];
                let d = unsafe { &mut *dst.add(t * stride + col) };
                *d = if matches!(out, GemmOut::Accumulate(_)) {
                    *d + y
                } else {
                    y
                };
            }
        }
    });
}
