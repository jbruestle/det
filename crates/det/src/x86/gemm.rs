//! The quantized GEMM (SPEC §3). Weights are repacked at load into strips of 8 rows; a strip's codes
//! come in blocks of 16 k holding the unsigned codes `u = wq + OFF` of its 8 rows as `[row][4 k]`
//! bytes, nibble-packed for W4 and W6 so that the four `vpdpbusd` operands of a block come out of two
//! or three loads and a few mask and shift ops. The strip's `m[g][row]` and `c[row]` follow its codes.
//!
//! A work item is a pair of strips (16 rows) times a chunk of tokens. It unpacks the pair's codes of
//! a K chunk once into an L1 scratch, then runs four tokens at a time with one 8-lane s32 accumulator
//! per (token, strip). At the end of every 128-group the lanes are corrected by `OFF · Σxq`,
//! multiplied by `m · mx` and added into int64 accumulators, which the epilogue converts.

use super::simd::*;
use crate::backend::{Buf, GemmOut};
use crate::cpu::CpuLinear;
use crate::spec::{self, GROUP};
use rayon::prelude::*;
use std::arch::x86_64::*;

const ROWS: usize = 8;
const PAIR: usize = 2 * ROWS;
const BLOCK: usize = 16;
const KC: usize = 1024;
const TC: usize = 64;

pub struct X86Linear {
    data: Aligned,
    pub widths: Vec<usize>,
    pub n: usize,
    pub k: usize,
    pub bits: u32,
    strip_bytes: usize,
    m_off: usize,
    c_off: usize,
}

fn block_bytes(bits: u32) -> usize {
    4 * spec::bytes_per_32(bits)
}

/// The code offset: `u = wq + OFF`.
fn off(bits: u32) -> i32 {
    1 << (bits - 1)
}

impl X86Linear {
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
        self.data.as_slice().len()
    }
}

pub fn repack(src: &CpuLinear) -> X86Linear {
    let (k, bits) = (src.k, src.bits);
    let n: usize = src.widths.iter().sum();
    assert!(k.is_multiple_of(GROUP));
    let strips = n.div_ceil(PAIR) * 2;
    let bb = block_bytes(bits);
    let m_off = k / BLOCK * bb;
    let c_off = m_off + k / GROUP * ROWS * 2;
    let strip_bytes = (c_off + ROWS * 4).next_multiple_of(64);
    let data = Aligned::zeroed(strips * strip_bytes);
    let o = off(bits);
    data.as_mut_slice()
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
                        u[r * k + i] = (wq[i] as i32 + o) as u8;
                    }
                    (dr.c, dr.m)
                } else {
                    u[r * k..(r + 1) * k].fill(o as u8);
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
                                    blk[qi * 32 + i] = q(qi);
                                }
                            }
                            4 => {
                                blk[i] = q(0) | q(1) << 4;
                                blk[32 + i] = q(2) | q(3) << 4;
                            }
                            _ => {
                                blk[i] = (q(0) & 15) | (q(1) & 15) << 4;
                                blk[32 + i] = (q(2) & 15) | (q(3) & 15) << 4;
                                blk[64 + i] = (q(0) >> 4)
                                    | (q(1) >> 4) << 2
                                    | (q(2) >> 4) << 4
                                    | (q(3) >> 4) << 6;
                            }
                        }
                    }
                }
            }
        });
    X86Linear {
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

#[repr(align(32))]
#[derive(Clone, Copy)]
struct Line(#[allow(dead_code)] [u8; 32]);

/// The 4 code operands (`[row][4 k]` bytes, k-quads 0..4) of one 16-k block.
#[inline(always)]
unsafe fn unpack_block<const BITS: u32>(p: *const u8) -> [__m256i; 4] {
    let ld = |i: usize| _mm256_load_si256(p.add(32 * i) as *const __m256i);
    match BITS {
        8 => [ld(0), ld(1), ld(2), ld(3)],
        4 => {
            let nib = _mm256_set1_epi8(0x0F);
            let (l0, l1) = (ld(0), ld(1));
            [
                _mm256_and_si256(l0, nib),
                _mm256_and_si256(_mm256_srli_epi16::<4>(l0), nib),
                _mm256_and_si256(l1, nib),
                _mm256_and_si256(_mm256_srli_epi16::<4>(l1), nib),
            ]
        }
        _ => {
            let nib = _mm256_set1_epi8(0x0F);
            let two = _mm256_set1_epi8(0x30);
            let (l0, l1, h) = (ld(0), ld(1), ld(2));
            [
                _mm256_or_si256(
                    _mm256_and_si256(l0, nib),
                    _mm256_and_si256(_mm256_slli_epi16::<4>(h), two),
                ),
                _mm256_or_si256(
                    _mm256_and_si256(_mm256_srli_epi16::<4>(l0), nib),
                    _mm256_and_si256(_mm256_slli_epi16::<2>(h), two),
                ),
                _mm256_or_si256(_mm256_and_si256(l1, nib), _mm256_and_si256(h, two)),
                _mm256_or_si256(
                    _mm256_and_si256(_mm256_srli_epi16::<4>(l1), nib),
                    _mm256_and_si256(_mm256_srli_epi16::<2>(h), two),
                ),
            ]
        }
    }
}

/// `p · mm` for 8 lanes of s32 `p` (|p| < 2^22) and u32 `mm`, as int64: the even lanes (rows 0, 2, 4,
/// 6) and the odd lanes (rows 1, 3, 5, 7). An unsigned 32×32 multiply, minus `mm << 32` where `p < 0`.
#[inline(always)]
unsafe fn mul_s32_u32(p: __m256i, mm: __m256i) -> (__m256i, __m256i) {
    let neg = _mm256_srai_epi32::<31>(p);
    let (po, mo, no) = (
        _mm256_srli_epi64::<32>(p),
        _mm256_srli_epi64::<32>(mm),
        _mm256_srli_epi64::<32>(neg),
    );
    let fe = _mm256_slli_epi64::<32>(_mm256_and_si256(mm, neg));
    let fo = _mm256_slli_epi64::<32>(_mm256_and_si256(mo, no));
    (
        _mm256_sub_epi64(_mm256_mul_epu32(p, mm), fe),
        _mm256_sub_epi64(_mm256_mul_epu32(po, mo), fo),
    )
}

struct Item<'a> {
    lin: &'a X86Linear,
    pair: usize,
    xq: &'a [i8],
    xsum: &'a [i32],
    mx: &'a [u16],
    t0: usize,
    t1: usize,
}

/// `T` tokens from `t` over the unpacked chunk: the group dot products and the int64 accumulation.
/// `acc[token][strip · 8 + lane]` holds the even rows in lanes 0..4 and the odd rows in 4..8.
#[inline(always)]
unsafe fn tile<const VNNI: bool, const BITS: u32, const T: usize>(
    it: &Item,
    scratch: *const u8,
    kc0: usize,
    groups: usize,
    t: usize,
    acc: &mut [[i64; PAIR]],
) {
    let (k, g_all) = (it.lin.k, it.lin.k / GROUP);
    let o = off(BITS);
    let xrow = |tt: usize| it.xq.as_ptr().add((t + tt) * k + kc0);
    for gi in 0..groups {
        let g = kc0 / GROUP + gi;
        let mut a = [[_mm256_setzero_si256(); 2]; T];
        for kq in 0..GROUP / 4 {
            let w = scratch.add(((gi * (GROUP / 4) + kq) * 2) * 32);
            let w0 = _mm256_load_si256(w as *const __m256i);
            let w1 = _mm256_load_si256(w.add(32) as *const __m256i);
            for tt in 0..T {
                let x = _mm256_set1_epi32(*(xrow(tt).add(gi * GROUP + kq * 4) as *const i32));
                a[tt][0] = dp::<VNNI, BITS>(a[tt][0], w0, x);
                a[tt][1] = dp::<VNNI, BITS>(a[tt][1], w1, x);
            }
        }
        let m = [
            _mm256_cvtepu16_epi32(_mm_loadu_si128(it.lin.m(2 * it.pair, g) as *const __m128i)),
            _mm256_cvtepu16_epi32(_mm_loadu_si128(
                it.lin.m(2 * it.pair + 1, g) as *const __m128i
            )),
        ];
        for tt in 0..T {
            let xs = _mm256_set1_epi32(o * it.xsum[(t + tt) * g_all + g]);
            let mxv = _mm256_set1_epi32(it.mx[(t + tt) * g_all + g] as i32);
            for s in 0..2 {
                let p = _mm256_sub_epi32(a[tt][s], xs);
                let (even, odd) = mul_s32_u32(p, _mm256_mullo_epi32(m[s], mxv));
                let dst = acc[t + tt - it.t0].as_mut_ptr().add(s * ROWS) as *mut __m256i;
                _mm256_storeu_si256(dst, _mm256_add_epi64(_mm256_loadu_si256(dst), even));
                _mm256_storeu_si256(
                    dst.add(1),
                    _mm256_add_epi64(_mm256_loadu_si256(dst.add(1)), odd),
                );
            }
        }
    }
}

#[inline(always)]
unsafe fn item<const VNNI: bool, const BITS: u32>(it: &Item, acc: &mut [[i64; PAIR]]) {
    let k = it.lin.k;
    let bb = block_bytes(BITS);
    let mut scratch = vec![Line([0; 32]); KC / 4 * 2];
    let sp = scratch.as_mut_ptr() as *mut u8;
    let mut kc0 = 0;
    while kc0 < k {
        let kc = KC.min(k - kc0);
        for s in 0..2 {
            let codes = it.lin.strip(2 * it.pair + s).add(kc0 / BLOCK * bb);
            for b in 0..kc / BLOCK {
                let q = unpack_block::<BITS>(codes.add(b * bb));
                for (qi, v) in q.iter().enumerate() {
                    _mm256_store_si256(sp.add(((b * 4 + qi) * 2 + s) * 32) as *mut __m256i, *v);
                }
            }
        }
        let groups = kc / GROUP;
        let mut t = it.t0;
        while t + 4 <= it.t1 {
            tile::<VNNI, BITS, 4>(it, sp, kc0, groups, t, acc);
            t += 4;
        }
        while t < it.t1 {
            tile::<VNNI, BITS, 1>(it, sp, kc0, groups, t, acc);
            t += 1;
        }
        kc0 += kc;
    }
}

#[target_feature(enable = "avx2,avxvnni")]
unsafe fn item_vnni<const BITS: u32>(it: &Item, acc: &mut [[i64; PAIR]]) {
    item::<true, BITS>(it, acc)
}

#[target_feature(enable = "avx2")]
unsafe fn item_avx2<const BITS: u32>(it: &Item, acc: &mut [[i64; PAIR]]) {
    item::<false, BITS>(it, acc)
}

pub fn gemm(
    lin: &X86Linear,
    vnni: bool,
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
    let group_sum = |c: &[i8]| c.iter().map(|v| *v as i32).sum::<i32>();
    let xsum: Vec<i32> = if m <= 8 {
        xq.chunks(GROUP).map(group_sum).collect()
    } else {
        xq.par_chunks(GROUP).map(group_sum).collect()
    };
    let mut offs = vec![0usize];
    for w in &lin.widths {
        offs.push(offs.last().unwrap() + w);
    }
    let pairs = n.div_ceil(PAIR);
    let tchunks = m.div_ceil(TC);
    (0..pairs * tchunks).into_par_iter().for_each(|i| {
        let (pair, tc) = (i % pairs, i / pairs);
        let (t0, t1) = (tc * TC, ((tc + 1) * TC).min(m));
        let it = Item {
            lin,
            pair,
            xq,
            xsum: &xsum,
            mx,
            t0,
            t1,
        };
        let mut acc = vec![[0i64; PAIR]; t1 - t0];
        unsafe {
            match (vnni, lin.bits) {
                (true, 8) => item_vnni::<8>(&it, &mut acc),
                (true, 6) => item_vnni::<6>(&it, &mut acc),
                (true, _) => item_vnni::<4>(&it, &mut acc),
                (false, 8) => item_avx2::<8>(&it, &mut acc),
                (false, 6) => item_avx2::<6>(&it, &mut acc),
                (false, _) => item_avx2::<4>(&it, &mut acc),
            }
        }
        for r in 0..PAIR {
            let row = pair * PAIR + r;
            if row >= n {
                break;
            }
            let lane = (r / ROWS) * ROWS + (r % 2) * 4 + (r % ROWS) / 2;
            let c = lin.c(row);
            let (dst, stride, col) = match out {
                GemmOut::Write(d) | GemmOut::Accumulate(d) => (d.as_mut_ptr(), n, row),
                GemmOut::Parts(ds) => {
                    let p = offs.iter().rposition(|o| *o <= row).unwrap();
                    (ds[p].as_mut_ptr(), lin.widths[p], row - offs[p])
                }
            };
            for t in t0..t1 {
                let y = ((acc[t - t0][lane] as f32) * c) * sx[t];
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
