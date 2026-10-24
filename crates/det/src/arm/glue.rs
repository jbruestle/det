//! Row-wise norms and activation quantization (SPEC §2, §4): the `treesum` of the squares as the
//! balanced tree built from in-register pair sums, then the scaling and `quant_act_i8g` on 4 lanes.
//! Rows are independent and run in parallel.

use super::simd::*;
use crate::backend::Buf;
use crate::spec::{self, GROUP};
use rayon::prelude::*;
use std::arch::aarch64::*;

/// 16 values to 4 consecutive sums of 4, each the balanced tree over its 4 (levels 1–2 of `treesum`).
#[inline(always)]
unsafe fn quads(p: *const f32) -> float32x4_t {
    let ld = |i: usize| vld1q_f32(p.add(4 * i));
    vpaddq_f32(vpaddq_f32(ld(0), ld(1)), vpaddq_f32(ld(2), ld(3)))
}

/// `treesum(x · x)`.
unsafe fn treesum_sq(x: &[f32], tmp: &mut Vec<f32>) -> f32 {
    let n = x.len();
    let p = n.next_power_of_two().max(4);
    tmp.clear();
    tmp.resize(p, 0.0);
    let mut i = 0;
    while i + 4 <= n {
        let v = vld1q_f32(x.as_ptr().add(i));
        vst1q_f32(tmp.as_mut_ptr().add(i), vmulq_f32(v, v));
        i += 4;
    }
    for j in i..n {
        tmp[j] = x[j] * x[j];
    }
    let mut len = p;
    while len >= 16 {
        for g in 0..len / 16 {
            let q = quads(tmp.as_ptr().add(16 * g));
            vst1q_f32(tmp.as_mut_ptr().add(4 * g), q);
        }
        len /= 4;
    }
    let s = (tmp[0] + tmp[1]) + (tmp[2] + tmp[3]);
    if len == 8 {
        s + ((tmp[4] + tmp[5]) + (tmp[6] + tmp[7]))
    } else {
        s
    }
}

/// `quant_act_i8g` of one row.
unsafe fn quant_row(x: &[f32], q: &mut [i8], mx: &mut [u16]) -> f32 {
    let g = x.len() / GROUP;
    let mut s_g = [0f32; 256];
    assert!(g <= s_g.len());
    for gi in 0..g {
        let mut m = vdupq_n_f32(0.0);
        for c in 0..GROUP / 4 {
            m = vmaxq_f32(m, vabsq_f32(vld1q_f32(x.as_ptr().add(gi * GROUP + 4 * c))));
        }
        s_g[gi] = vmaxvq_f32(m) / 127.0;
    }
    let smax = s_g[..g].iter().fold(0.0f32, |m, v| spec::fmax(m, *v));
    let sx = if smax == 0.0 { 1.0 } else { smax / 65535.0 };
    let (lo, hi) = (vdupq_n_s32(-127), vdupq_n_s32(127));
    for gi in 0..g {
        let m = spec::rint(s_g[gi] / sx);
        let m = if m < 1.0 {
            1.0
        } else if m > 65535.0 {
            65535.0
        } else {
            m
        };
        mx[gi] = m as u16;
        let inv = vdupq_n_f32(1.0 / (sx * (mx[gi] as f32)));
        for c in 0..GROUP / 16 {
            // clamp_i8: `fcvtzs` saturates and maps NaN (0 · inf in an all-denormal row) to 0, as `as i8` does
            let cv = |i: usize| {
                let r = vrndnq_f32(vmulq_f32(
                    vld1q_f32(x.as_ptr().add(gi * GROUP + 16 * c + 4 * i)),
                    inv,
                ));
                vqmovn_s32(vmaxq_s32(vminq_s32(vcvtq_s32_f32(r), hi), lo))
            };
            let b = vcombine_s8(
                vqmovn_s16(vcombine_s16(cv(0), cv(1))),
                vqmovn_s16(vcombine_s16(cv(2), cv(3))),
            );
            vst1q_s8(q.as_mut_ptr().add(gi * GROUP + 16 * c), b);
        }
    }
    sx
}

pub unsafe fn rmsnorm_row(x: &[f32], w: &[f32], eps: f32, y: &mut [f32], tmp: &mut Vec<f32>) {
    let n = x.len();
    assert!(n.is_multiple_of(4));
    let ss = treesum_sq(x, tmp);
    let inv = vdupq_n_f32(spec::rsqrt_f(ss / (n as f32) + eps));
    for i in (0..n).step_by(4) {
        let v = vmulq_f32(
            vmulq_f32(vld1q_f32(x.as_ptr().add(i)), inv),
            vld1q_f32(w.as_ptr().add(i)),
        );
        vst1q_f32(y.as_mut_ptr().add(i), v);
    }
}

/// `y = x · (1 / sqrt(treesum(x·x) + eps)) · scale`
pub unsafe fn l2norm_row(x: &[f32], eps: f32, scale: f32, y: &mut [f32], tmp: &mut Vec<f32>) {
    assert!(x.len().is_multiple_of(4));
    let inv = spec::rsqrt_f(treesum_sq(x, tmp) + eps);
    let (inv, scale) = (vdupq_n_f32(inv), vdupq_n_f32(scale));
    for i in (0..x.len()).step_by(4) {
        vst1q_f32(
            y.as_mut_ptr().add(i),
            vmulq_f32(vmulq_f32(vld1q_f32(x.as_ptr().add(i)), inv), scale),
        );
    }
}

/// `y = rmsnorm(x, w, eps) · silu(z)`
pub unsafe fn rmsnorm_silu_row(
    x: &[f32],
    w: &[f32],
    eps: f32,
    z: &[f32],
    y: &mut [f32],
    tmp: &mut Vec<f32>,
) {
    rmsnorm_row(x, w, eps, y, tmp);
    for i in (0..x.len()).step_by(4) {
        vst1q_f32(
            y.as_mut_ptr().add(i),
            vmulq_f32(
                vld1q_f32(y.as_ptr().add(i)),
                silu4(vld1q_f32(z.as_ptr().add(i))),
            ),
        );
    }
}

unsafe fn silu_mul_row(g: &[f32], u: &[f32], h: &mut [f32]) {
    for i in (0..g.len()).step_by(4) {
        let v = vmulq_f32(
            silu4(vld1q_f32(g.as_ptr().add(i))),
            vld1q_f32(u.as_ptr().add(i)),
        );
        vst1q_f32(h.as_mut_ptr().add(i), v);
    }
}

/// `f(row, codes, multipliers) -> scale` over the `m` rows of `n`, in parallel.
fn each_row(
    xq: &Buf<i8>,
    sx: &Buf<f32>,
    mx: &Buf<u16>,
    m: usize,
    n: usize,
    f: impl Fn(usize, &mut [i8], &mut [u16]) -> f32 + Sync,
) {
    assert!(n.is_multiple_of(GROUP));
    xq.host_mut()[..m * n]
        .par_chunks_mut(n)
        .zip(mx.host_mut()[..m * n / GROUP].par_chunks_mut(n / GROUP))
        .zip(sx.host_mut()[..m].par_iter_mut())
        .enumerate()
        .for_each(|(t, ((q, mxr), sxr))| *sxr = f(t, q, mxr));
}

pub fn rmsnorm_quant(
    x: &Buf<f32>,
    w: &Buf<f32>,
    eps: f32,
    m: usize,
    n: usize,
    xq: &Buf<i8>,
    sx: &Buf<f32>,
    mx: &Buf<u16>,
) {
    let w = &w.host()[..n];
    each_row(xq, sx, mx, m, n, |t, q, mxr| unsafe {
        let mut y = vec![0f32; n];
        rmsnorm_row(
            &x.host()[t * n..(t + 1) * n],
            w,
            eps,
            &mut y,
            &mut Vec::new(),
        );
        quant_row(&y, q, mxr)
    });
}

pub fn quant(x: &Buf<f32>, m: usize, n: usize, xq: &Buf<i8>, sx: &Buf<f32>, mx: &Buf<u16>) {
    each_row(xq, sx, mx, m, n, |t, q, mxr| unsafe {
        quant_row(&x.host()[t * n..(t + 1) * n], q, mxr)
    });
}

pub fn silu_mul_quant(
    g: &Buf<f32>,
    u: &Buf<f32>,
    m: usize,
    n: usize,
    hq: &Buf<i8>,
    sh: &Buf<f32>,
    mh: &Buf<u16>,
) {
    each_row(hq, sh, mh, m, n, |t, q, mxr| unsafe {
        let mut h = vec![0f32; n];
        silu_mul_row(
            &g.host()[t * n..(t + 1) * n],
            &u.host()[t * n..(t + 1) * n],
            &mut h,
        );
        quant_row(&h, q, mxr)
    });
}
