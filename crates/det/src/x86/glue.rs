//! Row-wise norms and activation quantization (SPEC §2, §4): the `treesum` of the squares as the
//! balanced tree built from in-register pair sums, then the scaling and `quant_act_i8g` on 8 lanes.
//! Rows are independent and run in parallel.

use super::simd::*;
use crate::backend::Buf;
use crate::spec::{self, GROUP};
use rayon::prelude::*;
use std::arch::x86_64::*;

/// 32 values to 4 consecutive sums of 8, each the balanced tree over its 8 (levels 1–3 of `treesum`).
#[inline(always)]
unsafe fn octets(p: *const f32) -> __m128 {
    let ld = |i: usize| _mm256_loadu_ps(p.add(8 * i));
    let h = _mm256_hadd_ps(_mm256_hadd_ps(ld(0), ld(1)), _mm256_hadd_ps(ld(2), ld(3)));
    _mm_add_ps(_mm256_castps256_ps128(h), _mm256_extractf128_ps::<1>(h))
}

/// `treesum(x · x)`. The tree is padded with +0.0 beyond the spec's power of two; sums of squares are
/// never −0.0, so this cannot change the result.
#[target_feature(enable = "avx2")]
unsafe fn treesum_sq(x: &[f32], tmp: &mut Vec<f32>) -> f32 {
    let n = x.len();
    let p = n.next_power_of_two().max(32);
    tmp.clear();
    tmp.resize(p, 0.0);
    let mut i = 0;
    while i + 8 <= n {
        let v = _mm256_loadu_ps(x.as_ptr().add(i));
        _mm256_storeu_ps(tmp.as_mut_ptr().add(i), _mm256_mul_ps(v, v));
        i += 8;
    }
    for j in i..n {
        tmp[j] = x[j] * x[j];
    }
    let mut len = p;
    loop {
        if len < 32 {
            tmp[len..32].fill(0.0);
        }
        for g in 0..len.div_ceil(32) {
            let o = octets(tmp.as_ptr().add(32 * g));
            _mm_storeu_ps(tmp.as_mut_ptr().add(4 * g), o);
        }
        if len < 32 {
            return (tmp[0] + tmp[1]) + (tmp[2] + tmp[3]);
        }
        len /= 8;
    }
}

/// `quant_act_i8g` of one row.
#[target_feature(enable = "avx2")]
unsafe fn quant_row(x: &[f32], q: &mut [i8], mx: &mut [u16]) -> f32 {
    let g = x.len() / GROUP;
    let sign = _mm256_set1_ps(-0.0);
    let mut s_g = [0f32; 256];
    assert!(g <= s_g.len());
    for gi in 0..g {
        let mut m = _mm256_setzero_ps();
        for c in 0..GROUP / 8 {
            m = _mm256_max_ps(m, _mm256_andnot_ps(sign, _mm256_loadu_ps(x.as_ptr().add(gi * GROUP + 8 * c))));
        }
        let mut a = [0f32; 8];
        _mm256_storeu_ps(a.as_mut_ptr(), m);
        s_g[gi] = a.iter().fold(0.0f32, |acc, v| spec::fmax(acc, *v)) / 127.0;
    }
    let smax = s_g[..g].iter().fold(0.0f32, |m, v| spec::fmax(m, *v));
    let sx = if smax == 0.0 { 1.0 } else { smax / 65535.0 };
    let (lo, hi) = (_mm256_set1_ps(-127.0), _mm256_set1_ps(127.0));
    let order = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
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
        let inv = _mm256_set1_ps(1.0 / (sx * (mx[gi] as f32)));
        for c in 0..GROUP / 32 {
            // clamp_i8: beyond ±127 saturates, NaN (0 · inf in an all-denormal row) is 0, as `as i8` does
            let cv = |i: usize| {
                let v = _mm256_mul_ps(_mm256_loadu_ps(x.as_ptr().add(gi * GROUP + 32 * c + 8 * i)), inv);
                let r = _mm256_round_ps::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(v);
                let q = _mm256_cvtps_epi32(r);
                let q = _mm256_blendv_epi8(q, _mm256_set1_epi32(127), _mm256_castps_si256(_mm256_cmp_ps::<_CMP_GT_OQ>(r, hi)));
                let q = _mm256_blendv_epi8(q, _mm256_set1_epi32(-127), _mm256_castps_si256(_mm256_cmp_ps::<_CMP_LT_OQ>(r, lo)));
                _mm256_andnot_si256(_mm256_castps_si256(_mm256_cmp_ps::<_CMP_UNORD_Q>(r, r)), q)
            };
            let b = _mm256_packs_epi16(_mm256_packs_epi32(cv(0), cv(1)), _mm256_packs_epi32(cv(2), cv(3)));
            _mm256_storeu_si256(q.as_mut_ptr().add(gi * GROUP + 32 * c) as *mut __m256i, _mm256_permutevar8x32_epi32(b, order));
        }
    }
    sx
}

#[target_feature(enable = "avx2")]
pub unsafe fn rmsnorm_row(x: &[f32], w: &[f32], eps: f32, y: &mut [f32], tmp: &mut Vec<f32>) {
    let n = x.len();
    let ss = treesum_sq(x, tmp);
    let inv = _mm256_set1_ps(spec::rsqrt_f(ss / (n as f32) + eps));
    let mut i = 0;
    while i + 8 <= n {
        let v = _mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(x.as_ptr().add(i)), inv), _mm256_loadu_ps(w.as_ptr().add(i)));
        _mm256_storeu_ps(y.as_mut_ptr().add(i), v);
        i += 8;
    }
}

/// `y = x · (1 / sqrt(treesum(x·x) + eps)) · scale`
#[target_feature(enable = "avx2")]
pub unsafe fn l2norm_row(x: &[f32], eps: f32, scale: f32, y: &mut [f32], tmp: &mut Vec<f32>) {
    let inv = spec::rsqrt_f(treesum_sq(x, tmp) + eps);
    let (inv, scale) = (_mm256_set1_ps(inv), _mm256_set1_ps(scale));
    for i in (0..x.len()).step_by(8) {
        _mm256_storeu_ps(y.as_mut_ptr().add(i), _mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(x.as_ptr().add(i)), inv), scale));
    }
}

/// `y = rmsnorm(x, w, eps) · silu(z)`
#[target_feature(enable = "avx2")]
pub unsafe fn rmsnorm_silu_row(x: &[f32], w: &[f32], eps: f32, z: &[f32], y: &mut [f32], tmp: &mut Vec<f32>) {
    rmsnorm_row(x, w, eps, y, tmp);
    for i in (0..x.len()).step_by(8) {
        _mm256_storeu_ps(y.as_mut_ptr().add(i), _mm256_mul_ps(_mm256_loadu_ps(y.as_ptr().add(i)), silu8(_mm256_loadu_ps(z.as_ptr().add(i)))));
    }
}

#[target_feature(enable = "avx2")]
unsafe fn silu_mul_row(g: &[f32], u: &[f32], h: &mut [f32]) {
    for i in (0..g.len()).step_by(8) {
        let v = _mm256_mul_ps(silu8(_mm256_loadu_ps(g.as_ptr().add(i))), _mm256_loadu_ps(u.as_ptr().add(i)));
        _mm256_storeu_ps(h.as_mut_ptr().add(i), v);
    }
}

/// `f(row, codes, multipliers) -> scale` over the `m` rows of `n`, in parallel.
fn each_row(xq: &Buf<i8>, sx: &Buf<f32>, mx: &Buf<u16>, m: usize, n: usize, f: impl Fn(usize, &mut [i8], &mut [u16]) -> f32 + Sync) {
    assert!(n.is_multiple_of(GROUP));
    xq.host_mut()[..m * n]
        .par_chunks_mut(n)
        .zip(mx.host_mut()[..m * n / GROUP].par_chunks_mut(n / GROUP))
        .zip(sx.host_mut()[..m].par_iter_mut())
        .enumerate()
        .for_each(|(t, ((q, mxr), sxr))| *sxr = f(t, q, mxr));
}

pub fn rmsnorm_quant(x: &Buf<f32>, w: &Buf<f32>, eps: f32, m: usize, n: usize, xq: &Buf<i8>, sx: &Buf<f32>, mx: &Buf<u16>) {
    let w = &w.host()[..n];
    each_row(xq, sx, mx, m, n, |t, q, mxr| unsafe {
        let mut y = vec![0f32; n];
        rmsnorm_row(&x.host()[t * n..(t + 1) * n], w, eps, &mut y, &mut Vec::new());
        quant_row(&y, q, mxr)
    });
}

pub fn quant(x: &Buf<f32>, m: usize, n: usize, xq: &Buf<i8>, sx: &Buf<f32>, mx: &Buf<u16>) {
    each_row(xq, sx, mx, m, n, |t, q, mxr| unsafe { quant_row(&x.host()[t * n..(t + 1) * n], q, mxr) });
}

pub fn silu_mul_quant(g: &Buf<f32>, u: &Buf<f32>, m: usize, n: usize, hq: &Buf<i8>, sh: &Buf<f32>, mh: &Buf<u16>) {
    each_row(hq, sh, mh, m, n, |t, q, mxr| unsafe {
        let mut h = vec![0f32; n];
        silu_mul_row(&g.host()[t * n..(t + 1) * n], &u.host()[t * n..(t + 1) * n], &mut h);
        quant_row(&h, q, mxr)
    });
}
