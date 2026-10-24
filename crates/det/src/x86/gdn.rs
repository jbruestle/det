//! Gated DeltaNet (SPEC §5.2). The conv, gates and norms of all tokens run first (parallel over
//! tokens); the recurrence then runs per (v head, block of 32 state columns) over the batch in order,
//! the block staying in L1: lanes are state columns, and the `treesum` over the dk rows is the
//! balanced tree built as the rows stream by. The gated output norm runs last (parallel over tokens
//! and heads).

use super::glue::{l2norm_row, rmsnorm_silu_row};
use super::simd::*;
use crate::backend::{Buf, GdnExport, GdnLayer, GdnShape, GdnState};
use crate::spec;
use rayon::prelude::*;
use std::arch::x86_64::*;

const COLS: usize = 32;

#[target_feature(enable = "avx2")]
unsafe fn conv_silu(x: [&[f32]; 4], taps: &[f32], n: usize, y: &mut [f32]) {
    for c in (0..n).step_by(8) {
        let w = |i: usize| _mm256_loadu_ps(taps.as_ptr().add(i * n + c));
        let xi = |i: usize| _mm256_loadu_ps(x[i].as_ptr().add(c));
        let mut acc = _mm256_add_ps(_mm256_mul_ps(w(0), xi(0)), _mm256_mul_ps(w(1), xi(1)));
        acc = _mm256_add_ps(acc, _mm256_mul_ps(w(2), xi(2)));
        acc = _mm256_add_ps(acc, _mm256_mul_ps(w(3), xi(3)));
        _mm256_storeu_ps(y.as_mut_ptr().add(c), silu8(acc));
    }
}

/// Adds leaf `i` to the running balanced tree over 4 vectors of columns.
#[inline(always)]
unsafe fn push(lvl: &mut [[__m256; 4]; 16], i: usize, mut cur: [__m256; 4]) {
    let mut l = 0;
    while (i >> l) & 1 == 1 {
        for c in 0..4 {
            cur[c] = _mm256_add_ps(lvl[l][c], cur[c]);
        }
        l += 1;
    }
    lvl[l] = cur;
}

/// The balanced tree over 8 leaves (levels 1–3).
#[inline(always)]
unsafe fn oct(p: &[__m256; 8]) -> __m256 {
    let a = _mm256_add_ps(_mm256_add_ps(p[0], p[1]), _mm256_add_ps(p[2], p[3]));
    let b = _mm256_add_ps(_mm256_add_ps(p[4], p[5]), _mm256_add_ps(p[6], p[7]));
    _mm256_add_ps(a, b)
}

struct Cols<'a> {
    s: *mut f32,
    dv: usize,
    dk: usize,
    q: &'a [f32],
    k: &'a [f32],
    v: &'a [f32],
    beta: f32,
    decay: f32,
    o: *mut f32,
}

/// One pass over the rows of a 32-column state block: `s = s · scale + add · k[i]` (`add` all zero
/// for the decay pass, where `scale` is the decay; `scale` 1 and `add` the delta for the update),
/// returning `treesum_i(s[i] · f[i])` for the 4 column vectors.
#[inline(always)]
unsafe fn pass(c: &Cols, scale: __m256, add: &[__m256; 4], f: &[f32], update: bool) -> [__m256; 4] {
    let mut lvl = [[_mm256_setzero_ps(); 4]; 16];
    for i8 in 0..c.dk / 8 {
        let mut o = [_mm256_setzero_ps(); 4];
        for j in 0..4 {
            let mut p = [_mm256_setzero_ps(); 8];
            for r in 0..8 {
                let i = i8 * 8 + r;
                let ptr = c.s.add(i * c.dv + 8 * j);
                let s = if update {
                    _mm256_add_ps(_mm256_loadu_ps(ptr), _mm256_mul_ps(_mm256_set1_ps(c.k[i]), add[j]))
                } else {
                    _mm256_mul_ps(_mm256_loadu_ps(ptr), scale)
                };
                _mm256_storeu_ps(ptr, s);
                p[r] = _mm256_mul_ps(s, _mm256_set1_ps(f[i]));
            }
            o[j] = oct(&p);
        }
        push(&mut lvl, i8, o);
    }
    lvl[(c.dk / 8).trailing_zeros() as usize]
}

/// One token's state update and output for 32 columns of one head's state (`s` points at column 0
/// of row 0; rows are `dv` apart).
#[target_feature(enable = "avx2")]
unsafe fn step(c: &Cols) {
    let zero = [_mm256_setzero_ps(); 4];
    let kv = pass(c, _mm256_set1_ps(c.decay), &zero, c.k, false);
    let beta = _mm256_set1_ps(c.beta);
    let mut delta = [_mm256_setzero_ps(); 4];
    for j in 0..4 {
        delta[j] = _mm256_mul_ps(_mm256_sub_ps(_mm256_loadu_ps(c.v.as_ptr().add(8 * j)), kv[j]), beta);
    }
    let o = pass(c, _mm256_setzero_ps(), &delta, c.q, true);
    for j in 0..4 {
        _mm256_storeu_ps(c.o.add(8 * j), o[j]);
    }
}

pub fn gdn(scratch: &mut Vec<f32>, l: &GdnLayer, st: &GdnState, s: &GdnShape, qkv: &Buf<f32>, z: &Buf<f32>, a: &Buf<f32>, b: &Buf<f32>, on: &Buf<f32>, ex: GdnExport) {
    let (n_k, n_v, dk, dv, m) = (s.n_k, s.n_v, s.dk, s.dv, s.m);
    assert!(dk.is_power_of_two() && dk <= 1 << 15 && dv.is_multiple_of(COLS));
    let conv_dim = 2 * n_k * dk + n_v * dv;
    assert!(conv_dim.is_multiple_of(8) && dv.is_multiple_of(8));
    let kd = n_k * dk;
    let rep = n_v / n_k;
    let gw = 2 * dk + dv * rep + 2 * rep;
    let conv_state = st.conv_state.host_mut();
    let qkv = &qkv.host()[..m * conv_dim];
    let conv_w = l.conv.host();
    let mut taps = vec![0f32; 4 * conv_dim];
    for c in 0..conv_dim {
        for i in 0..4 {
            taps[i * conv_dim + c] = conv_w[c * 4 + i];
        }
    }
    let row = |t: isize| -> &[f32] {
        if t >= 0 {
            &qkv[t as usize * conv_dim..(t as usize + 1) * conv_dim]
        } else {
            &conv_state[(3 + t) as usize * conv_dim..(4 + t) as usize * conv_dim]
        }
    };
    let need = m * (conv_dim + 2 * kd + 2 * n_v + n_v * dv);
    if scratch.len() < need {
        scratch.resize(need, 0.0);
    }
    let (y, rest) = scratch.split_at_mut(m * conv_dim);
    let (q, rest) = rest.split_at_mut(m * kd);
    let (k, rest) = rest.split_at_mut(m * kd);
    let (beta, rest) = rest.split_at_mut(m * n_v);
    let (decay, gout_s) = rest.split_at_mut(m * n_v);
    y.par_chunks_mut(conv_dim)
        .zip(q.par_chunks_mut(kd))
        .zip(k.par_chunks_mut(kd))
        .zip(beta.par_chunks_mut(n_v))
        .zip(decay.par_chunks_mut(n_v))
        .enumerate()
        .for_each(|(t, ((((yt, qt), kt), bt), dt))| {
            let t = t as isize;
            unsafe { conv_silu([row(t - 3), row(t - 2), row(t - 1), row(t)], &taps, conv_dim, yt) };
            let mut tmp = Vec::new();
            for h in 0..n_k {
                unsafe {
                    l2norm_row(&yt[h * dk..(h + 1) * dk], s.eps, spec::QSCALE_GDN, &mut qt[h * dk..(h + 1) * dk], &mut tmp);
                    l2norm_row(&yt[kd + h * dk..kd + (h + 1) * dk], s.eps, 1.0, &mut kt[h * dk..(h + 1) * dk], &mut tmp);
                }
            }
            let t = t as usize;
            for h in 0..n_v {
                bt[h] = spec::sigmoid_f(b.host()[t * n_v + h]);
                dt[h] = spec::gdn_decay(a.host()[t * n_v + h], l.dt_bias.host()[h], l.a.host()[h]);
            }
        });
    if m < 3 {
        conv_state.copy_within(m * conv_dim..3 * conv_dim, 0);
    }
    conv_state[(3 - m.min(3)) * conv_dim..].copy_from_slice(&qkv[(m - m.min(3)) * conv_dim..]);
    if let Some(gin) = ex.gin {
        for t in 0..m {
            let (qt, kt, vt) = (&q[t * kd..(t + 1) * kd], &k[t * kd..(t + 1) * kd], &y[t * conv_dim + 2 * kd..(t + 1) * conv_dim]);
            let g = &mut gin.host_mut()[t * n_k * gw..(t + 1) * n_k * gw];
            for kh in 0..n_k {
                let r = &mut g[kh * gw..(kh + 1) * gw];
                r[..dk].copy_from_slice(&qt[kh * dk..(kh + 1) * dk]);
                r[dk..2 * dk].copy_from_slice(&kt[kh * dk..(kh + 1) * dk]);
                r[2 * dk..2 * dk + rep * dv].copy_from_slice(&vt[kh * rep * dv..(kh + 1) * rep * dv]);
                r[2 * dk + rep * dv..2 * dk + rep * dv + rep].copy_from_slice(&beta[t * n_v + kh * rep..t * n_v + (kh + 1) * rep]);
                r[2 * dk + rep * dv + rep..].copy_from_slice(&decay[t * n_v + kh * rep..t * n_v + (kh + 1) * rep]);
            }
        }
    }
    let gout: &mut [f32] = match ex.gout {
        Some(g) => &mut g.host_mut()[..m * n_v * dv],
        None => &mut gout_s[..m * n_v * dv],
    };
    let (sp, gp) = (st.s.as_mut_ptr() as usize, gout.as_mut_ptr() as usize);
    let (q, k, y, beta, decay) = (&*q, &*k, &*y, &*beta, &*decay);
    let blocks = dv / COLS;
    (0..n_v * blocks).into_par_iter().for_each(|task| {
        let (h, j0) = (task / blocks, (task % blocks) * COLS);
        let kh = h / rep;
        for t in 0..m {
            let c = Cols {
                s: unsafe { (sp as *mut f32).add(h * dk * dv + j0) },
                dv,
                dk,
                q: &q[t * kd + kh * dk..t * kd + (kh + 1) * dk],
                k: &k[t * kd + kh * dk..t * kd + (kh + 1) * dk],
                v: &y[t * conv_dim + 2 * kd + h * dv + j0..t * conv_dim + 2 * kd + h * dv + j0 + COLS],
                beta: beta[t * n_v + h],
                decay: decay[t * n_v + h],
                o: unsafe { (gp as *mut f32).add((t * n_v + h) * dv + j0) },
            };
            unsafe { step(&c) };
        }
    });
    let norm = &l.norm.host()[..dv];
    on.host_mut()[..m * n_v * dv].par_chunks_mut(dv).enumerate().for_each(|(i, dst)| unsafe {
        rmsnorm_silu_row(&gout[i * dv..(i + 1) * dv], norm, s.eps, &z.host()[i * dv..(i + 1) * dv], dst, &mut Vec::new());
    });
}
