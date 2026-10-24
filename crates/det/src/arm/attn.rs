//! Full attention (SPEC §5.1). The KV rows of the batch are written first (parallel over tokens and
//! kv heads), then every (token, head) runs independently: scores as `sdot` of the query against each
//! key, the probabilities through `exp4`, and PV as `smlal` of the int16 values widened to int32
//! against `pq`, straight into int64 accumulators.

use super::simd::*;
use crate::backend::{AttnLayer, Buf, KvCache, Shape};
use crate::spec;
use rayon::prelude::*;
use std::arch::aarch64::*;

struct Head<'a> {
    kq: *const i8,
    sk: *const f32,
    vq: *const i16,
    n_kv: usize,
    hd: usize,
    kvh: usize,
    npos: usize,
    qq: &'a [i8],
    sq: f32,
    v_scale: f32,
    o: &'a mut [f32],
}

unsafe impl Send for Head<'_> {}

#[target_feature(enable = "dotprod")]
unsafe fn head(h: &mut Head) {
    let (hd, n_kv, kvh, npos) = (h.hd, h.n_kv, h.kvh, h.npos);
    let npad = npos.next_multiple_of(4);
    let mut scores = vec![f32::NEG_INFINITY; npad];
    let mut mx = f32::NEG_INFINITY;
    for j in 0..npos {
        let kp = h.kq.add((j * n_kv + kvh) * hd);
        let mut acc = [vdupq_n_s32(0); 2];
        for c in 0..hd / 32 {
            for p in 0..2 {
                acc[p] = sdot(
                    acc[p],
                    vld1q_s8(h.qq.as_ptr().add(32 * c + 16 * p)),
                    vld1q_s8(kp.add(32 * c + 16 * p)),
                );
            }
        }
        let s = ((vaddvq_s32(vaddq_s32(acc[0], acc[1])) as f32) * h.sq) * *h.sk.add(j * n_kv + kvh);
        scores[j] = s;
        mx = spec::fmax(mx, s);
    }
    let mut pq = vec![0i32; npad];
    let (mxv, fix) = (vdupq_n_f32(mx), vdupq_n_f32(spec::P_FIX));
    for j in (0..npad).step_by(4) {
        let e = exp4(vsubq_f32(vld1q_f32(scores.as_ptr().add(j)), mxv));
        vst1q_s32(pq.as_mut_ptr().add(j), vcvtnq_s32_f32(vmulq_f32(e, fix)));
    }
    let den: i64 = pq.iter().map(|p| *p as i64).sum();
    let mut num = vec![0i64; hd];
    for ic in 0..hd / 32 {
        let mut acc = [vdupq_n_s64(0); 16];
        for j in 0..npos {
            let vp = h.vq.add((j * n_kv + kvh) * hd + 32 * ic);
            let p = vdupq_n_s32(pq[j]);
            for q in 0..4 {
                let v = vld1q_s16(vp.add(8 * q));
                let (lo, hi) = (vmovl_s16(vget_low_s16(v)), vmovl_high_s16(v));
                acc[4 * q] = vmlal_s32(acc[4 * q], vget_low_s32(lo), vget_low_s32(p));
                acc[4 * q + 1] = vmlal_high_s32(acc[4 * q + 1], lo, p);
                acc[4 * q + 2] = vmlal_s32(acc[4 * q + 2], vget_low_s32(hi), vget_low_s32(p));
                acc[4 * q + 3] = vmlal_high_s32(acc[4 * q + 3], hi, p);
            }
        }
        for (q, v) in acc.iter().enumerate() {
            vst1q_s64(num.as_mut_ptr().add(32 * ic + 2 * q), *v);
        }
    }
    spec::attn_output(&num, den, h.v_scale, h.o);
}

pub fn attention(
    l: &AttnLayer,
    kv: &KvCache,
    s: &Shape,
    qg: &Buf<f32>,
    k: &Buf<f32>,
    v: &Buf<f32>,
    o: &Buf<f32>,
) {
    let (n_h, n_kv, hd, half, m, pos0) = (s.n_h, s.n_kv, s.hd, s.rot / 2, s.m, s.pos0);
    assert!(hd.is_multiple_of(32));
    let rep = n_h / n_kv;
    let qscale = 1.0f32 / (hd as f32).sqrt();
    assert_eq!(
        qscale.to_bits().trailing_zeros(),
        23,
        "qscale must be a power of two"
    );
    let (kq, sk, vq) = (kv.kq.host_mut(), kv.sk.host_mut(), kv.vq.host_mut());
    let rope = |pos: usize| {
        (
            &l.rope_cos.host()[pos * half..(pos + 1) * half],
            &l.rope_sin.host()[pos * half..(pos + 1) * half],
        )
    };
    let kqp = kq.as_mut_ptr() as usize;
    let vqp = vq.as_mut_ptr() as usize;
    sk[pos0 * n_kv..(pos0 + m) * n_kv]
        .par_iter_mut()
        .enumerate()
        .for_each(|(i, skr)| {
            let (t, h) = (i / n_kv, i % n_kv);
            let pos = pos0 + t;
            let (rc, rs) = rope(pos);
            let kr = &k.host()[(t * n_kv + h) * hd..(t * n_kv + h + 1) * hd];
            let mut kn = vec![0f32; hd];
            spec::rmsnorm(kr, &l.k_norm.host()[..hd], s.eps, &mut kn);
            spec::rope(&mut kn, rc, rs);
            let row = (pos * n_kv + h) * hd;
            let kqr = unsafe { std::slice::from_raw_parts_mut((kqp as *mut i8).add(row), hd) };
            let vqr = unsafe { std::slice::from_raw_parts_mut((vqp as *mut i16).add(row), hd) };
            *skr = spec::quant_act_i8(&kn, kqr);
            let vr = &v.host()[(t * n_kv + h) * hd..(t * n_kv + h + 1) * hd];
            spec::quant_v(vr, l.v_inv_scale.host()[h], vqr);
        });
    o.host_mut()[..m * n_h * hd]
        .par_chunks_mut(hd)
        .enumerate()
        .for_each(|(i, oh)| {
            let (t, h) = (i / n_h, i % n_h);
            let pos = pos0 + t;
            let (rc, rs) = rope(pos);
            let base = (t * n_h + h) * 2 * hd;
            let (qr, gate) = (
                &qg.host()[base..base + hd],
                &qg.host()[base + hd..base + 2 * hd],
            );
            let mut qn = vec![0f32; hd];
            spec::rmsnorm(qr, &l.q_norm.host()[..hd], s.eps, &mut qn);
            spec::rope(&mut qn, rc, rs);
            for x in qn.iter_mut() {
                *x = *x * qscale;
            }
            let mut qq = vec![0i8; hd];
            let sq = spec::quant_act_i8(&qn, &mut qq);
            let kvh = h / rep;
            let mut hh = Head {
                kq: kq.as_ptr(),
                sk: sk.as_ptr(),
                vq: vq.as_ptr(),
                n_kv,
                hd,
                kvh,
                npos: pos + 1,
                qq: &qq,
                sq,
                v_scale: l.v_scale.host()[kvh],
                o: oh,
            };
            unsafe { head(&mut hh) };
            spec::attn_gate(oh, gate);
        });
}
