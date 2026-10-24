//! Full attention (SPEC §5.1). The KV rows of the batch are written first (parallel over tokens and
//! kv heads), then every (token, head) runs independently: scores as `vpdpbusd` dots with the query's
//! sign moved onto the key, the probabilities through `exp8`, and PV as `vpmaddwd` over key pairs
//! with `pq` split into its low and high 12 bits so that every product fits 16-bit lanes; the s32
//! partials are widened into int64 every 8 keys, well inside their bound.

use super::simd::*;
use crate::backend::{AttnLayer, Buf, KvCache, Shape};
use crate::spec;
use rayon::prelude::*;
use std::arch::x86_64::*;

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

#[inline(always)]
unsafe fn head<const VNNI: bool>(h: &mut Head) {
    let (hd, n_kv, kvh, npos) = (h.hd, h.n_kv, h.kvh, h.npos);
    let chunks = hd / 32;
    let mut qa = [_mm256_setzero_si256(); 16];
    let mut qs = [_mm256_setzero_si256(); 16];
    for c in 0..chunks {
        qs[c] = _mm256_loadu_si256(h.qq.as_ptr().add(32 * c) as *const __m256i);
        qa[c] = _mm256_abs_epi8(qs[c]);
    }
    let npad = npos.next_multiple_of(8);
    let mut scores = vec![f32::NEG_INFINITY; npad];
    let mut mx = f32::NEG_INFINITY;
    for j in 0..npos {
        let kp = h.kq.add((j * n_kv + kvh) * hd);
        let mut acc = _mm256_setzero_si256();
        for c in 0..chunks {
            let kk = _mm256_loadu_si256(kp.add(32 * c) as *const __m256i);
            acc = dp::<VNNI, 6>(acc, qa[c], _mm256_sign_epi8(kk, qs[c]));
        }
        let s = ((hsum_epi32(acc) as f32) * h.sq) * *h.sk.add(j * n_kv + kvh);
        scores[j] = s;
        mx = spec::fmax(mx, s);
    }
    let mut pq = vec![0i32; npad];
    let (mxv, fix) = (_mm256_set1_ps(mx), _mm256_set1_ps(spec::P_FIX));
    for j in (0..npad).step_by(8) {
        let e = exp8(_mm256_sub_ps(_mm256_loadu_ps(scores.as_ptr().add(j)), mxv));
        _mm256_storeu_si256(pq.as_mut_ptr().add(j) as *mut __m256i, _mm256_cvtps_epi32(_mm256_mul_ps(e, fix)));
    }
    let den: i64 = pq.iter().map(|p| *p as i64).sum();
    let pairs = npad / 2;
    let pw: Vec<(i32, i32)> = (0..pairs).map(|p| (((pq[2 * p] & 0xFFF) | (pq[2 * p + 1] & 0xFFF) << 16), ((pq[2 * p] >> 12) | (pq[2 * p + 1] >> 12) << 16))).collect();
    let mut num = vec![0i64; hd];
    for ic in 0..hd / 16 {
        let mut acc = [_mm256_setzero_si256(); 4];
        for pc in (0..pairs).step_by(4) {
            let mut part = [_mm256_setzero_si256(); 4];
            for p in pc..(pc + 4).min(pairs) {
                let j0 = 2 * p;
                let j1 = (j0 + 1).min(npos - 1);
                let v0 = _mm256_loadu_si256(h.vq.add((j0 * n_kv + kvh) * hd + 16 * ic) as *const __m256i);
                let v1 = _mm256_loadu_si256(h.vq.add((j1 * n_kv + kvh) * hd + 16 * ic) as *const __m256i);
                let (a, b) = (_mm256_unpacklo_epi16(v0, v1), _mm256_unpackhi_epi16(v0, v1));
                let (wl, wh) = (_mm256_set1_epi32(pw[p].0), _mm256_set1_epi32(pw[p].1));
                part[0] = dp16::<VNNI>(part[0], a, wl);
                part[1] = dp16::<VNNI>(part[1], b, wl);
                part[2] = dp16::<VNNI>(part[2], a, wh);
                part[3] = dp16::<VNNI>(part[3], b, wh);
            }
            let lo = |v: __m256i| _mm256_cvtepi32_epi64(_mm256_castsi256_si128(v));
            let hi = |v: __m256i| _mm256_cvtepi32_epi64(_mm256_extracti128_si256::<1>(v));
            let sh = |v: __m256i| _mm256_slli_epi64::<12>(v);
            acc[0] = _mm256_add_epi64(acc[0], _mm256_add_epi64(lo(part[0]), sh(lo(part[2]))));
            acc[1] = _mm256_add_epi64(acc[1], _mm256_add_epi64(lo(part[1]), sh(lo(part[3]))));
            acc[2] = _mm256_add_epi64(acc[2], _mm256_add_epi64(hi(part[0]), sh(hi(part[2]))));
            acc[3] = _mm256_add_epi64(acc[3], _mm256_add_epi64(hi(part[1]), sh(hi(part[3]))));
        }
        for (q, v) in acc.iter().enumerate() {
            _mm256_storeu_si256(num.as_mut_ptr().add(16 * ic + 4 * q) as *mut __m256i, *v);
        }
    }
    spec::attn_output(&num, den, h.v_scale, h.o);
}

#[target_feature(enable = "avx2,avxvnni")]
unsafe fn head_vnni(h: &mut Head) {
    head::<true>(h)
}

#[target_feature(enable = "avx2")]
unsafe fn head_avx2(h: &mut Head) {
    head::<false>(h)
}

pub fn attention(vnni: bool, l: &AttnLayer, kv: &KvCache, s: &Shape, qg: &Buf<f32>, k: &Buf<f32>, v: &Buf<f32>, o: &Buf<f32>) {
    let (n_h, n_kv, hd, half, m, pos0) = (s.n_h, s.n_kv, s.hd, s.rot / 2, s.m, s.pos0);
    assert!(hd.is_multiple_of(32) && hd <= 512);
    let rep = n_h / n_kv;
    let qscale = 1.0f32 / (hd as f32).sqrt();
    assert_eq!(qscale.to_bits().trailing_zeros(), 23, "qscale must be a power of two");
    let (kq, sk, vq) = (kv.kq.host_mut(), kv.sk.host_mut(), kv.vq.host_mut());
    let rope = |pos: usize| (&l.rope_cos.host()[pos * half..(pos + 1) * half], &l.rope_sin.host()[pos * half..(pos + 1) * half]);
    let kqp = kq.as_mut_ptr() as usize;
    let vqp = vq.as_mut_ptr() as usize;
    sk[pos0 * n_kv..(pos0 + m) * n_kv].par_iter_mut().enumerate().for_each(|(i, skr)| {
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
    o.host_mut()[..m * n_h * hd].par_chunks_mut(hd).enumerate().for_each(|(i, oh)| {
        let (t, h) = (i / n_h, i % n_h);
        let pos = pos0 + t;
        let (rc, rs) = rope(pos);
        let base = (t * n_h + h) * 2 * hd;
        let (qr, gate) = (&qg.host()[base..base + hd], &qg.host()[base + hd..base + 2 * hd]);
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
        unsafe {
            if vnni {
                head_vnni(&mut hh)
            } else {
                head_avx2(&mut hh)
            }
        }
        spec::attn_gate(oh, gate);
    });
}
