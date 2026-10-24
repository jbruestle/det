//! The CPU backend: the spec functions applied row by row, parallel only across independent rows and
//! heads (which cannot change a result). Arithmetic is written in the spec's op order.
#![allow(clippy::assign_op_pattern, clippy::needless_range_loop)]

use crate::backend::*;
use crate::gguf::{Gguf, TensorInfo};
use crate::spec;
use rayon::prelude::*;
use std::sync::Arc;

pub struct Cpu;

impl crate::hooks::Reader for Cpu {
    unsafe fn read_bytes(&self, src: *const u8, bytes: usize, out: &mut [u8]) {
        out.copy_from_slice(unsafe { std::slice::from_raw_parts(src, bytes) });
    }
}

pub struct CpuLinear {
    gguf: Arc<Gguf>,
    parts: Vec<(usize, usize)>,
    pub widths: Vec<usize>,
    pub k: usize,
    pub bits: u32,
}

impl CpuLinear {
    /// The det row (`[codes][c][m]`) of output row `n`.
    pub fn row(&self, n: usize) -> &[u8] {
        let rb = spec::det_row_bytes(self.bits, self.k);
        let mut n = n;
        for (i, w) in self.widths.iter().enumerate() {
            if n < *w {
                let (off, len) = self.parts[i];
                let data = &self.gguf.file_bytes()[off..off + len];
                return &data[n * rb..(n + 1) * rb];
            }
            n -= w;
        }
        panic!("row out of range");
    }
}

/// Builds a linear over the file mapping; shared by backends that read the det rows in place.
pub fn linear_parts(g: &Arc<Gguf>, parts: &[&TensorInfo]) -> CpuLinear {
    let bits = parts[0]
        .ty()
        .and_then(|t| t.bits())
        .unwrap_or_else(|| panic!("{} is not a det linear; run `det convert`", parts[0].name));
    let k = parts[0].dims[0];
    let mut offs = Vec::new();
    let mut widths = Vec::new();
    for t in parts {
        assert!(
            t.ty() == parts[0].ty() && t.dims[0] == k && t.dims.len() == 2,
            "fused parts differ in K or type: {}",
            t.name
        );
        let data = g.data(t);
        let off = data.as_ptr() as usize - g.file_bytes().as_ptr() as usize;
        offs.push((off, data.len()));
        widths.push(t.dims[1]);
    }
    CpuLinear {
        gguf: g.clone(),
        parts: offs,
        widths,
        k,
        bits,
    }
}

fn rows<T: Copy>(b: &Buf<T>, m: usize, n: usize) -> Vec<&mut [T]> {
    assert!(b.len() >= m * n);
    b.host_mut()[..m * n].chunks_mut(n).collect()
}

impl Backend for Cpu {
    type Linear = CpuLinear;

    fn name(&self) -> String {
        "cpu".into()
    }
    fn alloc<T: Copy>(&self, len: usize) -> Buf<T> {
        Buf::new_host(len)
    }
    fn write<T: Copy>(&self, b: &Buf<T>, offset: usize, data: &[T]) {
        b.host_mut()[offset..offset + data.len()].copy_from_slice(data);
    }
    fn read<T: Copy>(&self, b: &Buf<T>, offset: usize, out: &mut [T]) {
        out.copy_from_slice(&b.host()[offset..offset + out.len()]);
    }
    fn zero<T: Copy>(&self, b: &Buf<T>) {
        b.host_mut().fill(unsafe { std::mem::zeroed() });
    }
    fn sync(&self) {}

    fn linear(&self, g: &Arc<Gguf>, parts: &[&TensorInfo]) -> CpuLinear {
        linear_parts(g, parts)
    }
    fn linear_widths(lin: &CpuLinear) -> &[usize] {
        &lin.widths
    }

    fn rmsnorm_quant(
        &self,
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
        let g = n / spec::GROUP;
        rows(xq, m, n)
            .into_par_iter()
            .zip(rows(mx, m, g))
            .zip(rows(sx, m, 1))
            .enumerate()
            .for_each(|(t, ((q, mxr), sxr))| {
                let mut y = vec![0f32; n];
                spec::rmsnorm(&x.host()[t * n..(t + 1) * n], w, eps, &mut y);
                sxr[0] = spec::quant_act_i8g(&y, q, mxr);
            });
    }

    fn quant(&self, x: &Buf<f32>, m: usize, n: usize, xq: &Buf<i8>, sx: &Buf<f32>, mx: &Buf<u16>) {
        let g = n / spec::GROUP;
        rows(xq, m, n)
            .into_par_iter()
            .zip(rows(mx, m, g))
            .zip(rows(sx, m, 1))
            .enumerate()
            .for_each(|(t, ((q, mxr), sxr))| {
                sxr[0] = spec::quant_act_i8g(&x.host()[t * n..(t + 1) * n], q, mxr);
            });
    }

    fn gemm(
        &self,
        lin: &CpuLinear,
        xq: &Buf<i8>,
        sx: &Buf<f32>,
        mx: &Buf<u16>,
        m: usize,
        out: GemmOut,
    ) {
        let (k, n, bits) = (lin.k, lin.widths.iter().sum::<usize>(), lin.bits);
        let g = k / spec::GROUP;
        for t in 0..m {
            let (xr, sxt, mxr) = (
                &xq.host()[t * k..(t + 1) * k],
                sx.host()[t],
                &mx.host()[t * g..(t + 1) * g],
            );
            let y: Vec<f32> = (0..n)
                .into_par_iter()
                .map(|j| {
                    let r = spec::det_row(bits, k, lin.row(j));
                    let mut wq = vec![0i8; k];
                    spec::unpack_row(r.codes, bits, k, &mut wq);
                    spec::gemm_row(&wq, r.c, &r.m, xr, sxt, Some(mxr))
                })
                .collect();
            match &out {
                GemmOut::Write(dst) => dst.host_mut()[t * n..(t + 1) * n].copy_from_slice(&y),
                GemmOut::Accumulate(dst) => {
                    let d = &mut dst.host_mut()[t * n..(t + 1) * n];
                    for j in 0..n {
                        d[j] = d[j] + y[j];
                    }
                }
                GemmOut::Parts(dsts) => {
                    let mut off = 0;
                    for (i, w) in lin.widths.iter().enumerate() {
                        dsts[i].host_mut()[t * w..(t + 1) * w].copy_from_slice(&y[off..off + w]);
                        off += w;
                    }
                }
            }
        }
    }

    fn silu_mul_quant(
        &self,
        gb: &Buf<f32>,
        u: &Buf<f32>,
        m: usize,
        n: usize,
        hq: &Buf<i8>,
        sh: &Buf<f32>,
        mh: &Buf<u16>,
    ) {
        let g = n / spec::GROUP;
        rows(hq, m, n)
            .into_par_iter()
            .zip(rows(mh, m, g))
            .zip(rows(sh, m, 1))
            .enumerate()
            .for_each(|(t, ((q, mxr), sxr))| {
                let (gr, ur) = (
                    &gb.host()[t * n..(t + 1) * n],
                    &u.host()[t * n..(t + 1) * n],
                );
                let h: Vec<f32> = (0..n).map(|i| spec::silu_f(gr[i]) * ur[i]).collect();
                sxr[0] = spec::quant_act_i8g(&h, q, mxr);
            });
    }

    fn attention(
        &self,
        l: &AttnLayer,
        kv: &KvCache,
        s: &Shape,
        qg: &Buf<f32>,
        k: &Buf<f32>,
        v: &Buf<f32>,
        o: &Buf<f32>,
    ) {
        let (n_h, n_kv, hd, half) = (s.n_h, s.n_kv, s.hd, s.rot / 2);
        let rep = n_h / n_kv;
        let qscale = 1.0f32 / (hd as f32).sqrt();
        assert_eq!(
            qscale.to_bits().trailing_zeros(),
            23,
            "qscale must be a power of two"
        );
        let (kq, sk, vq) = (kv.kq.host_mut(), kv.sk.host_mut(), kv.vq.host_mut());
        for t in 0..s.m {
            let pos = s.pos0 + t;
            let (rc, rs) = (
                &l.rope_cos.host()[pos * half..(pos + 1) * half],
                &l.rope_sin.host()[pos * half..(pos + 1) * half],
            );
            for h in 0..n_kv {
                let kr = &k.host()[(t * n_kv + h) * hd..(t * n_kv + h + 1) * hd];
                let mut kn = vec![0f32; hd];
                spec::rmsnorm(kr, &l.k_norm.host()[..hd], s.eps, &mut kn);
                spec::rope(&mut kn, rc, rs);
                let row = (pos * n_kv + h) * hd;
                sk[pos * n_kv + h] = spec::quant_act_i8(&kn, &mut kq[row..row + hd]);
                let vr = &v.host()[(t * n_kv + h) * hd..(t * n_kv + h + 1) * hd];
                spec::quant_v(vr, l.v_inv_scale.host()[h], &mut vq[row..row + hd]);
            }
            let keys = |kvh: usize| -> Vec<spec::KvRow> {
                (0..=pos)
                    .map(|j| spec::KvRow {
                        kq: &kq[(j * n_kv + kvh) * hd..(j * n_kv + kvh + 1) * hd],
                        sk: sk[j * n_kv + kvh],
                        vq: &vq[(j * n_kv + kvh) * hd..(j * n_kv + kvh + 1) * hd],
                    })
                    .collect()
            };
            let outs: Vec<Vec<f32>> = (0..n_h)
                .into_par_iter()
                .map(|h| {
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
                    let mut oh = vec![0f32; hd];
                    spec::attn_head(&qq, sq, &keys(kvh), l.v_scale.host()[kvh], &mut oh);
                    spec::attn_gate(&mut oh, gate);
                    oh
                })
                .collect();
            for (h, oh) in outs.iter().enumerate() {
                o.host_mut()[(t * n_h + h) * hd..(t * n_h + h + 1) * hd].copy_from_slice(oh);
            }
        }
    }

    fn gdn(
        &self,
        l: &GdnLayer,
        st: &GdnState,
        s: &GdnShape,
        qkv: &Buf<f32>,
        z: &Buf<f32>,
        a: &Buf<f32>,
        b: &Buf<f32>,
        on: &Buf<f32>,
        ex: GdnExport,
    ) {
        let (n_k, n_v, dk, dv) = (s.n_k, s.n_v, s.dk, s.dv);
        let conv_dim = 2 * n_k * dk + n_v * dv;
        let kd = n_k * dk;
        let rep = n_v / n_k;
        let gw = 2 * dk + dv * rep + 2 * rep;
        let conv_state = st.conv_state.host_mut();
        let sm = st.s.host_mut();
        let conv_w = l.conv.host();
        for t in 0..s.m {
            let xt = &qkv.host()[t * conv_dim..(t + 1) * conv_dim];
            let mut y = vec![0f32; conv_dim];
            for c in 0..conv_dim {
                let win = [
                    conv_state[c],
                    conv_state[conv_dim + c],
                    conv_state[2 * conv_dim + c],
                    xt[c],
                ];
                let w: [f32; 4] = conv_w[c * 4..c * 4 + 4].try_into().unwrap();
                y[c] = spec::gdn_conv(&win, &w);
            }
            conv_state.copy_within(conv_dim..3 * conv_dim, 0);
            conv_state[2 * conv_dim..].copy_from_slice(xt);
            let mut q = vec![0f32; kd];
            let mut k = vec![0f32; kd];
            for h in 0..n_k {
                spec::l2norm(
                    &y[h * dk..(h + 1) * dk],
                    s.eps,
                    &mut q[h * dk..(h + 1) * dk],
                );
                for x in q[h * dk..(h + 1) * dk].iter_mut() {
                    *x = *x * spec::QSCALE_GDN;
                }
                spec::l2norm(
                    &y[kd + h * dk..kd + (h + 1) * dk],
                    s.eps,
                    &mut k[h * dk..(h + 1) * dk],
                );
            }
            let v = &y[2 * kd..];
            let beta: Vec<f32> = (0..n_v)
                .map(|h| spec::sigmoid_f(b.host()[t * n_v + h]))
                .collect();
            let decay: Vec<f32> = (0..n_v)
                .map(|h| spec::gdn_decay(a.host()[t * n_v + h], l.dt_bias.host()[h], l.a.host()[h]))
                .collect();
            if let Some(gin) = ex.gin {
                let g = &mut gin.host_mut()[t * n_k * gw..(t + 1) * n_k * gw];
                for kh in 0..n_k {
                    let r = &mut g[kh * gw..(kh + 1) * gw];
                    r[..dk].copy_from_slice(&q[kh * dk..(kh + 1) * dk]);
                    r[dk..2 * dk].copy_from_slice(&k[kh * dk..(kh + 1) * dk]);
                    r[2 * dk..2 * dk + rep * dv]
                        .copy_from_slice(&v[kh * rep * dv..(kh + 1) * rep * dv]);
                    r[2 * dk + rep * dv..2 * dk + rep * dv + rep]
                        .copy_from_slice(&beta[kh * rep..(kh + 1) * rep]);
                    r[2 * dk + rep * dv + rep..].copy_from_slice(&decay[kh * rep..(kh + 1) * rep]);
                }
            }
            let heads: Vec<&mut [f32]> = sm.chunks_mut(dk * dv).collect();
            let outs: Vec<Vec<f32>> = heads
                .into_par_iter()
                .enumerate()
                .map(|(h, sh)| {
                    let kh = h / rep;
                    let mut o = vec![0f32; dv];
                    spec::gdn_head_step(
                        sh,
                        dk,
                        dv,
                        &k[kh * dk..(kh + 1) * dk],
                        &q[kh * dk..(kh + 1) * dk],
                        &v[h * dv..(h + 1) * dv],
                        beta[h],
                        decay[h],
                        &mut o,
                    );
                    o
                })
                .collect();
            let norm = &l.norm.host()[..dv];
            for h in 0..n_v {
                let base = (t * n_v + h) * dv;
                if let Some(gout) = ex.gout {
                    gout.host_mut()[base..base + dv].copy_from_slice(&outs[h]);
                }
                let mut yn = vec![0f32; dv];
                spec::rmsnorm(&outs[h], norm, s.eps, &mut yn);
                let zr = &z.host()[base..base + dv];
                let dst = &mut on.host_mut()[base..base + dv];
                for j in 0..dv {
                    dst[j] = yn[j] * spec::silu_f(zr[j]);
                }
            }
        }
    }

    fn argmax(&self, logits: &Buf<f32>, v: usize) -> i32 {
        spec::argmax(&logits.host()[..v]) as i32
    }

    fn sample(
        &self,
        logits: &Buf<f32>,
        v: usize,
        inv_temp: f32,
        top_k: usize,
        top_p: f32,
        z: u64,
    ) -> i32 {
        spec::sample_with_draw(&logits.host()[..v], inv_temp, top_k, top_p, z) as i32
    }
}
