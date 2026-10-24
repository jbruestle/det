//! The Qwen3.5 hybrid network (SPEC §5): 3 Gated DeltaNet layers then 1 full-attention layer, each
//! followed by a SwiGLU MLP; zero-centered RMSNorm everywhere; tied or separate output head.

use crate::backend::*;
use crate::hooks::{Ctx, Observer, Tap};
use crate::model::{Mixer, Model};

pub fn forward<B: Backend, O: Observer + ?Sized>(m: &Model<B>, pos0: usize, t: usize, mut obs: Option<&mut O>) {
    let c = &m.cfg;
    let bk = &m.backend;
    let w = &m.w;
    let h = c.hidden;
    let sync = || bk.sync();
    let ctx = Ctx { pos0, t, sync: &sync, reader: bk };
    let gdn_io = obs.as_ref().is_some_and(|o| o.wants_gdn_io());
    let mut observe = |tap: Tap| {
        if let Some(o) = obs.as_deref_mut() {
            o.observe(&ctx, tap);
        }
    };
    for (li, l) in m.layers.iter().enumerate() {
        observe(Tap::Residual { layer: li, x: &w.x });
        bk.rmsnorm_quant(&w.x, &l.attn_norm, c.eps, t, h, &w.xq, &w.sx, &w.mx);
        match &l.mixer {
            Mixer::Attn(a) => {
                m.project(&a.qkv, &w.xq, &w.sx, &w.mx, t, &[&w.qg, &w.k, &w.v]);
                observe(Tap::AttnV { layer: li, v: &w.v });
                let layer = AttnLayer { q_norm: &a.q_norm, k_norm: &a.k_norm, v_scale: &a.v_scale, v_inv_scale: &a.v_inv_scale, rope_cos: &m.rope_cos, rope_sin: &m.rope_sin };
                let shape = Shape { m: t, pos0, n_h: c.n_heads, n_kv: c.n_kv, hd: c.head_dim, rot: c.rot_dim, eps: c.eps };
                bk.attention(&layer, &a.kv, &shape, &w.qg, &w.k, &w.v, &w.o);
                observe(Tap::Kv { layer: li, kv: &a.kv });
                observe(Tap::AttnOut { layer: li, o: &w.o });
                observe(Tap::MixerOut { layer: li, y: &w.o });
                bk.quant(&w.o, t, c.n_heads * c.head_dim, &w.oq, &w.so, &w.mo);
                bk.gemm(&a.o.lins[0], &w.oq, &w.so, &w.mo, t, GemmOut::Accumulate(&w.x));
            }
            Mixer::Gdn(g) => {
                m.project(&g.qkvz, &w.xq, &w.sx, &w.mx, t, &[&w.qkv, &w.z]);
                m.project(&g.ab, &w.xq, &w.sx, &w.mx, t, &[&w.a, &w.b]);
                let layer = GdnLayer { conv: &g.conv, a: &g.a, dt_bias: &g.dt_bias, norm: &g.norm };
                let shape = GdnShape { m: t, n_k: c.gdn_k_heads, n_v: c.gdn_v_heads, dk: c.gdn_dk, dv: c.gdn_dv, eps: c.eps };
                let ex = if gdn_io { GdnExport { gin: Some(&w.gin), gout: Some(&w.gout) } } else { GdnExport { gin: None, gout: None } };
                bk.gdn(&layer, &g.state, &shape, &w.qkv, &w.z, &w.a, &w.b, &w.on, ex);
                if gdn_io {
                    observe(Tap::GdnIn { layer: li, gin: &w.gin });
                    observe(Tap::GdnOut { layer: li, gout: &w.gout });
                }
                observe(Tap::GdnState { layer: li, state: &g.state });
                observe(Tap::MixerOut { layer: li, y: &w.on });
                bk.quant(&w.on, t, c.gdn_v_heads * c.gdn_dv, &w.onq, &w.son, &w.mon);
                bk.gemm(&g.out.lins[0], &w.onq, &w.son, &w.mon, t, GemmOut::Accumulate(&w.x));
            }
        }
        observe(Tap::PostMixer { layer: li, x: &w.x });
        bk.rmsnorm_quant(&w.x, &l.ffn_norm, c.eps, t, h, &w.xq, &w.sx, &w.mx);
        m.project(&l.gu, &w.xq, &w.sx, &w.mx, t, &[&w.g, &w.u]);
        bk.silu_mul_quant(&w.g, &w.u, t, c.intermediate, &w.hq, &w.sh, &w.mh);
        bk.gemm(&l.down.lins[0], &w.hq, &w.sh, &w.mh, t, GemmOut::Accumulate(&w.x));
    }
    observe(Tap::Residual { layer: m.layers.len(), x: &w.x });
    if m.all_logits {
        bk.rmsnorm_quant(&w.x, &m.final_norm, c.eps, t, h, &w.xq, &w.sx, &w.mx);
        bk.gemm(&m.lm_head.lins[0], &w.xq, &w.sx, &w.mx, t, GemmOut::Write(&w.logits));
    } else {
        let last = unsafe { Buf::<f32>::view(w.x.ptr_at((t - 1) * h), h) };
        bk.rmsnorm_quant(&last, &m.final_norm, c.eps, 1, h, &w.xq, &w.sx, &w.mx);
        bk.gemm(&m.lm_head.lins[0], &w.xq, &w.sx, &w.mx, 1, GemmOut::Write(&w.logits));
    }
    observe(Tap::Logits { logits: &w.logits });
}
