//! Observation points of a forward pass. An observer sees the values a run produces, in the backend's
//! buffers, as they are produced: the residual stream before every layer, the attention inputs and
//! outputs, the GDN mixer's inputs, outputs and state, the logits and the sampled token.

use crate::backend::{Buf, GdnState, KvCache};

pub enum Tap<'a> {
    /// Residual stream before layer `layer` (`layer == n_layers`: before the final norm): `[t][hidden]`.
    Residual { layer: usize, x: &'a Buf<f32> },
    /// The token mixer's output as fed to its out projection (attention: the gated output; GDN: after
    /// the gated norm): `[t][n_h·hd]` or `[t][n_v·dv]`.
    MixerOut { layer: usize, y: &'a Buf<f32> },
    /// Residual stream after the token mixer of `layer`, before its MLP: `[t][hidden]`.
    PostMixer { layer: usize, x: &'a Buf<f32> },
    /// Attention value projections before quantization: `[t][n_kv][hd]`.
    AttnV { layer: usize, v: &'a Buf<f32> },
    /// The KV cache after the rows of this batch were written.
    Kv { layer: usize, kv: &'a KvCache },
    /// Gated attention output: `[t][n_h][hd]`.
    AttnOut { layer: usize, o: &'a Buf<f32> },
    /// GDN mixer inputs per k head: `[t][n_k][2·dk + rep·dv + 2·rep]` (q, k, then v, beta, decay of its rep v heads).
    GdnIn { layer: usize, gin: &'a Buf<f32> },
    /// GDN mixer output before the gated norm: `[t][n_v][dv]`.
    GdnOut { layer: usize, gout: &'a Buf<f32> },
    /// GDN state after the batch: `S[n_v][dk][dv]` and the conv window.
    GdnState { layer: usize, state: &'a GdnState },
    /// Logits of the last position of the batch: `[vocab]`.
    Logits { logits: &'a Buf<f32> },
    /// The token sampled from the last logits and the rng state after the draw.
    Token { id: i32, rng: u64 },
}

/// Context of an observation: the batch covers positions `pos0..pos0 + t`. `read` copies a tap's
/// buffer (or part of it) to the host after the backend's pending work; `sync` only waits, for
/// observers that enqueue work of their own on the backend's queue.
pub struct Ctx<'a> {
    pub pos0: usize,
    pub t: usize,
    pub sync: &'a dyn Fn(),
    pub reader: &'a dyn Reader,
}

/// Host access to a backend's buffers, type-erased.
pub trait Reader {
    /// # Safety
    /// `src` must point into one of this backend's buffers with at least `bytes` bytes left.
    unsafe fn read_bytes(&self, src: *const u8, bytes: usize, out: &mut [u8]);
}

impl Ctx<'_> {
    pub fn read<T: Copy>(&self, b: &Buf<T>, offset: usize, n: usize) -> Vec<T> {
        let mut v: Vec<T> = vec![unsafe { std::mem::zeroed() }; n];
        let bytes = n * std::mem::size_of::<T>();
        let out = unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr() as *mut u8, bytes) };
        unsafe { self.reader.read_bytes(b.ptr_at(offset) as *const u8, bytes, out) };
        v
    }
}

pub trait Observer {
    /// Whether the GDN mixer should export its inputs and outputs (`GdnIn`, `GdnOut`), which costs
    /// extra writes in the kernel.
    fn wants_gdn_io(&self) -> bool {
        false
    }
    fn observe(&mut self, ctx: &Ctx, tap: Tap);
}
