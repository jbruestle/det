//! The op set a backend implements. A network's forward pass is written once against this trait;
//! `cpu::Cpu` evaluates it with the spec functions, `metal::Metal` with the GPU kernels.
#![allow(clippy::too_many_arguments)]

use crate::gguf::{Gguf, TensorInfo};
use std::marker::PhantomData;
use std::sync::Arc;

/// A buffer owned by a backend: heap memory for the CPU backend, device memory for the GPU backends.
/// Only the owning backend touches its contents (`Backend::read` / `write` move data across); ops
/// write through `as_mut_ptr` while the forward pass holds shared references, which is sound as long
/// as no two ops run at once (backends serialize their work).
pub struct Buf<T> {
    ptr: *mut T,
    len: usize,
    owner: Owner,
    _p: PhantomData<T>,
}

enum Owner {
    Host,
    Device(unsafe fn(*mut u8)),
    View,
}

unsafe impl<T: Send> Send for Buf<T> {}
unsafe impl<T: Sync> Sync for Buf<T> {}

impl<T: Copy> Buf<T> {
    /// Wraps device memory owned elsewhere; `release` frees it on drop.
    ///
    /// # Safety
    /// `ptr` must stay valid for `len` elements until `release` runs.
    pub unsafe fn from_device(ptr: *mut T, len: usize, release: unsafe fn(*mut u8)) -> Self {
        Buf {
            ptr,
            len,
            owner: Owner::Device(release),
            _p: PhantomData,
        }
    }
    /// A non-owning window into another buffer's memory.
    ///
    /// # Safety
    /// The memory must outlive the view and hold `len` elements.
    pub unsafe fn view(ptr: *mut T, len: usize) -> Self {
        Buf {
            ptr,
            len,
            owner: Owner::View,
            _p: PhantomData,
        }
    }
    /// A zeroed host buffer, 64-byte aligned.
    pub fn new_host(len: usize) -> Self {
        let layout = host_layout::<T>(len);
        let ptr = if layout.size() == 0 {
            layout.align() as *mut T
        } else {
            unsafe { std::alloc::alloc_zeroed(layout) as *mut T }
        };
        assert!(
            !ptr.is_null(),
            "allocation of {} bytes failed",
            layout.size()
        );
        Buf {
            ptr,
            len,
            owner: Owner::Host,
            _p: PhantomData,
        }
    }
    pub fn host_from(data: &[T]) -> Self {
        let b = Self::new_host(data.len());
        b.host_mut().copy_from_slice(data);
        b
    }
    fn is_host(&self) -> bool {
        matches!(self.owner, Owner::Host | Owner::View)
    }
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    pub fn bytes(&self) -> usize {
        self.len * std::mem::size_of::<T>()
    }
    pub fn as_ptr(&self) -> *const T {
        self.ptr
    }
    pub fn as_mut_ptr(&self) -> *mut T {
        self.ptr
    }
    pub fn ptr_at(&self, i: usize) -> *mut T {
        assert!(i <= self.len);
        unsafe { self.ptr.add(i) }
    }
    /// The contents of a host buffer (the CPU backend's); device buffers go through `Backend::read`.
    pub fn host(&self) -> &[T] {
        assert!(self.is_host(), "host access to a device buffer");
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
    #[allow(clippy::mut_from_ref)]
    pub fn host_mut(&self) -> &mut [T] {
        assert!(self.is_host(), "host access to a device buffer");
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

fn host_layout<T>(len: usize) -> std::alloc::Layout {
    std::alloc::Layout::array::<T>(len)
        .unwrap()
        .align_to(64)
        .unwrap()
        .pad_to_align()
}

impl<T> Drop for Buf<T> {
    fn drop(&mut self) {
        match self.owner {
            Owner::Host => {
                let layout = host_layout::<T>(self.len);
                if layout.size() != 0 {
                    unsafe { std::alloc::dealloc(self.ptr as *mut u8, layout) }
                }
            }
            Owner::Device(f) => unsafe { f(self.ptr as *mut u8) },
            Owner::View => {}
        }
    }
}

/// Where a GEMM writes: a fresh output, a residual accumulation, or a fused projection routed into
/// its part buffers (part i gets columns `[offs[i], offs[i+1])`).
pub enum GemmOut<'a> {
    Write(&'a Buf<f32>),
    Accumulate(&'a Buf<f32>),
    Parts(&'a [&'a Buf<f32>]),
}

pub struct AttnLayer<'a> {
    pub q_norm: &'a Buf<f32>,
    pub k_norm: &'a Buf<f32>,
    pub v_scale: &'a Buf<f32>,
    pub v_inv_scale: &'a Buf<f32>,
    pub rope_cos: &'a Buf<f32>,
    pub rope_sin: &'a Buf<f32>,
}

/// KV cache of one attention layer: `kq[pos][n_kv][hd]` int8, `sk[pos][n_kv]`, `vq[pos][n_kv][hd]` int16.
pub struct KvCache {
    pub kq: Buf<i8>,
    pub sk: Buf<f32>,
    pub vq: Buf<i16>,
    pub capacity: usize,
    /// Metal keeps V as two int8 planes for the prefill attention path.
    pub vlo: Option<Buf<i8>>,
    pub vhi: Option<Buf<i8>>,
}

pub struct GdnLayer<'a> {
    pub conv: &'a Buf<f32>,
    pub a: &'a Buf<f32>,
    pub dt_bias: &'a Buf<f32>,
    pub norm: &'a Buf<f32>,
}

/// GDN state of one layer: the last `conv_k - 1` raw inputs (oldest first) and `S[n_v][dk][dv]`.
pub struct GdnState {
    pub conv_state: Buf<f32>,
    pub s: Buf<f32>,
}

/// Mixer-internal values a hook may ask for: `gin[M][n_k][256 + 130·rep]` (per k head: q, k, v[rep],
/// beta[rep], decay[rep]) and `gout[M][n_v][dv]` (the mixer output before the gated norm).
pub struct GdnExport<'a> {
    pub gin: Option<&'a Buf<f32>>,
    pub gout: Option<&'a Buf<f32>>,
}

pub struct Shape {
    pub m: usize,
    pub pos0: usize,
    pub n_h: usize,
    pub n_kv: usize,
    pub hd: usize,
    pub rot: usize,
    pub eps: f32,
}

pub struct GdnShape {
    pub m: usize,
    pub n_k: usize,
    pub n_v: usize,
    pub dk: usize,
    pub dv: usize,
    pub eps: f32,
}

pub trait Backend: Sized + crate::hooks::Reader {
    type Linear;
    /// Whether the KV cache carries the `vlo` / `vhi` planes this backend's attention reads.
    const V_PLANES: bool = false;

    fn name(&self) -> String;
    /// A zeroed buffer.
    fn alloc<T: Copy>(&self, len: usize) -> Buf<T>;
    fn upload<T: Copy>(&self, data: &[T]) -> Buf<T> {
        let b = self.alloc(data.len());
        self.write(&b, 0, data);
        b
    }
    /// Copies `data` into the buffer at `offset` (after the pending ops that may use it).
    fn write<T: Copy>(&self, b: &Buf<T>, offset: usize, data: &[T]);
    /// Copies the buffer's contents from `offset` into `out` (after the pending ops that produce it).
    fn read<T: Copy>(&self, b: &Buf<T>, offset: usize, out: &mut [T]);
    fn zero<T: Copy>(&self, b: &Buf<T>);
    fn download<T: Copy>(&self, b: &Buf<T>) -> Vec<T> {
        let mut v = vec![unsafe { std::mem::zeroed() }; b.len()];
        self.read(b, 0, &mut v);
        v
    }
    /// Waits for every op issued so far.
    fn sync(&self);

    /// A quantized linear from one or more tensors fused along N (all parts share K and type).
    fn linear(&self, g: &Arc<Gguf>, parts: &[&TensorInfo]) -> Self::Linear;
    fn linear_widths(lin: &Self::Linear) -> &[usize];

    /// `y = rmsnorm(x, w, eps)` per row (`m` rows of `n`), quantized to `xq`, `sx`, `mx` (`y` is not kept).
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
    );
    fn quant(&self, x: &Buf<f32>, m: usize, n: usize, xq: &Buf<i8>, sx: &Buf<f32>, mx: &Buf<u16>);
    fn gemm(
        &self,
        lin: &Self::Linear,
        xq: &Buf<i8>,
        sx: &Buf<f32>,
        mx: &Buf<u16>,
        m: usize,
        out: GemmOut,
    );
    /// `h = silu(g) * u`, quantized.
    fn silu_mul_quant(
        &self,
        g: &Buf<f32>,
        u: &Buf<f32>,
        m: usize,
        n: usize,
        hq: &Buf<i8>,
        sh: &Buf<f32>,
        mh: &Buf<u16>,
    );
    /// Full attention for `m` new tokens at `pos0..`: writes the KV rows, computes every new token against
    /// all positions up to its own, applies the output gate and writes `o[m][n_h·hd]`.
    fn attention(
        &self,
        l: &AttnLayer,
        kv: &KvCache,
        s: &Shape,
        qg: &Buf<f32>,
        k: &Buf<f32>,
        v: &Buf<f32>,
        o: &Buf<f32>,
    );
    /// Gated DeltaNet over `m` tokens in order; writes `on[m][n_v·dv]` after the gated norm.
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
    );
    fn argmax(&self, logits: &Buf<f32>, v: usize) -> i32;
    /// Sampling with the raw splitmix64 draw `z` (`top_k` 0 and `top_p` ≥ 1 disable the filters).
    fn sample(
        &self,
        logits: &Buf<f32>,
        v: usize,
        inv_temp: f32,
        top_k: usize,
        top_p: f32,
        z: u64,
    ) -> i32;
}
