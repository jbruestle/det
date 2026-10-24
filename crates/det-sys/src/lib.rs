//! Hand-written FFI to the det GPU runtimes: `det/kernels/metal/det_metal.h` (feature `metal`) or
//! `det/kernels/cuda/det_cuda.h` (feature `cuda`), the one `build.rs` compiles and links statically.
//! Both runtimes export the same C API; the handles for external code enqueuing its own kernels differ.

#![allow(non_camel_case_types)]
#![cfg(any(all(feature = "metal", target_os = "macos"), feature = "cuda"))]

use std::ffi::{c_char, c_int, c_void};

pub type Stream = *mut c_void;

#[repr(C)]
pub struct QLinear {
    pub wq: *const u8,
    pub c: *const f32,
    pub m: *const u16,
    pub n: c_int,
    pub k: c_int,
    pub bits: c_int,
}

#[repr(C)]
pub struct AttnLayer {
    pub q_norm: *const f32,
    pub k_norm: *const f32,
    pub v_scale: *const f32,
    pub v_inv_scale: *const f32,
}

#[repr(C)]
pub struct KvCache {
    pub kq: *mut i8,
    pub sk: *mut f32,
    pub vq: *mut i16,
    pub capacity: c_int,
    pub vlo: *mut i8,
    pub vhi: *mut i8,
}

#[repr(C)]
pub struct GdnLayer {
    pub conv: *const f32,
    pub a: *const f32,
    pub dt_bias: *const f32,
    pub norm: *const f32,
}

#[repr(C)]
pub struct GdnState {
    pub conv_state: *mut f32,
    pub s: *mut f32,
}

extern "C" {
    pub fn det_init() -> c_int;
    pub fn det_device_name() -> *const c_char;
    pub fn det_malloc(bytes: usize) -> *mut c_void;
    pub fn det_free(p: *mut c_void);
    pub fn det_allocated_bytes() -> usize;
    pub fn det_sync();
    pub fn det_flush();
    pub fn det_memset(p: *mut c_void, value: c_int, bytes: usize, st: Stream);
    pub fn det_memcpy(dst: *mut c_void, src: *const c_void, bytes: usize, st: Stream);

    pub fn det_rmsnorm_quant(
        x: *const f32,
        w: *const f32,
        eps: f32,
        m: c_int,
        n: c_int,
        y: *mut f32,
        xq: *mut i8,
        sx: *mut f32,
        mx: *mut u16,
        st: Stream,
    );
    pub fn det_quant(
        x: *const f32,
        m: c_int,
        n: c_int,
        xq: *mut i8,
        sx: *mut f32,
        mx: *mut u16,
        st: Stream,
    );
    pub fn det_embed(
        table_bf16: *const u16,
        ids: *const c_int,
        m: c_int,
        h: c_int,
        x: *mut f32,
        st: Stream,
    );
    pub fn det_add(x: *mut f32, y: *const f32, n: i64, st: Stream);
    pub fn det_silu_mul_quant(
        g: *const f32,
        u: *const f32,
        m: c_int,
        n: c_int,
        h: *mut f32,
        hq: *mut i8,
        sh: *mut f32,
        mh: *mut u16,
        st: Stream,
    );
    pub fn det_argmax(logits: *const f32, m: c_int, v: c_int, out: *mut c_int, st: Stream);

    pub fn det_gemm(
        l: *const QLinear,
        xq: *const i8,
        sx: *const f32,
        mx: *const u16,
        m: c_int,
        y: *mut f32,
        st: Stream,
    );
    pub fn det_gemm_scratch_bytes(m: c_int, n: c_int) -> usize;
    pub fn det_gemm_set_scratch(p: *mut c_void, bytes: usize);
    pub fn det_gemm_ex(
        l: *const QLinear,
        xq: *const i8,
        sx: *const f32,
        mx: *const u16,
        m: c_int,
        y: *mut f32,
        accumulate: c_int,
        nparts: c_int,
        offs: *const c_int,
        dsts: *const *mut f32,
        st: Stream,
    );

    pub fn det_attention(
        l: *const AttnLayer,
        kv: *const KvCache,
        qg: *const f32,
        k: *const f32,
        v: *const f32,
        rope_cos: *const f32,
        rope_sin: *const f32,
        rot: c_int,
        eps: f32,
        m: c_int,
        pos0: c_int,
        n_h: c_int,
        n_kv: c_int,
        hd: c_int,
        o: *mut f32,
        scratch: *mut c_void,
        st: Stream,
    );
    pub fn det_attn_scratch_bytes(m: c_int, n_h: c_int, hd: c_int) -> usize;

    pub fn det_gdn(
        l: *const GdnLayer,
        s: *mut GdnState,
        qkv: *const f32,
        z: *const f32,
        a: *const f32,
        b: *const f32,
        eps: f32,
        m: c_int,
        n_k: c_int,
        n_v: c_int,
        dk: c_int,
        dv: c_int,
        on: *mut f32,
        scratch: *mut c_void,
        st: Stream,
    );
    pub fn det_gdn_ex(
        l: *const GdnLayer,
        s: *mut GdnState,
        qkv: *const f32,
        z: *const f32,
        a: *const f32,
        b: *const f32,
        eps: f32,
        m: c_int,
        n_k: c_int,
        n_v: c_int,
        dk: c_int,
        dv: c_int,
        on: *mut f32,
        scratch: *mut c_void,
        gin: *mut f32,
        gout: *mut f32,
        st: Stream,
    );
    pub fn det_gdn_scratch_bytes(m: c_int, n_k: c_int, n_v: c_int, dk: c_int, dv: c_int) -> usize;

    pub fn det_sample(
        logits: *const f32,
        m: c_int,
        v: c_int,
        inv_temp: f32,
        z: *const u64,
        out: *mut c_int,
        st: Stream,
    );
    pub fn det_sample_topkp(
        logits: *const f32,
        m: c_int,
        v: c_int,
        inv_temp: f32,
        top_k: c_int,
        top_p: f32,
        z: *const u64,
        out: *mut c_int,
        scratch: *mut c_void,
        st: Stream,
    );
    pub fn det_sample_scratch_bytes(m: c_int, v: c_int) -> usize;

    pub fn det_test_prim(op: c_int, x: *const f32, m: c_int, n: c_int, out: *mut f32, st: Stream);
}

#[cfg(feature = "metal")]
extern "C" {
    pub fn det_metal_device() -> *mut c_void;
    pub fn det_metal_queue() -> *mut c_void;
    pub fn det_metal_buffer(p: *const c_void, offset: *mut usize) -> *mut c_void;
}

#[cfg(feature = "cuda")]
extern "C" {
    pub fn det_cuda_stream() -> *mut c_void;
}
