//! The x86-64 backend: the spec on AVX2 and AVX-VNNI (`vpdpbusd` where the CPU has it, `vpmaddubsw`
//! otherwise), threaded with rayon over rows, heads and tokens. Weights are repacked at load into the
//! strips `gemm.rs` streams; sampling is the reference backend's.
#![allow(clippy::needless_range_loop, clippy::too_many_arguments, clippy::assign_op_pattern, clippy::manual_clamp, clippy::missing_safety_doc)]

mod attn;
mod gdn;
mod gemm;
mod glue;
mod simd;

use crate::backend::*;
use crate::cpu::{linear_parts, Cpu};
use crate::gguf::{Gguf, TensorInfo};
pub use gemm::X86Linear;
use std::cell::RefCell;
use std::sync::Arc;

pub struct X86 {
    cpu: Cpu,
    pub vnni: bool,
    scratch: RefCell<Vec<f32>>,
}

impl crate::hooks::Reader for X86 {
    unsafe fn read_bytes(&self, src: *const u8, bytes: usize, out: &mut [u8]) {
        out.copy_from_slice(unsafe { std::slice::from_raw_parts(src, bytes) });
    }
}

impl X86 {
    pub fn available() -> bool {
        is_x86_feature_detected!("avx2")
    }
    /// `DET_X86_VNNI=0` forces the AVX2 dot products on a CPU that has VNNI.
    pub fn new() -> Self {
        assert!(Self::available(), "the x86 backend needs AVX2");
        let vnni = is_x86_feature_detected!("avxvnni") && std::env::var("DET_X86_VNNI").map_or(true, |v| v != "0");
        X86 {
            cpu: Cpu,
            vnni,
            scratch: RefCell::new(Vec::new()),
        }
    }
    pub fn with_vnni(vnni: bool) -> Self {
        assert!(Self::available() && (!vnni || is_x86_feature_detected!("avxvnni")));
        X86 {
            cpu: Cpu,
            vnni,
            scratch: RefCell::new(Vec::new()),
        }
    }
}

impl Default for X86 {
    fn default() -> Self {
        Self::new()
    }
}

impl Backend for X86 {
    type Linear = X86Linear;

    fn name(&self) -> String {
        format!("x86 ({}, {} threads)", if self.vnni { "avx-vnni" } else { "avx2" }, rayon::current_num_threads())
    }
    fn alloc<T: Copy>(&self, len: usize) -> Buf<T> {
        Buf::new_host(len)
    }
    fn write<T: Copy>(&self, b: &Buf<T>, offset: usize, data: &[T]) {
        self.cpu.write(b, offset, data)
    }
    fn read<T: Copy>(&self, b: &Buf<T>, offset: usize, out: &mut [T]) {
        self.cpu.read(b, offset, out)
    }
    fn zero<T: Copy>(&self, b: &Buf<T>) {
        self.cpu.zero(b)
    }
    fn sync(&self) {}

    fn linear(&self, g: &Arc<Gguf>, parts: &[&TensorInfo]) -> X86Linear {
        gemm::repack(&linear_parts(g, parts))
    }
    fn linear_widths(lin: &X86Linear) -> &[usize] {
        &lin.widths
    }

    fn rmsnorm_quant(&self, x: &Buf<f32>, w: &Buf<f32>, eps: f32, m: usize, n: usize, xq: &Buf<i8>, sx: &Buf<f32>, mx: &Buf<u16>) {
        glue::rmsnorm_quant(x, w, eps, m, n, xq, sx, mx)
    }
    fn quant(&self, x: &Buf<f32>, m: usize, n: usize, xq: &Buf<i8>, sx: &Buf<f32>, mx: &Buf<u16>) {
        glue::quant(x, m, n, xq, sx, mx)
    }
    fn gemm(&self, lin: &X86Linear, xq: &Buf<i8>, sx: &Buf<f32>, mx: &Buf<u16>, m: usize, out: GemmOut) {
        gemm::gemm(lin, self.vnni, xq, sx, mx, m, &out)
    }
    fn silu_mul_quant(&self, g: &Buf<f32>, u: &Buf<f32>, m: usize, n: usize, hq: &Buf<i8>, sh: &Buf<f32>, mh: &Buf<u16>) {
        glue::silu_mul_quant(g, u, m, n, hq, sh, mh)
    }
    fn attention(&self, l: &AttnLayer, kv: &KvCache, s: &Shape, qg: &Buf<f32>, k: &Buf<f32>, v: &Buf<f32>, o: &Buf<f32>) {
        attn::attention(self.vnni, l, kv, s, qg, k, v, o)
    }
    fn gdn(&self, l: &GdnLayer, st: &GdnState, s: &GdnShape, qkv: &Buf<f32>, z: &Buf<f32>, a: &Buf<f32>, b: &Buf<f32>, on: &Buf<f32>, ex: GdnExport) {
        gdn::gdn(&mut self.scratch.borrow_mut(), l, st, s, qkv, z, a, b, on, ex)
    }
    fn argmax(&self, logits: &Buf<f32>, v: usize) -> i32 {
        self.cpu.argmax(logits, v)
    }
    fn sample(&self, logits: &Buf<f32>, v: usize, inv_temp: f32, top_k: usize, top_p: f32, z: u64) -> i32 {
        self.cpu.sample(logits, v, inv_temp, top_k, top_p, z)
    }
}
