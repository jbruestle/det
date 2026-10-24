//! The AArch64 backend: the spec on NEON with the `sdot` dot products (FEAT_DotProd, required) and
//! `smmla` (FEAT_I8MM, where the CPU has it), threaded with rayon over rows, heads and tokens.
//! Weights are repacked at load into the strips `gemm.rs` streams; sampling is the reference backend's.
#![allow(
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::assign_op_pattern,
    clippy::manual_clamp,
    clippy::missing_safety_doc
)]

mod attn;
mod gdn;
mod gemm;
mod glue;
mod simd;

use crate::backend::*;
use crate::cpu::{linear_parts, Cpu};
use crate::gguf::{Gguf, TensorInfo};
pub use gemm::ArmLinear;
use std::arch::is_aarch64_feature_detected;
use std::cell::RefCell;
use std::sync::Arc;

pub struct Arm {
    cpu: Cpu,
    pub i8mm: bool,
    scratch: RefCell<Vec<f32>>,
}

impl crate::hooks::Reader for Arm {
    unsafe fn read_bytes(&self, src: *const u8, bytes: usize, out: &mut [u8]) {
        out.copy_from_slice(unsafe { std::slice::from_raw_parts(src, bytes) });
    }
}

impl Arm {
    pub fn available() -> bool {
        is_aarch64_feature_detected!("dotprod")
    }
    pub fn new() -> Self {
        Self::with_i8mm(is_aarch64_feature_detected!("i8mm"))
    }
    /// The `sdot` GEMM on a CPU that has i8mm (the conformance suites run both).
    pub fn with_i8mm(i8mm: bool) -> Self {
        assert!(Self::available() && (!i8mm || is_aarch64_feature_detected!("i8mm")));
        Arm {
            cpu: Cpu,
            i8mm,
            scratch: RefCell::new(Vec::new()),
        }
    }
}

impl Default for Arm {
    fn default() -> Self {
        Self::new()
    }
}

impl Backend for Arm {
    type Linear = ArmLinear;

    fn name(&self) -> String {
        format!(
            "arm ({}, {} threads)",
            if self.i8mm { "i8mm" } else { "dotprod" },
            rayon::current_num_threads()
        )
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

    fn linear(&self, g: &Arc<Gguf>, parts: &[&TensorInfo]) -> ArmLinear {
        gemm::repack(&linear_parts(g, parts))
    }
    fn linear_widths(lin: &ArmLinear) -> &[usize] {
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
        glue::rmsnorm_quant(x, w, eps, m, n, xq, sx, mx)
    }
    fn quant(&self, x: &Buf<f32>, m: usize, n: usize, xq: &Buf<i8>, sx: &Buf<f32>, mx: &Buf<u16>) {
        glue::quant(x, m, n, xq, sx, mx)
    }
    fn gemm(
        &self,
        lin: &ArmLinear,
        xq: &Buf<i8>,
        sx: &Buf<f32>,
        mx: &Buf<u16>,
        m: usize,
        out: GemmOut,
    ) {
        gemm::gemm(lin, self.i8mm, xq, sx, mx, m, &out)
    }
    fn silu_mul_quant(
        &self,
        g: &Buf<f32>,
        u: &Buf<f32>,
        m: usize,
        n: usize,
        hq: &Buf<i8>,
        sh: &Buf<f32>,
        mh: &Buf<u16>,
    ) {
        glue::silu_mul_quant(g, u, m, n, hq, sh, mh)
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
        attn::attention(l, kv, s, qg, k, v, o)
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
        gdn::gdn(
            &mut self.scratch.borrow_mut(),
            l,
            st,
            s,
            qkv,
            z,
            a,
            b,
            on,
            ex,
        )
    }
    fn argmax(&self, logits: &Buf<f32>, v: usize) -> i32 {
        self.cpu.argmax(logits, v)
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
        self.cpu.sample(logits, v, inv_temp, top_k, top_p, z)
    }
}
