//! The Metal backend: unified-memory buffers and the kernels of `det/kernels/metal` through `det-sys`.

use crate::backend::*;
use crate::gguf::{Gguf, TensorInfo};
use crate::spec;
use det_sys as ffi;
use rayon::prelude::*;
use std::cell::RefCell;
use std::ffi::{c_int, c_void, CStr};
use std::ptr::null_mut;
use std::sync::Arc;

const ST: ffi::Stream = null_mut();
pub const TILE_ROWS: usize = 64;

pub struct Metal {
    gemm_scratch: RefCell<Option<Buf<u8>>>,
    sample_scratch: RefCell<Option<Buf<u8>>>,
    attn_scratch: RefCell<Option<Buf<u8>>>,
    gdn_scratch: RefCell<Option<Buf<u8>>>,
    pick: Buf<c_int>,
    z: Buf<u64>,
}

pub struct MetalLinear {
    wq: Buf<u8>,
    c: Buf<f32>,
    m: Buf<u16>,
    widths: Vec<usize>,
    n: usize,
    k: usize,
    bits: u32,
}

unsafe fn release(p: *mut u8) {
    ffi::det_free(p as *mut c_void);
}

fn device_buf<T: Copy>(len: usize) -> Buf<T> {
    let bytes = (len * std::mem::size_of::<T>()).max(16);
    let p = unsafe { ffi::det_malloc(bytes) };
    assert!(!p.is_null(), "det_malloc({bytes}) failed");
    unsafe { Buf::from_device(p as *mut T, len, release) }
}

/// The unified-memory contents of a device buffer (valid once the GPU work on it is done).
fn mem<'a, T: Copy>(b: &Buf<T>, offset: usize, len: usize) -> &'a mut [T] {
    assert!(offset + len <= b.len());
    unsafe { std::slice::from_raw_parts_mut(b.ptr_at(offset), len) }
}

impl crate::hooks::Reader for Metal {
    unsafe fn read_bytes(&self, src: *const u8, bytes: usize, out: &mut [u8]) {
        unsafe { ffi::det_sync() };
        out.copy_from_slice(unsafe { std::slice::from_raw_parts(src, bytes) });
    }
}

impl Metal {
    pub fn new() -> Self {
        assert_eq!(
            unsafe { ffi::det_init() },
            0,
            "Metal runtime failed to initialize"
        );
        Metal {
            gemm_scratch: RefCell::new(None),
            sample_scratch: RefCell::new(None),
            attn_scratch: RefCell::new(None),
            gdn_scratch: RefCell::new(None),
            pick: device_buf(1),
            z: device_buf(1),
        }
    }

    pub fn allocated_bytes() -> usize {
        unsafe { ffi::det_allocated_bytes() }
    }

    fn scratch(slot: &RefCell<Option<Buf<u8>>>, bytes: usize) -> *mut c_void {
        let mut s = slot.borrow_mut();
        if s.as_ref().is_none_or(|b| b.len() < bytes) {
            *s = Some(device_buf(bytes.max(256)));
        }
        s.as_ref().unwrap().as_mut_ptr() as *mut c_void
    }
}

impl Default for Metal {
    fn default() -> Self {
        Self::new()
    }
}

/// The tiled code layout `gemm.metal` streams: `[ceil(N/64)][K/128][64 rows][4 · bytes_per_32]`, rows
/// past N zero, from the row-major packed codes of row `r` given by `codes(r)`.
pub fn tile_codes<'a>(
    bits: u32,
    k: usize,
    n: usize,
    codes: &(dyn Fn(usize) -> &'a [u8] + Sync),
    out: &mut [u8],
) {
    let g = k / 128;
    let gb = 4 * spec::bytes_per_32(bits);
    let tile = g * TILE_ROWS * gb;
    assert_eq!(out.len(), n.div_ceil(TILE_ROWS) * tile);
    out.par_chunks_mut(tile).enumerate().for_each(|(tn, t)| {
        t.fill(0);
        for ri in 0..TILE_ROWS {
            let r = tn * TILE_ROWS + ri;
            if r >= n {
                break;
            }
            let src = codes(r);
            for gi in 0..g {
                t[(gi * TILE_ROWS + ri) * gb..(gi * TILE_ROWS + ri + 1) * gb]
                    .copy_from_slice(&src[gi * gb..(gi + 1) * gb]);
            }
        }
    });
}

impl Backend for Metal {
    type Linear = MetalLinear;
    const V_PLANES: bool = true;

    fn name(&self) -> String {
        unsafe {
            CStr::from_ptr(ffi::det_device_name())
                .to_string_lossy()
                .into_owned()
        }
    }
    fn alloc<T: Copy>(&self, len: usize) -> Buf<T> {
        device_buf(len)
    }
    fn write<T: Copy>(&self, b: &Buf<T>, offset: usize, data: &[T]) {
        self.sync();
        mem(b, offset, data.len()).copy_from_slice(data);
    }
    fn read<T: Copy>(&self, b: &Buf<T>, offset: usize, out: &mut [T]) {
        self.sync();
        out.copy_from_slice(mem(b, offset, out.len()));
    }
    fn zero<T: Copy>(&self, b: &Buf<T>) {
        unsafe { ffi::det_memset(b.as_mut_ptr() as *mut c_void, 0, b.bytes(), ST) }
    }
    fn sync(&self) {
        unsafe { ffi::det_sync() }
    }

    fn linear(&self, g: &Arc<Gguf>, parts: &[&TensorInfo]) -> MetalLinear {
        let cpu = crate::cpu::linear_parts(g, parts);
        let (bits, k) = (cpu.bits, cpu.k);
        let n = cpu.widths.iter().sum::<usize>();
        let nc = k / 32 * spec::bytes_per_32(bits);
        let wq: Buf<u8> = device_buf(
            n.div_ceil(TILE_ROWS) * (k / 128) * TILE_ROWS * 4 * spec::bytes_per_32(bits),
        );
        tile_codes(bits, k, n, &|r| &cpu.row(r)[..nc], mem(&wq, 0, wq.len()));
        let (mut c, mut m) = (Vec::with_capacity(n), Vec::with_capacity(n * k / 128));
        for r in 0..n {
            let row = spec::det_row(bits, k, cpu.row(r));
            c.push(row.c);
            m.extend_from_slice(&row.m);
        }
        MetalLinear {
            wq,
            c: self.upload(&c),
            m: self.upload(&m),
            widths: cpu.widths.clone(),
            n,
            k,
            bits,
        }
    }
    fn linear_widths(lin: &MetalLinear) -> &[usize] {
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
        unsafe {
            ffi::det_rmsnorm_quant(
                x.as_ptr(),
                w.as_ptr(),
                eps,
                m as c_int,
                n as c_int,
                null_mut(),
                xq.as_mut_ptr(),
                sx.as_mut_ptr(),
                mx.as_mut_ptr(),
                ST,
            )
        }
    }

    fn quant(&self, x: &Buf<f32>, m: usize, n: usize, xq: &Buf<i8>, sx: &Buf<f32>, mx: &Buf<u16>) {
        unsafe {
            ffi::det_quant(
                x.as_ptr(),
                m as c_int,
                n as c_int,
                xq.as_mut_ptr(),
                sx.as_mut_ptr(),
                mx.as_mut_ptr(),
                ST,
            )
        }
    }

    fn gemm(
        &self,
        lin: &MetalLinear,
        xq: &Buf<i8>,
        sx: &Buf<f32>,
        mx: &Buf<u16>,
        m: usize,
        out: GemmOut,
    ) {
        let q = ffi::QLinear {
            wq: lin.wq.as_ptr(),
            c: lin.c.as_ptr(),
            m: lin.m.as_ptr(),
            n: lin.n as c_int,
            k: lin.k as c_int,
            bits: lin.bits as c_int,
        };
        if m <= 8 {
            let need = unsafe { ffi::det_gemm_scratch_bytes(m as c_int, lin.n as c_int) };
            let mut s = self.gemm_scratch.borrow_mut();
            if s.as_ref().is_none_or(|b| b.len() < need) {
                unsafe { ffi::det_gemm_set_scratch(null_mut(), 0) };
                let b: Buf<u8> = device_buf(need);
                unsafe { ffi::det_gemm_set_scratch(b.as_mut_ptr() as *mut c_void, need) };
                *s = Some(b);
            }
        }
        unsafe {
            match out {
                GemmOut::Write(y) => ffi::det_gemm_ex(
                    &q,
                    xq.as_ptr(),
                    sx.as_ptr(),
                    mx.as_ptr(),
                    m as c_int,
                    y.as_mut_ptr(),
                    0,
                    0,
                    std::ptr::null(),
                    std::ptr::null(),
                    ST,
                ),
                GemmOut::Accumulate(y) => ffi::det_gemm_ex(
                    &q,
                    xq.as_ptr(),
                    sx.as_ptr(),
                    mx.as_ptr(),
                    m as c_int,
                    y.as_mut_ptr(),
                    1,
                    0,
                    std::ptr::null(),
                    std::ptr::null(),
                    ST,
                ),
                GemmOut::Parts(dsts) => {
                    assert_eq!(dsts.len(), lin.widths.len());
                    let mut offs = Vec::with_capacity(dsts.len());
                    let mut acc = 0;
                    for w in &lin.widths {
                        offs.push(acc as c_int);
                        acc += w;
                    }
                    let ptrs: Vec<*mut f32> = dsts.iter().map(|d| d.as_mut_ptr()).collect();
                    ffi::det_gemm_ex(
                        &q,
                        xq.as_ptr(),
                        sx.as_ptr(),
                        mx.as_ptr(),
                        m as c_int,
                        null_mut(),
                        0,
                        dsts.len() as c_int,
                        offs.as_ptr(),
                        ptrs.as_ptr(),
                        ST,
                    )
                }
            }
        }
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
        unsafe {
            ffi::det_silu_mul_quant(
                g.as_ptr(),
                u.as_ptr(),
                m as c_int,
                n as c_int,
                null_mut(),
                hq.as_mut_ptr(),
                sh.as_mut_ptr(),
                mh.as_mut_ptr(),
                ST,
            )
        }
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
        let layer = ffi::AttnLayer {
            q_norm: l.q_norm.as_ptr(),
            k_norm: l.k_norm.as_ptr(),
            v_scale: l.v_scale.as_ptr(),
            v_inv_scale: l.v_inv_scale.as_ptr(),
        };
        let cache = ffi::KvCache {
            kq: kv.kq.as_mut_ptr(),
            sk: kv.sk.as_mut_ptr(),
            vq: kv.vq.as_mut_ptr(),
            capacity: kv.capacity as c_int,
            vlo: kv.vlo.as_ref().map_or(null_mut(), |b| b.as_mut_ptr()),
            vhi: kv.vhi.as_ref().map_or(null_mut(), |b| b.as_mut_ptr()),
        };
        let bytes =
            unsafe { ffi::det_attn_scratch_bytes(s.m as c_int, s.n_h as c_int, s.hd as c_int) };
        let scratch = Self::scratch(&self.attn_scratch, bytes);
        unsafe {
            ffi::det_attention(
                &layer,
                &cache,
                qg.as_ptr(),
                k.as_ptr(),
                v.as_ptr(),
                l.rope_cos.as_ptr(),
                l.rope_sin.as_ptr(),
                s.rot as c_int,
                s.eps,
                s.m as c_int,
                s.pos0 as c_int,
                s.n_h as c_int,
                s.n_kv as c_int,
                s.hd as c_int,
                o.as_mut_ptr(),
                scratch,
                ST,
            )
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
        let layer = ffi::GdnLayer {
            conv: l.conv.as_ptr(),
            a: l.a.as_ptr(),
            dt_bias: l.dt_bias.as_ptr(),
            norm: l.norm.as_ptr(),
        };
        let mut state = ffi::GdnState {
            conv_state: st.conv_state.as_mut_ptr(),
            s: st.s.as_mut_ptr(),
        };
        let bytes = unsafe {
            ffi::det_gdn_scratch_bytes(
                s.m as c_int,
                s.n_k as c_int,
                s.n_v as c_int,
                s.dk as c_int,
                s.dv as c_int,
            )
        };
        let scratch = Self::scratch(&self.gdn_scratch, bytes);
        let (gin, gout) = (
            ex.gin.map_or(null_mut(), |g| g.as_mut_ptr()),
            ex.gout.map_or(null_mut(), |g| g.as_mut_ptr()),
        );
        unsafe {
            ffi::det_gdn_ex(
                &layer,
                &mut state,
                qkv.as_ptr(),
                z.as_ptr(),
                a.as_ptr(),
                b.as_ptr(),
                s.eps,
                s.m as c_int,
                s.n_k as c_int,
                s.n_v as c_int,
                s.dk as c_int,
                s.dv as c_int,
                on.as_mut_ptr(),
                scratch,
                gin,
                gout,
                ST,
            )
        }
    }

    fn argmax(&self, logits: &Buf<f32>, v: usize) -> i32 {
        unsafe { ffi::det_argmax(logits.as_ptr(), 1, v as c_int, self.pick.as_mut_ptr(), ST) };
        self.download(&self.pick)[0]
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
        self.write(&self.z, 0, &[z]);
        if top_k == 0 && top_p >= 1.0 {
            unsafe {
                ffi::det_sample(
                    logits.as_ptr(),
                    1,
                    v as c_int,
                    inv_temp,
                    self.z.as_ptr(),
                    self.pick.as_mut_ptr(),
                    ST,
                )
            };
        } else {
            let bytes = unsafe { ffi::det_sample_scratch_bytes(1, v as c_int) };
            let scratch = Self::scratch(&self.sample_scratch, bytes);
            unsafe {
                ffi::det_sample_topkp(
                    logits.as_ptr(),
                    1,
                    v as c_int,
                    inv_temp,
                    top_k as c_int,
                    top_p,
                    self.z.as_ptr(),
                    self.pick.as_mut_ptr(),
                    scratch,
                    ST,
                )
            };
        }
        self.download(&self.pick)[0]
    }
}

/// The elementwise primitive test hook (`det_test_prim`): `op` as in `det_metal.h`.
pub fn test_prim(op: i32, x: &[f32], m: usize, n: usize) -> Vec<f32> {
    let b = Metal::new();
    let xb: Buf<f32> = b.upload(x);
    let out: Buf<f32> = device_buf(x.len().max(m));
    unsafe {
        ffi::det_test_prim(
            op,
            xb.as_ptr(),
            m as c_int,
            n as c_int,
            out.as_mut_ptr(),
            ST,
        )
    };
    b.download(&out)
}
