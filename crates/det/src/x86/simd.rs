//! AVX2 forms of the spec's fp32 primitives (`spec/prim.rs`, op for op; no FMA, no fast math) and the
//! integer dot products the kernels are built on.
use crate::spec::prim::*;
use std::alloc::{alloc_zeroed, dealloc, Layout};
use std::arch::x86_64::*;

/// 64-byte aligned zeroed bytes.
pub struct Aligned {
    ptr: *mut u8,
    len: usize,
}

unsafe impl Send for Aligned {}
unsafe impl Sync for Aligned {}

impl Aligned {
    pub fn zeroed(len: usize) -> Self {
        let layout = Layout::from_size_align(len.max(64), 64).unwrap();
        let ptr = unsafe { alloc_zeroed(layout) };
        assert!(!ptr.is_null(), "allocation of {len} bytes failed");
        Aligned { ptr, len }
    }
    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
    #[allow(clippy::mut_from_ref)]
    pub fn as_mut_slice(&self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
    pub fn as_ptr(&self) -> *const u8 {
        self.ptr
    }
}

impl Drop for Aligned {
    fn drop(&mut self) {
        unsafe {
            dealloc(
                self.ptr,
                Layout::from_size_align(self.len.max(64), 64).unwrap(),
            )
        }
    }
}

/// `acc + Σ u8·s8` over each lane's 4 byte pairs: `vpdpbusd`, or `vpmaddubsw` + `vpmaddwd` without
/// VNNI. The maddubs form saturates its 16-bit pair sums, so it is exact only for `u ≤ 127`: W8 codes
/// (`u = wq + 128 ≤ 255`) go through their two nibbles.
#[inline(always)]
pub unsafe fn dp<const VNNI: bool, const BITS: u32>(
    acc: __m256i,
    u: __m256i,
    x: __m256i,
) -> __m256i {
    if VNNI {
        _mm256_dpbusd_avx_epi32(acc, u, x)
    } else if BITS == 8 {
        let nib = _mm256_set1_epi8(0x0F);
        let lo = _mm256_and_si256(u, nib);
        let hi = _mm256_and_si256(_mm256_srli_epi16::<4>(u), nib);
        let ones = _mm256_set1_epi16(1);
        let a = _mm256_madd_epi16(_mm256_maddubs_epi16(lo, x), ones);
        let b = _mm256_madd_epi16(_mm256_maddubs_epi16(hi, x), ones);
        _mm256_add_epi32(acc, _mm256_add_epi32(a, _mm256_slli_epi32::<4>(b)))
    } else {
        _mm256_add_epi32(
            acc,
            _mm256_madd_epi16(_mm256_maddubs_epi16(u, x), _mm256_set1_epi16(1)),
        )
    }
}

/// `acc + Σ s16·s16` over each lane's pair: `vpdpwssd`, or `vpmaddwd` + add.
#[inline(always)]
pub unsafe fn dp16<const VNNI: bool>(acc: __m256i, a: __m256i, b: __m256i) -> __m256i {
    if VNNI {
        _mm256_dpwssd_avx_epi32(acc, a, b)
    } else {
        _mm256_add_epi32(acc, _mm256_madd_epi16(a, b))
    }
}

#[inline(always)]
pub unsafe fn hsum_epi32(v: __m256i) -> i32 {
    let s = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256::<1>(v));
    let s = _mm_add_epi32(s, _mm_shuffle_epi32::<0b01_00_11_10>(s));
    let s = _mm_add_epi32(s, _mm_shuffle_epi32::<0b10_11_00_01>(s));
    _mm_cvtsi128_si32(s)
}

/// `exp_f` on 8 lanes.
#[inline(always)]
pub unsafe fn exp8(x: __m256) -> __m256 {
    let x = _mm256_min_ps(x, _mm256_set1_ps(EXP_HI));
    let under = _mm256_cmp_ps::<_CMP_LT_OQ>(x, _mm256_set1_ps(EXP_LO));
    let n = _mm256_floor_ps(_mm256_add_ps(
        _mm256_mul_ps(x, _mm256_set1_ps(LOG2E)),
        _mm256_set1_ps(0.5),
    ));
    let r = _mm256_sub_ps(x, _mm256_mul_ps(n, _mm256_set1_ps(EXP_C1)));
    let r = _mm256_sub_ps(r, _mm256_mul_ps(n, _mm256_set1_ps(EXP_C2)));
    let z = _mm256_mul_ps(r, r);
    let mut y = _mm256_set1_ps(EXP_P[0]);
    for c in &EXP_P[1..] {
        y = _mm256_add_ps(_mm256_mul_ps(y, r), _mm256_set1_ps(*c));
    }
    y = _mm256_add_ps(_mm256_mul_ps(y, z), r);
    y = _mm256_add_ps(y, _mm256_set1_ps(1.0));
    let e = _mm256_slli_epi32::<23>(_mm256_add_epi32(
        _mm256_cvtps_epi32(n),
        _mm256_set1_epi32(127),
    ));
    let y = _mm256_mul_ps(y, _mm256_castsi256_ps(e));
    _mm256_andnot_ps(under, y)
}

/// `sigmoid_f` on 8 lanes.
#[inline(always)]
pub unsafe fn sigmoid8(x: __m256) -> __m256 {
    let lo = _mm256_set1_ps(SIGMOID_LO);
    let under = _mm256_cmp_ps::<_CMP_LT_OQ>(x, lo);
    let xs = _mm256_max_ps(x, lo);
    let neg = _mm256_xor_ps(xs, _mm256_set1_ps(-0.0));
    let r = _mm256_div_ps(
        _mm256_set1_ps(1.0),
        _mm256_add_ps(_mm256_set1_ps(1.0), exp8(neg)),
    );
    _mm256_blendv_ps(r, exp8(x), under)
}

/// `silu_f` on 8 lanes.
#[inline(always)]
pub unsafe fn silu8(x: __m256) -> __m256 {
    _mm256_mul_ps(x, sigmoid8(x))
}
