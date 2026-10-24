//! NEON forms of the spec's fp32 primitives (`spec/prim.rs`, op for op; no FMA, no estimates) and the
//! integer dot products the kernels are built on.
use crate::spec::prim::*;
use std::arch::aarch64::*;
use std::arch::asm;

/// `acc + Σ s8·s8` over each lane's 4 byte pairs (`sdot`, exact).
#[inline(always)]
pub unsafe fn sdot(acc: int32x4_t, a: int8x16_t, b: int8x16_t) -> int32x4_t {
    vdotq_s32(acc, a, b)
}

/// `acc + A·Bᵀ` for the 2×8 s8 matrices `a` and `b` (`smmla`, exact): lanes `[a0·b0, a0·b1, a1·b0, a1·b1]`.
#[inline]
#[target_feature(enable = "i8mm")]
pub unsafe fn smmla(mut acc: int32x4_t, a: int8x16_t, b: int8x16_t) -> int32x4_t {
    asm!("smmla {0:v}.4s, {1:v}.16b, {2:v}.16b", inout(vreg) acc, in(vreg) a, in(vreg) b, options(pure, nomem, nostack));
    acc
}

/// `exp_f` on 4 lanes.
#[inline(always)]
pub unsafe fn exp4(x: float32x4_t) -> float32x4_t {
    let x = vminq_f32(x, vdupq_n_f32(EXP_HI));
    let under = vcltq_f32(x, vdupq_n_f32(EXP_LO));
    let n = vrndmq_f32(vaddq_f32(
        vmulq_f32(x, vdupq_n_f32(LOG2E)),
        vdupq_n_f32(0.5),
    ));
    let r = vsubq_f32(x, vmulq_f32(n, vdupq_n_f32(EXP_C1)));
    let r = vsubq_f32(r, vmulq_f32(n, vdupq_n_f32(EXP_C2)));
    let z = vmulq_f32(r, r);
    let mut y = vdupq_n_f32(EXP_P[0]);
    for c in &EXP_P[1..] {
        y = vaddq_f32(vmulq_f32(y, r), vdupq_n_f32(*c));
    }
    y = vaddq_f32(vmulq_f32(y, z), r);
    y = vaddq_f32(y, vdupq_n_f32(1.0));
    let e = vshlq_n_s32::<23>(vaddq_s32(vcvtq_s32_f32(n), vdupq_n_s32(127)));
    let y = vmulq_f32(y, vreinterpretq_f32_s32(e));
    vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(y), under))
}

/// `sigmoid_f` on 4 lanes.
#[inline(always)]
pub unsafe fn sigmoid4(x: float32x4_t) -> float32x4_t {
    let lo = vdupq_n_f32(SIGMOID_LO);
    let under = vcltq_f32(x, lo);
    let xs = vmaxq_f32(x, lo);
    let one = vdupq_n_f32(1.0);
    let r = vdivq_f32(one, vaddq_f32(one, exp4(vnegq_f32(xs))));
    vbslq_f32(under, exp4(x), r)
}

/// `silu_f` on 4 lanes.
#[inline(always)]
pub unsafe fn silu4(x: float32x4_t) -> float32x4_t {
    vmulq_f32(x, sigmoid4(x))
}
