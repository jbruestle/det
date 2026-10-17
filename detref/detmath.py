"""Deterministic fp32 math primitives (the SPEC).

Every function here is defined by an exact sequence of IEEE-754 binary32 operations
(add, sub, mul, div, sqrt, floor, rint, comparisons, int<->float conversion, bit casts).
No FMA, no library transcendentals.  Any implementation (CUDA, Metal, C) must reproduce
these bit-for-bit.  numpy float32 ufuncs are correctly rounded per op, so this file is
both the specification and the golden reference.

Conventions:
  * all tensors are np.float32 unless stated; constants are np.float32
  * `treesum` is the one and only floating-point reduction: a balanced binary tree over
    the input padded with +0.0 to the next power of two.
"""
import numpy as np

f32 = np.float32
i32 = np.int32
u32 = np.uint32

# ---------------------------------------------------------------- constants
LOG2E   = f32(1.44269504088896341)
EXP_C1  = f32(0.693359375)        # ln2 split (Cephes)
EXP_C2  = f32(-2.12194440e-4)
EXP_P   = [f32(1.9875691500E-4), f32(1.3981999507E-3), f32(8.3334519073E-3),
           f32(4.1665795894E-2), f32(1.6666665459E-1), f32(5.0000001201E-1)]
EXP_LO  = f32(-87.0)   # exp(x) := 0 for x < EXP_LO
EXP_HI  = f32(88.0)    # x := min(x, EXP_HI)

LOG_SQRTHF = f32(0.707106781186547524)
LOG_P = [f32(7.0376836292E-2), f32(-1.1514610310E-1), f32(1.1676998740E-1),
         f32(-1.2420140846E-1), f32(1.4249322787E-1), f32(-1.6668057665E-1),
         f32(2.0000714765E-1), f32(-2.4999993993E-1), f32(3.3333331174E-1)]
LOG_Q1 = f32(-2.12194440e-4)
LOG_Q2 = f32(0.693359375)

SOFTPLUS_THRESH = f32(20.0)
SIGMOID_LO = f32(-80.0)

ONE = f32(1.0); HALF = f32(0.5); ZERO = f32(0.0)


def _bits(x):
    return x.view(u32)

def _from_bits(b):
    return b.view(f32)


def exp_f(x):
    """exp(x) for fp32 array x. Cephes-style: 2^n * poly(r)."""
    x = np.asarray(x, dtype=f32)
    x = np.minimum(x, EXP_HI)
    small = x < EXP_LO
    # n = floor(x*log2e + 0.5)
    n = np.floor((x * LOG2E) + HALF)
    r = x - (n * EXP_C1)
    r = r - (n * EXP_C2)
    z = r * r
    y = EXP_P[0]
    y = (y * r) + EXP_P[1]
    y = (y * r) + EXP_P[2]
    y = (y * r) + EXP_P[3]
    y = (y * r) + EXP_P[4]
    y = (y * r) + EXP_P[5]
    y = (y * z) + r
    y = y + ONE
    # scale by 2^n via exponent bits; n in [-125, 127] here
    ni = n.astype(i32)
    scale = _from_bits(((ni + i32(127)) << i32(23)).astype(u32))
    y = y * scale
    y = np.where(small, ZERO, y)
    return y.astype(f32)


def log_f(x):
    """natural log for fp32 array x > 0 (normal). Cephes-style."""
    x = np.asarray(x, dtype=f32)
    b = _bits(x)
    e = ((b >> u32(23)) & u32(0xFF)).astype(i32) - i32(126)   # frexp exponent
    m = _from_bits((b & u32(0x007FFFFF)) | u32(0x3F000000))  # mantissa in [0.5, 1)
    lt = m < LOG_SQRTHF
    e = np.where(lt, e - i32(1), e)
    m = np.where(lt, (m + m) - ONE, m - ONE)
    z = m * m
    y = LOG_P[0]
    for c in LOG_P[1:]:
        y = (y * m) + c
    y = y * m
    y = y * z
    ef = e.astype(f32)
    y = y + (LOG_Q1 * ef)
    y = y + (f32(-0.5) * z)
    r = m + y
    r = r + (LOG_Q2 * ef)
    return r.astype(f32)


SOFTPLUS_LO = f32(-15.0)

def softplus_f(x):
    """softplus: x if x > 20; exp(x) if x < -15; else log(1 + exp(x))."""
    x = np.asarray(x, dtype=f32)
    big = x > SOFTPLUS_THRESH
    lo = x < SOFTPLUS_LO
    xs = np.minimum(x, SOFTPLUS_THRESH)
    y = log_f(ONE + exp_f(xs))
    y = np.where(lo, exp_f(x), y)
    return np.where(big, x, y).astype(f32)


def sigmoid_f(x):
    x = np.asarray(x, dtype=f32)
    lo = x < SIGMOID_LO
    xs = np.maximum(x, SIGMOID_LO)
    y = ONE / (ONE + exp_f(-xs))
    return np.where(lo, exp_f(x), y).astype(f32)


def silu_f(x):
    x = np.asarray(x, dtype=f32)
    return (x * sigmoid_f(x)).astype(f32)


def treesum(x, axis=-1):
    """Balanced binary tree sum along `axis`, padding with +0.0 to a power of two.
    level l: s[i] = s[2i] + s[2i+1]."""
    x = np.asarray(x, dtype=f32)
    x = np.moveaxis(x, axis, -1)
    n = x.shape[-1]
    p = 1
    while p < n:
        p <<= 1
    if p != n:
        pad = np.zeros(x.shape[:-1] + (p - n,), dtype=f32)
        x = np.concatenate([x, pad], axis=-1)
    while x.shape[-1] > 1:
        x = x[..., 0::2] + x[..., 1::2]
    return x[..., 0].astype(f32)


def rsqrt_f(x):
    x = np.asarray(x, dtype=f32)
    return (ONE / np.sqrt(x)).astype(f32)


def rmsnorm(x, w, eps):
    """x[..., n] * (1/sqrt(mean(x^2)+eps)) * w   (w already includes the +1 if zero-centered)."""
    x = np.asarray(x, dtype=f32)
    n = x.shape[-1]
    ss = treesum(x * x)
    mean = ss / f32(n)
    inv = rsqrt_f(mean + f32(eps))
    return ((x * inv[..., None]) * w).astype(f32)


def l2norm(x, eps):
    x = np.asarray(x, dtype=f32)
    ss = treesum(x * x)
    inv = rsqrt_f(ss + f32(eps))
    return (x * inv[..., None]).astype(f32)


def quant_act_i8(x):
    """Per-row symmetric int8: returns (xq int8 [..., K], s fp32 [...])."""
    x = np.asarray(x, dtype=f32)
    amax = np.max(np.abs(x), axis=-1)
    s = amax / f32(127.0)
    s = np.where(amax == ZERO, ONE, s).astype(f32)
    inv = ONE / s
    q = np.rint(x * inv[..., None])
    q = np.clip(q, -127, 127).astype(np.int8)
    return q, s


def quant_act_i8g(x, group=128):
    """Per-row int8 with per-(row, 128-group) integer multipliers (the GEMM activation format).
    Returns (xq int8 [..., K], sx fp32 [...], mx uint16 [..., K/group]).
    effective scale of group g = sx * mx[g];  xq = clamp(rint(x / (sx*mx[g])), -127, 127)."""
    x = np.asarray(x, dtype=f32)
    K = x.shape[-1]; assert K % group == 0, K
    xg = x.reshape(x.shape[:-1] + (K // group, group))
    amax_g = np.max(np.abs(xg), axis=-1)                     # [..., G]
    s_g = amax_g / f32(127.0)
    smax = np.max(s_g, axis=-1)                              # [...]
    sx = np.where(smax == ZERO, ONE, smax / f32(65535.0)).astype(f32)
    mx = np.clip(np.rint(s_g / sx[..., None]), 1, 65535).astype(np.uint16)
    eff = sx[..., None] * mx.astype(f32)
    inv = ONE / eff
    q = np.rint(xg * inv[..., None])
    q = np.clip(q, -127, 127).astype(np.int8).reshape(x.shape)
    return q, sx, mx


# ---------------------------------------------------------------- self test
if __name__ == "__main__":
    xs = np.linspace(-100, 100, 200001).astype(f32)
    e = exp_f(xs); ref = np.exp(xs.astype(np.float64))
    ok = xs >= -87
    rel = np.abs(e[ok] - ref[ok]) / ref[ok]
    print("exp max rel err", rel.max(), "zeros below -87:", np.all(e[~ok] == 0))
    xl = np.concatenate([np.linspace(1e-30, 1, 100001), np.linspace(1, 1e30, 100001)]).astype(f32)
    l = log_f(xl); refl = np.log(xl.astype(np.float64))
    print("log max abs err", np.abs(l - refl).max())
    sp = softplus_f(xs); refsp = np.log1p(np.exp(np.minimum(xs.astype(np.float64), 50))); refsp[xs > 20] = xs[xs > 20]
    print("softplus max rel err", (np.abs(sp - refsp) / np.maximum(refsp, 1e-30)).max())
    sg = sigmoid_f(xs); refsg = 1 / (1 + np.exp(-xs.astype(np.float64)))
    print("sigmoid max rel err", (np.abs(sg - refsg) / refsg).max(), "min", sg.min(), "denormal?", np.any((sg > 0) & (sg < 1.17549435e-38)))
    a = np.random.RandomState(0).randn(7, 5120).astype(f32)
    print("treesum vs f64", np.abs(treesum(a) - a.astype(np.float64).sum(-1)).max())
    assert treesum(a).dtype == f32 and exp_f(xs).dtype == f32
