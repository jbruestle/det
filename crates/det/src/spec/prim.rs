//! Scalar fp32 primitives: explicit IEEE binary32 op sequences (SPEC §4).
//! Inputs are finite or infinite; NaN is outside the model's domain.

// Constants are written as f64 literals converted to f32, which is the path np.float32(<python float>) takes.
pub const LOG2E: f32 = 1.44269504088896341f64 as f32;
pub const EXP_C1: f32 = 0.693359375f64 as f32;
pub const EXP_C2: f32 = -2.12194440e-4f64 as f32;
pub const EXP_P: [f32; 6] = [
    1.9875691500E-4f64 as f32,
    1.3981999507E-3f64 as f32,
    8.3334519073E-3f64 as f32,
    4.1665795894E-2f64 as f32,
    1.6666665459E-1f64 as f32,
    5.0000001201E-1f64 as f32,
];
pub const EXP_LO: f32 = -87.0;
pub const EXP_HI: f32 = 88.0;

pub const LOG_SQRTHF: f32 = 0.707106781186547524f64 as f32;
pub const LOG_P: [f32; 9] = [
    7.0376836292E-2f64 as f32,
    -1.1514610310E-1f64 as f32,
    1.1676998740E-1f64 as f32,
    -1.2420140846E-1f64 as f32,
    1.4249322787E-1f64 as f32,
    -1.6668057665E-1f64 as f32,
    2.0000714765E-1f64 as f32,
    -2.4999993993E-1f64 as f32,
    3.3333331174E-1f64 as f32,
];
pub const LOG_Q1: f32 = -2.12194440e-4f64 as f32;
pub const LOG_Q2: f32 = 0.693359375f64 as f32;

pub const SOFTPLUS_THRESH: f32 = 20.0;
pub const SOFTPLUS_LO: f32 = -15.0;
pub const SIGMOID_LO: f32 = -80.0;

/// `np.minimum` semantics: a NaN operand is returned.
#[inline]
pub fn fmin(a: f32, b: f32) -> f32 {
    if a.is_nan() || b.is_nan() {
        if a.is_nan() { a } else { b }
    } else if a < b { a } else { b }
}

/// `np.maximum` semantics: a NaN operand is returned.
#[inline]
pub fn fmax(a: f32, b: f32) -> f32 {
    if a.is_nan() || b.is_nan() {
        if a.is_nan() { a } else { b }
    } else if a > b { a } else { b }
}

/// `np.rint`: round half to even.
#[inline]
pub fn rint(x: f32) -> f32 {
    x.round_ties_even()
}

pub fn exp_f(x: f32) -> f32 {
    let x = fmin(x, EXP_HI);
    if x < EXP_LO {
        return 0.0;
    }
    let n = ((x * LOG2E) + 0.5).floor();
    let mut r = x - (n * EXP_C1);
    r = r - (n * EXP_C2);
    let z = r * r;
    let mut y = EXP_P[0];
    y = (y * r) + EXP_P[1];
    y = (y * r) + EXP_P[2];
    y = (y * r) + EXP_P[3];
    y = (y * r) + EXP_P[4];
    y = (y * r) + EXP_P[5];
    y = (y * z) + r;
    y = y + 1.0;
    let ni = n as i32;
    let scale = f32::from_bits(((ni + 127) << 23) as u32);
    y * scale
}

pub fn log_f(x: f32) -> f32 {
    let b = x.to_bits();
    let mut e = ((b >> 23) & 0xFF) as i32 - 126;
    let mut m = f32::from_bits((b & 0x007F_FFFF) | 0x3F00_0000);
    if m < LOG_SQRTHF {
        e -= 1;
        m = (m + m) - 1.0;
    } else {
        m = m - 1.0;
    }
    let z = m * m;
    let mut y = LOG_P[0];
    for c in &LOG_P[1..] {
        y = (y * m) + c;
    }
    y = y * m;
    y = y * z;
    let ef = e as f32;
    y = y + (LOG_Q1 * ef);
    y = y + (-0.5f32 * z);
    let mut r = m + y;
    r = r + (LOG_Q2 * ef);
    r
}

pub fn softplus_f(x: f32) -> f32 {
    if x > SOFTPLUS_THRESH {
        return x;
    }
    if x < SOFTPLUS_LO {
        return exp_f(x);
    }
    let xs = fmin(x, SOFTPLUS_THRESH);
    log_f(1.0 + exp_f(xs))
}

pub fn sigmoid_f(x: f32) -> f32 {
    if x < SIGMOID_LO {
        return exp_f(x);
    }
    let xs = fmax(x, SIGMOID_LO);
    1.0 / (1.0 + exp_f(-xs))
}

pub fn silu_f(x: f32) -> f32 {
    x * sigmoid_f(x)
}

pub fn rsqrt_f(x: f32) -> f32 {
    1.0 / x.sqrt()
}

/// Balanced binary tree sum over `x` padded with +0.0 to the next power of two:
/// at every level `s[i] = s[2i] + s[2i+1]`.
pub fn treesum(x: &[f32]) -> f32 {
    if x.is_empty() {
        return 0.0;
    }
    let p = x.len().next_power_of_two();
    let mut s = Vec::with_capacity(p);
    s.extend_from_slice(x);
    s.resize(p, 0.0);
    let mut n = p;
    while n > 1 {
        n /= 2;
        for i in 0..n {
            s[i] = s[2 * i] + s[2 * i + 1];
        }
    }
    s[0]
}

/// `y = (x * (1 / sqrt(treesum(x*x)/n + eps))) * w`
pub fn rmsnorm(x: &[f32], w: &[f32], eps: f32, y: &mut [f32]) {
    let sq: Vec<f32> = x.iter().map(|v| v * v).collect();
    let ss = treesum(&sq);
    let mean = ss / (x.len() as f32);
    let inv = rsqrt_f(mean + eps);
    for i in 0..x.len() {
        y[i] = (x[i] * inv) * w[i];
    }
}

/// `y = x * (1 / sqrt(treesum(x*x) + eps))`
pub fn l2norm(x: &[f32], eps: f32, y: &mut [f32]) {
    let sq: Vec<f32> = x.iter().map(|v| v * v).collect();
    let inv = rsqrt_f(treesum(&sq) + eps);
    for i in 0..x.len() {
        y[i] = x[i] * inv;
    }
}

/// Order-free max of |x|.
pub fn absmax(x: &[f32]) -> f32 {
    x.iter().fold(0.0f32, |m, v| fmax(m, v.abs()))
}

/// bf16 bits to fp32 (exact).
#[inline]
pub fn bf16_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

/// splitmix64: returns (next state, output).
pub fn splitmix64(state: u64) -> (u64, u64) {
    let state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    (state, z)
}
