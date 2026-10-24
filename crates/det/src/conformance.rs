//! Backend conformance: every op of a backend against the reference CPU backend (the spec applied row by row) on
//! random inputs and on the edge cases of the numerics (denormal and zero rows, a GDN state decaying
//! through the denormal range, ties, degenerate sampling parameters), bit for bit. `det conformance`
//! runs it on an installed build; `tests/backends.rs` runs it under cargo.

use crate::backend::*;
use crate::cpu::Cpu;
use crate::gguf::{write_header, GgmlType, Gguf, TensorInfo, Value};
use crate::spec;
use std::sync::Arc;

/// The elementwise primitive hook of a GPU backend (`det_test_prim`; `op` as in `det_metal.h`).
pub type Prim<'a> = &'a dyn Fn(i32, &[f32], usize, usize) -> Vec<f32>;

/// Every check by name with the first mismatch it found, if any.
#[derive(Default)]
pub struct Report {
    pub checks: Vec<(String, Option<String>)>,
}

impl Report {
    fn add(&mut self, name: String, mismatch: Option<String>) {
        self.checks.push((name, mismatch));
    }
    pub fn failures(&self) -> impl Iterator<Item = (&str, &str)> {
        self.checks.iter().filter_map(|(n, m)| m.as_deref().map(|m| (n.as_str(), m)))
    }
    pub fn ok(&self) -> bool {
        self.failures().next().is_none()
    }
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn unit(&mut self) -> f32 {
        ((self.next() >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0) as f32
    }
    fn byte(&mut self) -> u8 {
        (self.next() >> 56) as u8
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
    /// A nonzero denormal of either sign.
    fn denorm(&mut self) -> f32 {
        let b = (self.next() as u32) & 0x807F_FFFF;
        f32::from_bits(if b & 0x7FFF_FFFF == 0 { b | 1 } else { b })
    }
}

fn same(a: &[f32], b: &[f32]) -> Option<String> {
    let bad: Vec<usize> = (0..a.len()).filter(|&i| a[i].to_bits() != b[i].to_bits()).collect();
    bad.first().map(|&i| format!("{} of {} differ, first at {i}: cpu {:e} ({:#010x}) gpu {:e} ({:#010x})", bad.len(), a.len(), a[i], a[i].to_bits(), b[i], b[i].to_bits()))
}

fn same_i<T: PartialEq + std::fmt::Debug>(a: &[T], b: &[T]) -> Option<String> {
    let bad: Vec<usize> = (0..a.len()).filter(|&i| a[i] != b[i]).collect();
    bad.first().map(|&i| format!("{} of {} differ, first at {i}: cpu {:?} gpu {:?}", bad.len(), a.len(), a[i], b[i]))
}

fn fill(b: &Buf<f32>, f: &mut dyn FnMut(usize) -> f32) {
    for (i, v) in b.host_mut().iter_mut().enumerate() {
        *v = f(i);
    }
}

fn upload<B: Backend, T: Copy>(gpu: &B, src: &Buf<T>) -> Buf<T> {
    gpu.upload(src.host())
}

/// A GGUF file holding one det linear of random codes and scales.
fn fake_gguf(rng: &mut Rng, ty: GgmlType, n: usize, k: usize) -> Arc<Gguf> {
    let bits = ty.bits().unwrap();
    let rb = ty.row_bytes(k);
    let mut data = vec![0u8; n * rb];
    let nc = k / 32 * spec::bytes_per_32(bits);
    for r in 0..n {
        let row = &mut data[r * rb..(r + 1) * rb];
        for b in row[..nc].iter_mut() {
            *b = rng.byte();
        }
        let c = rng.unit().abs() * 1e-6 + 1e-8;
        row[nc..nc + 4].copy_from_slice(&c.to_le_bytes());
        for g in 0..k / 128 {
            let m = 1 + rng.below(65535) as u16;
            row[nc + 4 + 2 * g..nc + 6 + 2 * g].copy_from_slice(&m.to_le_bytes());
        }
    }
    let info = TensorInfo { name: "w".into(), dims: vec![k, n], type_id: ty.id(), offset: 0 };
    let mut file = Vec::new();
    write_header(&mut file, &[("general.architecture".into(), Value::Str("test".into()))], std::slice::from_ref(&info), 32);
    file.extend_from_slice(&data);
    let path = std::env::temp_dir().join(format!("det-conformance-{}-{n}x{k}-{}.gguf", ty.name(), std::process::id()));
    std::fs::write(&path, &file).unwrap();
    let g = Arc::new(Gguf::open(&path).unwrap());
    let _ = std::fs::remove_file(&path);
    g
}

/// Group-quantized activations of `m` rows of `k` with a scale that varies by group.
fn acts(rng: &mut Rng, cpu: &Cpu, m: usize, k: usize) -> (Buf<i8>, Buf<f32>, Buf<u16>) {
    let x: Buf<f32> = cpu.alloc(m * k);
    for t in 0..m {
        for g in 0..k / 128 {
            let scale = 10f32.powi(-(((g + t) % 5) as i32));
            for i in 0..128 {
                x.host_mut()[t * k + g * 128 + i] = rng.unit() * scale;
            }
        }
    }
    let (xq, sx, mx) = (cpu.alloc(m * k), cpu.alloc(m), cpu.alloc(m * k / 128));
    cpu.quant(&x, m, k, &xq, &sx, &mx);
    (xq, sx, mx)
}

/// Everything, in order (`prim` is the GPU backends' primitive hook; the x86 backend has none, its
/// primitives are reached through the ops).
pub fn run<B: Backend>(gpu: &B, prim: Option<Prim>) -> Report {
    let mut r = Report::default();
    if let Some(prim) = prim {
        primitives(prim, &mut r);
    }
    gemm(gpu, &mut r);
    norms(gpu, &mut r);
    attention(gpu, &mut r);
    gdn(gpu, &mut r);
    sampling(gpu, &mut r);
    r
}

/// The fp32 primitives (SPEC §4, §8) on random, denormal, zero and infinite inputs; the int64 → fp32
/// conversions (the software routine and the native cast) on crafted round-half-even ties; the saturating
/// fp32 → int32 conversion; the tree sum.
pub fn primitives(prim: Prim, r: &mut Report) {
    let mut rng = Rng(1);
    let n = 1 << 16;
    let mut x: Vec<f32> = (0..n).map(|i| if i % 7 == 0 { f32::from_bits(rng.next() as u32 & 0x00FF_FFFF) } else { rng.unit() * 100.0 }).collect();
    // pairs (x[i], x[n-1-i]) for the binary ops: denormal and zero against infinity, zero and each other
    let lhs = [1e-40f32, f32::INFINITY, 0.0, -1e-42, f32::NEG_INFINITY, 1e-39, 0.0, 1.0, f32::MIN_POSITIVE, -1e-41, 0.0, -0.0, 3e-39, f32::INFINITY];
    let rhs = [f32::INFINITY, 1e-40, 1e-40, 0.0, -1e-39, f32::NEG_INFINITY, 0.0, 1e-40, f32::INFINITY, f32::INFINITY, -0.0, 0.0, 2e-39, f32::INFINITY];
    for (i, (&l, &rv)) in lhs.iter().zip(&rhs).enumerate() {
        x[i] = l;
        x[n - 1 - i] = rv;
    }
    x[lhs.len()] = f32::MIN_POSITIVE;
    x[lhs.len() + 1] = -f32::MIN_POSITIVE / 2.0;
    let unary: [(i32, &str, fn(f32) -> f32); 5] = [(0, "exp_f", spec::exp_f), (3, "sigmoid_f", spec::sigmoid_f), (4, "silu_f", spec::silu_f), (6, "rint", spec::rint), (8, "sqrt", |v| v.abs().sqrt())];
    for (op, name, f) in unary {
        let input: Vec<f32> = if op == 8 { x.iter().map(|v| v.abs()).collect() } else { x.clone() };
        let want: Vec<f32> = input.iter().map(|&v| f(v)).collect();
        r.add(format!("prim {name}"), same(&want, &prim(op, &input, 1, n)[..n]));
    }
    let pos: Vec<f32> = x.iter().map(|v| v.abs() + 1e-30).collect();
    r.add("prim log_f".into(), same(&pos.iter().map(|&v| spec::log_f(v)).collect::<Vec<_>>(), &prim(1, &pos, 1, n)[..n]));
    r.add("prim softplus_f".into(), same(&x.iter().map(|&v| spec::softplus_f(v)).collect::<Vec<_>>(), &prim(2, &x, 1, n)[..n]));
    let binary: [(i32, &str, fn(f32, f32) -> f32); 5] = [(9, "mul", |a, b| a * b), (10, "add", |a, b| a + b), (14, "div", |a, b| a / b), (15, "sub", |a, b| a - b), (16, "fmax", spec::fmax)];
    for (op, name, f) in binary {
        let want: Vec<f32> = (0..n).map(|i| f(x[i], x[n - 1 - i])).collect();
        r.add(format!("prim {name}"), same(&want, &prim(op, &x, 1, n)[..n]));
    }
    let mut rows: Vec<f32> = (0..8 * 5120).map(|_| rng.unit()).collect();
    for i in (0..rows.len()).step_by(97) {
        rows[i] = rng.denorm();
    }
    let ts: Vec<f32> = rows.chunks(5120).map(spec::treesum).collect();
    r.add("prim treesum".into(), same(&ts, &prim(11, &rows, 8, 5120)[..8]));

    let mut vals: Vec<i64> = (0..n / 2).map(|i| if i % 3 == 0 { (rng.next() as i64) >> (rng.below(40) as u32) } else { rng.next() as i64 >> 20 }).collect();
    for s in 1..=39u32 {
        for _ in 0..300 {
            let m = ((rng.next() & 0x7F_FFFF) | 0x80_0000) as i64;
            let half = 1i64 << (s - 1);
            for tail in [half, half + 1, half - 1, half | 1, 1, (1i64 << s) - 1] {
                let v = (m << s) + tail;
                vals.push(v);
                vals.push(-v);
            }
        }
    }
    let bits: Vec<f32> = vals.iter().flat_map(|v| [f32::from_bits(*v as u32), f32::from_bits((*v >> 32) as u32)]).collect();
    let want: Vec<f32> = vals.iter().map(|&v| v as f32).collect();
    for (op, name) in [(5, "i64_to_f"), (12, "native int64 to fp32 cast")] {
        let got = prim(op, &bits, 1, bits.len());
        let got: Vec<f32> = (0..vals.len()).map(|i| got[2 * i]).collect();
        r.add(format!("prim {name}"), same(&want, &got));
    }
    let big: Vec<f32> = vec![3e9, -3e9, 2147483648.0, 2147483520.0, -2147483648.0, -2147483904.0, 1e20, -1e20, f32::INFINITY, f32::NEG_INFINITY, 2.5, -2.5, 3.5, 1e38, -1e38, 4294967296.0];
    let want: Vec<f32> = big.iter().map(|v| (v.round_ties_even() as i32) as f32).collect();
    r.add("prim fp32 to int32 saturation".into(), same(&want, &prim(13, &big, 1, big.len())[..big.len()]));
}

/// Every GEMM kernel: all widths, odd shapes, every batch-size path, residual accumulation, fused parts.
pub fn gemm<B: Backend>(gpu: &B, r: &mut Report) {
    let cpu = Cpu;
    let mut rng = Rng(11);
    for ty in [GgmlType::Det8, GgmlType::Det6, GgmlType::Det4] {
        for (n, k) in [(96, 256), (200, 384), (640, 2560), (2560, 1024), (48, 128)] {
            let g = fake_gguf(&mut rng, ty, n, k);
            let info = g.tensor("w").unwrap().clone();
            let lc = cpu.linear(&g, &[&info]);
            let lg = gpu.linear(&g, &[&info]);
            let ms: &[usize] = if (n, k) == (640, 2560) { &[1, 3, 8, 9, 16, 17, 32, 45, 64, 65, 100, 300] } else { &[1, 3, 8, 9, 16, 17, 32, 45, 64] };
            for &m in ms {
                let (xq, sx, mx) = acts(&mut rng, &cpu, m, k);
                let yc: Buf<f32> = cpu.alloc(m * n);
                cpu.gemm(&lc, &xq, &sx, &mx, m, GemmOut::Write(&yc));
                let (xqg, sxg, mxg) = (upload(gpu, &xq), upload(gpu, &sx), upload(gpu, &mx));
                let yg: Buf<f32> = gpu.alloc(m * n);
                gpu.gemm(&lg, &xqg, &sxg, &mxg, m, GemmOut::Write(&yg));
                gpu.sync();
                r.add(format!("gemm {} n={n} k={k} m={m}", ty.name()), same(yc.host(), &gpu.download(&yg)));
                let acc_c: Buf<f32> = cpu.alloc(m * n);
                fill(&acc_c, &mut |_| rng.unit());
                let acc_g: Buf<f32> = upload(gpu, &acc_c);
                cpu.gemm(&lc, &xq, &sx, &mx, m, GemmOut::Accumulate(&acc_c));
                gpu.gemm(&lg, &xqg, &sxg, &mxg, m, GemmOut::Accumulate(&acc_g));
                gpu.sync();
                r.add(format!("gemm accumulate {} n={n} k={k} m={m}", ty.name()), same(acc_c.host(), &gpu.download(&acc_g)));
            }
        }
    }
    let k = 512;
    let g1 = fake_gguf(&mut rng, GgmlType::Det6, 100, k);
    let g2 = fake_gguf(&mut rng, GgmlType::Det6, 60, k);
    let i1 = g1.tensor("w").unwrap().clone();
    let i2 = g2.tensor("w").unwrap().clone();
    let combined = {
        let mut file = Vec::new();
        let infos = vec![TensorInfo { name: "a".into(), ..i1.clone() }, TensorInfo { name: "b".into(), offset: g1.data(&i1).len(), ..i2.clone() }];
        write_header(&mut file, &[("general.architecture".into(), Value::Str("test".into()))], &infos, 32);
        file.extend_from_slice(g1.data(&i1));
        file.extend_from_slice(g2.data(&i2));
        let path = std::env::temp_dir().join(format!("det-conformance-fused-{}.gguf", std::process::id()));
        std::fs::write(&path, &file).unwrap();
        let g = Arc::new(Gguf::open(&path).unwrap());
        let _ = std::fs::remove_file(&path);
        g
    };
    let parts = [combined.tensor("a").unwrap().clone(), combined.tensor("b").unwrap().clone()];
    let lc = cpu.linear(&combined, &[&parts[0], &parts[1]]);
    let lg = gpu.linear(&combined, &[&parts[0], &parts[1]]);
    for m in [1, 7, 20, 90] {
        let (xq, sx, mx) = acts(&mut rng, &cpu, m, k);
        let (ac, bc): (Buf<f32>, Buf<f32>) = (cpu.alloc(m * 100), cpu.alloc(m * 60));
        cpu.gemm(&lc, &xq, &sx, &mx, m, GemmOut::Parts(&[&ac, &bc]));
        let (ag, bg): (Buf<f32>, Buf<f32>) = (gpu.alloc(m * 100), gpu.alloc(m * 60));
        gpu.gemm(&lg, &upload(gpu, &xq), &upload(gpu, &sx), &upload(gpu, &mx), m, GemmOut::Parts(&[&ag, &bg]));
        gpu.sync();
        r.add(format!("gemm fused parts a m={m}"), same(ac.host(), &gpu.download(&ag)));
        r.add(format!("gemm fused parts b m={m}"), same(bc.host(), &gpu.download(&bg)));
    }
}

/// rmsnorm + quantization, quantization and silu·mul + quantization on rows of mixed scales, and on rows
/// made of denormals, zeros and values whose squares underflow.
pub fn norms<B: Backend>(gpu: &B, r: &mut Report) {
    let cpu = Cpu;
    let mut rng = Rng(3);
    for (m, n) in [(1, 2560), (6, 2560), (33, 4096), (9, 9216), (4, 5120), (3, 17408)] {
        let x: Buf<f32> = cpu.alloc(m * n);
        let u: Buf<f32> = cpu.alloc(m * n);
        fill(&x, &mut |i| {
            let (row, col) = (i / n, i % n);
            match row % 6 {
                0 => rng.denorm(),
                1 => if col % 2 == 0 { rng.denorm() } else { 0.0 },
                2 => match col % 8 { 0 => rng.unit() * 1e-20, 1 => rng.unit() * 1e-23, 2 => rng.denorm(), 3 => rng.unit(), 4 => 0.0, 5 => rng.unit() * 1e-30, 6 => -0.0, _ => rng.unit() * 1e3 },
                3 => if col % 5 == 0 { rng.unit() * 1e-19 } else { rng.unit() * 1e-38 },
                _ => rng.unit() * 10f32.powi(-(((i / 128) % 4) as i32)),
            }
        });
        fill(&u, &mut |i| match (i % n) % 3 { 0 => rng.unit() * 1e-20, 1 => rng.denorm(), _ => rng.unit() * 3.0 });
        let w: Buf<f32> = cpu.alloc(n);
        fill(&w, &mut |_| 1.0 + rng.unit());
        let (xg, ug, wg) = (upload(gpu, &x), upload(gpu, &u), upload(gpu, &w));
        let g = n / 128;
        let (qc, sc, mc): (Buf<i8>, Buf<f32>, Buf<u16>) = (cpu.alloc(m * n), cpu.alloc(m), cpu.alloc(m * g));
        let (qg, sg, mg): (Buf<i8>, Buf<f32>, Buf<u16>) = (gpu.alloc(m * n), gpu.alloc(m), gpu.alloc(m * g));
        let check = |r: &mut Report, name: &str| {
            gpu.sync();
            r.add(format!("{name} codes m={m} n={n}"), same_i(qc.host(), &gpu.download(&qg)));
            r.add(format!("{name} multipliers m={m} n={n}"), same_i(mc.host(), &gpu.download(&mg)));
            r.add(format!("{name} scales m={m} n={n}"), same(sc.host(), &gpu.download(&sg)));
        };
        if n <= 8192 {
            cpu.rmsnorm_quant(&x, &w, 1e-6, m, n, &qc, &sc, &mc);
            gpu.rmsnorm_quant(&xg, &wg, 1e-6, m, n, &qg, &sg, &mg);
            check(r, "rmsnorm_quant");
        }
        cpu.quant(&x, m, n, &qc, &sc, &mc);
        gpu.quant(&xg, m, n, &qg, &sg, &mg);
        check(r, "quant");
        cpu.silu_mul_quant(&x, &u, m, n, &qc, &sc, &mc);
        gpu.silu_mul_quant(&xg, &ug, m, n, &qg, &sg, &mg);
        check(r, "silu_mul_quant");
    }
}

fn kv<B: Backend>(b: &B, ctx: usize, n_kv: usize, hd: usize) -> KvCache {
    KvCache { kq: b.alloc(ctx * n_kv * hd), sk: b.alloc(ctx * n_kv), vq: b.alloc(ctx * n_kv * hd), capacity: ctx, vlo: B::V_PLANES.then(|| b.alloc(ctx * n_kv * hd)), vhi: B::V_PLANES.then(|| b.alloc(ctx * n_kv * hd)) }
}

/// Full attention over a growing cache: decode and prefill batch sizes, key splitting, rows of q, k and v
/// at every scale from denormal to 1e3.
pub fn attention<B: Backend>(gpu: &B, r: &mut Report) {
    let cpu = Cpu;
    let mut rng = Rng(9);
    let (n_h, n_kv, hd, rot, ctx) = (16, 4, 256, 64, 400);
    let half = rot / 2;
    let (cos, sin): (Vec<f32>, Vec<f32>) = ((0..ctx * half).map(|i| ((i as f64) * 0.001).cos() as f32).collect(), (0..ctx * half).map(|i| ((i as f64) * 0.001).sin() as f32).collect());
    let (q_norm, k_norm): (Vec<f32>, Vec<f32>) = ((0..hd).map(|_| 1.0 + rng.unit() * 0.5).collect(), (0..hd).map(|_| 1.0 + rng.unit() * 0.5).collect());
    let v_scale: Vec<f32> = (0..n_kv).map(|i| 0.001 * (i + 1) as f32).collect();
    let v_inv: Vec<f32> = v_scale.iter().map(|s| 1.0 / s).collect();
    let lc = AttnLayer { q_norm: &cpu.upload(&q_norm), k_norm: &cpu.upload(&k_norm), v_scale: &cpu.upload(&v_scale), v_inv_scale: &cpu.upload(&v_inv), rope_cos: &cpu.upload(&cos), rope_sin: &cpu.upload(&sin) };
    let lg = AttnLayer { q_norm: &gpu.upload(&q_norm), k_norm: &gpu.upload(&k_norm), v_scale: &gpu.upload(&v_scale), v_inv_scale: &gpu.upload(&v_inv), rope_cos: &gpu.upload(&cos), rope_sin: &gpu.upload(&sin) };
    let (kvc, kvg) = (kv(&cpu, ctx, n_kv, hd), kv(gpu, ctx, n_kv, hd));
    let scale_of = |row: usize| -> f32 {
        match row % 6 {
            0 => 1e-25,
            1 => 1e-38,
            2 => 2.0,
            3 => 1e-10,
            4 => 2.0,
            _ => 1e3,
        }
    };
    let mut pos0 = 0;
    for m in [40, 1, 7, 100, 3, 16, 200, 5, 1] {
        let (qg, k, v): (Buf<f32>, Buf<f32>, Buf<f32>) = (cpu.alloc(m * n_h * 2 * hd), cpu.alloc(m * n_kv * hd), cpu.alloc(m * n_kv * hd));
        fill(&qg, &mut |i| if i % (2 * hd) < hd { rng.unit() * scale_of(i / (2 * hd)) } else { rng.unit() * 200.0 });
        fill(&k, &mut |i| rng.unit() * scale_of(i / hd + 1));
        fill(&v, &mut |i| if (i / hd) % 4 == 1 { rng.denorm() } else { rng.unit() * scale_of(i / hd + 2) });
        let oc: Buf<f32> = cpu.alloc(m * n_h * hd);
        let og: Buf<f32> = gpu.alloc(m * n_h * hd);
        let shape = Shape { m, pos0, n_h, n_kv, hd, rot, eps: 1e-6 };
        cpu.attention(&lc, &kvc, &shape, &qg, &k, &v, &oc);
        gpu.attention(&lg, &kvg, &shape, &upload(gpu, &qg), &upload(gpu, &k), &upload(gpu, &v), &og);
        gpu.sync();
        r.add(format!("attention m={m} pos0={pos0}"), same(oc.host(), &gpu.download(&og)));
        r.add(format!("attention kq cache m={m}"), same_i(kvc.kq.host(), &gpu.download(&kvg.kq)));
        r.add(format!("attention vq cache m={m}"), same_i(kvc.vq.host(), &gpu.download(&kvg.vq)));
        r.add(format!("attention sk cache m={m}"), same(kvc.sk.host(), &gpu.download(&kvg.sk)));
        pos0 += m;
    }
}

/// Gated DeltaNet over a sequence whose state rows decay through the denormal range (k rows held at zero
/// after a warmup), heads with decays near the underflow limit, every batch boundary.
pub fn gdn<B: Backend>(gpu: &B, r: &mut Report) {
    let cpu = Cpu;
    let mut rng = Rng(21);
    let (n_k, n_v, dk, dv) = (16, 32, 128, 128);
    let conv_dim = 2 * n_k * dk + n_v * dv;
    let rep = n_v / n_k;
    let gw = 2 * dk + rep * dv + 2 * rep;
    let kd = n_k * dk;
    let mut conv = vec![0f32; conv_dim * 4];
    for c in 0..conv_dim {
        conv[c * 4 + 3] = 1.0;
        conv[c * 4 + 2] = 0.01 * rng.unit();
    }
    let a: Vec<f32> = (0..n_v).map(|h| match h { 0 => 4.35, 1 => 4.6, 2 => 0.5, _ => 0.03 + 0.02 * (h as f32 / n_v as f32) }).collect();
    let dt_bias = vec![20.0f32; n_v];
    let norm: Vec<f32> = (0..dv).map(|_| 1.0 + rng.unit() * 0.3).collect();
    let lc = GdnLayer { conv: &cpu.upload(&conv), a: &cpu.upload(&a), dt_bias: &cpu.upload(&dt_bias), norm: &cpu.upload(&norm) };
    let lg = GdnLayer { conv: &gpu.upload(&conv), a: &gpu.upload(&a), dt_bias: &gpu.upload(&dt_bias), norm: &gpu.upload(&norm) };
    let stc = GdnState { conv_state: cpu.alloc(3 * conv_dim), s: cpu.alloc(n_v * dk * dv) };
    let stg = GdnState { conv_state: gpu.alloc(3 * conv_dim), s: gpu.alloc(n_v * dk * dv) };
    let mut t0 = 0usize;
    let mut denormal_seen = 0usize;
    for m in [60, 20, 20, 20, 20, 1, 1, 1, 1, 1, 10, 10, 10, 10, 10, 10, 10, 10, 10, 100, 37, 1, 5] {
        let qkv: Buf<f32> = cpu.alloc(m * conv_dim);
        let (z, av, bv): (Buf<f32>, Buf<f32>, Buf<f32>) = (cpu.alloc(m * n_v * dv), cpu.alloc(m * n_v), cpu.alloc(m * n_v));
        fill(&qkv, &mut |i| {
            let (t, c) = (i / conv_dim, i % conv_dim);
            let tg = t0 + t;
            if c >= kd && c < 2 * kd && tg >= 20 && (c - kd) % dk >= 32 && tg % 97 != 0 {
                return -200.0;
            }
            rng.unit()
        });
        fill(&z, &mut |_| rng.unit() * 2.0);
        fill(&av, &mut |_| rng.unit() * 3.0);
        fill(&bv, &mut |_| rng.unit() * 3.0);
        let shape = GdnShape { m, n_k, n_v, dk, dv, eps: 1e-6 };
        let (onc, ginc, goutc): (Buf<f32>, Buf<f32>, Buf<f32>) = (cpu.alloc(m * n_v * dv), cpu.alloc(m * n_k * gw), cpu.alloc(m * n_v * dv));
        let (ong, ging, goutg): (Buf<f32>, Buf<f32>, Buf<f32>) = (gpu.alloc(m * n_v * dv), gpu.alloc(m * n_k * gw), gpu.alloc(m * n_v * dv));
        cpu.gdn(&lc, &stc, &shape, &qkv, &z, &av, &bv, &onc, GdnExport { gin: Some(&ginc), gout: Some(&goutc) });
        gpu.gdn(&lg, &stg, &shape, &upload(gpu, &qkv), &upload(gpu, &z), &upload(gpu, &av), &upload(gpu, &bv), &ong, GdnExport { gin: Some(&ging), gout: Some(&goutg) });
        gpu.sync();
        denormal_seen += stc.s.host().iter().filter(|&&x| x != 0.0 && x.abs() < f32::MIN_POSITIVE).count();
        r.add(format!("gdn output m={m} t0={t0}"), same(onc.host(), &gpu.download(&ong)));
        r.add(format!("gdn inputs m={m} t0={t0}"), same(ginc.host(), &gpu.download(&ging)));
        r.add(format!("gdn mixer output m={m} t0={t0}"), same(goutc.host(), &gpu.download(&goutg)));
        r.add(format!("gdn state m={m} t0={t0}"), same(stc.s.host(), &gpu.download(&stg.s)));
        r.add(format!("gdn conv state m={m} t0={t0}"), same(stc.conv_state.host(), &gpu.download(&stg.conv_state)));
        t0 += m;
    }
    r.add("gdn state reached the denormal range".into(), (denormal_seen == 0).then(|| "no denormal state entry was produced; the sequence no longer exercises the denormal path".into()));
}

/// Greedy and sampled picks: random logits, logits full of ties, ±0 maxima, every filter combination and
/// the degenerate parameters (`top_p` at and below the fixed-point resolution, `top_k` past the vocabulary).
pub fn sampling<B: Backend>(gpu: &B, r: &mut Report) {
    let cpu = Cpu;
    let mut rng = Rng(77);
    let v = 248320;
    let random: Buf<f32> = cpu.alloc(v);
    fill(&random, &mut |_| rng.unit() * 12.0);
    let ties: Buf<f32> = cpu.alloc(v);
    fill(&ties, &mut |i| (i % 7) as f32 - 3.0);
    let zeros: Buf<f32> = cpu.alloc(v);
    fill(&zeros, &mut |i| match i { 7 => 0.0, 11 => -0.0, 100 => 0.0, _ => -1.0 - (i % 13) as f32 });
    for (name, logits) in [("random", &random), ("ties", &ties), ("zero maxima", &zeros)] {
        let lg = upload(gpu, logits);
        let (c, g) = (cpu.argmax(logits, v), gpu.argmax(&lg, v));
        r.add(format!("argmax {name}"), (c != g).then(|| format!("cpu {c} gpu {g}")));
        let mut params: Vec<(f32, usize, f32)> = (0..24)
            .map(|i| {
                let inv_temp = 1.0 / (0.3 + rng.unit().abs() * 1.5);
                let top_k = match i % 4 { 0 => 0, 1 => 1 + rng.below(100) as usize, 2 => 40, _ => 0 };
                let top_p = match i % 3 { 0 => 1.0, 1 => 0.5 + rng.unit().abs() * 0.49, _ => 0.95 };
                (inv_temp, top_k, top_p)
            })
            .collect();
        params.extend([(1.0, 0, 1e-9), (1.0, 0, 2e-8), (1.0, 0, 0.0), (1.0, 0, -0.5), (1.0, 3, 1e-9), (1.0, 1, 1.0), (1.0, 0, 0.999999), (1.0, v, 0.5), (1.0, v + 5, 0.5), (1.0, 5, 0.3)]);
        for (inv_temp, top_k, top_p) in params {
            let z = rng.next();
            let (c, g) = (cpu.sample(logits, v, inv_temp, top_k, top_p, z), gpu.sample(&lg, v, inv_temp, top_k, top_p, z));
            r.add(format!("sample {name} inv_temp={inv_temp} top_k={top_k} top_p={top_p}"), (c != g).then(|| format!("cpu {c} gpu {g}")));
        }
    }
}
