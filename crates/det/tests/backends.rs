//! Every backend op against the CPU backend (the spec, row by row) on random inputs, bit for bit.

#![cfg(any(all(feature = "metal", target_os = "macos"), feature = "cuda"))]

use det::backend::*;
use det::cpu::Cpu;
#[cfg(feature = "cuda")]
use det::cuda::{test_prim, Cuda as Gpu};
use det::gguf::{write_header, GgmlType, Gguf, TensorInfo, Value};
#[cfg(not(feature = "cuda"))]
use det::metal::{test_prim, Metal as Gpu};
use det::spec;
use std::sync::{Arc, Mutex, MutexGuard};

/// The GPU runtime is one global in-order queue: the tests take turns.
static LOCK: Mutex<()> = Mutex::new(());
fn serial() -> MutexGuard<'static, ()> {
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
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
}

/// A GGUF in memory holding one det linear of random codes and scales.
fn fake_gguf(rng: &mut Rng, ty: GgmlType, n: usize, k: usize) -> Arc<Gguf> {
    let bits = ty.bits().unwrap();
    let rb = ty.row_bytes(k);
    let mut data = vec![0u8; n * rb];
    let nc = k / 32 * det::spec::bytes_per_32(bits);
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
    let path = std::env::temp_dir().join(format!("det-test-{}-{n}x{k}.gguf", ty.name()));
    std::fs::write(&path, &file).unwrap();
    Arc::new(Gguf::open(&path).unwrap())
}

fn acts(rng: &mut Rng, cpu: &Cpu, m: usize, k: usize) -> (Buf<i8>, Buf<f32>, Buf<u16>, Buf<f32>) {
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
    (xq, sx, mx, x)
}

fn upload<T: Copy>(b: &Gpu, src: &Buf<T>) -> Buf<T> {
    b.upload(src.host())
}

fn same(name: &str, a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    let bad: Vec<usize> = (0..a.len()).filter(|&i| a[i].to_bits() != b[i].to_bits()).collect();
    assert!(bad.is_empty(), "{name}: {} of {} differ, first at {} (cpu {} gpu {})", bad.len(), a.len(), bad[0], a[bad[0]], b[bad[0]]);
}

#[test]
fn gemm_both_types_all_paths() {
    let _g = serial();
    let (cpu, gpu) = (Cpu, Gpu::new());
    let mut rng = Rng(11);
    for ty in [GgmlType::Det8, GgmlType::Det6, GgmlType::Det4] {
        for (n, k) in [(96, 256), (200, 384), (640, 2560), (2560, 1024), (48, 128)] {
            let g = fake_gguf(&mut rng, ty, n, k);
            let info = g.tensor("w").unwrap().clone();
            let lc = cpu.linear(&g, &[&info]);
            let lg = gpu.linear(&g, &[&info]);
            for m in [1, 3, 8, 9, 16, 17, 32, 45, 64] {
                let (xq, sx, mx, _) = acts(&mut rng, &cpu, m, k);
                let yc: Buf<f32> = cpu.alloc(m * n);
                cpu.gemm(&lc, &xq, &sx, &mx, m, GemmOut::Write(&yc));
                let (xqg, sxg, mxg) = (upload(&gpu, &xq), upload(&gpu, &sx), upload(&gpu, &mx));
                let yg: Buf<f32> = gpu.alloc(m * n);
                gpu.gemm(&lg, &xqg, &sxg, &mxg, m, GemmOut::Write(&yg));
                gpu.sync();
                same(&format!("gemm {} n={n} k={k} m={m}", ty.name()), yc.host(), &gpu.download(&yg));
                let acc_c: Buf<f32> = cpu.alloc(m * n);
                for i in 0..m * n {
                    acc_c.host_mut()[i] = rng.unit();
                }
                let acc_g: Buf<f32> = upload(&gpu, &acc_c);
                cpu.gemm(&lc, &xq, &sx, &mx, m, GemmOut::Accumulate(&acc_c));
                gpu.gemm(&lg, &xqg, &sxg, &mxg, m, GemmOut::Accumulate(&acc_g));
                gpu.sync();
                same(&format!("gemm accumulate {} n={n} k={k} m={m}", ty.name()), acc_c.host(), &gpu.download(&acc_g));
            }
        }
    }
}

#[test]
fn gemm_fused_parts() {
    let _g = serial();
    let (cpu, gpu) = (Cpu, Gpu::new());
    let mut rng = Rng(5);
    let k = 512;
    let g1 = fake_gguf(&mut rng, GgmlType::Det6, 100, k);
    let g2 = fake_gguf(&mut rng, GgmlType::Det6, 60, k);
    let i1 = g1.tensor("w").unwrap().clone();
    let i2 = g2.tensor("w").unwrap().clone();
    let i2 = TensorInfo { offset: i2.offset, ..i2 };
    let combined = {
        let mut file = Vec::new();
        let infos = vec![TensorInfo { name: "a".into(), ..i1.clone() }, TensorInfo { name: "b".into(), offset: g1.data(&i1).len(), ..i2.clone() }];
        write_header(&mut file, &[("general.architecture".into(), Value::Str("test".into()))], &infos, 32);
        file.extend_from_slice(g1.data(&i1));
        file.extend_from_slice(g2.data(&i2));
        let path = std::env::temp_dir().join("det-test-fused.gguf");
        std::fs::write(&path, &file).unwrap();
        Arc::new(Gguf::open(&path).unwrap())
    };
    let parts = [combined.tensor("a").unwrap().clone(), combined.tensor("b").unwrap().clone()];
    let lc = cpu.linear(&combined, &[&parts[0], &parts[1]]);
    let lg = gpu.linear(&combined, &[&parts[0], &parts[1]]);
    for m in [1, 7, 20] {
        let (xq, sx, mx, _) = acts(&mut rng, &cpu, m, k);
        let (ac, bc): (Buf<f32>, Buf<f32>) = (cpu.alloc(m * 100), cpu.alloc(m * 60));
        cpu.gemm(&lc, &xq, &sx, &mx, m, GemmOut::Parts(&[&ac, &bc]));
        let (ag, bg): (Buf<f32>, Buf<f32>) = (gpu.alloc(m * 100), gpu.alloc(m * 60));
        gpu.gemm(&lg, &upload(&gpu, &xq), &upload(&gpu, &sx), &upload(&gpu, &mx), m, GemmOut::Parts(&[&ag, &bg]));
        gpu.sync();
        same(&format!("parts a m={m}"), ac.host(), &gpu.download(&ag));
        same(&format!("parts b m={m}"), bc.host(), &gpu.download(&bg));
    }
}

#[test]
fn norms_and_quantization() {
    let _g = serial();
    let (cpu, gpu) = (Cpu, Gpu::new());
    let mut rng = Rng(3);
    for (m, n) in [(1, 2560), (5, 2560), (33, 4096), (9, 9216), (2, 5120), (3, 17408)] {
        let x: Buf<f32> = cpu.alloc(m * n);
        let u: Buf<f32> = cpu.alloc(m * n);
        for i in 0..m * n {
            x.host_mut()[i] = rng.unit() * 10f32.powi(-(((i / 128) % 4) as i32));
            u.host_mut()[i] = rng.unit() * 3.0;
        }
        let w: Buf<f32> = cpu.alloc(n);
        for i in 0..n {
            w.host_mut()[i] = 1.0 + rng.unit();
        }
        let (xg, ug, wg) = (upload(&gpu, &x), upload(&gpu, &u), upload(&gpu, &w));
        let g = n / 128;
        let (qc, sc, mc): (Buf<i8>, Buf<f32>, Buf<u16>) = (cpu.alloc(m * n), cpu.alloc(m), cpu.alloc(m * g));
        let (qg, sg, mg): (Buf<i8>, Buf<f32>, Buf<u16>) = (gpu.alloc(m * n), gpu.alloc(m), gpu.alloc(m * g));
        let check = |name: &str| {
            gpu.sync();
            assert_eq!(qc.host(), &gpu.download(&qg)[..], "{name} codes m={m} n={n}");
            assert_eq!(mc.host(), &gpu.download(&mg)[..], "{name} multipliers m={m} n={n}");
            same(&format!("{name} scales m={m} n={n}"), sc.host(), &gpu.download(&sg));
        };
        if n <= 8192 {
            cpu.rmsnorm_quant(&x, &w, 1e-6, m, n, &qc, &sc, &mc);
            gpu.rmsnorm_quant(&xg, &wg, 1e-6, m, n, &qg, &sg, &mg);
            check("rmsnorm_quant");
        }
        cpu.quant(&x, m, n, &qc, &sc, &mc);
        gpu.quant(&xg, m, n, &qg, &sg, &mg);
        check("quant");
        cpu.silu_mul_quant(&x, &u, m, n, &qc, &sc, &mc);
        gpu.silu_mul_quant(&xg, &ug, m, n, &qg, &sg, &mg);
        check("silu_mul_quant");
    }
}

fn kv<B: Backend>(b: &B, ctx: usize, n_kv: usize, hd: usize) -> KvCache {
    KvCache { kq: b.alloc(ctx * n_kv * hd), sk: b.alloc(ctx * n_kv), vq: b.alloc(ctx * n_kv * hd), capacity: ctx, vlo: B::V_PLANES.then(|| b.alloc(ctx * n_kv * hd)), vhi: B::V_PLANES.then(|| b.alloc(ctx * n_kv * hd)) }
}

#[test]
fn attention_prefill_and_decode() {
    let _g = serial();
    let (cpu, gpu) = (Cpu, Gpu::new());
    let mut rng = Rng(9);
    let (n_h, n_kv, hd, rot, ctx) = (16, 4, 256, 64, 300);
    let half = rot / 2;
    let (cos, sin): (Vec<f32>, Vec<f32>) = ((0..ctx * half).map(|i| ((i as f64) * 0.001).cos() as f32).collect(), (0..ctx * half).map(|i| ((i as f64) * 0.001).sin() as f32).collect());
    let (q_norm, k_norm): (Vec<f32>, Vec<f32>) = ((0..hd).map(|_| 1.0 + rng.unit() * 0.5).collect(), (0..hd).map(|_| 1.0 + rng.unit() * 0.5).collect());
    let v_scale: Vec<f32> = (0..n_kv).map(|i| 0.001 * (i + 1) as f32).collect();
    let v_inv: Vec<f32> = v_scale.iter().map(|s| 1.0 / s).collect();
    let layer_c = AttnLayer { q_norm: &cpu.upload(&q_norm), k_norm: &cpu.upload(&k_norm), v_scale: &cpu.upload(&v_scale), v_inv_scale: &cpu.upload(&v_inv), rope_cos: &cpu.upload(&cos), rope_sin: &cpu.upload(&sin) };
    let layer_g = AttnLayer { q_norm: &gpu.upload(&q_norm), k_norm: &gpu.upload(&k_norm), v_scale: &gpu.upload(&v_scale), v_inv_scale: &gpu.upload(&v_inv), rope_cos: &gpu.upload(&cos), rope_sin: &gpu.upload(&sin) };
    let (kvc, kvg) = (kv(&cpu, ctx, n_kv, hd), kv(&gpu, ctx, n_kv, hd));
    let mut pos0 = 0;
    for m in [40, 1, 7, 100, 3, 16] {
        let fill = |b: &Buf<f32>, rng: &mut Rng, scale: f32| {
            for v in b.host_mut().iter_mut() {
                *v = rng.unit() * scale;
            }
        };
        let (qg, k, v): (Buf<f32>, Buf<f32>, Buf<f32>) = (cpu.alloc(m * n_h * 2 * hd), cpu.alloc(m * n_kv * hd), cpu.alloc(m * n_kv * hd));
        fill(&qg, &mut rng, 2.0);
        fill(&k, &mut rng, 2.0);
        fill(&v, &mut rng, 1.0);
        let oc: Buf<f32> = cpu.alloc(m * n_h * hd);
        let og: Buf<f32> = gpu.alloc(m * n_h * hd);
        let shape = Shape { m, pos0, n_h, n_kv, hd, rot, eps: 1e-6 };
        cpu.attention(&layer_c, &kvc, &shape, &qg, &k, &v, &oc);
        gpu.attention(&layer_g, &kvg, &shape, &upload(&gpu, &qg), &upload(&gpu, &k), &upload(&gpu, &v), &og);
        gpu.sync();
        same(&format!("attention m={m} pos0={pos0}"), oc.host(), &gpu.download(&og));
        assert_eq!(kvc.kq.host(), &gpu.download(&kvg.kq)[..], "kq cache m={m}");
        assert_eq!(kvc.vq.host(), &gpu.download(&kvg.vq)[..], "vq cache m={m}");
        same("sk cache", kvc.sk.host(), &gpu.download(&kvg.sk));
        pos0 += m;
    }
}

#[test]
fn gdn_with_exports() {
    let _g = serial();
    let (cpu, gpu) = (Cpu, Gpu::new());
    let mut rng = Rng(21);
    let (n_k, n_v, dk, dv) = (16, 32, 128, 128);
    let conv_dim = 2 * n_k * dk + n_v * dv;
    let rep = n_v / n_k;
    let gw = 2 * dk + rep * dv + 2 * rep;
    let conv: Vec<f32> = (0..conv_dim * 4).map(|_| rng.unit() * 0.5).collect();
    let a: Vec<f32> = (0..n_v).map(|_| rng.unit().abs() * 0.3 + 0.01).collect();
    let dt_bias: Vec<f32> = (0..n_v).map(|_| rng.unit() * 5.0).collect();
    let norm: Vec<f32> = (0..dv).map(|_| 1.0 + rng.unit() * 0.3).collect();
    let layer_c = GdnLayer { conv: &cpu.upload(&conv), a: &cpu.upload(&a), dt_bias: &cpu.upload(&dt_bias), norm: &cpu.upload(&norm) };
    let layer_g = GdnLayer { conv: &gpu.upload(&conv), a: &gpu.upload(&a), dt_bias: &gpu.upload(&dt_bias), norm: &gpu.upload(&norm) };
    let stc = GdnState { conv_state: cpu.alloc(3 * conv_dim), s: cpu.alloc(n_v * dk * dv) };
    let stg = GdnState { conv_state: gpu.alloc(3 * conv_dim), s: gpu.alloc(n_v * dk * dv) };
    for m in [1, 5, 33, 2] {
        let fill = |b: &Buf<f32>, rng: &mut Rng, scale: f32| {
            for v in b.host_mut().iter_mut() {
                *v = rng.unit() * scale;
            }
        };
        let (qkv, z, av, bv): (Buf<f32>, Buf<f32>, Buf<f32>, Buf<f32>) = (cpu.alloc(m * conv_dim), cpu.alloc(m * n_v * dv), cpu.alloc(m * n_v), cpu.alloc(m * n_v));
        fill(&qkv, &mut rng, 1.0);
        fill(&z, &mut rng, 2.0);
        fill(&av, &mut rng, 3.0);
        fill(&bv, &mut rng, 3.0);
        let shape = GdnShape { m, n_k, n_v, dk, dv, eps: 1e-6 };
        let (onc, ginc, goutc): (Buf<f32>, Buf<f32>, Buf<f32>) = (cpu.alloc(m * n_v * dv), cpu.alloc(m * n_k * gw), cpu.alloc(m * n_v * dv));
        let (ong, ging, goutg): (Buf<f32>, Buf<f32>, Buf<f32>) = (gpu.alloc(m * n_v * dv), gpu.alloc(m * n_k * gw), gpu.alloc(m * n_v * dv));
        cpu.gdn(&layer_c, &stc, &shape, &qkv, &z, &av, &bv, &onc, GdnExport { gin: Some(&ginc), gout: Some(&goutc) });
        gpu.gdn(&layer_g, &stg, &shape, &upload(&gpu, &qkv), &upload(&gpu, &z), &upload(&gpu, &av), &upload(&gpu, &bv), &ong, GdnExport { gin: Some(&ging), gout: Some(&goutg) });
        gpu.sync();
        same(&format!("gdn on m={m}"), onc.host(), &gpu.download(&ong));
        same(&format!("gdn gin m={m}"), ginc.host(), &gpu.download(&ging));
        same(&format!("gdn gout m={m}"), goutc.host(), &gpu.download(&goutg));
        same(&format!("gdn state m={m}"), stc.s.host(), &gpu.download(&stg.s));
        same(&format!("gdn conv state m={m}"), stc.conv_state.host(), &gpu.download(&stg.conv_state));
    }
}

#[test]
fn sampling() {
    let _g = serial();
    let (cpu, gpu) = (Cpu, Gpu::new());
    let mut rng = Rng(77);
    let v = 248320;
    let logits: Buf<f32> = cpu.alloc(v);
    for x in logits.host_mut().iter_mut() {
        *x = rng.unit() * 12.0;
    }
    let lg = upload(&gpu, &logits);
    assert_eq!(cpu.argmax(&logits, v), gpu.argmax(&lg, v));
    for i in 0..40 {
        let z = rng.next();
        let inv_temp = 1.0 / (0.3 + rng.unit().abs() * 1.5);
        let top_k = match i % 4 { 0 => 0, 1 => 1 + rng.below(100) as usize, 2 => 40, _ => 0 };
        let top_p = match i % 3 { 0 => 1.0, 1 => 0.5 + rng.unit().abs() * 0.49, _ => 0.95 };
        assert_eq!(cpu.sample(&logits, v, inv_temp, top_k, top_p, z), gpu.sample(&lg, v, inv_temp, top_k, top_p, z), "sample inv_temp={inv_temp} top_k={top_k} top_p={top_p}");
    }
}

#[test]
fn primitives() {
    let _g = serial();
    let mut rng = Rng(1);
    let n = 1 << 16;
    let mut x: Vec<f32> = (0..n).map(|i| if i % 7 == 0 { f32::from_bits(rng.next() as u32 & 0x00FF_FFFF) } else { rng.unit() * 100.0 }).collect();
    x[0] = 0.0;
    x[1] = -0.0;
    x[2] = f32::MIN_POSITIVE;
    x[3] = -f32::MIN_POSITIVE / 2.0;
    let ops: [(i32, fn(f32) -> f32); 5] = [(0, spec::exp_f), (3, spec::sigmoid_f), (4, spec::silu_f), (6, spec::rint), (8, |v| v.abs().sqrt())];
    for (op, f) in ops {
        let input: Vec<f32> = if op == 8 { x.iter().map(|v| v.abs()).collect() } else { x.clone() };
        let got = test_prim(op, &input, 1, n);
        let want: Vec<f32> = input.iter().map(|&v| f(v)).collect();
        same(&format!("prim op {op}"), &want, &got[..n]);
    }
    let pos: Vec<f32> = x.iter().map(|v| v.abs() + 1e-30).collect();
    same("prim log", &pos.iter().map(|&v| spec::log_f(v)).collect::<Vec<_>>(), &test_prim(1, &pos, 1, n)[..n]);
    same("prim softplus", &x.iter().map(|&v| spec::softplus_f(v)).collect::<Vec<_>>(), &test_prim(2, &x, 1, n)[..n]);
    let mul: Vec<f32> = (0..n).map(|i| x[i] * x[n - 1 - i]).collect();
    same("prim mul", &mul, &test_prim(9, &x, 1, n)[..n]);
    let addv: Vec<f32> = (0..n).map(|i| x[i] + x[n - 1 - i]).collect();
    same("prim add", &addv, &test_prim(10, &x, 1, n)[..n]);
    let divv: Vec<f32> = (0..n).map(|i| x[i] / x[n - 1 - i]).collect();
    same("prim div", &divv, &test_prim(14, &x, 1, n)[..n]);
    let rows: Vec<f32> = (0..8 * 5120).map(|_| rng.unit()).collect();
    let ts: Vec<f32> = rows.chunks(5120).map(spec::treesum).collect();
    same("prim treesum", &ts, &test_prim(11, &rows, 8, 5120)[..8]);
    let pairs: Vec<i64> = (0..n / 2).map(|i| if i % 3 == 0 { (rng.next() as i64) >> (rng.below(40) as u32) } else { rng.next() as i64 >> 20 }).collect();
    let bits: Vec<f32> = pairs.iter().flat_map(|v| [f32::from_bits(*v as u32), f32::from_bits((*v >> 32) as u32)]).collect();
    let got = test_prim(5, &bits, 1, n);
    let want: Vec<f32> = pairs.iter().map(|&v| v as f32).collect();
    let got_even: Vec<f32> = (0..n / 2).map(|i| got[2 * i]).collect();
    same("prim i64->f32", &want, &got_even);
}
