use det::gguf::{ggml_type_name, Gguf, Value};
use det::spec::dequant_row;
use det::{Backend, Buf, Model, Sampling};
use std::path::Path;
use std::time::Instant;

const DEFAULT_BACKEND: &str = if cfg!(target_os = "linux") {
    "cuda"
} else if cfg!(target_os = "macos") {
    "metal"
} else {
    "cpu"
};

fn usage() -> ! {
    eprintln!(
        "usage:\n  det convert <float.gguf> [--out det.gguf] [--bits 6,ffn_down=8,...] [--rope-positions N] [--no-calibrate]\n  det generate <model.gguf> [--prompt TEXT|@file | --ids 1,2,3|@file] [--chat] [--n 64] [--temp T] [--top-k K] [--top-p P] [--seed S]\n               [--backend cpu|ref|metal|cuda (default: the GPU backend built in)] [--ctx 8192] [--batch 256] [--logits out.bin] [--dump]\n  det calibrate <model.gguf> [--out calibrated.gguf] [--random 512]   (default: rewrites the file in place)\n  det bench <model.gguf> [--pp 512] [--tg 128] [--batch 256] [--gemm]\n  det ppl <model.gguf> <text.txt> [--ctx 2048] [--chunks 8] [--batch 512]\n  det conformance [--backend cpu|metal|cuda]   (every op of the backend against the reference, bit for bit; exit code 1 on a mismatch)\n  det info <model.gguf>\n  det tensor <model.gguf> <name> [n]\n  det tokens <model.gguf> <from> <to>"
    );
    std::process::exit(2)
}

struct NoObs;
impl det::hooks::Observer for NoObs {
    fn observe(&mut self, _: &det::hooks::Ctx, _: det::hooks::Tap) {}
}

/// Prints the first and last values and the sum of the residual stream and mixer outputs of every
/// position, in the layout of llama-eval-callback, for comparing against llama.cpp.
struct Dump {
    hidden: usize,
    mixer_w: usize,
}
impl det::hooks::Observer for Dump {
    fn observe(&mut self, ctx: &det::hooks::Ctx, tap: det::hooks::Tap) {
        use det::hooks::Tap;
        let (name, buf, width, layer) = match tap {
            Tap::Residual { layer, x } => ("res", x, self.hidden, layer),
            Tap::PostMixer { layer, x } => ("postmix", x, self.hidden, layer),
            Tap::MixerOut { layer, y } => ("mixout", y, self.mixer_w, layer),
            _ => return,
        };
        let rows = ctx.read(buf, 0, ctx.t * width);
        for t in 0..ctx.t {
            let r = &rows[t * width..(t + 1) * width];
            let n = r.len();
            let sum: f64 = r.iter().map(|v| *v as f64).sum();
            eprintln!("{name} layer {layer} pos {}: [{:.4} {:.4} {:.4} ... {:.4} {:.4} {:.4}] sum {sum:.6}", ctx.pos0 + t, r[0], r[1], r[2], r[n - 3], r[n - 2], r[n - 1]);
        }
    }
}

struct Args {
    pos: Vec<String>,
    opts: Vec<(String, Option<String>)>,
}

impl Args {
    fn parse() -> Self {
        let argv: Vec<String> = std::env::args().skip(1).collect();
        let mut pos = Vec::new();
        let mut opts = Vec::new();
        let mut i = 0;
        while i < argv.len() {
            if let Some(name) = argv[i].strip_prefix("--") {
                let takes = !matches!(name, "chat" | "think" | "dump" | "gemm" | "no-calibrate");
                let val = if takes {
                    argv.get(i + 1).cloned()
                } else {
                    None
                };
                if takes {
                    i += 1;
                }
                opts.push((name.to_string(), val));
            } else {
                pos.push(argv[i].clone());
            }
            i += 1;
        }
        Args { pos, opts }
    }
    fn opt(&self, name: &str) -> Option<&str> {
        self.opts
            .iter()
            .rev()
            .find(|(n, _)| n == name)
            .and_then(|(_, v)| v.as_deref())
    }
    fn flag(&self, name: &str) -> bool {
        self.opts.iter().any(|(n, _)| n == name)
    }
    fn num<T: std::str::FromStr>(&self, name: &str, default: T) -> T {
        self.opt(name)
            .map(|v| {
                v.parse()
                    .unwrap_or_else(|_| panic!("--{name}: bad value {v}"))
            })
            .unwrap_or(default)
    }
}

fn show(v: &Value) -> String {
    match v {
        Value::Str(s) => format!(
            "{:?}",
            if s.len() > 80 {
                format!("{}…", &s[..80])
            } else {
                s.clone()
            }
        ),
        Value::Array(a) => {
            let head: Vec<String> = a.iter().take(8).map(show).collect();
            format!(
                "[{}{}] ({} items)",
                head.join(", "),
                if a.len() > 8 { ", …" } else { "" },
                a.len()
            )
        }
        other => format!("{other:?}"),
    }
}

fn info(path: &Path) {
    let g = Gguf::open(path).expect("open gguf");
    for (k, v) in &g.kv {
        println!("{k} = {}", show(v));
    }
    println!(
        "data_start {} alignment {} tensors {}",
        g.data_start,
        g.alignment,
        g.tensors.len()
    );
    for t in &g.tensors {
        println!(
            "{:40} {:?} {} @{}",
            t.name,
            t.dims,
            ggml_type_name(t.type_id),
            t.offset
        );
    }
}

fn tensor(path: &Path, name: &str, n: usize) {
    let g = Gguf::open(path).expect("open gguf");
    let t = g.tensor(name).expect("no such tensor");
    let ty = t.ty().expect("unsupported type");
    let k = t.dims[0];
    let rows = t.n_elems() / k;
    let data = g.data(t);
    let rb = ty.row_bytes(k);
    let mut row = vec![0f32; k];
    let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
    for r in 0..rows {
        dequant_row(ty, &data[r * rb..(r + 1) * rb], k, &mut row);
        if r == 0 {
            println!("{name} {:?} {}: {:?}", t.dims, ty.name(), &row[..n.min(k)]);
        }
        for &v in &row {
            lo = lo.min(v);
            hi = hi.max(v);
        }
    }
    println!("min {lo} max {hi}");
}

fn tokens(path: &Path, from: usize, to: usize) {
    let g = Gguf::open(path).expect("open gguf");
    let toks = g
        .get("tokenizer.ggml.tokens")
        .and_then(Value::as_array)
        .expect("tokens");
    let types = g
        .get("tokenizer.ggml.token_type")
        .and_then(Value::as_array)
        .expect("token types");
    for i in from..to.min(toks.len()) {
        println!("{i} {:?} type {:?}", toks[i].as_str().unwrap(), types[i]);
    }
}

/// An argument's text, or the contents of a file for `@path`.
fn text_arg(v: &str) -> String {
    match v.strip_prefix('@') {
        Some(p) => std::fs::read_to_string(p).unwrap_or_else(|e| panic!("{p}: {e}")),
        None => v.to_string(),
    }
}

fn generate<B: Backend>(a: &Args, path: &Path, backend: B) {
    let mut m = load(a, path, backend);
    let tk = det::tokenizer::from_gguf(&m.gguf).unwrap_or_else(|e| panic!("tokenizer: {e}"));
    let ids: Vec<i32> = match (a.opt("ids").map(text_arg), a.opt("prompt").map(text_arg)) {
        (Some(s), _) => s
            .trim()
            .split(',')
            .map(|v| v.parse().expect("token id"))
            .collect(),
        (None, Some(p)) => {
            let text = if a.flag("chat") {
                det::tokenizer::chat_prompt(&m.gguf, &p, a.flag("think"))
                    .unwrap_or_else(|e| panic!("chat template: {e}"))
            } else {
                p
            };
            tk.tokenizer
                .encode(text, false)
                .expect("encode")
                .get_ids()
                .iter()
                .map(|&v| v as i32)
                .collect()
        }
        _ => usage(),
    };
    eprintln!("prompt: {} ids {:?}", ids.len(), &ids[..ids.len().min(32)]);
    let temp: f32 = a.num("temp", 0.0);
    let sampling = Sampling {
        inv_temp: (temp > 0.0).then(|| 1.0 / temp),
        top_k: a.num("top-k", 0),
        top_p: a.num("top-p", 1.0),
        rng: a.num("seed", 0u64),
    };
    let stop: Vec<i32> = tk.eos.iter().map(|&v| v as i32).collect();
    let n_new: usize = a.num("n", 64);
    let t0 = Instant::now();
    if a.flag("dump") {
        let c = &m.cfg;
        let mut d = Dump {
            hidden: c.hidden,
            mixer_w: c.gdn_v_heads * c.gdn_dv,
        };
        m.prefill(&ids, Some(&mut d))
            .unwrap_or_else(|e| panic!("{e}"));
    } else {
        m.prefill::<NoObs>(&ids, None)
            .unwrap_or_else(|e| panic!("{e}"));
    }
    let tp = t0.elapsed().as_secs_f64();
    let t0 = Instant::now();
    let mut text = String::new();
    let out = m
        .generate::<NoObs>(&[], n_new, sampling, &stop, None, |id| {
            let piece = tk.tokenizer.decode(&[id as u32], false).unwrap_or_default();
            eprint!("{piece}");
            text.push_str(&piece);
        })
        .unwrap_or_else(|e| panic!("{e}"));
    eprintln!();
    let td = t0.elapsed().as_secs_f64();
    eprintln!(
        "prefill {} tok in {tp:.3}s = {:.1} tok/s; decode {} tok in {td:.3}s = {:.2} tok/s",
        ids.len(),
        ids.len() as f64 / tp,
        out.len(),
        out.len() as f64 / td
    );
    println!(
        "{}",
        out.iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join(",")
    );
    if let Some(p) = a.opt("logits") {
        let bytes: Vec<u8> = m.logits().iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(p, bytes).unwrap();
    }
}

/// Runs the conformance suites of the GPU backend against the CPU reference and prints every check.
fn conformance<B: Backend>(backend: B, prim: Option<det::conformance::Prim>) -> ! {
    let t0 = Instant::now();
    let r = det::conformance::run(&backend, prim);
    for (name, mismatch) in &r.checks {
        match mismatch {
            None => println!("ok    {name}"),
            Some(m) => println!("FAIL  {name}: {m}"),
        }
    }
    let failed = r.failures().count();
    println!(
        "{}: {} checks, {failed} failed ({:.1}s)",
        backend.name(),
        r.checks.len(),
        t0.elapsed().as_secs_f64()
    );
    std::process::exit(if failed == 0 { 0 } else { 1 })
}

fn load<B: Backend>(a: &Args, path: &Path, backend: B) -> Model<B> {
    let t0 = Instant::now();
    let ctx: usize = a.num("ctx", 8192);
    let batch: usize = a.num("batch", 256);
    let m = Model::load(path, backend, ctx, batch).unwrap_or_else(|e| panic!("{e}"));
    eprintln!(
        "{}: loaded in {:.1}s",
        m.backend.name(),
        t0.elapsed().as_secs_f64()
    );
    m
}

fn calibrate<B: Backend>(a: &Args, path: &Path, backend: B) {
    let mut m = load(a, path, backend);
    let tk = det::tokenizer::from_gguf(&m.gguf).unwrap_or_else(|e| panic!("tokenizer: {e}"));
    let ids = det::calibrate::calibration_ids(&tk.tokenizer, m.cfg.vocab, a.num("random", 512));
    let scales = det::calibrate::v_scales(&mut m, &ids).unwrap_or_else(|e| panic!("{e}"));
    for (i, s) in scales.chunks(m.cfg.n_kv).enumerate() {
        eprintln!("attention layer {i}: v_scale {:?}", s);
    }
    match a.opt("out").map(Path::new) {
        Some(out) => {
            det::calibrate::write_copy(&m, &scales, out).expect("write gguf");
            eprintln!("wrote {} ({} value scales)", out.display(), scales.len());
        }
        None => {
            if det::calibrate::write_in_place(&m, &scales, path).expect("write gguf") {
                eprintln!(
                    "wrote {} value scales into {}",
                    scales.len(),
                    path.display()
                );
            } else {
                let tmp = path.with_extension("gguf.tmp");
                det::calibrate::write_copy(&m, &scales, &tmp).expect("write gguf");
                drop(m);
                std::fs::rename(&tmp, path).expect("rename");
                eprintln!(
                    "rewrote {} with {} value scales",
                    path.display(),
                    scales.len()
                );
            }
        }
    }
}

/// Times every distinct linear shape of the model at decode and prefill batch sizes.
fn bench_gemm<B: Backend>(m: &Model<B>) {
    use det::model::Mixer;
    let mut seen: Vec<(&str, usize, &B::Linear)> = Vec::new();
    let mut push = |name: &'static str, p: &'_ det::model::Proj<B>| {
        if seen.iter().any(|s| s.0 == name) {
            return;
        }
        let l = &p.lins[0];
        seen.push((name, B::linear_widths(l).iter().sum(), unsafe {
            &*(l as *const B::Linear)
        }));
    };
    let l0 = &m.layers[0];
    push("gate|up", &l0.gu);
    push("down", &l0.down);
    for l in &m.layers {
        match &l.mixer {
            Mixer::Attn(at) => {
                push("q|k|v", &at.qkv);
                push("o", &at.o);
            }
            Mixer::Gdn(g) => {
                push("qkv|z", &g.qkvz);
                push("a|b", &g.ab);
                push("out", &g.out);
            }
        }
    }
    push("lm_head", &m.lm_head);
    let bk = &m.backend;
    let k_of = |name: &str| -> usize {
        match name {
            "down" => m.cfg.intermediate,
            "o" => m.cfg.n_heads * m.cfg.head_dim,
            "out" => m.cfg.gdn_v_heads * m.cfg.gdn_dv,
            _ => m.cfg.hidden,
        }
    };
    let bpw = {
        let t = m.gguf.tensor("blk.0.ffn_down.weight").unwrap();
        t.ty().unwrap().row_bytes(t.dims[0]) as f64 / t.dims[0] as f64
    };
    for pass in 0..2 {
        for (name, n, lin) in &seen {
            let k = k_of(name);
            for &mm in &[1usize, 8, 256] {
                if mm > m.max_batch {
                    continue;
                }
                let (xq, sx, mx, y): (Buf<i8>, Buf<f32>, Buf<u16>, Buf<f32>) = (
                    bk.alloc(mm * k),
                    bk.alloc(mm),
                    bk.alloc(mm * k / 128),
                    bk.alloc(mm * n),
                );
                bk.write(&sx, 0, &vec![1.0f32; mm]);
                bk.write(&mx, 0, &vec![1u16; mm * k / 128]);
                let reps = if mm == 1 { 200 } else { 20 };
                for _ in 0..reps / 4 {
                    bk.gemm(lin, &xq, &sx, &mx, mm, det::backend::GemmOut::Write(&y));
                }
                bk.sync();
                let t0 = Instant::now();
                for _ in 0..reps {
                    bk.gemm(lin, &xq, &sx, &mx, mm, det::backend::GemmOut::Write(&y));
                }
                bk.sync();
                let dt = t0.elapsed().as_secs_f64() / reps as f64;
                let bytes = (n * k) as f64 * bpw;
                if pass == 1 {
                    println!(
                        "{name:8} N={n:6} K={k:5} M={mm:3}: {:7.3} ms  {:6.0} GB/s  {:5.1} TOPS",
                        dt * 1e3,
                        bytes / dt / 1e9,
                        2.0 * (n * k * mm) as f64 / dt / 1e12
                    );
                }
            }
        }
    }
}

fn bench<B: Backend>(a: &Args, path: &Path, backend: B) {
    let mut m = load(a, path, backend);
    if a.flag("gemm") {
        return bench_gemm(&m);
    }
    let pp: usize = a.num("pp", 512);
    let tg: usize = a.num("tg", 128);
    let ids: Vec<i32> = (0..pp).map(|i| 1000 + (i * 7919 % 50000) as i32).collect();
    m.prefill::<NoObs>(&ids[..1.min(pp)], None).unwrap();
    m.backend.sync();
    m.reset();
    let t0 = Instant::now();
    m.prefill::<NoObs>(&ids, None).unwrap();
    m.backend.sync();
    let tp = t0.elapsed().as_secs_f64();
    let t0 = Instant::now();
    let mut id = 1000;
    for _ in 0..tg {
        m.forward::<NoObs>(&[id], None).unwrap();
        id = m.pick(&mut Sampling::greedy()).unwrap();
    }
    m.backend.sync();
    let td = t0.elapsed().as_secs_f64();
    println!(
        "pp{pp}: {:.1} tok/s ({tp:.3}s)   tg{tg}: {:.2} tok/s ({:.1} ms/tok)",
        pp as f64 / tp,
        tg as f64 / td,
        td * 1000.0 / tg as f64
    );
}

/// Perplexity over `text` with the llama-perplexity protocol: chunks of `ctx` tokens, the second half
/// of each chunk scored.
fn ppl<B: Backend>(a: &Args, path: &Path, text_path: &Path, backend: B) {
    let text = std::fs::read_to_string(text_path).expect("read text");
    let mut m = load(a, path, backend);
    let tk = det::tokenizer::from_gguf(&m.gguf).unwrap_or_else(|e| panic!("tokenizer: {e}"));
    let ids: Vec<i32> = tk
        .tokenizer
        .encode(text, false)
        .expect("encode")
        .get_ids()
        .iter()
        .map(|&v| v as i32)
        .collect();
    let ctx: usize = a.num("ctx", 2048);
    let n_chunks: usize = a.num("chunks", usize::MAX).min(ids.len() / ctx);
    eprintln!("{} tokens, {n_chunks} chunks of {ctx}", ids.len());
    m.set_all_logits(true);
    let v = m.cfg.vocab;
    let (mut nll, mut count) = (0f64, 0usize);
    let t0 = Instant::now();
    for c in 0..n_chunks {
        let chunk = &ids[c * ctx..(c + 1) * ctx];
        m.reset();
        let mut pos = 0;
        for batch in chunk.chunks(m.max_batch) {
            m.forward::<NoObs>(batch, None).unwrap();
            let logits = m.logits();
            for i in 0..batch.len() {
                let p = pos + i;
                if p + 1 >= chunk.len() {
                    break;
                }
                if p + 1 < ctx / 2 {
                    continue;
                }
                let row = &logits[i * v..(i + 1) * v];
                let mx = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
                let lse: f64 = row.iter().map(|&x| (x as f64 - mx).exp()).sum::<f64>().ln() + mx;
                nll += lse - row[chunk[p + 1] as usize] as f64;
                count += 1;
            }
            pos += batch.len();
        }
        eprintln!(
            "chunk {c}: ppl so far {:.4} ({count} tokens, {:.1}s)",
            (nll / count as f64).exp(),
            t0.elapsed().as_secs_f64()
        );
    }
    println!(
        "perplexity {:.4} over {count} tokens",
        (nll / count as f64).exp()
    );
}

fn main() {
    let a = Args::parse();
    let pos = |i: usize| -> &str { a.pos.get(i).map(String::as_str).unwrap_or_else(|| usage()) };
    let cmd = pos(0);
    match cmd {
        "convert" => {
            let src = Path::new(pos(1));
            let bits = det::convert::Bits::parse(a.opt("bits").unwrap_or(det::convert::DEFAULT_BITS)).unwrap_or_else(|e| panic!("--bits: {e}"));
            let out = a.opt("out").map(std::path::PathBuf::from).unwrap_or_else(|| det::convert::default_output(src, &bits));
            let source = det::convert::Source::open(src).unwrap_or_else(|e| panic!("{}: {e}", src.display()));
            let opt = det::convert::Options { bits, rope_positions: a.opt("rope-positions").map(|v| v.parse().unwrap()) };
            det::convert::convert(&source, &out, &opt).unwrap_or_else(|e| panic!("convert: {e}"));
            drop(source);
            if a.flag("no-calibrate") {
                return;
            }
            let b = Args { pos: vec!["calibrate".into(), out.to_string_lossy().into_owned()], opts: Vec::new() };
            return match DEFAULT_BACKEND {
                #[cfg(target_os = "macos")]
                "metal" => calibrate(&b, &out, det::metal::Metal::new()),
                #[cfg(target_os = "linux")]
                "cuda" => calibrate(&b, &out, det::cuda::Cuda::new()),
                _ => calibrate(&b, &out, det::cpu::Cpu),
            };
        }
        "conformance" => match a.opt("backend").unwrap_or(DEFAULT_BACKEND) {
            #[cfg(target_arch = "x86_64")]
            "cpu" if det::x86::X86::available() => conformance(det::x86::X86::new(), None),
            #[cfg(target_os = "macos")]
            "metal" => conformance(det::metal::Metal::new(), Some(&det::metal::test_prim)),
            #[cfg(target_os = "linux")]
            "cuda" => conformance(det::cuda::Cuda::new(), Some(&det::cuda::test_prim)),
            other => panic!("conformance checks a GPU or x86 backend against the reference ({other} is not one, or is not built in)"),
        },
        "info" => return info(Path::new(pos(1))),
        "tensor" => return tensor(Path::new(pos(1)), pos(2), a.pos.get(3).map_or(8, |v| v.parse().unwrap())),
        "tokens" => return tokens(Path::new(pos(1)), pos(2).parse().unwrap(), pos(3).parse().unwrap()),
        _ => {}
    }
    let path = Path::new(pos(1));
    macro_rules! run {
        ($backend:expr) => {
            match cmd {
                "generate" => generate(&a, path, $backend),
                "calibrate" => calibrate(&a, path, $backend),
                "bench" => bench(&a, path, $backend),
                "ppl" => ppl(&a, path, Path::new(pos(2)), $backend),
                _ => usage(),
            }
        };
    }
    match a.opt("backend").unwrap_or(DEFAULT_BACKEND) {
        #[cfg(target_arch = "x86_64")]
        "cpu" if det::x86::X86::available() => run!(det::x86::X86::new()),
        "cpu" | "ref" => run!(det::cpu::Cpu),
        #[cfg(target_os = "macos")]
        "metal" => run!(det::metal::Metal::new()),
        #[cfg(target_os = "linux")]
        "cuda" => run!(det::cuda::Cuda::new()),
        other => panic!("backend {other} is not built in"),
    }
}
