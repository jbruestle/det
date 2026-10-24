//! `det convert`: a float GGUF (BF16/F16/F32 tensors as llama.cpp's converter writes them) into a det
//! GGUF: the same metadata and vectors, every linear quantized into det rows, the output head always
//! present, the rotary tables as tensors, and the GDN value heads reordered so that v head `h` reads
//! k head `h / (n_v / n_k)` (SPEC §5.2).

use crate::config::Config;
use crate::gguf::{write_header, GgmlType, Gguf, TensorInfo, Value};
use crate::quantize::quantize_rows;
use crate::spec::dequant_row;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

/// A GGUF model, possibly split over several files.
pub struct Source {
    pub parts: Vec<Gguf>,
}

impl Source {
    pub fn open(path: &Path) -> io::Result<Self> {
        let first = Gguf::open(path)?;
        let count = first.u64("split.count").unwrap_or(1) as usize;
        let mut parts = vec![first];
        if count > 1 {
            let name = path.file_name().unwrap().to_string_lossy().into_owned();
            let marker = name.find("-00001-of-").ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "split GGUF without -00001-of- in its name"))?;
            for i in 2..=count {
                let sibling = format!("{}-{i:05}-of-{}", &name[..marker], &name[marker + 10..]);
                parts.push(Gguf::open(&path.with_file_name(sibling))?);
            }
        }
        Ok(Source { parts })
    }
    pub fn kv(&self) -> &[(String, Value)] {
        &self.parts[0].kv
    }
    pub fn tensor(&self, name: &str) -> Option<(&Gguf, &TensorInfo)> {
        self.parts.iter().find_map(|g| g.tensor(name).map(|t| (g, t)))
    }
    /// A tensor as fp32, row-major with `dims[0]` fastest.
    pub fn f32s(&self, name: &str) -> Vec<f32> {
        let (g, t) = self.tensor(name).unwrap_or_else(|| panic!("missing tensor {name}"));
        let ty = t.ty().unwrap_or_else(|| panic!("{name}: unsupported source type"));
        let k = t.dims[0];
        let rows = t.n_elems() / k;
        let data = g.data(t);
        let rb = ty.row_bytes(k);
        let mut out = vec![0f32; rows * k];
        for r in 0..rows {
            dequant_row(ty, &data[r * rb..(r + 1) * rb], k, &mut out[r * k..(r + 1) * k]);
        }
        out
    }
}

/// Bit widths: a default and overrides per linear name (`ffn_down`, `attn_q`, `ssm_out`, `output`, …),
/// parsed from `6,output=8,ffn_down=8`.
pub struct Bits {
    pub default: u32,
    pub per: Vec<(String, u32)>,
}

impl Bits {
    pub fn parse(s: &str) -> Result<Self, String> {
        let mut b = Bits { default: 6, per: Vec::new() };
        for item in s.split(',').map(str::trim).filter(|v| !v.is_empty()) {
            let (name, bits) = match item.split_once('=') {
                Some((n, v)) => (Some(n), v),
                None => (None, item),
            };
            let bits: u32 = bits.parse().map_err(|_| format!("bad bit width {bits:?}"))?;
            if !matches!(bits, 4 | 6 | 8) {
                return Err(format!("bit width {bits} is not 4, 6 or 8"));
            }
            match name {
                Some(n) => b.per.push((n.to_string(), bits)),
                None => b.default = bits,
            }
        }
        Ok(b)
    }
    pub fn of(&self, linear: &str) -> u32 {
        self.per.iter().find(|(n, _)| n == linear).map_or(self.default, |(_, b)| *b)
    }
}

impl Default for Bits {
    fn default() -> Self {
        Bits::parse(DEFAULT_BITS).unwrap()
    }
}

/// W6 everywhere except the output head and the GDN a/b projections.
pub const DEFAULT_BITS: &str = "6,output=8,ssm_alpha=8,ssm_beta=8";

pub struct Options {
    pub bits: Bits,
    /// Rotary table length; `None` takes the model's context length.
    pub rope_positions: Option<usize>,
}

enum Gen {
    /// The source bytes and type as they are.
    Copy(String),
    /// A float tensor, its outer index permuted (`perm[new] = old`).
    Vec(String, Vec<usize>),
    /// A linear quantized to det rows after permuting its rows (N) and columns (K).
    Quant { name: String, bits: u32, rows: Option<Vec<usize>>, cols: Option<Vec<usize>> },
    Rope { cos: bool },
}

struct Out {
    name: String,
    dims: Vec<usize>,
    ty: GgmlType,
    gen: Gen,
}

fn identity(n: usize) -> Vec<usize> {
    (0..n).collect()
}

/// `perm[new] = old` over `width`-element blocks of the v heads: new head `h'` takes GGUF head
/// `(h' mod rep) · n_k + h' / rep`, whose k head is `h' / rep`.
fn head_perm(n_k: usize, n_v: usize, width: usize, offset: usize, total: usize) -> Vec<usize> {
    let rep = n_v / n_k;
    let mut p = identity(total);
    for h2 in 0..n_v {
        let h = (h2 % rep) * n_k + h2 / rep;
        for j in 0..width {
            p[offset + h2 * width + j] = offset + h * width + j;
        }
    }
    p
}

pub fn convert(src: &Source, out: &Path, opt: &Options) -> io::Result<()> {
    let cfg = Config::from_gguf(&src.parts[0]).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    let (n_k, n_v, dk, dv) = (cfg.gdn_k_heads, cfg.gdn_v_heads, cfg.gdn_dk, cfg.gdn_dv);
    let kd = n_k * dk;
    let conv_dim = cfg.conv_dim();
    let n_pos = opt.rope_positions.unwrap_or_else(|| src.parts[0].u64(&format!("{}.context_length", cfg.arch)).unwrap_or(32768) as usize);
    let dims_of = |name: &str| src.tensor(name).unwrap_or_else(|| panic!("missing tensor {name}")).1.dims.clone();
    let mut plan: Vec<Out> = Vec::new();
    let copy = |plan: &mut Vec<Out>, name: &str| {
        let (_, t) = src.tensor(name).unwrap_or_else(|| panic!("missing tensor {name}"));
        plan.push(Out { name: name.into(), dims: t.dims.clone(), ty: t.ty().unwrap_or_else(|| panic!("{name}: unsupported source type")), gen: Gen::Copy(name.into()) });
    };
    let quant = |plan: &mut Vec<Out>, name: &str, rows: Option<Vec<usize>>, cols: Option<Vec<usize>>| {
        let d = dims_of(name);
        let linear = name.rsplit('.').nth(1).unwrap_or(name);
        let bits = opt.bits.of(linear);
        plan.push(Out { name: name.into(), dims: vec![d[0], d[1]], ty: GgmlType::det(bits), gen: Gen::Quant { name: name.into(), bits, rows, cols } });
    };
    copy(&mut plan, "token_embd.weight");
    copy(&mut plan, "output_norm.weight");
    let head = if src.tensor("output.weight").is_some() { "output.weight" } else { "token_embd.weight" };
    let d = dims_of(head);
    let head_bits = opt.bits.of("output");
    plan.push(Out { name: "output.weight".into(), dims: vec![d[0], d[1]], ty: GgmlType::det(head_bits), gen: Gen::Quant { name: head.into(), bits: head_bits, rows: None, cols: None } });
    for cos in [true, false] {
        plan.push(Out { name: if cos { "rope_cos" } else { "rope_sin" }.into(), dims: vec![cfg.rot_dim / 2, n_pos], ty: GgmlType::F32, gen: Gen::Rope { cos } });
    }
    for i in 0..cfg.layers.len() {
        let n = |s: &str| format!("blk.{i}.{s}");
        copy(&mut plan, &n("attn_norm.weight"));
        copy(&mut plan, &n("post_attention_norm.weight"));
        for s in ["ffn_gate", "ffn_up", "ffn_down"] {
            quant(&mut plan, &n(&format!("{s}.weight")), None, None);
        }
        if cfg.is_attention(i) {
            for s in ["attn_q", "attn_k", "attn_v", "attn_output"] {
                quant(&mut plan, &n(&format!("{s}.weight")), None, None);
            }
            copy(&mut plan, &n("attn_q_norm.weight"));
            copy(&mut plan, &n("attn_k_norm.weight"));
        } else {
            let vrows = head_perm(n_k, n_v, dv, 2 * kd, conv_dim);
            quant(&mut plan, &n("attn_qkv.weight"), Some(vrows.clone()), None);
            quant(&mut plan, &n("attn_gate.weight"), Some(head_perm(n_k, n_v, dv, 0, n_v * dv)), None);
            let heads = head_perm(n_k, n_v, 1, 0, n_v);
            quant(&mut plan, &n("ssm_alpha.weight"), Some(heads.clone()), None);
            quant(&mut plan, &n("ssm_beta.weight"), Some(heads.clone()), None);
            quant(&mut plan, &n("ssm_out.weight"), None, Some(head_perm(n_k, n_v, dv, 0, n_v * dv)));
            for (name, perm) in [("ssm_conv1d.weight", vrows.clone()), ("ssm_a", heads.clone()), ("ssm_dt.bias", heads.clone())] {
                let (_, t) = src.tensor(&n(name)).unwrap_or_else(|| panic!("missing tensor {}", n(name)));
                plan.push(Out { name: n(name), dims: t.dims.clone(), ty: GgmlType::F32, gen: Gen::Vec(n(name), perm) });
            }
            copy(&mut plan, &n("ssm_norm.weight"));
        }
    }

    let align = 32usize;
    let mut infos = Vec::with_capacity(plan.len());
    let mut offset = 0usize;
    for o in &plan {
        let bytes = o.ty.row_bytes(o.dims[0]) * o.dims[1..].iter().product::<usize>();
        infos.push(TensorInfo { name: o.name.clone(), dims: o.dims.clone(), type_id: o.ty.id(), offset });
        offset = (offset + bytes).div_ceil(align) * align;
    }
    let mut kv: Vec<(String, Value)> = src.kv().iter().filter(|(k, _)| !k.starts_with("split.") && k != crate::model::V_SCALE_KEY).cloned().collect();
    let n_scales = cfg.n_attention_layers() * cfg.n_kv;
    kv.push((crate::model::V_SCALE_KEY.to_string(), Value::Array(vec![Value::F32(crate::model::DEFAULT_V_SCALE); n_scales])));
    let mut header = Vec::new();
    write_header(&mut header, &kv, &infos, align);
    let mut f = io::BufWriter::with_capacity(1 << 24, std::fs::File::create(out)?);
    f.write_all(&header)?;
    let mut written = 0usize;
    for (o, info) in plan.iter().zip(&infos) {
        let data: Vec<u8> = match &o.gen {
            Gen::Copy(name) => {
                let (g, t) = src.tensor(name).unwrap();
                g.data(t).to_vec()
            }
            Gen::Vec(name, perm) => {
                let v = src.f32s(name);
                let w = o.dims[0];
                let inner = if o.dims.len() > 1 { w } else { 1 };
                let outer = v.len() / inner;
                assert_eq!(perm.len(), outer, "{name}: permutation length");
                let mut out = Vec::with_capacity(v.len() * 4);
                for i in 0..outer {
                    for &x in &v[perm[i] * inner..(perm[i] + 1) * inner] {
                        out.extend_from_slice(&x.to_le_bytes());
                    }
                }
                out
            }
            Gen::Quant { name, bits, rows, cols } => {
                let (k, n) = (o.dims[0], o.dims[1]);
                let mut w = src.f32s(name);
                if let Some(p) = rows {
                    assert_eq!(p.len(), n);
                    let src_w = w;
                    w = Vec::with_capacity(n * k);
                    for &r in p {
                        w.extend_from_slice(&src_w[r * k..(r + 1) * k]);
                    }
                }
                if let Some(p) = cols {
                    assert_eq!(p.len(), k);
                    let src_w = w;
                    w = Vec::with_capacity(n * k);
                    for r in 0..n {
                        w.extend(p.iter().map(|&c| src_w[r * k + c]));
                    }
                }
                eprintln!("  {name}: {n} x {k} -> W{bits}");
                quantize_rows(&w, n, k, *bits)
            }
            Gen::Rope { cos } => {
                let half = cfg.rot_dim / 2;
                let mut out = Vec::with_capacity(n_pos * half * 4);
                let inv_freq: Vec<f64> = (0..half).map(|i| cfg.rope_theta.powf(-((2 * i) as f64) / cfg.rot_dim as f64)).collect();
                for pos in 0..n_pos {
                    for &f in &inv_freq {
                        let a = pos as f64 * f;
                        out.extend_from_slice(&((if *cos { a.cos() } else { a.sin() }) as f32).to_le_bytes());
                    }
                }
                out
            }
        };
        assert_eq!(written, info.offset, "{}: offset", o.name);
        f.write_all(&data)?;
        written += data.len();
        let pad = written.div_ceil(align) * align - written;
        f.write_all(&vec![0u8; pad])?;
        written += pad;
    }
    f.flush()?;
    eprintln!("wrote {} ({} tensors, {:.2} GB)", out.display(), infos.len(), (header.len() + written) as f64 / 1e9);
    Ok(())
}

pub fn default_output(src: &Path, bits: &Bits) -> PathBuf {
    let stem = src.file_stem().unwrap().to_string_lossy();
    let stem = stem.trim_end_matches("-00001-of-00002").trim_end_matches("-BF16").trim_end_matches("-bf16").trim_end_matches("-F16").trim_end_matches("-f16");
    src.with_file_name(format!("{stem}-det-w{}.gguf", bits.default))
}
