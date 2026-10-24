//! GGUF container: header, metadata key/values and tensor table over a memory-mapped file.

use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::File;
use std::io;
use std::path::Path;

#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    Str(String),
    Array(Vec<Value>),
    U64(u64),
    I64(i64),
    F64(f64),
}

impl Value {
    pub fn as_u64(&self) -> Option<u64> {
        match *self {
            Value::U8(v) => Some(v as u64),
            Value::U16(v) => Some(v as u64),
            Value::U32(v) => Some(v as u64),
            Value::U64(v) => Some(v),
            Value::I8(v) if v >= 0 => Some(v as u64),
            Value::I16(v) if v >= 0 => Some(v as u64),
            Value::I32(v) if v >= 0 => Some(v as u64),
            Value::I64(v) if v >= 0 => Some(v as u64),
            _ => None,
        }
    }
    pub fn as_f32(&self) -> Option<f32> {
        match *self {
            Value::F32(v) => Some(v),
            Value::F64(v) => Some(v as f32),
            _ => self.as_u64().map(|v| v as f32),
        }
    }
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::Str(s) => Some(s),
            _ => None,
        }
    }
    pub fn as_array(&self) -> Option<&[Value]> {
        match self {
            Value::Array(a) => Some(a),
            _ => None,
        }
    }
    fn type_id(&self) -> u32 {
        match self {
            Value::U8(_) => 0,
            Value::I8(_) => 1,
            Value::U16(_) => 2,
            Value::I16(_) => 3,
            Value::U32(_) => 4,
            Value::I32(_) => 5,
            Value::F32(_) => 6,
            Value::Bool(_) => 7,
            Value::Str(_) => 8,
            Value::Array(_) => 9,
            Value::U64(_) => 10,
            Value::I64(_) => 11,
            Value::F64(_) => 12,
        }
    }
}

/// The tensor types det accepts: float vectors and embeddings, and det's own quantized linears
/// (type ids 0xDE04, 0xDE06, 0xDE08: `[codes][c][m]` rows, SPEC §2).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GgmlType {
    F32,
    F16,
    BF16,
    Det4,
    Det6,
    Det8,
}

impl GgmlType {
    pub fn from_id(id: u32) -> Option<Self> {
        Some(match id {
            0 => GgmlType::F32,
            1 => GgmlType::F16,
            30 => GgmlType::BF16,
            0xDE04 => GgmlType::Det4,
            0xDE06 => GgmlType::Det6,
            0xDE08 => GgmlType::Det8,
            _ => return None,
        })
    }
    pub fn id(self) -> u32 {
        match self {
            GgmlType::F32 => 0,
            GgmlType::F16 => 1,
            GgmlType::BF16 => 30,
            GgmlType::Det4 => 0xDE04,
            GgmlType::Det6 => 0xDE06,
            GgmlType::Det8 => 0xDE08,
        }
    }
    pub fn det(bits: u32) -> Self {
        match bits {
            4 => GgmlType::Det4,
            6 => GgmlType::Det6,
            8 => GgmlType::Det8,
            _ => panic!("unsupported bit width {bits}"),
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            GgmlType::F32 => "F32",
            GgmlType::F16 => "F16",
            GgmlType::BF16 => "BF16",
            GgmlType::Det4 => "DET_W4",
            GgmlType::Det6 => "DET_W6",
            GgmlType::Det8 => "DET_W8",
        }
    }
    /// Code width of a det linear.
    pub fn bits(self) -> Option<u32> {
        match self {
            GgmlType::Det4 => Some(4),
            GgmlType::Det6 => Some(6),
            GgmlType::Det8 => Some(8),
            _ => None,
        }
    }
    pub fn row_bytes(self, k: usize) -> usize {
        match self {
            GgmlType::F32 => 4 * k,
            GgmlType::F16 | GgmlType::BF16 => 2 * k,
            _ => crate::spec::det_row_bytes(self.bits().unwrap(), k),
        }
    }
}

pub fn ggml_type_name(id: u32) -> String {
    if let Some(t) = GgmlType::from_id(id) {
        return t.name().to_string();
    }
    const NAMES: [&str; 31] = [
        "F32", "F16", "Q4_0", "Q4_1", "?4", "?5", "Q5_0", "Q5_1", "Q8_0", "Q8_1", "Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K", "Q8_K",
        "IQ2_XXS", "IQ2_XS", "IQ3_XXS", "IQ1_S", "IQ4_NL", "IQ3_S", "IQ2_S", "IQ4_XS", "I8", "I16", "I32", "I64", "F64", "IQ1_M", "BF16",
    ];
    NAMES.get(id as usize).map_or_else(|| format!("type{id}"), |s| s.to_string())
}

#[derive(Clone, Debug)]
pub struct TensorInfo {
    pub name: String,
    /// ggml order: `dims[0]` is the fastest-varying (the row length of a matrix).
    pub dims: Vec<usize>,
    pub type_id: u32,
    pub offset: usize,
}

impl TensorInfo {
    pub fn ty(&self) -> Option<GgmlType> {
        GgmlType::from_id(self.type_id)
    }
    pub fn n_elems(&self) -> usize {
        self.dims.iter().product()
    }
    pub fn bytes(&self) -> usize {
        let ty = self.ty().unwrap_or_else(|| panic!("tensor {} has unsupported type {}", self.name, ggml_type_name(self.type_id)));
        ty.row_bytes(self.dims[0]) * self.dims[1..].iter().product::<usize>()
    }
}

pub struct Gguf {
    map: Mmap,
    pub kv: Vec<(String, Value)>,
    index: HashMap<String, usize>,
    pub tensors: Vec<TensorInfo>,
    tindex: HashMap<String, usize>,
    pub data_start: usize,
    pub alignment: usize,
}

struct Cursor<'a> {
    b: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn take(&mut self, n: usize) -> io::Result<&'a [u8]> {
        if self.pos + n > self.b.len() {
            return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "truncated GGUF header"));
        }
        let s = &self.b[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }
    fn u32(&mut self) -> io::Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> io::Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn string(&mut self) -> io::Result<String> {
        let n = self.u64()? as usize;
        Ok(String::from_utf8_lossy(self.take(n)?).into_owned())
    }
    fn value(&mut self, ty: u32) -> io::Result<Value> {
        Ok(match ty {
            0 => Value::U8(self.take(1)?[0]),
            1 => Value::I8(self.take(1)?[0] as i8),
            2 => Value::U16(u16::from_le_bytes(self.take(2)?.try_into().unwrap())),
            3 => Value::I16(i16::from_le_bytes(self.take(2)?.try_into().unwrap())),
            4 => Value::U32(self.u32()?),
            5 => Value::I32(self.u32()? as i32),
            6 => Value::F32(f32::from_bits(self.u32()?)),
            7 => Value::Bool(self.take(1)?[0] != 0),
            8 => Value::Str(self.string()?),
            9 => {
                let et = self.u32()?;
                let n = self.u64()? as usize;
                let mut v = Vec::with_capacity(n.min(1 << 20));
                for _ in 0..n {
                    v.push(self.value(et)?);
                }
                Value::Array(v)
            }
            10 => Value::U64(self.u64()?),
            11 => Value::I64(self.u64()? as i64),
            12 => Value::F64(f64::from_bits(self.u64()?)),
            _ => return Err(io::Error::new(io::ErrorKind::InvalidData, format!("unknown GGUF value type {ty}"))),
        })
    }
}

impl Gguf {
    pub fn open(path: &Path) -> io::Result<Self> {
        let file = File::open(path)?;
        let map = unsafe { Mmap::map(&file)? };
        let mut c = Cursor { b: &map, pos: 0 };
        if c.take(4)? != b"GGUF" {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "not a GGUF file"));
        }
        let version = c.u32()?;
        if version != 3 && version != 2 {
            return Err(io::Error::new(io::ErrorKind::InvalidData, format!("unsupported GGUF version {version}")));
        }
        let n_tensors = c.u64()? as usize;
        let n_kv = c.u64()? as usize;
        let mut kv = Vec::with_capacity(n_kv);
        for _ in 0..n_kv {
            let key = c.string()?;
            let ty = c.u32()?;
            kv.push((key, c.value(ty)?));
        }
        let mut tensors = Vec::with_capacity(n_tensors);
        for _ in 0..n_tensors {
            let name = c.string()?;
            let nd = c.u32()? as usize;
            let mut dims = Vec::with_capacity(nd);
            for _ in 0..nd {
                dims.push(c.u64()? as usize);
            }
            let type_id = c.u32()?;
            let offset = c.u64()? as usize;
            tensors.push(TensorInfo { name, dims, type_id, offset });
        }
        let index = kv.iter().enumerate().map(|(i, (k, _))| (k.clone(), i)).collect();
        let tindex = tensors.iter().enumerate().map(|(i, t)| (t.name.clone(), i)).collect();
        let alignment = kv.iter().find(|(k, _)| k == "general.alignment").and_then(|(_, v)| v.as_u64()).unwrap_or(32) as usize;
        let data_start = c.pos.div_ceil(alignment) * alignment;
        Ok(Gguf { map, kv, index, tensors, tindex, data_start, alignment })
    }

    pub fn get(&self, key: &str) -> Option<&Value> {
        self.index.get(key).map(|&i| &self.kv[i].1)
    }
    pub fn u64(&self, key: &str) -> Option<u64> {
        self.get(key).and_then(Value::as_u64)
    }
    pub fn f32(&self, key: &str) -> Option<f32> {
        self.get(key).and_then(Value::as_f32)
    }
    pub fn str(&self, key: &str) -> Option<&str> {
        self.get(key).and_then(Value::as_str)
    }
    pub fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tindex.get(name).map(|&i| &self.tensors[i])
    }
    pub fn has_tensor(&self, name: &str) -> bool {
        self.tindex.contains_key(name)
    }
    /// The raw bytes of a tensor.
    pub fn data(&self, t: &TensorInfo) -> &[u8] {
        let start = self.data_start + t.offset;
        &self.map[start..start + t.bytes()]
    }
    pub fn file_bytes(&self) -> &[u8] {
        &self.map
    }
    /// The whole mapping (page aligned), for backends that wrap it as a device buffer.
    pub fn mapping(&self) -> &Mmap {
        &self.map
    }
}

/// Serializes a GGUF header (metadata + tensor table) for `tensors` laid out at their stated offsets.
pub fn write_header(out: &mut Vec<u8>, kv: &[(String, Value)], tensors: &[TensorInfo], alignment: usize) {
    fn string(out: &mut Vec<u8>, s: &str) {
        out.extend_from_slice(&(s.len() as u64).to_le_bytes());
        out.extend_from_slice(s.as_bytes());
    }
    fn value(out: &mut Vec<u8>, v: &Value) {
        match v {
            Value::U8(x) => out.push(*x),
            Value::I8(x) => out.push(*x as u8),
            Value::U16(x) => out.extend_from_slice(&x.to_le_bytes()),
            Value::I16(x) => out.extend_from_slice(&x.to_le_bytes()),
            Value::U32(x) => out.extend_from_slice(&x.to_le_bytes()),
            Value::I32(x) => out.extend_from_slice(&x.to_le_bytes()),
            Value::F32(x) => out.extend_from_slice(&x.to_bits().to_le_bytes()),
            Value::Bool(x) => out.push(*x as u8),
            Value::Str(s) => string(out, s),
            Value::Array(a) => {
                let et = a.first().map_or(0, Value::type_id);
                out.extend_from_slice(&et.to_le_bytes());
                out.extend_from_slice(&(a.len() as u64).to_le_bytes());
                for e in a {
                    assert_eq!(e.type_id(), et, "mixed array element types");
                    value(out, e);
                }
            }
            Value::U64(x) => out.extend_from_slice(&x.to_le_bytes()),
            Value::I64(x) => out.extend_from_slice(&x.to_le_bytes()),
            Value::F64(x) => out.extend_from_slice(&x.to_bits().to_le_bytes()),
        }
    }
    out.extend_from_slice(b"GGUF");
    out.extend_from_slice(&3u32.to_le_bytes());
    out.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    out.extend_from_slice(&(kv.len() as u64).to_le_bytes());
    for (k, v) in kv {
        string(out, k);
        out.extend_from_slice(&v.type_id().to_le_bytes());
        value(out, v);
    }
    for t in tensors {
        string(out, &t.name);
        out.extend_from_slice(&(t.dims.len() as u32).to_le_bytes());
        for d in &t.dims {
            out.extend_from_slice(&(*d as u64).to_le_bytes());
        }
        out.extend_from_slice(&t.type_id.to_le_bytes());
        out.extend_from_slice(&(t.offset as u64).to_le_bytes());
    }
    let pad = out.len().div_ceil(alignment) * alignment - out.len();
    out.extend(std::iter::repeat_n(0u8, pad));
}

impl Gguf {
    fn header_with(&self, key: &str, value: Value) -> Vec<u8> {
        let mut kv: Vec<(String, Value)> = self.kv.iter().filter(|(k, _)| k != key).cloned().collect();
        kv.push((key.to_string(), value));
        let mut header = Vec::new();
        write_header(&mut header, &kv, &self.tensors, self.alignment);
        header
    }

    /// Sets `key` in the file's own header when the new header fits before the tensor data (true), or
    /// does nothing (false). `det convert` reserves the space for `det.v_scale`.
    pub fn set_metadata_in_place(&self, path: &Path, key: &str, value: Value) -> io::Result<bool> {
        let header = self.header_with(key, value);
        if header.len() > self.data_start {
            return Ok(false);
        }
        use std::io::{Seek, SeekFrom, Write};
        let mut f = std::fs::OpenOptions::new().write(true).open(path)?;
        f.seek(SeekFrom::Start(0))?;
        f.write_all(&header)?;
        f.write_all(&vec![0u8; self.data_start - header.len()])?;
        f.flush()?;
        Ok(true)
    }

    /// Writes a copy of this file with `key` set to `value` (replacing an existing key) to `out`.
    pub fn write_with_metadata(&self, key: &str, value: Value, out: &Path) -> io::Result<()> {
        let header = self.header_with(key, value);
        let mut f = std::io::BufWriter::new(File::create(out)?);
        use std::io::Write;
        f.write_all(&header)?;
        let data = &self.map[self.data_start..];
        for chunk in data.chunks(1 << 26) {
            f.write_all(chunk)?;
        }
        f.flush()
    }
}
