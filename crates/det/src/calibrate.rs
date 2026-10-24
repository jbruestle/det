//! Value-cache calibration: the int16 KV cache stores V with one static scale per (layer, kv head),
//! set to 8× the largest |v| seen on a calibration run so that real inputs have headroom.

use crate::backend::Backend;
use crate::gguf::Value;
use crate::hooks::{Ctx, Observer, Tap};
use crate::model::{Model, V_SCALE_KEY};
use std::path::Path;

pub const HEADROOM: f32 = 8.0;

struct AbsMax {
    n_kv: usize,
    hd: usize,
    /// Indexed by attention layer order, then kv head.
    layers: Vec<usize>,
    absmax: Vec<f32>,
}

impl Observer for AbsMax {
    fn observe(&mut self, ctx: &Ctx, tap: Tap) {
        if let Tap::AttnV { layer, v } = tap {
            let li = self.layers.iter().position(|&l| l == layer).unwrap();
            let v = ctx.read(v, 0, ctx.t * self.n_kv * self.hd);
            for t in 0..ctx.t {
                for h in 0..self.n_kv {
                    let row = &v[(t * self.n_kv + h) * self.hd..(t * self.n_kv + h + 1) * self.hd];
                    let m = row.iter().fold(0f32, |m, x| m.max(x.abs()));
                    let slot = &mut self.absmax[li * self.n_kv + h];
                    *slot = slot.max(m);
                }
            }
        }
    }
}

/// English text plus random ids, as the calibration input.
pub fn calibration_ids(tk: &tokenizers::Tokenizer, vocab: usize, n_random: usize) -> Vec<i32> {
    const TEXT: &str = "The quick brown fox jumps over the lazy dog. In 1969, astronauts landed on the Moon and returned \
safely to Earth, an achievement that required the coordinated work of hundreds of thousands of people. \
def fibonacci(n):\n    if n < 2:\n        return n\n    return fibonacci(n - 1) + fibonacci(n - 2)\n\n\
Photosynthesis converts light energy into chemical energy, producing oxygen as a by-product. \
The committee will meet on Tuesday to discuss the proposed amendments to the budget; please bring the revised figures. \
東京は日本の首都であり、世界で最も人口の多い都市圏の一つです。 \
La Révolution française a profondément transformé la société et la politique du pays. \
SELECT name, COUNT(*) FROM orders WHERE total > 100 GROUP BY name ORDER BY 2 DESC;\n\
Water boils at 100 degrees Celsius at sea level, but at higher altitudes the boiling point drops because the air pressure is lower. \
<|im_start|>user\nWrite a haiku about autumn.<|im_end|>\n<|im_start|>assistant\nCrisp leaves drift and fall,\nGolden light on quiet paths,\nThe year exhales slow.<|im_end|>\n";
    let mut ids: Vec<i32> = tk.encode(TEXT, false).expect("encode").get_ids().iter().map(|&v| v as i32).collect();
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    for _ in 0..n_random {
        let (next, z) = crate::spec::splitmix64(state);
        state = next;
        ids.push((z % vocab as u64) as i32);
    }
    ids
}

/// Runs the calibration input through `m` and returns the per-(attention layer, kv head) value scales.
pub fn v_scales<B: Backend>(m: &mut Model<B>, ids: &[i32]) -> Result<Vec<f32>, String> {
    let layers: Vec<usize> = (0..m.cfg.layers.len()).filter(|&l| m.cfg.is_attention(l)).collect();
    let mut obs = AbsMax { n_kv: m.cfg.n_kv, hd: m.cfg.head_dim, absmax: vec![0.0; layers.len() * m.cfg.n_kv], layers };
    m.reset();
    m.prefill(ids, Some(&mut obs))?;
    Ok(obs.absmax.iter().map(|&a| if a == 0.0 { crate::model::DEFAULT_V_SCALE } else { a * HEADROOM / 32767.0 }).collect())
}

/// Writes `scales` into the metadata of the model's file in place (converted files reserve the room);
/// returns false when the header has no room, in which case `write_copy` is the way.
pub fn write_in_place<B: Backend>(m: &Model<B>, scales: &[f32], path: &Path) -> std::io::Result<bool> {
    let value = Value::Array(scales.iter().map(|&s| Value::F32(s)).collect());
    m.gguf.set_metadata_in_place(path, V_SCALE_KEY, value)
}

/// Writes a copy of the model's file with `scales` in its metadata to `out`.
pub fn write_copy<B: Backend>(m: &Model<B>, scales: &[f32], out: &Path) -> std::io::Result<()> {
    let value = Value::Array(scales.iter().map(|&s| Value::F32(s)).collect());
    m.gguf.write_with_metadata(V_SCALE_KEY, value, out)
}
