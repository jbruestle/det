//! Model hyperparameters, read from the GGUF metadata of a supported architecture.

use crate::gguf::Gguf;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayerKind {
    Attention,
    Gdn,
}

#[derive(Clone, Debug)]
pub struct Config {
    pub arch: String,
    pub hidden: usize,
    pub layers: Vec<LayerKind>,
    pub vocab: usize,
    pub eps: f32,
    pub n_heads: usize,
    pub n_kv: usize,
    pub head_dim: usize,
    pub rot_dim: usize,
    pub rope_theta: f64,
    pub intermediate: usize,
    pub gdn_k_heads: usize,
    pub gdn_v_heads: usize,
    pub gdn_dk: usize,
    pub gdn_dv: usize,
    pub conv_k: usize,
}

impl Config {
    pub fn from_gguf(g: &Gguf) -> Result<Self, String> {
        let arch = g
            .str("general.architecture")
            .ok_or("no general.architecture")?
            .to_string();
        if arch != "qwen35" {
            return Err(format!(
                "architecture {arch} is not supported (qwen35 only)"
            ));
        }
        let key = |s: &str| format!("{arch}.{s}");
        let u = |s: &str| {
            g.u64(&key(s))
                .map(|v| v as usize)
                .ok_or_else(|| format!("missing {}", key(s)))
        };
        let n_layers =
            u("block_count")? - g.u64(&key("nextn_predict_layers")).unwrap_or(0) as usize;
        let interval = u("full_attention_interval")?;
        let layers = (0..n_layers)
            .map(|i| {
                if (i + 1) % interval == 0 {
                    LayerKind::Attention
                } else {
                    LayerKind::Gdn
                }
            })
            .collect();
        let vocab = match g.tensor("token_embd.weight") {
            Some(t) => t.dims[1],
            None => g
                .get("tokenizer.ggml.tokens")
                .and_then(|v| v.as_array())
                .ok_or("no token_embd.weight and no tokenizer tokens")?
                .len(),
        };
        let gdn_dk = u("ssm.state_size")?;
        let gdn_v_heads = u("ssm.time_step_rank")?;
        Ok(Config {
            hidden: u("embedding_length")?,
            layers,
            vocab,
            eps: g
                .f32(&key("attention.layer_norm_rms_epsilon"))
                .ok_or("missing eps")?,
            n_heads: u("attention.head_count")?,
            n_kv: u("attention.head_count_kv")?,
            head_dim: u("attention.key_length")?,
            rot_dim: u("rope.dimension_count")?,
            rope_theta: g
                .f32(&key("rope.freq_base"))
                .ok_or("missing rope.freq_base")? as f64,
            intermediate: u("feed_forward_length")?,
            gdn_k_heads: u("ssm.group_count")?,
            gdn_v_heads,
            gdn_dk,
            gdn_dv: u("ssm.inner_size")? / gdn_v_heads,
            conv_k: u("ssm.conv_kernel")?,
            arch,
        })
    }

    pub fn is_attention(&self, layer: usize) -> bool {
        self.layers[layer] == LayerKind::Attention
    }
    pub fn n_attention_layers(&self) -> usize {
        self.layers
            .iter()
            .filter(|l| **l == LayerKind::Attention)
            .count()
    }
    pub fn conv_dim(&self) -> usize {
        2 * self.gdn_k_heads * self.gdn_dk + self.gdn_v_heads * self.gdn_dv
    }
}
