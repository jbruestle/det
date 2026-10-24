//! A byte-level BPE tokenizer built from the GGUF `tokenizer.ggml.*` metadata.

use crate::gguf::{Gguf, Value};
use tokenizers::models::bpe::BPE;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::pre_tokenizers::sequence::Sequence;
use tokenizers::pre_tokenizers::split::{Split, SplitPattern};
use tokenizers::pre_tokenizers::PreTokenizerWrapper;
use tokenizers::{AddedToken, Tokenizer};

const QWEN_PATTERN: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

pub struct TokenizerInfo {
    pub tokenizer: Tokenizer,
    pub eos: Vec<u32>,
}

pub fn from_gguf(g: &Gguf) -> Result<TokenizerInfo, String> {
    let model = g.str("tokenizer.ggml.model").unwrap_or("");
    if model != "gpt2" {
        return Err(format!("tokenizer model {model:?} is not byte-level BPE"));
    }
    let tokens = g.get("tokenizer.ggml.tokens").and_then(Value::as_array).ok_or("no tokenizer.ggml.tokens")?;
    let types = g.get("tokenizer.ggml.token_type").and_then(Value::as_array).ok_or("no tokenizer.ggml.token_type")?;
    let merges = g.get("tokenizer.ggml.merges").and_then(Value::as_array).ok_or("no tokenizer.ggml.merges")?;
    let vocab: tokenizers::models::bpe::Vocab = tokens.iter().enumerate().map(|(i, t)| (t.as_str().unwrap().to_string(), i as u32)).collect();
    let merges: Vec<(String, String)> = merges
        .iter()
        .map(|m| {
            let s = m.as_str().unwrap();
            let (a, b) = s.split_once(' ').unwrap();
            (a.to_string(), b.to_string())
        })
        .collect();
    let bpe = BPE::builder().vocab_and_merges(vocab, merges).byte_fallback(false).build().map_err(|e| e.to_string())?;
    let mut tk = Tokenizer::new(bpe);
    let split = Split::new(SplitPattern::Regex(QWEN_PATTERN.to_string()), tokenizers::SplitDelimiterBehavior::Isolated, false)
        .map_err(|e| e.to_string())?;
    let byte_level = ByteLevel::new(false, false, false);
    tk.with_pre_tokenizer(Some(Sequence::new(vec![PreTokenizerWrapper::Split(split), PreTokenizerWrapper::ByteLevel(byte_level)])));
    tk.with_decoder(Some(tokenizers::decoders::byte_level::ByteLevel::new(false, false, false)));
    let special: Vec<AddedToken> = tokens
        .iter()
        .zip(types)
        .filter(|(_, ty)| matches!(ty, Value::I32(3) | Value::I32(4)))
        .map(|(t, _)| AddedToken::from(t.as_str().unwrap(), true))
        .collect();
    tk.add_special_tokens(&special);
    let mut eos = Vec::new();
    if let Some(id) = g.u64("tokenizer.ggml.eos_token_id") {
        eos.push(id as u32);
    }
    for name in ["<|endoftext|>", "<|im_end|>"] {
        if let Some(id) = tk.token_to_id(name) {
            if !eos.contains(&id) {
                eos.push(id);
            }
        }
    }
    Ok(TokenizerInfo { tokenizer: tk, eos })
}

/// The prompt for a single user turn, rendered with the model's own chat template
/// (`tokenizer.chat_template`, Jinja as the HF transformers convention: `messages`,
/// `add_generation_prompt`, `enable_thinking`); ChatML when the file has none.
pub fn chat_prompt(g: &Gguf, user: &str, thinking: bool) -> Result<String, String> {
    let Some(template) = g.str("tokenizer.chat_template") else {
        return Ok(format!("<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"));
    };
    let mut env = minijinja::Environment::new();
    env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
    env.add_function("raise_exception", |msg: String| -> Result<minijinja::Value, minijinja::Error> {
        Err(minijinja::Error::new(minijinja::ErrorKind::InvalidOperation, msg))
    });
    env.add_template("chat", template).map_err(|e| e.to_string())?;
    let messages = vec![minijinja::context! { role => "user", content => user }];
    env.get_template("chat")
        .unwrap()
        .render(minijinja::context! { messages => messages, add_generation_prompt => true, enable_thinking => thinking })
        .map_err(|e| format!("{e:#}"))
}
