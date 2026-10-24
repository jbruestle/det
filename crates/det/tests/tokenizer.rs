//! The tokenizer built from the GGUF metadata against the HF tokenizer.json of the same model.

use det::gguf::Gguf;
use std::path::PathBuf;

fn find(pattern: &str) -> Option<PathBuf> {
    let home = std::env::var("HOME").ok()?;
    let mut parts = pattern.split('*');
    let base = PathBuf::from(home).join(parts.next()?);
    let rest = parts.next()?;
    for e in std::fs::read_dir(base).ok()? {
        let p = e.ok()?.path().join(rest.trim_start_matches('/'));
        if p.exists() {
            return Some(p);
        }
    }
    None
}

#[test]
fn matches_hf_tokenizer() {
    let gguf = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models/Qwen3.5-4B-det-w6.gguf");
    let hf = find(".cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/*/tokenizer.json");
    let (Some(hf), true) = (hf, gguf.exists()) else {
        eprintln!("skipped: model or HF tokenizer not present");
        return;
    };
    let g = Gguf::open(&gguf).unwrap();
    let ours = det::tokenizer::from_gguf(&g).unwrap().tokenizer;
    let theirs = tokenizers::Tokenizer::from_file(&hf).unwrap();
    let texts = [
        "The capital of France is",
        "Hello, world!  Multiple   spaces\tand\ttabs\n\nand newlines.\n",
        "<|im_start|>user\nWrite a haiku about autumn.<|im_end|>\n<|im_start|>assistant\n",
        "東京は日本の首都であり、世界で最も人口の多い都市圏の一つです。",
        "def fibonacci(n):\n    if n < 2:\n        return n\n    return fibonacci(n - 1) + fibonacci(n - 2)\n",
        "Numbers: 1234567890, 3.14159, -42, 1e10; emoji 🚀🔥 and accents: café naïve résumé",
        "it's we're they'd I'LL CAN'T",
        "    leading spaces and trailing   ",
    ];
    for t in texts {
        let a = ours.encode(t, false).unwrap().get_ids().to_vec();
        let b = theirs.encode(t, false).unwrap().get_ids().to_vec();
        assert_eq!(a, b, "tokenization differs for {t:?}");
        assert_eq!(ours.decode(&a, false).unwrap(), t, "decode roundtrip for {t:?}");
    }
}
