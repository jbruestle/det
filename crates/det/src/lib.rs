//! det: deterministic, bit-exact LLM inference over stock GGUF files.

pub mod backend;
pub mod calibrate;
pub mod config;
pub mod convert;
pub mod cpu;
pub mod gguf;
pub mod hooks;
#[cfg(feature = "cuda")]
pub mod cuda;
#[cfg(all(feature = "metal", target_os = "macos"))]
pub mod metal;
pub mod model;
pub mod quantize;
pub mod qwen35;
pub mod spec;
pub mod tokenizer;

pub use backend::{Backend, Buf};
pub use config::Config;
pub use model::{Model, Sampling};
