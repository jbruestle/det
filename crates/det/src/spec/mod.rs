//! The numerics of det, written as plainly as possible: this module is the executable form of
//! `docs/SPEC.md`. Every other implementation (the CPU backend's batched loops, the GPU kernels) must
//! reproduce these functions bit for bit.
//!
//! Every expression is written in the spec's op order, so lints that would rewrite arithmetic are off.
#![allow(
    clippy::approx_constant,
    clippy::excessive_precision,
    clippy::assign_op_pattern,
    clippy::needless_range_loop,
    clippy::manual_clamp,
    clippy::too_many_arguments
)]

pub mod layers;
pub mod prim;
pub mod quant;
pub mod sample;
pub mod tensor;

pub use layers::*;
pub use prim::*;
pub use quant::*;
pub use sample::{argmax, sample, sample_with_draw, weights};
pub use tensor::*;
