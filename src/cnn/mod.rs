//! Convolutional networks (CNNs), the second kind of model of the MLP mode.
//!
//! CNNs train through candle, a tensor library with automatic gradients
//! (ADR 0007). candle computes some backward passes wrongly on the CPU, so the
//! crate reaches it through one file only: `ops`. That file wraps candle's
//! tensor in its own type and offers only operations whose gradients we
//! checked against finite differences. No other file names candle; a test in
//! `ops` scans the sources for it.
//!
//! A CNN genome (`genome`) is the `Arch::Cnn` of a `Genome`: a stem, stages and a head.
//! `shape` infers the layers it describes, with their shapes, parameters and FLOPs; this
//! is the shape check of `Genome::validate`.
#![forbid(unsafe_code)]

pub mod genome;
pub mod ops;
pub mod shape;
