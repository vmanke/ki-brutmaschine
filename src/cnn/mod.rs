//! Convolutional networks (CNNs), the second kind of model of the MLP mode.
//!
//! CNNs train through candle, a tensor library with automatic gradients
//! (ADR 0007). candle computes some backward passes wrongly on the CPU, so the
//! crate reaches it through one file only: `ops`. That file wraps candle's
//! tensor in its own type and offers only operations whose gradients we
//! checked against finite differences. No other file names candle; a test in
//! `ops` scans the sources for it.
#![forbid(unsafe_code)]

pub mod ops;
