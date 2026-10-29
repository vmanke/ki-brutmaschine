//! Cascade breeding (Stage 2c, ADR 0005): Darwin-Gödel-style search over Cascades of
//! Workers that solve grid puzzles, judge each other blind, and are recombined.
//!
//! Everything here is pure computation: no files, no network, no processes, no clock.
//! Model calls go through the `ModelBackend` trait only; the binary `brut-breed` does the
//! I/O. The evaluator side (puzzle generation, ground truth, scoring) never depends on a
//! Cascade's content, so no Cascade can change how it is judged.
#![forbid(unsafe_code)]

pub mod arena;
pub mod circuit_text;
pub mod designer;
pub mod domain;
pub mod genome;
pub mod observe;
pub mod puzzle;
pub mod qubit;
pub mod sim;
pub mod variation;
