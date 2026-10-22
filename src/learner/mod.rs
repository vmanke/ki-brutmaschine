//! Learner mode: candidates are small learning algorithms in a closed, interpreted DSL.
#![forbid(unsafe_code)]

pub mod arc_cells;
pub mod baselines;
pub mod dsl;
pub mod families;
pub mod fec;
pub mod grammar;
pub mod interp;
pub mod llm_proposer;
pub mod mutate;
pub mod protocol;
pub mod seeds;
pub mod stats;
pub mod types;
pub mod wire;

pub use types::*;
