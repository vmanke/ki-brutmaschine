//! Brutmaschine: a safe, reproducible prototype of an AI model incubator.
//!
//! Six components:
//! 1. Variant generator   -> `variation`, `genome`
//! 2. Task environment    -> `task`
//! 3. Scoring             -> `score`
//! 4. Training            -> `train`, `nn`
//! 5. Hidden evaluation   -> `hidden_eval` (+ binary `brut-eval`)
//! 6. Lineage archive     -> `archive`
//!
//! Glue: `config`, `safety`, `evolve`, `util`.

pub mod arc_data;
pub mod archive;
pub mod backend;
pub mod config;
pub mod evolve;
pub mod genome;
pub mod hidden_eval;
pub mod hitl;
pub mod learner;
pub mod learner_hidden;
pub mod ledger;
pub mod lineage_store;
pub mod meta_evolve;
pub mod meta_report;
pub mod nn;
pub mod safety;
pub mod score;
pub mod task;
pub mod train;
pub mod util;
pub mod variation;
