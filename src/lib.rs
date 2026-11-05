//! Brutmaschine: a safe, reproducible prototype of an AI model incubator.
//!
//! Six components:
//! 1. Variant generator   -> `variation`, `genome`
//! 2. Task environment    -> `task`
//! 3. Scoring             -> `score`
//! 4. Training            -> `train`, `nn`, `cnn`; the trained `model`
//! 5. Hidden evaluation   -> `hidden_eval` (+ binary `brut-eval`)
//! 6. Lineage archive     -> `archive`
//!
//! Glue: `config`, `safety`, `evolve`, `util`.
//!
//! Stage 2c (ADR 0005): `cascade` breeds Cascades of model Workers.
//!
//! CNN models (ADR 0007): `cnn`; its file `ops` is the only one that calls candle. Their
//! families, base models and mutations (ADR 0008): `cnn::families`, `cnn::variation`. The
//! winners of a run, MLPs and CNNs, are promoted into `candidates/models/` through a
//! human-reviewed pull request (ADR 0008, ADR 0004): `promote`, `hitl::ModelAdapter`. The ARC
//! grid tasks of the CNNs, one colour class per cell, train and score genomes on ARC-AGI-1
//! training tasks and their re-arc generations (ticket 06): `arc_grid`.

pub mod arc_data;
pub mod arc_grid;
pub mod archive;
pub mod backend;
pub mod cascade;
pub mod cnn;
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
pub mod model;
pub mod nn;
pub mod promote;
// Universes that invent (spec .scratch/universum-erfinden): exact qubit simulator.
pub mod quantum;
pub mod safety;
pub mod score;
pub mod task;
pub mod train;
pub mod util;
pub mod variation;
