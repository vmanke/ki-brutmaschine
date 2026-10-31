# Stage 1 Implementation Contract: Brutmaschine "learner mode"

Version: contract-v1, 2026-10-02. Companion of `PLAN_GENERALISIERUNG.md` (German plan).
This document is normative. "MUST" items are acceptance-relevant. Where this contract and
the plan disagree, this contract wins for code; the plan wins for intent.

Notation: Rust blocks list public items with doc comments. A trailing `;` after a function
signature means "body omitted". WP0 creates every file below with these exact signatures and
`todo!()` bodies so that all work packages compile against the same API from day one.

---

## 0. Preconditions, scope, non-goals

Preconditions (MUST, checked by the lead before WP0):

1. The bug-fixing work currently running in `/home/user/brutmaschine` is merged and
   `cargo test`, `cargo clippy --all-targets -- -D warnings` and `cargo audit` are clean on it.
2. The user has answered the open decisions in plan section 13 (budget plan S or L, global
   secret dir, primary endpoint, E2 on/off). Defaults in this contract assume plan S, E2 off.

Scope: a new, additive "learner mode" in which a candidate is a small learning algorithm
(Setup / Predict / Learn programs in a closed DSL, after Real et al., ICML 2020), scored on
how well it learns *held-out task families*, with a hidden evaluator and an external ARC check.

Non-goals for Stage 1: any network access, any API key, any LLM, any third-party code at run
time, GPU code, changes to the existing MLP mode's behaviour.

---

## 1. Invariants (hold for every work package)

- I1. **No new dependencies.** `[dependencies]` in `Cargo.toml` stays exactly
  `hex, rand, rand_chacha, rayon, serde, serde_json, sha2` (a test enforces this).
  Amended by ADR 0007 (errata E15): `candle-core` and `candle-nn` from commit 9515d410 only.
- I2. **No unsafe code.** Every new `.rs` file starts with `#![forbid(unsafe_code)]`.
- I3. **No I/O in `src/learner/`.** Files under `src/learner/` never mention `std::fs`,
  `std::net`, `std::process`, `File::`, `Command`, `TcpStream`, `env::var`, `include_bytes`,
  `include_str`, `unsafe {`, `unsafe fn`, `unsafe impl` (a test enforces this).
  All file access lives in `src/arc_data.rs`, `src/ledger.rs`, `src/learner_hidden.rs`,
  `src/lineage_store.rs`, `src/meta_evolve.rs`, `src/meta_report.rs`, `src/main.rs`.
- I4. **Determinism.** All randomness comes from `ChaCha8Rng` seeded via
  `util::derive_seed(master, tag)` with the tags in section 6.1. Same `meta.json` gives the same
  archive (except `unix_ms`) and the same lineage-chunk SHA-256 for any thread count, unless a
  wall-clock limit or STOP fires. Parallel results are always merged in slot order.
- I5. **Library code never panics on bad input.** Return `util::Result` with `BrutError`.
  `todo!()`/`unwrap()` are forbidden in merged code outside tests.
- I6. **Backward compatibility.** Existing MLP mode, its config, archives, CLI and all
  existing tests are unchanged in behaviour (section 12).
- I7. **Secrets.** Only `src/learner_hidden.rs` and `src/bin/brut-eval.rs` read secret files.
  Orchestrator files never mention the forbidden identifiers in section 9.
- I8. **Finite numbers only in records.** Every `f64`/`f32` written to the archive, lineage
  chunks, snapshots, ledger or reports is finite. Use `max(x, 1)` before any `log10`.

---

## 2. Work packages and file ownership

Each file has exactly one owner. A package may read any file but edits only its own.
Cross-package API changes go through the lead and a contract revision.

| WP | Name | Owns (create or edit) | Depends on (API only) |
|---|---|---|---|
| WP0 | Skeleton (lead, first, ~0.5 day) | `src/lib.rs` (mod lines only), `src/learner/mod.rs`, `src/learner/types.rs` (complete, section 4), stub files for every module below | — |
| WP1 | DSL core | `src/learner/dsl.rs`, `src/learner/interp.rs`, `src/learner/seeds.rs`, `src/learner/mutate.rs`, `src/learner/fec.rs` | types |
| WP2 | Tasks, protocol, statistics | `src/learner/protocol.rs`, `src/learner/families.rs`, `src/learner/grammar.rs`, `src/learner/baselines.rs`, `src/learner/stats.rs`, `src/learner/wire.rs` | types, WP1 API |
| WP3 | ARC | `src/learner/arc_cells.rs`, `src/arc_data.rs`, `tools/rearc_to_jsonl.py` (optional E2), `data/manifest.example.json` | types, WP1/WP2 API |
| WP4 | Hidden evaluation | `src/hidden_eval.rs`, `src/learner_hidden.rs`, `src/ledger.rs`, `src/bin/brut-eval.rs` | all learner APIs, arc_data |
| WP5 | Orchestrator, archive, CLI | `src/meta_evolve.rs`, `src/meta_report.rs`, `src/lineage_store.rs`, `src/archive.rs`, `src/config.rs` (only: `diff_fields` → `pub(crate)`), `src/main.rs` | all of the above |
| WP6 | Tests, docs, hygiene | `tests/learner_e2e.rs`, `tests/learner_security.rs`, `tests/e2e.rs` (extend grep test only), `tests/bench.rs` (add interpreter bench), `README.md`, `docs/ARCHITEKTUR.md`, `docs/SICHERHEIT_UND_RECHT.md`, `.gitignore` (add `data/`), `.env.example` (document learner keys, placeholders only) | everything |

Order: WP0 → (WP1, WP2, WP3, WP4, WP5, WP6 in parallel against stubs) → lead integration →
acceptance run. Unit tests live inside each owned file (`#[cfg(test)] mod tests`).

Size estimate (lines incl. tests): WP1 ≈ 1 400, WP2 ≈ 1 900, WP3 ≈ 800, WP4 ≈ 1 300,
WP5 ≈ 1 800, WP6 ≈ 1 000. Total ≈ 8 000.

---

## 3. Module map

```
src/lib.rs                + pub mod learner; arc_data; ledger; learner_hidden; lineage_store;
                            meta_evolve; meta_report;
src/learner/mod.rs          re-exports; #![forbid(unsafe_code)]
src/learner/types.rs        shared plain data (WP0, complete)
src/learner/dsl.rs          op signatures, costs, validate, id, simplify          (pure)
src/learner/interp.rs       Memory<F>, run()                                       (pure)
src/learner/seeds.rs        4 hand-written seed programs                           (pure)
src/learner/mutate.rs       mutation operators, ProgramProposer trait              (pure)
src/learner/fec.rs          fingerprint cache keyed per task                       (pure)
src/learner/protocol.rs     run_learner, quality, AULC, fingerprint, protocol hash (pure)
src/learner/families.rs     24 hand families, split-v1, panels                     (pure)
src/learner/grammar.rs      expression-grammar families, signatures, pools        (pure)
src/learner/baselines.rs    tuned baselines, compute-matched MLP                  (pure)
src/learner/stats.rs        sign-flip test, bootstrap, Holm, Spearman, Hurdle      (pure)
src/learner/wire.rs         Episode JSON-lines protocol + DSL oracle (Stage-2 hook)(pure)
src/learner/arc_cells.rs    phi32, ARC grid -> per-cell task, pass@2, scoring      (pure)
src/arc_data.rs             ARC/ConceptARC/re-arc loading, manifest, contamination (I/O)
src/ledger.rs               global hash-chained query ledger                        (I/O)
src/learner_hidden.rs       hidden tiers, LearnerEvaluator, HiddenToken            (I/O, eval side only)
src/hidden_eval.rs          + tagged request envelope, generic child-process call
src/bin/brut-eval.rs        + dispatch learner / arc requests
src/lineage_store.rs        lineage chunks + snapshots                             (I/O)
src/meta_evolve.rs          search loop, final phase, hidden phase                 (I/O, orchestrator)
src/meta_report.rs          report.json + console table                            (I/O, orchestrator)
src/archive.rs              + new RecordKind variants, learner queries
src/main.rs                 + learner CLI
```

---

## 4. Shared types: `src/learner/types.rs` (WP0 writes this file completely)

```rust
//! Shared plain-data types for learner mode. No I/O and no logic beyond trivial helpers.
//! Changing this file requires the lead and a contract revision.
#![forbid(unsafe_code)]

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::config::Limits;
use crate::genome::Activation;

/// Scalar registers s0..s9.
pub const N_S: usize = 10;
/// Vector registers v0..v7, each F wide.
pub const N_V: usize = 8;
/// Matrix registers m0..m3, each F x F.
pub const N_M: usize = 4;
/// Feature widths the interpreter is monomorphised for.
pub const SUPPORTED_F: [usize; 3] = [8, 16, 32];
/// Largest number of classes.
pub const MAX_K: usize = 10;
/// Part of the protocol hash; bump on any change of tasks, metrics or seeds.
pub const PROTOCOL_VERSION: &str = "learner-v1";
/// Training-stream length of every synthetic task.
pub const STEPS_PER_TASK: usize = 384;
/// Checkpoints as fractions of the stream: after 24, 96 and 384 steps.
pub const CHECKPOINT_DIVISORS: [usize; 3] = [16, 4, 1];
/// Weights of q at the three checkpoints in the AULC.
pub const AULC_WEIGHTS: [f32; 3] = [0.25, 0.25, 0.5];
/// |prediction - label| above this on a training step aborts the task (AutoML-Zero S3).
pub const MAX_ABS_TRAIN_ERROR: f32 = 100.0;
/// Static cost cap: predict+learn cost per step <= this times the reference NN seed.
pub const COST_CAP_RATIO: u64 = 4;
/// Wall-clock and STOP checks happen every this many training steps.
pub const CHECK_EVERY_STEPS: usize = 1024;

/// What a task asks the learner to predict.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum LearnKind {
    /// One real target. Link: identity. Output width 1.
    Regression,
    /// Two classes {0, 1}. Link: sigmoid on v1[0]. Output width 1.
    Binary,
    /// k classes, 3 <= k <= 10. Link: softmax over v1[0..k]. Output width k.
    Multiclass { k: u8 },
}

impl LearnKind {
    /// Number of values the program writes into v1.
    pub fn k_out(self) -> usize {
        match self {
            LearnKind::Regression | LearnKind::Binary => 1,
            LearnKind::Multiclass { k } => k as usize,
        }
    }
}

/// Labels of one split. `Classes` holds class indices < k (Binary: 0 or 1).
#[derive(Clone, Debug, PartialEq)]
pub enum Labels {
    Values(Vec<f32>),
    Classes(Vec<u8>),
}

/// Evaluation tiers. Only `Search` and `Select` may be materialised outside `brut-eval`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Tier {
    Search,
    Select,
    HiddenInstance,
    HiddenNear,
    HiddenFar,
    HiddenGrammar,
}

impl Tier {
    pub fn is_hidden(self) -> bool {
        !matches!(self, Tier::Search | Tier::Select)
    }
}

/// Mechanism group of a family.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Group {
    Linear,
    Interaction,
    Radial,
    Periodic,
    Piecewise,
    Robustness,
    Grammar,
    ArcCell,
}

/// Stable family id: "lin-cls", "grammar/search/017", "grammar/hidden/003",
/// "arc-cell/rearc/<task-id>".
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct FamilyId(pub String);

/// One fully materialised learning task. Plain data; never contains code.
#[derive(Clone, Debug, PartialEq)]
pub struct TaskInstance {
    /// First 8 bytes (LE) of SHA-256 over "family|seed|f|n_train|n_valid|level".
    pub id: u64,
    pub family: FamilyId,
    pub group: Group,
    /// Input width; one of SUPPORTED_F.
    pub f: usize,
    pub kind: LearnKind,
    /// Row-major n_train x f.
    pub x_train: Vec<f32>,
    pub y_train: Labels,
    /// For drift tasks: labels used from stream step `drift_at` on. Same length as y_train.
    pub y_train_late: Option<Labels>,
    pub drift_at: Option<usize>,
    /// Row-major n_valid x f.
    pub x_valid: Vec<f32>,
    pub y_valid: Labels,
    pub n_train: usize,
    pub n_valid: usize,
    /// Training-stream length (STEPS_PER_TASK for synthetic tasks).
    pub steps: usize,
    /// Seed of the stream permutation and of the interpreter RNG (common random numbers).
    pub seed: u64,
    /// ARC cells only: true where the target colour differs from the input colour.
    pub changed_mask: Option<Vec<bool>>,
}

/// Why a task run ended early. The task then scores A_t = 0.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Abort {
    NonFinite,
    ErrorTooLarge,
    OverCost,
    Timeout,
    Stopped,
}

/// Outcome of running one learner on one task.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TaskResult {
    /// Quality at the three checkpoints, each in [0, 1].
    pub q: [f32; 3],
    /// 0.25 q0 + 0.25 q1 + 0.5 q2; 0 when aborted.
    pub aulc: f32,
    pub abort: Option<Abort>,
    /// Static FLOP count of the work actually done (charged to the Guard).
    pub flops: u64,
    pub steps_done: u32,
}

/// Run-time knobs that do not change scores (only when a task is cut off).
/// Orchestrator: max_seconds_per_task = limits.max_candidate_seconds; brut-eval: 30.0.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EvalSettings {
    /// Wall-clock limit per task; checked every CHECK_EVERY_STEPS steps.
    pub max_seconds_per_task: f64,
}

/// One released hidden result as recorded in the archive and the report.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LearnerHiddenResult {
    pub candidate_id: String,
    pub tier: Tier,
    pub delta: f64,
    pub decision: Option<Decision>,
    pub quarantined: bool,
}

/// Operation codes of the DSL. Semantics and costs: contract section 5.1.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum OpCode {
    // scalar out (15)
    SConst, SAdd, SSub, SMul, SDiv, SMin, SMax, SAbs, SRecip, SHeaviside,
    SSin, SCos, SExp, SLog, SUniform,
    // vector out (17)
    VBroadcast, VAdd, VSub, VMul, VDiv, VMin, VMax, VMax0, VScale, VAbs, VHeaviside,
    VSet, VGaussian, VUniform, MatVec, MatTVec, MRowMean,
    // scalar from vector (5)
    VDot, VNorm, VMean, VStd, VGet,
    // matrix out (8)
    MOuter, MAdd, MSub, MMul, MScale, MTranspose, MGaussian, MUniform,
    // scalar from matrix (1)
    MNorm,
}

/// One instruction. Unused operand fields MUST be 0; `c` MUST be 0.0 when unused.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Instr {
    pub op: OpCode,
    #[serde(default)]
    pub a: u8,
    #[serde(default)]
    pub b: u8,
    #[serde(default)]
    pub out: u8,
    #[serde(default)]
    pub c: f32,
}

/// A candidate learner: three straight-line programs. Plain data, interpreted only.
/// No seed field: randomness comes from the task (common random numbers).
#[derive(Clone, Debug, PartialEq, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LearnerGenome {
    /// Runs once per task, on zeroed memory.
    pub setup: Vec<Instr>,
    /// Runs before every training step and for every validation example.
    pub predict: Vec<Instr>,
    /// Runs after every training-step Predict, with the label in v2/s0.
    pub learn: Vec<Instr>,
}

/// Which program part an instruction belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Component {
    Setup,
    Predict,
    Learn,
}

/// Structural limits of a genome.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DslLimits {
    pub max_setup: usize,
    pub max_predict: usize,
    pub max_learn: usize,
    /// |c| <= this for value and scale constants.
    pub const_abs_max: f32,
}

impl Default for DslLimits {
    fn default() -> Self {
        DslLimits { max_setup: 16, max_predict: 24, max_learn: 32, const_abs_max: 1.0e3 }
    }
}

/// Kind of a mutation (AutoML-Zero section 3.2 plus a separate constant mutation).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Mutation {
    Insert,
    Remove,
    Randomize,
    Arg,
    Const,
}

/// Search-space ablations.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VariationMode {
    /// All mutation kinds.
    Full,
    /// Only `Const` mutations of seed-derived programs (the key ablation).
    ConstantsOnly,
    /// No selection: each child is a random seed with 1..=8 random Full mutations.
    RandomSearch,
}

/// Pass/fail of the pre-registered primary test (released as one bit).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Decision {
    Pass,
    Fail,
}

/// Pre-registered baselines.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BaselineId {
    LinearSgd,
    TwoLayerNn,
    Prototype,
    Perceptron,
    /// nn.rs Adam MLP, batch-trained on the examples seen so far, FLOP-matched per task.
    MlpMatched,
}

/// Recipe of the compute-matched MLP baseline (Adam, cosine decay to 5 %, full batch).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MlpRecipe {
    pub hidden: usize,
    pub activation: Activation,
    pub lr: f32,
    pub l2: f32,
}

/// Baselines after equal-effort constant tuning (deterministic; both sides compute them).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TunedBaselines {
    /// LinearSgd, TwoLayerNn, Prototype, Perceptron in this order.
    pub programs: Vec<(BaselineId, LearnerGenome)>,
    pub mlp: MlpRecipe,
    /// SHA-256 of the canonical JSON of this struct with this field set to "".
    pub hash: String,
}

/// ARC evaluation sets.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvalSet {
    A1Eval,
    A2Eval,
    ConceptArc,
}

/// Data sources known to the contamination rules.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceId {
    Synthetic,
    ArcAgi1Training,
    ArcAgi1Eval,
    ArcAgi2Training,
    ArcAgi2Eval,
    ReArc,
    ConceptArc,
    ArcGen,
    ArcTgi,
    Barc,
}

/// An ARC grid: 1 <= h, w <= 30, cells row-major, values 0..=9.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Grid {
    pub h: u8,
    pub w: u8,
    pub cells: Vec<u8>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ArcPair {
    pub input: Grid,
    pub output: Grid,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ArcTask {
    /// File stem, e.g. "0934a4d8".
    pub id: String,
    pub train: Vec<ArcPair>,
    pub test: Vec<ArcPair>,
}

/// Panel composition and rotation.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PanelConfig {
    /// Positions 0..hand_tasks: one instance of each SEARCH hand family (must be 10).
    pub hand_tasks: usize,
    /// Positions hand_tasks..: one instance from the SEARCH grammar pool each.
    pub grammar_tasks: usize,
    /// Rotate every this many batches; 0 = fixed panel (ablation).
    pub rotate_every: u64,
    /// Positions replaced per rotation.
    pub rotate_swap: usize,
    /// Panel positions evaluated in Stage A (hurdle).
    pub stage_a_positions: Vec<usize>,
}

impl Default for PanelConfig {
    fn default() -> Self {
        PanelConfig {
            hand_tasks: 10,
            grammar_tasks: 10,
            rotate_every: 50,
            rotate_swap: 5,
            stage_a_positions: vec![0, 3, 6, 10, 15],
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HurdleConfig {
    pub quantile: f64,
    pub window: usize,
    /// No hurdle until the window holds this many scores.
    pub warmup: usize,
}

impl Default for HurdleConfig {
    fn default() -> Self {
        HurdleConfig { quantile: 0.75, window: 1000, warmup: 100 }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MetaWeights {
    pub breadth: f64,
    /// Number of lowest per-task advantages averaged in the breadth term.
    pub breadth_count: usize,
    pub len_penalty: f64,
    pub cost_penalty: f64,
}

impl Default for MetaWeights {
    fn default() -> Self {
        MetaWeights { breadth: 0.25, breadth_count: 5, len_penalty: 0.005, cost_penalty: 0.002 }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SelectConfig {
    /// Stage C runs every this many batches and once at the end.
    pub every_batches: u64,
    pub hof_size: usize,
    /// Fresh instances per SELECT hand family per round.
    pub hand_per_family: usize,
    /// SELECT grammar families per round (one instance each).
    pub grammar_families: usize,
    pub finalists: usize,
    pub retune_trials: usize,
    pub retune_tasks: usize,
}

impl Default for SelectConfig {
    fn default() -> Self {
        SelectConfig {
            every_batches: 200,
            hof_size: 32,
            hand_per_family: 8,
            grammar_families: 32,
            finalists: 3,
            retune_trials: 200,
            retune_tasks: 36,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MetaHiddenConfig {
    /// Global learner secret file (default: $HOME/.brut/learner/hidden.env).
    pub secret_file: PathBuf,
    /// Tiers queried per finalist, in this order. Must contain HiddenInstance and HiddenGrammar.
    pub tiers: Vec<Tier>,
    pub gap_threshold: f64,
    pub score_step: f64,
    /// Upper bound passed as --budget (brut-eval also applies the secret file's caps).
    pub run_query_budget: usize,
}

/// Everything one learner-mode run needs. Stored as `<run_dir>/meta.json`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MetaConfig {
    /// Must equal PROTOCOL_VERSION.
    pub protocol_version: String,
    pub master_seed: u64,
    pub variation: VariationMode,
    pub population: usize,
    pub tournament: usize,
    pub batch: usize,
    pub max_batches: u64,
    pub dims_search: Vec<usize>,
    pub n_train_choices: Vec<usize>,
    pub n_valid_search: usize,
    pub panel: PanelConfig,
    pub hurdle: HurdleConfig,
    /// Capacity of the per-task score cache (entries).
    pub fec_capacity: usize,
    pub dsl: DslLimits,
    pub weights: MetaWeights,
    pub select: SelectConfig,
    /// Experiment E2: one re-arc cell task replaces grammar position 19 of the panel.
    pub arc_cells: bool,
    /// Lineage chunk + snapshot every this many batches.
    pub snapshot_every: u64,
    /// Reused for the Guard: max_total_flops, max_total_seconds, max_candidate_seconds.
    pub limits: Limits,
    pub threads: usize,
    pub hidden: MetaHiddenConfig,
}
```

`MetaConfig::default()` (implemented in `meta_evolve.rs`, WP5) uses: protocol_version
"learner-v1", master_seed 1, variation Full, population 256, tournament 10, batch 64,
max_batches 10 000 000, dims_search [8, 16], n_train_choices [16, 64, 256], n_valid_search 64,
the defaults above, fec_capacity 1 000 000, arc_cells false, snapshot_every 1 000,
`Limits { max_total_flops: 1_500_000_000_000, max_total_seconds: 14_400.0,
max_candidate_seconds: 30.0, ..Limits::default() }`, threads 4, hidden
{ secret_file: "$HOME/.brut/learner/hidden.env" resolved at init, tiers [HiddenInstance,
HiddenGrammar], gap_threshold 0.05, score_step 0.01, run_query_budget 12 }.

`src/learner/mod.rs`:

```rust
//! Learner mode: candidates are small learning algorithms in a closed, interpreted DSL.
#![forbid(unsafe_code)]

pub mod arc_cells;
pub mod baselines;
pub mod dsl;
pub mod families;
pub mod fec;
pub mod grammar;
pub mod interp;
pub mod mutate;
pub mod protocol;
pub mod seeds;
pub mod stats;
pub mod types;
pub mod wire;

pub use types::*;
```

---

## 5. Module contracts

### 5.1 `src/learner/dsl.rs` (WP1)

```rust
/// Operand slot type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Slot { None, S, V, M }

/// Meaning of the constant field `c`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConstKind {
    None,
    /// Any finite value with |c| <= const_abs_max.
    Value,
    /// Scale of a random op; |c| used; |c| <= const_abs_max.
    Scale,
    /// Integer index 0..=255 stored in c (c == c.trunc()); effective index = c as usize % F.
    Index,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OpSig { pub a: Slot, pub b: Slot, pub out: Slot, pub c: ConstKind }

/// All 46 opcodes in declaration order.
pub const ALL_OPS: [OpCode; 46];

/// Signature of an opcode (table below).
pub fn sig(op: OpCode) -> OpSig;

/// FLOP-equivalent cost of one execution at width f (table below).
pub fn cost(op: OpCode, f: u64) -> u64;

/// Per-step cost (predict + learn) of the reference 2-layer NN seed at width f:
/// 10 f^2 + 5 f. Defines the static cost cap.
pub fn reference_step_cost(f: usize) -> u64;

impl LearnerGenome {
    /// Structural check: lengths <= limits, every operand index valid for its slot
    /// (S < N_S, V < N_V, M < N_M), unused fields 0, constants finite and within limits,
    /// Index constants integral in 0..=255. Does not check cost.
    pub fn validate(&self, lim: &DslLimits) -> Result<()>;

    /// Static cost cap: step_cost(f) <= COST_CAP_RATIO * reference_step_cost(f)
    /// for every f in SUPPORTED_F. Err(Budget) otherwise. Called before any execution.
    pub fn check_cost(&self) -> Result<()>;

    /// validate + check_cost.
    pub fn check(&self, lim: &DslLimits) -> Result<()>;

    /// First 16 hex chars of SHA-256 over canonical_json() (same scheme as Genome::id).
    pub fn id(&self) -> String;

    /// serde_json::to_string of self (field order fixed by the struct).
    pub fn canonical_json(&self) -> String;

    /// Dead-code elimination by backward liveness over Setup -> (Predict -> Learn)*,
    /// iterated to a fixed point. Live outputs: v1 after Predict and every register read
    /// later. MUST preserve predictions bit-for-bit (property test).
    pub fn simplify(&self) -> LearnerGenome;

    /// Instruction count of simplify().
    pub fn effective_len(&self) -> usize;

    pub fn setup_cost(&self, f: usize) -> u64;
    pub fn predict_cost(&self, f: usize) -> u64;
    pub fn learn_cost(&self, f: usize) -> u64;
    /// predict_cost + learn_cost.
    pub fn step_cost(&self, f: usize) -> u64;

    /// setup + steps * step + 3 * n_valid * predict.
    pub fn task_cost(&self, f: usize, steps: usize, n_valid: usize) -> u64;

    /// Human-readable listing, one instruction per line ("MatVec v1 <- m0 . v0").
    pub fn render(&self) -> String;

    /// True when every instruction equals `other`'s except possibly `c`
    /// (used to accept constants-only variants of registered seeds).
    pub fn same_structure(&self, other: &LearnerGenome) -> bool;
}
```

Op table (normative). Registers: `s[i]`, `v[i]` (length F), `m[i]` (F x F, row-major).
All ops behave **as if inputs were copied before the output is written** (aliasing allowed).
Random ops draw from the memory's ChaCha8 stream. `idx(c) = (c as usize) % F`.

| Op | a | b | out | c | Semantics | Cost |
|---|---|---|---|---|---|---|
| SConst | – | – | S | Value | s[out] = c | 1 |
| SAdd, SSub, SMul, SDiv, SMin, SMax | S | S | S | – | s[out] = s[a] ∘ s[b] (IEEE; no checks) | 1 |
| SAbs, SRecip, SHeaviside | S | – | S | – | abs, 1/x, (x > 0 ? 1 : 0) | 1 |
| SSin, SCos, SExp, SLog | S | – | S | – | sin, cos, exp, ln | 10 |
| SUniform | – | – | S | Scale | s[out] = U[-\|c\|, \|c\|] | 10 |
| VBroadcast | S | – | V | – | v[out][i] = s[a] | F |
| VAdd, VSub, VMul, VDiv, VMin, VMax | V | V | V | – | elementwise | F |
| VMax0, VAbs, VHeaviside | V | – | V | – | max(x,0), abs, (x > 0 ? 1 : 0) | F |
| VScale | S | V | V | – | v[out] = s[a] * v[b] | F |
| VSet | S | – | V | Index | v[out][idx(c)] = s[a] (others unchanged) | 1 |
| VGaussian | – | – | V | Scale | v[out][i] = N(0, c²) | 10F |
| VUniform | – | – | V | Scale | v[out][i] = U[-\|c\|, \|c\|] | 10F |
| MatVec | M | V | V | – | v[out] = m[a] · v[b] | 2F² |
| MatTVec | M | V | V | – | v[out] = m[a]ᵀ · v[b] | 2F² |
| MRowMean | M | – | V | – | v[out][i] = mean_j m[a][i][j] | F² |
| VDot | V | V | S | – | s[out] = v[a] · v[b] | 2F |
| VNorm | V | – | S | – | s[out] = ‖v[a]‖₂ | 2F |
| VMean | V | – | S | – | mean | F |
| VStd | V | – | S | – | population std | 3F |
| VGet | V | – | S | Index | s[out] = v[a][idx(c)] | 1 |
| MOuter | V | V | M | – | m[out] = v[a] ⊗ v[b] (outer product) | F² |
| MAdd, MSub, MMul | M | M | M | – | elementwise | F² |
| MScale | S | M | M | – | m[out] = s[a] * m[b] | F² |
| MTranspose | M | – | M | – | m[out] = m[a]ᵀ | F² |
| MGaussian | – | – | M | Scale | m[out][i][j] = N(0, c²) | 10F² |
| MUniform | – | – | M | Scale | U[-\|c\|, \|c\|] | 10F² |
| MNorm | M | – | S | – | Frobenius norm | 2F² |

Excluded on purpose: matrix-matrix product, branches, loops, any ML-concept op.

### 5.2 `src/learner/interp.rs` (WP1)

```rust
/// Learner memory at width F. Zero-initialised at task start.
#[derive(Clone)]
pub struct Memory<const F: usize> {
    pub s: [f32; N_S],
    pub v: [[f32; F]; N_V],
    pub m: [[[f32; F]; F]; N_M],
    rng: rand_chacha::ChaCha8Rng,
}

impl<const F: usize> Memory<F> {
    /// Zeroed memory; `rng_seed` = derive_seed(task.seed, "interp/<attempt>").
    pub fn new(rng_seed: u64) -> Self;
    /// v0 = x (x.len() == F).
    pub fn set_input(&mut self, x: &[f32]);
    /// Zero v2 entirely and s0.
    pub fn clear_label(&mut self);
    /// Regression/Binary: v2[0] = y, rest 0. Multiclass: v2 = one-hot(y) over k. Then s0 = v2[0].
    pub fn set_label(&mut self, kind: LearnKind, y: f32);
    /// Apply the link in place on v1[0..k_out]: identity, sigmoid, or softmax
    /// (max-subtracted). Values outside 0..k_out are left as they are.
    pub fn apply_link(&mut self, kind: LearnKind);
    /// Copy of v1[0..k].
    pub fn output(&self, k: usize) -> [f32; MAX_K];
}

/// Execute one straight-line program on `mem`. Never panics: indices were validated;
/// in release builds out-of-range indices (impossible after validate) are clamped.
pub fn run<const F: usize>(code: &[Instr], mem: &mut Memory<F>);
```

Throughput target: ≥ 2.5 GFLOP/s counted FLOPs per thread for the NN seed at F = 16 in a
release build (scratch prototype measured 3.4).

### 5.3 `src/learner/seeds.rs` (WP1)

```rust
/// The four warm-start programs, in this fixed order:
/// "linear-sgd", "nn2-sgd", "prototype", "perceptron".
pub fn seed_programs() -> Vec<(&'static str, LearnerGenome)>;

/// The 2-layer ReLU NN seed ("nn2-sgd"); its step cost is reference_step_cost().
pub fn reference_nn() -> LearnerGenome;

/// SHA-256 over the canonical JSON of all seeds (part of the protocol hash).
pub fn seeds_hash() -> String;
```

Normative programs (registers as in section 6.2; `lr` constants are starting values):

- linear-sgd — Setup: `SConst s3 = 0.05`. Predict: `MatVec v1 = m0·v0`.
  Learn: `VSub v3 = v2 − v1; VScale v3 = s3·v3; MOuter m1 = v3⊗v0; MAdd m0 = m0 + m1`.
- nn2-sgd — Setup: `MGaussian m0 (c = 0.3); MGaussian m1 (c = 0.3); SConst s3 = 0.02`.
  Predict: `MatVec v4 = m0·v0; VMax0 v5 = max(v4, 0); MatVec v1 = m1·v5`.
  Learn: `VSub v3 = v2 − v1; VScale v3 = s3·v3; MOuter m2 = v3⊗v5; MatTVec v6 = m1ᵀ·v3;
  VHeaviside v7 = H(v4); VMul v6 = v6∘v7; MAdd m1 = m1 + m2; MOuter m3 = v6⊗v0;
  MAdd m0 = m0 + m3`. Step cost = 10F² + 5F.
- prototype — Setup: `SConst s3 = 0.05`. Predict: `MatVec v1 = m0·v0`.
  Learn: `MOuter m1 = v2⊗v0; MScale m1 = s3·m1; MAdd m0 = m0 + m1`.
- perceptron — Setup: `SConst s3 = 0.1; SConst s4 = 0.5`. Predict: `MatVec v1 = m0·v0`.
  Learn: `VSub v3 = v2 − v1; VAbs v4 = |v3|; VBroadcast v5 = s4; VSub v4 = v4 − v5;
  VHeaviside v4 = H(v4); VMul v3 = v3∘v4; VScale v3 = s3·v3; MOuter m1 = v3⊗v0;
  MAdd m0 = m0 + m1`.

### 5.4 `src/learner/mutate.rs` (WP1)

```rust
/// A uniformly random valid instruction for `comp` (op uniform over ALL_OPS, operands
/// uniform over their slot range, Value/Scale constants log-uniform in [1e-3, 1] with a
/// random sign for Value, Index uniform 0..=255).
pub fn random_instr(rng: &mut ChaCha8Rng, comp: Component) -> Instr;

/// One child that differs from `g` and passes `check(lim)`.
/// Full: P(Insert)=0.1, P(Remove)=0.2, P(Randomize component, keep length)=0.1,
///       P(Arg)=0.3, P(Const)=0.3. Const: multiply by U[0.5, 2.0], flip sign with p = 0.1
///       (Value only); Index constants are redrawn. Up to 8 retries for a valid, different
///       child; then fall back to Const, then Arg.
/// ConstantsOnly: only Const. RandomSearch: same as Full (selection is skipped elsewhere).
pub fn mutate(g: &LearnerGenome, mode: VariationMode, rng: &mut ChaCha8Rng, lim: &DslLimits)
    -> (LearnerGenome, Mutation);

/// Stage-2 hook (same role as variation::VariantGenerator). The result MUST pass check().
pub trait ProgramProposer: Send + Sync {
    fn name(&self) -> &str;
    fn propose(&self, parents: &[&LearnerGenome], rng: &mut ChaCha8Rng, lim: &DslLimits)
        -> (LearnerGenome, Mutation);
}

/// Default proposer wrapping `mutate`.
pub struct RandomProgramMutator { pub mode: VariationMode }
impl ProgramProposer for RandomProgramMutator { /* .. */ }

/// Initial population member i (0-based) of a run: seed[i % 4] mutated min(i / 4, 3) times
/// with derive_seed(master, "init/<i>").
pub fn initial_member(i: usize, master: u64, mode: VariationMode, lim: &DslLimits)
    -> LearnerGenome;
```

### 5.5 `src/learner/fec.rs` (WP1)

```rust
/// Per-task score cache keyed by (fingerprint, task id) -> AULC.
/// Reads during a batch are immutable; writes and LRU touches are staged and applied by
/// `commit` in slot order, so results do not depend on thread scheduling.
pub struct ScoreCache { /* private */ }

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CacheSnapshot {
    pub capacity: usize,
    /// Entries in LRU order, oldest first: (fingerprint, task id, aulc).
    pub entries: Vec<(u64, u64, f32)>,
}

impl ScoreCache {
    pub fn new(capacity: usize) -> Self;
    pub fn get(&self, fp: u64, task: u64) -> Option<f32>;
    pub fn stage_touch(&mut self, fp: u64, task: u64);
    pub fn stage_insert(&mut self, fp: u64, task: u64, aulc: f32);
    /// Apply staged touches/inserts in staging order; evict least recently used entries.
    pub fn commit(&mut self);
    pub fn len(&self) -> usize;
    pub fn snapshot(&self) -> CacheSnapshot;
    pub fn from_snapshot(s: CacheSnapshot) -> Self;
}
```

### 5.6 `src/learner/protocol.rs` (WP2)

```rust
/// Run learner `g` on `task` (attempt 0 for search; ARC uses attempts 0 and 1).
/// Implements section 6.2 exactly. `stop` is polled every CHECK_EVERY_STEPS steps.
/// Refuses (abort OverCost, flops 0) when g.check_cost() fails.
pub fn run_learner(g: &LearnerGenome, task: &TaskInstance, attempt: u32,
                   settings: &EvalSettings, stop: &dyn Fn() -> bool) -> TaskResult;

/// Like run_learner, but also returns the linked validation outputs at the final checkpoint
/// (n_valid rows of k_out values). Used by ARC solving and the wire oracle.
pub fn run_learner_with_outputs(g: &LearnerGenome, task: &TaskInstance, attempt: u32,
                                settings: &EvalSettings, stop: &dyn Fn() -> bool)
    -> (TaskResult, Vec<f32>);

/// q in [0, 1] for one checkpoint (formulas in section 6.3).
pub fn quality(kind: LearnKind, outputs: &[f32], labels: &Labels) -> f32;

/// 0.25 q0 + 0.25 q1 + 0.5 q2.
pub fn aulc(q: [f32; 3]) -> f32;

/// The two fixed probe tasks (F = 16; one regression, one binary; 10 train, 10 valid),
/// seeds derive_seed(0, "fec/probe/0") and ".../1".
pub fn probe_tasks() -> [TaskInstance; 2];

/// Functional fingerprint: run Setup + 10 steps + 10 validation predictions on each probe
/// (checkpoint only at the end), round the 20 outputs to 1e-3 (NaN -> i64::MIN,
/// +-inf -> i64::MAX/MIN+1), SHA-256 over the LE i64s, first 8 bytes LE.
pub fn fingerprint(g: &LearnerGenome, probes: &[TaskInstance; 2]) -> u64;

/// Everything that defines the measurement, as plain data.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ProtocolDescriptor {
    pub version: String,
    pub families: Vec<(FamilyId, Group, Tier, bool /* developer_seen */)>,
    pub grammar_search_pool_hash: String,
    pub grammar_select_pool_hash: String,
    pub steps_per_task: usize,
    pub checkpoint_divisors: [usize; 3],
    pub aulc_weights: [f32; 3],
    pub n_valid_hidden: usize,
    pub hidden_tier_sizes: Vec<(Tier, usize /* families */, usize /* instances each */)>,
    pub seeds_hash: String,
    pub baseline_tuning: String, // "trials=200;tasks=36;seed=derive_seed(0,\"baselines/tune\")"
    pub primary_test: String,    // "sign-flip;reps=100000;alpha=0.05;min_delta=0.01;coupling=0.05"
    pub dsl_ops_hash: String,    // SHA-256 over ALL_OPS names, sigs and costs
}

pub fn protocol_descriptor() -> ProtocolDescriptor;
/// SHA-256 hex over serde_json::to_string(&protocol_descriptor()).
pub fn protocol_hash() -> String;
```

### 5.7 `src/learner/families.rs` (WP2)

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FamilySpec {
    pub id: &'static str,
    pub group: Group,
    /// True when the developer has seen this family's behaviour in pilot runs.
    pub developer_seen: bool,
}

/// The 24 hand-written families (table below), in table order.
pub fn registry() -> &'static [FamilySpec];

/// split-v1 (golden; a unit test pins it): hidden-far group = smallest
/// SHA-256("brut-split-v1/group/" + name); inside every other group the families ordered by
/// SHA-256("brut-split-v1/family/" + name): first two Search, third Select, fourth HiddenNear.
pub fn split_v1() -> Vec<(FamilyId, Tier)>;

/// Families of a tier among the hand-written ones, in registry order.
pub fn hand_families(tier: Tier) -> Vec<FamilyId>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InstanceParams { pub f: usize, pub n_train: usize, pub n_valid: usize, pub level: u8 }

/// Draw f from `dims`, n_train from `n_train_choices`, level l in {1,2,3} with P ∝ 2^-l.
pub fn sample_params(rng: &mut ChaCha8Rng, dims: &[usize], n_train_choices: &[usize],
                     n_valid: usize) -> InstanceParams;

/// Materialise one instance of a hand-written family (Err for unknown ids).
pub fn hand_instance(fam: &FamilyId, seed: u64, p: &InstanceParams) -> Result<TaskInstance>;

/// One search panel position (section 6.4).
#[derive(Clone, Debug, PartialEq)]
pub struct Panel {
    /// SHA-256-derived id over the task ids (changes on every rotation).
    pub id: u64,
    pub rotation: u64,
    pub tasks: Vec<TaskInstance>,
    pub stage_a: Vec<usize>,
}

/// Panel after `rotation` rotations (deterministic from master_seed).
pub fn search_panel(master_seed: u64, rotation: u64, cfg: &PanelConfig, dims: &[usize],
                    n_train_choices: &[usize], n_valid: usize) -> Panel;

/// Positions swapped by rotation r >= 1: {(swap*(r-1) + j) mod len : j in 0..swap}.
pub fn rotation_positions(r: u64, cfg: &PanelConfig) -> Vec<usize>;

/// Fresh SELECT tasks for Stage C round `round` (hand: hand_per_family each with F cycling
/// 8,16,32; grammar: `grammar_families` from the SELECT pool, one each), n_valid 64.
pub fn select_tasks(master_seed: u64, round: u64, cfg: &SelectConfig) -> Vec<TaskInstance>;

/// 100 fresh public instances of the 10 SEARCH hand families (10 each), F alternating 8/16,
/// n_train cycling 16/64/256, n_valid 256. Used for Δ_public (quarantine).
pub fn public_reference_tasks(master_seed: u64) -> Vec<TaskInstance>;
```

Instance construction (normative): draw latent dimension d = 1 + level (2 for checkerboard
and spirals), latent points z ~ N(0, I_d) unless stated, embed x = Q · [z; ξ] with Q a random
F x F orthonormal matrix (Gram-Schmidt of a Gaussian matrix) and ξ ~ N(0, I_{F−d}); then
standardise every input feature with train-split mean and std (applied to both splits).
Regression targets are standardised with train-split statistics. Classification label noise
applies to training labels only; validation labels are clean.

| Family | Group | Definition (on latent z unless stated) |
|---|---|---|
| lin-reg | linear | y = w·z + N(0, 0.1²) |
| lin-cls | linear | y = 1[w·z + b > 0], b ~ N(0, 0.25²); 5 % train flips |
| sparse-lin-reg | linear | y = w·x on 1–2 random *embedded* coordinates + noise |
| multiclass-lin | linear | y = argmax_k (W z)_k, k = 2 + level |
| quad-reg | interaction | y = (a·z)² − 1 + noise, ‖a‖ = 1 |
| xor-cls | interaction | y = 1[(a·z)(b·z) > 0] |
| sign-parity-cls | interaction | parity of sign bits of min(d, 1 + level) latent axes |
| bilinear-reg | interaction | y = (a·z)(b·z) + noise |
| sphere-cls | radial | y = 1[‖z‖ < median radius] |
| blobs-multiclass | radial | k = 2 + level Gaussian blobs, σ ∈ [0.5, 1.5], centres N(0, 2² I) |
| rbf-reg | radial | y = Σ_{j≤1+level} a_j exp(−‖z − c_j‖² / 2s²) |
| prototype-cls | radial | 4–8 prototypes with random binary labels; nearest prototype |
| sine-reg | periodic | y = A sin(ω a·z + φ), A ∈ [0.1, 5], φ ∈ [0, π], ω ∈ [1, 3] |
| checkerboard-cls | periodic | 4 x 4 checkerboard on 2 latent dims mapped to [−1, 1]² (task.rs formula, reimplemented) |
| spirals-cls | periodic | two spirals on 2 latent dims (task.rs formula, reimplemented) |
| sawtooth-reg | periodic | y = frac(ω a·z + φ) − 0.5 |
| tree-cls | piecewise | random axis-aligned tree of depth 1 + level on latent axes |
| relu-teacher-reg | piecewise | y = Σ_{j≤1+level} v_j relu(w_j·z + b_j) |
| step-reg | piecewise | y = Σ_{j≤level} h_j 1[a_j·z > t_j] |
| dnf-cls | piecewise | random 3-term DNF over latent sign bits |
| drift-cls | robustness | linear classifier; weights negated from stream step steps/2; validation post-drift |
| noisy-label-cls | robustness | lin-cls with 30 % train flips |
| heavy-tail-reg | robustness | z ~ Student-t(ν = 2) per axis; y = w·z + t-noise |
| tanh-teacher-cls | robustness | y = 1[Σ v_j tanh(w_j·z) > 0] |

`developer_seen = true` for xor-cls, sine-reg, checkerboard-cls, spirals-cls, sign-parity-cls
(pilot or existing crate tasks). Do not edit `src/task.rs`.

### 5.8 `src/learner/grammar.rs` (WP2)

```rust
/// Expression over projections t_j = u_j · z (j < 3) of a 3-dim latent z.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Expr {
    T(u8),
    Add(Box<Expr>, Box<Expr>),
    Mul(Box<Expr>, Box<Expr>),
    Max(Box<Expr>, Box<Expr>),
    /// sin(w * e), w drawn per instance from [1, 3].
    Sin(Box<Expr>),
    Abs(Box<Expr>),
    Square(Box<Expr>),
    /// 1[e > t], t drawn per instance from N(0, 0.5²).
    Step(Box<Expr>),
}

#[derive(Clone, Debug, PartialEq)]
pub struct GrammarFamily {
    pub id: FamilyId,
    pub expr: Expr,
    /// Structure without constants, e.g. "add(sin(t0),mul(t1,t2))".
    pub signature: String,
    pub kind: LearnKind,
}

pub fn signature(e: &Expr) -> String;
/// Node count.
pub fn length(e: &Expr) -> usize;

/// Sample an expression of exactly `target_len` nodes, depth <= 3 (Err if impossible).
pub fn sample_expr(rng: &mut ChaCha8Rng, target_len: usize) -> Result<Expr>;

/// Target length distribution: L in 2..=9 with P(L) ∝ 2^-L.
pub fn sample_length(rng: &mut ChaCha8Rng) -> usize;

/// Public pools: Search = 64 families from derive_seed(0, "grammar/search/learner-v1");
/// Select = 32 families from derive_seed(0, "grammar/select/learner-v1") with signatures
/// disjoint from Search. Kind: Regression or Classification (Binary or Multiclass k in 3..=4)
/// with equal probability. Panics never; deterministic.
pub fn public_pool(tier: Tier) -> Vec<GrammarFamily>;

/// Pool of `n` families from `seed` whose signatures avoid `exclude` and whose length
/// histogram equals the histogram of `match_lengths` (scaled to n, largest-remainder
/// rounding). Err after 10 000 rejected draws for one family.
pub fn disjoint_pool(seed: u64, n: usize, exclude: &BTreeSet<String>,
                     match_lengths: &[usize], prefix: &str) -> Result<Vec<GrammarFamily>>;

/// Materialise one instance: classification = balanced quantile buckets of f(z) (quantiles
/// from 4 096 extra draws of the same instance), regression = standardised f(z) + N(0, 0.1²).
pub fn grammar_instance(fam: &GrammarFamily, seed: u64, p: &InstanceParams) -> TaskInstance;
```

### 5.9 `src/learner/baselines.rs` (WP2)

```rust
/// Equal-effort constant tuning: `trials` random perturbations of all Value/Scale
/// constants (each multiplied by exp(N(0, 0.5²)), kept if the mean AULC on `tasks` improves).
/// Structure never changes (same_structure holds).
pub fn tune_constants(g: &LearnerGenome, tasks: &[TaskInstance], trials: usize,
                      rng: &mut ChaCha8Rng, settings: &EvalSettings) -> (LearnerGenome, f64);

/// Random-search tuning of the MLP recipe over hidden ∈ {8,16,32,64}, the 4 activations,
/// lr log-uniform [1e-3, 1e-1], l2 ∈ {0, 1e-4, 1e-3}; `trials` draws, FLOP budget per task =
/// reference NN task cost.
pub fn tune_mlp(tasks: &[TaskInstance], trials: usize, rng: &mut ChaCha8Rng) -> (MlpRecipe, f64);

/// Deterministic tuned baselines: tasks = the even indices (36 tasks) of a SELECT task set
/// built like select_tasks but seeded with derive_seed(0, "baselines/tune/<family>/<i>");
/// rng = derive_seed(0, "baselines/tune"), trials 200. Orchestrator and brut-eval both call
/// this and MUST get identical `hash`.
pub fn tuned_baselines(settings: &EvalSettings) -> TunedBaselines;

/// Compute-matched MLP (nn.rs; Adam; cosine decay to 5 %; full batch). At each checkpoint c
/// it trains from scratch on the distinct examples seen in stream steps [0, step_c) with a
/// FLOP budget equal to the candidate's training cost up to step_c (6·P·n per epoch;
/// at least one epoch), then predicts the validation set. Uses nn::Mlp::new,
/// nn::Mlp::loss_and_grad_into and nn::OptimizerState; does not modify nn.rs or train.rs.
pub fn mlp_matched(task: &TaskInstance, recipe: &MlpRecipe, candidate_task_cost: u64)
    -> TaskResult;

/// Baselines in B(g) for a candidate with per-step cost `cand_step_cost_f16` at F = 16:
/// every program baseline whose step cost at F = 16 is <= the candidate's, plus MlpMatched.
pub fn compute_matched_set(tb: &TunedBaselines, cand_step_cost_f16: u64) -> Vec<BaselineId>;
```

### 5.10 `src/learner/stats.rs` (WP2)

```rust
/// One-sided sign-flip permutation p-value for H1: mean(d) > 0.
/// Exact enumeration when d.len() <= 20, else `reps` Monte-Carlo flips from `seed`.
/// p = (1 + #{flips with mean >= observed}) / (1 + reps).
pub fn sign_flip_pvalue(d: &[f64], reps: usize, seed: u64) -> f64;

/// Percentile CI of the mean over clusters (resample clusters with replacement).
/// Returns (mean, lo, hi). Caller documents that it is unreliable below ~20 clusters.
pub fn cluster_bootstrap_ci(clusters: &[Vec<f64>], reps: usize, alpha: f64, seed: u64)
    -> (f64, f64, f64);

/// Holm step-down; returns which hypotheses are rejected at family-wise level alpha.
pub fn holm(p: &[f64], alpha: f64) -> Vec<bool>;

/// Spearman rank correlation (average ranks for ties).
pub fn spearman(a: &[f64], b: &[f64]) -> f64;

/// Rolling-quantile hurdle (AutoML-Zero). Deterministic.
pub struct Hurdle { /* VecDeque<f64>, window, quantile, warmup */ }
impl Hurdle {
    pub fn new(cfg: &HurdleConfig) -> Self;
    /// None during warm-up (everyone passes), else the current quantile (nearest-rank).
    pub fn threshold(&self) -> Option<f64>;
    pub fn push(&mut self, score: f64);
    pub fn values(&self) -> Vec<f64>;
    pub fn from_values(cfg: &HurdleConfig, values: Vec<f64>) -> Self;
}
```

### 5.11 `src/learner/wire.rs` (WP2; Stage-2 bridge)

```rust
/// Host -> learner message (one JSON object per line).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
pub enum HostMsg {
    Setup { f: u16, kind: LearnKind, steps: u32 },
    Predict { x: Vec<f32> },
    Learn { x: Vec<f32>, y: f32 },
    /// Validation batch. The learner must not learn from it (only enforceable in-process).
    Eval { xs: Vec<Vec<f32>> },
    End,
}

/// Learner -> host message. Outputs are raw (pre-link); the host applies the link.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
pub enum LearnerMsg {
    Ready,
    Prediction { y: Vec<f32> },
    Learned,
    Evaluated { ys: Vec<Vec<f32>> },
}

pub trait WireLearner {
    fn handle(&mut self, msg: &HostMsg) -> Result<LearnerMsg>;
}

/// In-process DSL implementation; the oracle for any external (sandboxed) learner.
pub struct DslWireLearner { /* genome, memory, attempt seed */ }
impl DslWireLearner { pub fn new(g: LearnerGenome, task_seed: u64, attempt: u32) -> Self; }
impl WireLearner for DslWireLearner { /* .. */ }

/// Drive any WireLearner through section 6.2 and score it. For DslWireLearner the result
/// MUST equal run_learner bit-for-bit.
pub fn run_wire(learner: &mut dyn WireLearner, task: &TaskInstance) -> Result<TaskResult>;
```

Known difference (documented, not a bug): an external learner could use unlabeled `Eval`
inputs; the in-process DSL cannot (memory clone).

### 5.12 `src/learner/arc_cells.rs` (WP3; pure)

```rust
/// phi32 features of cell (r, c): one-hot centre colour (10); 8 "neighbour equals centre"
/// bits; 8 "neighbour non-zero" bits (neighbours in order (dr,dc) over (-1,0,1)^2 without
/// (0,0); outside the grid counts as 0); then r%2, c%2, r==0, r==h-1, c==0, c==w-1.
pub fn phi32(g: &Grid, r: usize, c: usize) -> [f32; 32];

/// True when every train pair keeps its shape (the solver predicts output shape = input).
pub fn train_keeps_shape(t: &ArcTask) -> bool;

/// One TaskInstance per test pair: F = 32, Multiclass{k: 10}, train = all cells of all
/// train pairs (input phi32 -> output colour), valid = cells of that test input with the
/// test output as labels, steps = clamp(3 * n_train_cells, 384, 8192),
/// changed_mask = test output != test input. None when !train_keeps_shape(t).
pub fn cell_tasks(t: &ArcTask) -> Option<Vec<TaskInstance>>;

#[derive(Clone, Debug, PartialEq)]
pub struct ArcResult {
    pub task_id: String,
    /// Per test pair: whether attempt 0 or attempt 1 matched exactly.
    pub correct: Vec<bool>,
    /// Balanced cell accuracy per test pair with both changed and unchanged cells.
    pub bca: Vec<f64>,
}

/// Two attempts (interpreter attempts 0 and 1), argmax colour per cell, exact grid match.
/// Tasks that change shape yield all-false.
pub fn solve_pass2(g: &LearnerGenome, t: &ArcTask, settings: &EvalSettings,
                   stop: &dyn Fn() -> bool) -> ArcResult;

/// Mean over ALL n_tasks of (correct outputs / test outputs); missing tasks count 0.
pub fn score_official(results: &[ArcResult], n_tasks: usize) -> f64;
/// Correct outputs / all test outputs.
pub fn score_kaggle(results: &[ArcResult], n_test_outputs: usize) -> f64;
pub fn mean_bca(results: &[ArcResult]) -> Option<f64>;

/// SHA-256 of the lexicographically smallest serialisation over the 8 D4 transforms
/// (applied to input and output together) with colours relabelled by first appearance
/// (0 kept as 0) across input then output.
pub fn canonical_pair_hash(p: &ArcPair) -> [u8; 32];

/// Port of arc_cell_ceiling.py "phi32 ceiling": all pairs (train and test) keep shape and
/// the map phi32-key -> output colour is consistent over train+test. Test helper.
pub fn phi32_ceiling(t: &ArcTask) -> bool;
```

### 5.13 `src/arc_data.rs` (WP3; I/O)

```rust
pub const MAX_ARC_FILE_BYTES: u64 = 1 << 20;
pub const MAX_REARC_FILE_BYTES: u64 = 32 << 20;
/// ARC-AGI-2 eval tasks that duplicate ARC-AGI-1 eval tasks.
pub const A1_A2_SHARED: [&str; 6] =
    ["0934a4d8", "136b0064", "16b78196", "981571dc", "aa4ec2a5", "da515329"];

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DataRole { Train, External }

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ManifestEntry {
    /// Relative to the data root; no "..", no absolute paths.
    pub rel_path: String,
    pub sha256: String,
    pub license: String,
    pub source: SourceId,
    pub commit: String,
    pub role: DataRole,
    /// For External entries: the set this directory/file belongs to.
    pub set: Option<EvalSet>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest { pub version: u32, pub entries: Vec<ManifestEntry> }

pub fn load_manifest(path: &Path) -> Result<Manifest>;
/// Recompute SHA-256 of every listed file; Integrity error on mismatch or missing file.
pub fn verify_manifest(root: &Path, m: &Manifest) -> Result<()>;
/// Regular file only (no symlink to devices), <= MAX_ARC_FILE_BYTES via read_limited_bytes,
/// JSON {"train": [...], "test": [...]}, grids rectangular 1..=30, values 0..=9,
/// 1..=10 train pairs, 1..=4 test pairs.
pub fn load_task_file(path: &Path) -> Result<ArcTask>;
/// All tasks of a set (ARC-AGI dirs: data/evaluation/*.json; ConceptARC: corpus/*/*.json),
/// sorted by id; only files listed in the manifest.
pub fn load_eval_set(root: &Path, m: &Manifest, set: EvalSet) -> Result<Vec<ArcTask>>;
/// Hard error when a training source touches the eval set:
/// A1Eval forbids ArcAgi1Eval, ArcAgi2Training, ArcGen, ArcTgi, Barc;
/// A2Eval forbids ArcAgi2Eval, ArcAgi1Eval, ArcTgi, Barc;
/// ConceptArc forbids ConceptArc, Barc. Barc is always forbidden (no licence).
pub fn check_contamination(train_sources: &[SourceId], set: EvalSet) -> Result<()>;
/// Canonical hashes of all training pairs of the given sources listed in the manifest.
pub fn training_pair_hashes(root: &Path, m: &Manifest, sources: &[SourceId])
    -> Result<BTreeSet<[u8; 32]>>;
/// E2 only: converted re-arc cell families (JSONL written by tools/rearc_to_jsonl.py).
pub fn load_rearc_jsonl(path: &Path, m: &Manifest) -> Result<Vec<(String, Vec<ArcPair>)>>;
```

`tools/rearc_to_jsonl.py` (optional, E2): Python standard library only (`zipfile`, `json`,
`hashlib`). Reads `re_arc.zip` (sha256 4a7c309499f450eb47c2117fa06f3c97fe741cf6e7912579c0b5b8cc759e9ac4),
whitelists `re_arc/tasks/<8 hex>.json`, skips `__MACOSX/`, rejects absolute or `..` paths,
caps each member at 32 MiB and the total at 800 MB, writes one JSON line per family with at
most 60 same-shape examples of area <= 225. MUST be run only via the sandbox recipe of plan
section 10 (setpriv to uid 65534 + no_new_privs, prlimit, unshare user/net/pid/mount/ipc/uts,
private tmpfs) from a staging directory owned by uid 65534 outside `/tmp/claude-0`.

### 5.14 `src/ledger.rs` (WP4)

```rust
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum LedgerKind {
    /// Written under the lock before scoring; counts against the budget.
    Charge { query: QueryKind },
    /// Written after scoring with exactly what was released.
    Release { query: QueryKind, released: serde_json::Value },
    /// Written when a query was refused (budget, protocol mismatch); does not count.
    Refused { query: QueryKind, reason: String },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum QueryKind {
    Learner { tier: Tier },
    Arc { set: EvalSet },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LedgerEntry {
    pub seq: u64,
    pub unix_ms: u64,
    pub prev_hash: String,
    pub protocol_hash: String,
    pub run_id: String,
    pub candidate_id: String,
    pub program_sha256: String,
    pub kind: LedgerKind,
    /// SHA-256 of this entry's JSON with hash = "" (same scheme as archive.rs).
    pub hash: String,
}

pub struct Ledger { /* path, entries */ }

impl Ledger {
    /// Open or create <dir>/ledger.jsonl (0600) and verify the whole chain.
    pub fn open(dir: &Path) -> Result<Ledger>;
    /// Append with fsync. Caller holds the eval lock.
    pub fn append(&mut self, e: LedgerEntry) -> Result<&LedgerEntry>;
    pub fn verify(path: &Path) -> Result<usize>;
    /// Charges for this protocol (all runs) and for this run, per query kind.
    pub fn charges(&self, protocol_hash: &str, run_id: Option<&str>, q: &QueryKind) -> usize;
    /// Total learner charges for this protocol (all tiers) and for one run.
    pub fn learner_charges(&self, protocol_hash: &str, run_id: Option<&str>) -> usize;
}
```

### 5.15 `src/learner_hidden.rs` (WP4; evaluator side only)

```rust
pub const LEARNER_SEED_KEY: &str = "BRUT_LEARNER_HIDDEN_SEED";
pub const BUDGET_TOTAL_KEY: &str = "BRUT_LEARNER_BUDGET_TOTAL";     // default 48
pub const BUDGET_PER_RUN_KEY: &str = "BRUT_LEARNER_BUDGET_PER_RUN"; // default 12
pub const ARC_ROOT_KEY: &str = "BRUT_ARC_DATA_ROOT";
pub const ARC_MANIFEST_KEY: &str = "BRUT_ARC_MANIFEST";
pub const ARC_BUDGET_KEY: &str = "BRUT_ARC_BUDGET_PER_SET";         // default 3
pub const LEARNER_EVAL_TIMEOUT_SECONDS: f64 = 600.0;
pub const ARC_EVAL_TIMEOUT_SECONDS: f64 = 900.0;

/// Capability: only this module can construct it (private field).
pub struct HiddenToken(());

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LearnerEvalRequest {
    /// Hash of the run's first archive record (MetaRunStarted).
    pub run_id: String,
    pub candidate_id: String,
    pub protocol_hash: String,
    pub tier: Tier,
    /// The program itself (no file reads by brut-eval).
    pub program: LearnerGenome,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LearnerEvalResponse {
    pub candidate_id: String,
    pub tier: Tier,
    /// Δ_T rounded DOWN to the step (>= 0.01). The only number released.
    pub delta: f64,
    /// Only for HiddenGrammar: the pre-registered decision (section 6.8). One bit.
    pub decision: Option<Decision>,
    pub queries_used_total: usize,
    pub queries_left_total: usize,
    pub queries_left_run: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArcEvalRequest {
    pub run_id: String,
    pub candidate_id: String,
    pub protocol_hash: String,
    pub set: EvalSet,
    pub program: LearnerGenome,
    /// Every source used to train or select this program.
    pub train_sources: Vec<SourceId>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArcEvalResponse {
    pub candidate_id: String,
    pub set: EvalSet,
    pub n_tasks: usize,
    pub n_test_outputs: usize,
    pub solved_outputs: usize,
    pub official: f64,
    pub kaggle: f64,
    /// Rounded down to 0.01.
    pub bca: Option<f64>,
    pub queries_left_set: usize,
}

/// Lives only inside brut-eval.
pub struct LearnerEvaluator { /* seed, caps, state dir, ledger, token */ }

impl LearnerEvaluator {
    /// Open the global learner secret (check_private on file and dir), read caps,
    /// open the ledger next to it. `cli_budget` and `cli_step` are the brut-eval flags:
    /// per-run cap = min(cli_budget, BUDGET_PER_RUN), step = max(cli_step, 0.01, BRUT_HIDDEN_STEP).
    pub fn open(secret_env: &Path, cli_budget: usize, cli_step: f64) -> Result<LearnerEvaluator>;

    /// Order: validate request and program (check()), protocol hash == protocol_hash()
    /// (else Refused, not charged), budget check + Charge entry under eval.lock, build tier
    /// tasks, run candidate and cached tuned baselines, compute Δ (and decision), quantize,
    /// Release entry, respond.
    pub fn evaluate(&mut self, req: &LearnerEvalRequest) -> Result<LearnerEvalResponse>;

    /// Order: validate, protocol hash, check_contamination, verify_manifest,
    /// canonical-hash overlap between training_pair_hashes(train_sources) and the eval set
    /// (any overlap: Refused), budget check + Charge, solve_pass2 on all tasks (rayon),
    /// Release, respond.
    pub fn evaluate_arc(&mut self, req: &ArcEvalRequest) -> Result<ArcEvalResponse>;
}

/// Create <dir> (0700) and <dir>/hidden.env (0600) with a fresh random seed and the
/// default caps. Refuses to overwrite. Returns the file path. `arc_root`/`manifest` are
/// written as ARC_ROOT_KEY/ARC_MANIFEST_KEY when given.
pub fn create_learner_secret(dir: &Path, arc_root: Option<&Path>, manifest: Option<&Path>)
    -> Result<PathBuf>;

/// Orchestrator side: send one request through hidden_eval::request_eval_json.
pub fn request_learner_eval(eval_bin: &Path, secret_env: &Path, budget: usize, step: f64,
                            req: &LearnerEvalRequest, stop_file: Option<&Path>)
    -> Result<LearnerEvalResponse>;
pub fn request_arc_eval(eval_bin: &Path, secret_env: &Path, budget: usize, step: f64,
                        req: &ArcEvalRequest, stop_file: Option<&Path>)
    -> Result<ArcEvalResponse>;
```

Private (not `pub`): `fn hidden_tier_tasks(token: &HiddenToken, secret_seed: u64, tier: Tier)
-> Result<Vec<TaskInstance>>` and `fn hidden_grammar_pool(token, secret_seed)`. Tier sizes
(normative, n_valid = 256 for all):

| Tier | Families | Instances | F | n_train |
|---|---|---|---|---|
| HiddenInstance | 10 SEARCH hand families | 10 each (100) | 8, 16 alternating | 16, 64, 256 cycling |
| HiddenNear | 5 HIDDEN-NEAR hand families | 20 each (100) | 8, 16, 32 cycling | cycling |
| HiddenFar | 4 periodic families | 25 each (100) | 8, 16, 32 cycling | cycling |
| HiddenGrammar | 40 from disjoint_pool(derive_seed(secret, "grammar/hidden"), 40, exclude = signatures of both public pools, match_lengths = search pool lengths, "grammar/hidden") | 5 each (200) | 8, 16, 32 cycling | cycling |

Instance seeds: `derive_seed(secret, "tier/<tier>/<family>/<i>")`.
Tuned-baseline results per (protocol_hash, tier, baseline) are cached in
`<state dir>/baseline_cache.json` and never charged. `MlpMatched` depends on the candidate's
cost and is recomputed per query (part of the query's work).

### 5.16 `src/hidden_eval.rs` changes (WP4; additive)

```rust
/// Any request brut-eval accepts. JSON without a "kind" key parses as Mlp (legacy).
#[derive(Clone, Debug, PartialEq)]
pub enum AnyEvalRequest {
    Mlp(EvalRequest),
    Learner(crate::learner_hidden::LearnerEvalRequest),
    Arc(crate::learner_hidden::ArcEvalRequest),
}

impl AnyEvalRequest {
    /// Parse: object with "kind": "learner" | "arc" -> tagged (the "kind" key is removed
    /// before deserialising the body with deny_unknown_fields); no "kind" -> Mlp.
    pub fn from_json(text: &str) -> Result<AnyEvalRequest>;
    /// Serialise with the tag (Mlp without a tag, byte-identical to today).
    pub fn to_json(&self) -> Result<String>;
}

/// Generic child-process call used by all request kinds. Same guarantees as today's
/// request_hidden_eval (env_clear, stdin/stdout JSON, output cap, timeout, STOP polling).
pub fn request_eval_json(eval_bin: &Path, secret_env: &Path, budget: usize, score_step: f64,
                         body: &[u8], timeout: Duration, stop_file: Option<&Path>)
    -> Result<Vec<u8>>;
```

`request_hidden_eval` keeps its signature and behaviour and delegates to `request_eval_json`.
`check_private` and `create_private_dir` become `pub(crate)`. Nothing else changes.

### 5.17 `src/bin/brut-eval.rs` changes (WP4)

Arguments unchanged (`--secret --budget --step`). After reading stdin (1 MiB cap):
`AnyEvalRequest::from_json` → `Mlp` → `HiddenEvaluator` (unchanged path);
`Learner` → `LearnerEvaluator::open(..).evaluate`; `Arc` → `...evaluate_arc`.
Error messages never quote request content or data. Exit codes unchanged (0, 2, 3).

### 5.18 `src/lineage_store.rs` (WP5)

```rust
/// One child that entered the population (hurdle passed). ~110 bytes as JSON.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LineageRecord {
    pub id: String,
    pub parent: Option<String>,
    pub batch: u64,
    pub slot: u16,
    pub mutation: Option<Mutation>,
    /// Fingerprint as 16 hex chars.
    pub fp: String,
    pub fitness: f64,
    /// Panel tasks served from the cache (0..=20).
    pub cached_tasks: u8,
}

/// Per-chunk counters (also copied into BatchDigest).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ChunkStats {
    pub children: u64,
    pub unique_evals: u64,
    pub fec_task_hits: u64,
    pub hurdled: u64,
    pub cost_rejected: u64,
    pub aborts_nonfinite: u64,
    pub aborts_error: u64,
    pub aborts_timeout: u64,
    pub flops: u64,
}

pub struct LineageWriter { /* buffer */ }
impl LineageWriter {
    pub fn new(dir: &Path, chunk: u64) -> Self;
    pub fn push(&mut self, r: LineageRecord);
    /// Atomically write <dir>/lineage/chunk-<chunk:06>.jsonl; return (relative path, sha256).
    pub fn finish(self) -> Result<(String, String)>;
}
pub fn read_chunk(path: &Path, expected_sha256: &str) -> Result<Vec<LineageRecord>>;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PopMember {
    pub id: String,
    pub genome: LearnerGenome,
    pub fp: String,
    pub age: u64,
    pub fitness: f64,
    /// Per panel position: AULC (needed to re-score only rotated tasks).
    pub task_scores: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Snapshot {
    pub batch: u64,
    pub rotation: u64,
    pub next_age: u64,
    pub population: Vec<PopMember>,
    pub hall_of_fame: Vec<PopMember>,
    pub hurdle_values: Vec<f64>,
    pub flops_used: u64,
    pub select_round: u64,
}

/// Atomically write <dir>/snapshots/pop-<batch:09>.json; return (relative path, sha256).
pub fn write_snapshot(dir: &Path, s: &Snapshot) -> Result<(String, String)>;
pub fn read_snapshot(path: &Path, expected_sha256: &str) -> Result<Snapshot>;
/// Overwrite <dir>/snapshots/cache-latest.json atomically; return sha256.
pub fn write_cache(dir: &Path, c: &CacheSnapshot) -> Result<String>;
pub fn read_cache(path: &Path, expected_sha256: &str) -> Result<CacheSnapshot>;
```

### 5.19 `src/meta_evolve.rs` (WP5; orchestrator; never touches secrets)

```rust
pub const META_CONFIG_FILE: &str = "meta.json";
pub const PROGRAMS_DIR: &str = "programs";
pub const LINEAGE_DIR: &str = "lineage";
pub const SNAPSHOT_DIR: &str = "snapshots";

impl Default for MetaConfig { /* values in section 4 */ }

impl MetaConfig {
    pub fn load(path: &Path) -> Result<MetaConfig>;
    pub fn save(&self, path: &Path) -> Result<()>;
    /// Ranges, protocol_version == PROTOCOL_VERSION, dims ⊆ SUPPORTED_F, panel sizes
    /// (hand_tasks == 10), score_step <= gap_threshold / 2, limits within HARD caps,
    /// hidden.tiers contains HiddenInstance and HiddenGrammar.
    pub fn validate(&self) -> Result<()>;
    /// Only max_batches, threads, limits.max_candidate_seconds, limits.max_total_seconds,
    /// limits.max_total_flops may change on resume (uses config::diff_fields).
    pub fn resume_conflicts(&self, earlier: &MetaConfig) -> Vec<String>;
}

pub struct LearnerEvalSetup { pub eval_bin: PathBuf, pub secret_env: PathBuf }
// LearnerHiddenResult is defined in types.rs.

#[derive(Clone, Debug, PartialEq)]
pub struct MetaRunSummary {
    pub batches_done: u64,
    pub children: u64,
    pub unique_evals: u64,
    pub fec_task_hit_rate: f64,
    pub hurdle_pass_rate: f64,
    pub flops_used: u64,
    pub wall_seconds: f64,
    pub stop_reason: String,
    pub finalists: Vec<String>,
    pub hidden: Vec<LearnerHiddenResult>,
}

/// True when <run_dir>/meta.json exists.
pub fn is_meta_run(run_dir: &Path) -> bool;

/// Create run_dir, write meta.json, create archive with MetaRunStarted and
/// ProtocolRegistered, write protocol.json. Refuses an existing archive.
pub fn init_meta_run(run_dir: &Path, cfg: &MetaConfig) -> Result<()>;

/// Full run (section 6.5–6.7). Resumes from the last BatchDigest. `eval = None` or
/// `no_hidden = true` skips the hidden phase (run stays open for more batches).
pub fn meta_run(run_dir: &Path, eval: Option<&LearnerEvalSetup>, no_hidden: bool)
    -> Result<MetaRunSummary>;

/// External ARC check for one archived program (finalist or baseline).
pub fn arc_check(run_dir: &Path, candidate_id: &str, set: EvalSet, eval: &LearnerEvalSetup)
    -> Result<ArcEvalResponse>;
```

### 5.20 `src/meta_report.rs` (WP5)

```rust
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Report {
    pub protocol_hash: String,
    pub config: MetaConfig,
    pub search: SearchHealth,
    pub finalists: Vec<FinalistReport>,
    pub baselines: TunedBaselines,
    pub external_arc: Vec<ExternalArcRow>,
    pub reference_lines: ReferenceLines,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SearchHealth {
    pub batches: u64, pub children: u64, pub unique_evals: u64,
    pub fec_task_hit_rate: f64, pub hurdle_pass_rate: f64, pub cost_rejected: u64,
    pub aborts: Vec<(Abort, u64)>, pub population_unique_fps: usize,
    pub flops_used: u64, pub wall_seconds: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FinalistReport {
    pub id: String,
    pub effective_len: usize,
    pub step_cost_f16: u64,
    pub program: String,                 // render() of simplify()
    pub panel_fitness: f64,
    pub delta_select: f64,
    pub delta_select_f32: f64,           // coupling, public
    pub delta_select_f8_16: f64,
    pub delta_public_instance: f64,      // Δ_public (quarantine input)
    pub winners_curse: f64,              // panel M(g) - mean AULC on public_reference_tasks
    pub q_by_checkpoint: [f64; 3],       // on SELECT
    pub q_by_n_train: Vec<(usize, f64)>, // on SELECT
    pub hidden: Vec<(Tier, f64, Option<Decision>)>,
    pub transfer_ratio: Option<f64>,     // Δ_GH / Δ_HI when Δ_HI > 0.01
    pub quarantined: bool,
    pub lineage_seed: String,            // seed name it descends from
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExternalArcRow { pub id: String, pub set: EvalSet, pub official: f64, pub kaggle: f64, pub bca: Option<f64> }

/// Fixed lines measured on the dev machine (plan section 6).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ReferenceLines {
    pub phi32_ceiling: [(EvalSet, usize, usize); 3], // (A1Eval 9/400), (A2Eval 0/120), (ConceptArc 17/160)
    pub icecuber_a1_pass2_outputs: f64,             // 0.289
    pub icecuber_a2_tasks: (usize, usize),          // (1, 120)
}

/// Build from archive + lineage; write <run_dir>/report.json atomically; return it.
pub fn write_report(run_dir: &Path) -> Result<Report>;
/// Plain-text table for `brut learner report`.
pub fn render_table(r: &Report) -> String;
```

### 5.21 `src/archive.rs` changes (WP5; additive)

New `RecordKind` variants (serde tag stays `"type"`; old archives keep parsing):

```rust
MetaRunStarted { config: MetaConfig, protocol_hash: String, crate_version: String },
ProtocolRegistered { protocol_hash: String, protocol: ProtocolDescriptor },
LearnerCandidate(LearnerEntry),
BatchDigest(BatchDigestEntry),
PanelRotated { rotation: u64, panel_id: String },
SelectEval { candidate_id: String, round: u64, select_mean: f64, delta_select: f64 },
LearnerHiddenEval { candidate_id: String, tier: Tier, delta: f64,
                    decision: Option<Decision>, delta_public: Option<f64>, quarantined: bool },
ExternalEval { candidate_id: String, set: EvalSet, official: f64, kaggle: f64,
               bca: Option<f64> },
```

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LearnerStatus { Seed, HallOfFame, Finalist, Baseline, Quarantined }

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LearnerEntry {
    pub id: String,
    pub parents: Vec<String>,
    /// "seed:<name>", "aging/<mutation>", "random-search", "retune", "baseline:<id>".
    pub generator: String,
    pub batch: u64,
    pub genome: LearnerGenome,
    pub effective_len: usize,
    pub step_cost_f16: u64,
    pub fp: String,
    pub panel_mean: f64,
    pub fitness: f64,
    /// Relative path under programs/ for Finalist/Baseline; None otherwise.
    pub program_file: Option<String>,
    pub status: LearnerStatus,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BatchDigestEntry {
    pub first_batch: u64,
    pub last_batch: u64,
    pub chunk_file: String,
    pub chunk_sha256: String,
    pub snapshot_file: String,
    pub snapshot_sha256: String,
    pub cache_sha256: String,
    pub stats: ChunkStats,
    pub best_fitness: f64,
}

impl Archive {
    pub fn meta_config(&self) -> Option<&MetaConfig>;
    pub fn protocol_hash(&self) -> Option<&str>;
    /// Latest entry per learner id, with Quarantined applied from LearnerHiddenEval.
    pub fn learner_candidates(&self) -> Vec<LearnerEntry>;
    pub fn learner_get(&self, id: &str) -> Option<LearnerEntry>;
    pub fn last_batch_digest(&self) -> Option<&BatchDigestEntry>;
    pub fn learner_hidden_evals(&self) -> Vec<LearnerHiddenResult>;
    /// Hash of record 0 (used as run_id).
    pub fn run_id(&self) -> Option<&str>;
}
```

`Archive::open` additionally verifies, for every `BatchDigest`, that the referenced chunk and
snapshot files exist and match their SHA-256 (cache file: only the latest digest).
`config.rs`: only `fn diff_fields` becomes `pub(crate)`.

### 5.22 `src/main.rs` CLI (WP5)

| Command | Behaviour |
|---|---|
| `brut init <dir> --mode learner [--seed N] [--flops X] [--variation full\|constants-only\|random-search] [--secret <file>] [--arc-cells]` | `init_meta_run`. Without `--mode` behaviour is unchanged (MLP). `--secret` default `$HOME/.brut/learner/hidden.env`; error with a hint if it does not exist. |
| `brut learner secret-init [--dir D] [--arc-root R] [--manifest M]` | `create_learner_secret`; prints only the path, never the seed. |
| `brut run <dir> [--no-hidden]` | dispatches on `is_meta_run`. |
| `brut status <dir>` | learner-aware summary (batches, best, finalists, hidden results). |
| `brut lineage <dir> <id>` | ancestors from lineage chunks + archive. |
| `brut verify <dir>` | archive chain + head + chunk/snapshot hashes; prints last hash. |
| `brut approve <dir> <id> --by B --note N` | learner run: requires a HiddenGrammar evaluation and not quarantined. |
| `brut stop <dir>` | unchanged. |
| `brut learner show <dir> <id>` | prints `render()` of the simplified program and its costs. |
| `brut learner report <dir>` | `write_report` + `render_table`. |
| `brut learner arc-check <dir> <id> --set a1-eval\|a2-eval\|concept-arc` | `arc_check`; appends `ExternalEval`. |

Exit codes stay 0 / 1 / 2.

---

## 6. Algorithms (normative)

### 6.1 Seed tags (all via `util::derive_seed`)

| Purpose | master | tag |
|---|---|---|
| initial member i | run master | `init/<i>` |
| batch b, slot s (tournament + mutation) | run master | `batch/<b>/slot/<s>` |
| panel position p, generation g | run master | `panel/<p>/<g>` |
| Stage C round r | run master | `select/<r>/<family>/<i>` |
| finalist retune | run master | `retune/<id>` |
| public reference tasks | run master | `pubref/<family>/<i>` |
| interpreter RNG | task.seed | `interp/<attempt>` |
| stream permutation, epoch e | task.seed | `stream/<e>` |
| probes | 0 | `fec/probe/<0\|1>` |
| grammar pools | 0 | `grammar/search/learner-v1`, `grammar/select/learner-v1` |
| tuned baselines | 0 | `baselines/tune` (tasks: `baselines/tune/<family>/<i>`) |
| hidden tier instances | secret | `tier/<tier>/<family>/<i>` |
| hidden grammar pool | secret | `grammar/hidden` |
| sign-flip test | secret | `test/<candidate_id>/<tier>` |

### 6.2 Per-task protocol

1. `mem = Memory::<F>::new(derive_seed(task.seed, "interp/<attempt>"))`; run Setup.
2. For step s in 0..steps: example index = permutation of epoch s / n_train at position
   s % n_train. `set_input(x)`, `clear_label()`, run Predict, `apply_link(kind)`.
   If any of v1[0..k_out] is non-finite → abort NonFinite. Regression: if
   |v1[0] − y| > 100 → abort ErrorTooLarge. Then `set_label(kind, y)` (drift tasks use
   `y_train_late` from step `drift_at`), run Learn.
3. After steps steps/16, steps/4 and steps: clone memory once; for each validation x in
   order: `set_input`, `clear_label`, run Predict on the clone, `apply_link`, record outputs.
   Compute q for that checkpoint.
4. Every CHECK_EVERY_STEPS steps: wall-clock limit → abort Timeout; `stop()` → abort Stopped.
5. `flops` = static `task_cost` prorated to the steps done. Aborted tasks have aulc = 0.

### 6.3 Quality

- Regression: `q = clip(1 − MSE / Var(y_valid), 0, 1)`; Var computed on y_valid; if
  Var < 1e-12 then q = 0.
- Binary: predicted class = (v1[0] ≥ 0.5). Multiclass: argmax (lowest index on ties).
  `q = clip((acc − a0) / (1 − a0), 0, 1)`, a0 = majority-class share of y_valid; if a0 = 1
  then q = 0.
- ARC cells use classification q for search (E2); BCA is reported separately.

### 6.4 Panels and rotation

Positions 0..10: SEARCH hand families in `hand_families(Search)` order. Positions 10..20:
SEARCH grammar pool; position p at generation g uses pool[(p − 10 + 10·g) mod 64]. The
generation of position p after rotation r is the number of rotations r' in 1..=r with
p ∈ `rotation_positions(r')`. Instance seed `panel/<p>/<g>`; params from `sample_params`
with that seed (dims_search, n_train_choices, n_valid_search). With `arc_cells = true`,
position 19 holds a re-arc cell task instead (E2 only). `rotate_every = 0` disables rotation.

### 6.5 Batch loop

```
load or initialise: population (P members, ages 0..P-1, fully evaluated, no hurdle),
                    hall of fame, hurdle window, cache, rotation, batch counter
for b in start.. while b < max_batches:
    guard.check()?                                   // STOP, time, FLOPs
    if rotate_every > 0 and b > 0 and b % rotate_every == 0:
        rotation += 1; rebuild panel; re-score every population/HoF member on the swapped
        positions only (parallel, merged in member order); recompute fitness; append PanelRotated
    hurdle_threshold = hurdle.threshold()            // fixed for the whole batch
    for slot s in 0..batch (parallel, rayon, threads):
        rng = ChaCha8(derive_seed(master, "batch/b/slot/s"))
        parent = tournament: T distinct members drawn uniformly; best fitness; ties: younger
                 (RandomSearch: parent = seed[rng % 4] mutated 1..=8 times; no tournament)
        child  = proposer.propose([parent]) ; child.check() guaranteed
        fp     = fingerprint(child)
        stage A: for each stage_a position: cache.get(fp, task.id) or run_learner
        if hurdle_threshold is Some(t) and mean_A < t: outcome = Hurdled
        else: remaining positions (cached or run); fitness (6.6); outcome = Passed
    merge in slot order:
        for every executed task: cache.stage_insert; for every hit: cache.stage_touch
        for every child that executed >= 1 stage-A task: hurdle.push(mean_A)
        Passed children: insert into population (age = next_age++), remove the oldest member
                         for each insertion; LineageRecord; update HoF (top hof_size by
                         fitness, dedup by fp; new entrants get a LearnerCandidate record)
        stats counters; guard.charge(sum of flops)
    cache.commit()
    if (b + 1) % select.every_batches == 0: Stage C (6.6)
    if (b + 1) % snapshot_every == 0: finish chunk, write snapshot + cache, append BatchDigest
at loop end: finish chunk + snapshot + BatchDigest; then final phase (6.7)
```

Resume: open archive, take `last_batch_digest()`, load its snapshot and the cache, delete
lineage chunks newer than it, continue at `last_batch + 1`. An uninterrupted run and a run
resumed from any digest MUST produce identical later chunk hashes (no wall-clock stops).

### 6.6 Fitness and Stage C

```
A_t(g)   = AULC of g on panel position t (cached or computed)
ref_t    = AULC of the untuned nn2-sgd seed on panel position t (computed once per position
           generation, cached)
M(g)     = mean_t A_t(g)                         over the 20 positions
B(g)     = mean of the breadth_count lowest values of (A_t(g) - ref_t)
fitness  = (1 - breadth) * M(g) + breadth * B(g)
           - len_penalty  * log10(1 + effective_len(g))
           - cost_penalty * log10(max(step_cost(g, 16), 1))
```

Tuned baselines: `tb = tuned_baselines()` is computed once at run start (≈ 3e10 FLOPs,
charged to the Guard) and written to `programs/baseline-<name>.json` with LearnerCandidate
status Baseline.

Stage C (every `select.every_batches` and once at the end): tasks =
`select_tasks(master, round)`; programs = hall of fame; baselines = `tb` restricted by
`compute_matched_set` (MlpMatched at each candidate's cost); Δ_select(g) = S_select(g) −
max over B(g) of S_select(b) (families weighted equally, each grammar family one family);
append `SelectEval` per HoF member. Stage C never feeds back into fitness or parent
selection.

### 6.7 Final phase

1. Final Stage C round. Finalists = top `select.finalists` HoF members by Δ_select
   (ties by id).
2. Retune each finalist with `tune_constants` (200 trials, rng `retune/<id>`, tasks = the
   36 tasks at even indices of `select_tasks(master, u64::MAX)`, i.e. 20 hand + 16 grammar).
   The retuned program gets a new id, generator "retune", parent = pre-retune id, status
   Finalist; write `programs/<id>.json`.
3. The final phase always runs after the search loop stops for budget or time reasons (as
   the MLP hidden phase does today); its cost (≈ 4e10 FLOPs) is charged and reported
   separately. STOP aborts it.
4. Δ_public(g) on `public_reference_tasks(master)` against `compute_matched_set`.
5. Hidden phase (unless `no_hidden` or `eval` is None): for each finalist, for each tier in
   `hidden.tiers`: `request_learner_eval`. After HiddenInstance: quarantined iff
   Δ_public − Δ_HI > gap_threshold. Append `LearnerHiddenEval`. Budget exhaustion or eval
   errors append `SafetyEvent` and do not fail the run (same as today).
6. After the first `LearnerHiddenEval`, the run is final: later `brut run` refuses new
   batches and only fills missing (finalist, tier) pairs of the same fixed list.
7. `write_report`; append `RunFinished`.

### 6.8 Hidden Δ and the primary decision (inside brut-eval)

For tier T with families f and tasks t: compute A_t for the candidate and every baseline in
B(g) (MlpMatched with the candidate's per-task cost). S_T = mean over families of the mean
over their tasks. b* = argmax over B(g) of S_T(b). Δ_T = S_T(g) − S_T(b*). Released:
`quantize(Δ_T, step)`.

HiddenGrammar only: d_f = mean over the family's 5 tasks of A_t(g) − A_t(b*);
p = `sign_flip_pvalue(d, 100_000, derive_seed(secret, "test/<id>/hidden_grammar"))`;
Δ32 = Δ restricted to F = 32 tasks, Δ816 = Δ restricted to F ∈ {8, 16}.
Decision = Pass iff p < 0.05 and Δ_T ≥ 0.01 and Δ32 ≥ Δ816 − 0.05; else Fail.
Nothing else (p, CI, per-family values) leaves brut-eval.

---

## 7. Data formats

`<run_dir>/meta.json` — `MetaConfig` as JSON (deny_unknown_fields). Excerpt:

```json
{"protocol_version":"learner-v1","master_seed":1,"variation":"full","population":256,
 "tournament":10,"batch":64,"max_batches":10000000,"dims_search":[8,16],
 "n_train_choices":[16,64,256],"n_valid_search":64,
 "panel":{"hand_tasks":10,"grammar_tasks":10,"rotate_every":50,"rotate_swap":5,
          "stage_a_positions":[0,3,6,10,15]},
 "hurdle":{"quantile":0.75,"window":1000,"warmup":100},"fec_capacity":1000000,
 "dsl":{"max_setup":16,"max_predict":24,"max_learn":32,"const_abs_max":1000.0},
 "weights":{"breadth":0.25,"breadth_count":5,"len_penalty":0.005,"cost_penalty":0.002},
 "select":{"every_batches":200,"hof_size":32,"hand_per_family":8,"grammar_families":32,
           "finalists":3,"retune_trials":200,"retune_tasks":36},
 "arc_cells":false,"snapshot_every":1000,"limits":{"...":"config::Limits"},"threads":4,
 "hidden":{"secret_file":"/root/.brut/learner/hidden.env",
           "tiers":["hidden_instance","hidden_grammar"],"gap_threshold":0.05,
           "score_step":0.01,"run_query_budget":12}}
```

`<run_dir>/programs/<id>.json` — `LearnerGenome`:

```json
{"setup":[{"op":"SConst","out":3,"c":0.05}],
 "predict":[{"op":"MatVec","a":0,"b":0,"out":1}],
 "learn":[{"op":"VSub","a":2,"b":1,"out":3},{"op":"VScale","a":3,"b":3,"out":3},
          {"op":"MOuter","a":3,"b":0,"out":1},{"op":"MAdd","a":0,"b":1,"out":0}]}
```

`<run_dir>/lineage/chunk-000001.jsonl` — one `LineageRecord` per line:

```json
{"id":"3fa1c2d4e5f60718","parent":"0b1c2d3e4f506172","batch":1234,"slot":17,
 "mutation":"const","fp":"9a8b7c6d5e4f3a2b","fitness":0.3412,"cached_tasks":15}
```

`<run_dir>/snapshots/pop-000001000.json` — `Snapshot`; `cache-latest.json` — `CacheSnapshot`.
`<run_dir>/protocol.json` — `ProtocolDescriptor` (human copy; the archive record is binding).
`<run_dir>/report.json` — `Report`.

Global learner secret `~/.brut/learner/hidden.env` (0600; dir 0700):

```
BRUT_LEARNER_HIDDEN_SEED=<u64, created by brut learner secret-init>
BRUT_LEARNER_BUDGET_TOTAL=48
BRUT_LEARNER_BUDGET_PER_RUN=12
BRUT_HIDDEN_STEP=0.01
BRUT_ARC_DATA_ROOT=/abs/path/data/external
BRUT_ARC_MANIFEST=/abs/path/data/external/manifest.json
BRUT_ARC_BUDGET_PER_SET=3
```

Same directory: `ledger.jsonl` (hash-chained `LedgerEntry` lines), `eval.lock`,
`baseline_cache.json`.

Requests (stdin of brut-eval):

```json
{"kind":"learner","run_id":"<64 hex>","candidate_id":"3fa1c2d4e5f60718",
 "protocol_hash":"<64 hex>","tier":"hidden_grammar","program":{"setup":[],"predict":[],"learn":[]}}
{"kind":"arc","run_id":"<64 hex>","candidate_id":"3fa1c2d4e5f60718","protocol_hash":"<64 hex>",
 "set":"a1_eval","program":{"setup":[],"predict":[],"learn":[]},"train_sources":["synthetic"]}
```

Responses: `LearnerEvalResponse`, `ArcEvalResponse` as JSON. Legacy MLP request/response:
unchanged.

`data/manifest.example.json` (WP3; real manifest lives under gitignored `data/`):

```json
{"version":1,"entries":[
 {"rel_path":"arc-agi-1/data/evaluation/0934a4d8.json","sha256":"<64 hex>",
  "license":"Apache-2.0","source":"arc_agi1_eval","commit":"399030444e0ab0cc8b4e199870fb20b863846f34",
  "role":"external","set":"a1_eval"}]}
```

---

## 8. CLI summary

See 5.22. `brut help` lists the new commands. The README documents them (WP6, after the
other agent's doc work is merged).

---

## 9. Security requirements

- S1. Orchestrator files `src/meta_evolve.rs`, `src/meta_report.rs`, `src/lineage_store.rs`,
  `src/main.rs`, `src/evolve.rs` MUST NOT contain: `HiddenEvaluator`, `LearnerEvaluator`,
  `parse_env_file`, `HIDDEN_SEED_KEY`, `LEARNER_SEED_KEY`, `HiddenToken`, `hidden_tier_tasks`,
  `hidden_grammar_pool`, `BRUT_ARC_DATA_ROOT`, `BRUT_ARC_MANIFEST`, `data/evaluation`,
  `arc-agi-2`, `ConceptARC/corpus`. (`main.rs` may call `create_learner_secret`.)
- S2. `src/learner/**` obeys invariant I3; every file contains `#![forbid(unsafe_code)]`.
- S3. brut-eval reads no program files; programs arrive inline (≤ 1 MiB request).
- S4. Hidden tier construction requires `HiddenToken`, constructible only inside
  `learner_hidden.rs` (compile-time hygiene; not a security boundary: same Unix user).
- S5. Charge before scoring, under `eval.lock`; refused queries are logged and not charged.
- S6. Per-run and total caps come from the secret file; CLI flags can only lower them.
- S7. ARC: manifest hashes verified before every ARC query; contamination check and
  canonical-hash audit before scoring; any overlap refuses.
- S8. All loaders: regular files only, size-capped reads (`read_limited_bytes`), strict
  shape/value checks, no content quoted in errors.
- S9. No third-party code runs during `brut` or `brut-eval`. `tools/rearc_to_jsonl.py` runs
  only under the sandbox recipe, once, offline.
- S10. `cargo audit` (advisory-db ≥ 117edb3) before every build used for experiments; 0
  vulnerabilities required.

---

## 10. Test list

Unit tests (in the owning file):

| ID | File | Test |
|---|---|---|
| T-DSL-1 | dsl.rs | `sig` covers all 46 ops; `ALL_OPS` has 46 distinct entries; cost table matches section 5.1 |
| T-DSL-2 | dsl.rs | `validate` rejects each bad field (index out of range per slot, non-zero unused field, NaN/inf/oversize constant, non-integral Index, too long) |
| T-DSL-3 | dsl.rs | `check_cost` refuses a program at 4·ref + 1 and accepts one at exactly 4·ref |
| T-DSL-4 | dsl.rs | `simplify` preserves predictions bit-for-bit on 1 000 random valid programs × 3 tasks |
| T-DSL-5 | dsl.rs | `id` stable, content-sensitive; JSON round trip; unknown fields rejected |
| T-INT-1 | interp.rs | each op equals a naive reference (rel. err ≤ 1e-6) on random inputs at F = 8, 16, 32 |
| T-INT-2 | interp.rs | aliasing (out == a == b) equals copy-first semantics |
| T-INT-3 | interp.rs | same seed → same random-op outputs; different attempts differ |
| T-SEED-1 | seeds.rs | linear-sgd and nn2-sgd outputs equal a plain-Rust reference implementation (abs ≤ 1e-5) on 20 tasks |
| T-SEED-2 | seeds.rs | nn2-sgd step cost == 10F² + 5F |
| T-MUT-1 | mutate.rs | 10 000 mutations: always valid, always different from parent |
| T-MUT-2 | mutate.rs | ConstantsOnly changes only `c` (same_structure) |
| T-MUT-3 | mutate.rs | Remove ≈ 2 × Insert frequency (chi-square, p > 0.001) |
| T-FEC-1 | fec.rs | commit order determinism; LRU eviction at capacity; snapshot round trip |
| T-PROT-1 | protocol.rs | cheating program (copies v2/s0 into v1 in Predict) has mean AULC ≤ 0.02 on 100 tasks |
| T-PROT-2 | protocol.rs | validation does not change trained memory (state after = state with validation skipped) |
| T-PROT-3 | protocol.rs | aborts: NaN output, regression error > 100, timeout, stop; aborted AULC = 0 |
| T-PROT-4 | protocol.rs | checkpoints at steps 24, 96, 384; CRN: two programs reading only random ops see equal streams |
| T-PROT-5 | protocol.rs | `protocol_hash` stable across runs; changes when any family, seed or constant changes |
| T-FAM-1 | families.rs | golden split-v1 table (plan section 4) |
| T-FAM-2 | families.rs | every family × F ∈ {8,16,32} × n_train ∈ {16,64,256} builds; features standardised (mean ≈ 0, std ≈ 1 on train); finite |
| T-FAM-3 | families.rs | determinism; class-balance bounds; drift labels flip at steps/2 |
| T-FAM-4 | families.rs | `rotation_positions` and panel generations match section 6.4 |
| T-GRA-1 | grammar.rs | pools deterministic; Search ∩ Select signatures = ∅; `disjoint_pool` avoids `exclude` and matches the length histogram exactly; Err after the rejection cap |
| T-BAS-1 | baselines.rs | `tune_constants` keeps structure; `tuned_baselines` hash identical across two calls and thread counts |
| T-BAS-2 | baselines.rs | `mlp_matched` FLOPs within ±10 % of the candidate budget per checkpoint (≥ 1 epoch rule documented) |
| T-STAT-1 | stats.rs | sign-flip null calibration: false-positive rate at α = 0.05 over 2 000 null datasets of 40 families in [0.035, 0.065] |
| T-STAT-2 | stats.rs | power sanity: 40 families with d_f = 0.05 + N(0, 0.05²) → p < 0.001 |
| T-STAT-3 | stats.rs | Holm, Spearman, Hurdle quantile against hand-computed fixtures |
| T-WIRE-1 | wire.rs | `run_wire(DslWireLearner)` == `run_learner` bit-for-bit on 50 tasks; JSON round trip of every message |
| T-ARC-1 | arc_cells.rs | phi32 golden vectors for fixed grids; `cell_tasks` sizes; steps clamp |
| T-ARC-2 | arc_cells.rs | `canonical_pair_hash` invariant under 8 D4 transforms × colour permutations (keeping 0) |
| T-ARC-3 | arc_cells.rs | `score_official` / `score_kaggle` / `mean_bca` on hand fixtures incl. multi-output and missing tasks |
| T-ARC-4 | arc_cells.rs | `#[ignore]` unless BRUT_ARC_TEST_ROOT is set: phi32_ceiling counts 9 / 400 (A1 eval), 0 / 120 (A2 eval), 17 / 160 (ConceptARC); same-shape counts 270 / 81 / 119 |
| T-DATA-1 | arc_data.rs | rejects oversize, non-rectangular, value 10, 31 × 31, symlink to /dev/zero, `..` in manifest path, hash mismatch |
| T-DATA-2 | arc_data.rs | `check_contamination` blocklist table; A1_A2_SHARED flagged |
| T-LED-1 | ledger.rs | chain verify; tamper (edit, delete, reorder) detected; two processes charging concurrently count exactly |
| T-HID-1 | learner_hidden.rs | per-run and total caps; a new run_id does not reset the total; refused queries not charged |
| T-HID-2 | learner_hidden.rs | protocol mismatch refused; invalid or over-cost program refused before charge |
| T-HID-3 | learner_hidden.rs | response has exactly the documented fields; delta is a multiple of the step and ≤ the true Δ; decision only for HiddenGrammar |
| T-HID-4 | learner_hidden.rs | hidden grammar signatures disjoint from both public pools; tiers sized per table |
| T-ENV-1 | hidden_eval.rs | legacy request JSON (no kind) parses as Mlp and round-trips byte-identically; tagged kinds parse; unknown kind rejected |

Integration tests:

| ID | File | Test |
|---|---|---|
| T-E2E-1 | tests/learner_e2e.rs | tiny run (max_total_flops 1e9, population 32, batch 8) with hidden phase: archive verifies, programs saved, report.json has every finalist × tier |
| T-E2E-2 | tests/learner_e2e.rs | same seed, threads 1 vs 4: identical archive (ignoring unix_ms) and chunk hashes |
| T-E2E-3 | tests/learner_e2e.rs | STOP during a batch ends the run within 2 s; resume reaches the same chunk hashes as an uninterrupted run |
| T-E2E-4 | tests/learner_e2e.rs | resume refuses a changed experiment and appends nothing |
| T-E2E-5 | tests/learner_e2e.rs | budget exhaustion → SafetyEvent, run completes |
| T-E2E-6 | tests/learner_e2e.rs | quarantine is final; fixed list does not shift |
| T-E2E-7 | tests/learner_e2e.rs | tampered chunk or snapshot → `brut verify` fails (exit 1) |
| T-E2E-8 | tests/learner_e2e.rs | CLI cycle: secret-init, init --mode learner, run, status, lineage, learner show, learner report, verify, approve |
| T-E2E-9 | tests/learner_e2e.rs | ARC check on a 5-task fixture set (in `tests/fixtures/arc/`, hand-made) end-to-end with ledger entries |
| T-SEC-1 | tests/learner_security.rs | S1 grep test (forbidden identifiers in orchestrator files) |
| T-SEC-2 | tests/learner_security.rs | S2/I3 grep test over `src/learner/**` + forbid attribute present |
| T-SEC-3 | tests/learner_security.rs | Cargo.toml `[dependencies]` keys == the seven allowed (since ADR 0007, errata E15: also `candle-core` and `candle-nn` from commit 9515d410 only) |
| T-SEC-4 | tests/learner_security.rs | brut-eval with a 2 MiB request fails without charging |
| T-BENCH-1 | tests/bench.rs | `#[ignore]` interpreter throughput report (GFLOP/s, tasks/s at F = 8, 16, 32) |
| existing | tests/e2e.rs | all existing tests unchanged; `orchestrator_never_touches_the_secret` extended to the new orchestrator files and identifiers |

---

## 11. Acceptance criteria (release build, 4 vCPU dev machine)

| ID | Criterion |
|---|---|
| AC-1 | `cargo test` passes (all old and new non-ignored tests); `cargo clippy --all-targets -- -D warnings` clean; `cargo audit` 0 vulnerabilities; `[dependencies]` unchanged (since ADR 0007, errata E15: except `candle-core` and `candle-nn` from commit 9515d410). |
| AC-2 | Interpreter: nn2-sgd at F = 16 runs at ≥ 2.5 GFLOP/s counted per thread and ≥ 3 000 full tasks (384 steps + 3 × 64 predictions) per second on 4 threads. |
| AC-3 | E0 smoke run (`--flops 1e11`, defaults otherwise) finishes in ≤ 5 min wall, evaluates ≥ 10 000 children, and writes report.json. |
| AC-4 | Determinism: T-E2E-2 and T-E2E-3 pass. |
| AC-5 | Leakage: cheating program mean AULC ≤ 0.02 over 100 tasks (T-PROT-1); validation side-effect free (T-PROT-2). |
| AC-6 | Static cost cap enforced before execution: an over-cap program never reaches `interp::run` (instrumented test counter = 0). |
| AC-7 | Hidden phase for 3 finalists × {HiddenInstance, HiddenGrammar} completes in ≤ 5 min wall including first-time baseline tuning; each response carries exactly one Δ (multiple of 0.01) plus at most one decision bit; ledger verifies; a second run directory sees the reduced total budget. |
| AC-8 | Meta-fitness on hidden families is computed and reported: report.json contains, per finalist, Δ for every configured tier, the HiddenGrammar decision, Δ_public, quarantine flag, transfer ratio, public F = 32 coupling and winner's-curse numbers. |
| AC-9 | ARC: `brut learner arc-check --set a1-eval` runs end-to-end over all 400 tasks in ≤ 10 min wall and appends `ExternalEval`; official and Kaggle scores match fixture computations; with BRUT_ARC_TEST_ROOT set, T-ARC-4 reproduces 9 / 0 / 17. |
| AC-10 | Contamination: configuring `arc_agi2_training` or `arc_gen` as a training source while checking a1-eval is refused before any scoring and without charge; a planted rotated + recoloured duplicate pair is detected. |
| AC-11 | Resources: a 1e13-FLOP run writes ≤ 500 MB under the run directory and peaks ≤ 2 GB RSS. |
| AC-12 | STOP: the run ends ≤ 2 s after `brut stop` (checked before each batch and every 1 024 steps). |
| AC-13 | Backward compatibility (section 12) holds; existing e2e tests pass unchanged. |
| AC-14 | Statistics: T-STAT-1 false-positive rate within [0.035, 0.065]. |

Experiment-level deliverable (after acceptance, plan section 8): plan S — E0, E1 (Full, 3
seeds × 1.5e12), ablation ConstantsOnly (3 seeds × 1.5e12), hidden phase and ARC checks on
a1-eval, a2-eval, concept-arc for the best finalist of each E1 seed; total ≈ 9.5e12 FLOPs.

---

## 12. Backward compatibility

- MLP mode: `config.json`, `brut init` without `--mode`, `brut run` on MLP run dirs, archive
  records, hidden evaluation, quarantine, approve: unchanged behaviour and output.
- `RecordKind`: variants only added; old archives open and verify; MLP code paths ignore
  learner records.
- `EvalRequest`/`EvalResponse` JSON unchanged; brut-eval accepts untagged legacy requests;
  arguments and exit codes unchanged; `EVAL_TIMEOUT_SECONDS` stays 120 for MLP requests.
- `request_hidden_eval` keeps its signature.
- `config.rs`: only `diff_fields` visibility changes. `nn.rs`, `train.rs`, `task.rs`,
  `genome.rs`, `variation.rs`, `score.rs`, `evolve.rs`, `safety.rs`, `util.rs` are not edited.
- Docs: README's "Genome sind Daten / kein selbstverändernder Code" gains a learner-mode
  clause: "Kandidaten sind Programme in einer geschlossenen Sprache; sie werden nur
  interpretiert und können nur rechnen" (WP6).

---

## 13. Deferred (not Stage 1)

- Stage 2a `LlmProposer` (implements `ProgramProposer`; separate `broker/` package with
  `ureq` 3.4.2; key only in `.env`, read only by the broker; USD caps). Interface frozen here.
- Stage 2b sandboxed external learners speaking `wire.rs` JSON lines.
- Stage 2c DGM-style self-modifying agent scaffold.
- ARC-AGI-3 (interactive), GPU code, Docker.
