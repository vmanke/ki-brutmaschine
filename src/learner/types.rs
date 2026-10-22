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
/// learner-v2 (review fixes): tiny values flushed, symmetric min/max, non-finite outputs
/// score q = 0, MlpMatched matched to the training cost, twin-free hidden grammar pool.
/// learner-v3 (errata V14): K1 re-costs SSin/SCos/SExp/SLog/VGaussian/MGaussian so counted
/// cost tracks real time (changes `dsl_ops_hash`), and K3 adds the all-time-best memory
/// (changes the search dynamics). Both shift which programs the search keeps and ships.
pub const PROTOCOL_VERSION: &str = "learner-v3";
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
/// Wall-clock and STOP checks happen every this many training steps and validation rows
/// (SEC-2: at most 64 Predict + Learn runs between two checks; was 1024, more than a
/// whole synthetic task).
pub const CHECK_EVERY_STEPS: usize = 64;

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
    /// Wall-clock limit per task; checked every CHECK_EVERY_STEPS steps and validation rows.
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
    SConst,
    SAdd,
    SSub,
    SMul,
    SDiv,
    SMin,
    SMax,
    SAbs,
    SRecip,
    SHeaviside,
    SSin,
    SCos,
    SExp,
    SLog,
    SUniform,
    // vector out (17)
    VBroadcast,
    VAdd,
    VSub,
    VMul,
    VDiv,
    VMin,
    VMax,
    VMax0,
    VScale,
    VAbs,
    VHeaviside,
    VSet,
    VGaussian,
    VUniform,
    MatVec,
    MatTVec,
    MRowMean,
    // scalar from vector (5)
    VDot,
    VNorm,
    VMean,
    VStd,
    VGet,
    // matrix out (8)
    MOuter,
    MAdd,
    MSub,
    MMul,
    MScale,
    MTranspose,
    MGaussian,
    MUniform,
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
        DslLimits {
            max_setup: 16,
            max_predict: 24,
            max_learn: 32,
            const_abs_max: 1.0e3,
        }
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
        HurdleConfig {
            quantile: 0.75,
            window: 1000,
            warmup: 100,
        }
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
        MetaWeights {
            breadth: 0.25,
            breadth_count: 5,
            len_penalty: 0.005,
            cost_penalty: 0.002,
        }
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
    /// Errata K3: cap of the permanent all-time-best memory (best programs ever seen by
    /// their best SELECT Δ, kept across panel rotations as parents and finalist candidates).
    /// 0 turns the memory off; the hard cap is `HARD_MAX_ALL_TIME_BEST` (64). Absent in
    /// learner-v2 archives, where it defaults to 64.
    #[serde(default = "default_all_time_best")]
    pub all_time_best: usize,
}

/// Default cap of the all-time-best memory (errata K3). Also used for older archives that
/// predate the field.
fn default_all_time_best() -> usize {
    64
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
            all_time_best: default_all_time_best(),
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

#[cfg(test)]
mod tests {
    use super::*;

    /// JSON text of a serialisable value.
    fn json<T: Serialize>(value: &T) -> String {
        serde_json::to_string(value).expect("serialise")
    }

    #[test]
    fn output_widths_and_hidden_tiers() {
        assert_eq!(LearnKind::Regression.k_out(), 1);
        assert_eq!(LearnKind::Binary.k_out(), 1);
        assert_eq!(LearnKind::Multiclass { k: 7 }.k_out(), 7);

        assert!(!Tier::Search.is_hidden());
        assert!(!Tier::Select.is_hidden());
        assert!(Tier::HiddenInstance.is_hidden());
        assert!(Tier::HiddenNear.is_hidden());
        assert!(Tier::HiddenFar.is_hidden());
        assert!(Tier::HiddenGrammar.is_hidden());
    }

    #[test]
    fn checkpoints_and_weights_are_consistent() {
        let steps: Vec<usize> = CHECKPOINT_DIVISORS
            .iter()
            .map(|d| STEPS_PER_TASK / d)
            .collect();

        assert_eq!(steps, vec![24, 96, 384]);
        assert_eq!(AULC_WEIGHTS.iter().sum::<f32>(), 1.0);
        assert!(SUPPORTED_F.iter().all(|f| (8..=32).contains(f)));
    }

    #[test]
    fn serde_names_match_the_data_formats() {
        assert_eq!(json(&Tier::HiddenGrammar), "\"hidden_grammar\"");
        assert_eq!(json(&VariationMode::ConstantsOnly), "\"constants_only\"");
        assert_eq!(json(&Mutation::Const), "\"const\"");
        assert_eq!(json(&EvalSet::A1Eval), "\"a1_eval\"");
        assert_eq!(json(&EvalSet::ConceptArc), "\"concept_arc\"");
        assert_eq!(json(&SourceId::ArcAgi1Eval), "\"arc_agi1_eval\"");
        assert_eq!(json(&SourceId::ArcAgi2Training), "\"arc_agi2_training\"");
        assert_eq!(json(&SourceId::ArcGen), "\"arc_gen\"");
        assert_eq!(json(&Group::ArcCell), "\"arc_cell\"");
        assert_eq!(json(&BaselineId::MlpMatched), "\"mlp_matched\"");
        assert_eq!(json(&Abort::ErrorTooLarge), "\"error_too_large\"");
        assert_eq!(json(&OpCode::MatTVec), "\"MatTVec\"");
        assert_eq!(json(&LearnKind::Binary), "{\"kind\":\"binary\"}");
        assert_eq!(
            json(&LearnKind::Multiclass { k: 4 }),
            "{\"kind\":\"multiclass\",\"k\":4}"
        );
    }

    #[test]
    fn defaults_match_the_meta_json_excerpt() {
        assert_eq!(
            json(&PanelConfig::default()),
            "{\"hand_tasks\":10,\"grammar_tasks\":10,\"rotate_every\":50,\"rotate_swap\":5,\
             \"stage_a_positions\":[0,3,6,10,15]}"
        );
        assert_eq!(
            json(&HurdleConfig::default()),
            "{\"quantile\":0.75,\"window\":1000,\"warmup\":100}"
        );
        assert_eq!(
            json(&DslLimits::default()),
            "{\"max_setup\":16,\"max_predict\":24,\"max_learn\":32,\"const_abs_max\":1000.0}"
        );
        assert_eq!(
            json(&MetaWeights::default()),
            "{\"breadth\":0.25,\"breadth_count\":5,\"len_penalty\":0.005,\"cost_penalty\":0.002}"
        );
        assert_eq!(
            json(&SelectConfig::default()),
            "{\"every_batches\":200,\"hof_size\":32,\"hand_per_family\":8,\
             \"grammar_families\":32,\"finalists\":3,\"retune_trials\":200,\"retune_tasks\":36,\
             \"all_time_best\":64}"
        );
    }

    #[test]
    fn program_json_accepts_the_compact_form_and_writes_every_field() {
        let text = r#"{"setup":[{"op":"SConst","out":3,"c":0.05}],
            "predict":[{"op":"MatVec","a":0,"b":0,"out":1}],
            "learn":[{"op":"VSub","a":2,"b":1,"out":3},{"op":"VScale","a":3,"b":3,"out":3},
                     {"op":"MOuter","a":3,"b":0,"out":1},{"op":"MAdd","a":0,"b":1,"out":0}]}"#;
        let g: LearnerGenome = serde_json::from_str(text).expect("parse");

        assert_eq!(g.setup.len(), 1);
        assert_eq!(g.predict.len(), 1);
        assert_eq!(g.learn.len(), 4);
        assert_eq!(g.setup[0].out, 3);
        assert_eq!(g.setup[0].a, 0);
        assert_eq!(g.setup[0].c, 0.05);
        assert_eq!(g.learn[1].op, OpCode::VScale);

        // Canonical output always carries all five instruction fields (errata E2).
        assert_eq!(
            json(&g.setup[0]),
            "{\"op\":\"SConst\",\"a\":0,\"b\":0,\"out\":3,\"c\":0.05}"
        );

        let back: LearnerGenome = serde_json::from_str(&json(&g)).expect("round trip");

        assert_eq!(back, g);
    }

    #[test]
    fn unknown_fields_are_rejected() {
        let instr = r#"{"op":"SAdd","a":1,"b":2,"out":3,"c":0.0,"d":1}"#;
        let genome = r#"{"setup":[],"predict":[],"learn":[],"seed":1}"#;
        let op = r#"{"op":"MatMul","a":0,"b":0,"out":0}"#;

        assert!(serde_json::from_str::<Instr>(instr).is_err());
        assert!(serde_json::from_str::<LearnerGenome>(genome).is_err());
        assert!(serde_json::from_str::<Instr>(op).is_err());
    }
}
