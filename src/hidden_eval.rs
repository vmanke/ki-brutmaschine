//! Hidden evaluation: a separate process scores models on a secret test split.
//!
//! Why: selecting on the validation score makes it optimistic over time.
//! A test split the search never sees gives an honest final estimate.
//!
//! Leakage controls:
//! - The hidden seed lives in a secret `.env` file read only by `brut-eval`.
//! - Only one aggregated score, rounded down to a step of at least 0.01, is released per query.
//! - A persistent query budget limits (but does not prevent) adaptive overfitting.
//!
//! Rounding and a query budget are the simple heuristics that Blum & Hardt 2015
//! ("The Ladder", ICML) improve on; their guarantees do not apply here. The Ladder releases
//! a new score only when it beats the best so far by a margin, which does not fit the
//! quarantine check: that needs each candidate's own score. Dwork et al. 2015
//! ("The reusable holdout", Science 349:636) add noise for their guarantees; this does not.
//!
//! Limit: on one Unix user this is process separation, not a security boundary.
//! Production needs a separate user, container or machine.
//!
//! `brut-eval` accepts three request kinds (`AnyEvalRequest`): legacy MLP requests (no "kind"
//! key, unchanged), learner requests and ARC requests (see `learner_hidden`). The model file
//! of a legacy request holds an MLP or a CNN (`model::Model`). A CNN file is checked against
//! the task and the caps of this module before any tensor is built (`check_cnn`).
#![forbid(unsafe_code)]

use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::cnn::model::CnnModel;
use crate::config::{HARD_MAX_HIDDEN_LAYERS, HARD_MAX_PARAMS, HARD_MAX_WIDTH};
use crate::model::Model;
use crate::nn::Mlp;
use crate::score;
use crate::task::{Split, TaskKind, TaskSpec, generate};
use crate::util::{BrutError, Result, parse_env_file, write_atomic};

/// Env key in the secret file that holds the hidden seed (u64, decimal).
pub const HIDDEN_SEED_KEY: &str = "BRUT_HIDDEN_SEED";

/// Optional env key in the secret file: an upper bound on the query budget.
/// The evaluator uses the smaller of this and the `--budget` argument.
pub const HIDDEN_BUDGET_KEY: &str = "BRUT_HIDDEN_BUDGET";

/// Optional env key in the secret file: a lower bound on the score step.
/// The evaluator uses the largest of this, `MIN_HIDDEN_SCORE_STEP` and the `--step` argument.
pub const HIDDEN_STEP_KEY: &str = "BRUT_HIDDEN_STEP";

/// Optional env key in the secret file: the only directory whose model files may be scored.
/// Default: `<secret file dir>/../models`, the `models/` folder of the run directory.
pub const HIDDEN_MODELS_DIR_KEY: &str = "BRUT_MODELS_DIR";

/// Smallest score step the evaluator ever uses. A finer step (or 0, no rounding) would
/// release more information about the hidden data per query, whatever the caller asks for.
pub const MIN_HIDDEN_SCORE_STEP: f64 = 0.01;

/// File name of the evaluator state, next to the secret file.
pub const STATE_FILE: &str = "eval_state.json";

/// Lock file next to the secret file. Held while a query is counted, so concurrent
/// `brut-eval` processes sharing one secret directory count every query exactly once.
pub const LOCK_FILE: &str = "eval.lock";

/// Largest model file the evaluator reads (defense against huge or malicious files).
pub const MAX_MODEL_BYTES: u64 = 64 * 1024 * 1024;

/// Largest model the evaluator scores (parameters).
pub const MAX_EVAL_PARAMS: usize = HARD_MAX_PARAMS;

/// Most forward FLOPs the evaluator spends on one CNN: its forward FLOPs per sample
/// (`cnn::shape`) times the hidden split. A CNN reuses each weight at every position, so
/// its parameter count does not bound its cost. A CNN within the default limits needs less
/// than 3 % of it.
///
/// The count alone does not bound the time. A plan of grouped or narrow convolutions with
/// norms and activations spends most of its time copying values and calling candle, not
/// multiplying: a file of 9.2e10 FLOPs took 179 s at one thread (ticket 02). So
/// `MAX_EVAL_SECONDS` caps the estimated time as well, and that cap binds for such plans.
pub const MAX_EVAL_FLOPS: u64 = 100_000_000_000;

/// Most seconds that scoring one CNN may take by the time model of `cnn::shape`
/// (`Plan::eval_nanos` for the hidden split, in the chunks of `score::evaluate`): half of
/// `EVAL_TIMEOUT_SECONDS`. Every file of the measurements of ticket 02 scored in less than
/// 0.92 of its estimate, at one thread.
pub const MAX_EVAL_SECONDS: f64 = 60.0;

/// How long the orchestrator waits for one `brut-eval` answer before it stops the process.
pub const EVAL_TIMEOUT_SECONDS: f64 = 120.0;

/// Largest stdout or stderr the orchestrator keeps from `brut-eval`.
const MAX_EVAL_OUTPUT_BYTES: u64 = 1024 * 1024;

/// How often the orchestrator checks whether `brut-eval` has finished.
const EVAL_POLL: Duration = Duration::from_millis(10);

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EvalRequest {
    pub candidate_id: String,
    /// Path to the model file (`model::Model`): the JSON of an `nn::Mlp`, or of a
    /// `cnn::model::CnnModel`.
    pub model_path: PathBuf,
    pub task: TaskKind,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EvalResponse {
    pub candidate_id: String,
    /// Primary metric on the hidden split, rounded to `score_step`.
    pub hidden_primary: f64,
    pub queries_used: usize,
    pub queries_left: usize,
}

/// Persistent evaluator state, stored next to the secret file.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EvalState {
    pub queries_used: usize,
}

/// Any request brut-eval accepts. JSON without a "kind" key parses as Mlp (legacy).
#[derive(Clone, Debug, PartialEq)]
pub enum AnyEvalRequest {
    Mlp(EvalRequest),
    Learner(crate::learner_hidden::LearnerEvalRequest),
    Arc(crate::learner_hidden::ArcEvalRequest),
}

/// Value of the "kind" key of a learner request.
pub(crate) const KIND_LEARNER: &str = "learner";

/// Value of the "kind" key of an ARC request.
pub(crate) const KIND_ARC: &str = "arc";

impl AnyEvalRequest {
    /// Parse: object with "kind": "learner" | "arc" -> tagged (the "kind" key is removed
    /// before deserialising the body with deny_unknown_fields); no "kind" -> Mlp.
    ///
    /// Errors name the problem and position only; they never quote request content.
    pub fn from_json(text: &str) -> Result<AnyEvalRequest> {
        let value: serde_json::Value = serde_json::from_str(text).map_err(|e| bad_request(&e))?;

        let serde_json::Value::Object(mut fields) = value else {
            return Err(BrutError::Invalid("request is not a JSON object".into()));
        };

        let Some(kind) = fields.remove("kind") else {
            // Legacy MLP request: parse the original text exactly as before.
            let req: EvalRequest = serde_json::from_str(text).map_err(|e| bad_request(&e))?;

            return Ok(AnyEvalRequest::Mlp(req));
        };

        let body = serde_json::Value::Object(fields);

        match kind.as_str() {
            Some(KIND_LEARNER) => serde_json::from_value(body)
                .map(AnyEvalRequest::Learner)
                .map_err(|e| bad_request(&e)),
            Some(KIND_ARC) => serde_json::from_value(body)
                .map(AnyEvalRequest::Arc)
                .map_err(|e| bad_request(&e)),
            _ => Err(BrutError::Invalid(
                "request has an unknown \"kind\" (expected \"learner\" or \"arc\")".into(),
            )),
        }
    }

    /// Serialise with the tag (Mlp without a tag, byte-identical to today).
    /// Tagged requests start with `{"kind":"..."` followed by the fields in struct order.
    pub fn to_json(&self) -> Result<String> {
        match self {
            AnyEvalRequest::Mlp(req) => Ok(serde_json::to_string(req)?),
            AnyEvalRequest::Learner(req) => tagged_json(KIND_LEARNER, req),
            AnyEvalRequest::Arc(req) => tagged_json(KIND_ARC, req),
        }
    }
}

/// `{"kind":"<kind>",` followed by the fields of `body` (a struct with named fields).
fn tagged_json(kind: &str, body: &impl Serialize) -> Result<String> {
    let text = serde_json::to_string(body)?;

    let Some(fields) = text.strip_prefix('{') else {
        return Err(BrutError::Invalid(
            "request body is not a JSON object".into(),
        ));
    };

    if fields == "}" {
        return Ok(format!("{{\"kind\":\"{kind}\"}}"));
    }

    Ok(format!("{{\"kind\":\"{kind}\",{fields}"))
}

/// A generic parse error: category and position, never the offending text.
fn bad_request(e: &serde_json::Error) -> BrutError {
    let place = if e.line() > 0 {
        format!(" at line {} column {}", e.line(), e.column())
    } else {
        String::new()
    };

    BrutError::Invalid(format!(
        "request does not match its schema ({:?} error{place})",
        e.classify()
    ))
}

/// Lives only inside the `brut-eval` process.
pub struct HiddenEvaluator {
    hidden_seed: u64,
    budget: usize,
    score_step: f64,
    state_path: PathBuf,
    lock_path: PathBuf,
    /// Only model files inside this directory are read.
    models_dir: PathBuf,
}

impl HiddenEvaluator {
    /// Load the seed from `secret_env`. The query count lives in
    /// `<secret_env dir>/eval_state.json` and is read under a lock for each query.
    /// Refuses a secret file readable by group or others (Unix mode & 0o077 != 0).
    ///
    /// The score step actually used is the largest of `score_step`, `MIN_HIDDEN_SCORE_STEP`
    /// and the optional `BRUT_HIDDEN_STEP` in the secret file, so the caller cannot lower it.
    pub fn open(secret_env: &Path, budget: usize, score_step: f64) -> Result<HiddenEvaluator> {
        check_private(secret_env)?;

        if !score_step.is_finite() || score_step < 0.0 {
            return Err(BrutError::Invalid(format!(
                "score step {score_step} must be finite and >= 0"
            )));
        }

        let env = parse_env_file(secret_env)?;
        let Some(seed_text) = env.get(HIDDEN_SEED_KEY) else {
            return Err(BrutError::Invalid(format!(
                "secret file has no {HIDDEN_SEED_KEY}"
            )));
        };

        let hidden_seed: u64 = seed_text.parse().map_err(|_| {
            BrutError::Invalid(format!("{HIDDEN_SEED_KEY} is not an unsigned integer"))
        })?;

        let budget = match env.get(HIDDEN_BUDGET_KEY) {
            Some(text) => {
                let cap: usize = text.parse().map_err(|_| {
                    BrutError::Invalid(format!("{HIDDEN_BUDGET_KEY} is not an unsigned integer"))
                })?;

                budget.min(cap)
            }
            None => budget,
        };

        let secret_step = match env.get(HIDDEN_STEP_KEY) {
            Some(text) => {
                let step: f64 = text.parse().map_err(|_| {
                    BrutError::Invalid(format!("{HIDDEN_STEP_KEY} is not a number"))
                })?;

                if !(0.0..1.0).contains(&step) {
                    return Err(BrutError::Invalid(format!(
                        "{HIDDEN_STEP_KEY} {step} must be in [0, 1)"
                    )));
                }

                step
            }
            None => 0.0,
        };

        let score_step = score_step.max(MIN_HIDDEN_SCORE_STEP).max(secret_step);

        let dir = match secret_env.parent() {
            Some(d) if !d.as_os_str().is_empty() => d,
            _ => Path::new("."),
        };

        let models_dir = match env.get(HIDDEN_MODELS_DIR_KEY) {
            Some(text) => PathBuf::from(text),
            None => dir.join("..").join("models"),
        };

        Ok(HiddenEvaluator {
            hidden_seed,
            budget,
            score_step,
            state_path: dir.join(STATE_FILE),
            lock_path: dir.join(LOCK_FILE),
            models_dir,
        })
    }

    /// The score step this evaluator rounds to (after applying the floors).
    pub fn score_step(&self) -> f64 {
        self.score_step
    }

    /// Score one model.
    ///
    /// First the model path must lie inside the models directory (after resolving symlinks
    /// and `..`), and the file is loaded and checked against the task. These checks do not
    /// touch the hidden data, so a refused request costs no query. Then one query is charged and
    /// persisted before scoring (a crash during scoring still counts it); when the budget is
    /// spent, the request is refused.
    pub fn evaluate(&mut self, req: &EvalRequest) -> Result<EvalResponse> {
        let path = self.allowed_model_path(&req.model_path)?;
        let model = load_model(&path)?;
        let spec = req.task.spec();

        match &model {
            Model::Mlp(mlp) => {
                if mlp.n_in() != spec.n_in || mlp.n_out() != spec.n_out {
                    return Err(BrutError::Invalid(format!(
                        "model sizes {:?} do not match task {:?}",
                        mlp.sizes, req.task
                    )));
                }
            }
            Model::Cnn(cnn) => check_cnn(cnn, &spec)?,
        }

        let used = self.charge()?;
        let data = generate(&spec, Split::Hidden, self.hidden_seed);
        let metrics = score::evaluate(&model, &spec, &data);

        Ok(EvalResponse {
            candidate_id: req.candidate_id.clone(),
            hidden_primary: quantize(metrics.primary, self.score_step),
            queries_used: used,
            queries_left: self.budget.saturating_sub(used),
        })
    }

    /// Resolve `path` (symlinks, `..`) and accept it only inside the models directory.
    ///
    /// Without this check the evaluator would read any file it can open, for example
    /// another user's files in a separate-user deployment.
    fn allowed_model_path(&self, path: &Path) -> Result<PathBuf> {
        let root = std::fs::canonicalize(&self.models_dir).map_err(|e| {
            BrutError::Invalid(format!(
                "models directory {}: {e}",
                self.models_dir.display()
            ))
        })?;
        let real = std::fs::canonicalize(path)
            .map_err(|e| BrutError::Invalid(format!("model path {}: {e}", path.display())))?;

        if real == root || !real.starts_with(&root) {
            return Err(BrutError::Invalid(format!(
                "model path {} is outside the models directory {}",
                path.display(),
                root.display()
            )));
        }

        Ok(real)
    }

    /// Count one query and return the new count.
    ///
    /// Read, check and write of the state happen under an exclusive lock on `eval.lock`,
    /// so two evaluators sharing a secret directory never lose or double a query.
    fn charge(&self) -> Result<usize> {
        // Released when `_lock` is dropped (the file is closed).
        let _lock = lock_exclusive(&self.lock_path)?;

        let mut state = match std::fs::read_to_string(&self.state_path) {
            Ok(text) => serde_json::from_str::<EvalState>(&text)?,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => EvalState { queries_used: 0 },
            Err(e) => return Err(e.into()),
        };

        if state.queries_used >= self.budget {
            return Err(BrutError::Eval(format!(
                "hidden query budget exhausted ({} of {} used)",
                state.queries_used, self.budget
            )));
        }

        state.queries_used += 1;
        write_atomic(&self.state_path, &serde_json::to_vec_pretty(&state)?)?;

        Ok(state.queries_used)
    }
}

/// Read and check a model file: regular file, size cap, JSON, shape, parameter cap, finite weights.
///
/// Devices, FIFOs and directories are refused before they are opened (opening a FIFO could
/// block, and /dev/zero reports length 0). The read itself stops after `MAX_MODEL_BYTES`,
/// so a file that grows after the size check cannot exhaust memory either.
///
/// A CNN file (`"kind":"cnn"`) is parsed with unknown fields refused, at most
/// `HARD_MAX_PARAMS` trained values and at most `MAX_RUNNING_VALUES` running statistics
/// (`cnn::model`). Its dense head gets the caps of an MLP here; its shapes are checked
/// against the task by `check_cnn`.
fn load_model(path: &Path) -> Result<Model> {
    let not_regular = || {
        BrutError::Invalid(format!(
            "model path {} is not a regular file",
            path.display()
        ))
    };

    if !std::fs::metadata(path)?.is_file() {
        return Err(not_regular());
    }

    let file = std::fs::File::open(path)?;
    let meta = file.metadata()?;

    // Check the opened handle again: the path may have changed since the first check.
    if !meta.is_file() {
        return Err(not_regular());
    }

    if meta.len() > MAX_MODEL_BYTES {
        return Err(BrutError::Invalid(format!(
            "model file is {} bytes, max {MAX_MODEL_BYTES}",
            meta.len()
        )));
    }

    let bytes = read_limited_bytes(file, MAX_MODEL_BYTES)?;

    // Generic messages: the parser's own error would quote strings from the file.
    match Model::from_json(&bytes)? {
        Model::Mlp(mlp) => check_mlp(&mlp).map(|()| Model::Mlp(mlp)),
        Model::Cnn(cnn) => check_cnn_head(&cnn).map(|()| Model::Cnn(cnn)),
    }
}

/// The checks of an MLP file: shape, depth, width, parameter cap and finite weights.
fn check_mlp(model: &Mlp) -> Result<()> {
    model.check_shape()?;

    // Depth and width bound the activation memory and the run time of scoring;
    // the parameter cap alone allows a 400 000-wide layer or a million thin layers.
    let hidden_layers = model.sizes.len().saturating_sub(2);

    if hidden_layers > HARD_MAX_HIDDEN_LAYERS {
        return Err(BrutError::Invalid(format!(
            "model has {hidden_layers} hidden layers, max {HARD_MAX_HIDDEN_LAYERS}"
        )));
    }

    if model.sizes.iter().any(|&w| w > HARD_MAX_WIDTH) {
        return Err(BrutError::Invalid(format!(
            "model has a layer wider than {HARD_MAX_WIDTH}"
        )));
    }

    if model.param_count() > MAX_EVAL_PARAMS {
        return Err(BrutError::Invalid(format!(
            "model has more than {MAX_EVAL_PARAMS} parameters"
        )));
    }

    if !model.is_finite() {
        return Err(BrutError::Invalid("model has non-finite weights".into()));
    }

    Ok(())
}

/// The caps of an MLP for the dense head of a CNN file: depth, width and parameters.
fn check_cnn_head(model: &CnnModel) -> Result<()> {
    if model.hidden.len() > HARD_MAX_HIDDEN_LAYERS {
        return Err(BrutError::Invalid(format!(
            "model has {} hidden layers, max {HARD_MAX_HIDDEN_LAYERS}",
            model.hidden.len()
        )));
    }

    if model.hidden.iter().any(|&w| w > HARD_MAX_WIDTH) {
        return Err(BrutError::Invalid(format!(
            "model has a layer wider than {HARD_MAX_WIDTH}"
        )));
    }

    if model.param_count() > MAX_EVAL_PARAMS {
        return Err(BrutError::Invalid(format!(
            "model has more than {MAX_EVAL_PARAMS} parameters"
        )));
    }

    Ok(())
}

/// Check a CNN file against the task before it is scored: the shape check of its genome,
/// the parameter cap, `MAX_EVAL_FLOPS` and `MAX_EVAL_SECONDS` for its architecture, then
/// every value against its tensor (`CnnModel::check_values`). This builds no tensor and
/// touches no hidden data.
fn check_cnn(model: &CnnModel, spec: &TaskSpec) -> Result<()> {
    let plan = model.plan(spec)?;

    if plan.params > MAX_EVAL_PARAMS {
        return Err(BrutError::Invalid(format!(
            "model has more than {MAX_EVAL_PARAMS} parameters"
        )));
    }

    let flops = plan.forward_flops.saturating_mul(spec.n_hidden as u64);

    if flops > MAX_EVAL_FLOPS {
        return Err(BrutError::Invalid(format!(
            "scoring the model would cost {flops} FLOPs, max {MAX_EVAL_FLOPS}"
        )));
    }

    let seconds = plan.eval_nanos(spec.n_hidden, score::cnn_chunk_rows(&plan)) / 1e9;

    if seconds > MAX_EVAL_SECONDS {
        return Err(BrutError::Invalid(format!(
            "scoring the model would take about {seconds:.0} s by the time model, max \
             {MAX_EVAL_SECONDS} s"
        )));
    }

    model.check_values(&plan)
}

#[cfg(unix)]
pub(crate) fn check_private(path: &Path) -> Result<()> {
    use std::os::unix::fs::PermissionsExt;

    let mode = std::fs::metadata(path)?.permissions().mode();

    if mode & 0o077 != 0 {
        return Err(BrutError::Invalid(format!(
            "secret file {} is readable by others (mode {:o}); run chmod 600",
            path.display(),
            mode & 0o777
        )));
    }

    Ok(())
}

#[cfg(not(unix))]
pub(crate) fn check_private(path: &Path) -> Result<()> {
    std::fs::metadata(path)?;

    Ok(())
}

/// Round `score` down to a multiple of `step` (step <= 0 means no rounding).
/// Non-finite scores give 0. The result is always finite.
pub fn quantize(score: f64, step: f64) -> f64 {
    if !score.is_finite() {
        return 0.0;
    }

    if !step.is_finite() || step <= 0.0 {
        return score;
    }

    // The tiny epsilon keeps 0.97 / 0.01 = 96.999.. from dropping a whole step.
    let steps = (score / step + 1e-9).floor();
    let value = steps * step;

    // Remove float noise such as 0.9600000000000001.
    (value * 1e12).round() / 1e12
}

/// Generic child-process call used by all request kinds. Same guarantees as
/// `request_hidden_eval` (env_clear, stdin/stdout JSON, output cap, timeout, STOP polling).
///
/// Returns the child's stdout (at most 1 MiB) when it exits with status 0.
pub fn request_eval_json(
    eval_bin: &Path,
    secret_env: &Path,
    budget: usize,
    score_step: f64,
    body: &[u8],
    timeout: Duration,
    stop_file: Option<&Path>,
) -> Result<Vec<u8>> {
    let out = run_evaluator(
        eval_bin,
        secret_env,
        budget,
        score_step,
        body.to_vec(),
        timeout,
        stop_file,
    )?;

    Ok(out.stdout)
}

/// Orchestrator side: run `eval_bin` as a child process.
///
/// The child gets a cleared environment and these args:
/// `--secret <secret_env> --budget <budget> --step <score_step>`.
/// The request is written as JSON to stdin; the response is read as JSON from stdout.
/// Non-zero exit or bad JSON is an `Eval` error with stderr included.
/// A child that does not finish within `timeout` is killed (`Eval` error).
/// When `stop_file` (the kill switch) appears while waiting, the child is killed as well
/// (`Stopped` error). The orchestrator never opens `secret_env` itself.
pub fn request_hidden_eval(
    eval_bin: &Path,
    secret_env: &Path,
    budget: usize,
    score_step: f64,
    req: &EvalRequest,
    timeout: Duration,
    stop_file: Option<&Path>,
) -> Result<EvalResponse> {
    let body = serde_json::to_vec(req)?;
    let out = run_evaluator(
        eval_bin, secret_env, budget, score_step, body, timeout, stop_file,
    )?;

    let resp: EvalResponse = serde_json::from_slice(&out.stdout).map_err(|e| {
        BrutError::Eval(format!("bad evaluator output: {e}; stderr: {}", out.stderr))
    })?;

    if resp.candidate_id != req.candidate_id {
        return Err(BrutError::Eval(
            "evaluator answered for a different candidate".into(),
        ));
    }

    if !resp.hidden_primary.is_finite() {
        return Err(BrutError::Eval(
            "evaluator returned a non-finite score".into(),
        ));
    }

    Ok(resp)
}

/// What a successful `brut-eval` child left on its pipes.
struct EvalOutput {
    stdout: Vec<u8>,
    /// Trimmed, lossy UTF-8 (for error messages).
    stderr: String,
}

/// Shared core of `request_eval_json` and `request_hidden_eval`: start the child, send `body`
/// on stdin, wait with timeout and kill-switch polling, collect both pipes (capped).
fn run_evaluator(
    eval_bin: &Path,
    secret_env: &Path,
    budget: usize,
    score_step: f64,
    body: Vec<u8>,
    timeout: Duration,
    stop_file: Option<&Path>,
) -> Result<EvalOutput> {
    let mut child = Command::new(eval_bin)
        .env_clear()
        .arg("--secret")
        .arg(secret_env)
        .arg("--budget")
        .arg(budget.to_string())
        .arg("--step")
        .arg(format!("{score_step}"))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| BrutError::Eval(format!("cannot start {}: {e}", eval_bin.display())))?;

    // Read both pipes in the background, so a chatty child cannot block on a full pipe.
    let stdout = child.stdout.take().map(read_pipe);
    let stderr = child.stderr.take().map(read_pipe);

    // Write the request in the background too: a large body sent to a child that never reads
    // would otherwise block here, past the timeout and the kill switch. A child that exits
    // early closes stdin; the write then fails and its stderr explains why.
    let writer = child
        .stdin
        .take()
        .map(|mut stdin| std::thread::spawn(move || stdin.write_all(&body)));

    let started = Instant::now();

    let status = loop {
        if let Some(status) = child.try_wait()? {
            break status;
        }

        if started.elapsed() >= timeout {
            // Best effort: the timeout error matters more than a failed kill.
            let _ = child.kill();
            let _ = child.wait();

            return Err(BrutError::Eval(format!(
                "evaluator did not answer within {:.1} s and was stopped",
                timeout.as_secs_f64()
            )));
        }

        if let Some(path) = stop_file.filter(|p| p.exists()) {
            let _ = child.kill();
            let _ = child.wait();

            return Err(BrutError::Stopped(format!(
                "kill switch {} present; evaluator stopped",
                path.display()
            )));
        }

        std::thread::sleep(EVAL_POLL);
    };

    let write_result = match writer {
        Some(handle) => handle
            .join()
            .unwrap_or_else(|_| Err(std::io::Error::other("writer thread panicked"))),
        None => Ok(()),
    };

    let stdout = join_pipe(stdout);
    let stderr = String::from_utf8_lossy(&join_pipe(stderr))
        .trim()
        .to_string();

    if !status.success() {
        return Err(BrutError::Eval(format!(
            "evaluator failed ({status}): {stderr}"
        )));
    }

    if let Err(e) = write_result {
        return Err(BrutError::Eval(format!("could not send request: {e}")));
    }

    Ok(EvalOutput { stdout, stderr })
}

/// Read a child pipe to its end (at most `MAX_EVAL_OUTPUT_BYTES`) in a background thread.
fn read_pipe(pipe: impl Read + Send + 'static) -> std::thread::JoinHandle<Vec<u8>> {
    std::thread::spawn(move || {
        let mut bytes = Vec::new();

        // A read error only shortens the output; the exit status reports the failure.
        let _ = pipe.take(MAX_EVAL_OUTPUT_BYTES).read_to_end(&mut bytes);

        bytes
    })
}

fn join_pipe(handle: Option<std::thread::JoinHandle<Vec<u8>>>) -> Vec<u8> {
    handle.and_then(|h| h.join().ok()).unwrap_or_default()
}

/// Create a new secret file with a fresh random hidden seed, mode 0600.
/// Fails if the file already exists.
pub fn create_secret(secret_env: &Path) -> Result<()> {
    if let Some(dir) = secret_env.parent().filter(|d| !d.as_os_str().is_empty()) {
        create_private_dir(dir)?;
    }

    let seed: u64 = rand::rng().random();
    let text = format!(
        "# Secret of the hidden evaluator. Only brut-eval reads this file.\n\
         # Do not share, commit or print it.\n\
         {HIDDEN_SEED_KEY}={seed}\n"
    );

    let mut options = std::fs::OpenOptions::new();

    options.write(true).create_new(true);

    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;

        options.mode(0o600);
    }

    let mut file = options.open(secret_env)?;

    file.write_all(text.as_bytes())?;
    file.sync_all()?;

    Ok(())
}

pub(crate) fn create_private_dir(dir: &Path) -> Result<()> {
    if dir.is_dir() {
        return Ok(());
    }

    let mut builder = std::fs::DirBuilder::new();

    builder.recursive(true);

    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;

        builder.mode(0o700);
    }

    builder.create(dir)?;

    Ok(())
}

/// Open (create with mode 0600) and exclusively lock `path`.
/// The lock is released when the returned file is dropped (closed).
pub(crate) fn lock_exclusive(path: &Path) -> Result<std::fs::File> {
    let mut options = std::fs::OpenOptions::new();

    options.read(true).write(true).create(true).truncate(false);

    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;

        options.mode(0o600);
    }

    let lock = options.open(path)?;

    lock.lock()?;

    Ok(lock)
}

/// Like `util::write_atomic`, but the new file has mode 0600 (state next to a secret).
/// Temp file next to `path`, fsync, rename, then fsync the directory (best effort).
pub(crate) fn write_private_atomic(path: &Path, bytes: &[u8]) -> Result<()> {
    use std::sync::atomic::{AtomicU64, Ordering};

    static COUNTER: AtomicU64 = AtomicU64::new(0);

    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let name = path
        .file_name()
        .map(|f| f.to_string_lossy().into_owned())
        .unwrap_or_default();
    let tmp = path.with_file_name(format!(".{name}.{}.{n}.tmp", std::process::id()));

    let written = (|| -> Result<()> {
        let mut options = std::fs::OpenOptions::new();

        options.write(true).create_new(true);

        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;

            options.mode(0o600);
        }

        let mut file = options.open(&tmp)?;

        file.write_all(bytes)?;
        file.sync_all()?;
        std::fs::rename(&tmp, path)?;

        Ok(())
    })();

    if written.is_err() {
        // Best effort: the write error matters more than a leftover temp file.
        let _ = std::fs::remove_file(&tmp);

        return written;
    }

    #[cfg(unix)]
    {
        let dir = match path.parent() {
            Some(d) if !d.as_os_str().is_empty() => d,
            _ => Path::new("."),
        };

        // Best effort, as in util::write_atomic.
        if let Ok(d) = std::fs::File::open(dir) {
            let _ = d.sync_all();
        }
    }

    Ok(())
}

/// Read at most `limit` bytes from `reader` (for stdin of `brut-eval`).
pub fn read_limited(reader: impl Read, limit: u64) -> Result<String> {
    let mut text = String::new();

    reader.take(limit + 1).read_to_string(&mut text)?;

    if text.len() as u64 > limit {
        return Err(BrutError::Invalid(format!(
            "input larger than {limit} bytes"
        )));
    }

    Ok(text)
}

/// Read at most `limit` bytes from `reader`; more than `limit` bytes is an error.
pub fn read_limited_bytes(reader: impl Read, limit: u64) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();

    reader
        .take(limit.saturating_add(1))
        .read_to_end(&mut bytes)?;

    if bytes.len() as u64 > limit {
        return Err(BrutError::Invalid(format!(
            "input larger than {limit} bytes"
        )));
    }

    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::genome::Activation;

    fn tmp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("brut-hidden-{name}-{}", std::process::id()));

        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("temp dir");

        dir
    }

    /// The run-directory layout the evaluator expects: `<dir>/secret/hidden.env` (not yet
    /// created) and the models directory `<dir>/models/`. Returns (dir, secret, models).
    fn layout(name: &str) -> (PathBuf, PathBuf, PathBuf) {
        let dir = tmp_dir(name);
        let models = dir.join("models");

        std::fs::create_dir_all(&models).expect("models dir");

        (dir.clone(), dir.join("secret").join("hidden.env"), models)
    }

    fn request(path: &Path) -> EvalRequest {
        EvalRequest {
            candidate_id: "abc".into(),
            model_path: path.to_path_buf(),
            task: TaskKind::TwoSpirals,
        }
    }

    fn write_model(path: &Path, sizes: Vec<usize>) {
        let layers: Vec<(usize, usize)> = sizes.windows(2).map(|w| (w[0], w[1])).collect();
        let model = Mlp {
            sizes: sizes.clone(),
            activation: Activation::Tanh,
            weights: layers.iter().map(|&(i, o)| vec![0.1; i * o]).collect(),
            biases: layers.iter().map(|&(_, o)| vec![0.0; o]).collect(),
        };

        std::fs::write(path, serde_json::to_vec(&model).expect("json")).expect("write");
    }

    #[test]
    fn quantize_rounds_down() {
        assert_eq!(quantize(0.97, 0.01), 0.97);
        assert_eq!(quantize(0.979, 0.01), 0.97);
        assert_eq!(quantize(0.5, 0.25), 0.5);
        assert_eq!(quantize(0.74, 0.25), 0.5);
        assert_eq!(quantize(0.123, 0.0), 0.123);
        assert_eq!(quantize(0.123, -1.0), 0.123);
        assert_eq!(quantize(f64::NAN, 0.01), 0.0);
        assert_eq!(quantize(1.0, 0.01), 1.0);
    }

    #[test]
    fn secret_is_private_and_not_overwritten() {
        let dir = tmp_dir("secret");
        let path = dir.join("secret").join("hidden.env");

        create_secret(&path).expect("create");
        assert!(create_secret(&path).is_err());

        let env = parse_env_file(&path).expect("parse");

        assert!(
            env.get(HIDDEN_SEED_KEY)
                .and_then(|s| s.parse::<u64>().ok())
                .is_some()
        );

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;

            let mode = std::fs::metadata(&path).expect("meta").permissions().mode();

            assert_eq!(mode & 0o777, 0o600);
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[cfg(unix)]
    #[test]
    fn open_refuses_readable_secret() {
        use std::os::unix::fs::PermissionsExt;

        let dir = tmp_dir("readable");
        let path = dir.join("hidden.env");

        std::fs::write(&path, format!("{HIDDEN_SEED_KEY}=5\n")).expect("write");
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o644)).expect("chmod");

        assert!(HiddenEvaluator::open(&path, 3, 0.01).is_err());

        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).expect("chmod");

        assert!(HiddenEvaluator::open(&path, 3, 0.01).is_ok());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn evaluate_charges_budget_and_checks_model() {
        let (dir, secret, models) = layout("eval");
        let good = models.join("good.json");
        let wrong = models.join("wrong.json");

        create_secret(&secret).expect("secret");
        write_model(&good, vec![2, 4, 2]);
        write_model(&wrong, vec![3, 2]);

        let req = |path: &Path| EvalRequest {
            candidate_id: "abc".into(),
            model_path: path.to_path_buf(),
            task: TaskKind::TwoSpirals,
        };

        let mut ev = HiddenEvaluator::open(&secret, 2, 0.01).expect("open");
        let resp = ev.evaluate(&req(&good)).expect("eval");

        assert_eq!(resp.candidate_id, "abc");
        assert!((0.0..=1.0).contains(&resp.hidden_primary));
        assert_eq!(resp.queries_used, 1);
        assert_eq!(resp.queries_left, 1);

        // A model that does not fit the task is refused before the hidden data is touched,
        // so it costs no query.
        assert!(matches!(
            ev.evaluate(&req(&wrong)),
            Err(BrutError::Invalid(_))
        ));

        let resp = ev.evaluate(&req(&good)).expect("eval");

        assert_eq!(resp.queries_used, 2);
        assert_eq!(resp.queries_left, 0);

        // Budget persists across reopen.
        let mut ev = HiddenEvaluator::open(&secret, 2, 0.01).expect("open");

        assert!(matches!(ev.evaluate(&req(&good)), Err(BrutError::Eval(_))));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn cnn_files_are_scored_and_bad_ones_cost_no_query() {
        use crate::cnn::genome::{
            Block, CnnActivation, CnnGenome, Dims, Downsample, Head, Norm, Stage, Stem,
        };

        let (dir, secret, models) = layout("cnn");
        let spec = TaskKind::Shapes.spec();
        let good =
            crate::cnn::model::testing::untrained(&crate::cnn::genome::testing::vgg(), &spec, 1);
        let good_path = models.join("good.json");

        create_secret(&secret).expect("secret");
        std::fs::write(
            &good_path,
            serde_json::to_vec(&Model::Cnn(good.clone())).expect("json"),
        )
        .expect("write");

        let req = |path: &Path| EvalRequest {
            candidate_id: "cnn".into(),
            model_path: path.to_path_buf(),
            task: TaskKind::Shapes,
        };
        let mut ev = HiddenEvaluator::open(&secret, 10, 0.01).expect("open");
        let resp = ev.evaluate(&req(&good_path)).expect("a valid CNN file");

        assert!((0.0..=1.0).contains(&resp.hidden_primary));
        assert_eq!(resp.queries_used, 1);

        // Architectures without trained values: they must be refused before the values.
        let wide = |width: usize, kernel: usize, stages: usize| {
            let stage = Stage {
                block: Block::Plain,
                width,
                kernel,
                stride: 1,
                down: Downsample::Conv,
                dilation: 1,
                groups: 1,
                se: false,
                residual: false,
                norm: Norm::None,
                activation: CnnActivation::Relu,
            };
            let arch = CnnGenome {
                dims: Dims::TwoD,
                stem: Stem {
                    width: 64,
                    kernel: 3,
                    stride: 1,
                    down: Downsample::Conv,
                    norm: Norm::None,
                    activation: CnnActivation::Relu,
                },
                stages: vec![stage; stages],
                head: Head::GlobalAvgPool,
            };

            CnnModel::new(arch, Vec::new(), Activation::Relu, Vec::new(), Vec::new())
        };
        // The file of the review of ticket 02 that took 145 s to score (release build): 12
        // depthwise 5x5 convolutions of 256 channels with layer norm and GELU. It passes
        // the parameter and FLOP caps (9.2e10 FLOPs); the time model refuses it.
        let depthwise = {
            let stage = Stage {
                block: Block::Plain,
                width: 256,
                kernel: 5,
                stride: 1,
                down: Downsample::Conv,
                dilation: 1,
                groups: 256,
                se: false,
                residual: false,
                norm: Norm::Layer,
                activation: CnnActivation::Gelu,
            };
            let arch = CnnGenome {
                dims: Dims::TwoD,
                stem: Stem {
                    width: 256,
                    kernel: 5,
                    stride: 1,
                    down: Downsample::Conv,
                    norm: Norm::Layer,
                    activation: CnnActivation::Gelu,
                },
                stages: vec![stage; 12],
                head: Head::GlobalAvgPool,
            };

            CnnModel::new(arch, Vec::new(), Activation::Relu, Vec::new(), Vec::new())
        };
        let depthwise_plan = depthwise.plan(&spec).expect("plan");

        assert!(depthwise_plan.params <= MAX_EVAL_PARAMS);
        assert!(depthwise_plan.forward_flops * spec.n_hidden as u64 <= MAX_EVAL_FLOPS);

        let text = |model: &CnnModel| serde_json::to_string(model).expect("json");
        let good_text = text(&good);
        let mut short = good.clone();

        short.params[0].pop();

        let mut one_d = good.clone();

        one_d.arch.dims = Dims::OneD;

        let mut deep = good.clone();

        deep.hidden = vec![1; HARD_MAX_HIDDEN_LAYERS + 1];

        let over_cap = format!("[{}]", vec!["0"; HARD_MAX_PARAMS + 1].join(","));
        let bad: Vec<(&str, String, &str)> = vec![
            (
                "only the kind",
                "{\"kind\":\"cnn\"}".into(),
                "not a valid CNN model",
            ),
            (
                "an unknown field",
                good_text.replacen("\"hidden\":", "\"extra\":0,\"hidden\":", 1),
                "not a valid CNN model",
            ),
            (
                "an unknown kind",
                good_text.replacen("\"cnn\"", "\"rnn\"", 1),
                "not a valid model",
            ),
            (
                "too many values",
                good_text.replacen(
                    "\"batch_norm\":[",
                    &format!("\"batch_norm\":[{over_cap},"),
                    1,
                ),
                "not a valid CNN model",
            ),
            ("too costly to score", text(&wide(256, 3, 2)), "FLOPs, max"),
            ("too slow to score", text(&depthwise), "by the time model"),
            ("too many parameters", text(&wide(256, 11, 2)), "parameters"),
            ("too deep a head", text(&deep), "hidden layers"),
            ("a tensor too short", text(&short), "tensor 1 has"),
            ("a 1-D genome on an image", text(&one_d), "needs a sequence"),
            // serde_json refuses a number beyond the range of f32 while it parses.
            (
                "a value beyond f32",
                good_text.replacen("[1.0,", "[1e40,", 1),
                "not a valid CNN model",
            ),
        ];

        for (what, contents, expected) in bad {
            let path = models.join("bad.json");

            std::fs::write(&path, &contents).expect("write");

            let message = match ev.evaluate(&req(&path)) {
                Err(BrutError::Invalid(m)) => m,
                other => panic!("{what}: expected Invalid, got {other:?}"),
            };

            assert!(message.contains(expected), "{what}: {message}");
        }

        // None of the refused files cost a query.
        assert_eq!(ev.evaluate(&req(&good_path)).expect("eval").queries_used, 2);

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A CNN file just below `MAX_EVAL_SECONDS` scores within `EVAL_TIMEOUT_SECONDS`. The
    /// file is the slowest kind of the review of ticket 02: depthwise convolutions of 256
    /// channels with layer norm and GELU, with as many stages and as large a kernel as the
    /// time model admits. This takes up to a minute and measures time, so it runs only on
    /// request, alone: `cargo test --lib just_below -- --ignored --test-threads 1`.
    #[test]
    #[ignore]
    fn a_file_just_below_the_time_cap_scores_within_the_timeout() {
        use crate::cnn::genome::{CnnActivation, CnnGenome, Dims, Downsample, Head, Norm, Stem};
        use crate::cnn::genome::{MAX_STAGES, Stage, testing};

        let spec = TaskKind::Shapes.spec();
        let stack = |kernel: usize, stages: usize| {
            let stage = Stage {
                groups: 256,
                norm: Norm::Layer,
                ..testing::plain(256, kernel, CnnActivation::Gelu)
            };
            let cnn = CnnGenome {
                dims: Dims::TwoD,
                stem: Stem {
                    width: 256,
                    kernel,
                    stride: 1,
                    down: Downsample::Conv,
                    norm: Norm::Layer,
                    activation: CnnActivation::Gelu,
                },
                stages: vec![stage; stages],
                head: Head::GlobalAvgPool,
            };

            testing::genome(cnn, Vec::new(), Activation::Relu)
        };
        let estimate = |genome: &crate::genome::Genome| {
            let plan = genome.cnn_plan(&spec).expect("plan");

            plan.eval_nanos(spec.n_hidden, score::cnn_chunk_rows(&plan)) / 1e9
        };
        let genome = [1, 3, 5]
            .into_iter()
            .flat_map(|kernel| (1..=MAX_STAGES).map(move |stages| stack(kernel, stages)))
            .filter(|genome| estimate(genome) <= MAX_EVAL_SECONDS)
            .max_by(|a, b| estimate(a).total_cmp(&estimate(b)))
            .expect("a layout below the cap");
        let model = crate::cnn::model::testing::untrained(&genome, &spec, 1);
        let (dir, secret, models) = layout("time-cap");
        let path = models.join("slow.json");

        create_secret(&secret).expect("secret");
        std::fs::write(&path, serde_json::to_vec(&Model::Cnn(model)).expect("json"))
            .expect("write");

        let mut ev = HiddenEvaluator::open(&secret, 10, 0.01).expect("open");
        let started = std::time::Instant::now();
        let resp = ev
            .evaluate(&EvalRequest {
                candidate_id: "slow".into(),
                model_path: path,
                task: TaskKind::Shapes,
            })
            .expect("a file below the cap is scored");
        let seconds = started.elapsed().as_secs_f64();

        eprintln!(
            "estimate {:.1} s, scored in {seconds:.1} s: {}",
            estimate(&genome),
            genome.summary()
        );

        assert_eq!(resp.queries_used, 1);
        assert!(seconds < EVAL_TIMEOUT_SECONDS, "{seconds} s");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn concurrent_queries_are_counted_exactly() {
        let (dir, secret, models) = layout("concurrent");
        let model = models.join("m.json");
        let budget = 5;

        create_secret(&secret).expect("secret");
        write_model(&model, vec![2, 4, 2]);

        let req = EvalRequest {
            candidate_id: "c".into(),
            model_path: model,
            task: TaskKind::TwoSpirals,
        };

        let results: Vec<Result<EvalResponse>> = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..16)
                .map(|_| {
                    scope.spawn(|| {
                        let mut ev = HiddenEvaluator::open(&secret, budget, 0.01)?;

                        ev.evaluate(&req)
                    })
                })
                .collect();

            handles
                .into_iter()
                .map(|h| {
                    h.join()
                        .unwrap_or_else(|_| Err(BrutError::Eval("thread panicked".into())))
                })
                .collect()
        });

        let mut used: Vec<usize> = Vec::new();

        for r in &results {
            match r {
                Ok(resp) => used.push(resp.queries_used),
                Err(BrutError::Eval(m)) => assert!(m.contains("budget exhausted"), "{m}"),
                Err(e) => panic!("unexpected error: {e}"),
            }
        }

        used.sort_unstable();

        assert_eq!(used, (1..=budget).collect::<Vec<_>>());

        let state_text =
            std::fs::read_to_string(dir.join("secret").join(STATE_FILE)).expect("state");
        let state: EvalState = serde_json::from_str(&state_text).expect("state json");

        assert_eq!(state.queries_used, budget);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn secret_budget_cap_applies() {
        let (dir, secret, models) = layout("cap");

        std::fs::create_dir_all(dir.join("secret")).expect("secret dir");
        write_private(
            &secret,
            &format!("{HIDDEN_SEED_KEY}=5\n{HIDDEN_BUDGET_KEY}=0\n"),
        );

        let model = models.join("m.json");

        write_model(&model, vec![2, 2]);

        let mut ev = HiddenEvaluator::open(&secret, 100, 0.01).expect("open");
        let req = EvalRequest {
            candidate_id: "x".into(),
            model_path: model,
            task: TaskKind::TwoSpirals,
        };

        assert!(ev.evaluate(&req).is_err());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn model_paths_outside_the_models_directory_are_refused() {
        let (dir, secret, models) = layout("outside");
        let outside = dir.join("outside.json");
        let inside = models.join("m.json");

        create_secret(&secret).expect("secret");
        write_model(&outside, vec![2, 4, 2]);
        write_model(&inside, vec![2, 4, 2]);

        let mut refused = vec![outside.clone(), models.join("..").join("outside.json")];

        #[cfg(unix)]
        {
            let link = models.join("link.json");

            std::os::unix::fs::symlink(&outside, &link).expect("symlink");
            refused.push(link);
        }

        let mut ev = HiddenEvaluator::open(&secret, 10, 0.01).expect("open");

        for path in refused {
            match ev.evaluate(&request(&path)) {
                Err(BrutError::Invalid(m)) => assert!(m.contains("models directory"), "{m}"),
                other => panic!("{}: expected Invalid, got {other:?}", path.display()),
            }
        }

        // Refused paths cost no query.
        let resp = ev.evaluate(&request(&inside)).expect("inside");

        assert_eq!(resp.queries_used, 1);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn invalid_model_files_do_not_echo_their_content() {
        let (dir, secret, models) = layout("echo");
        let path = models.join("private.json");

        create_secret(&secret).expect("secret");
        std::fs::write(&path, r#"{"sizes":"some-private-json-value"}"#).expect("write");

        let mut ev = HiddenEvaluator::open(&secret, 10, 0.01).expect("open");
        let err = ev.evaluate(&request(&path)).expect_err("not a model");

        assert!(matches!(err, BrutError::Invalid(_)), "{err}");
        assert!(!err.to_string().contains("private-json"), "{err}");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn models_dir_can_be_set_in_the_secret_file() {
        let (dir, secret, models) = layout("models-key");
        let elsewhere = dir.join("elsewhere");
        let there = elsewhere.join("m.json");
        let default = models.join("m.json");

        std::fs::create_dir_all(&elsewhere).expect("dir");
        std::fs::create_dir_all(dir.join("secret")).expect("secret dir");
        write_private(
            &secret,
            &format!(
                "{HIDDEN_SEED_KEY}=5\n{HIDDEN_MODELS_DIR_KEY}={}\n",
                elsewhere.display()
            ),
        );
        write_model(&there, vec![2, 4, 2]);
        write_model(&default, vec![2, 4, 2]);

        let mut ev = HiddenEvaluator::open(&secret, 10, 0.01).expect("open");

        assert!(ev.evaluate(&request(&there)).is_ok());
        assert!(matches!(
            ev.evaluate(&request(&default)),
            Err(BrutError::Invalid(_))
        ));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn score_step_has_a_floor_the_caller_cannot_lower() {
        let dir = tmp_dir("step");
        let secret = dir.join("hidden.env");

        create_secret(&secret).expect("secret");

        for (arg, expected) in [(0.0, 0.01), (1e-300, 0.01), (0.009, 0.01), (0.05, 0.05)] {
            let ev = HiddenEvaluator::open(&secret, 3, arg).expect("open");

            assert_eq!(ev.score_step(), expected, "--step {arg}");
        }

        // The secret file can raise the floor further, and the larger value wins.
        let text = std::fs::read_to_string(&secret).expect("read");

        std::fs::remove_file(&secret).expect("remove");
        write_private(&secret, &format!("{text}{HIDDEN_STEP_KEY}=0.1\n"));

        assert_eq!(
            HiddenEvaluator::open(&secret, 3, 0.01)
                .expect("open")
                .score_step(),
            0.1
        );
        assert_eq!(
            HiddenEvaluator::open(&secret, 3, 0.2)
                .expect("open")
                .score_step(),
            0.2
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Write a file with mode 0600 (the evaluator refuses readable secrets).
    fn write_private(path: &Path, text: &str) {
        std::fs::write(path, text).expect("write");

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;

            std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600)).expect("chmod");
        }
    }

    #[test]
    fn read_limited_caps_input() {
        assert_eq!(read_limited(&b"hello"[..], 5).expect("fits"), "hello");
        assert!(read_limited(&b"hello!"[..], 5).is_err());
    }

    #[test]
    fn read_limited_bytes_stops_at_the_limit() {
        assert_eq!(
            read_limited_bytes(&b"abc"[..], 3).expect("fits"),
            b"abc".to_vec()
        );

        // An endless reader must not be read to the end.
        let endless = std::io::repeat(0);

        assert!(matches!(
            read_limited_bytes(endless, 1024),
            Err(BrutError::Invalid(_))
        ));
    }

    #[cfg(unix)]
    #[test]
    fn load_model_refuses_devices_without_reading_them() {
        // /dev/zero reports length 0 and never ends; it must be refused before reading.
        for dev in ["/dev/null", "/dev/zero"] {
            match load_model(Path::new(dev)) {
                Err(BrutError::Invalid(m)) => assert!(m.contains("regular file"), "{m}"),
                other => panic!("{dev}: expected Invalid, got {other:?}"),
            }
        }
    }

    #[test]
    fn missing_binary_is_eval_error() {
        let req = EvalRequest {
            candidate_id: "x".into(),
            model_path: "m.json".into(),
            task: TaskKind::Parity,
        };
        let r = request_hidden_eval(
            Path::new("/nonexistent/brut-eval"),
            Path::new("s.env"),
            1,
            0.01,
            &req,
            Duration::from_secs(5),
            None,
        );

        assert!(matches!(r, Err(BrutError::Eval(_))));
    }

    #[cfg(unix)]
    #[test]
    fn stop_file_kills_a_running_evaluator() {
        let dir = tmp_dir("stop");
        let bin = script(&dir, "hang.sh", "exec /bin/sleep 30");
        let stop = dir.join("STOP");
        let started = std::time::Instant::now();

        std::thread::scope(|scope| {
            scope.spawn(|| {
                std::thread::sleep(Duration::from_millis(300));
                std::fs::write(&stop, "stop").expect("write STOP");
            });

            let r = request_with_retry(&bin, Duration::from_secs(60), Some(&stop));

            assert!(matches!(&r, Err(BrutError::Stopped(_))), "{r:?}");
        });

        assert!(started.elapsed() < Duration::from_secs(10));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn load_model_refuses_too_wide_or_too_deep_models() {
        let dir = tmp_dir("shape-caps");
        let path = dir.join("m.json");
        let deep = |hidden: usize| {
            let mut sizes = vec![2];

            sizes.extend(vec![1; hidden]);
            sizes.push(2);

            sizes
        };

        write_model(&path, vec![2, HARD_MAX_WIDTH + 1, 2]);
        assert!(matches!(load_model(&path), Err(BrutError::Invalid(_))));

        write_model(&path, deep(HARD_MAX_HIDDEN_LAYERS + 1));
        assert!(matches!(load_model(&path), Err(BrutError::Invalid(_))));

        // Exactly at the caps is fine.
        write_model(&path, vec![2, HARD_MAX_WIDTH, 2]);
        assert!(load_model(&path).is_ok());

        write_model(&path, deep(HARD_MAX_HIDDEN_LAYERS));
        assert!(load_model(&path).is_ok());

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Write an executable shell script (Unix only).
    #[cfg(unix)]
    fn script(dir: &Path, name: &str, body: &str) -> PathBuf {
        use std::os::unix::fs::PermissionsExt;

        let path = dir.join(name);

        std::fs::write(&path, format!("#!/bin/sh\n{body}\n")).expect("write script");
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).expect("chmod");

        path
    }

    /// Call `request_hidden_eval`, retrying when another test thread still holds the
    /// fresh script open ("Text file busy" right after writing it).
    #[cfg(unix)]
    fn request_with_retry(
        bin: &Path,
        timeout: Duration,
        stop: Option<&Path>,
    ) -> Result<EvalResponse> {
        let req = EvalRequest {
            candidate_id: "x".into(),
            model_path: "m.json".into(),
            task: TaskKind::Parity,
        };

        for _ in 0..20 {
            let r = request_hidden_eval(bin, Path::new("s.env"), 1, 0.01, &req, timeout, stop);

            match &r {
                Err(BrutError::Eval(m)) if m.contains("busy") => {
                    std::thread::sleep(Duration::from_millis(50));
                }
                _ => return r,
            }
        }

        Err(BrutError::Eval("script stayed busy".into()))
    }

    #[cfg(unix)]
    #[test]
    fn hanging_evaluator_is_stopped_after_the_timeout() {
        let dir = tmp_dir("hang");
        let bin = script(&dir, "hang.sh", "exec /bin/sleep 30");
        let started = std::time::Instant::now();
        let r = request_with_retry(&bin, Duration::from_millis(300), None);

        assert!(
            matches!(&r, Err(BrutError::Eval(m)) if m.contains("did not answer")),
            "{r:?}"
        );
        assert!(started.elapsed() < Duration::from_secs(10));

        let _ = std::fs::remove_dir_all(&dir);
    }

    fn learner_request() -> crate::learner_hidden::LearnerEvalRequest {
        use crate::learner::types::{Instr, LearnerGenome, OpCode, Tier};

        crate::learner_hidden::LearnerEvalRequest {
            run_id: "a".repeat(64),
            candidate_id: "3fa1c2d4e5f60718".into(),
            protocol_hash: "b".repeat(64),
            tier: Tier::HiddenGrammar,
            program: LearnerGenome {
                setup: vec![Instr {
                    op: OpCode::SConst,
                    a: 0,
                    b: 0,
                    out: 3,
                    c: 0.05,
                }],
                predict: vec![Instr {
                    op: OpCode::MatVec,
                    a: 0,
                    b: 0,
                    out: 1,
                    c: 0.0,
                }],
                learn: Vec::new(),
            },
        }
    }

    fn arc_request() -> crate::learner_hidden::ArcEvalRequest {
        use crate::learner::types::{EvalSet, SourceId};

        let l = learner_request();

        crate::learner_hidden::ArcEvalRequest {
            run_id: l.run_id,
            candidate_id: l.candidate_id,
            protocol_hash: l.protocol_hash,
            set: EvalSet::A1Eval,
            program: l.program,
            train_sources: vec![SourceId::Synthetic],
        }
    }

    /// T-ENV-1 (legacy part): no "kind" parses as Mlp and round-trips byte-identically.
    #[test]
    fn legacy_request_parses_as_mlp_and_round_trips_byte_identically() {
        for task in [TaskKind::TwoSpirals, TaskKind::Parity] {
            let req = EvalRequest {
                candidate_id: "abc".into(),
                model_path: "runs/r/models/m.json".into(),
                task,
            };
            let text = serde_json::to_string(&req).expect("json");
            let parsed = AnyEvalRequest::from_json(&text).expect("parse");

            assert_eq!(parsed, AnyEvalRequest::Mlp(req.clone()));
            assert_eq!(parsed.to_json().expect("to_json"), text);

            // Exactly the bytes request_hidden_eval sends.
            assert_eq!(
                parsed.to_json().expect("to_json").into_bytes(),
                serde_json::to_vec(&req).expect("bytes")
            );
        }
    }

    /// T-ENV-1 (tagged part): learner and ARC requests carry the tag first and round-trip.
    #[test]
    fn tagged_requests_round_trip() {
        let learner = AnyEvalRequest::Learner(learner_request());
        let text = learner.to_json().expect("json");

        assert!(
            text.starts_with("{\"kind\":\"learner\",\"run_id\":"),
            "{text}"
        );
        assert_eq!(AnyEvalRequest::from_json(&text).expect("parse"), learner);

        let arc = AnyEvalRequest::Arc(arc_request());
        let text = arc.to_json().expect("json");

        assert!(text.starts_with("{\"kind\":\"arc\",\"run_id\":"), "{text}");
        assert_eq!(AnyEvalRequest::from_json(&text).expect("parse"), arc);

        // The kind may appear anywhere in the object.
        let moved = text.replacen("\"kind\":\"arc\",", "", 1).replacen(
            "\"set\":",
            "\"kind\":\"arc\",\"set\":",
            1,
        );

        assert_eq!(AnyEvalRequest::from_json(&moved).expect("parse"), arc);
    }

    #[test]
    fn compact_program_json_from_the_contract_is_accepted() {
        let text = format!(
            "{{\"kind\":\"learner\",\"run_id\":\"{a}\",\"candidate_id\":\"3fa1c2d4e5f60718\",\
             \"protocol_hash\":\"{b}\",\"tier\":\"hidden_grammar\",\"program\":{{\
             \"setup\":[{{\"op\":\"SConst\",\"out\":3,\"c\":0.05}}],\
             \"predict\":[{{\"op\":\"MatVec\",\"a\":0,\"b\":0,\"out\":1}}],\"learn\":[]}}}}",
            a = "a".repeat(64),
            b = "b".repeat(64)
        );

        assert_eq!(
            AnyEvalRequest::from_json(&text).expect("parse"),
            AnyEvalRequest::Learner(learner_request())
        );
    }

    /// T-ENV-1 (refusals): unknown kinds and bad shapes; errors never quote the request.
    #[test]
    fn bad_requests_are_rejected_without_quoting_them() {
        let private = "zz-private-request-text";
        let mut learner = serde_json::to_value(learner_request()).expect("json");

        learner["kind"] = serde_json::json!("learner");

        let mut extra_field = learner.clone();

        extra_field[private] = serde_json::json!(1);

        let mut bad_tier = learner.clone();

        bad_tier["tier"] = serde_json::json!(private);

        let mut bad_op = learner.clone();

        bad_op["program"]["setup"][0]["op"] = serde_json::json!(private);

        let mut missing = learner.clone();

        missing.as_object_mut().expect("object").remove("program");

        let mut legacy_extra = serde_json::to_value(EvalRequest {
            candidate_id: "abc".into(),
            model_path: "m.json".into(),
            task: TaskKind::Parity,
        })
        .expect("json");

        legacy_extra["task"] = serde_json::json!(private);

        let cases = [
            format!("{{\"kind\":\"{private}\"}}"),
            "{\"kind\":\"mlp\",\"candidate_id\":\"x\"}".to_string(),
            "{\"kind\":7}".to_string(),
            "{\"kind\":null}".to_string(),
            format!("[\"{private}\"]"),
            format!("\"{private}\""),
            extra_field.to_string(),
            bad_tier.to_string(),
            bad_op.to_string(),
            missing.to_string(),
            legacy_extra.to_string(),
            format!("{{\"candidate_id\":\"{private}\""),
            format!("not json {private}"),
            String::new(),
        ];

        for text in cases {
            let err = AnyEvalRequest::from_json(&text).expect_err("must be refused");

            assert!(matches!(err, BrutError::Invalid(_)), "{text}: {err}");
            assert!(!err.to_string().contains("zz-private"), "{err}");
        }
    }

    /// Retry while a fresh script is still busy ("Text file busy" from a parallel test).
    #[cfg(unix)]
    fn retry_busy<T>(call: impl Fn() -> Result<T>) -> Result<T> {
        for _ in 0..20 {
            match call() {
                Err(BrutError::Eval(m)) if m.contains("busy") => {
                    std::thread::sleep(Duration::from_millis(50));
                }
                other => return other,
            }
        }

        Err(BrutError::Eval("script stayed busy".into()))
    }

    #[cfg(unix)]
    #[test]
    fn request_eval_json_sends_the_body_and_returns_stdout() {
        let dir = tmp_dir("json-call");
        let body_file = dir.join("body.txt");
        let bin = script(
            &dir,
            "echo.sh",
            &format!(
                "cat > '{}'\nprintf '%s' '{{\"ok\":1}}'",
                body_file.display()
            ),
        );
        let out = retry_busy(|| {
            request_eval_json(
                &bin,
                Path::new("s.env"),
                1,
                0.01,
                b"{\"x\":2}",
                Duration::from_secs(20),
                None,
            )
        })
        .expect("call");

        assert_eq!(out, b"{\"ok\":1}");
        assert_eq!(std::fs::read(&body_file).expect("body"), b"{\"x\":2}");

        // A failing child is an Eval error that carries its stderr.
        let failing = script(&dir, "fail.sh", "echo broken-thing >&2\nexit 3");
        let r = retry_busy(|| {
            request_eval_json(
                &failing,
                Path::new("s.env"),
                1,
                0.01,
                b"{}",
                Duration::from_secs(20),
                None,
            )
        });

        assert!(
            matches!(&r, Err(BrutError::Eval(m)) if m.contains("broken-thing")),
            "{r:?}"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[cfg(unix)]
    #[test]
    fn large_request_to_a_hanging_evaluator_still_times_out() {
        let dir = tmp_dir("big-body");
        let bin = script(&dir, "hang.sh", "exec /bin/sleep 30");
        let body = vec![b' '; 4 << 20];
        let started = Instant::now();
        let r = retry_busy(|| {
            request_eval_json(
                &bin,
                Path::new("s.env"),
                1,
                0.01,
                &body,
                Duration::from_millis(300),
                None,
            )
        });

        assert!(
            matches!(&r, Err(BrutError::Eval(m)) if m.contains("did not answer")),
            "{r:?}"
        );
        assert!(started.elapsed() < Duration::from_secs(10));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn private_atomic_write_and_lock() {
        let dir = tmp_dir("private-write");
        let path = dir.join("state.json");

        write_private_atomic(&path, b"one").expect("write");
        write_private_atomic(&path, b"two").expect("overwrite");

        assert_eq!(std::fs::read(&path).expect("read"), b"two");
        assert_eq!(std::fs::read_dir(&dir).expect("list").count(), 1);

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;

            let mode = std::fs::metadata(&path).expect("meta").permissions().mode();

            assert_eq!(mode & 0o777, 0o600);
        }

        // While the lock is held, a second handle on the same file cannot lock it.
        let lock = lock_exclusive(&dir.join(LOCK_FILE)).expect("lock");
        let second = std::fs::File::open(dir.join(LOCK_FILE)).expect("open");

        assert!(matches!(
            second.try_lock(),
            Err(std::fs::TryLockError::WouldBlock)
        ));

        drop(lock);
        assert!(second.try_lock().is_ok());

        let _ = std::fs::remove_dir_all(&dir);
    }
}
