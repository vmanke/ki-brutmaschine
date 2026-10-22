//! Episode JSON-lines protocol and the in-process DSL oracle (Stage-2 bridge).
//!
//! Owner: WP2. Contract section 5.11. Pure: no I/O.
//! Known difference (documented, not a bug): an external learner could use unlabeled `Eval`
//! inputs; the in-process DSL cannot (memory clone).
//!
//! One episode (section 6.2 over the wire):
//! Setup -> Ready; per step Predict{x} -> Prediction{raw y}, the host links and checks,
//! then Learn{x, y} -> Learned; at each checkpoint Eval{xs} -> Evaluated{raw ys};
//! End -> Ready. `Learn.x` repeats the input for stateless learners; the DSL learner
//! ignores it (its Predict may have changed v0, exactly as in `run_learner`).
//! Differences to `run_learner`: no wall-clock or stop checks (an external sandbox
//! enforces its own limits), and a malformed task or protocol violation is an Err.
#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};

use crate::learner::interp::{self, Memory};
use crate::learner::protocol::{
    self, Stream, aborted, aulc, checkpoint_steps, label_values, quality, static_flops,
};
use crate::learner::types::{
    Abort, LearnKind, LearnerGenome, MAX_ABS_TRAIN_ERROR, MAX_K, TaskInstance, TaskResult,
};
use crate::util::{BrutError, Result, derive_seed};

/// Host -> learner message (one JSON object per line).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
pub enum HostMsg {
    Setup {
        f: u16,
        kind: LearnKind,
        steps: u32,
    },
    Predict {
        x: Vec<f32>,
    },
    Learn {
        x: Vec<f32>,
        y: f32,
    },
    /// Validation batch. The learner must not learn from it (only enforceable in-process).
    Eval {
        xs: Vec<Vec<f32>>,
    },
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

    /// Static (setup, predict, learn) FLOP costs at width f, when the learner can state
    /// them. The host fills `TaskResult::flops` from them; None gives flops = 0.
    /// (Additive default method; external learners keep the default.)
    fn static_costs(&self, f: usize) -> Option<(u64, u64, u64)> {
        let _ = f;

        None
    }
}

/// In-process DSL implementation; the oracle for any external (sandboxed) learner.
pub struct DslWireLearner {
    genome: LearnerGenome,
    /// derive_seed(task_seed, "interp/<attempt>").
    rng_seed: u64,
    state: State,
}

/// Memory of the running episode, monomorphised per width.
enum State {
    Idle,
    F8(Box<Episode<8>>),
    F16(Box<Episode<16>>),
    F32(Box<Episode<32>>),
}

struct Episode<const F: usize> {
    mem: Memory<F>,
    kind: LearnKind,
}

impl DslWireLearner {
    pub fn new(g: LearnerGenome, task_seed: u64, attempt: u32) -> Self {
        DslWireLearner {
            genome: g,
            rng_seed: derive_seed(task_seed, &format!("interp/{attempt}")),
            state: State::Idle,
        }
    }

    /// Validate the episode, refuse over-cost programs (Err(Budget)), run Setup.
    fn setup(&mut self, f: usize, kind: LearnKind) -> Result<LearnerMsg> {
        if let LearnKind::Multiclass { k } = kind
            && !(3..=MAX_K).contains(&(k as usize))
        {
            return Err(wire_error("class count out of range"));
        }

        if kind.k_out() > f || !matches!(f, 8 | 16 | 32) {
            return Err(wire_error("unsupported width"));
        }

        self.genome.check_cost()?;

        self.state = match f {
            8 => State::F8(Box::new(Episode::start(&self.genome, self.rng_seed, kind))),
            16 => State::F16(Box::new(Episode::start(&self.genome, self.rng_seed, kind))),
            _ => State::F32(Box::new(Episode::start(&self.genome, self.rng_seed, kind))),
        };

        Ok(LearnerMsg::Ready)
    }
}

impl WireLearner for DslWireLearner {
    fn handle(&mut self, msg: &HostMsg) -> Result<LearnerMsg> {
        match msg {
            HostMsg::Setup { f, kind, .. } => self.setup(*f as usize, *kind),
            HostMsg::End => {
                self.state = State::Idle;

                Ok(LearnerMsg::Ready)
            }
            _ => match &mut self.state {
                State::Idle => Err(wire_error("message before Setup")),
                State::F8(ep) => ep.handle(&self.genome, msg),
                State::F16(ep) => ep.handle(&self.genome, msg),
                State::F32(ep) => ep.handle(&self.genome, msg),
            },
        }
    }

    fn static_costs(&self, f: usize) -> Option<(u64, u64, u64)> {
        Some((
            self.genome.setup_cost(f),
            self.genome.predict_cost(f),
            self.genome.learn_cost(f),
        ))
    }
}

impl<const F: usize> Episode<F> {
    fn start(g: &LearnerGenome, rng_seed: u64, kind: LearnKind) -> Self {
        let mut mem = Memory::<F>::new(rng_seed);

        interp::run(&g.setup, &mut mem);

        Episode { mem, kind }
    }

    fn handle(&mut self, g: &LearnerGenome, msg: &HostMsg) -> Result<LearnerMsg> {
        let k = self.kind.k_out();

        match msg {
            HostMsg::Predict { x } => {
                check_width(x, F)?;

                Ok(LearnerMsg::Prediction {
                    y: predict_raw(&mut self.mem, g, self.kind, x, k),
                })
            }
            HostMsg::Learn { x, y } => {
                check_width(x, F)?;

                self.mem.set_label(self.kind, *y);
                interp::run(&g.learn, &mut self.mem);

                Ok(LearnerMsg::Learned)
            }
            HostMsg::Eval { xs } => {
                let mut eval = self.mem.clone();
                let mut ys = Vec::with_capacity(xs.len());

                for x in xs {
                    check_width(x, F)?;
                    ys.push(predict_raw(&mut eval, g, self.kind, x, k));
                }

                Ok(LearnerMsg::Evaluated { ys })
            }
            HostMsg::Setup { .. } | HostMsg::End => Err(wire_error("unexpected message")),
        }
    }
}

/// set_input, clear_label, Predict; copy the raw outputs, then link in memory (as the
/// protocol does, so later programs see the linked v1).
fn predict_raw<const F: usize>(
    mem: &mut Memory<F>,
    g: &LearnerGenome,
    kind: LearnKind,
    x: &[f32],
    k: usize,
) -> Vec<f32> {
    mem.set_input(x);
    mem.clear_label();
    interp::run(&g.predict, mem);

    let raw = mem.v[1][..k.min(F)].to_vec();

    mem.apply_link(kind);

    raw
}

fn check_width(x: &[f32], f: usize) -> Result<()> {
    if x.len() == f {
        Ok(())
    } else {
        Err(wire_error("input has the wrong width"))
    }
}

fn wire_error(what: &str) -> BrutError {
    BrutError::Invalid(format!("wire: {what}"))
}

/// Drive any WireLearner through section 6.2 and score it. For DslWireLearner the result
/// MUST equal run_learner bit-for-bit.
///
/// The host links raw outputs with the interpreter's own `Memory::apply_link` (on a scratch
/// memory), so linked values are bit-identical to the in-process run. A learner that
/// answers Setup with Err(Budget) scores as abort OverCost (flops 0), like `run_learner`.
pub fn run_wire(learner: &mut dyn WireLearner, task: &TaskInstance) -> Result<TaskResult> {
    protocol::validate_task(task)?;

    let setup = HostMsg::Setup {
        f: task.f as u16,
        kind: task.kind,
        steps: task.steps as u32,
    };

    match learner.handle(&setup) {
        Ok(LearnerMsg::Ready) => {}
        Ok(_) => return Err(wire_error("expected Ready after Setup")),
        Err(BrutError::Budget(_)) => return Ok(aborted(Abort::OverCost, 0, 0)),
        Err(e) => return Err(e),
    }

    match task.f {
        8 => run_wire_width::<8>(learner, task),
        16 => run_wire_width::<16>(learner, task),
        _ => run_wire_width::<32>(learner, task),
    }
}

fn run_wire_width<const F: usize>(
    learner: &mut dyn WireLearner,
    task: &TaskInstance,
) -> Result<TaskResult> {
    let kind = task.kind;
    let k = kind.k_out().min(F);
    let checkpoints = checkpoint_steps(task.steps);
    let labels = label_values(&task.y_train);
    let late = task.y_train_late.as_ref().map(label_values);
    let drift_at = match (&late, task.drift_at) {
        (Some(_), Some(at)) => at,
        _ => usize::MAX,
    };

    let mut linker = Memory::<F>::new(0);
    let mut stream = Stream::new(task.seed, task.n_train);
    let mut outputs = vec![0.0f32; task.n_valid * k];
    let mut linked = [0.0f32; MAX_K];
    let eval_msg = HostMsg::Eval {
        xs: task.x_valid.chunks_exact(F).map(<[f32]>::to_vec).collect(),
    };

    let mut q = [0.0f32; 3];
    let mut evaluated = 0usize;
    let mut steps_done = 0usize;
    let mut partial_predicts = 0u64;
    let mut abort = None;

    while evaluated < 3 && checkpoints[evaluated] == 0 {
        q[evaluated] = wire_checkpoint(learner, &mut linker, task, &eval_msg, &mut outputs)?;
        evaluated += 1;
    }

    for s in 0..task.steps {
        let i = stream.index(s);
        let x = task.x_train[i * F..(i + 1) * F].to_vec();

        let raw = match learner.handle(&HostMsg::Predict { x: x.clone() })? {
            LearnerMsg::Prediction { y } if y.len() == k => y,
            _ => return Err(wire_error("expected a Prediction of k_out values")),
        };

        link(&mut linker, kind, &raw, &mut linked[..k]);

        let y = if s >= drift_at {
            late.as_ref().map_or(labels[i], |l| l[i])
        } else {
            labels[i]
        };

        if linked[..k].iter().any(|v| !v.is_finite()) {
            partial_predicts = 1;
            abort = Some(Abort::NonFinite);
            break;
        }

        if kind == LearnKind::Regression && (linked[0] - y).abs() > MAX_ABS_TRAIN_ERROR {
            partial_predicts = 1;
            abort = Some(Abort::ErrorTooLarge);
            break;
        }

        match learner.handle(&HostMsg::Learn { x, y })? {
            LearnerMsg::Learned => {}
            _ => return Err(wire_error("expected Learned")),
        }

        steps_done = s + 1;

        while evaluated < 3 && checkpoints[evaluated] == steps_done {
            q[evaluated] = wire_checkpoint(learner, &mut linker, task, &eval_msg, &mut outputs)?;
            evaluated += 1;
        }
    }

    match learner.handle(&HostMsg::End)? {
        LearnerMsg::Ready => {}
        _ => return Err(wire_error("expected Ready after End")),
    }

    let flops = learner
        .static_costs(F)
        .map_or(0, |(setup, predict, learn)| {
            static_flops(
                setup,
                predict,
                learn,
                steps_done,
                partial_predicts,
                evaluated,
                task.n_valid,
            )
        });

    Ok(match abort {
        Some(a) => aborted(a, flops, steps_done),
        None => TaskResult {
            q,
            aulc: aulc(q),
            abort: None,
            flops,
            steps_done: steps_done as u32,
        },
    })
}

/// Send the validation batch, link every row, score it.
fn wire_checkpoint<const F: usize>(
    learner: &mut dyn WireLearner,
    linker: &mut Memory<F>,
    task: &TaskInstance,
    eval_msg: &HostMsg,
    outputs: &mut [f32],
) -> Result<f32> {
    let k = task.kind.k_out().min(F);

    let ys = match learner.handle(eval_msg)? {
        LearnerMsg::Evaluated { ys } if ys.len() == task.n_valid => ys,
        _ => {
            return Err(wire_error(
                "expected Evaluated with one row per validation input",
            ));
        }
    };

    for (row, out) in ys.iter().zip(outputs.chunks_exact_mut(k.max(1))) {
        if row.len() != k {
            return Err(wire_error("an Evaluated row has the wrong length"));
        }

        link(linker, task.kind, row, out);
    }

    Ok(quality(task.kind, outputs, &task.y_valid))
}

/// Link raw outputs with the interpreter's own link on a scratch memory.
fn link<const F: usize>(linker: &mut Memory<F>, kind: LearnKind, raw: &[f32], out: &mut [f32]) {
    let k = raw.len().min(F).min(out.len());

    linker.v[1][..k].copy_from_slice(&raw[..k]);
    linker.apply_link(kind);
    out[..k].copy_from_slice(&linker.v[1][..k]);
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::learner::families::{public_reference_tasks, search_panel, select_tasks};
    use crate::learner::protocol::tests::over_cost;
    use crate::learner::protocol::{run_learner, run_learner_with_outputs};
    use crate::learner::seeds;
    use crate::learner::types::{EvalSettings, Instr, OpCode, PanelConfig, SelectConfig};

    const RELAXED: EvalSettings = EvalSettings {
        max_seconds_per_task: 1.0e9,
    };

    fn json_round_trip<T>(value: &T)
    where
        T: Serialize + for<'de> Deserialize<'de> + PartialEq + std::fmt::Debug,
    {
        let text = serde_json::to_string(value).expect("serialize");

        assert!(!text.contains('\n'), "one message per line");

        let back: T = serde_json::from_str(&text).expect("parse");

        assert_eq!(&back, value);
    }

    #[test]
    fn every_message_round_trips_as_one_json_line() {
        let host = [
            HostMsg::Setup {
                f: 16,
                kind: LearnKind::Multiclass { k: 4 },
                steps: 384,
            },
            HostMsg::Setup {
                f: 8,
                kind: LearnKind::Regression,
                steps: 1,
            },
            HostMsg::Predict {
                x: vec![0.5, -1.25, 3.0e-7],
            },
            HostMsg::Learn {
                x: vec![1.0, 2.0],
                y: 1.0,
            },
            HostMsg::Eval {
                xs: vec![vec![0.1, 0.2], vec![]],
            },
            HostMsg::End,
        ];
        let learner = [
            LearnerMsg::Ready,
            LearnerMsg::Prediction {
                y: vec![0.25, 0.75],
            },
            LearnerMsg::Learned,
            LearnerMsg::Evaluated {
                ys: vec![vec![1.0], vec![-2.5]],
            },
        ];

        for m in &host {
            json_round_trip(m);
        }

        for m in &learner {
            json_round_trip(m);
        }

        assert_eq!(
            serde_json::to_string(&HostMsg::End).expect("serialize"),
            "{\"op\":\"end\"}"
        );
        assert_eq!(
            serde_json::to_string(&host[1]).expect("serialize"),
            "{\"op\":\"setup\",\"f\":8,\"kind\":{\"kind\":\"regression\"},\"steps\":1}"
        );
    }

    #[test]
    fn malformed_messages_are_rejected() {
        let bad = [
            r#"{"op":"predict","x":[1.0],"extra":1}"#,
            r#"{"op":"train","x":[1.0]}"#,
            r#"{"x":[1.0]}"#,
            r#"{"op":"learn","x":[1.0]}"#,
        ];

        for text in bad {
            assert!(serde_json::from_str::<HostMsg>(text).is_err(), "{text}");
        }

        // Struct variants reject extra keys. Known serde limitation: with internal tagging,
        // unit variants (Ready, Learned, End) ignore extra keys despite deny_unknown_fields.
        let extra = r#"{"op":"prediction","y":[],"z":0}"#;
        let unit = r#"{"op":"learned","x":1}"#;

        assert!(serde_json::from_str::<LearnerMsg>(extra).is_err());
        assert_eq!(
            serde_json::from_str::<LearnerMsg>(unit).expect("unit variant"),
            LearnerMsg::Learned
        );
    }

    #[test]
    fn the_dsl_learner_refuses_out_of_order_and_bad_episodes() {
        let mut learner = DslWireLearner::new(LearnerGenome::default(), 1, 0);

        assert!(
            learner
                .handle(&HostMsg::Predict { x: vec![0.0; 8] })
                .is_err()
        );

        let wide = HostMsg::Setup {
            f: 12,
            kind: LearnKind::Binary,
            steps: 1,
        };
        let too_many = HostMsg::Setup {
            f: 8,
            kind: LearnKind::Multiclass { k: 9 },
            steps: 1,
        };

        assert!(learner.handle(&wide).is_err());
        assert!(learner.handle(&too_many).is_err());
        assert_eq!(
            learner.handle(&HostMsg::End).expect("end"),
            LearnerMsg::Ready
        );
    }

    /// A learner that answers Setup with a fixed reply.
    struct Scripted(Result<LearnerMsg>);

    impl WireLearner for Scripted {
        fn handle(&mut self, _msg: &HostMsg) -> Result<LearnerMsg> {
            match &self.0 {
                Ok(m) => Ok(m.clone()),
                Err(BrutError::Budget(m)) => Err(BrutError::Budget(m.clone())),
                Err(_) => Err(wire_error("scripted")),
            }
        }
    }

    #[test]
    fn run_wire_maps_setup_replies() {
        let task = &public_reference_tasks(1)[0];

        let refused = run_wire(&mut Scripted(Err(BrutError::Budget("cap".into()))), task)
            .expect("over-cost is a score");

        assert_eq!(refused.abort, Some(Abort::OverCost));
        assert_eq!((refused.flops, refused.aulc), (0, 0.0));

        assert!(run_wire(&mut Scripted(Ok(LearnerMsg::Learned)), task).is_err());
        assert!(run_wire(&mut Scripted(Err(wire_error("x"))), task).is_err());

        let mut broken = task.clone();

        broken.x_train.pop();

        assert!(run_wire(&mut Scripted(Ok(LearnerMsg::Ready)), &broken).is_err());
    }

    // ---- Tests below need the WP1 interpreter (skipped while it is a stub). ----

    fn ins(op: OpCode, a: u8, b: u8, out: u8, c: f32) -> Instr {
        Instr { op, a, b, out, c }
    }

    /// Programs that exercise random ops, links, aborts and drift.
    fn programs() -> Vec<LearnerGenome> {
        let mut list: Vec<LearnerGenome> =
            seeds::seed_programs().into_iter().map(|(_, g)| g).collect();
        let mut noisy = seeds::reference_nn();

        noisy.predict.push(ins(OpCode::VGaussian, 0, 0, 7, 0.01));
        noisy.learn.push(ins(OpCode::VAdd, 6, 7, 6, 0.0));
        list.push(noisy);

        // Writes v0 in Predict: Learn must see the overwritten input.
        list.push(LearnerGenome {
            setup: vec![ins(OpCode::SConst, 0, 0, 3, 0.05)],
            predict: vec![
                ins(OpCode::VAbs, 0, 0, 0, 0.0),
                ins(OpCode::MatVec, 0, 0, 1, 0.0),
            ],
            learn: vec![
                ins(OpCode::VSub, 2, 1, 3, 0.0),
                ins(OpCode::VScale, 3, 3, 3, 0.0),
                ins(OpCode::MOuter, 3, 0, 1, 0.0),
                ins(OpCode::MAdd, 0, 1, 0, 0.0),
            ],
        });

        // Diverges on regression (error too large) after a while.
        list.push(LearnerGenome {
            setup: vec![ins(OpCode::SConst, 0, 0, 4, 1.5)],
            predict: vec![ins(OpCode::VBroadcast, 5, 0, 1, 0.0)],
            learn: vec![
                ins(OpCode::SMul, 5, 4, 5, 0.0),
                ins(OpCode::SAdd, 5, 4, 5, 0.0),
            ],
        });

        list
    }

    /// T-WIRE-1: run_wire(DslWireLearner) equals run_learner bit-for-bit on 50 tasks.
    #[test]
    fn wire_oracle_matches_run_learner_bit_for_bit() {
        let mut tasks =
            search_panel(7, 0, &PanelConfig::default(), &[8, 16], &[16, 64, 256], 64).tasks;

        tasks.extend(
            select_tasks(7, 0, &SelectConfig::default())
                .into_iter()
                .take(30),
        );

        assert_eq!(tasks.len(), 50);

        let programs = programs();
        let mut aborted_runs = 0;

        for (n, task) in tasks.iter().enumerate() {
            for (p, g) in programs.iter().enumerate() {
                if (n + p) % 2 == 1 {
                    continue;
                }

                let direct = run_learner(g, task, 0, &RELAXED, &|| false);
                let mut learner = DslWireLearner::new(g.clone(), task.seed, 0);
                let wired = run_wire(&mut learner, task).expect("wire run");

                assert_eq!(wired, direct, "task {n} program {p}");
                assert_eq!(wired.q.map(f32::to_bits), direct.q.map(f32::to_bits));
                assert_eq!(wired.aulc.to_bits(), direct.aulc.to_bits());

                aborted_runs += usize::from(direct.abort.is_some());
            }
        }

        assert!(aborted_runs > 0, "the set should include aborted runs");
    }

    #[test]
    fn wire_attempts_and_raw_outputs() {
        let task = &select_tasks(3, 0, &SelectConfig::default())[41];
        let g = seeds::reference_nn();

        for attempt in 0..2 {
            let mut learner = DslWireLearner::new(g.clone(), task.seed, attempt);

            assert_eq!(
                run_wire(&mut learner, task).expect("wire"),
                run_learner(&g, task, attempt, &RELAXED, &|| false)
            );
        }

        // Over-cost programs score OverCost on both paths.
        let heavy = over_cost();
        let mut learner = DslWireLearner::new(heavy.clone(), task.seed, 0);
        let wired = run_wire(&mut learner, task).expect("wire");

        assert_eq!(wired.abort, Some(Abort::OverCost));
        assert_eq!(wired, run_learner(&heavy, task, 0, &RELAXED, &|| false));

        // In-process outputs are the linked final checkpoint, empty after an abort.
        let (r, linked) = run_learner_with_outputs(&g, task, 0, &RELAXED, &|| false);
        let expected = if r.abort.is_none() {
            task.n_valid * task.kind.k_out()
        } else {
            0
        };

        assert_eq!(linked.len(), expected);
    }
}
