//! Performance benchmarks. Ignored by default; run them in release mode:
//!
//! `cargo test --release --test bench -- --ignored --nocapture --test-threads 1`
//!
//! Each benchmark prints one line per measurement. They assert nothing about speed,
//! only that the work was done, so they never fail on a slow machine.

use std::hint::black_box;
use std::time::Instant;

use brutmaschine::config::{Config, Limits};
use brutmaschine::evolve::{self, CONFIG_FILE};
use brutmaschine::genome::{Activation, Genome, Optimizer};
use brutmaschine::learner::interp::{self, Memory};
use brutmaschine::learner::protocol::run_learner;
use brutmaschine::learner::seeds::reference_nn;
use brutmaschine::learner::{
    EvalSettings, FamilyId, Group, Instr, Labels, LearnKind, LearnerGenome, OpCode, STEPS_PER_TASK,
    TaskInstance,
};
use brutmaschine::nn::{Gradients, Mlp, TargetsRef, Workspace};
use brutmaschine::task::{Objective, Split, Targets, TaskKind, generate};
use brutmaschine::train;
use brutmaschine::util::normal_f32;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;

const ACTIVATIONS: [Activation; 4] = [
    Activation::Relu,
    Activation::Tanh,
    Activation::Gelu,
    Activation::Sine,
];

/// Best wall time in seconds over `reps` calls of `f`.
fn best_of(reps: usize, mut f: impl FnMut()) -> f64 {
    let mut best = f64::INFINITY;

    for _ in 0..reps {
        let started = Instant::now();

        f();
        best = best.min(started.elapsed().as_secs_f64());
    }

    best
}

#[test]
#[ignore]
fn bench_forward_backward() {
    let spec = TaskKind::TwoSpirals.spec();
    let data = generate(&spec, Split::Train, 1);
    let Targets::Labels(labels) = &data.targets else {
        panic!("spirals has labels");
    };
    let batch = 32;
    let iters = 4000;
    let objective = Objective::Classification { classes: 2 };

    for act in ACTIVATIONS {
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let model = Mlp::new(&[2, 64, 64, 2], act, 1.0, &mut rng);
        let mut ws = Workspace::new();
        let mut grads = Gradients::zeros_like(&model);
        let mut sink = 0.0f32;

        let seconds = best_of(3, || {
            for i in 0..iters {
                let start = (i * batch) % (data.n - batch);
                let x = &data.x[start * 2..(start + batch) * 2];
                let t = TargetsRef::Labels(&labels[start..start + batch]);

                sink += model.loss_and_grad_into(x, t, batch, objective, &mut ws, &mut grads);
            }
        });

        let rate = (iters * batch) as f64 / seconds / 1e6;

        assert!(sink.is_finite());
        println!("bench fwd+bwd 2-64-64-2 batch {batch} {act:?}: {rate:.3} M samples/s");
    }
}

fn spirals_genome(activation: Activation, optimizer: Optimizer) -> Genome {
    Genome {
        hidden: vec![64, 64],
        activation,
        optimizer,
        learning_rate: 0.01,
        l2: 0.0,
        batch_size: 16,
        epochs: 100,
        init_scale: 1.0,
        seed: 1,
    }
}

#[test]
#[ignore]
fn bench_train_spirals() {
    let spec = TaskKind::TwoSpirals.spec();
    let data = generate(&spec, Split::Train, 1);
    let limits = Limits::default();
    let cases = [
        (Activation::Gelu, Optimizer::Adam),
        (Activation::Tanh, Optimizer::Adam),
        (Activation::Relu, Optimizer::Adam),
        (Activation::Relu, Optimizer::Sgd),
        (Activation::Relu, Optimizer::Momentum),
    ];

    for (act, opt) in cases {
        let genome = spirals_genome(act, opt);
        let mut loss = 0.0f32;

        let seconds = best_of(3, || {
            let (_, stats) = train::train(&genome, &spec, &data, &limits).expect("valid genome");

            loss = stats.final_train_loss;
        });

        println!(
            "bench train 2-64-64-2 batch 16 x 100 epochs {act:?}+{opt:?}: {seconds:.3} s (final loss {loss:.8})"
        );
    }
}

#[test]
#[ignore]
fn bench_run_spirals_default() {
    let dir = std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("bench-run");

    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("bench dir");

    let config = Config::default();

    config.save(&dir.join(CONFIG_FILE)).expect("save config");

    let started = Instant::now();
    let summary = evolve::run(&dir, None).expect("run");
    let seconds = started.elapsed().as_secs_f64();

    println!(
        "bench run spirals default: {seconds:.2} s, {} candidates, flops {:.4e}, best {:?} fitness {:.6} valid {:.4}",
        summary.candidates,
        summary.flops_used as f64,
        summary.best_id,
        summary.best_fitness,
        summary.best_valid_primary
    );

    let _ = std::fs::remove_dir_all(&dir);
}

/// Training examples per bench task and validation examples (AC-2: 3 x 64 predictions).
const BENCH_N_TRAIN: usize = 64;
const BENCH_N_VALID: usize = 64;

/// Counted FLOPs each interpreter measurement aims for (about 0.2 s at 2.5 GFLOP/s).
const BENCH_INTERP_FLOPS: u64 = 500_000_000;

/// Counted FLOPs of the single-thread task measurement.
const BENCH_TASK_FLOPS: u64 = 2_000_000_000;

/// Threads of the parallel task measurement (AC-2 counts on 4 threads).
const BENCH_THREADS: usize = 4;

/// A binary task built by hand: Gaussian inputs, labels from a random linear rule.
fn bench_task(f: usize, seed: u64) -> TaskInstance {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let w: Vec<f32> = (0..f).map(|_| normal_f32(&mut rng)).collect();

    let mut draw = |n: usize| {
        let x: Vec<f32> = (0..n * f).map(|_| normal_f32(&mut rng)).collect();
        let y: Vec<u8> = x
            .chunks(f)
            .map(|row| {
                let dot: f32 = row.iter().zip(&w).map(|(a, b)| a * b).sum();

                u8::from(dot > 0.0)
            })
            .collect();

        (x, y)
    };

    let (x_train, y_train) = draw(BENCH_N_TRAIN);
    let (x_valid, y_valid) = draw(BENCH_N_VALID);

    TaskInstance {
        id: seed,
        family: FamilyId("bench/linear-binary".to_string()),
        group: Group::Linear,
        f,
        kind: LearnKind::Binary,
        x_train,
        y_train: Labels::Classes(y_train),
        y_train_late: None,
        drift_at: None,
        x_valid,
        y_valid: Labels::Classes(y_valid),
        n_train: BENCH_N_TRAIN,
        n_valid: BENCH_N_VALID,
        steps: STEPS_PER_TASK,
        seed,
        changed_mask: None,
    }
}

/// Interpreter alone: Setup once, then `steps` training steps (Predict, link, Learn).
/// Returns counted GFLOP/s on one thread.
fn interp_gflops<const F: usize>(g: &LearnerGenome, task: &TaskInstance) -> f64 {
    let kind = task.kind;
    let steps = (BENCH_INTERP_FLOPS / g.step_cost(F).max(1)).max(1_000) as usize;
    let Labels::Classes(labels) = &task.y_train else {
        panic!("bench task has classes");
    };

    let seconds = best_of(3, || {
        let mut mem = Memory::<F>::new(task.seed);

        interp::run(&g.setup, &mut mem);

        for s in 0..steps {
            let i = s % task.n_train;

            mem.set_input(&task.x_train[i * F..(i + 1) * F]);
            mem.clear_label();
            interp::run(&g.predict, &mut mem);
            mem.apply_link(kind);
            mem.set_label(kind, f32::from(labels[i]));
            interp::run(&g.learn, &mut mem);
        }

        black_box(mem.output(1));
    });

    let flops = g.setup_cost(F) + steps as u64 * g.step_cost(F);

    flops as f64 / seconds / 1e9
}

/// Full protocol tasks per second, on one thread and on `BENCH_THREADS` threads.
fn tasks_per_second(g: &LearnerGenome, tasks: &[TaskInstance]) -> (f64, f64) {
    let settings = EvalSettings {
        max_seconds_per_task: 30.0,
    };
    let never = || false;
    let cost = g
        .task_cost(tasks[0].f, STEPS_PER_TASK, BENCH_N_VALID)
        .max(1);
    let n = (BENCH_TASK_FLOPS / cost).max(50) as usize;

    let one = best_of(3, || {
        for i in 0..n {
            black_box(run_learner(
                g,
                &tasks[i % tasks.len()],
                0,
                &settings,
                &never,
            ));
        }
    });

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(BENCH_THREADS)
        .build()
        .expect("thread pool");
    let n_par = n * BENCH_THREADS;

    let par = best_of(3, || {
        pool.install(|| {
            (0..n_par).into_par_iter().for_each(|i| {
                black_box(run_learner(
                    g,
                    &tasks[i % tasks.len()],
                    0,
                    &settings,
                    &never,
                ));
            });
        });
    });

    (n as f64 / one, n_par as f64 / par)
}

/// One line per width: interpreter GFLOP/s and full tasks/s for the nn2-sgd seed.
fn bench_learner_width<const F: usize>(g: &LearnerGenome) {
    let tasks: Vec<TaskInstance> = (0..8).map(|i| bench_task(F, 100 + i)).collect();
    let gflops = interp_gflops::<F>(g, &tasks[0]);
    let (one, par) = tasks_per_second(g, &tasks);

    println!(
        "bench learner nn2-sgd F={F}: interpreter {gflops:.2} GFLOP/s counted (1 thread); \
         full tasks ({STEPS_PER_TASK} steps + 3 x {BENCH_N_VALID} predictions): \
         {one:.0}/s (1 thread), {par:.0}/s ({BENCH_THREADS} threads)"
    );
}

/// T-BENCH-1. Targets of AC-2 (release build, F = 16): >= 2.5 GFLOP/s counted per thread
/// and >= 3 000 full tasks per second on 4 threads. Reported, not asserted.
#[test]
#[ignore]
fn bench_learner_interpreter() {
    let g = reference_nn();

    bench_learner_width::<8>(&g);
    bench_learner_width::<16>(&g);
    bench_learner_width::<32>(&g);
}

/// One instruction (bench programs).
fn op(op: OpCode, a: u8, b: u8, out: u8, c: f32) -> Instr {
    Instr { op, a, b, out, c }
}

/// Counted GFLOP/s of full protocol tasks of `g` at width F (one thread).
fn task_gflops<const F: usize>(g: &LearnerGenome) -> f64 {
    let tasks: Vec<TaskInstance> = (0..4).map(|i| bench_task(F, 200 + i)).collect();
    let settings = EvalSettings {
        max_seconds_per_task: 30.0,
    };
    let never = || false;
    let mut flops = 0u64;

    let seconds = best_of(3, || {
        flops = 0;

        for t in &tasks {
            let r = run_learner(g, t, 0, &settings, &never);

            flops += r.flops;
            black_box(r);
        }
    });

    flops as f64 / seconds / 1e9
}

/// SEC-1: programs that pass the static cost cap but feed subnormal or tiny values into
/// products, next to the same structure on normal values. Before the flush fix the
/// subnormal variants ran about 100 times slower per counted FLOP.
#[test]
#[ignore]
fn bench_learner_adversarial() {
    let matvecs = |out: u8, v: u8| -> Vec<Instr> {
        (0..20)
            .map(|_| op(OpCode::MatVec, 0, v, out, 0.0))
            .collect()
    };
    let fill = |c: f32| LearnerGenome {
        setup: vec![op(OpCode::MUniform, 0, 0, 0, c)],
        predict: matvecs(1, 0),
        learn: Vec::new(),
    };
    let tiny_products = LearnerGenome {
        setup: vec![
            op(OpCode::MUniform, 0, 0, 0, 1e-20),
            op(OpCode::VUniform, 0, 0, 3, 1e-20),
        ],
        predict: matvecs(4, 3),
        learn: Vec::new(),
    };
    let gaussian_learn = LearnerGenome {
        setup: Vec::new(),
        predict: vec![op(OpCode::MatVec, 0, 0, 1, 0.0)],
        learn: (0..3)
            .map(|k| op(OpCode::MGaussian, 0, 0, 1 + k, 0.1))
            .collect(),
    };
    let small = |c: f32| LearnerGenome {
        setup: vec![
            op(OpCode::MUniform, 0, 0, 0, c),
            op(OpCode::VUniform, 0, 0, 3, c),
        ],
        predict: matvecs(4, 3),
        learn: Vec::new(),
    };
    let programs = [
        ("normal fill (c = 1)", fill(1.0)),
        ("small products (c = 1e-18)", small(1e-18)),
        ("small products (c = 2e-15)", small(2e-15)),
        ("small products (c = 1e-7)", small(1e-7)),
        ("subnormal fill (c = 1e-39)", fill(1e-39)),
        ("tiny products (c = 1e-20)", tiny_products),
        ("3 x MGaussian in Learn", gaussian_learn),
        ("nn2-sgd", reference_nn()),
    ];
    let normal = task_gflops::<32>(&programs[0].1);

    for (name, g) in &programs {
        assert!(g.check_cost().is_ok(), "{name} passes the cost cap");

        let rate = task_gflops::<32>(g);

        println!(
            "bench adversarial F=32 {name:<28} {rate:>6.3} GFLOP/s counted ({:>6.1} x slower than the normal fill)",
            normal / rate
        );
    }
}
