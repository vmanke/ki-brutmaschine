# Stage 1 contract errata (contract-v1)

Written by WP0 (skeleton) on 2026-10-02. Each entry names the contract places that disagree
or leave a gap, and the decision WP0 took. The decisions are minimal. The code skeleton
already follows them where they touch signatures. Entries marked **lead: confirm** change a
measurement rule. The lead should confirm those before the protocol hash is frozen.

## Decisions in force (lead defaults)

- D1. Budget plan S: `MetaConfig::default()` keeps `limits.max_total_flops = 1.5e12` per run.
- D2. Experiment E2 is off. `arc_cells` stays in `MetaConfig` (types are frozen), default
  `false`. `MetaConfig::validate` may refuse `arc_cells = true` until E2 is enabled.
  `tools/rearc_to_jsonl.py` is optional and need not ship. `load_rearc_jsonl` keeps its
  signature.
- D3. The global secret directory `~/.brut/learner` is accepted (contract default).
- D4. The only primary endpoint is HIDDEN-GRAMMAR (section 6.8). The other tiers are
  descriptive.

## Errata

### E1. WP0 adds stubs to files owned by WP4 and WP5

Section 2 gives WP0 only `lib.rs`, `learner/mod.rs`, `learner/types.rs` and the stub files.
But sections 5.16, 5.21 and 5.19 change the signatures of existing files that other packages
own. Without those signatures the stubs would not compile against each other.

Decision: WP0 added these additive API stubs. Each one is marked
`// WP0 stub: remove when implemented`:

- `src/archive.rs` (WP5): the new `RecordKind` variants, `LearnerStatus`, `LearnerEntry`,
  `BatchDigestEntry`, and the learner query methods with `todo!()` bodies.
- `src/hidden_eval.rs` (WP4): `AnyEvalRequest` with `from_json` / `to_json`, and
  `request_eval_json`, all with `todo!()` bodies. `check_private` and `create_private_dir`
  are now `pub(crate)`. `request_hidden_eval` is unchanged and does not delegate yet.
- `src/config.rs` (WP5): `diff_fields` is now `pub(crate)`.

The owners replace the bodies. MLP mode never calls any of the stubs.

### E2. The program JSON example in section 7 is not the canonical form

Section 7 shows instructions without zero fields (`{"op":"SConst","out":3,"c":0.05}`).
Section 4 defines `Instr` with `#[serde(default)]` but no `skip_serializing_if`. So
`canonical_json()` (section 5.1) always writes all five fields.

Decision: types stay as in section 4. The compact form is a valid *input* (missing fields
default to 0). The canonical *output*, and therefore `id()`, always has
`op, a, b, out, c`. A unit test in `types.rs` pins both facts.

### E3. Output width can exceed the vector width (k > F)

`LearnKind::Multiclass` allows k up to `MAX_K = 10`. `Memory::output`, `set_label` and
`apply_link` (section 5.2) work on `v1` / `v2`, which are only F wide. F = 8 is supported.

Decision: every `TaskInstance` MUST satisfy `kind.k_out() <= f`. All constructors in the
contract already do: the hand families and grammar families have k <= 5, and ARC cells
have k = 10 at F = 32. The interpreter helpers use `min(k, F)` and never index past F.
`hand_instance`, `grammar_instance` and `cell_tasks` never produce k > f.

### E4. `RecordKind::MetaRunStarted` trips clippy `large_enum_variant`

With `config: MetaConfig`, the variant is 536 bytes. The next largest is 296 bytes. Clippy
then fails with `-D warnings`, which AC-1 requires to pass.

Decision: the field is `config: Box<MetaConfig>`. Serde writes `Box<T>` exactly like `T`,
so the archive JSON is unchanged. `Archive::meta_config() -> Option<&MetaConfig>` keeps its
signature. A `#[allow]` was not used because it is permitted only in stub files.

### E5. `ScoreCache::len` without `is_empty` trips clippy `len_without_is_empty`

Decision: `ScoreCache` also gets `pub fn is_empty(&self) -> bool`. This is the only
addition to the API in section 5.5.

### E6. One-time costs exceed the tiny test budgets (confirmed by the lead, L1; metered since V11)

Section 6.6 charges the tuned baselines (about 3e10 FLOPs) to the Guard at run start.
Section 6.7.3 charges the final phase (about 4e10 FLOPs). T-E2E-1 runs with
`max_total_flops = 1e9` and still expects finalists, a hidden phase and a report. Under the
literal rule that run is over budget before its first batch. Plan section 8 also counts
these costs apart from the search budget ("verdeckt, ARC, Endphasen ≈ 4·10¹¹").

Decision: `limits.max_total_flops` bounds the **search phase**. That phase covers the
initial population, the batches, the rotation re-scores and the periodic Stage C rounds.
The one-time `tuned_baselines()` call and the final phase (section 6.7) are metered in
their own counters. They are written to the archive and the report, and do not count
against `max_total_flops`. STOP still aborts both. `max_total_seconds` still covers the
baseline tuning. The final phase still runs after a time stop, as section 6.7.3 says.

### E7. Cycling F and n_train together confounds the F = 32 coupling check (confirmed by the lead, L2)

The hidden tiers (table in section 5.15), `select_tasks` and the baseline tuning set say
"F 8, 16, 32 cycling" and "n_train cycling (16, 64, 256)". If both cycle on the same index,
every F = 32 task has n_train = 256, and every F = 8 task has n_train = 16. Then the primary
decision's coupling clause `Δ32 >= Δ816 - 0.05` (section 6.8) and `delta_select_f32` would
measure data size, not width. These sets also leave `level` unspecified.

Decision, one rule for every set built by cycling (all hidden tiers,
`public_reference_tasks`, `select_tasks` and the baseline tuning set):

- k is the running index of the task within the set, in output order. Families come in
  their listed order, with instances i = 0.. inside each family. In `select_tasks` the hand
  tasks come first and the grammar tasks continue the count.
- `f = dims[k % dims.len()]`. Here dims is `[8, 16]` for the sets that "alternate" and
  `[8, 16, 32]` for the sets that "cycle".
- `n_train = [16, 64, 256][(k / dims.len()) % 3]`. With this rule all (f, n_train) pairs
  appear about equally often.
- `level` comes from `sample_params(&mut ChaCha8Rng::seed_from_u64(instance_seed), &[f],
  &[n_train], n_valid).level`. Its distribution then matches the search panel. Grammar
  instances ignore `level`.

`public_reference_tasks` and HiddenInstance use the same rule. Δ_public and Δ_HI then stay
like-for-like (quarantine).

### E8. `tuned_baselines` does not say which `SelectConfig` it uses

Section 5.9 says the tasks are "built like select_tasks". `select_tasks` takes a
`SelectConfig`. `brut-eval` has no run config, but it MUST compute the same hash.

Decision: `tuned_baselines` always uses `SelectConfig::default()` (8 per SELECT hand
family, 32 grammar families, 72 tasks). It keeps the even indices, which gives 20 hand and
16 grammar tasks. The run's `MetaConfig.select` does not affect it.

### E9. ARC cell tasks have no seed tag

`cell_tasks(t)` takes no seed, and section 6.1 has no ARC tag. Invariant I4 still requires
every seed to come from `derive_seed`.

Decision: for test pair j of task `<id>`, the seed is
`derive_seed(0, "arc-cell/<id>/<j>")`. The family is `FamilyId("arc-cell/<id>")` (E2 uses
`arc-cell/rearc/<id>`). `level` is 0 in the task-id string. Attempts 0 and 1 share the task
seed and differ only in the interpreter tag `interp/<attempt>`.

### E10. How many mutations a random-search child gets

The `VariationMode::RandomSearch` doc (section 4) says each child is "a random seed with
1..=8 random Full mutations". The pseudo-code in section 6.5 first builds such a parent and
then applies `proposer.propose` to it, which would give 2..=9 mutations.

Decision: a RandomSearch child is `seed[r % 4]` with exactly n mutations, n uniform in
1..=8. Both r and n come from the slot rng. `propose` is not applied on top. The generator
string is `"random-search"`.

### E11. STOP and time checks on tasks shorter than `CHECK_EVERY_STEPS` (amended by V5)

`CHECK_EVERY_STEPS = 1024` is larger than `STEPS_PER_TASK = 384`. If the checks only ran at
steps 1024, 2048 and so on, a synthetic task would never poll `stop()`, and T-PROT-3 could
not trigger Stopped or Timeout on one.

Decision: the protocol checks at every step s with `s % CHECK_EVERY_STEPS == 0`, including
s = 0. Every task therefore checks at least once, before its first step.

### E12. Grammar positions in the panel hard-code 10 and 64

Section 6.4 uses `pool[(p - 10 + 10·g) mod 64]`. But `PanelConfig.grammar_tasks` is
configurable, and `validate` only pins `hand_tasks == 10`.

Decision: the general form is `pool[(p - hand_tasks + grammar_tasks·g) mod pool.len()]`.
It gives the same panels for the defaults. `MetaConfig::validate` also requires
`1 <= grammar_tasks <= 64` and every `stage_a_positions` entry `< hand_tasks + grammar_tasks`.

### E13. How the placeholders in seed tags are spelled

Section 6.1 uses placeholders such as `<tier>` and `<family>` without saying how to write
them.

Decision:

- `<tier>` is the serde snake_case name, e.g. `tier/hidden_grammar/...` and
  `test/<id>/hidden_grammar` (this matches the literal text in section 6.8).
- `<family>` is `FamilyId.0` verbatim and may contain `/`.
- Integers are plain decimal without padding, so `u64::MAX` in the retune set is written
  `18446744073709551615`.
- `<id>` is the 16-hex candidate id.

### E14. Invariant I5 versus the `todo!()` bodies WP0 is told to write

I5 forbids `todo!()` in merged code. The contract preamble tells WP0 to write `todo!()`
bodies, and the skeleton is committed to master.

Decision: the bodies are allowed only until integration. Every stub file or stub block
carries `// WP0 stub: remove when implemented`. The `#![allow(dead_code, unused_variables)]`
lines appear only in the stub files. Before acceptance, the lead checks that
`grep -rn "todo!()\|WP0 stub" src` returns nothing.

## WP0 choices that are not errata

- Private fields in `ScoreCache`, `Hurdle`, `DslWireLearner`, `Ledger`, `LearnerEvaluator`
  and `LineageWriter` are placeholders. The owners may change them freely.
- The private `hidden_grammar_pool` returns `Result<Vec<GrammarFamily>>`. The contract
  leaves the type open.
- Trailing `// default N` comments on constants became doc comments.

## Risks (no contract conflict, but they affect acceptance)

- R1. **Test runtime in debug builds.** `cargo test` builds without optimisation. Several
  required tests do release-scale work:
  - `tuned_baselines`, about 3e10 FLOPs. The orchestrator and `brut-eval` each compute it
    once per state directory, in every learner e2e run.
  - T-BAS-1, which calls `tuned_baselines` at least twice.
  - T-STAT-1, 2 000 datasets × `reps` flips. It should use a small `reps` such as 2 000; the
    calibration claim does not depend on `reps`.
  - The final-phase retune.

  WP1 should measure debug against release interpreter throughput as soon as `interp.rs`
  works. If debug is more than about 10x slower, the lead should set
  `[profile.dev] opt-level = 2` in `Cargo.toml`. That does not touch `[dependencies]`
  (I1), and Rust does not change float results with the optimisation level.
- R2. I1 to I8 and S1/S2 are grep-checked today only by hand (all clean at this commit).
  The tests T-SEC-1/2/3 that enforce them are WP6's.

## Lead decisions at integration (2026-10-02)

The work packages WP1 to WP6 were merged on master. Their reports listed the deviations
below. The lead accepts each one as stated unless the entry says otherwise. All of them
are in force for the protocol hash `d430e049…` (`learner-v1`).

### Confirmed errata

- L1. **E6 confirmed.** `max_total_flops` bounds the search phase only. Baseline tuning and
  the final phase are metered apart. The guard is checked before each batch, so a periodic
  Stage C round in the last batch can overshoot the cap by one round. The 1e10 smoke run
  stopped at 1.48e10. At 1.5e12 one round is well under 1 %.
- L2. **E7 confirmed** (one cycling rule for all cycled task sets).
- L3. **Grammar pools (WP2 D1) confirmed.** The length histogram of section 5.8 cannot be
  met: only 4 distinct structures exist at length 2 and 13 at length 3. Signatures are
  canonical (operand order of add/mul/max and the naming of t0..t2 do not count). Collapsing
  expressions are never sampled. Signatures are distinct inside each pool. Pool lengths
  follow 2^-L, capped per length at a quarter of the signatures that exist. Hidden pool of
  40 for L = 2..9: [1, 2, 9, 14, 7, 4, 2, 1].
- L4. **Non-finite validation outputs (WP1 issue C; replaced by V2 in `learner-v2`).**
  `quality()` maps a non-finite q to 0 for that checkpoint. The task is not aborted. A non-finite training-step output still
  aborts with NonFinite (section 6.2.2). I8 holds.
- L5. **Primary decision details (WP4 1).** Δ32 and Δ816 are measured against the overall
  b* of the tier, not re-maximised on the subset; families weigh equally. The rule
  Δ ≥ 0.01 uses the unrounded Δ. A missing width class or a non-finite p-value gives Fail.
- L6. **Program limits in brut-eval (WP4 2).** brut-eval checks programs against
  `DslLimits::default()`. A run with larger `MetaConfig.dsl` limits would have its finalists
  refused (logged, not charged). All Stage 1 runs use the defaults.
- L7. **`--budget` also lowers the ARC per-set cap** (WP4 3, reading of S6).
- L8. **Seed programs stay as in section 5.3.** The perceptron seed never updates on Binary
  tasks from its zero start (|y − σ(0)| = 0.5 is not above its 0.5 margin). Untuned nn2-sgd
  can diverge at F = 32. Both are part of the protocol and are not changed. Tuning moves the
  perceptron margin (tuned baseline 0.072), and the tuned NN baseline is stable. The panel
  reference ref_t uses F ∈ {8, 16} only.
- L9. **Extra seed tags** (WP2 D8): `gen/params`, `gen/train`, `gen/valid`, `gen/reference`,
  `mlp-matched/init`, `fec/over-cost`; ARC cells as in E9.
- L10. **WP3 manifest rules stand.** Commits are hex, licences short names. The test
  fixture manifest uses commit `0000000` and `CC0-1.0`.
- L11. **ARC overlap.** At the pinned commits, A1 eval 070dd51e and 50a16a69 repeat pairs of
  A1 training tasks. ARC-AGI-1 training can therefore never be declared as a training
  source for A1 eval. Stage 1 trains on synthetic data only, so nothing is affected.
  A2 eval db695cfb is a rotated, recoloured copy of A1 eval 55783887. The source rules
  already block that pairing.
- L12. **Integration changes by the lead.** `[profile.dev] opt-level = 2` (R1). The
  orchestrator calls `tuned_baselines_with_stop` instead of polling a detached thread
  (WP5 deviation 8). The report's wall time ends at `RunFinished`, so a later ARC check
  does not add to it. The WP2 test probe `wp1_ready` is removed.

### Known gaps (not closed in Stage 1)

- G1. The protocol hash covers the grammar pool definitions, not the hand-family generator
  code. A code change to a hand family does not change the hash.
- G2. The 48-query total is counted per protocol hash. A changed protocol gets 48 fresh
  queries against the same secret, which builds the same hidden families. Rotate the
  secret (`brut learner secret-init` into a new directory) whenever the protocol changes.
- G3. Deleting both `ledger.jsonl` and `ledger.head` resets the budget (same Unix user).
- G4. AC-11 fails. A measured 1e13-FLOP run (seed 11, no hidden phase) wrote 1 593 MB
  (lineage 1 184 MB, archive 311 MB, of which 310 MB are 331 021 hall-of-fame
  `LearnerCandidate` records with full genomes; snapshots 99 MB) and peaked at 2 374 MB RSS.
  `Archive::open` alone holds about 590 MB for that archive. The record formats of sections
  5.18 and 5.21 drive this; a fix needs a contract revision (for example hall-of-fame
  records only at Stage C or snapshot time, packed lineage chunks, a streaming archive).
  Plan S runs (1.5e12) stay at 69 to 181 MB and at most 380 MB RSS.

## Pre-registration of the plan S evaluation (fixed before any query with the real secret)

The plan says "the best candidate passes in at least 2 of 3 seeds" but does not say which
finalist is the best one. These rules were written down before the first hidden query
against the global secret `~/.brut/learner/hidden.env`:

- P1. **Best candidate of a run:** the finalist with the highest Δ_select of the final
  Stage C round after retuning (`report.json`, `finalists[].delta_select`). Ties go to the
  earlier finalist. Only public numbers decide this.
- P2. **Primary endpoint:** the HIDDEN-GRAMMAR decision bit of the best candidate (section
  6.8: p < 0.05, Δ ≥ 0.01, Δ32 ≥ Δ816 − 0.05). Stage 1 succeeds iff it is Pass in at least
  2 of the 3 E1 seeds (master seeds 1, 2, 3, variation Full, 1.5e12 search FLOPs each).
  The other finalists' results are descriptive only.
- P3. **Ablation condition:** the mean released Δ_GH of the best candidates over the three
  E1 seeds must be strictly greater than the same mean for the three ConstantsOnly runs
  (same seeds and budget). A tie counts as "not beaten".
- P4. **Stop rule (plan section 9):** if Δ_select of the best candidate is ≤ 0 in all three
  E1 seeds and P3 fails, the program evolution is not extended further.
- P5. **ARC:** the best candidate of each E1 seed is checked once on a1-eval, a2-eval and
  concept-arc (3 queries per set, the per-set cap), with training source synthetic only.
- P6. The E0 smoke run uses its own secret directory. Its hidden results only test the
  machinery and are not part of the evaluation.

## Review fixes (`learner-v2`, 2026-10-02)

An independent review of the integrated Stage 1 found the issues below. The fixes change
measurements, so `PROTOCOL_VERSION` is now `learner-v2` (protocol hash `c4b67a43…`). Per
G2 the re-run uses a new secret directory (the E0 smoke run again uses its own). A first
re-run attempt with `~/.brut/learner-v2` was cut off by a container restart after one E1
run had queried the evaluator, before COR-2, COR-4 and SCI-R4 were finished; that
secret counts as used and its results were discarded. The reported re-run uses a third,
fresh secret directory (`/tmp/claude-0/stage1-runs-v2/secret`), created with
`brut learner secret-init` after all code fixes. The plan S re-run follows P1 to P6 unchanged; the v1 results stay in the
history of `docs/ERGEBNISSE_STUFE1.md`.

### Measurement changes

- V1. **MlpMatched budget = training cost (COR-1).** Section 5.9 asks for "the
  candidate's training cost up to step_c". The v1 code prorated the whole task cost,
  which also counts the 3 · n_valid validation predictions: about 1.95× the contracted
  budget at n_valid = 256 and 1.24× on SELECT. Now the budget at checkpoint c is exactly
  `setup_cost + step_c · step_cost` (`baselines::TrainingCost`), in Stage C, the final
  phase and `brut-eval`. The recipe tuning keeps its reference-NN task cost (5.9). The
  one-epoch floor stays, so at small budgets the MLP still spends more than the
  candidate (see SCI-R2 in the results document).
- V2. **Non-finite validation outputs score q = 0 (COR-2; replaces L4).** A NaN output
  used to count as a class-0 prediction (6.3 read literally: `NaN >= 0.5` is false, and
  argmax skips NaN), so NaN rows earned accuracy on class-0 labels. `quality()` now
  returns 0 for a checkpoint with any non-finite output, for every kind.
- V3. **min / max are symmetric (COR-4).** SMin, SMax, VMin and VMax follow IEEE 754-2019
  minimumNumber / maximumNumber: a NaN operand yields the other operand, and −0 < +0.
  In v1 the result depended on operand order when a NaN was involved.
- V4. **Tiny values are flushed to +0 (SEC-1).** The op table says "IEEE; no checks". On
  x86 every product with a subnormal result takes a microcode assist, so a valid,
  capped program (e.g. `MUniform c = 1e-39` and 20 × MatVec) ran about 100× slower per
  counted FLOP, and the static FLOP count no longer bounded wall time. FTZ/DAZ cannot be
  set under `#![forbid(unsafe_code)]`. Now scalars and vectors keep magnitudes ≥ 2^-34,
  matrices ≥ 2^-68; every op stores smaller results as +0, and so do inputs, labels,
  constants and random fills. Divisions are computed in f64 and rounded (bit-identical
  to f32 division outside the flushed range); `exp` below −87 is 0. With these floors no
  op forms a subnormal value (proofs in the module doc of `interp.rs`). The adversarial
  bench now runs at the normal rate. Not changed: the op costs of VGaussian / MGaussian
  (about 10× under-costed against MatVec), SSin / SCos on large arguments (about 9×) and
  SExp / SLog (about 5×). Known gap K1 below.
- V5. **Wall-clock and STOP polling inside a task (SEC-2; amends E11).**
  `CHECK_EVERY_STEPS` is 64 (was 1024, more than a whole synthetic task). The limit and
  STOP are checked at step 0 and every 64 steps, and before every 64th validation row of
  each checkpoint. An abort during validation charges the predictions done. Results
  change only when a limit fires (I4 exempts that).
- V6. **Hidden grammar pool without exact twins (SCI-R4).** Distinct signatures do not
  guarantee distinct task distributions. Two rewrites leave the distribution unchanged:
  (a) a sum of single-use projections under only mul / abs / square is one projection;
  (b) for classification, square at the root equals abs. The hidden pool now also
  rejects a family whose semantic key (`grammar::semantic_key`) equals that of a public
  family or of an earlier hidden family; the draw is repeated. The public pools are
  unchanged. The key catches these two rewrites only, so it is a lower bound on twins.

### Bookkeeping and engineering changes (results bit-identical)

- V7. **Grouped archive writes (PERF-1).** Two fsyncs and a rename per record cost about
  13 to 23 % of the search wall time. During the learner search `Archive` now buffers
  records and writes them as one group with one fsync: at every Stage C round, at every
  `BatchDigest`, after 4 096 records, when the search ends (every path) and on drop.
  `archive.head` gets an optional field `pending`: before a group write the anchor
  announces the group, and `check_head` accepts an anchor at most that many records
  behind. A crash loses at most the records written since the last group, all of them
  after the last digest; a resume from that digest recreates them. Record contents and
  order are unchanged, so the hash chain is the same (timestamps aside). MLP mode and
  every learner record after the search (finalists, hidden results, ARC checks) are
  written one by one as before.
- V8. **MlpMatched fits run once per distinct (task, cost) (PERF-2).** A fit depends only
  on the task, the recipe and the candidate's training cost. Hall-of-fame members often
  share a cost, so 39 to 97 % of the v1 MLP work repeated identical fits. Stage C and the
  final phase now run and charge each distinct fit once. SelectEval values are unchanged;
  the charged Stage C FLOPs drop, so a run gets more children for the same budget.
  Stage C still counts against `max_total_flops` (6.6 and E6 unchanged). Its FLOPs are
  now also counted on their own (`ChunkStats.stage_c_flops`, `search.stage_c_flops` in
  the report). See known gap K2 for the ablation.
- V9. **Training-stream order computed once per task (PERF-3).** `protocol::prepare`
  computes the example order and the attempt-0 interpreter seed of a task once
  (`TaskPrep`); the search, Stage C and the tuning reuse it. Bit-identical to `Stream`.
- V10. **Report after `RunFinished`; `arc-check` refreshes it (COR-6).** 6.7 step 7 writes
  the report and then appends `RunFinished`. The report's wall time ends at
  `RunFinished` (L12), so the stored report showed the time before writing it (about 5 s
  less than a regenerated one). Now `RunFinished` comes first and the report is written
  after it; both reports agree. `brut learner arc-check` rewrites `report.json` after it
  appends `ExternalEval`, so the report lists the check (a failure to rewrite is a
  warning; the check result stands).
- V11. **One-time costs are metered (SCI-R5).** E6 said the baseline tuning and the final
  phase are "written to the archive and the report"; v1 metered neither in the report
  and the tuning nowhere. A new archive record `LearnerCost { phase, flops }`
  (`baseline_tuning`, `final_phase`) stores the static FLOPs of each, and `report.json`
  sums them in `one_time`. The final phase's `SafetyEvent` text stays.
- V12. **ARC load errors stay on the evaluator side (SEC-3).** `arc_data` errors quote
  manifest paths and task ids. `brut-eval` used to forward them, and `arc-check` wrote
  them into the orchestrator's archive. Now the caller gets the fixed text
  `REFUSED_ARC_DATA`; the detailed message goes only into the ledger's `Refused` reason
  in the private secret directory. The query is not charged (as before).
- V13. The report also names b* per finalist on SELECT and on the public reference tasks
  (`best_baseline_select`, `best_baseline_public`). The hidden b* stays inside
  `brut-eval`.

### Known gaps after the review

- K1. **FLOP counts still do not bound wall time exactly.** After V4 the worst measured
  cases are the Gaussian fills (about 10× slower per counted FLOP than MatVec), SSin /
  SCos on huge arguments (about 9×) and SExp / SLog (about 5×). Not done: re-costing
  these ops and metering measured time in the Guard. Wall time stays bounded by
  `max_total_seconds`, the per-task limit (now polled every 64 steps), the `brut-eval`
  kill and STOP. **Addressed in `learner-v3` (V14 below): the six ops are re-costed.
  Metering measured time in the Guard stays open.**
- K2. **Stage C shares the search budget.** Stage C never feeds back into the search, but
  it is charged to `max_total_flops`. In v1 it took 44 to 49 % of the budget in the E1
  runs and 12 to 25 % in the ConstantsOnly runs (plan section 8 assumed about 9 %), so
  ConstantsOnly had about 1.4 to 1.6× more search compute outside Stage C than Full.
  V1 and V8 shrink Stage C; the remaining share is reported per run, next to the
  ablation result. Taking Stage C out of the budget would change the pre-registered
  design and is left to the lead.
- K3. **The search can lose every good program.** The hall of fame is re-scored on each
  panel rotation and keeps only the currently best 32, so it is no permanent memory. In
  the `learner-v2` re-run, E1 seed 3 lost all linear learners by about batch 30 000 of
  48 000; its finalists output random numbers (Δ_GH −0.06). The plain linear-sgd seed
  still scores 0.19 to 0.27 panel mean on those late panels, so this is search dynamics,
  not a measurement fault. An all-time best list is a design change for the lead.
  **Addressed in `learner-v3` (V15 below): a permanent, bounded all-time-best memory.**

## Protocol learner-v3 (K1 re-cost and K3 all-time-best memory, 2026-10-21)

K1 and K3 above were left for the lead. They change the op costs and the search dynamics,
so both go in under one protocol bump: `PROTOCOL_VERSION` is now `learner-v3`. The protocol
hash moves for two reasons — the DSL op-cost table (`dsl_ops_hash`) and the version string.
The K3 search config does not feed the hash; the version bump records it. Per known gap G2
a changed protocol gets a fresh secret directory, so **the hidden re-measurement is deferred
to the Stage-2 control runs (fresh secret, new protocol) and was NOT run now**. No hidden
query, learner secret or `brut-eval` call was made for this change.

### Measurement change

- V14. **DSL ops re-costed so counted cost tracks real time (K1).** The adversarial-FLOP
  bench (V4, K1) measured these ops slower per counted FLOP than MatVec: VGaussian and
  MGaussian about 10×, SSin / SCos about 9×, SExp / SLog about 5×. `dsl::cost` now raises
  exactly these, by exactly those factors, so the counted-cost-to-real-time ratio drops to
  about 1.0×:

  | op | old cost | new cost | factor |
  | --- | --- | --- | --- |
  | SSin, SCos | 10 | 90 | ×9 |
  | SExp, SLog | 10 | 50 | ×5 |
  | VGaussian | 10 f | 100 f | ×10 (per element) |
  | MGaussian | 10 f² | 100 f² | ×10 |

  MatVec stays f² · 2 and is the 1.0× reference. SUniform, VUniform and MUniform are left
  as they were; the bench did not flag them. The static FLOP count is a type-time constant,
  so the flush fix (V4) and these costs are the only wall-time bounds inside a task besides
  `max_total_seconds` and the per-task limit; metering measured time in the Guard stays open
  (K1). The re-cost is confirmed by the `#[ignore]` cost table and adversarial benches, not
  by any hidden query. The step cost of the `nn2-sgd` reference (predict + learn, no
  re-costed op) is unchanged, so `reference_step_cost` and the cost cap are unchanged; its
  setup cost (two MGaussian) rises 10×, which is a one-time setup cost, not a per-step cost.
  This shifts which programs fit `COST_CAP_RATIO` (Gaussian / trig in predict or learn now
  cost more) and the FLOP budget (Gaussian setup now costs more) — that is expected.

- V15. **Permanent all-time-best memory (K3).** `meta_evolve` keeps a permanent, bounded
  set of the best programs ever seen, so a good program is not lost when a panel rotation
  pushes it out of the rotating hall of fame. The set is keyed by a rotation-invariant
  score — the program's best observed Δ_select over the completed periodic Stage C rounds,
  measured on the fixed SELECT tasks, not on the current search panel. It is capped by the
  new config field `select.all_time_best` (default 64, hard cap 64, 0 turns it off; validated
  in `MetaConfig::validate`), so it stores at most 64 genomes and does not reintroduce the
  AC-11 hall-of-fame blow-up (G4). Each periodic Stage C round offers its hall-of-fame
  members to the memory in a fixed order (`atb_offer`, best Δ first, ties by id, one entry
  per id). The memory feeds the search in two ways: at every panel rotation its members that
  have left the population are re-scored on the new panel and re-injected as fresh members,
  so they are eligible parents again; and the final phase draws its finalists from the hall
  of fame together with the memory, so a dropped program can still be a finalist. The memory
  is not written to the snapshot; it is rebuilt on resume from the archive's SelectEval and
  LearnerCandidate records (`rebuild_all_time_best`), which is provably equal to the
  incremental memory over the same rounds, so a resume keeps the same memory as an unbroken
  run. Everything merges in a fixed order and is independent of the thread count, so the
  archive and the lineage-chunk SHA-256 stay reproducible. This changes the search dynamics
  (new programs survive, breed and ship) — that is the intended effect of K3.
