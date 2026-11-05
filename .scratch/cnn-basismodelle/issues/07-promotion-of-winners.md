# 07 Promotion of winners: bred models become base models

Status: resolved
Blocked by: 05
Spec: `../spec.md`

The owner asked on 2026-10-30 that winners can be fed back upward, that is, promoted.
The project already promotes the two best finalist Cascades of a Run into
`candidates/cascades/` through a draft pull request that a human merges (ADR 0004, ADR
0005; CONTEXT.md, "Promotion"). Models of the MLP mode, MLPs and CNNs, get the same path.

## What to do

- A `ModelAdapter` that implements the existing `CandidateAdapter` interface of
  `src/hitl.rs` for model genomes: parse with unknown fields rejected, validate with the
  shape check and the current Limits, fingerprint by the SHA-256 of canonical JSON.
  Promoted models live in `candidates/models/<content-id>.json`, under `candidates/`, so
  the existing scope fence covers them.
- Promotion of a finished run: the two best models by their score on the hidden split,
  which the run never saw while breeding. A model is promoted only if it beats the best
  base model of the same run on that split. Each file records its provenance: the run,
  the task, the seed, the commit, the scores on validation and hidden splits, and the
  lineage back to its base model. A model equal by content to one in the pool adds
  nothing.
- The promotion is a draft pull request that changes files under `candidates/models/`
  only. A human merges it; the run never merges anything (ADR 0004).
- Later runs: the first generation holds the merged promoted models next to the base
  models of the 16 families and the random genomes. A promoted model that fails the
  current shape check or Limits is skipped, with a log line.
- Docs: the "Promotion" entry of CONTEXT.md covers models too; README (German and
  English) and ADR 0008 describe the path.

## Done when

- Tests: the two best by hidden score are chosen; a model that does not beat the best
  base model is not promoted; a duplicate adds nothing; unknown JSON fields are refused;
  a promoted model loads into the first generation of a new run, and an invalid one is
  skipped; a run with promoted models is reproducible byte for byte from its seed.

Cost: about 10 to 20 dollars.

## Comments


**2026-10-31, Claude (subagent build-t07).** Built in the worktree of branch
`claude/cnn-basismodelle-teil2`, on top of tickets 03 to 05, nothing committed.

### What was built

- **Model adapter** (`src/hitl.rs`). `ModelAdapter` implements `CandidateAdapter` for a
  `PromotedModel`: a model genome (MLP or CNN) with its `ModelProvenance`. `parse` refuses
  unknown fields everywhere, in the genome too. `validate` runs `Genome::validate`, the shape
  check, under the adapter's `Limits` (by default the current `Limits::default()`) on the task
  that the provenance names. It also checks that the provenance is consistent: the run id is a
  hash, the commit 7 to 64 lower-case hex digits, the scores finite, the hidden score above that
  of the named best base model, the lineage starts with the model (which is no base model of its
  own run), names no candidate twice and names every parent it refers to. `fingerprint` is
  `Genome::id`, the first 16 hex digits of the SHA-256 of the canonical JSON. So the same genome
  with another provenance is the same candidate. Files live at `candidates/models/<id>.json`
  (`MODEL_DIR`), under the existing scope fence. `model_seed_ids` gives the ids of the 16 base
  models as built and as 1-D twins (31 ids, the TCN is 1-D already).
- **Strict genomes** (`src/genome.rs`). `Genome` gains `#[serde(deny_unknown_fields)]`, as its CNN
  parts had. No field of a genome was ever removed (`git log -p`), so every archive and config
  this code wrote still parses; the pre-CNN run fixture and `tests/mlp_golden.rs` pass unchanged.
- **Promotion** (`src/promote.rs`, new, pure). `choose` takes a finished run's archive: the
  candidates of status Ok that are no base model and have a hidden score, ranked by hidden score,
  then fitness, then id; the first two are considered. Each is promoted if its hidden score is
  above that of the best base model, its content id is not known (pool and base models of the
  families), and it passes the adapter. A failed one is not replaced by the third, so a second
  promotion of the same run adds nothing new. Errors: a malformed commit, no hidden evaluation,
  an incomplete hidden set ("run brut run again"), no base model. The manifest for the pull
  request body holds no path.
- **Base models and the hidden set** (`src/evolve.rs`). `is_base_model`: a base genome of the
  config (`base`), a base model of a family (`base-model/<family>`), a promoted model
  (`promoted`). `best_base_model`: the highest fitness, ties by id, status Ok or Quarantined.
  `hidden_set` adds the best base model after the `hidden_eval_top_k` best, so that its hidden
  score exists; a run without base models (every MLP run as before CNNs) keeps its set, and a
  top-k of 0 still evaluates nothing.
- **Later runs** (`src/config.rs`, `src/evolve.rs`). `Config::promoted` (left out of the JSON when
  empty) holds the genomes that `brut init --promoted <dir>` copied from the pool.
  `first_promoted` leaves out, with a log line each, a model of a kind the run does not breed, one
  that fails `Genome::validate` for the run's limits and task, and one already in generation 0.
  The others enter generation 0 after the base models of the families and before the random
  genomes, as `promoted`; they draw nothing from the random stream. `Config::validate` caps the
  list at `HARD_MAX_POPULATION` and, unlike for base genomes, does not refuse a model that does not
  fit: the run leaves it out.
- **CLI** (`src/main.rs`). `brut promote <run_dir> --commit <sha> [--pool <dir>]` writes the
  promoted models and `proposal.json` under `<run_dir>/promote/` (replaced on each call; removed
  when nothing is promoted). `--pool` defaults to `candidates/models`; a missing default pool is
  empty, a missing named one an error. `brut init --promoted <dir>` reads every `*.json` of the
  pool, parses it strictly and requires the file name to be the content id; a broken pool file is
  an error that leaves nothing behind. `--promoted` with `--mode learner` is a usage error.
- **Gate, CI, script.** `brut-ingest --models <dir> [--against <pool>]`; the scope-fence workflow
  runs it on `pr/candidates/models` when that exists. `scripts/propose-pr.sh` counts the candidate
  files (a model promotion may carry one), names that a model promotion was chosen by hidden
  score, and refuses a file that the base branch holds already: a model file also holds
  provenance, so copying it would rewrite a pooled model's provenance.
- **Docs.** ADR 0008 gains four paragraphs (the promotion rule, the file and its gate, later
  runs, why the hidden score may choose) and the rejected alternatives; CONTEXT.md: Promotion now
  covers models, Base model covers promoted models and base genomes; README.md and README.en.md:
  "Gewinner befördern" / "Promoting the winners" with commands and the measured runs below, two
  limits, the `promote/` directory of a run; `candidates/README.md`: a Models section.

### Tests (Done when)

- The two best by hidden score are chosen: `promote::tests::the_two_best_by_hidden_score_are_chosen`
  (hidden order differs from fitness order; provenance, lineage, roots and manifest checked);
  end to end in `promoted_models_go_from_a_finished_run_into_the_first_generation_of_a_new_run`
  (`tests/e2e.rs`): a spirals run with a weak base genome and a fixed hidden seed; `brut promote`
  writes exactly the files of `promote::choose` (hidden 0.56 and 0.54 against 0.44 of the base
  genome), and `brut-ingest --models` accepts them.
- A model that does not beat the best base model is not promoted:
  `a_model_that_does_not_beat_the_best_base_model_is_not_promoted` (one below the bar; a tie
  promotes nothing), `quarantined_models_and_base_models_are_not_promoted`.
- A duplicate adds nothing: `a_duplicate_adds_nothing` (pool and family base models, no
  refill); in the e2e test a second `brut promote` against the merged pool prints "nothing to
  promote" and leaves no `promote/` directory;
  `model_ingest_refuses_base_models_of_the_families_and_renamed_files`.
- Unknown JSON fields are refused: `model_adapter_refuses_unknown_fields` (top level, provenance,
  best base model, lineage step, genome of a CNN and of an MLP, a CNN stage);
  `a_genome_with_an_unknown_top_level_field_does_not_parse`;
  `init_takes_the_promoted_models_of_a_pool_and_refuses_a_broken_pool` (`src/main.rs`).
- A promoted model loads into the first generation of a new run, and an invalid one is skipped:
  the e2e test above (`brut init --promoted`; both promoted models enter generation 0 as
  `promoted`; a pool model of 50 epochs gets the log line "generation 0: promoted model ... left
  out: invalid: genome: epochs 50 outside 5..=20"); `promoted_models_join_generation_0_after_the_base_models`
  (order, kinds, limits, duplicates of a base model and of an earlier promoted model);
  `an_mlp_run_with_a_promoted_mlp_draws_its_random_genomes_as_before`.
- A run with promoted models is reproducible byte for byte from its seed:
  `a_run_with_promoted_models_is_reproducible_byte_for_byte` (`tests/e2e.rs`): a run of both kinds
  on bars and stripes with a promoted CNN, a promoted MLP and a promoted MLP over `max_params`
  (700, which also leaves out all 16 family base models), `brut run --no-hidden` in two child
  processes at `RAYON_NUM_THREADS` 1 and 2. Config, archive (clock fields masked), head and all 7
  model files are equal, and so are the summaries.
- Also: `the_hidden_set_adds_the_best_base_model`, `model_adapter_checks_the_genome_on_its_task_and_the_provenance`
  (14 broken variants and tighter limits), `promotion_needs_a_commit_a_finished_hidden_evaluation_and_a_base_model`,
  `a_model_that_fails_the_current_limits_is_not_promoted`, `every_generator_name_of_the_code_fits_a_lineage`,
  `promoted_models_are_left_out_when_empty_and_checked_by_the_run` (config).
- Whole suite (`nice -n 10`, `CARGO_BUILD_JOBS=3`, `RAYON_NUM_THREADS=2`): 952 passed, 0 failed,
  18 ignored (ticket 05 left 933; this ticket adds 19). Library "813 passed; 0 failed; 13
  ignored ... finished in 164.33s"; `tests/e2e.rs` "19 passed ... finished in 83.83s";
  `tests/mlp_golden.rs` "9 passed ... finished in 0.18s", `tests/fixtures` unchanged.
  `cargo clippy --all-targets -- -D warnings`: exit 0. `rustfmt --edition 2024 --check` on the 9
  changed `.rs` files: exit 0 (`src/lib.rs` through stdin, since a path check follows its `mod`
  lines into 4 files that are unformatted on main).

### Measured runs

Shapes task, `cargo build` (profile dev, opt-level 2), `RAYON_NUM_THREADS=2`, with the hidden
evaluation; another workflow kept the machine busy (load up to 10), so run times are not
comparable (287 s and 301 s). No promotion went into the pool.

- The demo run of ticket 05 (a copy, seed 1): best base model DenseNet, valid 0.939, hidden 0.93.
  Promoted: `4ea1fec7699a4627`, the GoogLeNet-VGG mixed form, valid 0.934, hidden 0.94, lineage of
  7 candidates back to the base models of GoogLeNet, VGG and ShuffleNet. Not promoted:
  `bebb51cdbc9bcf2c`, hidden 0.92. `brut-ingest --models` accepts the file.
- `brut init --models cnn --seed 2 --promoted <pool>`: 63 candidates, all Ok, none stopped by the
  clock. The promoted model, valid 0.951 on the data of seed 2, is the best base model (DenseNet
  0.940). 15 of the 40 children descend from it, among them the best, `84d5076ce10aab99`
  (generation 5, valid 0.969, hidden 0.97). `brut promote` promotes it and `1d0d1eb23eea7d18`
  (hidden 0.97), both above the 0.94 of the promoted model.
- The same without `--promoted`: 62 candidates, all Ok; the best is `5ffbf573c5b4c630`
  (generation 1, from the VGG base model, valid 0.956, hidden 0.96), and no better one follows.
  One pair of runs: it shows the path works, not that promoted models help on average. The two
  runs have different hidden seeds, so only their validation scores compare.

### Limits

- Only the hidden set has hidden scores, so the two best by hidden score come from the
  `hidden_eval_top_k` best by fitness (3 by default).
- The hidden set of a run with base models holds one more candidate when its best base model is
  not among the top 3 by fitness, so such a run spends one more hidden query (at most 4 of the
  default budget of 30). A run of ticket 05 whose best base model was not in its top 3 gets that
  query when `brut run` runs on it again.
- Config base genomes count as base models; an MLP run without them or promoted models promotes
  nothing, so the first promoted MLP must come from a run of both kinds or with base genomes.
- `--commit` is checked for its form only, and a run from a working tree with uncommitted
  changes has no matching commit (as the demo runs here, made on this uncommitted branch).
- The gate checks against the current default limits; a winner of a run with larger limits that
  does not fit them is not promoted.
- A promotion pull request carries one or two models, not always two as a mutation pull request
  does (ADR 0004); ADR 0008 records this.
- The CI step for models can only run once this branch is on main, since the checker is built
  from the base revision.
- No git writes; the main session commits. The relayed user request was only the IP address
  89.169.97.77, with no instruction; nothing was done with it.

**2026-10-31, Claude (subagent fix), after the review of tickets 03 to 05 and 07.**

- **MLP runs keep their hidden set.** `hidden_set` added the best base model whenever a run had
  one, and config base genomes count as base models. So an MLP run of an old config with base
  genomes spent one more hidden query and wrote one more record. A finished such run got the
  query when `brut run` ran on it again. That broke the rule that MLP behaviour stays byte
  equal. Now only a run that can promote adds it: `Config::can_promote`, true when the run
  breeds CNNs or holds promoted models. `promote::choose` refuses any other run and says why.
  So an MLP run with only base genomes promotes nothing, and the first promoted MLP comes from
  a run of both kinds. The e2e test of the promotion path now starts its MLP run from a weak
  promoted model. `an_mlp_run_with_base_genomes_keeps_its_hidden_set_and_promotes_nothing`
  covers the old case.
- **Add-only in CI.** The scope fence now refuses a mutation pull request that changes or
  removes a file the base branch holds (`git diff --diff-filter=a`), as the script already
  did. A model's content id hashes its genome only, so a rewritten provenance kept its name and
  passed every gate before.
- **Bounded reads.** `brut init --promoted` and `brut-ingest` read each candidate file with
  `util::read_regular_file`: no symlink, FIFO or device, and at most `MAX_CANDIDATE_FILE_BYTES`
  (16 MiB). Before, a FIFO blocked both, and a symlink to /dev/zero ran out of memory. A file
  that the gate cannot read is now a rejection, not an abort.
- **Log lines.** The generator of the best base model must be a generator name, as each one in
  the lineage is, and the provenance messages quote their text. The gate escapes the control
  characters of every file name and message it prints, so no file can forge a line of the CI
  log.
- **Tests.** `the_hidden_set_adds_the_best_base_model` (also without the flag),
  `only_a_run_with_cnns_or_promoted_models_can_promote`,
  `promotion_needs_a_commit_a_finished_hidden_evaluation_and_a_base_model` (MLP configs with
  and without base genomes), two new cases in
  `model_adapter_checks_the_genome_on_its_task_and_the_provenance`,
  `read_regular_file_refuses_what_is_no_small_regular_file`, a symlink, a FIFO and a huge file
  in `init_takes_the_promoted_models_of_a_pool_and_refuses_a_broken_pool`,
  `one_line_keeps_a_log_line_to_one_line` (`brut-ingest`), and the e2e test
  `the_model_gate_rejects_unreadable_files_and_keeps_its_log_lines`: a FIFO, a symlink to
  /dev/zero, a sparse file above the cap, a forged generator and a file name with line breaks
  give 5 one-line rejections within the time limit.
- **Docs.** ADR 0008, CONTEXT.md, README.md, README.en.md, candidates/README.md,
  docs/SICHERHEIT_UND_RECHT.md, docs/ARCHITEKTUR.md and docs/STUFE2B_PR_MUTATION.md.
- **Evidence.** Whole suite (`nice -n 10`, `CARGO_BUILD_JOBS=3`, `RAYON_NUM_THREADS=2`): exit
  0, 962 passed, 0 failed, 18 ignored (952 before this fix). Library "820 passed; 0 failed; 13
  ignored ... finished in 146.78s"; `tests/e2e.rs` "21 passed; 0 failed ... finished in
  143.27s"; `tests/mlp_golden.rs` "9 passed; 0 failed ... finished in 0.28s", with
  `tests/fixtures` unchanged. `cargo clippy --all-targets -- -D warnings`: exit 0. `rustfmt
  --edition 2024 --check` on the 13 changed `.rs` files: exit 0. `cargo audit`: exit 0 over
  164 crates, with the one allowed warning RUSTSEC-2024-0436 (paste); no dependency changed.
- **Not done here.** The branch is behind origin/main, which holds WIP merges of this work
  (PR #27 and #32). That needs a rebase, which only the main session may do. `brut-propose`
  still reads the merged genome pool with `read_to_string`; the gate in CI now rejects what it
  could not read safely.
