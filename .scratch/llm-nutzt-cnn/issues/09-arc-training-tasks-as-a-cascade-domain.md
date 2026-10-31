# 09 ARC-AGI-1 training tasks as a Cascade Domain

Status: needs-triage (waits for the owner's approval of the design)
Type: task
Blocked by: the merge of PR 1 (both change `src/cascade/domain.rs` and
`src/cascade/arena.rs`); outside: for real data, cnn-basismodelle 06 pins the ARC-AGI-1
training files in the manifest. Today the manifest lists 680 evaluation files and no
training file (`data/manifest.example.json`). cnn-basismodelle 06 uses the split rule of
this ticket, and 07 records the provenance fields below (spec section 5, rule 7).
Spec: ../spec.md (sections 5, 8; decisions Q92, Q97, Q98)
Pull request: H2, a small one of its own

## What to do

- `DomainConfig::Arc(ArcConfig { manifest_sha256, select_share })`. Paths stay out of the
  config: brut-breed takes `--arc-root` and `--arc-manifest`, checks the manifest's
  SHA-256 against the config, and refuses a resume with another manifest. `--verify` and
  `--promote` accept both options too (`READ_KEYS`, `src/bin/brut-breed.rs:95`), since
  their rebuild needs the Domain. The ARC family takes its pool from `sim_pool`, as
  `grid_family` does (`src/bin/brut-breed.rs:953-962`), so `brut-breed-ollama` serves ARC
  runs.
- Load only manifest entries with role `Train` and source ARC-AGI-1 training. Refuse every
  other entry, `External` ones included (`src/arc_data.rs:150-197`).
- Drop the tasks 40853293 and caa06a1f; they repeat pairs of the evaluation tasks
  070dd51e and 50a16a69 (`src/arc_data.rs:249-251`).
- Split the tasks into a search and a select pool by a fixed rule over the SHA-256 of
  each task id, with the share `select_share`. The rule is one pure function, which
  cnn-basismodelle 06 uses too: select-pool tasks and their re-arc generations never enter
  genome breeding, validation, hidden scoring or promotion.
- An ARC run refuses a `cnn:` genome whose promotion file lacks provenance, whose split
  rule differs from the run's (the recorded hash covers `select_share`), or whose training
  sources fail `check_contamination` for A1Eval (`src/arc_data.rs:196-241`).
- The size method of Q92 computes the smallest and largest padded size and the most
  examples from the loaded pool.
- One puzzle per test pair. The prompts use the shared renderers of ticket 06, so grid
  prompts keep their bytes.
- `Breeder::start_with` and `resume_with` take a Domain built by the binary, since
  `from_config` cannot load files (`src/cascade/arena.rs:879`). A default method
  `Domain::select_panel` (used at `src/cascade/arena.rs:885`) lets the ARC Domain draw the
  select panel from its own pool.
- The ARC gene pool: the grid seeds and phrases, with 2048 answer tokens, since a 30 × 30
  answer needs about 900 whitespace tokens (`src/cascade/genome.rs:34-38`).
- The audit of section 5, rule 10: a tool that reads the CNN training sets back from a
  run's Call records, hashes them with `canonical_pair_hash` and checks them against the
  pair hashes of both evaluation sets (`src/arc_data.rs:243-284`).
- Hand-made training-role fixtures under `tests/fixtures/arc/`, as for the existing
  evaluation fixtures.

## Tests (write them first)

- `the_arc_domain_loads_only_arc_agi_1_training_entries`: `External`, ARC-AGI-2 training,
  ARC-GEN, ARC-TGI and BARC entries are refused.
- `the_two_tasks_that_repeat_evaluation_pairs_never_enter_a_panel`.
- `search_and_select_pools_are_disjoint_and_fixed`.
- `a_longer_panel_extends_a_shorter_one` (`src/cascade/domain.rs:46`).
- `each_test_pair_is_its_own_puzzle`.
- `prompts_hold_no_test_output`.
- `a_resume_with_another_manifest_is_refused`.
- `verify_recomputes_an_arc_run`.
- `a_genome_bred_on_a_select_task_is_refused`.
- `the_audit_finds_a_planted_evaluation_pair`.
- `grid_runs_keep_every_golden_fixture`.

## Acceptance criteria

- The fixture tests pass, and a short simulated run on the fixtures finishes and
  verifies.
- clippy is clean; the golden tests pass unchanged.

## Comments
