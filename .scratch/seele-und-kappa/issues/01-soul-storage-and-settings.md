# 01 Soul storage and settings, off by default

Status: needs-triage (waits for the owner's approval of the design; no open question)
Blocked by: none
Spec: `../spec.md`, sections 4, 12.2 and 14

## What to do

- `src/cascade/genome.rs`: `Traits` gets `phase_state: Option<PhaseState>` and `coupling: u16`
  (thousandths), both with serde defaults and left out of the JSON at their default (Q42, Q43,
  Q44). `PhaseState` is a list of `u16`. `Traits` keeps `deny_unknown_fields`.
- `Cascade::gene_id()`: the id with both fields removed from every Worker. `Cascade::check`
  refuses `coupling` above 1000.
- `src/cascade/arena.rs`: `BreedConfig` gets `soul: Option<SoulConfig>`, left out of the JSON
  when `None`. `SoulConfig` holds the fields of the table in spec 12.2, all required, unknown
  fields refused. Until part B of ticket 03 gives Souls to the seeds and children, the breeder
  refuses every Run with Souls.
- The breeder's rule of spec 4.4 for Runs without Souls: no Worker carries `phase_state` or a
  non-zero `coupling`, whether it comes from the seeds, a resumed archive or a Mutator answer.
  The rule for Runs with Souls (exactly n phases per Worker) follows in part B of ticket 03.
- The Mutator never sees or writes a Soul (spec 4.5): `llm_mutate_request_with` and
  `llm_crossover_request_with` get the parents with both fields removed; `parse_llm_cascade`,
  or the admission step after it, refuses an answer that carries either field.
- `--promote` refuses a Run with Souls with exit code 1 (spec 4.6).

## Done when

- `cargo test` passes and clippy is clean.
- The tests of `tests/universe.rs` and `tests/mlp_golden.rs` pass with their fixtures unchanged
  (`git diff --exit-code tests/fixtures`).
- `old_traits_write_back_byte_equal`: every seed Cascade of both gene pools parses and serializes
  back byte for byte, with its id unchanged. The repository holds no Cascade JSON fixture, so the
  recorded ids are checked where they are kept: the `born` ids of
  `tests/fixtures/universe/grid_run.txt` and `grid_llm_run.txt` come out again (the goldens of
  `tests/universe.rs`), and so does the head hash of the step 1 pilot (the last item).
- `a_soul_changes_the_id_but_not_the_gene_id`: setting `phase_state` or `coupling` changes `id()`;
  `gene_id()` equals the `id()` of the Cascade without them; without them `gene_id() == id()`.
- `the_mutator_never_sees_or_writes_a_soul`: the mutate and crossover requests for parents with
  Souls equal, byte for byte and by `request_key`, those for the same parents without; an answer
  with `phase_state` or `coupling` is refused with the reason of spec 4.5.
- `soul_settings_are_left_out_when_off`: the JSON of `BreedConfig::default()` holds no `"soul"`;
  `fixed_hash` matches the settings fixture; a `soul` object with an unknown or a missing field is
  refused.
- `a_run_without_souls_refuses_souled_workers`: a seed, a resumed member or a Mutator answer with
  a `phase_state` or a non-zero `coupling` is refused; a Run with Souls is refused with a message
  that Souls are not built yet (until part B of ticket 03).
- `souls_off_reproduces_the_step1_pilot` (ignored, run with `cargo test --release -- --ignored`):
  the 60-generation Run of pilot seed 101 in arm A of step 1 ends its event log in the head hash
  that `docs/pilot-step1/pilot.json` records.

## Comments
