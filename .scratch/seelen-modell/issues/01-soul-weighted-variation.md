# 01 Soul-weighted variation behind a switch (arm W)

Status: needs-triage
Blocked by: none
Spec: `../spec.md`, section 4.1

The owner decides first whether candidate W gets built. This ticket builds it, off by default,
so that ticket 02 can screen it. It changes `src/`, so it waits for the owner's approval.

## What to do

- `src/cascade/soul.rs`: `SoulConfig` gets a field `variation`, `off` or `prefer`, with a serde
  default of `off` and left out of the JSON when off. Every existing config, settings hash, log
  and test then stays as it is, and the pilot's `config.json` files still read. A resume refuses
  a change, as for every other `soul` field.
- The preference of section 4.1: `preference(soul, readout)` gives w_f = 1 + Re u(p_f) for each
  family, so 1 + cos(2π p_f / 65536) with complex phases and 0 or 2 with the sign readout.
- The gene pool gets a hook that names the families of a text, in family order. For circuits it
  uses the stems of `designer::FAMILY_STEMS`, as `designer::knowledge` does. A Run with
  `variation: prefer` on a Domain without the hook is refused.
- `src/cascade/variation.rs`: weighted forms of SectionReplace (which Worker, which section,
  which phrase), SectionAdd (which Worker, which phrase, times the protection of the section it
  would overwrite) and `cross_worker` (the coin of each section). The weights are those of
  section 4.1, each at least ε = 0.05, with one uniform draw per choice. With `variation` off
  the operators run exactly as today.
- `src/cascade/arena.rs`: the operators still get the parents' genes without Souls (spec 9.1),
  plus each parent Worker's preference from its birth Soul. The child's Souls come from
  Souletics as today.
- The `soul` group of the `state` line gets `variation_choices` and `variation_changed` per
  generation: the weighted choices, and those whose pick differs from the uniform pick with the
  same draw. Both are left out when `variation` is off.
- An amendment of ADR 0009: with `variation: prefer` the Soul shapes variation too, not only
  votes and the designer's replies.

## Done when

- `cargo test` passes and clippy is clean; the fixtures are unchanged
  (`git diff --exit-code tests/fixtures`), and the release test of pilot seed 101 still ends in
  its recorded head hash.
- Tests show: the mean preference over 2^20 fresh Souls is 1 within 0.01 per family; no weight
  falls below ε; a Worker with w = 2 on a family overwrites a text naming it less often than
  today's operator does, by the expected factor within 3 standard errors over 100 000 draws; with complex
  phases all at a quarter turn (w = 1) the weighted choices match today's choices in
  distribution, including the draw of `replacement_text` (0.7 from the same section name).
- A 60-generation circuit Run with `variation: prefer` passes `--verify`, and its five record
  files are byte-equal at 1 and at 4 threads.
- `CONTEXT.md` is not changed by this ticket: "preference" stays a word of the design note until
  the owner adopts the model.

## Comments

- 2026-11-02, built, not committed; the owner reviews and commits. The owner chose W and added
  arm WF, the upper bound: `"variation": "fixed"` uses the same weights with the preference 2
  for Linear, Phase and Hadamard and 1 for every other family, from no Soul. WF keeps the rest
  of arm S, as W does, so W and WF differ only in where the preference comes from. It is off
  by default, and a Domain without these three families refuses it.
- Decisions within the spec. Each choice takes one draw. A choice whose weights are all equal
  makes today's draw (`random_range`, or `random_bool(0.5)` for the crossover coin). Otherwise
  one u in [0, 1) picks by the summed weights. So with every preference at 1 the operators are
  exactly today's, draw for draw, and not only in distribution. Arm W draws as arm S until its
  first choice of unequal weights. `variation_choices` counts the choices of unequal weights in
  the operator call that made each child of the generation; failed attempts and refused children
  do not count. `variation_changed` counts those whose pick differs from ⌊u·n⌋, or for the coin
  from u < 1/2. The lean of a Worker's whole prompt reads its system prompt, section names
  included; the lean of a section or a phrase reads its text, since an overwrite keeps the name.
  In choice 1 each candidate Worker is weighed by its own birth Soul; choices 2 and 3 use the
  chosen Worker's. Each factor of a SectionAdd weight and their product are at least ε. The hook
  is `designer::named_families`: the families of `designer::knowledge`, in family order. The
  grid gene pool has none, so a grid Run with `variation` exits with code 1 before it writes
  a record. Like every other refusal of a fresh Run's settings, it leaves only the empty run
  directory and its LOCK file (found in the review of PR #52).
- Tests. `soul.rs`: `variation_is_off_by_default_and_left_out` (the pilot's soul object of
  `config-E.json` reads and writes back byte for byte), `the_preference_has_mean_one_per_family`
  (2^20 fresh Souls per Domain, both readouts) and
  `the_fixed_preference_favours_linear_phase_and_hadamard`. `variation.rs`:
  `no_weight_falls_below_the_floor`, `a_preferring_worker_protects_its_family_by_the_expected_factor`
  (100 000 draws), `each_choice_follows_its_weights`,
  `with_every_preference_one_the_operators_are_todays` (the 0.7 draw included),
  `a_choice_changes_when_the_uniform_pick_of_its_draw_differs` and
  `steered_operators_see_genes_only`. `arena.rs`: `variation_needs_a_gene_pool_that_names_families`,
  `the_birth_soul_steers_variation` (current Souls do not steer) and
  `a_run_with_variation_logs_its_choices_and_resumes_like_a_straight_one`. `tests/soul.rs`:
  `a_run_whose_soul_steers_variation_reproduces` (60 generations, `--verify`, five record
  files at 1 and 4 threads), `variation_needs_circuits`, and the release test
  `the_readme_run_of_arm_s_is_unchanged` (head e38339538f…).
- Gate, 2026-11-02. `cargo test` and `cargo clippy --all-targets -- -D warnings` pass; the
  fixtures are unchanged; both release tests pass. The protection test measured the factor
  0.072 for 0.073 (SectionReplace) and 0.095 for 0.095 (SectionAdd) at w = 2, and 0.600 for
  0.600 and 0.666 for 0.667 at w ≈ 1.5, each within 3 standard errors. Smoke Runs on pilot
  seed 913, 60 generations, the settings of ticket 02, release binary: W, W0, WF and N pass
  `--verify`, and their five record files are byte-equal at 1 and 4 threads. The weights
  changed 108 choices in W (1.8 per generation, the first in generation 1) and 77 in WF (1.3
  per generation, the first in generation 1). These Runs are no part of the screen.
