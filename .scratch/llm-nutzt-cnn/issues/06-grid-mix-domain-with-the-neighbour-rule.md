# 06 grid-mix Domain with the Neighbour rule

Status: needs-triage (waits for the owner's approval of the design)
Type: task
Blocked by: 02 (both CNN tests also need 04)
Spec: ../spec.md (section 4.2.4; decisions Q92, Q95)
Pull request: PR 1 (F1)

## What to do

- New pure module `src/cascade/neighbour.rs`: a seeded generator of Neighbour rule
  puzzles. A rule sets a cell's colour from its own colour and the colours of its four
  neighbours. A puzzle is well posed only if:
  - every grid keeps its size;
  - every example changes at least one cell and keeps at least one;
  - the test output differs from the test input;
  - every colour of the test input occurs in the examples;
  - no hypothesis that SimModel can form (`src/cascade/sim.rs:109-142`) fits any single
    example: Identity and every step of `MOVES` (`src/cascade/puzzle.rs:51-59`), each with
    and without a colour map fitted to that one example;
  - no step of `MOVES` applied to the test input gives the test output, so SimModel's
    guess fails too (`src/cascade/sim.rs:293`).
- Neighbour grids have 3 to 6 rows and columns, and a puzzle has `TRAIN_PAIRS` examples,
  as in the generator (`src/cascade/puzzle.rs:34, 343-359`). The grid-mix Domain returns
  these sizes through the size method of Q92.
- `DomainConfig::GridMix` and a grid-mix Domain in `src/cascade/domain.rs`. Task i of a
  panel is a generator puzzle when i is even and a Neighbour rule puzzle when i is odd.
  `--domain grid-mix` in brut-breed. Grid runs leave the field out, so their settings
  and hashes stay (`src/cascade/arena.rs:536-539`). The grid-mix family takes its pool
  from `sim_pool`, as `grid_family` does (`src/bin/brut-breed.rs:953-962`), so
  `brut-breed-ollama` serves grid-mix runs too.
- Share the grid renderers over (examples, test input), so every grid prompt keeps its
  bytes. Prompts never hold the rule or the truth (`src/cascade/puzzle.rs:470-480`).
- The gene pool of grid-mix: the grid seeds, phrases and briefs.

## Tests (write them first)

- `neighbour_puzzles_are_well_posed`.
- `no_simulated_hypothesis_fits_any_neighbour_example`: it fits each hypothesis with
  SimModel's own code (`hypotheses` and `Thinker::test`, `src/cascade/sim.rs:109-142,
  219-253`) on each example alone.
- `the_strongest_simulated_prompt_solves_no_neighbour_puzzle`: one `sim:large` Worker
  with 4096 thinking tokens and self-check whose strategy names every family.
- `the_reference_cnn_solves_most_neighbour_puzzles` (needs ticket 04): a calibration
  bound set 0.1 below the first measurement, then pinned.
- `cnn_then_sim_solves_both_halves` (needs ticket 04): [reference CNN; strong SimModel
  Worker] on a grid-mix panel.
- `grid_mix_never_renders_its_rule_or_truth`.
- `a_longer_grid_mix_panel_extends_a_shorter_one` (`src/cascade/domain.rs:46`).
- `grid_runs_keep_their_settings_prompts_and_panels`: the golden fixtures.

## Acceptance criteria

- All tests pass. clippy is clean.
- If the calibration fails, Comments record the measured share, and the rule is
  redesigned before ticket 07 starts.

## Comments
