# 09 Program, `--verify` and the Character analysis

Status: resolved
Blocked by: 08
Spec: `../spec.md`, sections 12.5, 15 and 16

## What to do

- `src/bin/brut-breed.rs`: `--soul s|v|k` sets the `soul` field of arm S, V or K (spec 12.2);
  finer settings come with `--config`; giving both is an error with exit code 2. `--soul` defines
  the experiment, so a resume refuses it. The usage text names the option.
- `--verify` for Runs with Souls (spec 16): `check_archive` also compares the five new `born`
  fields; ticket 05 already put the check of current Souls there. Today `check_archive` runs only
  for a finished circuit Run, and a grid Run's `check_finish` checks nothing. So a finished grid
  Run with Souls gets the same two checks, by an archive check that works for either Domain; a grid
  Run without Souls is checked as today, with the same output.
- `examples/soul_character.rs`: reads a Run directory and prints, per member and Worker, the
  Character and the normalised Character on each distinct target or rule, from the checkpoint's
  current Souls and the Domain's strengths, so any Character can be recomputed from the record
  (Q51, Q55).
- End-to-end tests in a new `tests/soul.rs`.

## Done when

- `the_presets_are_the_arms`: `--soul s`, `--soul v` and `--soul k` write exactly the `soul`
  objects of spec 12.2 into `config.json`; `--soul` with `--config` holding a `soul` field fails
  with exit code 2; a resume with `--soul` is refused.
- `a_souled_circuit_run_reproduces`: the same command twice, and at 1 and 4 threads, gives
  byte-equal `config.json`, `events.jsonl`, `snapshot.json`, `report.json` and `rediscovery.json`,
  for each of the three arms; all pass `--verify`.
- `verify_refuses_tampered_souls`: on a finished circuit Run and on a finished grid Run, a changed
  birth Soul in the checkpoint, a changed imprinted Soul, a changed source map and a changed current
  Soul each make `--verify` exit with code 1 and an "integrity: …" message.
- `grid_verify_is_unchanged_without_souls`: `--verify` on a finished grid Run without Souls prints
  the same line as before this ticket.
- `promote_refuses_souls`: `--promote` on a grid Run with Souls exits with code 1.
- `the_character_example_agrees`: its output equals `character()` and `character_norm()` applied
  to the Souls of the checkpoint, for a short Run of each Domain.
- `rediscovery_is_unchanged_by_souls`: `rediscovery.json` of a souled circuit Run equals the
  recount from its logged answers, as for Runs without Souls.

## Comments

- 2026-11-02, build. Not committed: this build makes no commits.
  - `brut-breed`: `--soul s|v|k` sets the `soul` field to `SoulConfig::arm_s`, `arm_v` or
    `arm_k`. Next to a `--config` that holds a `soul` object it is wrong usage (exit code
    2); next to a config without one it adds the Souls to that config. An unknown arm is
    wrong usage too. A resume refuses `--soul` as it refuses every option that defines the
    experiment. The usage text and the docs at the top of the file name it.
  - `--verify`: `check_archive` is generic over the Domain. A grid Run with Souls gets
    `check_souled_grid`: for a finished Run it runs `check_archive`, which compares the five
    Born fields of step 2 and the current Souls, and the line ends with "; archive and Souls
    match the log"; otherwise it says why it did not check. A grid Run without Souls keeps
    the old check and the old line (`grid_family` picks the check by the settings).
  - `examples/soul_character.rs`: reads `config.json`, the checkpoint and the run start, and
    prints a table with tabs: per member, Worker and distinct target (circuits) or kind of
    rule (grids: the geometric step, with "+colours" for a colour map), the Character and
    the normalised Character of the current Soul, with the Run's readout and strength
    setting. Numbers are printed in the shortest form that reads back to the same f64. It
    refuses a Run without Souls, reads only the first line of the log, and ends quietly when
    its reader closes the pipe early.
  - Tests in `tests/soul.rs`: the seven named above. `the_character_example_agrees` takes
    the example in as a module (`#[path]`), runs it on Runs of both Domains (arm V on grids,
    arm S on circuits, and arm S with equal strengths) and compares every row exactly with
    `character()` and `character_norm()`. `a_souled_circuit_run_reproduces` uses seed 41
    with 20 generations, which holds 6 complex votes in each arm.
    `grid_verify_is_unchanged_without_souls` compares with two lines recorded on the
    unchanged binary before this ticket. `promote_refuses_souls` replaces
    `promote_refuses_a_run_with_souls` of ticket 01 and runs through `--soul` for each arm.
    Added: `grid_verify_with_souls_says_what_it_checked`.
  - Seen failing first: six of the seven failed on the unchanged code (`--soul` was
    unknown, the example a stub); `grid_verify_is_unchanged_without_souls` passed, as a
    guard recorded there must. After `--soul` existed, `verify_refuses_tampered_souls` still
    failed on the grid Run until the grid check was built. Twelve deliberate faults (a wrong
    preset, a silent override, a resume that takes `--soul`, the grid check left out or
    added without Souls, the current-Soul check skipped, and six faults in the example)
    were each caught.
  - End of the build (spec 14, last item): the counted Run A-1 of step 1, run by hand with the
    command of `docs/step1-counted/step1-driver.sh` on the release build of tickets 01 to 09,
    gives the five hashes of `docs/step1-counted/hashes.txt` again (499 s, 4 threads).
- 2026-11-02, review fix. Not committed: this build makes no commits.
  - `--verify` let a finished grid Run with Souls pass with exit code 0 when its checkpoint
    was marked unfinished or had lost its select-panel answers. The report was then checked
    only against the logged finalists, and the archive and Souls were not checked, so a
    changed current Soul got through and a resume would breed on it. A Run with Souls keeps
    every select-panel answer, so `verify` now refuses such a report for grid Runs with Souls
    too, as for circuit Runs. Only a grid Run without Souls keeps the fallback for Runs from
    before the select answers were kept.
  - Tests: `verify_refuses_tampered_souls` also changes a current Soul and then marks the
    checkpoint unfinished or empties its select answers, on both Domains; the grid case failed
    before the fix. `grid_verify_is_unchanged_without_souls` also checks that a grid Run
    without Souls still falls back to the finalist check.
  - Docs: `docs/STEP2_SOUL.md` and `docs/SCHRITT2_SEELE.md`, section 3, and the notes at the
    top of `src/bin/brut-breed.rs`.
