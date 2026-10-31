# 09 Program, `--verify` and the Character analysis

Status: needs-triage (waits for the owner's approval of the design; no open question)
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
