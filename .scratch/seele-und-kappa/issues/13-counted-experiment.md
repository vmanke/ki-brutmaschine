# 13 The counted experiment and its report

Status: needs-triage (starts only after the frozen pre-registration of ticket 10)
Blocked by: 10, 11, 12
Spec: `../spec.md`, section 17

## What to do

- Run the counted triples by the frozen `docs/PREREG_STEP2_SOUL.md`, with a driver script modelled
  on `docs/step1-counted/step1-driver.sh`: the seeds in list order, one process at a time,
  `--verify` and the hashes of the five files after each triple, the frozen analysis after each
  triple, the thread check after triple 1.
- Recount with `reference_rule.py`, and recompute every reported number from the Run files with a
  script that stops at any difference, as `docs/step1-counted/make_counted.py` does.
- Put the data and scripts into `docs/step2-counted/`, and the result into
  `docs/SCHRITT2_SEELE.md` and `docs/STEP2_SOUL.md`: both verdicts with E_V and E_K at the stop, every
  deviation with its reason, the void conditions checked one by one, the descriptive results.

## Done when

- Every counted Run passes `--verify`, and the recorded hashes match the files.
- The thread check of gate G4 found no differing byte.
- The frozen analysis and `reference_rule.py` give the same rows, stop triple and verdicts.
- The recomputing script runs through on all counted Runs.
- The report states the verdicts as the pre-registration's section 13 words them, and nothing beyond
  rung 1 of the ladder.

## Comments
