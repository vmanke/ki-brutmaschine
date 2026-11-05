# 12 ADR 0009, READMEs, report skeleton and glossary

Status: resolved
Blocked by: 09
Spec: `../spec.md`, section 20

## What to do

- `docs/adr/0009-workers-carry-an-inherited-soul-whose-votes-interfere.md` (English): the decisions
  of step 2 (Q42 to Q46, Q51 to Q66) with the owner's answers to Q67 to Q74: storage, Family
  strengths, Character, Array pattern and complex vote with its tie rule, Observations and κ,
  Souletics, Upbringing, the Soul in the simulated models, controls, logs, byte-equality when off.
  It states which quantum forms of Q45 step 2 builds: the first as the Soul, the third as the
  interfering Stage vote, a simple form that models no single judgment, both against the real
  readout; the second, quantum Workers, follows in step 5 (Q47); a full model of one judgment has no
  step yet. One ADR for the step (Q47).
- `README.md` (German) and `README.en.md`: a section on the Soul with `--soul s|v|k` and one
  reproducible command on a pilot seed, with the line it prints.
- `docs/SCHRITT2_SEELE.md` (German original, by `docs/SCHREIBSTIL.md`) and `docs/STEP2_SOUL.md`
  (English): purpose, what step 2 built, how to run and verify a Run, what the `soul` group means.
  The results follow in ticket 13. German reports say "Seele (Phasenzustand)" (Q43), "Seeletik"
  (Q66), "Kopplungsstärke des Beobachters" (Q60) and "Gruppencharakteristik der Stufe" (Q61).
  `docs/SCHREIBSTIL.md` lists the new German report.
- The glossary in its own small pull request, never inside the code pull request (Q53):
  - the heading "Soul and Character (planned, step 2)" loses "planned";
  - Soul gains the birth Soul, the imprinted Soul and the current Soul;
  - Character keeps its definition, and the normalised Character (the Character divided by the
    sum of the squared Family strengths), which the logs use, is named there;
  - Universe state counts seven groups in Runs with Souls;
  - Trait says that the Soul and the Coupling strength are stored with the Traits but are
    inherited by Souletics, not as genes [Q71 a];
  - Souletics passes the Soul and κ [Q71 a];
  - Arm adds the arms of step 2: S, V (the real readout, German Vorzeichen), K, and in the pilot
    E and N;
  - any other term the build needs.

## Done when

- ADR 0009 exists, names every decision it implements by its number, and links the spec and the
  report skeleton. The link to `docs/PREREG_STEP2_SOUL.md` is added in the pull request of tickets
  10 and 11, since the pre-registration does not exist before it.
- The README command, run at the commit of the code pull request, prints the line both READMEs show;
  the check is recorded in this ticket's Comments.
- `git diff --stat` of the code pull request lists no change to `CONTEXT.md`, and the glossary pull
  request changes `CONTEXT.md` only.
- `docs/SCHREIBSTIL.md` names `docs/SCHRITT2_SEELE.md`.

## Comments

- 2026-11-02, build. Not committed: this build makes no commits.
  - ADR: `docs/adr/0009-workers-carry-an-inherited-soul-whose-votes-interfere.md`. It names
    Q42 to Q48 and Q51 to Q74 by number, says which quantum forms of Q45 step 2 builds, lists
    the options of Q67 to Q74 that were not chosen under "Rejected", and links the spec and
    both reports. The link to `docs/PREREG_STEP2_SOUL.md` follows with tickets 10 and 11.
  - READMEs: since pull request #44, `README.md` is the English README and `README.de.md` the
    German one, and `README.en.md` is gone, so this ticket's file names are out of date. Both
    READMEs got a section on the Soul ("The Soul (step 2)", "Seele und Kopplungsstärke
    (Schritt 2)") with `--soul s|v|k` and one command on pilot seed 901, plus the status
    lines, the table of building blocks, the tests and the limits.
  - Report skeleton: `docs/SCHRITT2_SEELE.md` (German original) and `docs/STEP2_SOUL.md`
    (English): purpose, what step 2 built, how to run and verify a Run, what the `soul` group
    means, what comes next, limits and sources. `docs/SCHREIBSTIL.md` names the German
    report.
  - Glossary (Q53): not in this branch; `CONTEXT.md` is unchanged here. The change is
    prepared as a patch that touches `CONTEXT.md` only, for a pull request of its own. It
    drops "planned" from the heading; gives Soul its birth, imprinted and current Soul;
    names the normalised Character under Character; counts seven groups in the Universe
    state of a Run with Souls; says under Trait that the Soul and κ are stored with the
    Traits but pass by Souletics; lets Souletics pass κ; adds the arms S, V, K, E and N under
    Arm; says under Coupling strength that the observer is pulled back by half as much; brings
    Upbringing in line with Q73 a and d; and adds Readout as a new term.
  - README check: the command of both READMEs ran on the release build of the worktree,
    which holds tickets 01 to 09: `rediscovery.json` line "the seeds solve 2 of 11 targets,
    bred Cascades 2 of the other 9; D > 1/2: no; calibrated", and `--verify` "2046 events,
    ..., head e38339538f19d85fccd1d1a7a1c3a0435e1989eec7838e6b6e0c5a46e0e692a1, ...;
    rediscovery recomputed and equal", as both READMEs show. The debug build gives the same
    bytes. There is no commit yet, so the check at the commit of the code pull request is
    still to be done.
  - Seen on that seed, for the pilot: the three arms make the same 484 newborns up to
    generation 60 (the Soul changed no outcome), and in arm V the contrast lies above its
    formula (1.217 against 0.927), unlike spec 13.1 expects; the reports say so.
