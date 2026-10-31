# 12 ADR 0009, READMEs, report skeleton and glossary

Status: needs-triage (waits for the owner's answers to Q67 to Q74)
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
