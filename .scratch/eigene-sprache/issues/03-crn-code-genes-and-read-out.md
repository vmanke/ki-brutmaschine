# 03 Chemical networks: code genes and a fitted linear read-out

Status: needs-triage
Blocked by: tickets 01 and 02 of `.scratch/chemische-netze/issues/` (simulator and genome)
Spec: `../spec.md`, sections 2 and 4

## What to do

- Write a design note, not code, as an addition to ticket 02 of `.scratch/chemische-netze/`.
- Input code genes: each task input chooses one code from a menu (dual-rail, single-rail with an
  offset, the fractional code of Salehi, Parhi and Riedel 2017, a population code over a few
  species), with integer parameters. State the species and the concentration range each code
  needs, and how the genome check refuses a code that leaves the caps.
- Reporter genes: a set of reporter species read at the fixed read-out time. A linear read-out,
  fitted by ridge regression on the training split, maps them to the logits. State the ridge
  constant, that the fit is deterministic, and that its weights count as parameters.
- Keep Fog 1 of `../../chemische-netze/map.md`, "semantics first": the reaction dynamics must
  be chosen first, and this ticket assumes mass action with a fixed read-out time. Add the
  decoder as a separate item of that Fog: a fixed linear decoder with a free code inside.
- Name four controls: the same read-out on the raw inputs, a fixed random reservoir with the same
  read-out, a random assignment of codes, and the fixed read-out of the German draft
  (`entwurf-rekurrentes-chemisches-netz.md:759-760`).
- State the efficiency: the fitness form of the German draft (lines 899-902) with the read-out
  weights and code genes among the parameters, and dissipation reported, not charged.

## Done when

- The note answers each point above with numbers or with an explicit "open", and the owner has
  read it.
- Every claim about chemistry cites a checked source; every claim about the code cites a file and
  a line.

## Comments

- 2026-11-07, design note for the owner: `../../chemische-netze/code-genes-and-read-out.md`.
- 2026-11-20, reviewed: the arithmetic, the code lines cited and source [S1] were checked again.
  The note found two errors in `../spec.md`, now fixed: section 4 cited the research note for the
  ADR on charging dissipation (it is the German draft, Decision 5), and the formose reservoir
  used ridge only for prediction and forecasting. The stale fitness line of Fog 5 in
  `../../chemische-netze/map.md` is fixed too. The ticket is done when the owner has read the
  note. Its main open choice: dissipation reported, as the spec says, or charged, as ticket 02
  says.
