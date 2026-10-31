# Spec: Universes that invent (overseers, soul, qubits)

Status: done (step 1), needs-triage (later steps)

Builds on Stage 2c (docs/adr/0005-*.md, `.scratch/kaskaden-zucht/`). Decided in the
grilling session of 2026-10-29, questions Q24 to Q47 (Q1 to Q23 are in the Stage 2c spec).

## Goal

Above all, the breeding machine should **invent**: produce artefacts that an exact checker
confirms as correct, that beat a pre-registered reference, and that are new. First field:
quantum circuits on exactly simulated qubits. Around it: overseers that watch execution,
a complex "soul" state with an interaction strength kappa, human feelings that steer the
search, and runs studied as universes.

## Decisions

| # | Decision |
|---|---|
| Q24 | The overseer watches execution (who answered what, tokens, limits hit, votes, pass-through, time), not only outcomes. |
| Q25 | Overseers are bred children on the same level as the Cascades (same archive, same DGM selection); below them, fixed code measures the hard facts. |
| Q26 | A report is a suspicion score in [0, 1], a category from a fixed list, and a short reason that goes to the log only. |
| Q27 | The generator acts on a report only when fixed code or a known answer confirms it; otherwise it only logs it. |
| Q28 | First a dedicated overseer lineage with its own fitness (detection skill); later "every child also oversees" as a comparison. |
| Q29 | Two bred levels (overseer, meta-overseer); the chain always ends in fixed code and the human. |
| Q30 | Protection against collusion: no relatives, redacted traces, planted fakes, quarantine on too many misses or false alarms. Backpropagation: blame flows up the overseer levels, trust weights are learned by gradient on confirmed cases, TextGrad-style textual feedback becomes a Mutator operator; blame flowing back to ancestors only as a pre-registered experiment. |
| Q31 | Fixed code plants five fake kinds (free rider, judge manipulation, voting cartel, constant answer, inconsistent numbers) at 5 to 10 % of watched executions; list and rate hidden from Mutator and overseer prompts. |
| Q32 | A run is a universe; universes with different laws (configs) are compared in pre-registered experiments; a "god of gods" that breeds laws comes later and may only choose configs within code ceilings. |
| Q33 | Six observables per generation, logged: performance, diversity (incl. Shannon entropy of building blocks), lineage tree, honesty (fakes caught, false alarms, confirmed collusion), judge skill and trust, cost. |
| Q34 | Human feelings enter as pair preferences, free-text feelings that Claude turns into mutations, and humans acting as the Mutator. |
| Q35 | Humans see only executions and search-panel numbers, never the select panel or sealed evaluation; every intervention is logged with name and time; such universes are labelled. |
| Q36 | Human feelings steer the search (parents, mates, which Traits mutate), never the measured fitness. |
| Q37 | Invent quantum circuits (a), programs and constructions with exact checkers (b), and let Cascades "think quantum" in the buildable forms of Q45 (c). Start with (a). |
| Q38 | An invention is exactly correct (fidelity >= 1 - 1e-9 up to global phase for circuits), better than a pre-registered reference, new in a canonical form, and called "new to the world" only after a human literature check. |
| Q39 | Qubits are simulated exactly in the Rust core without new dependencies (about 12 here, about 25 on the host); real quantum hardware later, through a broker, with its own ADR. |
| Q40 | First rediscover known optimal circuits (positive control), then query algorithms as in Spector et al. (1999). |
| Q41 | Gate set Clifford+T (H, S, S-dagger, T, T-dagger, X, Z, CNOT); T-count is the cost measure. |
| Q42 | Each Worker carries a complex phase state ("soul") as a Trait; tested against a real-valued control of equal size, budget and seeds. |
| Q43 | Code name `phase_state`; German reports say "Seele (Phasenzustand)". |
| Q44 | Observation is an interaction of strength kappa in [0, 1] (the feeling): weak measurement changes little, strong measurement fixes the phases; both sides change; kappa is an inherited Trait of the observer. Controls: kappa fixed at 0; real-valued states. |
| Q45 | Build all three quantum forms, each against a classical control: quantum-inspired amplitudes (Han and Kim 2002), quantum Workers (simulated circuits such as Grover search, after the simulator exists), quantum probability for judgments (Pothos and Busemeyer 2013). |
| Q46 | With real language models the soul acts in voting: votes add as complex numbers and the squared magnitude decides; in the simulator it also interferes over explicit hypotheses. |
| Q47 | Build order: (1) universe observables and inventing with qubits; (2) soul and kappa; (3) overseers; (4) human feelings; (5) quantum Workers and the TextGrad operator. One PR and one ADR per step. |

## What the experiments may claim

The owner wants the experiments to speak about the soul and about physics as well. We
record the ladder of claims the evidence can carry:

1. **The model (our universes):** whether complex phase states, kappa or overseers help
   Cascades invent, against matched controls. Measurable in every run.
2. **Human feelings:** whether human feeling input improves invention, under blind
   protocols. A claim about the value of human intuition in the loop.
3. **Physics:** invented circuits run on real quantum hardware (Q39, later) test that the
   inventions work in nature. That tests quantum mechanics as applied, which is settled
   physics, not the soul.
4. **The soul:** testable only with an operational definition: a measurable prediction
   that differs between "soul" and "no soul". Open question Q48 to the owner.

## Open

- Q48: an operational definition of "soul" that yields a measurable prediction.

## Tickets

See `issues/`.
