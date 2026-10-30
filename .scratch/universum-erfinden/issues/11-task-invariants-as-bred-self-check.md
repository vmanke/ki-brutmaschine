# 11 Task invariants as a bred self-check

Status: needs-triage
Type: research

On 2026-10-30 the owner forwarded a proposal from another AI assistant: "Generalized
Domain-Informed Evolution". It treats logic, mathematics and physics as "invariant fields"
and adds a penalty for every violation to the fitness, in the style of physics-informed
neural networks (PINNs). Most of it does not fit this repo. One kernel is worth a later
experiment.

## What does not fit

- Circuits. Every gate of the set is unitary (`src/cascade/qubit.rs`), so a residual of
  U†U = I is exactly zero for every candidate and separates nothing. The exact check
  already decides right or wrong, and `parse_circuit_answer` rejects a whole answer with a
  malformed line, a qubit out of range or too many gates.
- ARC. "The number of cells stays the same" is not a law of ARC. Many tasks fill, draw,
  crop or scale. A fixed penalty would punish correct answers.
- A weighted penalty, Fitness = Score - alpha * Residual, adds a guessed weight. Even for
  PINNs this is the known weak point: a small weight lets the physics be violated, a large
  one makes the problem hard to optimize (Krishnapriyan et al., NeurIPS 2021). Deb (2000)
  calls the penalty weight the hardest part of the penalty method and replaces it by
  feasibility rules: a feasible candidate always beats an infeasible one. Our ranking
  (correct first, then T count, then CNOT count; decision 2 of ADR 0006) is such a rule.
- The Learner DSL is already closed: 46 commands, no loops, no I/O, a cost known before
  running, and every program is checked before it runs.
- The repo does not name "a search space too vast and unguided" as the cause of the Plan S
  result. `docs/ERGEBNISSE_STUFE1.md` records that the search found only variants of the
  linear seed program, that one seed collapsed, and that the full search space did not
  beat "constants only".
- Hard-coded assumptions block inventions outside them. AlphaTensor-Quantum lowered T
  counts with gadgets that use ancilla qubits (Ruiz et al., Nature Machine Intelligence,
  2025). A rule "no extra qubits" would have forbidden that.

## The kernel

- Some invariants a task reveals in its own examples. For example, if every example keeps
  the grid size, an answer of another size is suspicious. Such an invariant holds for one
  task, never for all.
- Do not hard-code them. Offer them to evolution as phrases in a Worker's prompt sections,
  or through the self-check Trait, so that selection finds out which checks help.
- Test this after step 1 as a pre-registered comparison of universes with and without such
  phrases (Q32), on the grid Domain.

## Sources

- Raissi, Perdikaris & Karniadakis (2019). Physics-informed neural networks. J. Comput.
  Phys. 378:686-707.
- Krishnapriyan, Gholami, Zhe, Kirby & Mahoney (2021). Characterizing possible failure
  modes in physics-informed neural networks. NeurIPS.
- Deb (2000). An efficient constraint handling method for genetic algorithms. Comput.
  Methods Appl. Mech. Eng. 186:311-338.
- Ruiz et al. (2025). Quantum circuit optimization with AlphaTensor. Nature Machine
  Intelligence.

## Comments
