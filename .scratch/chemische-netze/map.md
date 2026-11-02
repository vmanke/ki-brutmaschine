# Map: chemical networks as generalized neural networks

Goal, set by the owner on 2026-10-31: "We want to ultimately introduce chemical networks as
generalized neural networks."

Research note: `research.md`. Child tickets: `issues/`.

## Notes

- **What the research says.** `research.md` section 2 gives the short answer, with numbered
  sources. ReLU networks and rate-independent chemical reaction networks (CRNs) compute the same
  functions. Mass-action CRNs compute more, and error-free discrete CRNs less. Physics demands
  reversible reactions with energy-derived rate constants. Steady-state outputs of catalytic
  weights and of non-trivial classification need fuel. An output read at a fixed time can run
  on fuel stored in the initial state. Lab systems are small. DNA, enzymatic and self-assembly
  networks take hours. Chemical reservoirs take 30 to 60 min per input. Most lab systems get
  their weights from a computer.
- **Where a chemical network would sit.** `research.md` section 8 sketches a third kind of
  Candidate in the MLP mode, next to MLPs and CNNs (`src/genome.rs:58-65`), under the shared
  `Limits` (`src/config.rs:34-67`). It is a sketch; nothing is decided.
- **Related efforts.** `.scratch/cnn-basismodelle/` builds the seam for a second kind of model:
  the `Arch` field, the shape check, base models, and promotion to `candidates/models/`. The
  draft `.scratch/llm-nutzt-cnn/spec.md` plans an Architect that writes CNN genomes; a
  reaction-network genome could take the same path later. That draft is uncommitted in the
  worktree of branch `claude/llm-nutzt-cnn`. It is not on the branch and not merged, so its
  section and question numbers may change.
- **No ADR conflict found.** A hand-written integrator adds no crate, so invariant I1 of both
  contracts holds (`docs/STAGE1_CONTRACT.md:33-35`, `docs/STAGE2_BACKEND_CONTRACT.md:37-41`).
  ADR 0007 decides candle for CNNs. Its condition for equal bits, that no Candidate hits a
  wall-clock limit, binds every new model kind (ADR 0007; `research.md` section 8.4).
- **Child tickets.**
  - `issues/01-deterministic-crn-simulator-spike.md` (prototype): does a hand-written
    fixed-step integrator give equal bits, a cost known in advance and enough speed?
  - `issues/02-thermodynamically-consistent-genome.md` (research): which genome keeps every
    mutant physical?
  - `issues/03-compile-mlp-into-crn.md` (prototype, blocked by 01): does a compiled ReLU MLP
    compute the same function, and what do finite reverse rates cost?
  - `issues/04-gradients-through-the-integrator.md` (prototype, blocked by 01): can a CRN
    Candidate be trained within the limits, and by which route?
- **Glossary gap.** CONTEXT.md has no term yet for a chemical network, its genome or its parts.
  Proposals for /domain-modeling, ours only: "Reaction network" as the Candidate kind,
  "Chemostat" for a species held at a fixed concentration (the term of `research.md` source 39),
  and "Dual-rail pair" for the two species of one signed value. CONTEXT.md is not changed now.
- **Reading limits.** Some sources were read as abstracts only, and two results only as other
  papers restate them. `research.md` sections 9.4 and 10 list them.

## Decisions so far

- **2026-10-31, owner.** The goal above. No other decision is made yet. Every design choice in
  `research.md` section 8 is a sketch.

## Fog

1. **Semantics first.** Compiled rate-independent CRNs, which match ReLU networks exactly but
   only with idealized irreversible reactions? Mass action with energies and fuels, which is
   rate-dependent and computes more? Or stochastic CRNs, which sample distributions and need a
   stochastic simulator?
2. **How strict the physics is.** Must every reaction be reversible with energy-derived rate
   constants, and every irreversible step be a chemostatted fuel (`research.md` section 6)? Or
   may a Run use idealized irreversible reactions, as the equivalence theorems do? No source of
   the research note says how large a free-energy drop per reaction keeps a compiled network
   within a given error (ticket 03).
3. **Training inside a Candidate.** Evolution only, exact gradients through the integrator, or
   training the equivalent network and compiling it (ticket 04)?
4. **Determinism.** Integer energy genes with one fixed table of rate constants, or continuous
   energies with a pinned `exp`? Would a new crate such as libm be acceptable under invariant I1
   of both contracts (`docs/STAGE1_CONTRACT.md:33-35`, `docs/STAGE2_BACKEND_CONTRACT.md:37-41`)
   and their test (`tests/learner_security.rs:278`)? Admitting candle took an ADR, erratum E15
   and a change of that test (ADR 0007).
5. **What counts as size and cost.** What is a parameter of a CRN, for `max_params` and for the
   fitness penalty (`src/score.rs:301-311`)? How are reactions, molecularity, peak
   concentration, convergence time and dissipation charged? The chemostats' work depends on the
   energy reference; the dissipation does not, but it needs `ln` (`research.md` section 8.3).
   Taken per reaction and step, it conflicts with the rule of no `exp`, `ln` or `powf` in the
   loop (`research.md` sections 7.3 and 8.4) unless `ln` is written by hand. Taken per species
   at the start and at the read-out, it keeps `ln` out of the loop.
6. **Tasks.** The existing 2-D, parity and sine tasks with dual-rail inputs first? The image
   tasks would need 512 input species. Are chemistry-native tasks wanted?
7. **Success.** Which compute-matched baseline and which pre-registered endpoint would show
   that bred chemical networks are worth having? No source of the research note compares evolved
   and gradient-trained CRNs at equal compute.
8. **Lab realism.** Stay with formal CRNs, or restrict the genes to motifs that have been built,
   such as seesaw gates or enzymatic templates, with their measured constraints: rate matching,
   leak and crosstalk?
9. **The Architect.** Should a language model also write reaction-network genomes, and with
   which physics brief? This waits for the CNN Architect.
10. **Beyond the MLP mode.** Could a chemical network later be a Worker or a tool of a Cascade,
    as the CNN draft plans for CNNs?
