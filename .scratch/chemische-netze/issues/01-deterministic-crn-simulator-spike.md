# 01 Deterministic CRN simulator spike

Type: prototype
Status: needs-triage
Map: `../map.md`
Research: `../research.md`, sections 6, 7.1 to 7.3, 8.3 and 8.4

## Question

Can a small mass-action integrator, written by hand in Rust, give equal bits on one machine at
any thread count, with a cost known before it runs? Is it fast enough for the default limits of
the MLP mode?

## What to do

- Work in a throwaway crate outside the repository, as
  `.scratch/cnn-basismodelle/issues/01-library-spike.md` did. Add no dependency.
- Keep the state in f64. Allow at most two reactants and two products per reaction.
- Derive rate constants from integer energies and barrier excesses (`research.md` section 6.3),
  through a fixed table stored as bit patterns. No `exp`, `ln` or `powf` may run in the loop
  (`research.md` section 7.3; sources 70 and 71).
- Implement three fixed-step schemes: forward Euler, classical RK4, and MPRK43I(1, 1/2) of
  Kopecz and Meister. It is the only MPRK43 scheme with p = q = 1, so its Patankar weights need
  no power (source 54, Eqs. 63b-63c and the text after Eq. 68). MPRK43I(1/2, 3/4) needs only a
  square; most other members need a real power.
- State how each reaction, bimolecular ones included, is split into production and destruction
  terms `p_ij` and `d_ij` (source 54, Defs. 1.1-1.2). Reactions such as `X+ → M + Y+` do not
  conserve the sum of the species, so the conservation property of source 54, Lemma 1.5, does
  not hold for them. The paper assumes a fully conservative system (source 54, Sec. 1).
- Use three test networks:
  - the Robertson network (source 53, Sec. IV), as a stiff probe;
  - the ReLU module `X+ → M + Y+`, `M + X− → Y−` (source 7, Sec. 3.1), whose output must
    converge to `max(x, 0)`;
  - a random closed network of reversible reactions with energy-derived rates (source 39,
    Eq. 50). It must relax to the equilibrium of source 39, Eq. 88, with the f_λ solved from the
    conserved totals of its initial state, and its entropy production (source 39, Eq. 55) must
    fall toward zero.
- Hash every trajectory. Compare two runs in a row, rayon pools of 1 and 4 threads with
  trajectories in parallel, and child processes with `RAYON_NUM_THREADS` at 1 and 3, as the CNN
  tests do (ADR 0007). If a second CPU model is at hand, compare its hashes too.
- Predict the floating-point operations of a trajectory from the network, count them in the
  code, and compare.
- Treat NaN, infinity, a negative concentration and a concentration above a cap as a failure.
  Never hash a NaN (source 70).
- Time one step and one trajectory of 1000 steps, for 10, 50 and 200 reactions, at 1 thread.

## Done when

- A report under Comments gives:
  - the bit-identity result for every setting: equal hashes, or the first difference and its
    cause;
  - the measured time per step and per trajectory;
  - the GFLOP/s at 1 thread for the slowest layouts, and a time model whose estimate bounds
    every measured time, since equal bits need that no Candidate hits `max_candidate_seconds`
    (ADR 0007; `research.md` section 8.4);
  - the predicted against the counted FLOPs per trajectory;
  - per scheme and test network, the largest step that stays finite and positive, and the
    error against a run of the same scheme with a step 1e-3 times as large;
  - whether the closed random network reaches its equilibrium, and how its entropy production
    behaves;
  - a recommendation of one scheme, and of the step rule a genome validator would use.
- If equal bits fail in some setting, the report names the setting and the cause.

## Comments
