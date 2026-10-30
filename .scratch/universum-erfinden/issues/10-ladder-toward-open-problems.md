# 10 Ladder toward open problems

Status: needs-triage
Type: grilling

The owner asked (2026-10-30) when the machine can be aimed at the Millennium Prize
Problems. This ticket records the honest answer and a ladder of steps. Each step has an
exact checker. Nothing here is decided.

## State of the art (checked 2026-10-30)

- Navier-Stokes. On 2026-09-08 OpenAI announced a proof, produced by an internal system of
  coordinating agents and formalized in Lean, that a fluid at rest with a smooth external
  force can develop a singularity in finite time. That is statement C (and D) of the
  official formulation. OpenAI reports on the order of 10 000 concurrent agents for about
  88 hours, and about 300 billion output tokens across all problems it tried. It does not
  claim the prize. On 2026-09-11 the Clay Mathematics Institute wrote that the problem has
  "apparently been settled". Its rules require publication, a two-year wait and general
  acceptance. Mathematicians quoted by Scientific American (2026-09-21) say the result uses
  an allowed but unnatural external force, and that the unforced problem stays open.
- P vs NP, the Riemann hypothesis, the Hodge conjecture, Birch and Swinnerton-Dyer and
  Yang-Mills remain open. The Poincare conjecture was proved by Perelman.
- Evolution with an exact evaluator has found new mathematics on construction problems:
  FunSearch (Romera-Paredes et al., Nature 625, 2024; larger cap sets) and AlphaEvolve
  (Novikov et al. 2025, arXiv:2506.13131, white paper; 48 multiplications for 4x4 complex
  matrices). Formal proofs at olympiad level: AlphaProof (Hubert et al., Nature, 2025;
  Lean). T-count records on arithmetic circuits: AlphaTensor-Quantum (Ruiz et al., Nature
  Machine Intelligence, 2025).

## Why the machine cannot be aimed at them now

1. Breeding needs a score that can rise step by step. A proof is only right or wrong.
2. The checker must be exact and automatic. For proofs that means a proof checker such as
   Lean. The repo has none.
3. The Workers of step 2 are small open models on one GPU. The pilot cap is 20 000 calls with
   at most 2048 answer tokens each, about 41 million tokens plus thinking tokens. The
   Navier-Stokes run used thousands of times more.
4. An optimizer solves the problem as written, not as meant. The Navier-Stokes result went
   through an allowed loophole, and the DGM paper shows objective hacking. Every target needs
   a statement that leaves no cheap way out.

## Ladder (proposal)

1. Step 1: rediscover known optimal circuits (positive control). In progress.
2. Step 2: the same with real open models on the host.
3. Invention in the circuit Domain: a circuit with a lower T count than a published
   reference, for example a row of Table 1 of AlphaTensor-Quantum, confirmed by the exact
   simulator and a human literature check (Q38).
4. Constructions with exact checkers (Q37 b), for example cap sets or packings.
5. Proofs checked by Lean as a new Domain: known theorems first, then small open statements.
6. Problems of the Millennium kind only after step 5 works, and only with a budget the owner
   approves (Q23: Claude as a Worker only through the Broker, with a euro cap).

## What the machine adds

Not raw power. Its contribution is breeding how a team of Workers is organized: Cascades,
Stages and blind judging, cheaply and on exact checkers. The Navier-Stokes run also relied on
groups of coordinating agents. How such teams should be organized is an open question the
machine can study.

## Comments
