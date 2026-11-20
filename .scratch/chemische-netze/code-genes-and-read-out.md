# Code genes and a fitted read-out for reaction networks

Design note of 2026-11-07, for the owner. Nothing is built, and nothing is decided. It adds to
ticket 02 (`issues/02-thermodynamically-consistent-genome.md`). It answers
`../eigene-sprache/issues/03-crn-code-genes-and-read-out.md`, under `../eigene-sprache/spec.md`,
sections 2 and 4.

**Conventions.**

- **Sources.** [R36] is source 36 of `research.md`, section 10. [S1] is source 1 of the list at
  the end of this note.
- **Ours.** "Our reading" and "our calculation" mark our own work; no source states it.
  "Recommendation" marks a design choice that the owner makes.
- **Units.** Concentrations are in units of the one reference concentration c°, which ticket 02
  fixes (`research.md` section 8.2, "One unit").
- **Repository.** File:line refers to commit 26ae4cb. "The draft" is
  `entwurf-rekurrentes-chemisches-netz.md`; "the spec" is `../eigene-sprache/spec.md`.

## 1 Assumptions

- **Semantics first.** Fog 1 of `map.md` stays open: the reaction dynamics come before the
  genome. This note assumes one of its options: mass action, with every output read at one
  fixed time T. T is the end of the integration that ticket 02 fixes.
- **A second choice.** Within these semantics the decoder has a fixed class: a linear map of
  concentrations at T (section 3). The code inside the network is free. `map.md` lists this as
  Fog 11, beside Fog 1 and not instead of it.
- **What the decoder reads.** Our reading: it reads only what is linear in concentrations at T.
  A value in the fractional code of section 2 is a ratio. If the pair's total varies, the
  decoder reads it only approximately.
- **Caps.** Ticket 02 sets the caps on the species count, S_max, and on peak concentration, c_max.
  Its design note is not written yet; its Comments hold only input
  (`issues/02-thermodynamically-consistent-genome.md:67-86`). So both values are open here.

## 2 Input code genes

A code gene says how a network writes one number into concentrations (spec, line 17). Each task
input has one. It names one menu entry and that entry's integer genes. The code's species are
ordinary species of ticket 02, with energies and compositions. Only their initial concentrations
come from the code, so they have no context genes.

- **Scaled input.** Recommendation: the Run fixes one input range [lo, hi] per task; it is not
  a gene. It is [−1, 1] for spirals, checkerboard and parity, and [−π, π] for sine
  (`src/task.rs:571-678`). The code reads s = clamp(2·(u − lo)/(hi − lo) − 1, −1, 1) and
  v = (s + 1)/2, in f64. Clipping touches only noisy points. Spirals and checkerboard add
  Gaussian noise with standard deviations of 0.03 and 0.02 (`src/task.rs:383`, `:395`).
- **Scale.** Every code has an integer scale gene α. Its scale is a = 2^α c°. A step of α doubles
  or halves every concentration of the code exactly: f64 multiplies by a power of two without
  rounding, unless the result under- or overflows. Recommendation: α in −6 … 2, default 0. The
  low end keeps a above √λ = 0.01 c°, the scale below which the read-out damps (section 3.1).

The menu:

- **Dual-rail.** Two species, X+ and X−. The value is [X+] − [X−] (`research.md` section 3.1).
  [X+] = a·(β/4 + max(s, 0)) and [X−] = a·(β/4 + max(−s, 0)). The overlap gene β, in 0 … 8, is
  the level both rails share at s = 0. Our calculation found that the kink error of the ReLU
  module grows with this overlap (`research.md` section 3.7, "Overlap of the rails").
- **Single-rail with an offset.** One species X, with [X] = a·(β/4 + v). The offset gene β, in
  0 … 8, sets the level at the low end of the range. `research.md` section 6.1 says that signed
  values need dual rail [R6; R18]. Our reading: that binds a network that must compute with the
  sign, as a compiled ReLU network does. Here no fixed function is asked of the network, and the
  read-out's bias absorbs the offset. So one species can carry a signed input.
- **Fractional.** Two species, X0 and X1. The value is [X1]/([X0] + [X1]) [S1, abstract]. With
  this code, reaction networks compute any polynomials that map [0, 1] to itself [S1, abstract].
  Here [X1] = a·v and [X0] = a·(1 − v).
- **Population.** K species X1 … XK, with K in 3 … 6. A population code spreads one value over
  several species, each strongest near its own preferred value c_k = −1 + 2·(k − 1)/(K − 1).
  Ours: [Xk] = a·max(0, 1 − |s − c_k|·(K − 1)/2), a hat that falls to 0 at the neighbouring
  preferred values. At most two species are nonzero. Their sum is always a, and
  s = Σ c_k·[Xk]/a. Hats need no exp (`research.md` section 7.3). With K = 2 the hats equal the
  fractional code, so K starts at 3.

| Code | Species | Each species lies in | Largest within the gene ranges |
|---|---|---|---|
| Dual-rail | 2 | [a·β/4, a·(1 + β/4)] | 12 c° |
| Single-rail with an offset | 1 | [a·β/4, a·(1 + β/4)] | 12 c° |
| Fractional | 2 | [0, a]; the two sum to a | 4 c° |
| Population | K, 3 to 6 | [0, a]; the K sum to a | 4 c° |

- At the defaults α = 0 and β = 0, every code stays within [0, 1] c°.
- **Zeros.** Codes can write exact zeros: dual-rail at β = 0 on one rail for every input,
  single-rail at β = 0 at the low end, the fractional code at the ends of the range, and most
  species of a population code. Parity inputs are ±1
  (`src/task.rs:629-656`), so the fractional code writes a 0 for every parity input. A Patankar
  scheme divides by the state (`research.md` section 3.7). The dissipation needs ln of every
  concentration at the start (`research.md` section 8.3). Open for tickets 01 and 02: may a
  code write an exact 0, or does it need a floor?

### 2.1 How the genome check refuses a code

Recommendation: the check of ticket 02 refuses the genome, and never repairs it, if a code

1. names no menu entry, or has a gene outside its range (α in −6 … 2, β in 0 … 8, K in 3 … 6);
2. uses a species that another code uses, or a chemostat, a species held at a fixed
   concentration. A chemostat's level follows from its chemical potential (`research.md`
   section 8.2), so it cannot carry an input;
3. brings the species count above S_max;
4. can write a concentration above c_max. The input is clipped, so the largest concentration of
   the table, a·(1 + β/4) or a, is exact and known before any run.

- If ticket 02 sets c_max at 12 c° or more, the gene ranges alone keep rule 4 from firing.
- Our reading: a code inside the cap cannot break the step rule. The rule holds at the cap
  (`research.md` section 8.2, "Step rule"). A state above the cap ends the Candidate as a
  failure (`research.md` section 8.3; ticket 01).
- A refused child is drawn again, as a CNN child is: up to 16 times, then the next operator
  (ADR 0008:52-59). `research.md` section 8.7 says that this rule fits a reaction-network
  validator.

## 3 Reporter genes and the fitted read-out

- **Reporter genes.** A list of R distinct species, the reporters, read at T. A chemostat cannot
  be a reporter, since its level never changes. A code species may be one; control C1b shows what
  that alone gives. Recommendation: R in 1 … 16. The read-out then has at most 34 parameters
  (section 3.2), far fewer than the smallest training split, 256 inputs for parity
  (`src/task.rs:404`).
- **Read-out.** A logit is the score of one class; the largest names the predicted class. The
  logit of class c is z_c = b_c + Σ_j W_cj·φ_j, where φ_j is the concentration of reporter j at
  T. For sine regression z is the predicted value. There are C logits: 2 for spirals,
  checkerboard and parity, 1 for sine (`src/task.rs:371-420`). W and b are fitted, not bred.
- **Targets.** One-hot for classes: 1 for the true class, 0 for the others. The value for sine.
  Ridge serves the classes too, because its solve has a closed form. So its FLOPs are known
  before the run (`research.md` section 8.3, "FLOPs before the run"). The draft fits its
  read-out the same way (lines 877-879).
- **The fit.** Ridge regression is least squares plus λ times the sum of the squared weights.
  W minimises (1/n)·Σ_s ‖y_s − ȳ − W·(φ_s − φ̄)‖² + λ·Σ W_cj² over the n training inputs, with
  φ̄ and ȳ the means over the training split. Then b = ȳ − W·φ̄, so the bias is not penalised.
  The fit sees only the training split. The validation split scores it, as for an MLP
  (`src/evolve.rs:727`, `src/evolve.rs:741`).

### 3.1 The ridge constant

Recommendation: λ = 1e-4 (c°)², on raw concentrations. Three reasons, all our calculation:

- **One solution.** λ > 0 makes the fit's matrix positive definite, so the fit has one solution
  even when reporters are collinear. Conservation can make them so. A fractional or population
  code always sums to a. If its species share a composition, a closed genome's conserved totals
  are the same for every input. The reporters that make up one total then sum to a constant.
  [R36] used L2 regularisation against the collinearity of its ion signals [R36, SI].
- **A bounded gain.** Each logit's weights have a norm of at most σ_y/(2√λ), with σ_y the
  standard deviation of its target over the training split. For one-hot targets of two balanced
  classes that is 25 per c°. A direction in reporter space with standard deviation σ over the
  training inputs keeps the share σ²/(σ² + λ) of its least-squares weight: 0.99 at σ = 0.1 c°,
  0.5 at 0.01 c° and 0.01 at 0.001 c°. So a contrast far below 1 % of the default code scale
  counts little against stronger ones, and rounding noise is not blown up. This is a soft form
  of the draft's rule that the chemistry must make the contrast itself (lines 759-760). The
  draft instead rounds its read-out to multiples of 2^(−20) c° (lines 400-401).
- **A well-conditioned solve.** Each reporter lies in [0, c_max], so its variance is at most
  c_max²/4. The condition number, the ratio of the largest to the smallest eigenvalue of the
  fit's matrix, is then at most 1 + R·c_max²/(4λ). That is 4e6 at R = 16 and an example c_max of
  10 c°. Our estimate: the relative error of the solve is about that number times 2^(−53), or
  4e-10.

- Accuracy uses only the order of the logits (`src/score.rs:204-211`). So λ acts on accuracy by
  weighting directions against each other, not through the overall scale.
- **Alternative.** Standardise each reporter to unit variance, as [R36] did for its
  classification tasks [R36, Methods], and use a λ without units. The read-out then ignores the
  overall scale. But a reporter that varies only by rounding noise is blown up to unit variance.
- [R36] fitted its prediction and forecasting tasks by ridge regression with a regularisation
  strength of 5e-5, and its classification tasks by a linear support vector classifier [R36,
  Methods; SI]. Its units differ, so that constant does not carry over.

### 3.2 A deterministic fit

Recommendation, all in f64, on one thread per Candidate, as for the integrator (`research.md`
section 8.4):

1. Simulate the training inputs in split order. Read the R reporters at T.
2. Take each mean as one running sum over the inputs in split order, divided once by n.
3. Form G_jk = Σ_s (φ_sj − φ̄_j)·(φ_sk − φ̄_k) for j ≤ k, and H_jc = Σ_s (φ_sj − φ̄_j)·(y_sc − ȳ_c).
   Each is one running sum in split order. Set G_kj = G_jk. Add n·λ to every G_jj.
4. Factor G = L·Lᵀ by Cholesky, with L lower triangular: column by column, each inner sum in
   increasing index, one sqrt per pivot. A pivot is the value whose sqrt a step takes.
5. Solve L·Z = H forward and Lᵀ·Wᵀ = Z backward, each inner sum in increasing index.
6. Set b_c = ȳ_c − Σ_j W_cj·φ̄_j, in increasing j.

- Only +, −, ×, ÷ and sqrt run. Rust gives their IEEE 754 results, sqrt is correctly rounded,
  and the compiler may not fuse a multiply and an add (`research.md` section 7.3). Our reading:
  the fit adds no source of bit differences beyond the simulator's.
- A pivot that is not positive and finite ends the Candidate as a failure, like a NaN
  (ticket 01).
- The logits reach the scorer as f32, as an MLP's outputs do (`src/score.rs:173-174`).
- Recommendation: the model file holds W and b. The hidden evaluation reads them and does not
  refit. The half-step test of section 5 reuses them.
- **Cost.** Our count: about n·R·(R + 2C + 3) + R³/3 + 2·C·R² FLOPs. At n = 1000, R = 16 and
  C = 2 that is 3.7e5. That is about 2e-4 of one simulation pass over checkerboard's training
  split at the rough size of `research.md` section 8.3 (2e9 FLOPs).
- **Parameters.** The read-out has (R + 1)·C parameters: R weights and one bias per logit. They
  count as parameters, as in the draft (lines 903-904).

## 4 Four controls

All four use the same splits and scorer. Where they fit a read-out, they use λ, the targets and
the fit of section 3. C1 and C2 breed nothing. C3 and C4 are arms, bred like the main arm under
the same limits. Seeds and the endpoint are open (`map.md`, Fog 7).

- **C1, the raw inputs.** The read-out on the scaled inputs s of section 2, one per task input,
  read as concentrations in c°. No chemistry runs. Parameters: (n_in + 1)·C. Recommendation, a
  variant C1b: the read-out on the winner's code concentrations at t = 0. The dual-rail split
  max(±s, 0) and the population hats are already nonlinear in s, so a code alone does part of
  the work. C1b measures that part.
- **C2, a fixed random reservoir.** Random genomes, drawn as for a first generation and never
  mutated. Each has the winner's codes, its counts of species, reactions and reporters, and the
  same T. `research.md` section 8.7 proposes this labelled baseline [R36; R37; R38].
  Recommendation: five draws from fixed seeds, each reported, and their median.
- **C3, random codes.** An arm in which each input's code gene is drawn at random once, then
  frozen. Every other gene is bred. It separates having a code from finding one that fits (spec,
  section 2, rule 3). Recommendation: draw among the entries with the species count of the main
  arm's winner, input by input, so that the codes have the same size, as rule 3 asks.
- **C4, the draft's fixed read-out.** An arm without fitted weights. Each logit is the
  difference of one output dual-rail pair at T, as `research.md` sketches it (sections 8.2 and
  8.5). The draft scales it by 8/A (lines 759-760). A is a level of the draft's own core
  (line 221); this genome has no A. For classes the gain changes only the loss, since accuracy
  takes the first index of the largest logit (`src/score.rs:204-211`). For sine the value is the
  difference in c°. No read-out weights count. The spec keeps this arm because a free read-out
  can do the work that the chemistry should do (spec, section 4.2).

## 5 Efficiency

- **Form.** The fitness stays primary − 0.005·log10(P) − 0.002·log10(F) (the draft, lines
  899-902; `src/score.rs:396-415`; defaults at `src/config.rs:122-123`). The primary is the
  validation accuracy, or R² clamped to [0, 1] for sine (`src/score.rs:226`, `:384`).
- **Parameters.** P = P_net + P_code + (R + 1)·C.
  - P_net: the numeric genes of the network. Which genes count is open (`map.md`, Fog 5).
  - P_code, a recommendation: the numeric code genes only. That is 2 for dual-rail and
    single-rail (α and β), and 1 for the fractional and the population code (α). The menu entry
    and K do not count, just as an MLP counts its weights and biases but not its widths or its
    activation (`src/genome.rs:190-204`). The code's species carry their own genes in P_net.
  - The read-out: (R + 1)·C.
- **FLOPs.** F is the simulation of the training split, the fit, and the half-step test, as in
  the draft (lines 906-908). The half-step test reruns fixed inputs at dt/2 and fails a
  Candidate whose outputs change too much (`research.md` section 8.3). The Run books only
  training FLOPs (`src/evolve.rs:791`).
- **Example.** Our numbers, for checkerboard with two dual-rail codes, R = 8, C = 2 and a
  placeholder P_net of 100. Then P = 100 + 4 + 18 = 122, a parameter term of 0.0104 instead of
  0.0100. One simulation pass costs 2e9 FLOPs at the rough size of `research.md` section 8.3.
  With a half-step test on 64 inputs, as in the draft (line 397), and the fit, F = 2.26e9. That
  is a FLOP term of 0.0187, well inside `max_train_flops` of 2e10 (`src/config.rs:101`).
- **What follows.** Doubling P costs 0.0015, and doubling F costs 0.0006. The read-out and code
  genes above cost 0.0004, and the fit's FLOPs 5e-8. So the fitness hardly limits the read-out;
  the cap R ≤ 16 does. Each code pays for its species in P and for their reactions in F (rule 1
  of the spec). Whether that pressure is strong enough is open.
- **Dissipation.** Reported, not charged. It is reported per input at T and as a mean over the
  validation split, in the form that ticket 02 chooses. The draft does the same (line 912). The
  codes set part of the start state, so the dissipation depends on them.
- **A conflict.** Ticket 02 asks to charge the dissipation
  (`issues/02-thermodynamically-consistent-genome.md:33-37`). `research.md` section 8.3,
  "Energy bill", reads the same. This note's ticket and the spec say: report, do not charge. The
  spec adds that charging needs an ADR, "as the research note says" (spec, line 252). That ADR
  appears in the draft's Decision 5 (lines 1309-1311), not in `research.md`. The owner decides.

## 6 The other rules of the spec

- **Rule 2, readable.** Each code is a menu entry with the definition above. The genome holds its
  genes; the model file holds W and b.
- **Rule 4, a switch.** Off, every input is dual-rail at α = 0 and β = 0, and the outputs are
  read as in C4. That is the sketch of `research.md` sections 8.2 and 8.5.
- **Rule 5, scoring.** The logits enter the scorer and the fitness unchanged.

## 7 Open

- The caps S_max and c_max (ticket 02). The gene ranges and R ≤ 16 are recommendations to check
  against them.
- Whether a code may write an exact 0 (tickets 01 and 02).
- What counts in P_net (`map.md`, Fog 5), and whether the menu entry and K should count.
- An absolute λ, as recommended, or a scale-free one.
- The number of C2 draws, the size rule of C3, the seeds and the endpoint (`map.md`, Fog 7).
- Dissipation reported or charged (section 5, "A conflict").
- Mutation operators for code and reporter genes. A new menu entry changes the species count, so
  it must add or drop species; that belongs to the operators of ticket 02.

## Sources

New:

1. [S1] Salehi SA, Parhi KK, Riedel MD (2017). Chemical reaction networks for computing
   polynomials. ACS Synthetic Biology 6(1):76-83. https://doi.org/10.1021/acssynbio.5b00163.
   Peer-reviewed journal. Checked 2026-11-07 at PubMed (PMID 27598466). Read: abstract only.
   PubMed dates the online version 2016-09-21.

Re-checked, for claims that `research.md` does not state:

- [R36] Baltussen MG, de Jong TJ, Duez Q, Robinson WE, Huck WTS (2024). Chemical reservoir
  computation in a self-organizing reaction network. Nature 631(8021):549-555.
  https://doi.org/10.1038/s41586-024-07567-x. Checked 2026-11-07 at nature.com (the article's
  Results and Methods) and in its Supplementary Information PDF at media.springernature.com,
  both read as extracts through the Exa search tool; PMC was not reachable from this session.
  Claims: ridge regression with a regularisation strength of 5e-5 for prediction and
  forecasting; a linear support vector classifier on features scaled to zero mean and unit
  variance for classification; L2 regularisation against the collinearity of some ions.

Used as `research.md` states them: [R6], [R18], [R37], [R38].
