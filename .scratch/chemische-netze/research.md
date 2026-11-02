# Chemical reaction networks as generalized neural networks

Research note of 2026-10-31. Nothing is built. The map of this effort is `map.md`, and its
first tickets are in `issues/`.

**Conventions.**

- **Sources.** A bracket such as [6, Thm. 5.10] names source 6 of section 10 and the place in
  it. Page, section and theorem numbers refer to the version that was read, which section 10
  names.
- **Partly verified.** [n, partly] marks a claim that the source check confirmed only in part.
  The sentence states only the confirmed part. Section 10 says what was not confirmed.
- **Preprints.** Sources 16, 19 and 60 are preprints. Each use says so.
- **Our reading.** A sentence that begins with "Our reading:" is our own inference. No source
  states it in that form.
- **This repository.** Facts about the code cite a file and a line, not a number.

## 1. Question

The owner set the goal on 2026-10-31: "We want to ultimately introduce chemical networks as
generalized neural networks." This note asks six questions.

1. In which precise senses is a chemical reaction network (CRN) a generalized neural network,
   and under which conditions?
2. What is proven about what CRNs compute?
3. What has been built in the lab, at what scale, speed and error?
4. Which physics must a simulated CRN obey?
5. How are CRNs simulated, trained and evolved?
6. How could the Brutmaschine later breed CRNs as a third kind of Candidate, next to MLPs and
   CNNs?

A CRN is a finite set of species and of reactions among them. Under mass-action kinetics it
defines a polynomial ODE on concentrations [43, Sec. 2, Eq. 2; 67, Sec. 2.3]. Under stochastic
kinetics it defines a Markov chain on molecule counts [14, Sec. 2.3], which Gillespie's
algorithm samples exactly [51, Sec. IIIB]. This note treats well-mixed solutions. Non-ideal
and spatial effects are outside the sources read (section 9.4).

## 2. Short answer

- **Equal function classes.** Dual-rail, rate-independent CRNs compute exactly the continuous
  piecewise rational affine functions. This holds when the CRN has a fixed nonzero rational
  initial context (the biases) and its output is read as t → ∞; the constructions use
  irreversible reactions [6, Thm. 5.10, Sec. 6.2]. Feedforward ReLU networks compute exactly
  the continuous piecewise affine functions [74, Thm. 2.1]. Our reading: with rational weights
  and coefficients on both sides, the two classes are the same.
- **A direct translation.** Binary-weight ReLU networks and a syntactic class of CRNs simulate
  each other, one ReLU node per bimolecular reaction [8, "BReLU Networks Simulate CRNs"].
- **More under mass action.** Mass-action CRNs stabilize exactly the algebraic functions of
  inputs that are held fixed [12, Thm. 1]. If the inputs may be consumed, they compute every
  computable function `f: R+ → R+`, with the input set by a polynomial initial condition [10,
  as restated in 12, Def. 1].
- **Less elsewhere.** Error-free discrete CRNs compute only semilinear functions, so not even
  `x1·x2` [5, Thm. 3.4, Sec. 1].
- **Distributions.** Detailed-balanced stochastic CRNs can sample Boltzmann-machine
  distributions exactly, with reactions of high molecularity [14, Thm. 3, Table 1].
- **Lab scale.** DNA networks have classified patterns of up to 144 bits [25, abstract]. One
  winner-take-all network sorted 100-bit patterns into up to nine classes [23, abstract]. A
  two-step DNA hierarchy reached 32 classes [25, abstract]. One DNA network learned two
  memories in vitro, by memorizing and not by optimizing a loss [28, Methods].
- **Lab speed and reuse.** DNA, enzymatic and self-assembly networks answer in hours [22,
  Fig. 3 legend, partly; 26, Ext. Data Fig. 6d,e; 27, "Pattern recognition by nucleation"].
  Freeze-thaw cut a DNA CNN "from hours to minutes" [25, abstract, partly]. Chemical
  reservoirs need 30 to 60 min per input [36, "Nonlinear classification"; 37, Methods].
  Current DNA networks are used once [28, Discussion]. Reuse with more than 200 species has so
  far been shown with a heat-cool cycle [29, abstract].
- **Weights.** Where a source says how the weights were set, a computer trained them or a
  person tuned them, except in that one network [7, Sec. 5; 22, abstract; 26, Ext. Data
  Fig. 6f; 27, Methods; 36, Conclusion; section 5]. A 2010 molecular automaton was instead
  programmed by example [32, abstract].
- **Physics.** Every elementary reaction is reversible, and its rate constants obey local
  detailed balance [39, Sec. I, Eq. 50]. A closed network then relaxes to an equilibrium that
  is unique for given rate constants and conserved totals, which the initial state fixes [39,
  Sec. II D]. Catalysts cannot shift that equilibrium [14, App. 7.1]. So a circuit that biases
  its steady-state outputs by catalysis needs fuel [14, App. 7.1], and so does non-trivial
  steady-state classification in first-order networks [47, "Expressivity requires
  non-equilibrium driving"]. An output read at a fixed time can also be driven by fuel stored
  in the initial state [28, Discussion].
- **Methods.** Exact stochastic simulation exists [51, Sec. IIID]. Gradients that integrate a
  stiff CRN backwards in time failed [53, Sec. III.A]. Differentiating the discrete integrator
  itself gives the exact gradient of the simulated model [57, Sec. IV].
- **For the Brutmaschine.** Our reading: CRNs could become a third kind of Candidate. Its genome
  would hold species energies, barriers, reactions and fuels, chemostatted or stored, so that
  local detailed balance holds by construction. A hand-written fixed-step integrator with only
  exactly rounded operations would keep the cost known before the run. It would keep the bits
  equal on one machine if no wall-clock limit stops a Candidate (section 8.4). No source here
  compares evolution with gradient training of CRNs at equal compute.

## 3. What "generalized neural network" means here

The sources give five precise senses. Each holds only under its conditions.

| Sense | Network side | CRN side | Exact or approximate | Sources |
|---|---|---|---|---|
| 3.1 Same functions | feedforward ReLU, rational weights | dual-rail, rate-independent | exact as t → ∞ | [6; 7; 8; 74] |
| 3.2 Steady state | feedforward, smoothed ReLU | mass action, catalytic weights | exact as t → ∞ | [9] |
| 3.3 Dynamics | continuous-time recurrent network | recurrent neural CRN | approximate, finite horizon | [17] |
| 3.4 Distributions | Boltzmann machine | stochastic CRN | exact or approximate | [14; 15] |
| 3.5 Rate law | one hidden layer, exponential activation | mass-action rate law | an identity | [59] |

### 3.1 ReLU networks and rate-independent CRNs compute the same functions

- **Statement.** With dual-rail values `x = [X+] − [X−]`, a function `R^k → R` is stably
  computable by a CRN if and only if it is continuous piecewise rational linear [6, Thm. 5.10,
  p. 31]. Feedforward, output-oblivious CRNs suffice [6, Thm. 5.10].
- **With an initial context.** With rational initial concentrations of the non-input species,
  exactly the continuous piecewise rational affine functions are computable [6, Sec. 6.2,
  p. 45]. [8, "Computational Power"] asserts that ReLU networks compute exactly this class. The
  theorem, for real weights, is [74, Thm. 2.1]. Chen et al. state, citing [8], that such CRNs
  with a fixed non-zero initial context "are equivalent to ReLU neural networks" [6, Sec. 1.1,
  p. 5].
- **ReLU module.** `X+ → M + Y+` and `M + X− → Y−` drive `y+ − y−` to `max(x, 0)` for any
  nonzero rate constants [7, Sec. 3.1, reactions (1)-(2)]. The module never consumes its
  output, so modules compose [7, p. 4]. Citing earlier work, the authors state that no
  composable CRN with fewer than 2 reactions and 5 species computes ReLU [7, p. 4, fn. 3].
- **Weights and biases.** Rational weights use chains that double and halve concentrations.
  The number of reactions grows with the bits of precision. Biases are initial concentrations
  [8, "Rational Weight ReLU Neural Networks"].
- **Binary weights.** Binary-weight ReLU networks and "CheLU" CRNs simulate each other, one
  ReLU node per bimolecular reaction [8, "BReLU Networks Simulate CRNs", Fig. 7]. A CheLU CRN
  has at most two reactants per reaction, is feed-forward, uses each species at most once per
  reaction with unit stoichiometry, and is noncompetitive [8, same section].
- **Condition: dual rail.** Concentrations are nonnegative, so a signed value is a difference of
  two species [6, Thm. 5.10; 7, Sec. 2.1]. The representation is not unique, and the choice
  changes the convergence time [7, Sec. 2.1].
- **Condition: read at convergence.** A CRN is noncompetitive if every species that a reaction
  decreases is a reactant of that reaction only [8, Def. 4]. If a line-segment path leads to a
  static state, a noncompetitive CRN converges to that state under every rate law that is fair
  and respects stoichiometry, mass action included [8, Theorem].
- **Condition: feedforward.** A feedforward CRN that stably computes f converges under mass
  action for every choice of positive rate constants, even after an adversary steers it [6,
  Cor. 4.11]. Without feedforward wiring this can fail. `X + X → Y + Y` with `Y + X → X + X`
  stably computes `f(x) = x`, but under mass action it settles in a dynamic equilibrium [6,
  p. 22].
- **Condition: irreversible reactions.** The theorems treat reactions as irreversible. No
  reaction is truly irreversible; strand displacement is effectively irreversible because base
  pairing releases free energy [7, fn. 2].
- **Condition: offline training.** The networks are trained in silico and then compiled [7,
  Sec. 5; 8, "Related Work"].
- **Open.** Whether feedback ReLU networks and feedback CRNs are equivalent is open [8, "BReLU
  Networks Simulate CRNs"].

### 3.2 Smoothed-ReLU networks are steady states of mass-action CRNs

- **Activation.** `φ(y) = (y + sqrt(y² + 4h))/2` with `h ≥ 0`. At `h = 0` it is the ReLU [9,
  Eq. 4, Fig. 4].
- **Statement.** The ODE `dx/dt = h + (W·x_prev + β)_i·x − x²` with `h > 0` implements a
  feedforward network with activation φ [9, Prop. 4.9]. "Implements" means that the
  activations of every layer are the limits of the concentrations, for any nonnegative start
  of the non-input layers [9, Def. 4.1].
- **Convergence.** The system converges from infinity in finite time and is exponentially
  reliable [9, Prop. 4.9]. Theorem 4.8 gives general sufficient conditions for other
  activation systems [9, Thm. 4.8].
- **Reactions.** Zero-order production `H → H + X`, the trimolecular weight reactions
  `X′ + W+ + X → X′ + W+ + 2X` and `X′ + W− + X → X′ + W−`, bias reactions with `B+` and `B−`,
  and `2X → X` [9, Table 1]. A weight is a difference of two catalyst concentrations,
  `w+ − w−` [9, Table 1].
- **Variants.** With `3X → X` the activation grows like `sqrt(y/2)`. With `qX → X` it grows
  like `y^(1/(q−1))` [9, Sec. 6.1].
- **Conditions.** Inputs must be nonnegative [9, Eq. 11, Def. 4.1]. The construction needs
  trimolecular reactions and zero-order production [9, Table 1].
- **Training.** In-silico gradient descent on MNIST (784-40-10) reached just over 95 % on
  training batches with `h = 1` and about 88 % with `h = 0` [9, Sec. 6]. These are batch
  accuracies, not held-out ones, and the paper says it is not about training [9, Sec. 6].
- **Same steady state elsewhere.** Dack et al. use it as a "chemical perceptron" with a
  globally stable equilibrium [17, Eqs. 3-4].

### 3.3 Recurrent neural CRNs approximate dynamics

- **Statement.** For a Lipschitz target vector field on a compact set in the positive orthant,
  recurrent neural CRNs follow its trajectories on `[0, T]` with error at most `c1·ε + c2·μ`
  [17, Thm. A.1].
- **Conditions.** The network needs enough perceptron species, and a perceptron time scale μ
  below a threshold `μ0` [17, Thm. A.1]. The bound covers a finite horizon only [17,
  Thm. A.1]. Targets outside the positive orthant need an affine shift [17, Sec. 3].
- **Reactions.** Zero-order production and uni- and bimolecular catalytic reactions [17,
  Eq. 14].
- **Training.** Parameters come from in-silico backpropagation [17, Algorithm 1]. A DNA
  version would need its fuel replenished, for example by a chemostat; the simulations hold the
  fuel constant [17, Sec. 4.1].

### 3.4 Stochastic CRNs sample Boltzmann-machine distributions

- **Detailed balance.** If each species has an energy and `k_r+/k_r− = exp(−ΔG_r)`, the
  stationary distribution on each reachability class is a product of Poissons restricted to
  that class [14, Sec. 2.3, Eqs. 11-13].
- **Direct construction.** It is exact. It needs fuel, a number of reactions exponential in the
  node degree, and a molecularity that grows with the degree [14, Sec. 3.2, Table 1].
- **Edge-species construction.** It is detailed balanced, needs no fuel and is exact. Its
  molecularity reaches `2d + 1`, with `O(N·2^(d+1))` reactions [14, Thm. 3, Table 1].
- **Taylor-series construction.** It uses only uni- and bimolecular reactions. It is not
  detailed balanced, and it is approximate [14, Sec. 4.1].
- **Any distribution.** Marginals of detailed-balanced CRNs approximate any distribution on
  `N^d` in sup norm, from a specific initial condition [15, Thm. 3, Def. 2, Sec. 3.1]. Robust
  (ergodic) CRNs, whose limit does not depend on the start, are universal as well [15, Thm. 6].
  A CRN that is both detailed balanced and robust has only product-Poisson limits [15,
  Sec. 3.1].
- **The word itself.** A preprint calls detailed-balanced CRNs "generalized Boltzmann machines"
  in its title [16, preprint].

### 3.5 The mass-action rate law is a neuron

- A mass-action rate `exp(ln k + Σ ν_i·ln[X_i])` is a neuron. Its inputs are log
  concentrations, its input weights are reaction orders, its bias is `ln k`, and its output
  weights are stoichiometric coefficients [59, Sec. 2, Eqs. 1-5].
- A reaction network is then one hidden layer with one node per reaction [59, Fig. 1]. It
  trains as a neural ODE [59, Sec. 2, Eqs. 6-8].

### 3.6 A chemical threshold neuron (1991)

- A reversible mechanism with a single stationary state, whose concentrations are high or low
  depending on a catalyst, has the properties of a McCulloch-Pitts neuron [1, abstract].
- Connected chemical neurons in one homogeneous solution form logic gates such as AND and NOR
  [1, abstract]. They compute either by relaxing to a stationary state or in time steps set by
  an oscillatory reaction [1, abstract].

**Our reading:** "generalized" holds for mass-action CRNs as a class. The class contains CRNs
that compute what ReLU networks compute [6; 8], and CRNs that compute products and other
algebraic functions [12, Thm. 1]. It does not hold for rate-independent or for error-free
discrete CRNs, which cannot multiply (section 4.5). A breeder must therefore fix the semantics
before it fixes a genome.

## 4. Proven results

### 4.1 Discrete counts, no error allowed

- Two-way population protocols stably compute exactly the semilinear predicates, which are the
  sets definable in Presburger arithmetic [4, Cor. 7, Sec. 6.1]. These protocols are the CRNs
  whose reactions all have exactly two reactants and two products [5, Sec. 2.2].
- The upper bound is proved for rules that replace one finite multiset by another, including
  rules that add or delete agents or involve more than two of them [4, Sec. 16]. Our reading,
  the standard one: it covers error-free computation by general CRNs.
- Semilinearity belongs to stable computation. Mere reachability sets are not semilinear in
  general [4, Sec. 1.1].
- A function `N^k → N^l` is stably computable if and only if its graph is semilinear [5,
  Thm. 3.4]. So `x1·x2`, `x²` and `2^x` are not computable this way [5, Sec. 1, p. 5].
- Every semilinear function is stably computable in expected time `O(log^5 n)` for n input
  molecules [5, Thm. 4.6]. The construction runs a fast error-prone computation beside a slow
  error-free one [5, Thm. 4.6].
- The systematic construction needs single-copy "leader" species in its initial context.
  Whether they can be dropped was left open [5, Sec. 5].
- Doty and Hajiaghayi removed the leaders. Every semilinear function f with `f(0) = 0` is
  stably computable from an initial state that holds only the input species, in expected time
  `O(n)` [73, Thm. 3.4].
- A preprint shows that semilinear predicates and functions stay exactly computable when
  reactions may also run backwards for a while [19, Thms. 6 and 10, preprint]. It reuses the
  leaderless construction of [73] and states that no initial leader is needed [19, Lemma 11,
  Thm. 13, Conclusion, preprint].

### 4.2 Discrete counts, small error allowed

- With any nonzero error probability allowed, stochastic CRNs with a fixed set of species are
  Turing universal. Error-free Turing-universal computation is impossible [3, Sec. 1, p. 2].
- Whether a state can be reached is decidable. Whether it is likely to be reached is
  uncomputable in general [3, Sec. 1, p. 2].
- Whether some state `y′ ≥ y` can be reached from x is decidable by the Karp-Miller
  construction. Hence no CRN simulates Turing machines without error [3, Thm. 4.2].
- For any Turing machine there is one CRN in which an initial amount of an "accuracy" species
  bounds the cumulative error by δ, over unboundedly many steps and unbounded space [3,
  Thm. 4.1]. Its expected time, given a correct run, is `O((1/δ + s0 + t·s)^5 · t·s / k)`, a
  12th-order polynomial slowdown for `s ~ t` [3, Thm. 4.1]. In this model the volume grows with
  the total molecule count [3, Thm. 4.1].
- Bounded computations are cheaper. A register machine runs with error at most δ in expected
  time `O(v·t²/(k·δ))` [3, Thm. 3.1]. A Turing machine runs with at most a 6th-order slowdown
  [3, Thm. 3.2, p. 10].
- The constructions use only bimolecular reactions with one rate constant k. Start-state
  species have one molecule, and a dummy catalyst has count 1 [3, Sec. 3, p. 5, Fig. 3.1].
- The error is set by the count of an "accuracy" species, not by rate constants or volume [3,
  Sec. 3, p. 4].
- Information sits in unary counts, so the total molecule count grows exponentially with the
  memory used [3, Sec. 1, p. 2]. Finite density then forces the volume to grow at least in
  proportion to the largest count [3, Sec. 3, p. 4].
- Earlier work proposed chemical finite-state machines and stack memories, with which a
  universal Turing machine is possible in principle [1, abstract; 2, abstract]. These networks
  inherit the halting problem [2, abstract].

### 4.3 Concentrations, rate-independent

Section 3.1 holds the main theorem. Further results:

- **Direct encoding.** With nonnegative values, exactly the positive-continuous piecewise
  rational linear functions are computable [6, Thm. 5.9]. A jump is allowed only where an input
  switches from zero to positive [6, Def. 5.8]. Feedforward CRNs suffice [6, Thm. 5.9].
- **Composition.** Dual rail lets a CRN never consume its output. CRNs then compose by
  concatenation and renaming, without loss of power [6, p. 4, Sec. 5.1]. With the direct
  encoding, never consuming the output limits the power to superadditive functions [6,
  Sec. 5.1].
- **Construction.** Every continuous piecewise affine function is a max of mins of its affine
  pieces (Ovchinnikov) [6, Thm. 5.11]. Modules for linear maps, min, max and fan-out assemble it
  [6, Lemmas 5.12-5.15, Cor. 5.14].
- **max.** A max CRN converges under mass action to `max([X1]0, [X2]0)` for any nonzero rate
  constants [5, Sec. 5].
- **Size.** After optimization, a binary-weight network needs one bimolecular reaction per ReLU
  node and two unimolecular reactions per input [7, p. 6; 8, "Translation Optimization"]. Some
  adversarial networks give reactions with a number of products exponential in the depth [7,
  p. 6; 8, "Translation Optimization"].
- **Concentrations.** Without cancellation reactions such as `S+ + S− → W`, dual-rail
  concentrations may grow exponentially with the number of layers [7, Sec. 6; 8, "Rational
  Weight ReLU Neural Networks"].
- **Time.** There is no formal convergence-time result [7, Sec. 6]. In simulations the 90 %
  completion time grew sublinearly, linearly or superlinearly with depth, by architecture [7,
  Sec. 6]. In a later study the time to 90 % of the output grew linearly with depth, and the
  stabilization time superlinearly [8, "Convergence Time Analysis"]. For one network, a 2-D
  heart pattern, stabilization took 0 to 12 time units across inputs [8, Fig. 9].
- **Simulated agreement.** Networks trained with BinaryConnect on Iris, MNIST and a virus
  gene-expression set were compiled and simulated under mass action, with all rate constants 1
  [7, Sec. 4]. MNIST at 14 × 14 with two hidden layers of 512 units reached 97.76 % test
  accuracy and needed 4488 reactions, or 1024 after optimization [7, Sec. 4]. The CRN outputs
  matched the network outputs in all tested examples [7, Sec. 4; 8, "Simulations"]. The later
  study's largest network was MNIST with 64 ReLU units [8, "Simulations"].

### 4.4 Concentrations under mass action

- **Turing completeness, as restated.** Every computable function `f: R+ → R+` can be
  computed by a mass-action CRN over finitely many species, with at most two reactants per
  reaction [10, as restated in 11, Sec. 1, p. 2, and 12, Def. 1]. The input enters through a
  polynomial initial condition `y(0) = q(x)` [10, as restated in 12, Def. 1]. "Computed" means
  that an output species `y1` satisfies `|y1(t) − f(x)| ≤ y2(t)` for `t > 1`, where an error
  species `y2` decreases to 0 [10, as restated in 12, Def. 1]. The primary text of [10] was not
  read.
- **Proof route, as restated.** It combines Turing completeness of polynomial initial value
  problems, dual-rail encoding of reals, and quadratization down to bimolecular reactions,
  implemented in BIOCHAM-4 [10, as restated in 11, Sec. 1].
- **Inputs are consumed.** The input is a decaying species that halts the generating ODE [11,
  Sec. 2.1]. A compiled CRN must be re-initialized for each computation, and online synthesis
  is called a harder problem [11, Sec. 5-6].
- **Sizes.** The logistic function compiles to 3 species and 5 reactions [11, Table 1]. A Hill
  function of order 5 compiles to 7 species and 11 reactions, or to 8 and 16 with a faster
  heuristic [11, Table 1]. Minimizing the dimension of the quadratization is NP-hard [11,
  Sec. 4.2].
- **Analytic only.** A polynomial ODE generates only analytic functions. A compiled solution is
  valid up to the first non-analytic point [11, Sec. 3.2].
- **Online, inputs pinned.** The functions that a mass-action CRN stabilizes are exactly the
  real algebraic functions [12, Thm. 1]. Inputs are pinned, and the output must converge to
  `f(X)` from every start in a domain of full dimension [12, Def. 3]. Pinning can come from
  separated concentration or volume scales, or from a mechanism that holds the input fixed [12,
  Def. 2]. Negative values use dual rail with fast mutual annihilation [12, Lemma 1,
  Eqs. 12-13].
- **Example.** A Hill function of order 5 is stabilized online with 6 species and 10
  mass-action reactions [12, Example 7]. The output follows inputs that change more slowly than
  a relaxation time τ [12, Sec. 2.2]. The authors remark, without proof, that faster adaptation
  usually costs more energy [12, Sec. 2.2].
- **Complexity.** A real function on an interval `[a, b]` is polynomial-time computable if and
  only if a polynomial ODE reaches precision `e^−μ` along a solution curve whose length is
  polynomial in the input size and μ [13, Thm. 1, Def. 3]. P equals the analog-recognizable
  languages [13, Thm. 2]. ODE time is not a robust cost measure because time can be
  contracted; curve length is [13, Sec. 3].
- **Online inputs.** Polynomial ODEs fed with an input that stays nearly constant on an
  interval converge, after a delay, to f of that input [13, Lemma 13].

### 4.5 What these results exclude

- **Winner-take-all.** Our reading: an indicator that is 1 for the largest input jumps where two
  positive inputs are equal. Theorems 5.9 and 5.10 of [6] allow a jump at most where an input
  leaves zero, so no rate-independent CRN computes it. The max itself is continuous and
  computable [5, Sec. 5; 6, Cor. 5.14]. Winner-take-all therefore needs rate-dependent
  kinetics. The DNA winner-take-all networks used gates whose rate constants lay within ±5 % of
  their median [23, Ext. Data Fig. 6b,c], and imperfect rates bias the competition [28, "Testing
  after training"].
- **Products.** Our reading: `x1·x2` is not piecewise linear, so no rate-independent CRN
  computes it [6, Thm. 5.10]. Error-free discrete CRNs cannot either [5, Sec. 1]. Under mass
  action, `X1 + X2 → X1 + X2 + Y` with `Y → ∅` has the steady state `y* = (k1/k2)·x1·x2`, which
  depends on the rate constants.
- **Sigmoids.** Our reading: the logistic function and tanh are not algebraic, so no mass-action
  CRN stabilizes them online in the sense of [12, Thm. 1]. The logistic function can still be
  compiled for one use with consumed inputs [11, Table 1]. Hill functions are rational, and the
  smoothed ReLU solves `φ² − y·φ − h = 0`, so both are algebraic. This fits their online and
  steady-state constructions [9, Prop. 4.9; 12, Example 7].
- **The repository's activations.** Our reading: of the four MLP activations
  (`src/genome.rs:42-47`), only ReLU is piecewise linear. So only ReLU networks have an exact
  rate-independent counterpart [6, Thm. 5.10].

### 4.6 Stochastic distributions

- **Clamping.** Clamping species of a detailed-balanced CRN yields exactly the conditional
  distribution, if clamping preserves reachability [14, Thm. 1].
- **Independence.** A reachability class of product form gives independent species [14,
  Thm. 2]. Non-trivial correlations need driving out of equilibrium, or conservation laws that
  couple the state spaces [14, discussion after Thm. 2].
- **Complex balance.** If the deterministic model is complex balanced with equilibrium c, the
  stochastic model has the product-form stationary distribution `∏ c_i^{x_i}/x_i!` on each
  closed irreducible class, restricted to the class and renormalized [21, Thm. 4.1]. Only if
  the whole lattice is irreducible is it a product of Poissons with means `c_i` [21,
  Thm. 4.1]. The product form holds for every choice of rate constants if the network is weakly
  reversible with deficiency zero [21, Thm. 4.2].
- **Restriction is needed.** If the whole nonnegative integer lattice is reachable, the
  equilibrium is a product of independent Poissons. A distribution far from that needs a
  restricted reachability class [16, Sec. 2, Eqs. 12-13, preprint]. Hidden species add power
  only if their reachable states are entangled with those of the visible species [16, Sec. 2,
  Eq. 17, preprint].
- **Size.** Universal constructions use high-molecularity reactions such as `17V → 16V`, and
  their size grows with the support of the target distribution [15, Sec. 1]. A bimolecular
  detailed-balanced version exists for connected supports [15, Construction 2].
- **Learning rule.** A gradient rule on species energies minimizes the KL divergence, using mean
  counts under free and clamped dynamics [14, Sec. 5, Eq. 24]. It ran in silico [14, Sec. 5].
- **MNIST.** A Boltzmann machine trained by stochastic gradient descent on 10 × 10 MNIST was
  converted to a CRN of 300 species, 300 unimolecular and 22 350 bimolecular reactions, and
  simulated by the stochastic simulation algorithm [14, App. 7.2].

### 4.7 Counts and concentrations

- With molecule numbers in proportion to the volume V, the rescaled stochastic model converges
  to the mass-action ODE over compact time intervals as V grows [20, partly]. Chen et al. add
  that this holds with high probability and is guaranteed only for times `O(log n)` in the
  scaling factor n [6, fn. 4, p. 5]. They give CRNs whose two models diverge after that [6,
  fn. 4].
- Stochastic rate constants scale with the volume as `κ_k = κ̂_k / V^(|ν_k| − 1)` [21, Sec. 4.1,
  Eq. 4.5].
- Except for simple linear systems, mean populations satisfy no closed set of rate equations
  [51, Sec. I].
- The ReLU compilation also applies to discrete counts "as long as real-valued signals are
  properly discretized" [7, fn. 1].

## 5. Built in the lab

Laboratory implementations of arbitrary CRNs by DNA strand displacement have been limited to
small systems [8, Conclusion]. The subsections give scale, speed, error and the limits the
authors state.

### 5.1 DNA strand displacement

- **Seesaw linear-threshold circuits (2011).** Linear threshold circuits were transformed
  systematically into DNA strand-displacement cascades [22, abstract].
  - A 3-input, 4-output threshold gate used 60 strands forming 38 initial species, at
    1x = 16.67 nM and 20 °C, with inputs at 0.1x for off and 0.9x for on [22, Fig. 1 legend,
    partly].
  - A 3-bit XOR circuit used 68 strands forming 42 initial species, at 1x = 50 nM and 25 °C
    [22, Fig. 2 legend, partly].
  - A Hopfield memory of four fully connected neurons used 112 strands forming 72 initial
    species at 1x = 25 nM [22, Fig. 3 legend, partly]. Its outputs were read at 8 h [22, Fig. 3
    legend, partly].
  - The memory was trained in silico. It recalls the most similar of four stored patterns from
    an incomplete one [22, abstract].
  - Each neuron reads out one of four values, among them "unknown" and "invalid" [22, Fig. 3e
    legend, partly].
- **Winner-take-all networks (2018).** Seesaw gates gained a component for cooperative
  hybridization [23, abstract].
  - Scale: 10 × 10 patterns, each made of 20 of 100 distinct molecules, sorted into up to nine
    classes, the handwritten digits 1 to 9 [23, abstract].
  - Error: test patterns with up to 30 of 100 bits flipped were classified correctly [23,
    abstract].
  - Size: `n + 3m` nodes for m memories of n bits [23, Ext. Data Fig. 2b].
  - Rate matching: 15 candidate catalytic gates lay within ±65 % of their median rate
    constant. Two groups of three within ±5 % were used in the networks [23, Ext. Data
    Fig. 6b,c].
  - Earlier DNA networks recognized no more than four patterns of four molecules each [23,
    abstract, partly].
  - The authors state that winner-take-all circuits are computationally more powerful than
    linear-threshold circuits and Hopfield networks, and simpler to build [23, abstract].
  - Lateral inhibition and competition replace negative weights, which removes the need for
    dual rail [24, abstract].
- **Loser-take-all (2021).** A three-input circuit was shown on nine input combinations,
  cascadable without dual rail [24, abstract]. The authors compensated signal loss and rate
  differences [24, abstract].
- **Convolutional network (2022).** DNA switching gates perform weight-sharing
  multiply-accumulate operations on 144-bit inputs and recognize up to eight classes [25,
  abstract]. Linked to other DNA circuits, a two-step hierarchy reaches 32 classes [25,
  abstract]. Cyclic freeze-thaw reduced the computation time "from hours to minutes" [25,
  abstract, partly].
- **Supervised learning in vitro (2025).**
  - Training patterns and class labels write weights into activator concentrations. The
    activators later move to weight gates that classify test patterns [28, abstract, "Concept
    and design"].
  - Three training sets of 100-bit MNIST digit pairs (0/1, 3/4, 6/7) were learned [28, "Learned
    weights", Fig. 4].
  - Scale: over 700 distinct species in one tube, and more than 1200 unique strands across
    learning and testing [28, "Testing after training"]. Up to 80 % of activators and weights
    must stay inhibited after training [28, same].
  - Result: correct classification in 72 representative tests after 3 training processes [28,
    same].
  - Speed: the weight output reached the activator level within 2 h, and exceeded the input more
    than 4-fold in 20 h; 1x = 50 nM [28, "Motif characterization", Fig. 2].
  - Protocol: each class was presented for 24 h, then a label inhibitor for 2 h [28, Fig. 4
    legend]. Learned weights were read over 8 h and quantified at 4 h [28, Fig. 4 legend].
  - Errors: pixels that were 0 in all training patterns gave 0.5 ± 0.3 % and 1.1 ± 0.4 % of the
    total signal, against up to 13-fold increases elsewhere [28, "Learned weights"]. Bits 27, 85
    and 86 stayed low because of malformed intermediate structures [28, Methods].
  - Crosstalk: matched pairs gave at least 88 % of the target signal, and 306 mismatched pairs
    at most 20 % [28, "Motif characterization"]. These tests ran without competition, one pair
    per tube [28, same].
  - Accuracy: without a molecular mechanism for selecting bits, accuracy falls below that of
    in-silico weights [28, "Activatable memories"]. Training patterns were filtered by a "good
    teacher" rule, and test patterns within 20 % of the decision diagonal were left out [28,
    same]. The expected lower bounds of accuracy on full MNIST classes were 46 % to 71 % [28,
    same]. These bounds come from that margin rule, not from measured errors [28, same].
  - Stated limits: only four of seven target features are met; accuracy, reusability and
    flexibility are not fully met [28, Introduction]. Performance falls as the ratio of unused
    to used memory bits grows [28, "Testing after training"]. More than two memories bring a quadratic
    number of annihilators and biased competition [28, same]. Computation consumes stored
    energy, so the network is used once [28, Discussion]. The rule is "learning by memorizing",
    not loss optimization, and it does not extend naturally to deeper networks [28, Methods].
- **Heat-rechargeable networks (2025).**
  - Heating and cooling return enzyme-free DNA circuits to kinetically trapped states that
    store energy for the next computation [29, abstract].
  - Circuits of more than 200 distinct species recharged within minutes, for at least 16 rounds
    with varying inputs [29, abstract].
  - A 100-bit, two-memory winner-take-all network used up to 289 strands, 213 of them in one
    tube [29, "Scalability"]. Ten test patterns were added in sequence to the same tube [29,
    Fig. 4a].
  - Reset: inhibitors equal to the previous inputs are added, the tube is heated to 95 °C for
    5 min, cooled to 20 °C in 1 min and held there 3 min [29, Methods].
  - Simulations predicted reset success of 93 % for annihilators and 85 % for summation gates
    [29, "Scalability"]. A reusable catalyst kept reset success above 90 %, even with cooling
    slower than 75 min [29, "A reusable catalyst"]. Its circuit worked largely as before after
    100 resets, which used one input and heating only [29, same]. DNA degradation at high
    temperature was the main limit [29, same].
  - Speed: the catalyst gave 10-fold amplification within 2 h and an 82-fold turnover within
    24 h for 1 nM input [29, same].
  - A circuit for the first 16 elements of the Fibonacci word ran 16 rounds with 15 resets over
    640 h [29, "Versatility"]. It has three layers by design, seven when its input wastes count
    as layer 0 [29, same].
  - Stated limits: input changes alone cannot update a decision, and outputs cannot revert
    without a heat-cool cycle [29, "Scalability"]. Resets are complete, so unsupervised
    learning would need selective reset [29, Discussion]. Earlier reusable enzyme-free circuits
    had fewer than a dozen species [29, Introduction].
- **Smaller DNA networks.**
  - A noncomplementary 4-bit DNA perceptron with removable lipid-oligonucleotide inputs ran
    several computing cycles [30, abstract]. The authors call it the first reusable DNA-based
    neural network [30, abstract].
  - An early classifier encoded weights as mixing ratios of labelled probe molecules in
    competitive hybridization [31, abstract].
  - Reconfigurable nucleic-acid catalyst units formed an automaton that covers all responses to
    two sets of four inputs and can be "programmed by example" [32, abstract].
  - A DNA classifier trained in silico on miRNA profiles diagnosed lung cancer in serum from 22
    people (8 healthy, 14 with cancer) at 86.4 % accuracy [33, abstract]. Our reading: that is
    19 of 22 samples.

### 5.2 Enzymatic and cell-free networks

- **Enzymatic neurons (2022).** DNA-encoded enzymatic neurons have tunable weights and biases
  [26, abstract].
  - One neuron computed majority functions on 10 bits [26, abstract].
  - A two-layer network formed rectangular functions of a microRNA input [26, abstract].
  - A hybrid circuit partitioned a concentration plane by a decision tree in cell-sized
    droplets [26, abstract].
  - Motivation: in non-enzymatic networks, leakage, sensitivity, speed, preparation and weak
    nonlinearity had made the composition of layers delicate [26, abstract].
  - Rates: at equal converter-template concentrations, production rates were 39 ± 18 pM/min,
    with a factor 4.6 between inputs [26, Ext. Data Fig. 6c]. Balancing the templates cut the
    coefficient of variation from 47 % to 14 % [26, same].
  - Time: single inputs of 5 nM triggered amplification between 112 and 148 min
    (123 ± 11 min), over 28 samples with 0 to 10 inputs [26, Ext. Data Fig. 6d,e].
  - Tuning: the decision threshold moved with the template concentration [26, Ext. Data
    Fig. 6f]. At 0.04x nothing amplified within 1000 min, and at 0.15x samples with 4 and 6
    inputs separated clearly [26, same].
  - Enzymes: 70 U/mL Vent(exo-), 300 U/mL Nb.BsmI and 23 nM ttRecJ [26, Ext. Data Fig. 6d,
    partly].
- **Metabolic perceptrons (2019).** Enzymes convert five input metabolites into benzoate, whose
  sigmoidal BenR/pBen response is the perceptron's nonlinearity [34, Results, Fig. 3]. Weights
  are enzyme DNA concentrations in a cell-free system [34, same].
  - Two four-input classifiers ran with model-predicted weights and inputs at 100 µM [34,
    "Cell-free perceptron for binary classifications"].
  - Whole-cell circuits were read after 18 h at 37 °C [34, Methods, partly].
  - Stated limits: one layer and positive weights only [34, Discussion]. One weight (HipO,
    0.03 nM DNA) had to be found by experiment, because the model's prediction gave too high a
    signal [34, "Cell-free perceptron for binary classifications"].
- **Riboregulator perceptron (2022).** A cell-free perceptron sums three toehold-switch inputs at
  the RNA level [35, "Direct-Expression WSO", partly].
  - Weights are DNA template concentrations: 0.6, 0.6 and 0.8 nM in the direct-expression sum,
    and 1, 5 and 7.5 nM in the coupled network, with inputs at 5 nM [35, same, partly].
  - The threshold comes from titrating anti-σ28 against σ28 [35, "Genetically Implemented
    Perceptron", partly]. Batch reactions ran at 29 °C for 15 to 16 h [35, Methods, partly].
  - Weights had to be rebalanced by hand, and sums grew sublinearly with more inputs because of
    resource competition [35, "Direct-Expression WSO", partly].
  - The authors state that it lacks learning; the threshold is tuned by hand [35, Discussion].

### 5.3 Self-assembly

- **Nucleation classifier (2024).** 917 DNA tiles with 698 binding domains assemble into one of
  three shapes, chosen by the concentration pattern; 371 of the tiles are shared between at
  least two shapes [27, Methods].
  - 18 grayscale 30 × 30 images were turned into tile concentrations and sorted into three
    classes after a 150 h anneal from 48 to 45 °C [27, "Pattern recognition by nucleation"].
  - All 18 gave more of the correct shape than of any other, and all but 5 were more than 80 %
    selective [27, same].
  - Of 12 degraded images, speckle distortions and all but one partly obscured image were
    recognized. Only 3 of 6 alternative handwritten digits were [27, same].
  - Of 37 probe patterns, about half nucleated selectively; the authors could not determine why
    the others failed [27, "Colocalization controls nucleation"].
  - Concentrations ran from 16.67 to 450 nM in ten levels, 60 nM on average [27, Methods].
  - Read-out: AFM counts by four authors differed by up to ±50 % in total but agreed within 5 %
    in shape ratios [27, Methods].
  - Control is kinetic: equal yields from an equal mix contradict an equilibrium Boltzmann
    distribution [27, "Molecular system design"].
  - Training: the pixel-to-tile map was optimized in silico by hill climbing, as in reservoir
    computing [27, Methods]. Performance drops as more patterns are trained [27, Discussion].
    Temperature trades speed against accuracy and pattern complexity [27, Discussion].
  - The authors call the approach slow compared with earlier biochemical networks [27,
    abstract].

### 5.4 Chemical reservoirs

- **Formose reservoir (2024).** A 435 µL stirred-tank reactor with a residence time of 2 min
  [36, Methods]. Up to 106 ion signals, read by ion-mobility mass spectrometry every 500 ms,
  feed one trained linear read-out [36, "A chemical reservoir computer"].
  - It emulated all Boolean gates and six nonlinear tasks (XOR, checkers, circle, sine,
    concentric circles, dots) on 132 input points, with 30 min of equilibration each [36,
    "Nonlinear classification"]. It scored like support vector classifiers, multilayer
    perceptrons and extreme learning machines, and lost clearly only on the circle task [36,
    same].
  - After 30 min of training it predicted an E. coli carbon-metabolism model (87 substrates, 92
    reactions) for 60 min [36, "Predicting complex dynamical systems"]. It missed species on
    very long time scales, which the formose network lacks [36, same].
  - It forecast two of three Lorenz inputs 120 s ahead for several hours [36, "Forecasting
    and mutual information"]. The Results give a 20 min training period and the Methods 30 min
    [36, "Forecasting and mutual information"; Methods, "Forecasting"].
  - The read-out is electronic. A fully chemical, autonomously learning read-out is named as the
    key open challenge [36, Conclusion].
- **Enzymatic reservoir (2025).** Seven immobilized enzymes and seven peptides in a 100 µL
  reactor give 103 ion traces after 1 h of equilibration per input [37, "Design of a recursive
  enzymatic competition network", Methods].
  - It classified XOR, circle and hourglass tasks, from peptide concentrations and from
    temperature and pH alone [37, "Chemical and physicochemical non-linear classification
    tasks"].
  - It sensed temperature over 30 °C with an average standard deviation of 1.33 °C [37,
    "Generalized physical information processing"]. It told apart blue-light pulse periods of
    2 to 8 min at constant mean intensity [37, "Extending the input space to light pulses"].
  - Subsets of three and of five enzymes failed the nonlinear tasks [37, "Chemical and
    physicochemical non-linear classification tasks", partly].
  - Weights live in an in-silico linear read-out [37, Methods]. The authors rate the network
    below the formose reservoir in computing power [37, Conclusions].
- **Belousov-Zhabotinsky array (2020).** A 5 × 5 array of coupled oscillating cells, each
  stirred on its own and filmed [38, "Pattern recognition using the BZ platform"].
  - A convolutional network on the video classified 20 input patterns at 80.5 % accuracy
    (750 features, 30 % test split), and at up to 92.5 % (500 features, 10 % test split) [38,
    same].
  - With water instead of the BZ medium, the read-out could not classify the patterns [38,
    same].
  - Oscillation periods of 40 s to 1 min set the speed. Reagents deplete, so each run used a
    30 min window after 10 min of stabilization [38, Discussion].
  - The classification runs outside chemistry; the authors call the system a hybrid chemical
    computer [38, Discussion].

### 5.5 Summary

| System | Year | Largest verified scale | Time | Weights set by | Source |
|---|---|---|---|---|---|
| Seesaw Hopfield memory | 2011 | 112 strands, 72 species | outputs read at 8 h | in silico | [22] |
| Winner-take-all | 2018 | 100-bit patterns, 9 classes | not verified | not verified | [23] |
| DNA CNN | 2022 | 144-bit inputs, 8 classes | "minutes" with freeze-thaw | not verified | [25] |
| Enzymatic neurons | 2022 | 10-bit majority, two layers | 112 to 148 min | concentration tuning | [26] |
| Nucleation | 2024 | 917 tiles, 30 × 30 images | 150 h anneal | in silico | [27] |
| In-vitro learning | 2025 | > 700 species in one tube | 24 h per class | in vitro, by memorizing | [28] |
| Heat recharge | 2025 | 289 strands, 16 rounds | recharge in minutes | not stated | [29] |
| Formose reservoir | 2024 | 106 ion signals | 30 min per point | in-silico read-out | [36] |

## 6. Physics a simulated chemical network must obey

Each rule names the source that owns it, and what breaks without it.

### 6.1 Mass action and nonnegative concentrations

- **Rule.** An elementary reaction's rate is its rate constant times its reactant
  concentrations, each raised to its stoichiometric coefficient [39, Eq. 4; 43, Eq. 2].
- **Rule.** A polynomial ODE comes from a mass-action network if and only if every negative term
  of `dx_m/dt` contains `x_m` [43, Def. 6, Lemma 1]. In the authors' words, "in order to lose a
  chemical species via a reaction, that species must participate as a reactant" [43, Sec. 3].
  The lemma is due to Hárs and Tóth (1979) [43, Lemma 1].
- **Consequence.** Positive stoichiometric compatibility classes are forward-invariant, so
  concentrations stay positive [43, Sec. 2].
- **What breaks.** A term such as `−x·z` in `dy/dt` makes a system non-kinetic [43, Sec. 3].
  Our reading: a negative weight written as a drain `−w·x_j` in `dx_i/dt` is therefore not
  chemistry. Signed values need dual rail [6, Thm. 5.10; 18, Sec. 2.2, Def. 1].

### 6.2 Reversible elementary steps and local detailed balance

- **Rule.** Every reaction is elementary and reversible. Each forward reaction has its backward
  reaction, and no species or reaction is hidden [39, Sec. I, p. 2].
- **Rule.** "Closed CRNs must be detailed-balanced"; Rao and Esposito call this the zeroth law
  for CRNs [39, Sec. II D]. They impose it on the rate constants as local detailed balance [39,
  Sec. II D, Eq. 18]. Its thermodynamic form is `k+ρ/k−ρ = exp(−Δ_r G°_ρ / RT)`, where
  `Δ_r G°_ρ` is the stoichiometry-weighted sum of standard chemical potentials [39, Sec. III B,
  Eq. 50].
- **Out of equilibrium too.** Rate constants are constants, so the relation holds for any
  distribution, also out of equilibrium [42, Sec. V C, Eqs. 21-22]. Local detailed balance does
  not mean that the network runs near equilibrium [39, Sec. III B].
- **Cycle form.** For linear kinetics, the product of the rate constants around each oriented
  cycle equals the product of the reverse rate constants (Wegscheider) [40, Sec. 1.1, Eq. 3].
  For mechanisms of fully reversible elementary reactions, the number of independent relations
  equals the number of linearly independent cycles [40, Sec. 5].
- **Stochastic form.** The same rule fixes the stochastic constructions: `k+/k− = exp(−ΔG)` from
  species energies [14, Eq. 11], the thermodynamic formula
  `κ_forward/κ_backward = exp(−ΔG/kT)` [15, Remark 1].
- **What breaks.** A model parameterized without this rule shows unphysical cyclic flows in its
  steady state [42, Sec. V D]. A steady flow `X → Y → Z → X` at equilibrium could power
  machines and would violate the second law [42, Sec. V C, Fig. 8]. A non-zero cyclic flux
  without entropy production would be "a chemical version of a perpetual motion machine" [45,
  "Power consumption and entropy production"]. Kinetic models violate detailed balance easily
  unless care is taken [41, abstract].

### 6.3 Parameters: energies first, then barriers

- **Only the ratio is fixed.** The forward and backward constants of a reaction may be scaled
  together without changing the equilibrium distribution [44, §2.4]. In a Markov-jump
  parameterization, symmetric barriers drop out of the equilibrium `π_i ∝ exp(−E_i)` [47,
  "Expressivity requires non-equilibrium driving"].
- **Advice.** Start from free energies and impose the rate ratio for each pair of reactions
  [42, Sec. V D]. Rate constants written down directly usually imply inconsistent free
  energies along different paths [42, Sec. V D].
- **A formalism.** A thermodynamic-kinetic formalism with compound capacities and reaction
  resistances obeys detailed balance for all values of its parameters [41, abstract].
- **Kinetics are empirical.** Strand-displacement rates can change by orders of magnitude at a
  fixed standard free energy [42, Sec. V D, partly]. In some cases most of a free-energy change
  shows up as a higher off-rate [42, Sec. V D, partly].
- **Units.** For `A + B ⇌ AB`, `φ_bind/φ_unbind = V0·exp(−ΔF0/kT)`, with `V0 = 1/C0` the
  volume of the standard concentration [42, Sec. V D, Eq. 25].
- **Our reading:** give each species a standard free energy `G°_i` and each reaction a
  nonnegative barrier excess `b_ρ ≥ 0` over its higher side, so that
  `G‡_ρ = max(G°_R, G°_P) + b_ρ`. Set `k+ρ = κ·exp(−(G‡_ρ − G°_R)/RT)` and
  `k−ρ = κ·exp(−(G‡_ρ − G°_P)/RT)`, where `G°_R` and `G°_P` are the summed standard free
  energies of the reactant and product sides. Then `k+ρ/k−ρ = exp(−(G°_P − G°_R)/RT)`, which
  is Eq. 50 of [39] for every gene value. A mutation of any energy or barrier excess keeps
  local detailed balance.
- **Our reading:** both rate constants then stay at or below κ. A barrier below a reactant side
  would give a rate constant above κ, faster than a barrierless step. κ is thus the largest rate
  constant in the chosen units; physically it would be the attempt rate, or the encounter limit
  for bimolecular steps. No source of this note gives that bound. The excess `b_ρ` is also
  unchanged when all energies shift by a constant times a conserved composition (section 8.3).

### 6.4 Irreversible steps only as limits or as spent fuel

- **Rule.** A system with irreversible reactions is a limit of detailed-balanced reversible
  systems if and only if it satisfies an extended detailed balance [40, Thm. 1]. That holds if
  and only if 0 is not in the convex hull of the irreversible reactions' stoichiometric
  vectors, taken modulo the span of the reversible ones, and the reversible part obeys detailed
  balance [40, Thm. 3]. Physically, irreversible reactions cannot be part of oriented cyclic
  pathways [40, abstract].
- **Examples.** In linear systems, irreversible transitions may only connect reversible parts,
  without cycles [40, Sec. 3.3]. `2H → H2` together with `H + H2 → 3H` forms a cycle, so both
  must be reversible [40, Sec. 3.4, Eq. 15].
- **Fuel.** Treating a reaction as irreversible is acceptable only in context, for example in a
  strongly fuel-driven process [42, Sec. VI A]. Networks that break local detailed balance must
  be coupled to reservoirs of fuel molecules [14, Sec. 2.3]. Living systems stay out of
  equilibrium through a continuous supply of fuel such as ATP [42, Sec. VII].
- **Strand displacement.** It is effectively irreversible because base pairing releases free
  energy [7, fn. 2]. A completely irreversible reaction would need an infinite energy
  difference [19, fn. 1, preprint].
- **What breaks.** A CRN parity algorithm designed with irreversible reactions loses accuracy
  once finite reverse reactions are added [42, Sec. VI A, Eqs. 26-28]. The free energy it needs
  grows with the logarithm of the volume [42, Eq. 29].

### 6.5 Open networks: chemostats and the energy bill

- **Chemostats.** Holding a species at a fixed concentration either breaks a conservation law or
  creates an emergent cycle: `|S_y| = |λ_b| + |ε|` [39, Sec. II D, Eqs. 19-28].
- **Driving.** A network is detailed balanced if and only if all emergent-cycle affinities
  vanish, which is equivalent to Wegscheider's condition [39, Sec. II D]. A network with only
  one chemostatted species is always detailed balanced: the first chemostat breaks mass
  conservation and creates no cycle [39, Sec. II D, Example 6].
- **Entropy production.** `T·dS_i/dt = RT·Σρ (J+ρ − J−ρ)·ln(J+ρ/J−ρ) ≥ 0` [39, Sec. III D,
  Eq. 55]. It vanishes only at equilibrium, and only to second order near it [39, Eq. 57].
- **Steady dissipation.** At a nonequilibrium steady state, the entropy production equals the
  chemostats' exchange currents times their chemical potentials [39, Eq. 60]. It also equals the
  emergent-cycle currents times their affinities [39, Eq. 61].
- **Work.** `T·dS_i/dt = dW_c/dt − dG/dt ≥ 0` [39, Sec. III E, Eq. 78, partly]. So entropy
  production limits the chemical work the chemostats must do, with equality only for
  quasi-static changes [39, Eqs. 78-79, partly].
- **Lyapunov functions.** In closed networks the nonequilibrium Gibbs free energy
  `G = G_eq + RT·L(Z|Z_eq)` falls at the rate `T·dS_i/dt` [39, Eqs. 74, 77]. In open but
  undriven detailed-balanced networks, a transformed free energy plays this role [39, Sec. V B].
  In undriven complex-balanced mass-action networks, the relative entropy to the steady state is
  a Lyapunov function [39, Sec. IV, Eq. 87].
- **Our reading:** Eq. 55 has no finite value when a backward current is exactly zero. A model
  with ideal irreversible reactions has no entropy bookkeeping, so a breeder that charges energy
  must keep the reverse rates finite.

### 6.6 What relaxation forbids

- Detailed-balanced networks always relax to an equilibrium. For given kinetics and chemostats it
  is unique and globally stable [39, Sec. II D]. It depends on the unbroken components, which
  the initial condition fixes [39, Sec. II D, Eq. 88].
- Feinberg's deficiency-zero theorem, as stated in [21]: a weakly reversible network of
  deficiency zero with mass-action kinetics has, for any rate constants, exactly one equilibrium
  in each positive stoichiometric compatibility class, and it is locally asymptotically stable
  [21, Thm. 3.1].
- Infinite networks can behave differently from equilibrium even without emergent cycles [39,
  Sec. II D].
- **Our reading:** in a finite network without driving, every trajectory ends at one
  equilibrium. Sustained bistability, oscillation and clocks therefore need driving, for example
  by chemostats, held fixed or changed by a protocol over time [39, Sec. II B]. A closed network
  that starts away from equilibrium shows them only transiently, while its stored free energy
  lasts [28, Discussion; 38, Discussion]. Driving is necessary but not sufficient, as the
  deficiency-zero theorem shows [21, Thm. 3.1]. Hjelmfelt's neuron fits this: it is reversible
  and has a single stationary state [1, abstract].

### 6.7 The stochastic picture

- A detailed-balanced CRN has a product-Poisson stationary distribution on each reachability
  class, with a finite partition function even for an infinite class [14, Sec. 2.3, Eq. 13].
- A reversible network's deterministic model has a detailed-balanced equilibrium if and only if
  its stochastic model is reversible with respect to its stationary distribution [21, Thm. 4.5].
- Species are independent unless conservation laws restrict which states can be reached [14,
  Thm. 2; 44, §3.1].
- The deterministic framework assumes molecule numbers large enough to ignore discreteness [39,
  Sec. II]. Gillespie's algorithm assumes a well-stirred system in a fixed volume [51,
  Sec. IIIB].

### 6.8 What needs fuel

- **Catalytic weights.** "The presence of catalysts cannot influence the equilibrium yield of a
  species." A circuit that biases stationary distributions by catalysis must therefore be
  powered by fuel [14, App. 7.1].
- **Sensing.** In a receptor and read-out network, learning about an external concentration
  requires breaking detailed balance and consuming energy, and more learning needs more energy
  [45, abstract]. Here "learning" means estimating the concentration, not training [45,
  "Energetics, information, and Landauer's principle"]. Entropy production is zero exactly where
  the estimation uncertainty diverges [45, Appendix, Eq. 16].
- **Copying.** A push-pull read-out network is a copy process. Its chemical work per copy is at
  least kT times the mutual information that the copy creates [46, Sec. II, Eq. 19]. With equal
  accuracies, an autonomous network reaches at most half of that efficiency, whatever its rates
  [46, Sec. II D]. Its cost diverges as accuracy approaches 1, while an optimal quasi-static
  protocol levels off at kT ln 2 for `p′ = 0.5` [46, Sec. II D, Fig. 2C]. Still, 99 % accuracy
  at `p′ = 0.5` costs less than 4 kT ln 2 per copy [46, Sec. II D, Discussion].
- **Classification.** In networks of (pseudo-)first-order reactions, the undriven steady state is
  the Boltzmann distribution and does not depend on the barriers [47, "Expressivity requires
  non-equilibrium driving"]. Classification accuracy rises with the allowed driving force, so
  non-trivial computation needs broken detailed balance [47, same]. When each input drives one
  edge, each response keeps a fixed sign, which limits the decision boundaries [47, "A
  fundamental limit on classification expressivity"]. Letting one input drive several edges
  lifts this limit [47, "Improving expressivity by increasing input multiplicity"].
- **A nonlinear exception.** The preprint version notes that with nonlinear chemical dynamics,
  computations can be expressive even at equilibrium [47, arXiv v1, p. 7].
- **Energies instead of fuel.** The edge-species Boltzmann machine needs no fuel; only molecular
  energies are tuned [14, Sec. 3.3].

### 6.9 Molecularity and the range of energies

- The exact Boltzmann-machine constructions keep high-molecularity reactions as an open issue
  [14, Sec. 3.3]. Universal distribution constructions use reactions such as `17V → 16V` [15,
  Sec. 1]. An autonomous learning CRN needs many reactants per reaction, which bimolecular
  reactions can approximate in theory [44, §4.4].
- A polynomial ODE can be reduced to bimolecular reactions [10, as restated in 11, Sec. 1], but
  minimizing the reduction is NP-hard [11, Sec. 4.2].
- Interactions in molecular systems can vary only over a finite range, relative to kBT, before
  associations become irreversible [50, Sec. 4.4].
- **Our reading:** a genome should cap molecularity at two per side and bound every energy and
  barrier to a fixed range. The range also bounds the ratio of the fastest to the slowest rate
  constant, which sets the stiffness (section 7.2).

### 6.10 Assumptions behind these rules

- The thermodynamic framework assumes a homogeneous, ideal dilute solution at constant
  temperature and pressure [39, Sec. II]. It assumes local equilibrium, with chemical potentials
  `μ_σ = μ°_σ + RT·ln(Z_σ/Z_tot)` [39, Sec. III A, Eq. 45].
- Non-ideal activities, diffusion limits and crowding are outside the sources read (section 9.4).

## 7. Simulation, training and evolution methods

### 7.1 Two semantics, two kinds of simulator

- **Exact stochastic.** For a well-stirred system in a fixed volume, the time τ to the next
  reaction and its channel μ have the joint density `P(τ, μ) = a_μ·exp(−a0·τ)` [51, Sec. IIIB,
  Eqs. 18-19]. This follows from the same hypothesis as the chemical master equation [51,
  Sec. IIIB].
- **Direct method.** Draw two uniform numbers r1 and r2. Set `τ = (1/a0)·ln(1/r1)`, and take μ
  as the first index whose cumulative propensity reaches `r2·a0` [51, Sec. IIIC, Eqs. 21a-21b].
  After each event, recompute only the propensities whose reactants changed [51, Sec. IIIC].
- **Properties.** The method is exact and never replaces dt by a finite step [51, Sec. IIID].
  Its run time grows roughly with the number of reaction events, and its memory is `N + 2M + 1`
  numbers [51, Sec. IIID]. It needs a reliable uniform generator; another seed gives a
  different but statistically equivalent chain [51, Sec. IIIC].
- **Exact variants.** The First Reaction and Next Reaction methods are mathematically equivalent
  to the Direct method [52, Sec. II]. The Next Reaction method needs asymptotically one random
  number per event [52, Sec. II].
- **Approximations.** Tau-leaping fires each channel a Poisson(`a_j·τ`) number of times while
  the propensities stay nearly constant [52, Sec. III]. Its author wrote that it was not ready
  for practical use until a robust step control existed [52, Sec. IX]. When every channel fires
  many times per leap, tau-leaping becomes the chemical Langevin equation [52, Sec. IV]. For
  infinite populations it reduces to an Euler step of the reaction-rate ODE [52, Sec. IV].
- **Deterministic.** Pure mass-action networks give polynomial ODEs [67, Sec. 2.3].

### 7.2 Stiffness and positivity

- **Stiffness.** Stiff problems are those for which explicit methods do not work, by Hairer's
  working definition [53, Sec. II.B]. The Robertson network (3 species, `k1 = 0.04`,
  `k2 = 3e7`, `k3 = 1e4`) and POLLU (20 species, 25 reactions) are standard stiff benchmarks
  [53, Sec. IV].
- **Implicit cost.** Implicit stiff solvers need Newton steps with LU factorizations that cost
  `O(k³)` in the number of states k [53, Sec. III.B].
- **Positive schemes.** Modified Patankar-Runge-Kutta (MPRK) schemes stay positive and conserve
  mass at every step size on conservative production-destruction systems [54, Sec. 1,
  Defs. 1.1-1.4]. Each step solves s linear `N × N` systems [54, Sec. 1]. Third-order MPRK43
  schemes crossed the Robertson problem over `[1e-6, 1e10]` in 29 steps, whose size grew by a
  factor of 4 per step from 1e-6 [54, Sec. 3-4, Fig. 4]. The authors state that a constant step
  is not appropriate for this problem [54, Sec. 3].
- **Their limits.** The evidence that MPRK schemes beat standard Runge-Kutta and Rosenbrock
  methods comes from biogeochemical models [54, Sec. 1, partly]. No thorough stability analysis
  yet ranks the MPRK43 schemes [54, Sec. 5].
- **Fixed steps in practice.** Two sources that train CRNs by gradients use explicit fixed
  steps: forward Euler and Runge-Kutta methods [55, Sec. 4], and RK4 in a preprint [60,
  App. A.1, preprint]. A winner-take-all MNIST example ran 1e5 steps of `dt = 1e-4` [55,
  Fig. 4].
- **Hazards.** Simple integrators perform poorly on stiff problems [55, Sec. 6]. A small dt
  raises the memory for backpropagation [55, Sec. 6]. Training can settle in low-loss regions
  that are numerical artifacts [55, Sec. 6]. Some networks, such as `2A → 3A`, blow up in
  finite time [65, Sec. 2, fn. 1].
- **Compiled programs.** CRNs compiled from programs carry errors from asymptotic convergence,
  from modules that never switch off completely, and from comparisons [65, Sec. 3.4]. In most
  examples the error grew during the computation [65, final section].

### 7.3 Bit reproducibility in Rust

- **Basic operations.** In Rust, `+`, `−`, `*`, `/`, abs, copysign, mul_add, sqrt, casts between
  floats and comparisons give exactly the IEEE 754-2008 results [70, Summary, partly]. Only the
  sign and payload of NaNs vary [70, partly]. The remainder `%` follows C's fmod,
  float-to-integer casts round toward zero, and 32-bit x86 without SSE2 can deviate [70,
  partly].
- **No contraction.** Strict semantics forbid the compiler to turn `a*b + c` into a fused
  multiply-add [70, "Rationale and alternatives"]. Rust supports no other rounding mode [70,
  "Reference-level explanation"]. 64-bit x86, 64-bit ARM and RISC-V comply fully [70,
  "Rationale and alternatives"].
- **Transcendentals.** `exp`, `ln`, `powf` and `powi` have unspecified precision, which may vary
  by platform, by Rust version and between calls in one execution [71, method docs]. `sqrt` and
  `mul_add` are guaranteed to return the correctly rounded result [71, method docs].
- **Background.** IEEE 754 does not specify transcendental functions, and libraries and
  processors differ [69, Sec. 4.1]. The x87's extended registers make results depend on
  register allocation and compiler options [69, Sec. 3.1]. Conformance is a property of the
  whole environment [69, Sec. 1].
- **libm.** The pure-Rust libm crate claims neither correct rounding nor determinism on its
  root page [72].
- **This repository.** The MLP's softmax and the scorer call `exp` and `ln` (`src/nn.rs:345`,
  `src/nn.rs:356`, `src/score.rs:191-193`). CNN training gives equal bits at any thread count
  on one machine, while other CPUs may give other bits (ADR 0007). Every random value comes
  from the crate's own ChaCha8 stream (ADR 0007, W8). The step-1 report notes that another
  toolchain may round a floating-point function differently
  (`docs/step1-counted/README.md:54`).
- **Our reading:** a fixed-step mass-action integrator needs only `+`, `−`, `*` and `/`. With a
  fixed operation order and one thread per Candidate, it should give equal bits on one machine
  at any thread count, provided no wall-clock limit stops a Candidate (ADR 0007; section 8.4).
  Since it would use no matrix library and no contraction, it might even give equal bits across
  CPUs of one target, unlike candle's products (ADR 0007). No source here has shown either for a
  CRN simulator (ticket 01).
- **Our reading:** rate constants from energies need `exp` once per genome. With integer
  energy genes, every rate constant is one entry of a fixed table stored as bit patterns, so no
  `exp` runs at all.
- **Our reading:** the stochastic algorithm needs `ln` only for its waiting times [51,
  Eq. 21a]. The choice of the next reaction needs no logarithm [51, Eq. 21b]. Time averages can
  weight each visited state by its mean holding time `1/a0` instead of a sampled waiting time.
- **Our reading:** a NaN should end a Candidate as a failure, and its bits should never be
  hashed, since they are not deterministic [70, Summary, partly].

### 7.4 Gradients through simulations

- **Adjoint.** The neural-ODE adjoint integrates `a(t) = dL/dz(t)` backwards together with the
  state, at a memory cost constant in depth [56, Sec. 2, Alg. 1]. It reconstructs the state
  backwards, which adds error when the reconstruction diverges; checkpointing is the remedy [56,
  Sec. 6]. With adaptive solvers, the number of function evaluations grows during training [56,
  Fig. 3d].
- **Reversal fails on dissipative systems.** For `dz/dt = −100z`, resolving forward and backward
  flows to 1 % takes about 200 000 steps [57, Sec. III]. For `λ = −1e4` the flow cannot be
  reversed in double precision [57, Sec. III]. On the Robertson network, the reverse solve had
  72 % error at tolerance 1e-6 [53, Sec. III.A]. The backsolve adjoint failed at every tolerance from 1e-6 to
  1e-14 [53, Sec. III.A].
- **Differentiate the discrete model.** Continuous-adjoint gradients are inconsistent with the
  discrete forward scheme, with errors of order dt [57, Sec. IV]. Automatic differentiation of
  the discrete scheme gives the gradient of the discrete model to machine precision [57,
  Sec. V]. Checkpointed and discrete adjoints agreed to three decimals on Robertson [53,
  Sec. III.A].
- **Forward mode for small systems.** Below about 100 parameters plus ODEs, forward-mode
  differentiation through the solver beat reverse mode and continuous sensitivities [58,
  abstract, Sec. III]. The backsolve adjoint was unstable on stiff problems [58, Sec. III.C].
  End-to-end fitting gave a mixed ranking [58, Table III]. Forward mode was also the more
  efficient choice for Robertson [53, Sec. IV.A.1].
- **Scaling.** A neural ODE learned Robertson only after outputs and loss were scaled per species
  and in time, and its training took 2 h on one CPU [53, Sec. III.C, IV.A].
- **Reaction networks as networks.** A chemical reaction neural network learned reaction orders,
  stoichiometry and rate constants from concentration data with Adam [59, Sec. 2, Sec. 3.1]. It
  recovered a four-reaction network within about 10 % under 5 % noise [59, Sec. 3.1]. Its
  authors assume a system that is not very stiff, and integer stoichiometry is not enforced [59,
  Sec. 4].
- **Structure by gradient (preprint).** A dense reaction tensor `T = softmax(W) − I` conserves
  mass and keeps products nonnegative [60, Sec. 2, Eqs. 1-2, preprint]. Removing small-rate
  reactions after training changed behaviour, so the sparse network was used during training
  [60, Sec. 2.1, preprint]. The learned winner-take-all network, once sparsified, equals the
  hand-designed approximate-majority protocol of Angluin et al. [60, Sec. 4, p. 8, preprint].
- **Through the stochastic algorithm.** A differentiable Gillespie algorithm replaces the
  reaction choice and the state update by smooth functions [61, Sec. I.A-B]. It is not exact:
  counts become real numbers, gradients grow unstable as the smoothing sharpens, moments are
  overestimated and tails are poor [61, Sec. I.D, II.B, partly]. Its demonstrations are steady
  states [61, Sec. I.D].
- **A GPU library.** icrn trains rate constants and initial conditions by backpropagation through
  time [55, Sec. 4, 5.2]. Its Hopfield example used Adam and gradient checkpointing [55,
  Sec. 5.3]. It trained a 256-neuron Hopfield CRN of 512 species and 131 072 catalytic rate
  constants, with rate constants `ReLU(W)` to keep them nonnegative [55, Sec. 5.3, partly]. The
  task was reaction-diffusion pattern growth into a photograph, and 512 degradation and 512
  diffusion constants were trained as well [55, Sec. 5.3]. Its speed figures are for an A100
  GPU [55, Fig. 3].

### 7.5 Training routes that avoid kinetics

- **Compile a trained network.** Noncompetitive CRNs compute their output by stoichiometry under
  any reasonable rate law [8, Theorem]. Networks can therefore be trained as ReLU networks and
  compiled, and their outputs matched in mass-action simulation [8, "Simulations"].
- Every compile route in this note trains offline, in silico [7, Sec. 5; 8, "Related Work"; 9,
  Sec. 6; 14, Sec. 5].
- **Equilibrium Propagation.** It gets the gradient of a cost at a free fixed point from one free
  and one weakly nudged relaxation of an energy-minimizing system [48, Thm. 1, partly]. The
  formula is exact in the limit of zero nudging; a finite nudge gives an estimate [48,
  Sec. 3.5, partly]. Fixed points must be energy minima [48, Sec. 3.2]. On digital hardware they
  are found slowly and approximately [48, Sec. 3.5].
- **Coupled learning.** It compares a free and a clamped state with a local rule, for systems
  whose physical degrees of freedom minimize a physical cost [49, Sec. II B]. It was shown in
  simulation, for flow and elastic networks among others [49, Sec. II]. Physical learning
  parameters cannot be negative, which limits what can be learned and slows learning [49,
  App. D].
- **Our reading:** closed detailed-balanced CRNs minimize a free energy within each compatibility
  class [39, Eqs. 74, 77]. So Equilibrium Propagation might apply with that free energy as its
  energy. No source here tests this. The Boltzmann-machine rule of [14, Eq. 24] already has the same
  contrastive form: clamped minus free mean counts.

### 7.6 Learning inside chemistry

- **Backpropagation in chemistry.** A supervised pipeline of feedforward, adjoint, gradient and
  update stages uses only uni- and bimolecular reactions, and was validated by simulation only
  [18, abstract, Sec. 3]. CRNs cannot run backwards in time, so the adjoint ODEs are rewritten
  in forward time `τ = t_f − t` [18, Sec. 2.2]. An oscillatory two-phase chemical clock
  multiplexes the stages, and the learning rate is fixed at 1 [18, Table 4, Stage N4]. The
  authors note that oscillators are hard to build because their autocatalytic loops amplify
  noise [18, Discussion].
- **Autonomous Boltzmann learning, in simulation.** A detailed-balanced CRN coupled to
  non-detailed-balanced "potential clamping" reactions performs gradient descent on relative
  entropy [44, §4.1-4.4]. The authors show that this online Boltzmann-machine rule is integral
  feedback control [44, §4]. It learned an XOR distribution with three-node machines [44, §4,
  Fig. 3].
- Species can go extinct together, so the fully stochastic version may have non-unique steady
  states; this motivates an ensemble of vesicles that share the potential species [44,
  §4.2-4.3]. The construction needs many reactants per reaction [44, §4.4]. The authors report an
  accuracy/dissipation trade-off in an appendix that this check did not read [44, §5].
- **In vitro.** The 2025 DNA network learns by memorizing (section 5.1) [28, Methods].
- **Review.** Physical learning needs physical degrees of freedom, learning degrees of freedom
  and a local rule [50, Sec. 1.1]. Physical systems are in principle more constrained in what
  they learn than in-silico networks [50, Sec. 1.3]. Learning is almost always irreversible; as
  a many-to-one map of weights it is erasure and must consume free energy, but how much is
  unclear [50, Sec. 5.3(a)].
- **Status in 2022.** Discussing active molecular systems, the same review stated that truly
  autonomous learning had yet to be demonstrated at the molecular scale [50, Sec. 2.2].

### 7.7 Evolution

- **Computing networks.** Evolution produced reaction networks that compute multiplication, roots
  and logarithms, and modularity evolved along the way [62, abstract].
- **BioJazz.** Networks are encoded as a binary genome with rule-based models, so their size is
  unbounded [63, Methods]. Mutations include duplications, deletions and horizontal transfer [63,
  Methods]. Fitness comes from integrating the generated ODEs [63, Methods]. It offers a random
  walk with Kimura's fixation probability and a fixed-size population, and it prunes evolved
  networks afterwards [63, Methods]. Its ultrasensitivity runs stopped at fitness 0.8 or after at
  most 120 h of computation each [63, Results, Fig. 4].
- **Artificial biochemical networks.** These were evolved as discrete-time maps, so no ODE was
  solved [64, Sec. IV]. Fixed-length genomes were used because variable length had shown size
  pathologies [64, Sec. V]. Without a mass-conservation rule, fit artificial metabolic networks
  were very hard to find [64, Sec. VI.A]. No network type was best on all three control tasks
  [64, Sec. VIII].

### 7.8 Software

- **Catalyst.jl.** It simulates reaction-rate ODEs, chemical Langevin equations and several exact
  stochastic algorithms, and it produces differentiable models [67, Sec. 2.2-2.4, Sec. 4.1]. In
  single-thread benchmarks it was often an order of magnitude or more faster than BioNetGen,
  COPASI, GillesPy2 and SimBiology [67, Sec. 2.2, Fig. 3]. At publication it had no tau-leaping
  or hybrid solvers [67, Sec. 1].
- **Visual DSD.** It compiles DNA strand-displacement programs into reactions and simulates them
  with Gillespie's algorithm or with Runge-Kutta-Fehlberg [66, Sec. 2].
- **CRN++.** It compiles an imperative language into mass-action CRNs whose steps a chemical
  oscillator orders [65, Sec. 3.2]. Its examples have 20 to 39 reactions and 24 to 32 species
  [65, Table 2].
- **icrn.** Python and JAX, with training by backpropagation (section 7.4) [55, Sec. 4].
- **diffsol.** A Rust crate with adaptive BDF, SDIRK and explicit Runge-Kutta methods and with
  forward and adjoint sensitivities [68, Summary, Features]. Its paper says nothing about bit
  reproducibility or fixed steps [68].
- **Our reading:** none of these sources claims bit-equal results. A small integrator inside the
  crate, written by hand like the MLP of `src/nn.rs`, fits the breeder better. Catalyst or
  diffsol could serve offline as reference solvers.

## 8. How a third model kind could fit the Brutmaschine

This section is a sketch, not a spec. Every sentence without a source or a file reference is
our reading.

### 8.1 Where it would sit

- The MLP mode trains and scores Candidates. A genome's `Arch` is `Mlp` or `Cnn` today
  (`src/genome.rs:58-65`).
- Our reading: a third variant would hold a reaction-network genome. It would follow the CNN
  pattern: old archives stay readable through a serde default, and every genome is checked
  before training (`.scratch/cnn-basismodelle/spec.md`, Design).
- MLPs and CNNs share `Limits`, so that neither wins by size (`src/config.rs:34-67`;
  `.scratch/cnn-basismodelle/spec.md`, Design, "Shape check"). Our reading: a chemical network
  would share them too.
- Hidden scores come from a separate process (CONTEXT.md, "Hidden evaluation"). Our reading: that
  process needs the same simulator with the same bits, so the simulator must live in the crate.
- The MLP is written by hand (`src/nn.rs:1-4`). Our reading: a hand-written integrator needs no
  new crate, so invariant I1 of both contracts holds (`docs/STAGE1_CONTRACT.md:33-35`,
  `docs/STAGE2_BACKEND_CONTRACT.md:37-41`).

### 8.2 Genome

Our reading, in one table:

| Part | Content | Rule it serves |
|---|---|---|
| Species | up to S species; each has an integer standard free energy, in steps of a fixed fraction of RT, and a composition vector over a few conserved units | 6.2, 6.3, conservation |
| Reactions | up to R reversible reactions with at most two species per side; each has a nonnegative integer barrier excess over its higher side and conserves composition | 6.1, 6.3, 6.9 |
| Chemostats | optional; fuel and waste species held at fixed concentration; each has an integer chemical potential | 6.4, 6.5 |
| Inputs | one dual-rail pair per task input; their initial concentrations come from the input value | 3.1 |
| Outputs | one dual-rail pair per task output, read at a fixed time | 3.1 |
| Context | initial concentrations of the other species, which act as biases or store fuel for outputs read at a fixed time | 3.1, 6.6 |
| Integration | fixed by the Run, or a gene accepted only under a step rule, for example that dt times the largest pseudo-first-order rate stays below the scheme's stability bound | 7.2 |

- **Rate constants.** Our reading: k+ and k− follow from the energies and the barrier excess as
  in section 6.3. A chemostat's concentration follows from its chemical potential through
  Eq. 45 of [39].
- **No exp at run time.** Our reading: with integer genes, every exponent is an integer multiple
  of one step. Every rate constant and every chemostat concentration is then one entry of a
  fixed table.
- **Exact checks.** Our reading: Wegscheider's cycle conditions [40, Sec. 1.1] and conservation
  become identities in integer arithmetic on the genes.
- **One unit.** Our reading: association rates depend on the standard state [42, Eq. 25], so the
  whole simulation would use one reference concentration as its unit.
- **Reported properties.** Whether a network is feedforward and noncompetitive can be read off
  its reactions [8, Def. 4]. These properties say when the exact ReLU equivalence applies in the
  irreversible limit (section 3.1). Our reading: report them, but do not make them separate
  kinds.

### 8.3 Limits and cost

- Our reading: caps on species, reactions, molecularity, the range of energies and barrier
  excesses, peak concentration, steps and simulated time.
- Our reading: a state that is not finite, or above the concentration cap, ends the Candidate
  as a failure.
- **FLOPs before the run.** Our reading: a fixed-step trajectory has a cost that follows from the
  genome. A rough count under our own assumptions: 50 reversible reactions with at most two
  reactants per side, 30 species, RK4 and 1000 steps cost about 2e6 FLOPs per input. One pass
  over the 1000 training points of checkerboard (`src/task.rs:279`) then costs about 2e9 FLOPs.
  The default `max_train_flops` of 2e10 (`src/config.rs:61`) buys about 9 such passes.
- **Run-wide caps.** The default `max_total_flops` of 2e12 (`src/config.rs:64`) covers about 100
  Candidates that spend the full 2e10. A default run has 16 + 30 × 16 = 496 Candidates
  (`src/config.rs:130-132`). The run also stops at `max_total_seconds`, 300 s by default
  (`src/config.rs:63`). Our reading: the cost estimate must be compared with these caps as
  well.
- **Parameters.** Fitness subtracts a penalty per log10 of the parameter count and of the
  training FLOPs (`src/score.rs:301-311`). What a parameter is for a CRN is open (`map.md`,
  Fog).
- **Further costs.** No source here bounds convergence time, reaction count or peak concentration for
  compiled networks [7, Sec. 6; 8, "Convergence Time Analysis"]. Our reading: they need cost
  terms of their own.
- **Energy bill.** Our reading: charge the dissipation `T·Δ_iS`, not the chemostats' work
  `W_c`. The work rate is exchange currents times chemical potentials at all times [39,
  Eq. 69]. It equals the dissipation rate at a steady state [39, Eq. 60]. Off a steady state the
  dissipation is `T·Δ_iS = W_c − ΔG` [39, Eqs. 78-79]. Outputs here are read at a fixed time
  (section 8.2).
- **The energy reference.** Our reading: shift every species energy, barrier and chemostat
  potential by a constant c times a conserved composition. Every rate constant and every
  chemostat concentration stays the same. `W_c` changes by c times the change in that
  composition's total amount. Chemostats can change that total, since they break conservation
  laws [39, Sec. II C, Eqs. 10, 69]. `G` changes by the same amount [39, Eq. 70], so `T·Δ_iS`
  does not change. Two genomes with equal dynamics would get different `W_c`, but equal
  dissipation.
- **Its price.** Our reading: two forms compute the dissipation, and both need `ln`, whose
  precision Rust leaves unspecified [71].
  - The sum of `Δt·RT·Σρ (J+ρ − J−ρ)·ln(J+ρ/J−ρ)` over the steps [39, Eq. 55] needs `ln` for
    every reaction at every step. That conflicts with the rule of no `exp`, `ln` or `powf` in
    the loop (sections 7.3 and 8.4), unless `ln` is written by hand from exactly rounded
    operations, for example as a table with interpolation.
  - `W_c − ΔG`, with `G` from [39, Eq. 70], needs `ln` of every species concentration only at
    the start and at the read-out time. With constant chemostats, `W_c` needs only their
    potentials and the reaction extents. This keeps `ln` out of the loop, at the bar of the
    scorer's `ln` (`src/score.rs:191-193`; `map.md`, Fog 5).
- **Numerical artifacts.** Training can settle in low-loss regions that are numerical artifacts,
  where the simulation no longer reflects the physical system [55, Sec. 6]. Our reading: rerun
  each scored Candidate at dt/2 on a fixed subset of inputs. It fails if its outputs change by
  more than a stated tolerance. The FLOPs of this check are charged.

### 8.4 Determinism

- Our reading: the integrator would use only `+`, `−`, `*`, `/` and perhaps `mul_add`, in a
  fixed order. Rust rounds these exactly on 64-bit x86, 64-bit ARM and RISC-V [70, partly; 71].
- Our reading: rate constants come from the table, so no `exp`, `ln` or `powf` runs in the loop
  [71].
- Our reading: one thread per Candidate, with Candidates in parallel, as for the MLP
  (`src/nn.rs:3`). The CNN instead gives equal bits at any thread count (ADR 0007).
- Equal bits also need that no Candidate is stopped by `max_candidate_seconds` (ADR 0007). For
  a CNN, `Genome::validate` holds the estimated training time to `max_train_flops /
  TIME_BUDGET_FLOPS_PER_SECOND`, 20 s at the defaults (`src/genome.rs:310-319`,
  `src/cnn/shape.rs:325`). Our reading: a CRN validator would estimate the simulation time from
  the genome at a measured worst-case FLOP rate and hold it to the same bound.
- `brut-eval` checks a CNN against `MAX_EVAL_FLOPS` (1e11) and against `MAX_EVAL_SECONDS`
  (60 s) by a time model (`src/hidden_eval.rs:84-91`, `src/hidden_eval.rs:501-532`). Our
  reading: a CRN needs a check of the same kind.
- The scorer already calls `exp` and `ln` (`src/score.rs:191-193`). Our reading: that holds a
  chemical network to the same bar as an MLP, equal bits on one machine.
- Our reading: a stochastic variant would draw from the ChaCha8 stream (ADR 0007, W8) and avoid
  `ln` as in section 7.3.

### 8.5 Training and scoring

Our reading: three routes, which ticket 04 compares.

- **Evolution only.** Our reading: mutation and selection set every gene, as in BioJazz and in
  the artificial biochemical networks [63, Methods; 64, Sec. V].
- **Exact gradients through the integrator.** Discretize-then-optimize gives the gradient of the
  simulated model [57, Sec. IV]. Forward mode suits few trained values [58, abstract]. Reverse
  mode stores the trajectory, as backpropagation through time does in icrn [55, Sec. 5.2]. Our
  reading: the trained values would be initial concentrations of catalysts and context species,
  since weights can be catalyst concentrations [9, Table 1], while energies stay integer genes.
- **Equivalent network.** Our reading: train the network that the CRN equals at its steady
  state, ReLU [8, "Computational Power"] or smoothed ReLU [9, Prop. 4.9]. Then simulate only to
  check convergence and to measure its time. Training then costs as much as for the network.
- **Scoring.** Our reading: the output pairs at the read-out time are the logits or the
  regression value. The argmax stays in the scorer, since no rate-independent CRN computes it
  (section 4.5).

### 8.6 Tasks

- Spirals and checkerboard have 2 inputs, sine regression 1 and parity 8
  (`src/task.rs:262-305`). Our reading: each input becomes one dual-rail pair.
- The image tasks have 16 × 16 = 256 inputs (`src/task.rs:29`, `src/task.rs:359-375`). Our
  reading: 512 input species are large for a first step.
- The formose reservoir's nonlinear tasks (XOR, checkers, circle, sine, concentric circles,
  dots) are 2-D classifications [36, "Nonlinear classification"]. Our reading: they are of the
  same kind as spirals and checkerboard.
- Our reading: the hidden evaluation and the quarantine rule would apply unchanged (CONTEXT.md,
  "Hidden evaluation", "Quarantine").

### 8.7 Mutations, seeds and promotion

- **Mutations.** Our reading, one operator per line:
  - an integer step on one energy, barrier excess or chemostat potential;
  - switching a reaction of a fixed-length list on or off, since variable-length genomes can
    show size pathologies [64, Sec. V];
  - changing one reactant or product within the composition rule;
  - changing one initial concentration;
  - crossover of reaction groups that share interface species.
- **Validity.** Every CNN child passes the shape check or is redrawn
  (`.scratch/cnn-basismodelle/spec.md`, Design, "Mutations"). Our reading: the same rule fits a
  reaction-network validator.
- **Conservation by construction.** Fit artificial metabolic networks were very hard to evolve
  without a conservation rule [64, Sec. VI.A]. A softmax reaction tensor keeps mass by design
  [60, Sec. 2, preprint].
- **Seeds.** Our reading: compiled ReLU MLPs of the Run's task [7; 8], smoothed-ReLU layers with
  catalytic weights [9], a winner-take-all module [23; 60, preprint], and a fixed random
  reservoir with a trained read-out as a labelled baseline [36; 37; 38].
- **Seeds need conversion.** Compiled and smoothed-ReLU seeds are not genomes of section 8.2
  until they are converted. Compiled networks use irreversible reactions with one product per
  fan-out branch, such as `X1+ → I+1,1 + I+1,2 + I+1,3` [7, Sec. 3.2, reactions (3)-(6)], or
  `X1+ → M1,1 + M1,2 + M1,3 + 2Y+ + Y−` after optimization [7, Sec. 3.3, reaction (29)].
  The construction of [6] divides with d reactants, `dW+ → Y+` [6, Lemma 5.12], where [8]
  uses halving chains [8, "Rational Weight ReLU Neural Networks"]. The smoothed-ReLU layers use
  irreversible reactions, and their weight reactions are trimolecular [9, Table 1]. Their
  reactions `H → H + X` and `qX → X` create or destroy X [9, Table 1]. Tickets 03 and 04
  handle the conversion.
- **Promotion.** CNN winners are to reach `candidates/models/` through a reviewed draft pull
  request (`.scratch/cnn-basismodelle/spec.md`, Design, "Promotion of winners"). Our reading:
  chemical networks could take the same path.

### 8.8 Link to the planned Architect

- A draft spec of 2026-10-31 plans an Architect, a language model that writes CNN genomes
  (Blueprints) in Architect rounds of the MLP mode (`.scratch/llm-nutzt-cnn/spec.md`,
  section 4.3; status needs-triage). The spec is an uncommitted draft in the worktree of branch
  `claude/llm-nutzt-cnn`. It is not on the branch and not merged, so its section and question
  numbers may change.
- There, a Blueprint is checked strictly and never repaired, and a refused one is replaced by a
  random child (same spec, Q28). Requests carry the task and the limits in words and the parents
  with their public validation scores (same spec, Q29). Answers arrive as files, are recorded in
  the lineage archive and are replayed (same spec, Q27 and Q30).
- Our reading: a reaction-network genome could take the same path once it exists. The Blueprint
  check would add the checks of section 8.2. The request would state the physics rules of
  section 6 in words. No source here tests whether a language model proposes useful CRNs.

### 8.9 What the sketch leaves open

The open choices are the Fog of `map.md`.

## 9. What is not known

### 9.1 Theory

- No formal bound on the convergence time of compiled ReLU CRNs exists; the data are empirical
  [7, Sec. 6; 8, "Convergence Time Analysis"].
- Whether feedback ReLU networks and feedback CRNs are equivalent is open [8, "BReLU Networks
  Simulate CRNs"].
- What irrational initial concentrations add to rate-independent CRNs is open [6, Sec. 6.2].
- If reverse reactions may run forever, no state is stable, and how to define computation then
  is open [19, fn. 2, Conclusion, preprint].
- Synthesizing a CRN that computes an input/output function online is called a harder problem
  [11, Sec. 5-6].
- Our reading: universal approximation by smoothed-ReLU CRNs would follow from their exact
  implementation [9, Prop. 4.9] and a universal approximation theorem for non-polynomial
  activations, which [17, Thm. A.1] uses. No source here states this corollary.
- How finite reverse rates degrade a compiled ReLU network is not quantified. The only worked
  case is a parity algorithm [42, Sec. VI A].
- The limits of [47] are proven for (pseudo-)first-order networks, and their extension to
  bimolecular reactions is preliminary [47, Discussion].
- No source here gives a general energy bound for inference or learning by a CRN neural network.
  Bounds exist for copying and sensing [45; 46]. The dissipation cost of learning is called
  unclear [50, Sec. 5.3(a)].
- No source here implements Equilibrium Propagation or coupled learning in a CRN (section 7.5).

### 9.2 Lab

- No system in these sources has learned in chemistry by optimizing a loss; the in-vitro
  learner memorizes [28, Methods].
- The learning network does not fully meet accuracy, reusability and flexibility [28,
  Introduction].
- Resets erase everything, and selective reset for unsupervised learning is future work [29,
  Discussion].
- Beyond two memories, annihilators grow quadratically and competition becomes biased [28,
  "Testing after training"].
- A chemical, autonomously learning read-out for reservoirs is open [36, Conclusion; 37,
  Conclusions; 38, Discussion].
- Why about half of the nucleation probe patterns failed is unknown to the authors [27,
  "Colocalization controls nucleation"].
- The verified depths are small: a two-layer enzymatic network [26, abstract] and a logic circuit
  of three layers by design [29, "Versatility"].

### 9.3 Methods

- No source here demonstrates bit-identical CRN simulation, or bounds the work of an adaptive
  stiff solver.
- MPRK schemes are not established for general multi-reactant mass-action networks [54, Sec. 1,
  partly; Sec. 5].
- The differentiable Gillespie algorithm has not been used on dynamical data [61, Discussion].
- Tau-leaping lacked a robust step control in 2001 [52, Sec. IX]; later work was not read.
- No source here compares evolution with gradient training of CRNs at equal compute.
- No source here tests whether trained or evolved CRNs generalize to unseen tasks, which is what the
  Brutmaschine's hidden evaluation measures (CONTEXT.md, "Hidden evaluation").
- Gradient methods do not enforce integer stoichiometry [59, Sec. 4], and sparsifying after
  training changes behaviour [60, Sec. 2.1, preprint].

### 9.4 Limits of this note

- The sources assume ideal dilute, well-mixed, isothermal solutions [39, Sec. II, III A].
  Activities, diffusion limits and crowding are not covered.
- Enzyme rate laws such as Michaelis-Menten and Hill kinetics enter only through one abstract
  [41] and through the rate laws that [6, Def. 4.1] admits.
- Read as abstracts only: [1], [2], [24], [30], [31], [32], [33], [41] and [62]. Read as
  abstract plus some figure legends: [22], [23] and [26]. Read through secondary copies of the
  abstract: [25].
- Not read in the primary: [10], restated by [11] and [12]; and [20], known through secondary
  statements.

## 10. Sources

Status is one of: peer-reviewed journal, peer-reviewed conference, preprint, or documentation
(not peer reviewed). "Read" names the version whose pages, sections and theorems this note
cites.

1. Hjelmfelt A, Weinberger ED, Ross J (1991). Chemical implementation of neural networks and
   Turing machines. Proc Natl Acad Sci USA 88(24):10983-10987.
   https://doi.org/10.1073/pnas.88.24.10983. Peer-reviewed journal. Read: abstract only (PubMed
   1763012).
2. Hjelmfelt A, Weinberger ED, Ross J (1992). Chemical implementation of finite-state machines.
   Proc Natl Acad Sci USA 89(1):383-387. https://doi.org/10.1073/pnas.89.1.383. Peer-reviewed
   journal. Read: abstract only (PubMed 11607249).
3. Soloveichik D, Cook M, Winfree E, Bruck J (2008). Computation with finite stochastic chemical
   reaction networks. Natural Computing 7(4):615-633.
   https://doi.org/10.1007/s11047-008-9067-y. Peer-reviewed journal. Read: Caltech technical
   report CaltechPARADISE 2007.ETR085 (etr085v2).
4. Angluin D, Aspnes J, Eisenstat D, Ruppert E (2007). The computational power of population
   protocols. Distributed Computing 20(4):279-304. Peer-reviewed journal. Read: arXiv
   cs/0608084v1, https://arxiv.org/abs/cs/0608084.
5. Chen H-L, Doty D, Soloveichik D (2014). Deterministic function computation with chemical
   reaction networks. Natural Computing 13(4):517-534.
   https://doi.org/10.1007/s11047-013-9393-6. Peer-reviewed journal. Read: arXiv 1204.4176v3.
6. Chen H-L, Doty D, Reeves W, Soloveichik D (2023). Rate-independent computation in continuous
   chemical reaction networks. Journal of the ACM 70(3), Article 22.
   https://doi.org/10.1145/3590776. Peer-reviewed journal; preliminary version ITCS 2014. Read:
   arXiv 2107.13681v3.
7. Vasic M, Chalk C, Khurshid S, Soloveichik D (2020). Deep molecular programming: a natural
   implementation of binary-weight ReLU neural networks. Proc. 37th ICML, PMLR 119:9701-9711.
   https://proceedings.mlr.press/v119/vasic20a.html. Peer-reviewed conference. Read: arXiv
   2003.13720v3.
8. Vasic M, Chalk C, Luchsinger A, Khurshid S, Soloveichik D (2022). Programming and training
   rate-independent chemical reaction networks. Proc Natl Acad Sci USA 119(24):e2111552119.
   https://doi.org/10.1073/pnas.2111552119. Peer-reviewed journal. Read: PMC9214506 full text
   and arXiv 2109.11422v1; places are section names.
9. Anderson DF, Joshi B, Deshpande A (2021). On reaction network implementations of neural
   networks. J R Soc Interface 18(177):20210031. https://doi.org/10.1098/rsif.2021.0031.
   Peer-reviewed journal. Read: arXiv 2010.13290v3.
10. Fages F, Le Guludec G, Bournez O, Pouly A (2017). Strong Turing completeness of continuous
    chemical reaction networks and compilation of mixed analog-digital programs. CMSB 2017,
    LNCS 10545:108-127. https://doi.org/10.1007/978-3-319-67471-1_7. Peer-reviewed conference.
    Not read; its result is cited as restated in [11] and [12]. The volume number was not
    confirmed independently.
11. Hemery M, Fages F, Soliman S (2021). Compiling elementary mathematical functions into finite
    chemical reaction networks via a polynomialization algorithm for ODEs. CMSB 2021,
    LNCS 12881:74-90. https://doi.org/10.1007/978-3-030-85633-5_5. Peer-reviewed conference.
    Read: arXiv 2106.15884v1.
12. Hemery M, Fages F (2022). Algebraic biochemistry: a framework for analog online computation
    in cells. CMSB 2022, LNCS 13447:3-20. https://arxiv.org/abs/2206.09624. Peer-reviewed
    conference. Read: arXiv 2206.09624v1. The chapter DOI was not confirmed.
13. Bournez O, Graça DS, Pouly A (2016). Polynomial time corresponds to solutions of polynomial
    ordinary differential equations of polynomial length. ICALP 2016, LIPIcs 55, 109:1-109:15.
    https://doi.org/10.4230/LIPIcs.ICALP.2016.109. Journal version: J ACM 64(6), Article 38
    (2017). Peer-reviewed conference. Read: arXiv 1601.05360v2.
14. Poole W, Ortiz-Muñoz A, Behera A, Jones NS, Ouldridge TE, Winfree E, Gopalkrishnan M (2017).
    Chemical Boltzmann machines. DNA 23, LNCS 10467:210-231.
    https://doi.org/10.1007/978-3-319-66799-7_14. Peer-reviewed conference. Read: arXiv
    1707.06221v1.
15. Cappelletti D, Ortiz-Muñoz A, Anderson DF, Winfree E (2020). Stochastic chemical reaction
    networks for robustly approximating arbitrary probability distributions. Theoretical
    Computer Science 801:64-95. Peer-reviewed journal. Read: arXiv 1810.02854v1,
    https://arxiv.org/abs/1810.02854.
16. Poole W, Ouldridge TE, Gopalkrishnan M, Winfree E (2022). Detailed balanced chemical
    reaction networks as generalized Boltzmann machines. arXiv 2205.06313v1.
    https://arxiv.org/abs/2205.06313. Preprint, not peer reviewed; no journal version found.
17. Dack A, Qureshi B, Ouldridge TE, Plesa T (2026). Recurrent neural chemical reaction networks
    that approximate arbitrary dynamics. Cell Systems 17(5):101572.
    https://doi.org/10.1016/j.cels.2026.101572. Peer-reviewed journal (PubMed 42025162).
    Read: arXiv 2406.03456v2 (5 June 2025), https://arxiv.org/abs/2406.03456; the places cited
    refer to this arXiv version.
18. Nagipogu RT, Reif JH (2025). Neural CRNs: a natural implementation of learning in chemical
    reaction networks. ACS Synthetic Biology 14(10):3899-3912.
    https://doi.org/10.1021/acssynbio.5c00099. Peer-reviewed journal (PubMed 40981009). Read:
    arXiv 2409.00034v4 (8 September 2025), https://arxiv.org/abs/2409.00034; the places cited
    refer to this arXiv version.
19. Kini R, Doty D. Reverse-robust computation with chemical reaction networks. arXiv
    2604.14355v1 (15 April 2026). https://arxiv.org/abs/2604.14355. Preprint.
20. Kurtz TG (1972). The relationship between stochastic and deterministic models for chemical
    reactions. J Chem Phys 57(7):2976-2978. Peer-reviewed journal; DOI not confirmed. Not read.
    Partly verified through secondary statements: the qualifications in section 4.7 come from
    [6, fn. 4].
21. Anderson DF, Craciun G, Kurtz TG (2010). Product-form stationary distributions for
    deficiency zero chemical reaction networks. Bull Math Biol 72(8):1947-1970.
    https://doi.org/10.1007/s11538-010-9517-4. Peer-reviewed journal. Read: arXiv 0803.3042v4.
22. Qian L, Winfree E, Bruck J (2011). Neural network computation with DNA strand displacement
    cascades. Nature 475(7356):368-372. https://doi.org/10.1038/nature10262. Peer-reviewed
    journal. Read: abstract (PubMed 21776082) and the legends of Figs. 1 to 3; the main text was
    not read. Partly verified: the legend numbers are confirmed. Not confirmed: that all four
    gate outputs were correct, the trace lengths, the size cost of dual rail, the stored bit
    patterns, the count of games played, the training algorithm, and the stated limits.
23. Cherry KM, Qian L (2018). Scaling up molecular pattern recognition with DNA-based
    winner-take-all neural networks. Nature 559(7714):370-376.
    https://doi.org/10.1038/s41586-018-0289-6. Peer-reviewed journal. Read: abstract (PubMed
    29973727) and the legends of Extended Data Figs. 2 and 6. Partly verified: the comparison
    with earlier networks is from the abstract, and the lateral-inhibition rationale is from
    [24]. Not checked: Extended Data Figs. 3, 4, 5, 7 and 8.
24. Rodriguez KR, Sarraf N, Qian L (2021). A loser-take-all DNA circuit. ACS Synth Biol
    10(11):2878-2885. https://doi.org/10.1021/acssynbio.1c00318. Peer-reviewed journal. Read:
    abstract (PubMed 34623152).
25. Xiong X, Zhu T, Zhu Y, Cao M, Xiao J, Li L, Wang F, Fan C, Pei H (2022). Molecular
    convolutional neural networks with DNA regulatory circuits. Nature Machine Intelligence
    4:625-635. https://doi.org/10.1038/s42256-022-00502-7. Peer-reviewed journal. Read: the
    abstract through a repository copy and a seminar listing. Partly verified: "from hours to
    minutes". Not checked: the exact times, the bead-separation step, the pooling details and
    the training method.
26. Okumura S, Gines G, Lobato-Dauzier N, Baccouche A, Deteix R, Fujii T, Rondelez Y, Genot AJ
    (2022). Nonlinear decision-making with enzymatic neural networks. Nature
    610(7932):496-501. https://doi.org/10.1038/s41586-022-05218-7. Peer-reviewed journal. Read:
    abstract (PubMed 36261553) and the legend of Extended Data Fig. 6. Partly verified: the
    enzyme concentrations; the three-template design, the account of bistability and the
    droplet imaging were not checked.
27. Evans CG, O'Brien J, Winfree E, Murugan A (2024). Pattern recognition in the nucleation
    kinetics of non-equilibrium self-assembly. Nature 625(7995):500-507.
    https://doi.org/10.1038/s41586-023-06890-z. Peer-reviewed journal. Read: PMC10794147 full
    text and arXiv 2207.06399v3.
28. Cherry KM, Qian L (2025). Supervised learning in DNA neural networks. Nature
    645(8081):639-647. https://doi.org/10.1038/s41586-025-09479-w. Peer-reviewed journal. Read:
    PMC12443606 full text.
29. Song T, Qian L (2025). Heat-rechargeable computation in DNA logic circuits and neural
    networks. Nature 646(8084):315-322. https://doi.org/10.1038/s41586-025-09570-2.
    Peer-reviewed journal. Read: PMC12507696 full text.
30. Sun C, Liu X, Zhong J, Zhou Q, Cheng J (2025). Reusable noncomplementary DNA-based neural
    network. J Am Chem Soc 147(38):34339-34349. https://doi.org/10.1021/jacs.5c04886.
    Peer-reviewed journal. Read: abstract (PubMed 40768299).
31. Lim H-W, Lee SH, Yang K-A, Lee JY, Yoo S-I, Park TH, Zhang B-T (2010). In vitro molecular
    pattern classification via DNA-based weighted-sum operation. Biosystems 100(1):1-7.
    https://doi.org/10.1016/j.biosystems.2009.12.001. Peer-reviewed journal. Read: abstract
    (PubMed 19995592).
32. Pei R, Matamoros E, Liu M, Stefanovic D, Stojanovic MN (2010). Training a molecular
    automaton to play a game. Nature Nanotechnology 5(11):773-777.
    https://doi.org/10.1038/nnano.2010.194. Peer-reviewed journal. Read: abstract (PubMed
    20972436).
33. Zhang C, Zhao Y, Xu X, Xu R, Li H, Teng X, Du Y, Miao Y, Lin H-C, Han D (2020). Cancer
    diagnosis with DNA molecular computation. Nature Nanotechnology 15(8):709-715.
    https://doi.org/10.1038/s41565-020-0699-0. Peer-reviewed journal. Read: abstract (PubMed
    32451504).
34. Pandi A, Koch M, Voyvodic PL, Soudier P, Bonnet J, Kushwaha M, Faulon J-L (2019). Metabolic
    perceptrons for neural computing in biological systems. Nature Communications 10:3880.
    https://doi.org/10.1038/s41467-019-11889-0. Peer-reviewed journal. Read: PMC6713752 full
    text. Partly verified: the 18 h whole-cell read-out; an 8 h cell-free time point was not
    found.
35. van der Linden AJ, Pieters PA, Bartelds MW, Nathalia BL, Yin P, Huck WTS, Kim J, de Greef TFA
    (2022). DNA input classification by a riboregulator-based cell-free perceptron. ACS Synth
    Biol 11(4):1510-1520. https://doi.org/10.1021/acssynbio.1c00596. Peer-reviewed journal.
    Read: PMC9016768 full text. Partly verified: the weight patterns differ between the direct
    and the coupled experiments. Not confirmed: the inhibitor concentrations for OR, majority
    and AND, the off-target level of input 2, and a model over-prediction in every case.
36. Baltussen MG, de Jong TJ, Duez Q, Robinson WE, Huck WTS (2024). Chemical reservoir
    computation in a self-organizing reaction network. Nature 631(8021):549-555.
    https://doi.org/10.1038/s41586-024-07567-x. Peer-reviewed journal. Read: PMC11254755 full
    text. The Results give a 20 min training period for the Lorenz task, the Methods
    ("Forecasting") 30 min.
37. Ghosh S, Baltussen MG, Knox AC, Haije R, Duez Q, Tsitsimeli AT, Chak MH, Beves JE, Huck WTS
    (2025). A recursive enzymatic competition network capable of multitask molecular
    information processing. Nature Chemistry 18(2):302-308.
    https://doi.org/10.1038/s41557-025-01981-y. Peer-reviewed journal. Read: PMC12872465 full
    text. Partly verified: the number of output features each task needed was not read. The
    light-pulse range of 2 to 8 min is the reported discrimination range; the Methods list 17
    tested periods from 30 s to 600 s.
38. Parrilla-Gutierrez JM, Sharma A, Tsuda S, Cooper GJT, Aragon-Camarasa G, Donkers K, Cronin L
    (2020). A programmable chemical computer with memory and pattern recognition. Nature
    Communications 11:1442. https://doi.org/10.1038/s41467-020-15190-3. Peer-reviewed journal.
    Read: PMC7080730 full text.
39. Rao R, Esposito M (2016). Nonequilibrium thermodynamics of chemical reaction networks:
    wisdom from stochastic thermodynamics. Physical Review X 6:041064.
    https://doi.org/10.1103/PhysRevX.6.041064. Peer-reviewed journal. Read: arXiv 1602.07257v3.
    Partly verified: the work relation (Eqs. 78-79) is confirmed; the Landauer-type
    inequalities of Section V C were not read.
40. Gorban AN, Yablonsky GS (2011). Extended detailed balance for systems with irreversible
    reactions. Chemical Engineering Science 66(21):5388-5399.
    https://doi.org/10.1016/j.ces.2011.07.054. Peer-reviewed journal. Read: arXiv 1101.5280v3.
41. Ederer M, Gilles ED (2007). Thermodynamically feasible kinetic models of reaction networks.
    Biophysical Journal 92(6):1846-1857. https://doi.org/10.1529/biophysj.106.094094.
    Peer-reviewed journal. Read: abstract only.
42. Ouldridge TE (2018). The importance of thermodynamics for molecular systems, and the
    importance of molecular systems for thermodynamics. Natural Computing 17(1):3-29.
    https://doi.org/10.1007/s11047-017-9646-x. Peer-reviewed journal, a review. Read: arXiv
    1702.00360v2. Partly verified: the off-rate statement holds "in some cases" only.
43. Craciun G, Johnston MD, Szederkényi G, Tonello E, Tóth J, Yu PY (2020). Realizations of
    kinetic differential equations. Mathematical Biosciences and Engineering 17(1):862-892.
    https://doi.org/10.3934/mbe.2020046. Peer-reviewed journal. Read: arXiv 1907.07266v2.
44. Poole W, Ouldridge TE, Gopalkrishnan M (2025). Autonomous learning of generative models with
    chemical reaction network ensembles. J R Soc Interface 22(222):20240373.
    https://doi.org/10.1098/rsif.2024.0373. Peer-reviewed journal. Read: PMC11771824 full text
    and arXiv 2311.00975v2; supplementary appendix H was not read.
45. Mehta P, Schwab DJ (2012). Energetic costs of cellular computation. Proc Natl Acad Sci USA
    109(44):17978-17982. https://doi.org/10.1073/pnas.1207814109. Peer-reviewed journal. Read:
    arXiv 1203.5426v3.
46. Ouldridge TE, Govern CC, ten Wolde PR (2017). Thermodynamics of computational copying in
    biochemical systems. Physical Review X 7(2):021004.
    https://doi.org/10.1103/PhysRevX.7.021004. Peer-reviewed journal. Read: arXiv 1503.00909v4.
47. Floyd C, Dinner AR, Murugan A, Vaikuntanathan S (2025). Limits on the computational
    expressivity of non-equilibrium biophysical processes. Nature Communications 16:7184.
    https://doi.org/10.1038/s41467-025-61873-0. Peer-reviewed journal. Read: PMC12325794 full
    text and arXiv 2409.05827v1.
48. Scellier B, Bengio Y (2017). Equilibrium Propagation: bridging the gap between energy-based
    models and backpropagation. Frontiers in Computational Neuroscience 11:24.
    https://doi.org/10.3389/fncom.2017.00024. Peer-reviewed journal. Read: arXiv 1602.05179v5.
    Partly verified: the gradient is exact only as the nudging goes to zero.
49. Stern M, Hexner D, Rocks JW, Liu AJ (2021). Supervised learning in physical networks: from
    machine learning to learning machines. Physical Review X 11(2):021045.
    https://doi.org/10.1103/PhysRevX.11.021045. Peer-reviewed journal. Read: arXiv 2011.03861v2.
50. Stern M, Murugan A (2023). Learning without neurons in physical systems. Annual Review of
    Condensed Matter Physics 14:417-441.
    https://doi.org/10.1146/annurev-conmatphys-040821-113439. Peer-reviewed journal, a review.
    Read: arXiv 2206.05831v1 (2022).
51. Gillespie DT (1977). Exact stochastic simulation of coupled chemical reactions. J Phys Chem
    81(25):2340-2361. https://doi.org/10.1021/j100540a008. Peer-reviewed journal. Read: a scan
    of the published article.
52. Gillespie DT (2001). Approximate accelerated stochastic simulation of chemically reacting
    systems. J Chem Phys 115(4):1716-1733. https://doi.org/10.1063/1.1378322. Peer-reviewed
    journal. Read: full text (Zenodo 1231854).
53. Kim S, Ji W, Deng S, Ma Y, Rackauckas C (2021). Stiff neural ordinary differential
    equations. Chaos 31(9):093122. https://doi.org/10.1063/5.0060697. Peer-reviewed journal.
    Read: arXiv 2103.15341v3.
54. Kopecz S, Meister A (2018). Unconditionally positive and conservative third order modified
    Patankar-Runge-Kutta discretizations of production-destruction systems. BIT Numerical
    Mathematics 58(3):691-728. https://doi.org/10.1007/s10543-018-0705-1. Peer-reviewed
    journal. Read: arXiv 1703.05052v1. Partly verified: the comparison with Runge-Kutta and
    Rosenbrock methods rests on one cited study of an MPRK22 scheme, and for the multi-reactant
    case the paper cites schemes with a nonlinear solve per step as well as explicit ones.
55. Lee I, Buse S, Winfree E (2025). Differentiable programming of indexed chemical reaction
    networks and reaction-diffusion systems. DNA 31, LIPIcs 347, 4:1-4:23.
    https://doi.org/10.4230/LIPIcs.DNA.31.4. Peer-reviewed conference. Read: full text.
    Partly verified: the Hopfield figures are confirmed; the Gray-Scott training difficulty had
    several causes, not stiffness alone.
56. Chen RTQ, Rubanova Y, Bettencourt J, Duvenaud D (2018). Neural ordinary differential
    equations. NeurIPS 31.
    https://proceedings.neurips.cc/paper/2018/hash/69386f6bb1dfed68692a24c8686939b9-Abstract.html.
    Peer-reviewed conference. Read: arXiv 1806.07366v6, a revision of 9 October 2026 whose
    correction of Fig. 3c is not in the NeurIPS version.
57. Gholaminejad A (also published as Gholami A), Keutzer K, Biros G (2019). ANODE:
    unconditionally accurate memory-efficient gradients for neural ODEs. IJCAI-19, pp. 730-736.
    https://doi.org/10.24963/ijcai.2019/103. Peer-reviewed conference. Read: arXiv
    1902.10298v3.
58. Ma Y, Dixit V, Innes MJ, Guo X, Rackauckas C (2021). A comparison of automatic
    differentiation and continuous sensitivity analysis for derivatives of differential
    equation solutions. IEEE HPEC 2021, pp. 1-9.
    https://doi.org/10.1109/HPEC49654.2021.9622796. Peer-reviewed conference. Read: arXiv
    1812.01892v2.
59. Ji W, Deng S (2021). Autonomous discovery of unknown reaction pathways from data by chemical
    reaction neural network. J Phys Chem A 125(4):1082-1092.
    https://doi.org/10.1021/acs.jpca.0c09316. Peer-reviewed journal. Read: arXiv 2002.09062v2.
60. Mordvintsev A, Randazzo E, Niklasson E (2023). Differentiable programming of chemical
    reaction networks. arXiv 2302.02714v1. https://arxiv.org/abs/2302.02714. Preprint, not peer
    reviewed; no peer-reviewed version found.
61. Rijal K, Mehta P (2025). A differentiable Gillespie algorithm for simulating chemical
    kinetics, parameter estimation, and designing synthetic biological circuits. eLife
    14:RP103877. https://doi.org/10.7554/eLife.103877. Peer-reviewed journal. Read: arXiv
    2407.04865v3. Partly verified: errors shrink with the smoothing only in theory; in the
    tests the error levels off and can rise again.
62. Deckard A, Sauro HM (2004). Preliminary studies on the in silico evolution of biochemical
    networks. ChemBioChem 5(10):1423-1431. https://doi.org/10.1002/cbic.200400178.
    Peer-reviewed journal. Read: abstract only.
63. Feng S, Ollivier JF, Swain PS, Soyer OS (2015). BioJazz: in silico evolution of cellular
    networks with unbounded complexity using rule-based modeling. Nucleic Acids Research
    43(19):e123. https://doi.org/10.1093/nar/gkv595. Peer-reviewed journal. Read: PMC4627059
    full text.
64. Lones MA, Fuente LA, Turner AP, Caves LSD, Stepney S, Smith SL, Tyrrell AM (2014).
    Artificial biochemical networks: evolving dynamical systems to control dynamical systems.
    IEEE Trans Evol Comput 18(2):145-166. https://doi.org/10.1109/TEVC.2013.2243732.
    Peer-reviewed journal. Read: accepted version (White Rose eprint 75277).
65. Vasić M, Soloveichik D, Khurshid S (2020). CRN++: molecular programming language. Natural
    Computing 19(2):391-407. https://doi.org/10.1007/s11047-019-09775-1. Peer-reviewed journal.
    Read: arXiv 1809.07430v3.
66. Lakin MR, Youssef S, Polo F, Emmott S, Phillips A (2011). Visual DSD: a design and analysis
    tool for DNA strand displacement systems. Bioinformatics 27(22):3211-3213.
    https://doi.org/10.1093/bioinformatics/btr543. Peer-reviewed journal. Read: PMC3208393 full
    text.
67. Loman TE, Ma Y, Ilin V, Gowda S, Korsbo N, Yewale N, Rackauckas C, Isaacson SA (2023).
    Catalyst: fast and flexible modeling of reaction networks. PLoS Comput Biol
    19(10):e1011530. https://doi.org/10.1371/journal.pcbi.1011530. Correction: PLoS Comput Biol
    21(6):e1013175 (2025), which replaces a duplicated panel of Fig. 3. Peer-reviewed journal.
    Read: PMC10584191 full text.
68. Robinson M, Allmont A (2026). diffsol: Rust crate for solving differential equations.
    Journal of Open Source Software 11(117):9384. https://doi.org/10.21105/joss.09384.
    Peer-reviewed journal. Read: the full paper.
69. Monniaux D (2008). The pitfalls of verifying floating-point computations. ACM TOPLAS 30(3),
    Article 12. https://doi.org/10.1145/1353445.1353446. Peer-reviewed journal. Read: arXiv
    cs/0701192v5.
70. Rust RFC 3514: Float semantics (start date 2023-10-14). The Rust RFC Book.
    https://rust-lang.github.io/rfcs/3514-float-semantics.html. Documentation, not peer
    reviewed. Partly verified: `%` follows fmod, float-to-integer casts round toward zero, the
    quiet bit of a NaN can also vary, and 32-bit x86 without SSE2 deviates.
71. The Rust Standard Library, primitive type f64, std 1.99.1, as fetched on 2026-10-31.
    https://doc.rust-lang.org/std/primitive.f64.html. Documentation, not peer reviewed.
72. libm 0.2.16, Rust crate documentation, docs.rs build of 13 September 2026.
    https://docs.rs/libm/0.2.16/libm/. Documentation, not peer reviewed.
73. Doty D, Hajiaghayi M (2015). Leaderless deterministic chemical reaction networks. Natural
    Computing 14(2):213-223. https://doi.org/10.1007/s11047-014-9435-8. Peer-reviewed journal;
    published online 10 June 2014. Read: the authors' copy of the journal version,
    https://www.cs.ucdavis.edu/~doty/papers/ldcrn-journal.pdf, and the CaltechAUTHORS record
    CaltechAUTHORS:20150624-093546162.
74. Arora R, Basu A, Mianjy P, Mukherjee A (2018). Understanding deep neural networks with
    rectified linear units. ICLR 2018. Peer-reviewed conference. Read: arXiv 1611.01491v6,
    https://arxiv.org/abs/1611.01491.

**Left out.** Two works found by the search were left out, because no claim from them could be
checked: Lopez, Wang and Seelig (2018), Nature Chemistry 10(7):746-754, and Dittrich, Ziegler
and Banzhaf (2001), Artificial Life 7(3):225-275.
