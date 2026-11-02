# Spec: a Soul model that acts early (candidates and a screening plan)

Status: needs-triage. This is a design note for the owner. Nothing is built, and no candidate
is chosen yet. The owner decides which candidate, if any, gets built and screened.

It follows the owner's decision of 2026-11-02 to drop the counted experiment of step 2. The
pilot and that decision are on the branch `claude/schritt2-pilot`: `docs/pilot-step2/README.md`,
`docs/PREREG_STEP2_SOUL.md` and `.scratch/seele-und-kappa/rule/diagnose.txt`. The terms are
those of `CONTEXT.md`. The Soul, its settings and the arms S, V, K, E and N are those of
`.scratch/seele-und-kappa/spec.md` (here "spec 9.1" and so on) and of ADR 0009. The pass time is
that of `docs/PREREG_STEP1_CIRCUITS.md`, section 7.

Three words are new in this note, and only here. A Worker's **preference** for a family is a
number that its Soul gives each family. A text's **lean** adds up each preference minus 1 over
the families that the text names (both in section 4.1). A **promising Worker** is one whose
system prompt names two of the three families Linear, Phase and Hadamard.

## 1. The problem

In step 2 the Soul changes an outcome in only two ways: a complex Stage vote whose winner
differs from the count's, or a reply of a self-checking Worker in which the phases choose among
equally cheap circuits that two or more families found. Arms with the same genes share every
random number until one of these happens. In the pilot both were rare and late. In arm S the
first flipped vote came in generation 12 to 536 (median 57.5), the first phase-decided reply in
generation 41 to 458 (median 147.5), and only 7.2 % of the generations before the pass held
either; 3 of the 12 Runs had neither before they passed. So on 4 of 12 seeds the arms S, V and
K all passed before they parted, and among the untied seeds arm S passed first half the time.
Our new count over the 60 pilot Runs adds a deeper reason: the pass waits for one event of
variation, the birth of a Worker whose prompt names Linear, Phase and Hadamard together. In 50
of the 57 Runs that passed, the pass time is the generation of the first such Worker, and no Run
passed before it. The step 2 Soul acts on answers, not on variation. So even a frequent vote
could move the pass only through small changes of fitness. A Soul model that is to change the
breeding measurably must act on variation or on selection, from generation 1.

## 2. What the pilot Runs show

We read the 60 compressed pilot Runs (seeds 901 to 912, arms S, V, K, E and N) again for this
note; section 9 says how. Table 1 gives what we measured.

Table 1: measured in the pilot Runs.

| Measure | Value |
|---|---|
| Runs that passed | 57 of 60 |
| Pass time equal to the generation of the first Worker naming Linear, Phase and Hadamard | 50 of 57; in the other 7 the pass came 2 to 214 generations later, never earlier |
| How that first Worker was made (58 Runs made one) | SectionReplace 22, SectionAdd 13, section crossover 22, StageAdd 1; in all 58 a parent already held a promising Worker |
| First promising Worker | generation 1 to 41, median 8 |
| Phrase draws (SectionReplace, SectionAdd) on a parent with a promising Worker, before the first Worker with all three | 4 928 over 60 Runs |
| ... that changed a promising Worker | 32.2 % (1 586) |
| ... of those, that brought the missing family | 8.8 % (139) |
| ... of those, that lost one of the two families | 57.5 % (912) |
| ... of those, that brought the missing family and kept both | 2.2 % (35), so 0.71 % of the 4 928 draws |
| Crossovers on a parent with a promising Worker, arm S, same period | 3 883; the partner named the missing family in 64 %; 0.10 % made the Worker with all three |
| Children per generation, generations 1 to 50, arm N | stage crossover 2.01, section crossover 1.86, SectionReplace 0.77, SectionAdd 0.47, the twelve other operators 2.89 |
| The step 2 Soul, arm S, per generation and Run, generations 1 to 50 | 0.97 complex votes, 0.08 flipped votes, 0.01 phase-decided replies; 4.2 % of generations with a flip or a phase-decided reply (15.1 % over generations 1 to 1000) |
| Arm S before the pass | 7.2 % of 3 399 generations with a flip or a phase-decided reply; 3 of 12 Runs with none |

Three findings follow.

- **The pass waits for the third family.** A promising Worker exists early, and the wait is for
  a change that adds the third family to it without losing the other two. Most attempts fail:
  two in three change another Worker, and more than half of those that change a promising Worker
  overwrite one of its two families.
- **The step 2 Soul acts on answers.** A flipped vote changes which answer a Stage passes on, and
  a phase-decided reply changes which circuit the designer gives. Neither makes a new Worker.
  Both reach the pass only through fitness, and both are rare in the early generations.
- **A new Soul must act on variation or selection.** Section crossover, SectionReplace and
  SectionAdd make about 3.1 children per generation from generation 1 on, and every child needs
  a parent. Every candidate below acts there. They differ in how strongly they can touch the wait
  for the third family.

## 3. How large an effect a test can confirm

Once two arms part early, the two Runs of a seed no longer share outcomes. Their pass times then
behave like two independent draws from a wide distribution. We pooled the 39 Runs of arm A of
step 1 (27 counted seeds) and of arm N of the pilot (12 seeds). 36 of them passed, with a median
pass time of 199 generations and quartiles of 62 and 296; 3 did not pass by generation 1000. If
one arm's Runs passed k times sooner than these, P(first | untied) would be 0.60 for k = 1.5,
0.65 for k = 2 and 0.72 for k = 3 (our calculation over all pairs of the 39 Runs).

The rule of the step 2 draft (p₁ = 3/4, proof at E ≥ 200, failure at E ≤ 1/20, at most 120
pairs) then proves a claim with these probabilities (our exact calculation):

| P(first \| untied) | Power, no ties | Mean pairs, no ties | Power at the pilot's tie rate 5/12 |
|---|---|---|---|
| 0.60 | 0.14 | 49 | 0.10 |
| 0.65 | 0.44 | 59 | 0.30 |
| 0.70 | 0.79 | 53 | 0.61 |
| 0.72 | 0.88 | 47 | 0.74 |
| 0.75 | 0.96 | 40 | 0.88 |

Early parting removes the ties by construction, which helps. Still, a confirmatory test on the
pass time needs a speed-up of about 2.5 to 3. A smaller effect can be seen in process measures
within 8 seeds, but it cannot be confirmed on the pass time within 120 pairs. A process measure
here is a count along the way to the pass, such as how often a change keeps both families of a
promising Worker. This bound guides the ranking in section 7.

## 4. Candidates

Each candidate keeps the Soul of step 2: inherited phases, one per family, passed on by
Souletics, never shown to a Worker or to the Mutator. Each adds one place where the Soul changes
an outcome. Each keeps the settings, the call cap and the evaluation of arm S, and the Soul adds
no model call. The calls per Run can still differ between arms, as in the pilot, since different
children make different calls; the analysis reports them. The new arm letters avoid A and R of
step 1 and S, V, K, E and N of step 2.

### 4.1 W: Soul-weighted variation

**Mechanism.** The Soul steers the section operators. For a Worker with birth Soul P, family f
gets the preference w_f = |u(p_f) + 1|² / 2 = 1 + cos(2π p_f / 65536). It is the intensity of
the family's phasor added to a fixed reference phasor 1, halved. It lies between 0 and 2, and its
mean over uniform phases is 1, so the operators stay unbiased on average. With the sign readout
of arm V it is 0 or 2. A text's lean is L = Σ (w_f − 1) over the families its words name, by the
stems of `designer::FAMILY_STEMS`; a text that names no family has L = 0. Three choices then
follow the lean, each weight at least ε = 0.05 (a placeholder), so no choice becomes impossible:

1. SectionReplace and SectionAdd choose the Worker with weight 1 + L(its whole prompt).
2. SectionReplace chooses the section to overwrite with weight 1 − L(that section), so a Worker
   protects texts it prefers.
3. Both draw the phrase with weight 1 + L(phrase). SectionAdd multiplies this by the weight of
   choice 2 for the section that the phrase would overwrite.

Section crossover takes the other source's section with probability (1 + L(theirs)) / ((1 +
L(mine)) + (1 + L(theirs))) instead of 1/2. The Soul that steers a change is the birth Soul of
the parent Worker that the operator changes; for section crossover it is the first parent's
Worker. Souletics then passes that Soul to the child, so the preference that shaped a change
travels with its result. Everything else stays as in arm S: the complex vote, the designer's
phase-decided replies, Observations and imprinting. W acts on the first child that these
operators make. In the pilot the first SectionReplace or SectionAdd child came in generation 1 in
50 of the 60 Runs and in generation 2 in the other 10.

**Reason.** This is self-adaptation, the standard way of evolution strategies to tune their own
mutation. Each individual carries strategy parameters that steer its mutation, and "they have a
higher probability of survival when they 'encode' object parameter variations that produce (on
average) fitter object parameters" (Beyer and Schwefel 2002, section 4.2.1). Spears (1995) let an
inherited bit choose the form of crossover. Promptbreeder evolves the mutation-prompts of a
language model together with its task-prompts, and removing these self-referential parts hurt
in nearly all of its cases (Fernando et al. 2023, a preprint). In W the preference is such an
inherited strategy parameter. Table 1 shows where it can help. Two in three draws on a promising
parent change another Worker, and 57.5 % of the draws that change a promising Worker overwrite
one of its two families. Linear and Phase pay on their own: swap needs Linear, cs needs Phase
and CCZ needs both (spec 5.2). Fitness counts partial credit too, since on circuits it uses the
graded value of each answer. So a lineage whose Soul prefers these families protects them, keeps
its fitness and breeds more, and its preference spreads before the last step. The third family
needs no preference of its own: protecting the other two already raises the success of a draw.
For an inherited weight that protects prompt texts from being overwritten, no source found; the
sources above cover inherited weights on variation in general.

**Controls.**

- **W0, not inherited.** Arm W with `copy: false` and `imprint: false`. Every child Worker gets
  a fresh uniform Soul and keeps it. The weights then have the same mean and act as often, but
  carry nothing from parent to child: a fixed random bias that is not inherited. W0 also catches
  the effect that Spears (1995) reports in his abstract, an improvement that helps a
  non-adaptive algorithm as well.
- **N, no Soul.** Today's unweighted operators.
- **Later, if W passes the screen.** WV with the sign readout, and W with current Souls instead
  of birth Souls; then Observations and imprinting move the preference, and κ = 0 is the control.
  Arm K is no control for W itself, since W reads birth Souls, which κ never moves.

**Predicted effect and frequency.** In generations 1 to 50 the pilot made 1.24 children per
generation by SectionReplace or SectionAdd and 1.86 by section crossover (arm N). So W steers
about 3 children per generation from generation 1, with two or three weighted choices each. We
predict that W and N part in generation 1 or 2 on every seed. On the way to the pass we predict:

- the mean preference for Linear, Phase and Hadamard among the parents' Workers rises above 1
  within 50 generations in W and stays near 1 in W0;
- among draws that change a promising Worker, the share that loses one of its families falls from
  57.5 % toward 20 % in W;
- among draws on a promising parent, the share that changes a promising Worker rises from 32 %
  toward 50 %.

If the last two reach these values, a draw makes the Worker that the pass waits for 3 to 5 times
as often (our rough calculation from the factors of table 1). Phrase draws made 35 of the 58
first such Workers in the pilot. If their rate rises 3 to 5 times and the crossover rate stays,
the wait shrinks by a factor of 2.2 to 3.4. That is the range that section 3 calls confirmable.
It is a best case: the preference must adapt first, and if it does not, W behaves like W0, a
random bias with no gain.

**Screening test.** Arms W, W0 and N on seeds 913 to 920, with the settings of the pilot (1000
generations, a cap of 13 000 000 calls, two Runs at a time with `RAYON_NUM_THREADS=2`): 24 Runs,
about 2 hours. The criteria are fixed before the first Run:

1. **Early.** W parts from N before either Run passes on at least 7 of 8 seeds (the divergence
   generation of `diagnose.py`), and the weights change a choice in generation 1 or 2 on all 8.
2. **Self-adaptation.** The mean preference for Linear, Phase and Hadamard over the Workers of
   the parents chosen in generations 41 to 50 is above 1.2 in W on at least 6 of 8 seeds, and
   below 1.1 in W0 on at least 6 of 8.
3. **Protection.** Pooled over the seeds, before the first Worker with all three families, at
   most 40 % of the draws that change a promising Worker lose one of its families in W; in W0
   and N the share stays near the pilot's 57.5 %.
4. **Direction, reported.** The seeds on which W passes before N, and before W0. With no effect,
   W passes first on at least 5 of 8 seeds with probability 0.36, so this is a signal, not a
   test.

Go if 1, 2 and 3 hold and N passes before W on at most 4 of the 8 seeds; then the owner may
pre-register a confirmatory test (section 8). Stop if 1 or 2 fails. In every other case the
owner decides. Criteria 2 and 3 carry the screen, since each counts hundreds of events per Run.

**Code size and risks.** About 300 to 450 lines of Rust and 250 lines of tests: weighted choices
in `src/cascade/variation.rs` (the three choices and `cross_worker`), the preference passed from
`src/cascade/arena.rs` to the operators, a hook in the gene pool that names the families of a
text (circuits: the stems of `designer::FAMILY_STEMS`; a grid Run with this switch is refused at
first), a field `variation` in `SoulConfig` (off by default and left out of the JSON when off, so
every existing config, hash, log and test stays as it is), and one count per generation in the
`soul` group: the choices whose pick the weights changed. Risks:

- W reopens an option that ADR 0009 rejected: genetic operators that see the Souls. The ADR gave
  two reasons. A crossover child with one parent's genes and the other's Souls would pass the
  novelty test, and the random stream would change. W keeps the first safeguard, since the
  operators get genes plus preferences and novelty stays by gene id. It gives up the second on
  purpose: the arms part in generation 1 and no longer share random numbers.
- New preferences arise slowly. A birth moves a phase by at most 512 units (2.8 degrees), and the
  pilot had only 7 to 17 Soul origins per Run. So selection works mostly on the phases of the 4
  seed Souls, which mixing at section crossover recombines family by family. If all 4 seed Souls
  disfavour one of the three families, W may be slower than N on that seed. Under uniform phases
  this happens for a given family with probability 1/16, and for at least one of the three with
  probability 0.18. A larger `phase_drift` would speed adaptation and add noise; the first screen
  keeps 512.
- The preference may converge too early. Glickman and Sycara (2000) saw self-adapted mutation
  rates fall below the point where search still works. Protected texts can freeze a lineage,
  and the families that CH and Fredkin need (Search, Meet, Compose, Simplify) may vanish. The
  pass needs 5 of 9 targets, so this would hurt the share D more than the pass time.
- The gain may come from protection as such. A plain rule that never overwrites a text naming a
  family would help every arm, without a Soul. W0 tests the Soul against a random bias, not
  against such a rule, and a hand-written rule may beat the Soul. An optional fourth arm, every
  preference fixed at 2 and no inheritance, would show this for 8 more Runs.
- `CONTEXT.md` says that a Soul is seen only through its Character. W lets it shape the genome
  through variation, so W needs an amendment of ADR 0009 and of that glossary entry.

### 4.2 I: interference among a Worker's own families (a variant of W)

**Mechanism.** As W, except the phrase weight of choice 3: a phrase gets ω = |Σ u(p_f)|² / m,
summed over the m families that the phrase and the Worker's prompt name together. ω has mean 1
over uniform phases. It favours a phrase whose families are in phase with the families the Worker
already holds. This keeps the interference of step 2 inside one Soul, and the sign readout gives
a different pattern from the complex one.

**Reason.** No source found for interference as a rule of operator choice. The hope is that
selection favours coherent Souls, which favour every combination of families.

**Controls, test, size.** As W, with W itself as a further control. About 60 lines on top of W.

**Predicted effect.** At most W's. The phase of the third family relative to the other two is
random until a Worker with all three exists, so ω cannot learn it before the pass. I is worth a
screen only after W shows self-adaptation.

### 4.3 M: the Soul chooses the crossover partner

**Mechanism.** Today a crossover draws three candidate partners by the parent weights and mates
the one with the best peer rating. In M it mates the one whose Soul overlaps least with the
parent's. The overlap of two members is the mean, over all pairs of one Worker of each, of
|(1/n) Σ_f u(p_f) conj(u(q_f))|²: 1 for equal Souls and about 1/n for independent ones. The three
draws stay as they are; only the choice among them changes. M acts at every crossover, 3.9 per
generation in generations 1 to 50. If ratings and Souls are unrelated, M picks another partner
than the rating would in about 2 of 3 crossovers, so the arms part in generation 1.

**Reason.** Mating restrictions against similar parents keep diversity. Incest prevention is one
(Eshelman and Schaffer 1991; only the bibliographic record seen). Negative assortative mating
beat a simple genetic algorithm on a royal road function (Fernandes and Rosa 2001). Spears (1994)
formed subpopulations by inherited tag bits instead of distances; Souletics makes the Soul such a
tag of ancestry. The hope is that a partner from another lineage brings the missing family.

**Pilot data against it.** In arm S, before the first Worker with all three families, the partner
of a parent with a promising Worker already named the missing family in 64 % of 3 883
crossovers. By the parents' overlap (below 0.2, 0.2 to 0.4, 0.4 to 0.6, 0.6 to 0.8, at least
0.8) that share was 73, 67, 59, 65 and 46 % in arm S, but 43, 53, 70, 72 and 66 % in arm V. And
a crossover whose partner named the family made the needed Worker in only 4 of 2 485 cases
(0.16 %). The partner is rarely what is missing; the mechanics of section crossover are. Sareni
and Krähenbühl (1998) also note that crossovers between niches often give poor children.

**Controls.** M0, the same rule with `copy: false` and `imprint: false` (a random one of the
three draws); N (the best peer rating); M+, the most similar Soul, as the opposite direction.

**Predicted effect.** Small: at most from 64 % to about 73 % of partners naming the missing
family, on the channel that made 22 of the 58 first such Workers. We expect no visible change
of the pass time.

**Screening test.** As W, with M, M0 and N; the process measure is the share of crossovers on a
promising parent whose partner names the missing family.

**Code size and risks.** About 40 to 60 lines in `partner` of `src/cascade/arena.rs` and 80 lines
of tests. Mating by peer rating is part of Darwin Gödel selection in this repository (Q2), and M
gives it up.

### 4.4 F: fitness sharing by Soul origin

**Mechanism.** The parent weights of Darwin Gödel selection are divided by a niche count: the
number of archive members whose Soul has the same origin (Hoppe's colour, which the `soul` group
already logs as `origins`). A smooth form would use the overlap of section 4.3 instead. F acts at
every parent draw, 8 or more per generation, from generation 1.

**Reason.** Fitness sharing keeps stable subpopulations at several peaks (Goldberg and Richardson
1987; Sareni and Krähenbühl 1998). Spears (1994) formed such subpopulations by tags. MAP-Elites
keeps the best solution of each cell of a chosen feature space and often finds a better overall
solution (Mouret and Clune 2015, a preprint). The pass counts targets that any bred Cascade
reaches, so lineages that search in parallel might help.

**Data against it.** In step 1, uniform parent choice passed first in only 5 of 25 untied pairs
against Darwin Gödel selection (`docs/STEP1_CIRCUITS.md`): spreading the choice slowed the pass.
The DGM paper found that always taking the best is worse too (Zhang et al. 2026, appendix A.3:
39.7 % against 50.0 % on SWE-bench), so the best spread lies between the two. F moves it toward
uniform. And the pass waits for one Worker in one lineage, not for coverage across lineages. Of
the 55 pilot Runs that reached 7 of the 9 targets, the sixth target came in the generation of
the fifth in 49 Runs, and the seventh too in 37.

**Controls.** F0, sharing by a label drawn at birth and not inherited (`copy: false`); N.

**Predicted effect.** A slower pass: P(F first) below 1/2.

**Code size and risks.** 60 to 100 lines. Sharing by origin is cheap; sharing by overlap costs
time in the square of the archive size (8 004 members by generation 1000), so the origin form
comes first.

### 4.5 D: the Soul orders the designer's methods

**Mechanism.** The simulated designer tries the families its prompt names in the order of their
first mention. In D it tries them in the order of the Worker's preference w_f, highest first. D
acts on every solve call from generation 0.

**Reason.** A consistent individual bias in behaviour can be adaptive (Wolf et al. 2007, animal
personalities). In the designer, the order decides what the thinking budget reaches first.

**Data against it.** The pass waits for a Worker that names the three families, not for their
order. Once such a Worker exists, the pass follows in the same generation in 50 of 57 Runs. The
designer is also the positive control of step 1: its calibration tests pin routes, and a Soul
inside it has no counterpart with real models, where the Soul never reaches a prompt.

**Code size and risks.** 30 to 50 lines in `src/cascade/designer.rs`, but the calibration of step
1 must be derived again. We advise against D.

### 4.6 Directions we considered and dropped

- **Division of labour inside a Stage.** The Soul could decide which Worker of a Stage answers or
  thinks most, as response thresholds split tasks among the members of an insect colony
  (Bonabeau, Theraulaz and Deneubourg 1996). It acts on answers, like the step 2 Soul, so it
  cannot reach the wait for the third family.
- **An archive with one cell per kind of Soul.** MAP-Elites needs a feature that describes
  behaviour (Mouret and Clune 2015). The Soul describes ancestry, so such cells would be
  candidate F with more rules.
- **Other placeholders of step 2.** The pilot showed that no placeholder moves the first chance of
  the vote or of the designer (`docs/pilot-step2/README.md`, "Degenerate? No second batch").
- **Teaching.** It is inert with the simulated models (spec 10.2).

## 5. Lever: a change of endpoint

**Later circuit endpoints add little.** The Worker that names Linear, Phase and Hadamard unlocks
QFT2, Toffoli, Peres, RTOF and CCiX together. Of the 55 pilot Runs that reached 7 of the 9
targets, the sixth target came in the generation of the fifth in 49, and the seventh too in 37.
So the time to 6 or to 7 of the 9 targets still tied S and N on 7 and 6 of 12 seeds, mostly
because both Runs got there before they parted. The share D at generation 1000 was 7/9 in all 48
Runs of S, K and E, so D could not have told these arms apart on any seed.

**Forking paths.** An endpoint chosen after seeing data inflates false findings, also when nobody
fishes on purpose (Gelman and Loken 2014; Simmons, Nelson and Simonsohn 2011). The pilot gives an
example. Had we chosen "the time to 7 of the 9 targets" after looking, arm S would lead arm V by
6 to 2 on the same 12 seeds on which the pass time, fixed in advance, gave 4 to 4. That lead says
nothing. Any new endpoint must be fixed in a pre-registration before its first Run and tested on
seeds that no pilot or screen used. Seeds 901 to 923 are pilot seeds; 913 to 920 serve the
screens of this note and never count.

**Process measures screen, they do not confirm.** The process measures of section 4 count
hundreds of events per Run, so 8 seeds suffice to see whether a mechanism works. They answer a
narrower question than the pass time: whether the mechanism does what it should, not whether
breeding gets faster.

**Advice.** Keep the pass time as the confirmatory endpoint for a candidate that acts on
variation. Use process measures only to screen. Fix any change before the Runs, on fresh seeds.

## 6. Lever: a change of domain

**Grids with the simulated grid models, screened here.** This lever needs no code:
`--domain grid --soul s` runs today. The grid judge flips a coin whenever both answers or
neither are explained (spec 11.3), so the judges of a Stage split often. We ran arms S, V, K and
N on seeds 913 to 920 of the grid Domain for 300 generations each, with `--search 33 --select 22
--children 8 --max-calls 13000000`, two Runs at a time with `RAYON_NUM_THREADS=2` and
`nice -n 5`. Seed 913 ran 1000 generations in arms S and N. All 32 Runs passed `--verify`.

Table 2: the grid screen, 8 seeds, generations 1 to 300.

| Measure | Arm S | Arm V | Arm K |
|---|---|---|---|
| First complex vote | generation 1 to 10, median 1.5 | the same | the same |
| First flipped vote | 1 to 30, median 3.5 | 2 to 177, median 11.5 | 1 to 30, median 3.5 |
| Generations with a flip, of the first 50 | 1 to 46, median 21 | 0 to 19, median 2.5 | 1 to 40, median 23.5 |
| Phase-decided replies | none | none | none |
| Generation in which the arm parts from S | – | 3 to 39, median 6.5 | 2 to 39, median 9 |

Arm N parted from S in generation 3 to 39, median 4, and in generation 3 on 4 of the 8 seeds.
On circuits the same arm S flipped its first vote in generation 12 to 536 (median 57.5) and had a
flip or a phase-decided reply in only 4.2 % of the first 50 generations. A 300-generation grid
Run took 43 to 64 seconds, two at a time. A 1000-generation grid Run took 8.5 minutes in arm N
and 11.4 in arm S, about as long as on circuits.

So on grids the step 2 Soul acts early and often, and the arms part before generation 40 on
every seed. Whether it changes the breeding is open. For description only: the best Cascade's
solved count on the 33 search puzzles at generation 300 was higher in S than in N on 3 seeds,
lower on 4 and equal on 1. Three cautions follow.

- Grids have no pre-registered endpoint. Any endpoint read off these 32 Runs would be a forking
  path (section 5), so ticket 03 fixes one before any new Run and uses fresh seeds.
- The grid lever leaves the Soul acting on answers. Section 2 argues that this reaches breeding
  only through fitness. On grids, though, the vote decides many more answers, so an effect there
  is possible.
- Seeds 913 to 920 have now run on grids. They stay free for the circuit screen of W, since a
  grid Run says nothing about a circuit Run of the same seed.

**Real language models.** One model's samples often end in different answers: self-consistency
samples several reasoning paths, and their majority vote raised accuracy on GSM8K by 17.9 points
(Wang et al. 2023). So the Workers of a Stage would split more often, and the complex vote would
act more often. But with real models the Soul acts only in the vote, since phases never reach a
real model (spec 11.1). It would still act on answers, not on the variation that the pass waits
for. And every Run needs the rented GPU machine (an H200 rented by the hour,
`docs/PREREG_BREEDING_VS_VOTE.md`, section 9), while a confirmatory test needs 50 to 120 pairs.
We advise against this lever until a candidate has worked on a simulated Domain.

## 7. Ranked recommendation

The candidates, best first:

1. **W, Soul-weighted variation.** Build it and screen it first (tickets 01 and 02). It is the
   only candidate that acts on the event the pass waits for, and it acts from generation 1 on
   every seed. Its process measures show within 8 seeds whether the Soul adapts. Its best case,
   a pass 2.2 to 3.4 times sooner, reaches the confirmable range of section 3. It costs about 300
   to 450 lines of Rust and 2 hours of screening.
2. **I, interference among a Worker's own families.** Only after W shows self-adaptation; it
   shares W's code.
3. **M, the Soul chooses the crossover partner.** Cheap, and it acts early, but the pilot data
   leave it little to change.
4. **F, fitness sharing by Soul origin.** It acts early, but we predict a slower pass.
5. **D, the Soul orders the designer's methods.** It touches the positive control and not the
   wait. Not advised.

The levers:

- **The grid Domain** needs no code, and the step 2 Soul as built acts there from generation 1
  (section 6). It is the cheapest way to learn whether the Soul of step 2 changes breeding at
  all when it acts often. It needs an endpoint fixed before any further Run, and fresh seeds
  (ticket 03). It can run beside W, since it touches no code.
- **A new endpoint on circuits** helps only as a screen; the pass time stays the confirmatory
  endpoint (section 5).
- **Real language models** wait until a candidate has worked on a simulated Domain.

Our recommendation for the first screening is W, with the grid lever as a second track that
needs no build. The owner decides which of the three tickets start.

## 8. After a screen: the confirmatory step

If W passes its screen, a pre-registration in the form of step 1 follows: arm W against W0 (claim:
the inherited Soul helps) and arm W against N (claim: the model helps), the pass time as the
endpoint, the sequential rule with p₁ fixed from the screen, a fresh seed list fixed in advance,
and N_max from the table of section 3. Early parting makes ties rare, so the table's column
without ties applies. The screen's seeds 913 to 920 never count.

## 9. How we counted

- **Pilot Runs.** We read each Run's `events.jsonl.xz` once, through `xz -dc` and Python's
  standard library with `python3 -I`, and never wrote a decompressed Run to disk. A Worker names
  a family when its section texts, lower-cased, hold one of the family's stems in
  `designer::FAMILY_STEMS`, as the designer reads them. We recomputed every pass time from
  `rediscovery.json` by section 7 of the step 1 pre-registration; all 60 agree with the
  README's table.
- **Draws on a promising parent.** For every SectionReplace or SectionAdd child up to the first
  Worker with all three families, we compared each promising Worker of the parent with the Worker
  at the same position in the child (these operators keep positions).
- **Crossovers and overlap.** For every crossover child up to that Worker, the overlap of the two
  parents is the mean of section 4.3 over their Workers' birth Souls.
- **Speed-up and power.** P(first | untied) for a speed-up k compares all ordered pairs of the 39
  Runs, the faster arm's pass time divided by k and rounded up, "never" later than every
  generation. Power comes from exact dynamic programming over the counts of wins and losses
  under the rule of the step 2 draft.
- **Grids.** The binary was built with `cargo build --release --bin brut-breed` at commit
  `4e0110b`, whose `src/` equals that of the pilot's commit `e315220`; its SHA-256 is the
  pilot's, `898b7b40…c220`. Each Run was checked with `--verify`, then its `events.jsonl` and
  `snapshot.json` were compressed with `xz -6`. Per seed we compared each arm's newborns with
  arm S's by the divergence rule of `diagnose.py`, reading genes from the `born` events without
  `phase_state` and `coupling`, and counted the vote tallies of the `born` events per generation.
  Every count of table 2 falls within generations 1 to 300, also for the two 1000-generation
  Runs of seed 913.
- **Scripts.** The scripts are not in the repository. Ticket 02 adds those that the screen needs,
  with their outputs.

## 10. Sources

Each entry says what we saw of it.

- Beyer, H.-G. and Schwefel, H.-P. (2002). Evolution strategies: a comprehensive introduction.
  Natural Computing 1, 3–52. doi:10.1023/A:1015059928466. Section 4.2 read.
- Bonabeau, E., Theraulaz, G. and Deneubourg, J.-L. (1996). Quantitative study of the fixed
  threshold model for the regulation of division of labour in insect societies. Proceedings of
  the Royal Society B 263 (1376), 1565. doi:10.1098/rspb.1996.0229. Abstract read.
- Eshelman, L. J. and Schaffer, J. D. (1991). Preventing premature convergence in genetic
  algorithms by preventing incest. Proceedings of the Fourth International Conference on Genetic
  Algorithms, 115–122. Only the bibliographic record seen.
- Fernandes, C. and Rosa, A. C. (2001). A study on non-random mating and varying population size
  in genetic algorithms using a royal road function. Proceedings of the 2001 Congress on
  Evolutionary Computation, 60–66. doi:10.1109/CEC.2001.934371. Abstract read.
- Fernando, C., Banarse, D., Michalewski, H., Osindero, S. and Rocktäschel, T. (2023).
  Promptbreeder: self-referential self-improvement via prompt evolution. arXiv:2309.16797, a
  preprint. Sections 1 to 6 read.
- Gelman, A. and Loken, E. (2014). The statistical crisis in science. American Scientist,
  November–December 2014. Read.
- Glickman, M. R. and Sycara, K. (2000). Reasons for premature convergence of self-adapting
  mutation rates. Proceedings of the 2000 Congress on Evolutionary Computation, 62–69.
  doi:10.1109/CEC.2000.870276. Abstract read.
- Goldberg, D. E. and Richardson, J. (1987). Genetic algorithms with sharing for multimodal
  function optimization. Proceedings of the Second International Conference on Genetic
  Algorithms, 41–49. Abstract read.
- Mouret, J.-B. and Clune, J. (2015). Illuminating search spaces by mapping elites.
  arXiv:1504.04909, a preprint. Sections 1 to 4 read.
- Sareni, B. and Krähenbühl, L. (1998). Fitness sharing and niching methods revisited. IEEE
  Transactions on Evolutionary Computation 2 (3), 97–106. doi:10.1109/4235.735432. Sections 1
  and 2 read.
- Simmons, J. P., Nelson, L. D. and Simonsohn, U. (2011). False-positive psychology:
  undisclosed flexibility in data collection and analysis allows presenting anything as
  significant. Psychological Science 22 (11), 1359–1366. doi:10.1177/0956797611417632. Abstract
  read (PubMed).
- Spears, W. M. (1994). Simple subpopulation schemes. Proceedings of the Third Annual Conference
  on Evolutionary Programming, 296–307. Abstract read.
- Spears, W. M. (1995). Adapting crossover in evolutionary algorithms. Evolutionary Programming
  IV: Proceedings of the Fourth Annual Conference on Evolutionary Programming, MIT Press.
  doi:10.7551/mitpress/2887.003.0035. Abstract read; the inherited bit that chooses the form of
  crossover is as later papers that cite it describe it.
- Wang, X., Wei, J., Schuurmans, D., Le, Q., Chi, E., Narang, S., Chowdhery, A. and Zhou, D.
  (2023). Self-consistency improves chain of thought reasoning in language models. ICLR 2023;
  arXiv:2203.11171. Sections 1 to 3 read.
- Wolf, M., van Doorn, G. S., Leimar, O. and Weissing, F. J. (2007). Life-history trade-offs
  favour the evolution of animal personalities. Nature 447, 581–584. doi:10.1038/nature05835.
  Abstract read (PubMed).
- Zhang, J., Hu, S., Lu, C., Lange, R. and Clune, J. (2026). Darwin Gödel Machine: open-ended
  evolution of self-improving agents. ICLR 2026; arXiv:2505.22954. Section 3 and appendices A.3
  and C.2 read.

## Comments
