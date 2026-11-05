# Spec: Soul and Coupling strength (step 2)

Status: needs-triage. This is the build design of step 2 of the build order (Q47). Nothing is
built yet. Eight design choices that the earlier decisions left open, Q67 to Q74, were
decided on 2026-10-31 by their recommended answers (`.scratch/universum-erfinden/spec.md`,
`owner-questions.md`). A marker such as [Q69 a] shows where a decision applies. The owner
approves this design before any ticket starts.

Builds on step 1 ([ADR 0006](../../docs/adr/0006-cascades-breed-on-domains-and-circuits-are-checked-exactly.md),
`docs/STEP1_CIRCUITS.md`, `docs/PREREG_STEP1_CIRCUITS.md`), on the decisions Q42 to Q47 and
Q51 to Q66 (`.scratch/universum-erfinden/spec.md`), and on the glossary section "Soul and
Character (planned, step 2)" of `CONTEXT.md`. Ticket 04 of `.scratch/universum-erfinden/`
is this feature. The sources are the verified ones gathered for Q52 and for Q58, Q60 and
Q61 (section 19).

## 1. Goal

Every Worker gets a Soul: inherited phases, one per hypothesis family (Q42, Q52). The Soul
is never seen; it shows itself through the Character it produces on each Task (Q51, Q57).
In a Stage, the Workers' votes add as complex numbers and the squared magnitude decides
(Q46); the result over the answers is the Stage's Array pattern (Q54, Q61). Judging is an
Observation: it pulls the Souls of both sides together, as strongly as the observer's
Coupling strength κ says (Q44, Q60, Q62). Souls pass to children by Souletics (Q64 to Q66),
and Upbringing shapes them at birth (Q59). Every part has a switch, and the controls of
Q42, Q44 and Q63 are switch settings. The Soul is off by default, so every existing Run,
log and test stays byte-equal (section 14). Step 2 ends with a pre-registered comparison:
the Soul against a real-valued control and against κ = 0, on circuits with the simulated
designer (section 17, ticket 10).

Not in step 2: overseers (step 3), human feelings (step 4), quantum Workers and the TextGrad
operator (step 5) (Q47); learned Family strengths, a later variant (Q52, Q63); any claim
about souls in nature (rung 4 of the ladder; Q48 stays open, Q57).

Q45 names three quantum forms, each against a classical control. Step 2 builds two of them, the
scope that ticket 04 of `.scratch/universum-erfinden/` gives it (Q42 to Q46). The first,
quantum-inspired amplitudes (Q45 names Han and Kim 2002), is the Soul. The third, quantum
probability for judgments (Pothos and Busemeyer 2013), is built as the interfering Stage vote of
Q46: the judges' amplitudes add, and the squared magnitude decides. Both have the real readout as
their classical control (section 12). The second form, quantum Workers, follows in step 5 (ticket
07 there). The vote is a simple form of the third: it combines judgments and models no single
judgment. A full quantum-probability model of one judgment has no step in the build order of Q47
yet, and ADR 0009 says so (section 18).

## 2. Decisions and where they are built

| Decision | What this spec does with it | Section |
|---|---|---|
| Q42 | A Soul per Worker as a Trait; a real-valued control of equal size, budget and seeds | 4, 12 |
| Q43 | Code name `phase_state`; German reports say "Seele (Phasenzustand)" | 4, 20 |
| Q44 | Observation as an interaction of strength κ in [0, 1]; weak changes little, strong fixes the phases; both sides change; κ an inherited Trait of the observer; controls κ = 0 and real-valued states | 8, 9.5, 12 |
| Q45 | Amplitudes and interfering judgments now, each against the real control; quantum Workers later | 1 |
| Q46 | Votes add as complex numbers, the squared magnitude decides; in the simulator also over explicit hypotheses | 7, 11 |
| Q47 | Step 2 of the build order; one pull request and one ADR (0009) | 20 |
| Q51 | The Character is computed and measured in Runs | 6, 13 |
| Q52 | The Soul is the inherited phases, one per family, and its identity; strengths come from the Task; equal strengths are the control | 4, 5 |
| Q53 | New terms go to the glossary in a pull request of their own | 20 |
| Q54 | A Character per Worker; an Array pattern per Stage | 6, 7 |
| Q55 | The Character is recomputed, never inherited; three channels of inheritance | 6.2, 9.9 |
| Q56 | Observations change Soul and Character during a Run; the change is logged per generation | 8.4, 13 |
| Q57 | The Soul shows itself through its Character; the contrast prediction | 6.3, 13, 17 |
| Q58 | Every child brings its own unit of Soul, nobody loses any, the archive keeps all | 9.4, 9.8 |
| Q59 | Upbringing: imprinting and teaching | 10 |
| Q60 | κ is the Coupling strength ("Kopplungsstärke des Beobachters"); "feeling" stays for humans | 8.1 |
| Q61 | The joint pattern of a Stage is its Array pattern ("Gruppencharakteristik der Stufe") | 7 |
| Q62 | Observations are the judgments of a Stage vote and peer judgments | 8.2 |
| Q63 | Family strengths come from the Task as the Domain states them; equal strengths the control | 5 |
| Q64 | Souletics: copy, mix after crossover, a small random angle, now and then a new Soul; each part can be switched off | 9 |
| Q65 | A parent passes on its birth Soul; experience passes only through Upbringing | 4.3, 9.2, 10 |
| Q66 | The channel is called Souletics, "Seeletik" in German reports | 3, 20 |

## 3. Terms and notation

The glossary terms are those of `CONTEXT.md`: Soul, Soul population, Family strength,
Character, Array pattern, Observation, Coupling strength, Upbringing and Souletics. German
reports use "Seele (Phasenzustand)" (Q43), "Kopplungsstärke des Beobachters" (Q60),
"Gruppencharakteristik der Stufe" (Q61), "Seeletik" (Q66), and from the glossary pull
requests "Seelengrundgesamtheit" and "Familienstärke"; for Upbringing "Erziehung", with
"Prägung" for imprinting and "Lehre" for teaching (our proposal).

- **Families.** A Domain names its hypothesis families in a fixed order, f = 1 to n: n = 8
  for circuits, n = 7 for grids (section 5).
- **Phase.** An integer p in 0..=65535 that stands for the angle 2πp / 65536. One phase unit
  is 1/65536 of a turn, about 0.0055 degrees.
- **Soul.** P = (p_1, ..., p_n), one phase per family.
- **Readout.** u(p) = e^(2πip/65536) for complex phases; for the real control, u(p) = +1
  when p lies in 0..16384 or 49152..65536 and −1 otherwise, that is the nearer of 0 and
  half a turn (section 12.1).
- **Family strengths** of task t: s(t) = (s_1, ..., s_n), each at least 0, not all 0.
- **Coupling strength.** κ in thousandths, an integer in 0..=1000.
- **Arms** of the comparison: S (the Soul), V (the real readout, which reads signs; German
  Vorzeichen) and K (κ fixed at 0); the pilot adds E (equal strengths) and N (no Soul). The
  letter V avoids a clash with arm R of step 1, uniform parent choice (CONTEXT, "Arm").

## 4. The Soul and how it is stored

**4.1 What a Soul is.** In a Run on a Domain with n families, every Worker carries a Soul of
n phases in the Domain's family order (Q42, Q52). The phases are its identity; it carries no
strengths, which come from the Task (Q52, section 5). The Soul is hidden (CONTEXT, "Soul"):
no Worker prompt, no request to a language model and no Mutator request shows it. Only the
simulated models receive the phases (section 11).

**4.2 Why integers.** Integer phases keep the genome's JSON, its content id and every update
of a Soul exact. Floating point enters only the Character, the vote and the logs
(section 15).

**4.3 Birth Soul and current Soul.** The Soul a Worker has at birth, its birth Soul, is part
of its genome. Observations change the Soul during a Run (Q56). That changed Soul, the
current Soul, lives in the archive member, never in the genome. So a member's content id
never changes, and a parent can only pass on its birth Soul, never the Soul that
Observations changed (Q65). A third value is fixed at birth: the imprinted Soul, with which a
child answers its tasks after the first (section 10.1).

**4.4 Storage that keeps old records.** In `src/cascade/genome.rs`, `Traits` gets two fields:

- `phase_state: Option<PhaseState>`, the code name of Q43, with `#[serde(default,
  skip_serializing_if = "Option::is_none")]`. `PhaseState` is a list of `u16`, serialized as a
  plain JSON array.
- `coupling: u16`, the Coupling strength in thousandths, with `#[serde(default,
  skip_serializing_if = "is_zero")]` (section 8.1).

`Traits` keeps `deny_unknown_fields`. A Traits object written before step 2 has neither
field. It reads as `None` and 0 and writes back byte for byte the same. So `Cascade::id()`,
the first 16 hex digits of the SHA-256 of the canonical JSON, gives every recorded id again,
and old archives, snapshots and candidate files read as before. A Run with the Soul off
never sets either field. Code from before step 2 refuses a genome with a Soul as an unknown
field, which is what we want: such a genome belongs to a Run with Souls.

- `Cascade::gene_id()` is the id of the Cascade with both fields removed from every Worker.
  Without Souls it equals `id()`. The novelty rule (section 9.6) and the seeds of model calls
  (section 15) use it.
- `Cascade::check` gains one rule: `coupling` is at most 1000. The breeder adds two rules for
  its Run. With Souls, every Worker carries a `phase_state` of exactly n phases. Without
  Souls, no Worker carries `phase_state` or a non-zero `coupling`. A seed, a resumed archive
  or a Mutator answer that breaks them is refused.

**4.5 The Mutator never sees a Soul.** The requests to the Mutator show the parent Cascades
with both fields removed, so the format text of the brief keeps its four Traits. A Run with
Souls therefore sends the Mutator the same requests, with the same keys, as a Run without
Souls whose members have the same genes (ADR 0002). An answer that carries either field is
refused with the reason "not admitted: the Mutator does not write Souls"; only Souletics
gives a Soul (section 9). The `born` event logs the genome with its birth Soul, since the log
is the Run's record, not a prompt.

**4.6 Promotion.** `--promote` refuses a Run with Souls with exit code 1, as it refuses
circuit Runs. A Soul belongs to its universe, and promoting Cascades with Souls is not part of
step 2.

## 5. Family strengths (Q52, Q63)

**5.1 What the Domain states.** The `Domain` trait gets two methods: `families()`, the n
family names in their fixed order, and `strengths(task)`, n numbers of at least 0 of which one
at least is positive. They are evaluator data: they never reach a prompt, a request or a
Worker (tests, tickets 02 and 04). The default `families()` is empty, and a Run with Souls on
such a Domain is refused. The strengths of a task are the same for every Worker on it (Q63). A
dressing of a circuit target relabels qubits and adds X gates; its tasks keep the target's
strengths.

**5.2 Circuits** [Q67 a]. The families are those of the simulated designer, in the order of
`FAMILY_STEMS`: Linear, Phase, Hadamard, Compose, Search, Meet, Simplify and Previous. A test
checks that `families()` equals these names in this order. Each verified library target
states its fitting families: strength 1 for every family that a route reaching the reference
names, among the routes that the step 1 calibration tests pin on the plain target; strength 0
for every other family. Previous has 0 everywhere: it reuses earlier answers and fits no target
by itself. The table goes into `LIBRARY` in `src/cascade/circuit.rs`, beside the references
and their sources; the routes are those of `docs/STEP1_CIRCUITS.md`, section 5.

| Target | Fitting families (strength 1) | Pinned routes to the reference |
|---|---|---|
| cz | Linear, Hadamard, Search | Search on `sim:large` from 104 thinking tokens; Hadamard and Linear at 4096 |
| swap | Linear | Linear, 0 tokens |
| cs | Phase | Phase, 0 tokens |
| ch | Search, Meet | Search and Meet from 3124 tokens; Meet alone at 4096 |
| qft2 | Linear, Phase, Hadamard | Hadamard, Phase and Linear at 4096 |
| ccz | Linear, Phase | Phase and Linear from 196 tokens |
| toffoli, peres, rtof, ccix | Linear, Phase, Hadamard | Hadamard, Phase and Linear from 78 to 158 tokens |
| fredkin | Linear, Phase, Hadamard, Compose, Simplify | Compose, Hadamard, Phase, Linear and Simplify from 1566 tokens |

**5.3 Grids.** The families are those of the simulated grid model in `src/cascade/sim.rs`, in
its order: Mirror, Rotate, Transpose, Gravity, Colour, Compose and Previous; a test checks
the order. A puzzle's strengths come from its hidden rule: 1 for the family of its geometric
step (Mirror for both flips, Rotate for the three rotations, Transpose, Gravity; none for the
identity), 1 for Colour when the rule carries a colour map, 1 for Compose when it carries both
a step and a map, and 0 for all others, Previous included.

**5.4 Equal strengths** are the control (Q52, Q63): the setting `strengths: equal` gives every
family the strength 1 on every task. Then a Worker's Character is the same on every task.

**5.5 Learned strengths** are a later variant (Q52, Q63). Step 2 leaves the seam
`strengths(task)` and nothing more.

## 6. The Character (Q51, Q54, Q55, Q57)

**6.1 Definition.** For a Worker with Soul P on task t, let Z(P, t) = Σ_f s_f(t) · u(p_f). Its
Character on t is χ(P, t) = |Z(P, t)|², the squared magnitude of the sum of its phases, each
weighted by its Family strength (CONTEXT, "Character"). Over Souls with independent, uniformly
distributed phases its mean is Σ_f s_f(t)². So the logs also use the normalised Character
χ̂(P, t) = χ(P, t) / Σ_f s_f(t)², whose mean over such Souls is 1 for any strengths (by our
calculation; a Monte Carlo check with 200 000 Souls gave 0.998 to 1.005), so that tasks with
different strengths can be averaged. The contrast of 6.3 is the same for χ and χ̂, and the vote
does not need the division (7.2). The Character of a Worker is its pattern over the Tasks:
χ(P, t) for every task of the search panel. Each Worker has one (Q54).

**6.2 Recomputed, never inherited** (Q55). `character()` is a pure function of the Soul, the
strengths and the readout. No Character is stored in a genome, member, snapshot or event;
every use computes it from the current Soul. When an Observation changes the Soul, the
Character changes with it (Q56). The logs hold statistics of it (section 13), and an analysis
recomputes any Character from the logged Souls and the Domain's strengths (ticket 09).

**6.3 The prediction of Q57.** For Souls with independent, uniformly distributed phases, the
contrast of the Character on a task, its standard deviation over Souls divided by its mean, is

- K_C = √(1 − Σ s⁴ / (Σ s²)²) with complex phases, and
- K_R = √2 · K_C exactly with the real readout.

Both formulas are our calculation. For k equal strengths K_C = √(1 − 1/k), the formula that
the Q52 check recomputed for speckle from few steps (Goodman 2008; only the abstract was seen).
For many comparable families K_C tends to 1, the contrast of fully developed speckle
(Goodman 1976), and K_R to √2. The Monte Carlo check of the design (200 000 Souls per case)
agreed within 0.003. Values:

| Fitting families | K_C | K_R |
|---|---|---|
| 1 (a single phasor: no pattern) | 0 | 0 |
| 2 | 0.707 | 1.000 |
| 3 | 0.816 | 1.155 |
| 5 | 0.894 | 1.265 |
| equal strengths, 8 circuit families | 0.935 | 1.323 |
| equal strengths, 7 grid families | 0.926 | 1.309 |
| mean over the 11 circuit targets with the strengths of 5.2, each target once | 0.655 | 0.927 |

So with the strengths of 5.2 the contrast stays well below 1, but the ratio √2 holds on every
task with at least two fitting families. Where one family fits (swap, cs, and about 7 in 10 grid
puzzles, whose rule is one geometric step or one colour map alone), the Character shows no
pattern in either readout. The formulas hold for independent Souls only. Shared ancestry and
coupling add a common part to the Souls, and a consensus part lowers the contrast
(Escamilla-Taylor 1990, a dissertation). The Runs log how far the Soul population falls below
the formula (section 13); the pre-registration tests the formula itself on fresh Souls
(section 17).

## 7. The Array pattern and the complex vote (Q46, Q54, Q61)

**7.1 Today.** `stage_vote` in `src/cascade/arena.rs` takes a Stage's d distinct answers in the
order they first appear. Every Worker j judges every pair (x, y), x < y, blind, in an order
drawn from a seed. A verdict adds 1 to the wins of the answer it picks; a call without a reply
or without a verdict adds nothing. The answer with the most wins goes on; a tie goes to the
lowest index.

**7.2 The vote phasor** [Q68 a]. Judge j's phasor on task t is v_j = Z(P_j, t), from its
current Soul P_j (frozen for the generation, section 8.4). Its direction comes from the Soul, and
its squared length is the judge's Character: |v_j|² = χ(P_j, t). All judges of a vote share the
task's strengths, so a common scale, such as the division of the normalised Character, would
change no decision.

**7.3 The Array pattern.** Let w_j(x) be the number of verdicts of Worker j for answer x in this
vote. Then A(x) = Σ_j w_j(x) · v_j, and Π(x) = |A(x)|². The values Π(x) over the d answers are
the Stage's Array pattern on this task: for each answer, the squared magnitude of the sum of
its Workers' contributions (CONTEXT, "Array pattern"). In the antenna picture, separate
elements with their own amplitude and phase add per direction, and shifting the phases steers
the pattern (Zichner and Chandra 2009; Kark, sections 11.1, 11.2.1 and 11.2.7, only the table of
contents seen). The blind pairwise judgments stay exactly as they are: the same prompts,
seeds, order draws and calls. Only the tally changes, from counts to amplitudes.

**7.4 The decision** (Q46). The answer with the largest Π goes on. A tie in Π, also when every Π
is 0, goes to the answer with more verdicts, as today's count would decide, and a tie there to the
lowest index, as today. The count comes before the index because the real readout cancels
exactly: two equally loud judges of opposite sign who both choose x give Π(x) = 0, the same as the
answer nobody chose. With the index first, the order of the answers would decide such unanimous
votes and lose those whose chosen answer came second. For independent Souls the order would decide
1/2 of the unanimous two-judge votes with one fitting family, 3/8 with two, 5/16 with three and
63/256 with five (by our calculation; a Monte Carlo check with 10⁶ votes per case agreed within
0.001), against almost none with complex phases. Claim V would then measure mainly this artefact.
Values are f64, summed in a fixed order (j ascending) and compared with `total_cmp`.

**7.5 It contains today's vote.** If every v_j is the same non-zero number, Π(x) = |v|² ·
wins(x)², so the winner is today's. A test checks this on 10 000 random win tables. The same holds
in both readouts when every judge gives the same verdicts, w_j(x) = w(x) for every j: then
Π(x) = w(x)² · |Σ_j v_j|², and when the phasors cancel, every Π is 0 and the count decides. So a
vote in which every judge gives the same verdicts is never lost (test). With the Soul off, or
with `vote: false`, `stage_vote` counts as today, byte for byte.

**7.6 What interference does** (examples by our calculation). Judges in phase add up: k judges
of equal Character and equal phase weigh k² times one judge, and with random phases k times on
average (Born and Wolf 1999, chapter 7; Feynman, Leighton and Sands, volume III, chapter 1,
seen only through secondary sources). Judges out of phase cancel: take three judges with equal
Characters at 0, 120 and 240 degrees who all prefer x to both other answers, so A(x) = 0. If they
split two to one on the other pair, both other answers keep Π > 0, and x loses at any index,
although it has the most verdicts. If they agree on that pair too, 7.5 applies and x wins, as in
the count. A split vote goes to the louder judge, the one whose Soul fits the task better. In arm
V, two judges of equal loudness and opposite sign cancel exactly; when both choose the same answer,
the tie rule of 7.4 keeps it.

**7.7 What a vote records.** For each search task, the member records the authors of its final
answer: the Workers of the Stage that chose it whose own answer it was. The authors of the
answers in a vote are known inside `run_cascade_in`. Each member also keeps a tally of its
birth votes: complex votes, flipped votes (the winner differs from the count's winner), tied
votes (two or more answers shared the largest Π, so the tie rule decided), and the sum of the
coherence |Σ_j v_j| / Σ_j |v_j| (0 when the denominator is 0). The tally also counts the replies
of its Workers in which the phases decided a simulated model's choice (section 11.1). Both go
into the `born` event (section 13.2).

**7.8 Peer judgments are not votes.** A peer judgment has one judge, the first Worker of the
judge Cascade's last Stage, as today. Its verdict, its weight ln(u / (1 − u)) in the
Bradley-Terry ratings, judging skill and fitness (`refresh`) stay unchanged. The Soul touches
peer judging only as an Observation (section 8): a peer request never carries phases (section
11.1). Q46 names voting; peer ratings are a separate mechanism (CONTEXT, "Peer rating").

## 8. Observations and the Coupling strength (Q44, Q56, Q60, Q62)

**8.1 The Coupling strength.** κ is stored in thousandths in `Traits::coupling` (Q44, Q60).
It is inherited (section 9.5) and stays the same during a Worker's life. German reports say
"Kopplungsstärke des Beobachters"; the word "feeling" stays reserved for human feelings
(Q34 to Q36, Q60). The name follows the Kuramoto model, where the coupling strength pulls the
phases of many oscillators together (Acebrón et al. 2005).

**8.2 Which judgments are Observations** (Q62).

- **Stage vote.** A judgment of the pair (x, y) by Worker j that got a reply, with or without a
  verdict, is an Observation by j of every author of x and of y other than j.
- **Peer judgment.** A judgment that got a reply is an Observation by the judge Worker of every
  author of the first Cascade's answer and then of the second's, on that task (the authors of
  section 7.7).
- **No reply.** A call that the backend could not answer observes nothing.
- **Select phase.** Its judgments are never Observations. So the select panel never changes a
  Soul, also when a resume breeds on after a finished select phase.
- **Imprinting** is Upbringing, not an Observation; Q62 names only judgments (section 10.1)
  [Q73 a]. Under Q73 (c), a parent's judgment of the child's first answer would be one.

**8.3 The update** [Q69 a]. When O observes T with Coupling strength κ_O, for each family f let
Δ_f be the signed phase difference from T to O, wrapped into −32768..=32767. Then T moves
toward O by round(κ_O · Δ_f / 1000), and O moves toward T by round(κ_O · Δ_f / 2000). Both
moves start from the phases before this Observation, all modulo 65536, with rounding half away
from zero in integers. A small κ changes little, as a weak measurement does, and at κ = 0
nothing changes. At κ = 1000 the observed takes the observer's phases: strong measurement fixes
the phases (Q44; CONTEXT, "Coupling strength"). The observer is pulled in turn, by half as much.
This is the rule of the atlas sketch, where two phases 150 degrees apart are 75 degrees apart
after an Observation at κ = 1; our integer check gives the same. All families move alike. In arm
V the pull acts on the same hidden angles, and the readout follows when an angle crosses a
quarter turn.

Physics separates weak from strong measurement by the coupling of apparatus and system
(Aharonov, Albert and Vaidman 1988; Tamir and Cohen 2013). There, a stronger measurement
destroys more interference (Englert 1996, only the summary seen; Dressel et al. 2014). Here an
Observation aligns phases. That is a choice of the model, as the Q60 research noted.

**8.4 When an Observation acts** (Q56). During a generation every Soul is frozen. Each
evaluation, vote and peer judgment reads the current Souls as committed after the previous
generation; a newborn uses its birth Soul on its first task and its imprinted Soul afterwards
(section 10.1). Observations are recorded with the evaluation and applied at the commit, after
each newborn's current Souls are set to its imprinted Souls, one after another, in log order: the
newborns in birth order; within a newborn, its tasks in panel order, its Stages in order, the
pairs (x, y) in order, the judges in order and the observed authors in order; then the peer
judgments in the order of the plan. A stopped or failed
generation commits nothing, so its Observations vanish with it, and a resume repeats them
exactly. Apart from a newborn's imprinting after its first task (section 10.1), which changes only
that newborn's own Souls, no evaluation changes a Soul, so the other tasks still run in parallel.

**8.5 The record.** After the `generation` event, one `souls` event lists every Worker whose
current Soul changed at this commit: member id, Worker index and the new phases. The `state`
line, which stays the last event of the step, counts the Observations (section 13).

**8.6 How κ reaches a score.** A member's search answers are fixed at its birth, and the
designer's verdicts do not read the Soul (section 11). So an Observation changes no answer
directly. It changes the current Souls, which a member uses later: in its judging, in its
select-phase votes and in imprinting its children (section 10.1). Since parents imprint a child
before most of its search tasks, κ reaches the search scores through Upbringing.

## 9. Souletics at birth (Q55, Q58, Q64, Q65, Q66)

**9.1 Sources.** The breeder hands every genetic operator its parents with `phase_state` and
`coupling` removed, as it does for the Mutator (section 4.5). So the operators, and their test
that a child differs from both parents, see genes only and make exactly today's draws, and the
generation's random stream is unchanged. Without this, a crossover child with the genes of one
parent and the Souls of the other would pass that test where today it is refused and the coins
are flipped again; in generation 1 of a circuit Run, whose seeds differ only in their strategy
section, this is the common case. Every operator returns, beside the child, the source of each
child Worker: the parent Worker it was copied from, or one Worker from each parent where the
operator pairs Workers. Souletics then writes each child Worker's Soul and κ from these sources.

- **Mutation operators** keep each Worker's source at its position. The clone of WorkerAdd has
  the cloned Worker as its source, the new Worker of StageAdd the Worker it was copied from.
  WorkerDrop and StageDrop shift positions, and the map follows them.
- **StageCrossover**: the prefix Workers come from parent A, the suffix Workers from B, each from
  its own position.
- **SectionCrossover**: Worker (s, w) has two sources, A's Worker (s, w) and B's partner Worker,
  the one `partner(B, s, w)` picks.
- **Mutator children**: for LlmMutate, Worker (s, w) has the source `partner(parent, s, w)`; for
  LlmCrossover, `partner(A, s, w)` and `partner(B, s, w)`. The Mutator never writes a Soul
  (section 4.5).
- **Seeds** have no source.

**9.2 Copy and mix** (Q64, Q65). A Worker with one source copies the source's birth Soul, from
the parent's genome, never its current Soul. A Worker with two sources mixes both: each phase
comes from one of them, by one coin per family, as `cross_worker` crosses Traits. Mixing
invents no phase; each one is a parent's phase.

**9.3 Drift** (Q64). Then each phase moves by a uniform random integer in −d..=d phase units.
The setting `phase_drift` holds d; 512 units, about 2.8 degrees, is a placeholder that the
pilot sets.

**9.4 An entirely new Soul** (Q58, Q64). Before the copy, a coin decides for each child Worker
whether it gets an entirely new Soul instead. Its probability is θ / (θ + N), where N counts the
Souls already in the urn: the Workers of the archive's members plus the child Workers drawn
before this one in this generation. A ball of the urn is a Soul, as the Soul population counts
them (section 13.1), so the coin and N count the same unit. This is the new colour of Hoppe's urn
(Hoppe 1984, only the summary seen; the procedure confirmed by Leckey and Neininger 2012, a
preprint): every child Worker adds its own unit, and nothing is removed. A new Soul has n uniform
random phases, and the drift of 9.3 follows as for a copy. The setting `new_soul_theta` holds θ,
with 1 as a placeholder; θ = 0 never gives a new Soul. After the 4 seed Workers, a Run that ends
with W Workers gets about θ ln((θ + W) / (θ + 4)) new Souls (by our calculation). With θ = 1 and
8004 members that is 7.4 at 1 Worker per member, 8.7 at 3.7, as a 60-generation circuit Run of
step 1 had (pilot seed 101 of arm A, by our count from its log), and 9.9 at 12, the most the
limits allow. A new Soul starts a new origin, Hoppe's colour; a copied or mixed Soul keeps the
origin of its first source.

**9.5 The Coupling strength at birth** [Q71 a]. κ follows the same sources: copied from the one
source, or taken from one of two sources by a coin, then moved by a uniform random integer in
−δ..=δ thousandths and clamped to 0..=1000. The setting `kappa_drift` holds δ, with 20 as a
placeholder. Seed Workers start at `kappa_seed`, with 200 as a placeholder. A Worker with a new
Soul keeps the κ of its source, since κ is a Trait of the observer, not part of the Soul. With
`kappa_fixed`, κ never moves; arm K sets `kappa_seed` to 0 and `kappa_fixed` (Q44). So κ is no
gene: it is not in `gene_id`, the Mutator never sees it, and `cross_worker` never crosses it. It
passes by Souletics, with the Soul, although it is not part of the Soul (9.9); Q71 (b) would make
it a gene instead.

**9.6 Novelty by genes.** A child must be new by `gene_id`, against the archive and against the
other children of its generation, as today by `id`. A genetic copy of a member with another
Soul is refused, so Souls are selected together with genes. Without Souls `gene_id` is `id`, and
drafting stays byte-equal. Members keep their full ids, birth Souls included.

**9.7 Own random streams.** Souletics draws from its own streams: for seed i from
`derive_seed(seed, "soul/seed/{i}")`, for child i of generation g from `derive_seed(seed,
"soul/{g}/{i}")`. Per child Worker, in Stage order, the stream draws: the new-Soul coin, n
uniform phases, n mix coins, n drift steps, the κ coin and the κ step. It draws all of them
whatever the switches; a part switched off ignores its draws. So the arms of the comparison share
every random number of the genes and of the Souls as long as their archives agree.

**9.8 The Soul population** (Q58). Every member brings the Souls of its Workers. Observations
change Souls but remove none, and the archive keeps every member. So the number of Souls in a
Run never falls (test).

**9.9 Three channels that stay separable** (Q55, Q65). Genes pass through the genetic
operators. Souletics passes birth Souls, which experience never touches, and κ [Q71 a].
Upbringing passes experience: imprinting from the parents' current Souls and teaching from their
solved examples (section 10). Two tests show the separation: a parent's changed current Soul
leaves its children's birth Souls unchanged, and with imprinting off a child's first current Soul
equals its birth Soul.

## 10. Upbringing at birth (Q59) [Q73 a, d]

**10.1 Imprinting** [Q73 a]. A child answers the first task of the search panel with its
birth Souls. Then each of its Workers is imprinted. For each source of the Worker, one or two in
parent order, the Worker's phases move toward the source Worker's current phases by round(κ_src ·
Δ_f / 1000) per family, the observed side of the rule of 8.3, so the step follows the answer to
Q69. The parents do not change. The child answers its other search tasks with the imprinted
Souls; these start its current Souls at the commit. Imprinting makes no model call. Seeds have no
parents and are not imprinted. With `imprint: false` the imprinted Soul equals the birth Soul.
The parents read no answer, so Q59's "the parents observe the child's first answers" holds only
in its timing: the shift comes after the first answer. Q73 (c) would make it a real
Observation: a Worker of each parent judges the child's first answer against the parent's own
answer on that task, at one call per parent and child, and both sides move: the authors of the
child's answer at once, the parent's judging Worker at the commit. Imprinting comes before most
search tasks because, if it came only after them, κ could change no search score in a circuit Run
(section 8.6).

**10.2 Teaching** [Q73 d]. With `teach: true`, fixed code writes one prompt section named
`upbringing` at birth, from the parents' best solved examples. For each parent it takes the first
search task with the highest graded value among those the parent solved, and the Domain renders
the task with the parent's answer (a new `Domain::example`), as long as the section stays within
`MAX_SECTION_CHARS`. The section ends every system prompt the child sends, to solve and to
judge. It lives in the member and in the `born` event, never in the genome, so it is never
inherited as a gene; the child's own children get a section written from the child's examples
(Q59). Lessons need a language model to write them and wait for a later step [Q73 d]. The
template holds no family stem of the simulated models and neither of their self-check texts
(test). So the simulated models read nothing from it, and in simulated Runs teaching changes no
answer, only the prompt length. It is off in the comparison.

## 11. The Soul in the simulated models (Q46) [Q72 a]

**11.1 How the phases get there.** `GenRequest` in `src/backend.rs` gets `phases:
Option<Vec<u16>>`, left out of the JSON when `None`, so every request key stays as it is.
`ModelBackend` gets `reads_phases()`, false by default and true for `SimModel` and
`SimDesigner`. The breeder fills `phases` only in the solve and Stage-vote requests of a
Cascade's own evaluation, on the search and on the select panel, and only for backends that read
them, with the readout applied: in arm V every phase becomes 0 or 32768. A peer judgment never
carries phases, so the Soul touches peer judging only as an Observation (section 7.8), and
generation 0, whose seeds hold no vote and whose only judgments are peer judgments, scores as
without Souls. Real model backends never receive phases; with real language models the Soul acts
only in the vote (Q46). Strengths never reach a request (section 5.1): the simulated models
interfere with unit phasors over their own explicit hypotheses. `GenResponse` gets
`soul_decided: bool`, with the serde default false, so responses written without it still read,
and left out of the JSON when false: a simulated model sets it when its reply differs from its
reply to the same request without phases, which it knows at no cost (11.2, 11.3). The logs count these replies (section 13.1). The two fields and `reads_phases()` are
contract-v3 of the normative `docs/STAGE2_BACKEND_CONTRACT.md` (ticket 06).

**11.2 The circuit designer.** Today the pool keeps the first verified candidate of the best
rank (T, then CNOT, then gates) and drops a later candidate of the same rank unverified, at no
cost. With phases, and only when the request carries the self-check text, the pool also keeps
such a candidate when it passes the same check on every column; this one check is not paid for, so
the budget and the reported thinking tokens stay as they are (an explicit exception to "all work
costs steps"; it costs wall time only). Without self-check the reply stays the first found: the
check then covers only the first and the last column, so a later candidate of the same rank could
be inexact where the first was exact, and the Soul would change grades. Each kept candidate has a
support: the set of families whose methods produced exactly that circuit, so a circuit found twice
merges both. The reply is the kept candidate of the best rank with the largest
|Σ_{f in support} e^(iφ_f)|², then the first found. A support of one family has intensity 1
whatever the Soul. So the Soul decides only when at least one equally cheap candidate was found by
two or more families, for example a circuit that Simplify or Previous offers again; between
one-family candidates the first found stays. A cheaper candidate still replaces them all. The
reply's costs, exactness and thinking tokens stay as they are, so the calibration of step 1 holds
(ticket 06). The circuit itself can differ, so the next Stage, and whether the Workers of a Stage
agree, can change. The designer's judge stays as it is: it verifies and compares costs, and it
knows no families.

**11.3 The grid model.** Its judge tests hypotheses until both answers are explained; when both or
neither are explained, it flips a coin. With phases, each answer gets the set of families whose
hypotheses, among those tested so far, explain it, each family once. When both answers are
explained, the answer whose families give the larger |Σ_f e^(iφ_f)|² wins; equal values still go
to the coin. The coin is drawn wherever it is drawn today, also when the phases decide, so the
random stream stays as it is. An answer
explained by one family has intensity 1 whatever the Soul, so the Soul decides only when one
answer is explained by two or more families before the loop stops. Counting each family once
keeps two hypotheses of one family from beating one hypothesis of another by 4 to 1 in every arm,
a count that would not come from the Soul. Nothing more is tested, so its thinking stays the same.
The solver stays as it is: the first fitting hypothesis in the order of the prompt, which is a
gene.

**11.4 What this means.** In the circuit domain the designer lets the Soul choose only among
equally cheap, exact circuits of Workers with self-check, and only where two or more families
found one of them; the grid judge only where two or more families explain an answer. There the
Soul acts mainly through the vote, the Observations and Upbringing. The pilot counts how often the
phases decide in the simulator (`sim_decided`, section 13.1).

## 12. Controls and switches (Q42, Q44, Q63, Q64)

**12.1 Readouts** [Q70 a]. Both readouts work on the same hidden angles: the same Souls, draws,
inheritance and coupling. Only u(p) differs (section 3). The real readout keeps the phasor's
length 1, so the Soul carries no strength in either arm (Q52), and it holds as many numbers as
the complex Soul (Q42). The contrast of a real field is √2 times that of a complex one
(section 6.3). Quantum theory with real numbers alone can be told apart from complex quantum
theory by experiment (Renou et al. 2021); arm V asks the same of the model.

**12.2 Settings.** `BreedConfig` gets `soul: Option<SoulConfig>`, left out of the JSON when
`None`, so configs and `fixed_hash` stay as they are. `SoulConfig` refuses unknown fields and
requires every field. It defines the experiment, so a resume refuses a change.

| Field | Meaning | Arm S | Arm V | Arm K |
|---|---|---|---|---|
| `readout` | `complex` or `real` (Q42, Q70) | complex | real | complex |
| `strengths` | `domain` or `equal` (Q63) | domain | domain | domain |
| `vote` | complex Stage vote (Q46); off: the count of today | on | on | on |
| `simulator` | phases reach the simulated models (Q46, Q72) | on | on | on |
| `observe_vote`, `observe_peer` | judgments of the Stage vote, of the peer round are Observations (Q62) | on | on | on |
| `kappa_seed` | κ of the seed Workers, thousandths (Q71) | 200* | 200* | 0 |
| `kappa_fixed` | κ never moves (Q44) | off | off | on |
| `kappa_drift` | largest κ step at a birth, thousandths | 20* | 20* | 20 |
| `copy` | children copy Souls; off: every child Worker gets a new Soul (Q64) | on | on | on |
| `mix` | two sources mix; off: the first source is copied (Q64) | on | on | on |
| `phase_drift` | largest phase step at a birth, phase units (Q64) | 512* | 512* | 512* |
| `new_soul_theta` | Hoppe's θ; 0: never a new Soul (Q64) | 1* | 1* | 1* |
| `imprint` | imprinting (Q59) | on | on | on |
| `teach` | teaching (Q59) | off | off | off |

A star marks a placeholder that the pilot sets (ticket 11). Arm K's `kappa_drift` has no effect,
since κ is fixed.

**12.3 Arms.** S is the Soul. V differs from S only in the readout (Q42); its letter avoids arm
R of step 1, which chose parents uniformly. K differs from S only in κ, fixed at 0 (Q44). With
equal strengths (arm E, Q63) and without a Soul (arm N, today's breeder, which is arm A of step 1)
the pilot runs two more arms, for description only [Q74 a]. With Q74 (d), arm E would be counted
too, against S.

**12.4 Every part can be switched off.** The Soul as a whole (no `soul` field), the complex vote,
the simulator, each kind of Observation, the coupling (κ fixed at 0), each part of Souletics,
imprinting and teaching. A test runs each switch and checks its degenerate case, for example:
with `copy` off every child Worker has a new Soul; with `phase_drift` 0 a copy is exact; with
`vote` off the votes are those of today on the same answers.

**12.5 Program.** `brut-breed` gets `--soul s|v|k`, which sets the `soul` field of arm S, V or
K; finer settings come with `--config`. Giving both is an error. `--soul` defines the
experiment, so a resume refuses it, as it refuses `--domain` and `--selection`.

## 13. What is logged (Q33, Q51, Q56, Q57, Q58)

**13.1 The `state` line.** It gets a seventh group, `soul`, which is left out of the JSON in Runs
without Souls. It is a pure function of committed state, as the other six groups are (ADR 0006),
so a stopped and resumed Run logs the same lines as a straight one. Means over tasks run over the
tasks of the search panel, each task once. A statistic of the Character is computed once per
distinct strength vector and weighted by the number of tasks that have that vector. With three
dressings of each of the 11 targets, every target then weighs alike, as in the table of 6.3;
weighting the 7 distinct vectors alike would give 0.563 and 0.796 there instead of 0.655 and 0.927.

| Field | Meaning |
|---|---|
| `souls` | Workers in the archive: the Soul population (Q58); it never falls |
| `souls_born`, `souls_new` | Workers born in this generation, and those of them with an entirely new Soul (Q64) |
| `origins`, `origin_entropy_bits` | distinct origins of Souls in the archive, Hoppe's colours, and the entropy of their frequencies |
| `character_norm_mean` | the mean normalised Character (6.1) over the archive's Workers and the search tasks, from current Souls (Q51) |
| `contrast` | the mean over the search tasks of the Character's contrast over the archive's Workers (Q57) |
| `contrast_independent` | the same mean of the formula of 6.3 for this readout: the value for independent Souls |
| `order`, `order_mean` | per family, the length of the mean readout phasor over all Workers (1: all equal; near 0: spread out), and their mean |
| `moved_mean` | the mean circular distance between current and birth Soul over Workers and families, in degrees (what experience did) |
| `kappa` | quantiles of κ over the archive's Workers, as fractions |
| `observations_vote`, `observations_peer` | Observations applied at this commit, by kind (Q62) |
| `shift_mean` | the mean phase change of the Workers that moved at this commit, in degrees (Q56) |
| `votes`, `votes_flipped`, `votes_tied` | complex Stage votes of this generation's newborns on the search panel; those whose winner differs from the count's; those in which two or more answers shared the largest Π, so the tie rule decided |
| `coherence_mean` | the mean over these votes of their coherence (section 7.7): 1 when all phasors point alike, near 0 when they cancel |
| `sim_decided` | replies to this generation's newborns on the search panel in which the phases decided a simulated model's choice (sections 11.1 to 11.3) |

The fields `contrast` and `contrast_independent` together carry the prediction of Q57: in arm S
the contrast starts near the complex formula, in arm V near √2 times it, and both fall below it
when inheritance and coupling make the Souls alike. At generation 0 the archive holds only the 4
seed Workers, so the contrast there is a rough estimate; the pre-registration tests the formula on
fresh Souls instead (section 17).

**13.2 The `born` event** gets five fields, each left out without Souls: `imprinted` (the
imprinted Soul of each Worker), `soul_sources` (each Worker's source Workers, whether its Soul is
new, and its origin), `teaching` (the section, when on), `authors` (per search task, section 7.7)
and `votes` (the tally of 7.7).

**13.3 The `souls` event** is written after each `generation` event of a Run with Souls
(section 8.5).

**13.4 Rules.** Sums run in a fixed order; no value is NaN or infinite; an undefined value is
absent; counts of an empty set are 0, as in `observe.rs`.

## 14. Off by default: nothing changes

With no `soul` field, which is the default, no new field is written anywhere, no new random draw
is made and no request changes. These tests must pass with their fixtures unchanged:

- `tests/universe.rs`: the grid Run, the grid Run with LLM rounds, the Mutator's request keys and
  the settings hashes (recorded before step 1), and the state-line tests.
- `tests/mlp_golden.rs`: all nine tests, with the ten fixture files unchanged.
- `tests/circuits.rs`, the calibration tests named in section 5 of `PREREG_STEP1_CIRCUITS.md`,
  and the tests of `examples/circuit_positive_control.rs`.
- A release test: the 60-generation Run of pilot seed 101 in arm A of step 1 ends its event log
  in the head hash that `docs/pilot-step1/pilot.json` records (`e93ed521…`).
- Once, by hand, at the end of the build: the counted Run A-1 of step 1 gives the five hashes of
  `docs/step1-counted/hashes.txt`.

## 15. Reproducibility

- **Integers.** Souls, κ, drift, mixing, Observations and imprinting are integer arithmetic. Only
  the Character, the vote phasors, Π and the logs use floating point.
- **Seeds of calls.** In a Run with Souls, every model call takes its seed from the member's
  `gene_id` instead of its `id` (the tags `search/{gene_id}` and `select/{gene_id}`). Without Souls
  the two are equal, so nothing changes. With them, arms whose members hold the same genes make the
  same calls with the same seeds, and arm K, whose members differ from arm S only in κ, starts
  identically too. Together with 9.1 and 9.7, the arms share every random number until the Soul
  changes an outcome; from then on they diverge. The pilot measures when, by the divergence
  generation of two arms on a seed: the first generation whose newborns differ, member by member
  in birth order, in gene id, operator, parents (by gene id), answers, solved count or graded value
  (ticket 11). The `born` events themselves differ earlier and say nothing about outcomes: they
  carry each Worker's κ, which differs between S and K from generation 0, and the vote tallies,
  whose coherence differs between S and V from the first vote.
- **Threads.** Souls are frozen within a generation and Observations act in log order, so the
  bytes of a Run do not depend on the thread count. Tests compare 1 and 4 threads.
- **Resume.** The checkpoint holds every member's current Soul, so a resumed Run continues exactly;
  Observations of an uncommitted generation are not in it.
- **Platform.** The phasors use `cos` and `sin` of Rust's standard library. As with `exp` and `ln`
  in step 1, their last bit may differ between platforms, and then a near tie in a vote may fall the
  other way. Runs are compared on one toolchain and platform, rustc 1.97.0 on x86_64 Linux as in
  step 1.

## 16. `--verify` and the record

- **Born fields.** `check_archive` compares the five new `born` fields of 13.2 with the
  checkpoint's archive, as it compares the others. The genome in the `born` event already holds the
  birth Soul and κ.
- **Current Souls.** Each member's current Soul in the checkpoint must equal its last entry in the
  `souls` events, or its imprinted Soul when there is none. A difference ends `--verify` with exit
  code 1 and a message "integrity: …".
- **Which Runs.** Today `check_archive` runs only for a finished circuit Run, through
  `check_rediscovery`; a grid Run's `check_finish` checks nothing. Ticket 05 puts the check of
  current Souls into `check_archive`, and ticket 09 runs both checks above for a finished grid Run
  with Souls too. A grid Run without Souls is checked as today.
- **Limits of the check.** `--verify` makes no model call, so it does not recompute the
  Observations. A re-run of a simulated Run repeats them byte for byte, and the determinism tests
  check that.

## 17. The comparison of step 2

Ticket 10 holds the outline of the pre-registration, modelled on step 1. It is frozen only after a
pilot (ticket 11) and the owner's approval [Q74 a].

- **Question.** Does the Soul with complex phases and an inherited Coupling strength make the
  breeder reach the pass criterion of step 1 sooner than the same Soul read as real numbers (arm V),
  and sooner than the same Soul with κ fixed at 0 (arm K)? On circuits with the simulated designer,
  as a positive control.
- **Endpoint.** The pass time T of step 1 (`PREREG_STEP1_CIRCUITS.md`, section 7) per Run; per seed,
  "S first", "control first" or a tie, as step 1 compares its arms.
- **Gates.** The seed Cascades solve exactly swap and cs in every Run; the formulas of 6.3 hold on
  2^20 = 1 048 576 fresh Souls per Domain; the Runs without Souls still reproduce step 1. The
  second gate checks the code, since for independent Souls the formulas are exact. At this size its
  tolerance of 0.01 is about seven standard errors or more; at 65 536 Souls it would be about two
  for the real readout, and a correct implementation would fail at 4 to 5 seeds in 10 (three Monte
  Carlo checks in the design review, on 300, 300 and 60 seeds, saw it fail at 39 %, 44 % and 53 % of
  the seeds; none failed at 2^20 Souls, on 40, 40 and 6 seeds).
- **Rule.** Two evidence values, one per control, updated in the order of a fixed seed list; each
  claim with an error bound of 1/200, both together 1/100, by Ville's inequality as in step 1
  [Q74 a]. Q74 (d) would add a third value, against arm E, at 1/300 per claim.
- **Calls and run time** (planning figures). The Soul adds no call beyond the bound of step 1,
  12 723 412 calls for 1000 generations: imprinting makes no call [Q73 a], and no Stage votes more
  often than that bound counts. Under Q73 (c), each child would add one judge call per parent, at
  most 16 per generation, and the bound would become 12 739 412, still below 13 000 000, the cap of
  step 1. The calls of the arms can differ, since the Soul can change whether the Workers of a Stage
  agree. If a Run of arm S, V or K takes as long as one of arm A of step 1 (mean 7.36 min, 4.87 to
  11.18 min; `docs/STEP1_CIRCUITS.md`), a triple takes about 22 min and N_max = 120 triples about
  44 hours. The pilot's 60 Runs (5 arms, 12 seeds) take about 7.4 hours, its
  optional second batch of 40 Runs about 4.9 hours, and the thread check of gate G4, six Runs at 1
  and 4 threads, about 1.1 hours at the times step 1 measured for A-1 (14.42 min at 1 thread, 8.00
  min at 4). The pilot measures calls and wall time per arm.
- **What it may claim.** Rung 1 of the ladder only: whether complex phases and coupling help this
  model of Cascades. Nothing about real language models, inventions or souls in nature (Q48, Q57).

## 18. Limits

- **A model.** Soul, Character and Coupling strength are terms of a model. Nothing here is
  physically quantum, and the comparison says nothing about souls in nature (Q48).
- **A positive control.** The simulated designer is not a language model. Effects found with it
  say nothing about real models.
- **Unanimous judges.** The designer judges by verifying, so the judges of a Stage mostly agree.
  A vote in which every judge gives the same verdicts is never lost, in either readout (7.4,
  7.5). So the complex vote changes a winner only where the judges split, or where cancelling
  phasors among three answers let another answer's Π pass the favourite's. The tie rule of 7.4
  matters here: with the index as the first tie-break, exact cancellation would hand a quarter to
  a half of arm V's unanimous two-judge votes to the order of the answers, which would lose those
  whose chosen answer came second, and claim V would measure that artefact. The effect on the pass
  time may be small, and the pilot must show whether the comparison has power (ticket 11 says what
  happens if it has too little).
- **Weak simulator effect.** By [Q72 a] the designer lets the Soul choose only among equally cheap,
  exact circuits of Workers with self-check, and only where two or more families found one of
  them; the grid judge only where two or more families explain an answer. The pilot counts these
  choices (`sim_decided`).
- **Form 3 of Q45 in a simple form.** The interfering Stage vote is quantum probability for
  judgments in a simple form: it combines the judges' verdicts and models no single judgment. A
  full model of one judgment has no step in the build order of Q47 yet.
- **Teaching is inert** with the simulated models; its effect needs real models.
- **Hand-written strengths.** The table of 5.2 comes from the routes that step 1 pinned. On swap
  and cs one family fits, so every Character there is 1. A judge's loudness on a task depends on
  this table; the equal-strength arm of the pilot shows how much.
- **The contrast prediction** holds for independent Souls; in a Run, ancestry and coupling make
  Souls alike, and 4 seed Workers give a rough estimate.
- **Batches.** An Observation acts only from the next generation on, and many Observations within
  one birth can align a Cascade's Workers almost completely when κ is large.
- **Representatives.** A Cascade judges through its first Worker of the last Stage, as today.
- **Selection with genes.** Novelty by `gene_id` means that a Soul is selected only together with
  its genes.
- **New Souls get rare.** By Hoppe's rate, most new Souls enter early in a Run.
- **No claim of novelty.** The Q52 check could not verify the claim that no reviewed
  quantum-inspired evolutionary algorithm uses cancelling interference against a real control, so
  we make no such claim.

## 19. Sources

Only verified sources from the research for Q52 and for Q58, Q60 and Q61; the notes say what the
checkers saw.

- Acebrón, Bonilla, Pérez Vicente, Ritort, Spigler (2005). The Kuramoto model: A simple paradigm for
  synchronization phenomena. Reviews of Modern Physics 77, 137.
  https://doi.org/10.1103/RevModPhys.77.137
- Aharonov, Albert, Vaidman (1988). How the result of a measurement of a component of the spin of a
  spin-1/2 particle can turn out to be 100. Physical Review Letters 60, 1351.
  https://doi.org/10.1103/PhysRevLett.60.1351
- Born, Wolf (1999). Principles of Optics, 7th edition, chapter 7. Cambridge University Press.
  https://doi.org/10.1017/CBO9781139644181.016 (monograph)
- Dressel, Malik, Miatto, Jordan, Boyd (2014). Colloquium: Understanding quantum weak values: Basics
  and applications. Reviews of Modern Physics 86, 307. https://doi.org/10.1103/RevModPhys.86.307
- Englert (1996). Fringe visibility and which-way information: An inequality. Physical Review
  Letters 77, 2154. https://doi.org/10.1103/PhysRevLett.77.2154 (only the summary seen)
- Escamilla-Taylor (1990). Some aspects of speckle patterns produced by a small number of
  scatterers. Dissertation, Imperial College London. http://hdl.handle.net/10044/1/46762 (not
  peer-reviewed)
- Feynman, Leighton, Sands (1965). The Feynman Lectures on Physics, volume III, chapter 1, "Quantum
  Behavior". Addison-Wesley. (textbook; seen only through secondary sources)
- Goodman (1976). Some fundamental properties of speckle. Journal of the Optical Society of America
  66(11), 1145–1150. https://doi.org/10.1364/JOSA.66.001145
- Goodman (2008). Speckle with a finite number of steps. Applied Optics 47(4), A111–A118.
  https://doi.org/10.1364/AO.47.00A111 (only the abstract seen; the formula √(1 − 1/k) is our own
  recomputation)
- Hoppe (1984). Pólya-like urns and the Ewens' sampling formula. Journal of Mathematical Biology
  20, 91–94. https://doi.org/10.1007/BF00275863 (only the summary seen; the procedure confirmed by
  Leckey and Neininger (2012), Asymptotic analysis of Hoppe trees, preprint,
  https://arxiv.org/abs/1202.2439)
- Kark (no year). Antennen und Strahlungsfelder, 2nd revised and extended edition, Vieweg, sections
  11.1, 11.2.1 and 11.2.7. (textbook; only the table of contents seen)
- Löwe, Lippe, Rudolph, Welling (2022). Complex-valued autoencoders for object discovery.
  Transactions on Machine Learning Research (11/2022). https://openreview.net/forum?id=1PfcmFTXoa
  (in the Q52 check: the magnitude says whether a feature is there, the relative phase what it
  belongs to; a reason to keep strengths beside phases)
- Oppenheim, Lim (1981). The importance of phase in signals. Proceedings of the IEEE 69(5),
  529–541. https://doi.org/10.1109/PROC.1981.12022 (review; in the Q52 check: identity lies in the
  phase where magnitudes are alike, which strengths from the Task provide)
- Pothos, Busemeyer (2013). Can quantum probability provide a new direction for cognitive modeling?
  Behavioral and Brain Sciences 36(3), 255–274. https://doi.org/10.1017/S0140525X12001525
- Renou et al. (2021). Quantum theory based on real numbers can be experimentally falsified. Nature
  600, 625–629. https://doi.org/10.1038/s41586-021-04160-4
- Tamir, Cohen (2013). Introduction to weak measurements and weak values. Quanta 2(1).
  https://doi.org/10.12743/quanta.v2i1.14
- Zichner, Chandra (2009). Array antennas design in dependence of element-phasing. Advances in
  Radio Science 7, 267–271. https://doi.org/10.5194/ars-7-267-2009

The statistics of the comparison (Ville's inequality, test supermartingales) follow the sources
that `PREREG_STEP1_CIRCUITS.md`, section 8, gives.

## 20. Tickets and pull requests

1. `issues/01-soul-storage-and-settings.md`: the fields, `gene_id`, `SoulConfig`, the Mutator kept
   blind, everything off by default.
2. `issues/02-family-strengths-and-character.md`: families, strengths, readouts, the Character and
   the test of the Q57 formulas.
3. `issues/03-souletics-at-birth.md`, in two parts built in order: part A, the operators name their
   sources and see no Souls (no change of behaviour); part B, copy, mix, drift, new Souls, κ at
   birth, own streams, novelty by genes, seeds of calls.
4. `issues/04-complex-stage-vote.md`: vote phasors, the Array pattern, authors and tallies.
5. `issues/05-observations-and-coupling.md`: recording, the update at the commit, the `souls`
   event, the select phase kept out, resume.
6. `issues/06-soul-in-the-simulated-models.md`: phases in requests to simulated models; the
   designer's and the grid judge's choices.
7. `issues/07-upbringing-at-birth.md`: imprinting after the first task; teaching, off by default.
8. `issues/08-soul-observables.md`: the `soul` group of the `state` line.
9. `issues/09-program-verify-and-analysis.md`: `--soul`, `--verify`, end-to-end tests, the
   Character analysis.
10. `issues/10-preregistration-outline.md`: the outline of the pre-registered comparison.
11. `issues/11-pilot.md`: the pilot that sets the open numbers.
12. `issues/12-docs-adr-and-glossary.md`: ADR 0009, READMEs, report skeleton, and the glossary in
    its own pull request (Q53).
13. `issues/13-counted-experiment.md`: the counted Runs and the report, after the freeze.

Pull requests (Q47): tickets 01 to 09 and the ADR, README and report skeleton of ticket 12 form the
one pull request of step 2. The glossary changes go in a small pull request of their own (Q53). The
pilot and the frozen pre-registration (tickets 10 and 11) form a third, whose squash merge freezes
the pre-registration, as pull request #19 did for step 1; that pull request also adds the ADR's
link to the pre-registration, which does not exist before it. The results (ticket 13) form the
last.
German reports say "Seele (Phasenzustand)" (Q43) and "Seeletik" (Q66). Each pull request is
reviewed once by an independent reviewer; further rounds only on the owner's request.

## Comments

- 2026-10-31, design review. Two reviewers checked the design against the decisions and against
  the code. Changed after their findings: ties in the vote go to the count before the index, so
  the real readout loses no unanimous vote by exact cancellation (7.4 to 7.6); the genetic
  operators see no Souls, so their draws stay as today (9.1); Hoppe's N counts Workers (9.4); the
  Character is |Z|² as the glossary defines it, and the logs use a normalised Character (6.1,
  13.1); the designer's and the grid judge's Soul choices are limited to where they can neither
  change a grade nor come from a mere count, and they are counted (11.2 to 11.4, 13.1); peer
  requests carry no phases (7.8, 11.1); the backend contract becomes contract-v3 (11.1); the
  divergence generation is defined by outcomes, not by `born` events (15); gate G2 uses 2^20
  Souls (17); calls and run time are estimated (17); the real-readout arm is called V, not R
  (3, 12.3); the scope of Q45's third form is stated (1, 18). In the owner's questions, Q73 has two
  parts, Q74 an option (d), and Q67 to Q72 are corrected where they said too much.
- 2026-10-31, check of the revision. Every change above was checked again against the code, the
  step 1 records and our own reruns of the review's Monte Carlo checks. Added in the same revision:
  a newborn's current Souls start as its imprinted Souls before the commit's Observations apply
  (8.4); `soul_decided` has a serde default (11.1); the call bound under Q73 (c) (17); the counts of
  the three Monte Carlo checks of gate G2 (17).
