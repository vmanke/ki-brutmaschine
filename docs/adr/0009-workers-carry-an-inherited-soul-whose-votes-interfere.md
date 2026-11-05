# Workers carry an inherited Soul whose votes interfere

Step 2 of the build order (Q47) gives every Worker of a Cascade a Soul and every observer a
Coupling strength κ, and it prepares a pre-registered comparison of the Soul against two
controls. This record decides how a Soul is stored, where its Family strengths come from, what
its Character is, how a Stage vote with Souls decides, how judging changes Souls, how Souls pass
to children, what a child receives at birth, where the simulated models read the Soul, which
switches and controls exist, what is logged and checked, and that a Run without Souls stays as
it was. The design is the [spec](../../.scratch/seele-und-kappa/spec.md) (tickets 01 to 13 in its
`issues/`), and it builds the decisions Q42 to Q46 and Q51 to Q66 of
`.scratch/universum-erfinden/spec.md` with the answers to Q67 to Q74 in
`.scratch/seele-und-kappa/owner-questions.md`. The report is
[docs/SCHRITT2_SEELE.md](../SCHRITT2_SEELE.md) (English: [docs/STEP2_SOUL.md](../STEP2_SOUL.md));
its results follow with the counted Runs (ticket 13). The terms are those of `CONTEXT.md`, section
"Soul and Character"; new terms reach it in a pull request of their own, never inside a code pull
request (Q53).

**Two quantum forms now, one later.** Q45 names three quantum forms, each against a classical
control. Step 2 builds two of them, the scope of Q42 to Q46. The first, quantum-inspired
amplitudes (Q45 names Han and Kim 2002), is the Soul. The third, quantum probability for
judgments (Pothos and Busemeyer 2013), is built as the interfering Stage vote of Q46: the
judges' amplitudes add, and the squared magnitude decides. This is a simple form. It combines
the verdicts of several judges and models no single judgment. A full quantum-probability model
of one judgment has no step in the build order of Q47 yet. Both forms have the real readout as
their control. The second form, quantum Workers, follows in step 5 (Q47).

**A Soul is integer phases in the Traits (Q42, Q43, Q52).** In a Run on a Domain with n
hypothesis families, every Worker carries n phases, one per family in the Domain's order. A
phase is an integer p in 0..=65535 and stands for the angle 2πp / 65536. The phases are the
Soul's identity; it holds no strengths. `Traits::phase_state` (the code name of Q43) holds the
Soul, and `Traits::coupling` holds κ in thousandths, 0..=1000. Both are left out of the JSON at
their default, and `Traits` keeps `deny_unknown_fields`, so every genome, archive and snapshot
written before step 2 reads and writes back byte for byte, with the same id. Integers keep the
JSON, the id and every update of a Soul exact; floating point enters only the Character, the
vote and the logs. `Cascade::gene_id()` is the id without both fields. The birth Soul is part of
the genome and never changes. The current Soul, which Observations change, lives in the archive
member (Q56), and so does the imprinted Soul of Upbringing. So a member's id never changes, and a
parent can pass on only its birth Soul (Q65). No prompt and no request to a language model shows
a Soul. The Mutator sees the parents' genes only, and an answer that carries either field is
refused: "not admitted: the Mutator does not write Souls". `--promote` refuses a Run with Souls,
since a Soul belongs to its Run.

**Family strengths come from the Task (Q52, Q63, Q67 a).** A Domain names its families
(`Domain::families`) and states for each task how well each fits (`Domain::strengths`): at
least 0, at least one positive, the same for every Worker on the task. The strengths are
evaluator data; they never reach a prompt, a request or a Worker. Circuits have the 8 families of
the simulated designer: Linear, Phase, Hadamard, Compose, Search, Meet, Simplify and Previous. A
library target gives strength 1 to every family of a route that the step 1 calibration pins to
its reference, and 0 to every other; Previous fits no target. The table is in `LIBRARY` of
`src/cascade/circuit.rs`, and a dressed task keeps its target's strengths. Grids have the 7
families of the simulated grid model: Mirror, Rotate, Transpose, Gravity, Colour, Compose and
Previous. A puzzle gives 1 to the family of its hidden rule's geometric step, to Colour when the
rule carries a colour map, and to Compose when it carries both. Equal strengths, 1 for every
family on every task, are the control (arm E, setting `strengths: equal`). Learned strengths are
a later variant; step 2 leaves only the seam.

**The Character is computed, never stored (Q51, Q54, Q55, Q57).** A Worker's Character on task t
is χ(P, t) = |Σ_f s_f(t) · u(p_f)|², the squared magnitude of the sum of its phasors, each
weighted by its Family strength. u(p) is the unit phasor e^(2πip/65536), or in arm V the sign of
the nearer of 0 and half a turn (below). The logs use the normalised Character χ / Σ_f s_f²,
whose mean over independent uniform Souls is 1 for any strengths. No genome, member, snapshot or
event stores a Character; every use computes it from the current Soul, so it changes when an
Observation changes the Soul. `examples/soul_character.rs` recomputes every Character of a Run
from its record. For independent uniform Souls the contrast of the Character, its standard
deviation over its mean, is K_C = √(1 − Σ s⁴ / (Σ s²)²) with complex phases and K_R = √2 · K_C
with signs (our calculation; for k equal strengths K_C = √(1 − 1/k), as for speckle from few
steps, Goodman 2008). A test checks both on 2^20 fresh Souls per Domain within 0.01. This is the
prediction of Q57 for the model; for souls in nature Q48 stays open.

**Votes interfere, and a tie goes to the count (Q46, Q54, Q61, Q68 a, Q70 a).** The blind
pairwise judgments of a Stage vote stay exactly as they are: the same prompts, seeds, order
draws and calls. Only the tally changes. Judge j's phasor is v_j = Σ_f s_f · u(p_f) of its
current Soul, so its squared length is its Character: a Worker whose Soul fits the task votes
louder. With w_j(x) the verdicts of judge j for answer x, the Stage's Array pattern is
Π(x) = |Σ_j w_j(x) · v_j|². The answer with the largest Π goes on. A tie in Π, also when every Π
is 0, goes to the answer with more verdicts, and then to the lowest index. The count comes
before the index because two equally loud judges of opposite sign in arm V cancel exactly; with
the index first, the order of the answers would decide a quarter to a half of their unanimous
votes, and the comparison with arm V would measure that artefact. With equal phasors the
complex vote is today's count, and a vote in which every judge gives the same verdicts is never
lost, in either readout. A peer judgment has one judge and is no vote; its weight in the
Bradley-Terry ratings, judging skill and fitness stay as they are. Each member records the
authors of its answers and a tally of its votes at birth: complex, flipped against the count,
tied, the sum of their coherence, and the replies in which the phases decided a simulated
model's choice.

**Judging is an Observation (Q44, Q56, Q60, Q62, Q69 a).** A judgment in a Stage vote or in the
peer round that got a reply is an Observation by the judging Worker of the authors of the judged
answers. A call without a reply observes nothing, and the select phase never observes, so it
changes no Soul. When O observes T, for each family Δ is the signed phase gap from T to O,
wrapped to half a turn; T moves toward O by round(κ_O · Δ / 1000) and O toward T by
round(κ_O · Δ / 2000), both from the phases before the Observation, in integers with rounding
half away from zero. At κ = 0 nothing moves; at κ = 1 the observed takes the observer's phases,
and the observer moves half the way, as the atlas sketch computes. Souls are frozen during a
generation. The Observations are recorded with the evaluation and applied at the commit, one
after another in log order, so the bytes of a Run do not depend on the thread count, and a
stopped generation leaves no trace. A `souls` event after each `generation` event lists every
Worker whose current Soul changed. κ is the Coupling strength (German "Kopplungsstärke des
Beobachters"); the word "feeling" stays reserved for human feelings (Q34 to Q36). The name
follows the Kuramoto model (Acebrón et al. 2005). That an Observation aligns phases, where in
physics a stronger measurement destroys more interference, is a choice of the model.

**Souletics passes birth Souls and κ (Q55, Q58, Q64, Q65, Q66, Q71 a).** The breeder hands every
genetic operator its parents' genes only, so the operators make exactly today's draws and their
test that a child differs from both parents compares genes. Each operator names the source of
every child Worker: one parent Worker, or one from each parent where it pairs Workers. A child
Worker with one source copies its birth Soul; with two, each phase comes from one of them by a
coin. Then each phase moves by a uniform random step in −d..=d (`phase_drift`). Before that, a
coin gives the Worker an entirely new Soul with probability θ / (θ + N), where N counts the
Workers of the archive and those drawn before in this generation: every child brings its own
unit of Soul, nobody loses any, as in Hoppe's urn (Hoppe 1984). κ follows the same sources, moves
by a uniform step in −δ..=δ thousandths (`kappa_drift`) and is clamped to 0..=1000; seed Workers
start at `kappa_seed`, and with `kappa_fixed` κ never moves. κ is no gene: it is not in
`gene_id`, the Mutator never sees it, and crossover never crosses it. Souletics draws from its
own random streams, a fixed number of draws per Worker whatever the switches. A child must be new
by `gene_id`, and every model call takes its seed from the gene id. So arms whose archives hold
the same genes share every random number until the Soul changes an outcome. Souletics is one of
three channels of inheritance besides genes and Upbringing; German reports call it Seeletik.

**Upbringing at birth (Q59, Q73 a and d).** Imprinting: a child answers its first search task
with its birth Souls. Then each of its Workers moves toward the current Soul of each source
Worker, by the observed side of the rule of Q69 with the source's κ. It makes no model call, and
the parents do not change. The child answers its other search tasks with these imprinted Souls,
and they start its current Souls. Imprinting comes after the first answer, not after all,
because otherwise κ could change no search score in a circuit Run. Teaching: with `teach` on,
fixed code writes one prompt section at birth from each parent's best solved example; it ends
every system prompt the child sends, lives in the member and its `born` event, and never in the
genome. The simulated models read nothing from it, and teaching is off in the comparison.
Lessons need a language model to write them and wait for a later step.

**The simulated models read the phases, and only where they take the first of equals (Q46,
Q72 a).** `GenRequest::phases`, `GenResponse::soul_decided` and `ModelBackend::reads_phases()`
are contract-v3 of `docs/STAGE2_BACKEND_CONTRACT.md`; the phases are left out of the JSON when
absent, so every request key of a Run without Souls stays as it was. The breeder sends a
Worker's phases, readout applied, only in the solve and Stage-vote requests of a Cascade's own
evaluation, only to a backend that reads them, and never in a peer judgment; real models never
receive them. The simulated designer lets the Soul choose only among equally cheap exact
circuits that two or more families found, and only for a Worker with self-check. The grid
judge decides by the Soul only where both answers are explained and one by two or more
families; its coin is drawn as before. Costs, scores and the step 1 calibration stay as they
are. So with the simulated models the Soul acts mainly through the vote, the Observations and
Upbringing, and the logs count the replies in which it decided (`sim_decided`).

**Every part has a switch, and the controls are settings (Q42, Q44, Q63, Q64).** `BreedConfig`
holds `soul: Option<SoulConfig>`. Every field of `SoulConfig` is required and no other is
allowed: the readout, the strengths, the complex vote, the phases in the simulator, each kind of
Observation, κ at the seeds, `kappa_fixed`, `kappa_drift`, copy, mix, `phase_drift`, θ,
imprinting and teaching. A test runs each switch and checks its degenerate case. The settings
define the experiment, so a resume refuses a change. `brut-breed --soul s|v|k` sets the three
arms of the comparison: S, the Soul; V, the same hidden angles read only as a sign [Q70 a], so
that both arms share every random number and differ only in the readout; K, κ fixed at 0. Finer
settings come as a `soul` object in `--config`; both together are wrong usage. The numbers marked
as placeholders in the spec (κ at the seeds 0.2, its step 0.02, a phase step of 512 units, θ = 1)
are set by the pilot (ticket 11), and the presets of `--soul` follow them. The pilot also runs
arm E (equal strengths) and arm N (no Soul, which is arm A of step 1), for description only
[Q74 a]. Arm V's letter avoids arm R of step 1.

**What is logged and checked (Q33, Q51, Q56, Q57, Q58).** The `state` line gets a seventh group,
`soul`, left out without Souls. It is a pure function of committed state, as the other six are
(ADR 0006): the Soul population and its origins, the mean normalised Character, the contrast and
its formula for independent Souls, the order of the phases, how far experience moved the Souls,
the quantiles of κ, the Observations applied, the votes with their flips and ties and their
coherence, and `sim_decided`. Statistics of the Character weigh each distinct strength vector by
its number of tasks. The `born` event gets the imprinted Souls, where each Soul came from, the
teaching section, the authors and the vote tally. `--verify` compares these fields of every
member of a finished Run with its `born` event and each current Soul with its last entry in the
`souls` events, on both Domains; a grid Run without Souls is checked as before.

**Off by default, nothing changes (spec 14).** With no `soul` setting no new field is written, no
new random number is drawn and no request changes. The goldens of `tests/universe.rs` and
`tests/mlp_golden.rs` pass with their fixtures unchanged, and the 60-generation Run of pilot
seed 101 of step 1 still ends in the head hash e93ed521… that `docs/pilot-step1/pilot.json`
records.

**The comparison (Q57, Q74 a).** On circuits with the simulated designer, as a positive control,
the pass time of step 1 is compared per seed: does arm S pass before arm V, and before arm K?
Two separate claims, "S before V" and "S before K", each wrong with probability at most 1/200,
both together at most 1/100, by Ville's inequality as in step 1. The gates include the seed
Cascades solving exactly swap and cs, the formulas of the Character on 2^20 fresh Souls per
Domain, and the reproduction of step 1 without Souls. The pre-registration is written after the
pilot and frozen by the squash merge of its pull request; that pull request adds its link here.
The comparison may claim rung 1 of the ladder only: whether complex phases and coupling help this
model of Cascades. It says nothing about real language models, inventions or souls in nature
(Q48).

Rejected: strengths from the shape of a target's table (Q67 b; Toffoli would not look like
Hadamard although its best construction needs it) and a small base strength for every family
(Q67 c); vote phasors of length 1 (Q68 b; the Character would not act in the vote); an observer
that moves toward the observed one's new phases, or both meeting halfway (Q69 b, c; the second
contradicts the glossary, where the observed takes the observer's phases at κ = 1); a real
control that stores only signs or reads the real part (Q70 b, c; the second gives the Soul
strengths, which Q52 rules out); the index before the count as the tie rule (arm V would lose
unanimous votes by exact cancellation); κ as a gene that a mutation operator changes, or fixed
for the whole Run (Q71 b, c; the second contradicts Q44); a Soul that also orders the designer's
methods (Q72 b; the positive control would need a new calibration) and no Soul in the simulator
(Q72 c; contradicts Q46); imprinting before the first task (Q73 b) or by real judgments of the
parents, at one call per parent and child (Q73 c), and imprinting after all tasks (κ could
change no search score); teaching that passes the parents' strategy texts as lessons (Q73 e;
like an inheritance of genes); a fixed order of the two claims (Q74 b), one joint claim against
both controls (Q74 c) and a third claim against arm E (Q74 d; it can still be chosen after the
pilot, before the freeze); genetic operators that see the Souls (a crossover child with one
parent's genes and the other's Souls would pass the test it fails today, and the random stream
would change); novelty by the full id (a genetic copy with another Soul would count as new); the
Character stored in the record (Q55 says it is recomputed); Hoppe's N counted in members instead
of Workers (the urn's balls are Souls).

Status: accepted on 2026-10-31, when the owner merged the revised design (pull request #29).
Q67 to Q74 were decided that day by their recommended answers, under the owner's standing rule
that Claude, the AI assistant, decides open design questions by its recommendation and reports
each decision; the owner can overrule any of them. Claude chose the details of the build within
the spec; the tickets' Comments list them. The numbers of the presets and of the rule are set by
the pilot and frozen with the pre-registration.
