# Step 2: the Soul and the Coupling strength

German original: [SCHRITT2_SEELE.md](SCHRITT2_SEELE.md).

## 1. Purpose

Step 2 of the build order (Q47) gives every Worker a **Soul**: hidden, inherited phases, one
per hypothesis family of a Run's Domain (Q42, Q43, Q52). No model sees the Soul. It shows itself
only through its **Character** on each task (Q51, Q57). In a Stage, the Workers' votes add as
complex numbers, and the squared magnitude decides (Q46). Every judgment is an **Observation**:
it pulls the Souls of both sides together, as strongly as the observer's **Coupling strength** κ
says (Q44, Q60, Q62). Souls pass to children by **Souletics** (Q64 to Q66), and **Upbringing**
shapes them at birth (Q59). Every part has a switch. Without a Soul, the default, every Run
writes the same bytes as before step 2.

Step 2 ends with a pre-registered comparison on circuits with the simulated designer, as a
positive control: the Soul (arm S) against the same Soul read as signs (arm V) and against
κ = 0 (arm K). This report describes what is built and how to run and verify a Run. The pilot
(ticket 11), the pre-registration (ticket 10) and the counted Runs with their results (ticket 13)
follow; we add those results here. The decisions are in
[ADR 0009](adr/0009-workers-carry-an-inherited-soul-whose-votes-interfere.md), the design in
[.scratch/seele-und-kappa/spec.md](../.scratch/seele-und-kappa/spec.md), and the terms in
`CONTEXT.md`.

## 2. What step 2 built

- **The Soul in the genome** (ticket 01, `src/cascade/genome.rs`). `Traits::phase_state`, the
  code name of Q43, holds the Soul: one phase per family, an integer p from 0 to 65 535 that
  stands for the angle 2πp/65 536. `Traits::coupling` holds κ in thousandths, from 0 to 1000.
  Both fields are left out of the JSON while they are empty. So every genome written before
  step 2 reads and writes back byte for byte, with the same id. `Cascade::gene_id()` is the id
  without both fields. Integers keep the JSON, the id and every change of a Soul exact; floating
  point appears only in the Character, the vote and the logs.
- **Three Souls of a Worker.** The birth Soul is part of the genome and never changes; a parent
  passes on only this one (Q65). The imprinted Soul is fixed at birth ("Upbringing" below). The
  current Soul lives in the archive member, and Observations change it (Q56).
- **The Mutator sees no Soul.** Its requests show only the parents' genes. The breeder refuses
  an answer that carries either field ("not admitted: the Mutator does not write Souls").
  `--promote` refuses Runs with Souls: a Soul belongs to its Run.
- **Family strengths** (ticket 02, `src/cascade/domain.rs`, `circuit.rs`, `sim.rs`). The Domain
  names its families and states, per task, how well each fits. Circuits have the 8 families of
  the designer: Linear, Phase, Hadamard, Compose, Search, Meet, Simplify and Previous. A target
  gives strength 1 to every family of a route that the step 1 calibration pins to its
  reference, and 0 to every other [Q67 a]. Grids have the 7 families of the grid model: Mirror,
  Rotate, Transpose, Gravity, Colour, Compose and Previous. A puzzle gives strength 1 to the
  family of its geometric step, to Colour for a colour map, and to Compose for both. Equal
  strengths (1 for every family) are the control. The strengths never reach a prompt or a
  request.
- **The Character** (ticket 02, `src/cascade/soul.rs`). On task t it is
  χ = |Σ_f s_f(t) · u(p_f)|². Here u(p) is the unit phasor e^(2πip/65 536), and in arm V the sign
  +1 or −1 of the nearer of 0 and half a turn [Q70 a]. The logs use the normalised Character
  χ / Σ s_f², whose mean over independent uniform Souls is 1. No record stores a Character
  (Q55); every use computes it from the Soul. For independent Souls the contrast of the
  Character, its standard deviation over its mean, is by our calculation
  K_C = √(1 − Σ s⁴ / (Σ s²)²) with complex phases and K_R = √2 · K_C with signs (Q57). A test
  checks both formulas on 2^20 fresh Souls per Domain within 0.01.
- **The complex vote** (ticket 04, `src/cascade/arena.rs`, `soul.rs`). The blind pairwise
  judgments of a Stage stay as they are; only the tally changes. Worker j's phasor is
  v_j = Σ_f s_f · u(p_f) of its current Soul, so its squared length is its Character [Q68 a].
  With w_j(x) the verdicts of j for answer x, the Stage's **Array pattern** is
  Π(x) = |Σ_j w_j(x) · v_j|² (Q61). The answer with the largest Π wins. A tie goes to the number
  of verdicts, then to the order of the answers [Q70 a]. With equal phasors this is today's
  count, and a vote in which all Workers judge alike is never lost. Each member records the
  authors of its answers and a tally of its votes.
- **Observations and κ** (ticket 05). A judgment with a reply, in a Stage vote or in the peer
  round, is an Observation (Q62); the select phase never observes. For each family the
  observed Worker moves toward the observer by round(κ · Δ / 1000) phase units, and the
  observer back by round(κ · Δ / 2000), in integers [Q69 a]. All Observations of a generation
  act at its commit, in log order. A `souls` event names every Worker whose Soul changed. A
  member votes on the select panel with its current Souls, so a resume to more generations
  scores again every member whose Soul changed after an earlier select phase.
- **Souletics** (ticket 03). The genetic operators see genes only, so they draw the same random
  numbers as before step 2. A child Worker copies the birth Soul of its source Worker or mixes
  two, phase by phase. Then each phase moves by a random step of at most `phase_drift` units.
  With probability θ / (θ + N) the Worker gets an entirely new Soul instead, as the new colour
  in Hoppe's urn (Q58, Q64); N counts the Workers in the archive and those drawn before. κ
  follows the same sources, with a step of at most `kappa_drift`, and is no gene [Q71 a].
  Souletics draws from its own random streams. A child must be new by its genes, and model
  calls take their seed from the gene id. So arms with the same genes share every random number.
- **Upbringing** (ticket 07, `src/cascade/upbringing.rs`). Imprinting: after its first search
  task each Worker of a child moves toward the current Soul of its source Worker, by the rule
  of the observed side [Q73 a]. It costs no call, and the parents do not change. Teaching: fixed
  code writes a section from the parents' best solved examples into the child's prompts
  [Q73 d]. The simulated models do not read it, and teaching is off in the comparison.
- **The Soul in the simulated models** (ticket 06, `src/backend.rs`, `designer.rs`, `sim.rs`).
  Requests to simulated models carry the Worker's phases (contract-v3 of
  [STAGE2_BACKEND_CONTRACT.md](STAGE2_BACKEND_CONTRACT.md)), never in a peer judgment and never to
  a real model. The designer lets the Soul choose only between equally cheap exact circuits that
  two or more families found, and only with self-check; the grid model only in its judgment,
  when both answers are explained [Q72 a]. Costs, scores and the step 1 calibration stay as they
  are. `sim_decided` counts the replies in which the phases decided.
- **The `soul` group of the `state` line** (ticket 08, `src/cascade/observe.rs`), see section 4.
- **Program, verification and analysis** (ticket 09). `brut-breed --soul s|v|k`, `--verify` for
  Runs with Souls on both Domains, and `examples/soul_character.rs`, see section 3.
- **Without a Soul nothing changes.** Without the `soul` setting no Run writes a new field,
  draws a new random number or changes a request. The goldens of `tests/universe.rs` and
  `tests/mlp_golden.rs` pass with their fixtures unchanged, and the 60-generation Run of pilot
  seed 101 of step 1 ends in its recorded head hash e93ed521… (release test
  `souls_off_reproduces_the_step1_pilot`). At the end of the build we ran the counted Run A-1
  of step 1 again by hand: its five files have the hashes that
  [step1-counted/hashes.txt](step1-counted/hashes.txt) records.

## 3. A Run with Souls and how to verify it

```sh
cargo build --release

# Arm S, the Soul (pilot seed 901, 60 generations)
./target/release/brut-breed --out runs/s901-S --domain circuits --search 33 --select 22 \
  --children 8 --generations 60 --seed 901 --max-calls 200000 --soul s

# The controls: arm V reads the same Soul as signs, arm K keeps κ at 0
./target/release/brut-breed --out runs/s901-V --domain circuits --search 33 --select 22 \
  --children 8 --generations 60 --seed 901 --max-calls 200000 --soul v
./target/release/brut-breed --out runs/s901-K --domain circuits --search 33 --select 22 \
  --children 8 --generations 60 --seed 901 --max-calls 200000 --soul k

# Verify the log, the report, rediscovery.json, the archive and the Souls, without a model call
./target/release/brut-breed --out runs/s901-S --verify

# Recompute the Character of every Worker from the record
cargo run --release --example soul_character -- runs/s901-S
```

`--soul` sets the settings of an arm from section 12.2 of the design:

| Field | Meaning | Arm S | Arm V | Arm K |
|---|---|---|---|---|
| `readout` | complex or signs (Q42, Q70) | complex | real | complex |
| `strengths` | the Domain's strengths or equal ones (Q63) | domain | domain | domain |
| `vote` | the complex vote (Q46) | on | on | on |
| `simulator` | phases reach the simulated models (Q72) | on | on | on |
| `observe_vote`, `observe_peer` | judgments of the Stage vote, of the peer round are Observations (Q62) | on | on | on |
| `kappa_seed` | κ of the seed Workers, in thousandths (Q71) | 200* | 200* | 0 |
| `kappa_fixed` | κ never moves (Q44) | off | off | on |
| `kappa_drift` | largest κ step at a birth, in thousandths | 20* | 20* | 20 |
| `copy`, `mix` | children copy Souls; two sources mix (Q64) | on | on | on |
| `phase_drift` | largest phase step at a birth, in phase units (Q64) | 512* | 512* | 512* |
| `new_soul_theta` | Hoppe's θ (Q64) | 1* | 1* | 1* |
| `imprint`, `teach` | imprinting; teaching (Q59) | on; off | on; off | on; off |

A star marks a placeholder that the pilot sets (ticket 11); the lines below change with it.
Finer settings come with `--config <file>`: a complete Run configuration with a `soul` object
that holds each of the 15 fields. `--soul` next to such a file is wrong usage (exit code 2).
`--soul` defines the experiment; a resume refuses it, as it refuses `--domain` and
`--selection`. The seeds 901 to 923 are pilot seeds of step 2 and never count.

A Run prints the same lines as in step 1. For arm S on seed 901 the line about
`rediscovery.json` reads:

```
brut-breed: wrote runs/s901-S/rediscovery.json: the seeds solve 2 of 11 targets, bred Cascades 2 of the other 9; D > 1/2: no; calibrated
```

and `--verify` prints:

```
brut-breed: 2046 events, hash chain intact, head e38339538f19d85fccd1d1a7a1c3a0435e1989eec7838e6b6e0c5a46e0e692a1; checkpoint at entry 2046 matches; settings match; report recomputed and equal; rediscovery recomputed and equal
```

For arm V the head hash is 3516f3d2b956a6002eddb95d1340d3ffbd59b84963bf89e60aec8301069ea41d,
for arm K e431bae607b74d20f2e810075f121c8a25ba8c2e579717125aec6274b4107afa; both also have 2046
events and the same line about `rediscovery.json`. We measure 3.9 to 4.2 s per Run (release
build, 4 threads); the debug build writes the same bytes.

**What `--verify` checks.** Everything it checks in step 1 (the hash chain, the checkpoint's
anchor, the settings, the report, and for circuits `rediscovery.json`). For a finished Run with
Souls it also compares every archive member with its `born` event, the five fields of step 2
included: imprinted Souls, where each Soul came from, teaching, the authors of the answers and
the tally of votes. The birth Souls and κ are in the genome and are compared with it. Each
Worker's current Soul must equal its last entry in the `souls` events, or its imprinted Soul when
no event names it. This now holds for grid Runs with Souls too; their line ends with "archive
and Souls match the log". A Run with Souls keeps every select-panel answer, so on grids too a
report that the checkpoint cannot recompute, because the checkpoint is unfinished or lacks
select-panel answers, is an error, as for a circuit Run. Without a report the line says why
nothing was checked: "archive and Souls not checked: the run is not finished", or, for a
finished Run whose report is missing, "… there is no report.json; resume the run to write it".
A grid Run without Souls is checked as before and prints the same line as before step 2. Any
difference ends `--verify` with exit code 1 and a message "integrity: …". `--verify` makes no
model call, so it does not recompute the Observations; a rerun repeats them byte for byte, and
the tests check that at 1 and 4 threads.

**The Character from the record.** `examples/soul_character.rs` reads `config.json`, the
checkpoint and the run start in the log. For every member, every Worker and every target (for
grids, every kind of rule: its geometric step, with "+colours" for a colour map) it prints the
Character and the normalised Character of the current Soul, with the Run's readout and
strengths. The output is a table with tabs; lines that start with "#" name the families and,
per target, its number of search tasks and its strengths. Each number is printed in the
shortest form that reads back to the same f64. A test compares every row exactly with
`character()` and `character_norm()` on the Souls of the checkpoint.

## 4. What the `soul` group means

In Runs with Souls, the `state` line after every committed generation holds a seventh group,
`soul`. Without a Soul it is absent. Like the other six groups it depends only on committed
state, so a stopped and resumed Run writes the same lines as a straight one. Means over tasks
run over the search panel, each task once; we compute a statistic of the Character once per
distinct strength vector and weight it by the number of its tasks. An undefined value is
absent rather than NaN.

| Field | Meaning |
|---|---|
| `souls` | Workers in the archive: the Soul population (Q58); it never falls |
| `souls_born`, `souls_new` | Workers born in this generation, and those of them with an entirely new Soul (Q64) |
| `origins`, `origin_entropy_bits` | distinct origins of the Souls in the archive, Hoppe's colours, and the entropy of their frequencies |
| `character_norm_mean` | the mean normalised Character over the Workers and the search tasks, from the current Souls (Q51) |
| `contrast` | the mean over the search tasks of the Character's contrast over the Workers (Q57) |
| `contrast_independent` | the same mean of the formula K_C or K_R: the value for independent Souls |
| `order`, `order_mean` | per family, the length of the mean readout phasor over all Workers (1: all alike; near 0: spread out), and their mean |
| `moved_mean` | the mean circular distance between current and birth Soul over Workers and families, in degrees: what experience did |
| `kappa` | quantiles of κ over the Workers, as fractions |
| `observations_vote`, `observations_peer` | Observations of this commit, by kind (Q62) |
| `shift_mean` | the mean phase change of the Workers that moved at this commit, in degrees (Q56) |
| `votes`, `votes_flipped`, `votes_tied` | complex votes of this generation's newborns on the search panel; those whose winner differs from the count's; those in which two or more answers shared the largest Π |
| `coherence_mean` | the mean coherence of these votes, \|Σ_j v_j\| / Σ_j \|v_j\|: 1 when all phasors point alike, near 0 when they cancel |
| `sim_decided` | replies to this generation's newborns in which the phases decided a simulated model's choice |

An example from the three Runs of section 3, measured in generation 60:

| Field | Arm S | Arm V | Arm K |
|---|---|---|---|
| `souls`, `origins` | 1554, 10 | 1554, 10 | 1554, 10 |
| `character_norm_mean` | 0.973 | 0.678 | 1.017 |
| `contrast` (formula) | 0.534 (0.655) | 1.217 (0.927) | 0.546 (0.655) |
| `order_mean` | 0.663 | 0.651 | 0.544 |
| `moved_mean` | 14.4° | 14.4° | 0° |
| `kappa`, median | 0.206 | 0.206 | 0 |

Over all 61 generations we count, in each arm, 32 complex votes, 64 Observations in votes and
2937 in peer rounds. No vote flipped against the count, and `sim_decided` stayed 0. In arm V the
tie rule decided 12 votes. On this seed the three arms produce the same 484 newborns, member by
member equal in genes, operator, solved tasks, graded values and answers: up to generation 60
the Soul changed no outcome here. In arms S and K the contrast lies below its formula, as shared
ancestry and, in arm S, coupling lead one to expect. In arm V it lies above, unlike what section
13.1 of the design expects. With signs, shared phases push the Character of many Workers to 0:
on the two targets with two fitting families, CH and CCZ, we measure with
`examples/soul_character.rs` a Character of 0 for 70.5 % and 90.3 % of the 1554 Workers, where
independent Souls would give 50 %. There the contrast is 1.55 and 3.06 instead of 1.00. This is a
measurement on one pilot seed, not a result; the pilot examines it on 12 seeds.

**The `born` and `souls` events.** Unchanged without a Soul. With Souls every `born` event holds
five more fields: `imprinted` (the imprinted Soul of each Worker), `soul_sources` (the source
Workers, whether the Soul is new, and its origin), `teaching` (the teaching section, when there
is one), `authors` (per search task, the Workers whose answer was chosen) and `votes` (the tally
of the votes). The genome in the event holds the birth Soul and κ. After each `generation` event
a `souls` event names every Worker whose current Soul changed at the commit, with its new phases.

## 5. The pilot and the decision

- **Pilot (ticket 11, pull request #50).** Arms S, V and K, plus E (equal strengths) and N (no
  Soul, which is arm A of step 1), ran on the pilot seeds 901 to 912 over 1000 generations. Its
  report is [docs/pilot-step2/README.md](pilot-step2/README.md). All 60 Runs passed `--verify`,
  and seed 901, run again in all five arms, gave the same bytes.
- **The Soul acts late.** It changes an outcome only through a complex vote that flips its winner
  or a reply in which the phases decide the simulated designer's choice. Both are rare, and they
  often come after the pass time. Against arm V, arm S passed first 4 times, arm V 4 times, and 4
  triples tied. Against arm K the counts were 3, 3 and 6. Arm S never passed before arm N.
- **Decision.** At the pilot's estimates, the planned comparison of ticket 10 would prove a claim
  with probability below 0.004. On 2026-11-02 the owner therefore decided not to run the counted
  experiment (tickets 10 and 13). [docs/PREREG_STEP2_SOUL.md](PREREG_STEP2_SOUL.md) stays as the
  record of the rejected design.
- **Next.** A search for a Soul model that changes outcomes early and often, so that its effect
  can be measured.

## 6. Limits

- **A model.** Soul, Character and Coupling strength are terms of a model. Nothing here is
  physically quantum, and the comparison says nothing about souls in nature (Q48 stays open).
- **A positive control.** The simulated designer is not a language model. What we find with it
  says nothing about real models.
- **A weak effect in the simulator.** The designer lets the Soul choose only between equally
  cheap exact circuits that two or more families found, and only with self-check [Q72 a]. So the
  Soul acts mainly through the vote, the Observations and Upbringing. On seed 901 it changed no
  outcome up to generation 60 (section 4). The pilot must show whether the comparison has enough
  power.
- **Unanimous judgments.** The designer judges by checking, so the Workers of a Stage mostly
  judge alike. Such a vote is lost in neither arm. The complex vote changes a winner only where
  the judgments split.
- **Hand-written strengths.** The circuit strengths come from the routes that step 1 pinned. On
  swap and cs one family fits, so every Character there is 1.
- **The contrast formula** holds for independent Souls. In a Run, ancestry and coupling make
  Souls alike, and 4 seed Workers give only a rough estimate in generation 0.
- **Teaching without effect.** The simulated models do not read the teaching section; its effect
  needs real models.
- **Platform.** The phasors use `cos` and `sin` of the standard library. Their last bit may
  differ between platforms. So we compare Runs on one toolchain and platform, rustc 1.97.0 on
  x86_64 Linux, as in step 1.

## 7. Sources

The verified sources of the design are in section 19 of the
[spec](../.scratch/seele-und-kappa/spec.md). This report relies on:

- Acebrón, Bonilla, Pérez Vicente, Ritort, Spigler (2005). The Kuramoto model: A simple paradigm
  for synchronization phenomena. Reviews of Modern Physics 77, 137.
  https://doi.org/10.1103/RevModPhys.77.137 (the name of the Coupling strength)
- Goodman (1976). Some fundamental properties of speckle. Journal of the Optical Society of
  America 66(11), 1145–1150. https://doi.org/10.1364/JOSA.66.001145 (contrast 1 for fully
  developed speckle)
- Goodman (2008). Speckle with a finite number of steps. Applied Optics 47(4), A111–A118.
  https://doi.org/10.1364/AO.47.00A111 (only the abstract seen; we recomputed √(1 − 1/k)
  ourselves)
- Hoppe (1984). Pólya-like urns and the Ewens' sampling formula. Journal of Mathematical Biology
  20, 91–94. https://doi.org/10.1007/BF00275863 (only the summary seen)
- Pothos, Busemeyer (2013). Can quantum probability provide a new direction for cognitive
  modeling? Behavioral and Brain Sciences 36(3), 255–274.
  https://doi.org/10.1017/S0140525X12001525
- Renou et al. (2021). Quantum theory based on real numbers can be experimentally falsified.
  Nature 600, 625–629. https://doi.org/10.1038/s41586-021-04160-4 (the question behind arm V)
