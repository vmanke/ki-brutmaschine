# Brutmaschine

A research incubator that breeds learning algorithms and measures, honestly, whether they generalise to tasks they have never seen.

## Language

### What is bred

**Learner**:
A candidate learning algorithm: three small programs (Setup, Predict, Learn) written in the DSL.
_Avoid_: agent, model, candidate (in learner mode)

**Candidate**:
In MLP mode, one network description (an MLP or a CNN, with its training settings) that gets trained and scored.
_Avoid_: learner (in MLP mode), individual

**DSL**:
The closed instruction set every Learner is written in; it has no loops, no I/O and a cost known before running.
_Avoid_: language, code

**Seed program**:
One of the hand-written Learners the search starts from.
_Avoid_: seed (alone; means a random-number seed)

### CNNs in MLP mode

**Model kind**:
Whether a Candidate's network is an MLP or a CNN. A Run breeds one kind or both, as `--models mlp`, `cnn` or `both` sets; MLP is the default, and MLP Runs behave as before CNNs.
_Avoid_: model type, architecture (alone)

**CNN**:
A Candidate whose network is convolutional: a stem, one to twelve CNN stages and a head, trained through candle.
_Avoid_: ConvNet, conv model

**CNN stage**:
One block of a CNN after its stem, with its block kind, width, kernel, stride, dilation, groups, options, normalization and activation.
_Avoid_: Stage (alone; that is part of a Cascade), layer

**CNN family**:
One of the 16 published designs that the CNN genome covers, such as ResNet or U-Net; a CNN stage belongs to the family of its block kind, and a CNN whose stages come from several families is a mixed form.
_Avoid_: family (alone; Task family means Learner tasks), architecture

**Base model**:
A known design that enters generation 0 of a Run instead of a random draw: the scaled-down CNN of one CNN family, which a Run with CNNs takes when it passes the Shape check of the Run's task (on a sequence task a 2-D one enters as its 1-D twin); a promoted model, which `brut init --promoted` copies from `candidates/models/` into the Run's config; or a base genome of the Run's config. In a Run that breeds CNNs or holds promoted models, the best base model by fitness is the bar for its Promotion.
_Avoid_: seed, seed program (that is a Learner), template

**Shape check**:
The check every CNN passes before it trains: its shapes fit the task, and its parameters, training FLOPs and estimated training time fit the Limits. A child that fails it is drawn again.
_Avoid_: validation (alone; Valid is a split), compile

**Time model**:
The estimate of a CNN's training time from its layers; the Shape check holds it to the time that `max_train_flops` buys at 1 GFLOP/s, so that the FLOP limit, not the clock, ends a training.
_Avoid_: benchmark, timing

**Operator**:
One kind of change the random mutator makes to a CNN, such as switching the CNN family of a CNN stage or inserting one; each child of one CNN gets exactly one, named in the archive.
_Avoid_: mutation operator, mutation (alone; a mutation applies one Operator)

### How Learners are made

**Proposer**:
Anything that turns parent Learners into a new child Learner.
_Avoid_: generator, mutator (unless it is the random one)

**Random mutator**:
The Proposer that changes parents by random edits to their programs.
_Avoid_: mutation operator

**LLM proposer**:
The Proposer that asks a language model for a new Learner, given parents and their public scores.
_Avoid_: AI mutator, agent

**Proposal**:
One child Learner returned by the LLM proposer, before it is checked and scored.
_Avoid_: suggestion, answer

**LLM round**:
A pause in the search, at fixed intervals, in which the LLM proposer is asked for a fixed number of Proposals from the current best Learners.
_Avoid_: Claude round, LLM step

**Broker**:
The only process that talks to the language-model API; it holds the API key and enforces the spending caps.
_Avoid_: proxy, client, gateway

### What Learners are tested on

**Task family**:
A rule that produces many related learning tasks, for example sine-like regression.
_Avoid_: dataset, task type

**Task**:
One concrete learning problem drawn from a Task family.
_Avoid_: episode, sample

**Tier**:
A group of Task families used for one purpose: Search and Select are public; the Hidden tiers are secret.
_Avoid_: split, level, stage

**Hidden grammar family**:
A secret Task family generated from a grammar that only the hidden evaluator knows.
_Avoid_: test family, secret task

**Canary task**:
A task with random labels that no honest Learner can learn; scoring above chance on it signals cheating.
_Avoid_: trap, honeypot

### How results are judged

**Hidden evaluation**:
Scoring finalists on the Hidden tiers in a separate process the search cannot see into.
_Avoid_: test run, final test

**Hidden query**:
One request to the hidden evaluator; it returns one rounded number and is counted against a fixed budget.
_Avoid_: call, lookup

**Primary endpoint**:
The single pre-registered result: the best Learner's lead over the compute-matched baseline on the Hidden grammar families, judged by a fixed rule.
_Avoid_: main score, headline number

**Compute-matched baseline**:
A standard learning method given the same compute as the Learner it is compared with.
_Avoid_: reference, control (alone)

**Constants-only ablation**:
A control search that may only change numeric constants, used to check whether program changes matter at all.
_Avoid_: baseline (this is not one)

**Finalist**:
A Learner chosen at the end of a search to go to Hidden evaluation.
_Avoid_: winner, champion

**Hall of fame**:
The search's running list of its best Learners.
_Avoid_: elite, archive

**All-time best**:
The permanent, bounded set of the best Learners ever seen, kept so a good Learner survives a panel rotation that drops it from the hall of fame; ranked by its best Select lead over the fixed Select tasks, not the current panel.
_Avoid_: hall of fame, elite, lineage archive

**Quarantine**:
The state of a Learner or Candidate whose hidden score is far below its public score; it may not become a parent or be approved.
_Avoid_: ban, blacklist

### Cascade breeding (Stage 2c)

**Worker**:
One model call configured as data: a model tag from the allow-list, a system prompt made of named sections, and Traits.
_Avoid_: agent, sub-AI, bot

**Trait**:
An inheritable setting that shapes how a Worker thinks, not what it thinks: thinking budget, temperature, answer length, self-check. Hard ceilings on Traits are code, not genes.
_Avoid_: hyperparameter (alone), gene (for prompts)

**Stage**:
The Workers of a Cascade that answer the same input in parallel; when they disagree they judge each other blind.
_Avoid_: layer, step

**Cascade**:
One to four Stages run in order, each seeing the task plus the answers chosen by all earlier Stages (the residual path); the unit that is bred and competes.
_Avoid_: pipeline, chain, ensemble, agent

**Judging skill**:
How often a Cascade picks the right answer of a pair whose truth is known, with a Laplace prior (a Cascade never judged this way keeps 1/2); a reply without a usable verdict counts as a wrong pick. It enters fitness with a fixed weight.
_Avoid_: judge score, reliability

**Peer rating**:
A Cascade's weighted Bradley-Terry rating from blind pairwise judgments by Cascades that are no parent, child or sibling of either judged Cascade; it breaks fitness ties and chooses crossover partners, and never replaces the solved rate.
_Avoid_: Elo (it is not one), reputation

**Mutator**:
The language model, normally Claude, that rewrites one parent Cascade or merges two in an LLM round; its answers are recorded and replayed, and a missing answer pauses the Run.
_Avoid_: LLM proposer (that one writes Learners), meta-agent

**Search panel**:
The public puzzles on which Cascades are scored during breeding.
_Avoid_: training set

**Select panel**:
Puzzles unseen during breeding, on which only finalists and seed Cascades are scored at the end; descriptive, not the Primary endpoint.
_Avoid_: test set, hidden panel

**Promotion**:
Moving the winners of a finished Run into the candidate pool through a human-reviewed draft pull request: the two best finalist Cascades into `candidates/cascades/`; or, in MLP mode and from a Run that breeds CNNs or holds promoted models, of the two best models (MLPs or CNNs) by hidden score those that beat the Run's best Base model on the hidden split, each with its provenance, into `candidates/models/`. A model equal by content to one in the pool adds nothing. Later Runs take the merged models into generation 0 as Base models.
_Avoid_: deploy, release, merge (the human merges, not the Run)

### LLMs using CNNs (planned)

**CNN Worker**:
A Worker whose model tag is `cnn:` plus the content id of a CNN genome. For each task it trains a fresh CNN of that genome on the task's worked examples and answers with the CNN's output grid. It ignores its prompt sections, its Traits and earlier answers. It refuses tasks whose examples change the grid size.
_Avoid_: CNN agent, vision model, net (alone)

**Per-task training**:
Training a CNN from its genome's own seed on the worked examples of one task only, just before it answers that task. No clock can stop it. Fixed limits bound its work before it starts. The trained CNN serves only that task and is never written to disk.
_Avoid_: test-time training, fine-tuning, pretraining

**Call record**:
The stored request and full reply, or error, of one call that a Run makes to a model that is not simulated: a CNN Worker, the CNN tool or a real language model. It is one line of the Run's `llm/worker-calls.jsonl`, filed under a key of its model tag and its request. A resumed Run uses a stored reply instead of calling again. A replayed Run answers these calls from their records and never calls the model. `brut-breed-ollama` already keeps such records for the answered calls of real language models (contract I7); CNN calls and errors are planned.
_Avoid_: cache, transcript, log

**Neighbour rule**:
A hidden grid rule that sets a cell's colour from its own colour and the colours of its four neighbours. No hypothesis of the simulated Worker model fits it. A CNN is meant to learn it from the worked examples. Half of the puzzles of the grid-mix Domain follow such a rule; it is the positive control of CNN Workers and of the CNN tool.
_Avoid_: local rule, cellular automaton

**Silent CNN**:
A CNN Worker or CNN tool that refuses every task, under the reserved tag `cnn:0000000000000000`. It costs its calls and gives nothing. It is the negative control: breeding should not keep it.
_Avoid_: dummy, null model

**Knock-out**:
Scoring a Cascade again on the same tasks with the same call seeds, with its CNN Workers or its CNN tool silenced. The paired difference is what the CNN contributes.
_Avoid_: ablation (here that names a control search, as in Constants-only ablation), lesion

**Architect**:
The language model that writes CNN genomes in Architect rounds of an MLP-mode Run, given parents and their public validation scores. Its answers arrive as files and are recorded and replayed; a missing answer pauses the Run.
_Avoid_: designer (that is the simulated circuit model), Mutator (that one writes Cascades), LLM proposer (that one writes Learners)

**Blueprint**:
One CNN genome returned by the Architect, before it is checked and trained. It is checked strictly and never repaired.
_Avoid_: proposal (that is a Learner), design, suggestion

**Architect round**:
A generation of an MLP-mode Run, at fixed intervals, in which the Architect is asked for a fixed number of Blueprints.
_Avoid_: LLM round (that one asks for Learners or Cascades)

**CNN tool**:
A CNN that a language-model Worker may ask, while it answers, for the CNN's output grid on a grid the Worker writes. It learns by Per-task training on the task's worked examples. A Run has at most one, and its tag is not on the Workers' allow-list.
_Avoid_: plugin, function call, oracle

**Tool turn**:
One ask and reply inside one answer: the Worker's request, the CNN tool's grid, and the Worker's next reply. Each Tool turn costs one more call of the model. It also costs one call of the CNN tool, unless the requested grid has the wrong size.
_Avoid_: step, round, function call

**Tool budget**:
The Trait `tool_calls`: the most Tool turns one answer of a Worker may take, from 0 to a ceiling of 3 in code. A Run may set a lower ceiling. At 0 the Worker has no tool.
_Avoid_: tool limit (alone), quota

### Universes and invention (step 1)

**Domain**:
A kind of task Cascades are bred on, seen from the evaluator side: it makes the panels, renders and reads prompts, grades answers and says which of two answers is better. It never sees a Cascade. Grid puzzles and circuits are Domains; a Run names its Domain with `--domain`.
_Avoid_: task family (that means Learner tasks; doc comments in `src/cascade/` still use it loosely for a Domain), environment

**Gene pool**:
The breeder side of a Domain: the seed Cascades, the phrase pool of the section operators, the Mutator's brief and the admission rule every new Cascade must pass.
_Avoid_: genome (that is one Cascade), library

**Universe state**:
The per-generation record of a Run's six observables (performance, diversity, lineage, honesty, judging, cost), logged as one `state` event after every committed generation. It is computed from committed state only, so a resumed Run writes the same records as a straight one.
_Avoid_: metrics, stats, telemetry

**Target**:
A unitary that a circuit must implement exactly, given by its action on every basis state and built from its definition, never from a circuit. Only Targets with a verified Reference enter a panel.
_Avoid_: goal, spec

**Reference**:
A Target's sourced T and CNOT counts, which one circuit reaches together, each with a bound status: Proven (a published lower bound), Derived (our argument from published results) or BestKnown (a published construction only).
_Avoid_: optimum (unless Proven), baseline

**Dressing**:
The disguise drawn per task: a relabelling of the qubits and X gates on inputs and outputs. It keeps the optimal T and CNOT counts, so the Reference still holds, while a memorised circuit no longer fits.
_Avoid_: augmentation, noise

**Exact check**:
Deciding a circuit's correctness by exact equality with the Target up to one global phase, in integer arithmetic over Z[omega]; the fidelity is only reported.
_Avoid_: fidelity test, tolerance

**Integrity alarm**:
The stop raised when an exact answer goes below a Proven or Derived Reference: then the checker or the Reference is wrong, and the generation is not committed.
_Avoid_: discovery

**Invention candidate**:
An exact answer below a BestKnown Reference. `rediscovery.json` records the cheapest one per Target, with `literature_checked = false`; any others are in the logged answers. It is called new only after a human has checked the literature (Q38).
_Avoid_: invention, discovery, record

**Designer**:
The simulated model that writes circuits in circuit Runs, under the tags sim:small and sim:large. It is a positive control: the method families a system prompt names switch its synthesis on, and its thinking budget bounds its search.
_Avoid_: oracle, solver

**Rediscovery**:
A bred Cascade solving a Target that the seed Cascades miss, exactly and at Reference cost. The share D of such Targets is counted in `rediscovery.json` and reported per Run at generation 1000; the pre-registered result of step 1 is the verdict of the sequential rule on the pass times.
_Avoid_: invention, success rate

**Pass time**:
The first generation in which a Run's bred Cascades have solved more than half of the Targets its seed Cascades miss, or never. It is read from the first generations in `rediscovery.json`; a counted Run passes when its pass time is at most 1000, and the sequential rule of step 1 compares the pass times of both Arms of a seed.
_Avoid_: T alone (T also counts T gates), success generation, hitting time

**Arm**:
One configuration of a paired experiment, run on the same seeds as the others: arm A uses Darwin Gödel selection, arm R uniform parent choice.
_Avoid_: condition, group

### Soul and Character (planned, step 2)

**Soul**:
A Worker's hidden, inherited phases, one per hypothesis family, code name `phase_state`. The phases are its identity; their strengths are not part of the Soul but are the Family strengths of the Task. It is never seen directly, only through the Character it produces.
_Avoid_: spirit, psyche, mind

**Soul population**:
All Souls of a universe. Every child brings its own unit of Soul and nobody gives any up; since the archive keeps every member, the Soul population only grows.
_Avoid_: soul pool, total soul

**Family strength**:
How well a hypothesis family fits a Task, as the Domain states it; the same for every Worker on that Task. It weights the Soul's phases when the Character is computed.
_Avoid_: prior, weight (alone)

**Character**:
The visible pattern a Worker's Soul produces: for each Task, the squared magnitude of the sum of the Soul's phases, each weighted by its Family strength. It is recomputed from the Soul each time and never inherited; when an Observation changes the Soul, the Character changes with it.
_Avoid_: personality, temperament, profile

**Array pattern**:
The joint pattern of a Stage: for each answer, the squared magnitude of the sum of its Workers' contributions. Observations steer it by shifting phases.
_Avoid_: stage character, collective judgment, vox populi

**Observation**:
A Worker judging answers of others, in the vote of its own Stage or as a peer judge. It changes the Souls of both sides, as strongly as the observer's Coupling strength says.
_Avoid_: measurement (alone), look

**Coupling strength**:
The inherited number κ between 0 and 1 with which an observer pulls the phases of the Worker it observes toward its own and is pulled in turn. At 1 the observed takes the observer's phases.
_Avoid_: feeling, measurement strength, gain

**Upbringing**:
What a child receives from its parents' lives besides genes and Soul. In imprinting, its parents observe its first answers and so shift its Soul toward theirs. In teaching, a prompt section is written fresh at birth from its parents' best solved examples and lessons; it is never inherited as a gene.
_Avoid_: training, education (alone)

**Souletics**:
Soul inheritance: a child's Soul is a copy of one parent's Soul from birth, or a mix of both parents' Souls, with small random changes, and now and then an entirely new Soul. It is one of three channels of inheritance besides genes and Upbringing; experience passes only through Upbringing. German reports call it Seeletik.
_Avoid_: soul genetics

### Records and control

**Run**:
One complete search in its own run directory, with its own configuration and archive.
_Avoid_: experiment (an experiment can span several Runs), job

**Control run**:
A Run identical to an LLM Run (same settings, secret and seeds) but without the LLM proposer, used to isolate the language model's effect.
_Avoid_: baseline run, reference run

**Pilot run**:
A short, cheap Run with no Hidden queries that measures cost and Proposal quality before full Runs are approved.
_Avoid_: test run, dry run

**Lineage archive**:
The tamper-evident, append-only record of every Learner or Candidate, its parents and its scores.
_Avoid_: log, history, database

**Kill switch**:
A STOP file in the run directory that ends a Run cleanly within about a second.
_Avoid_: abort, emergency stop
