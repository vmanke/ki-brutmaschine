# Brutmaschine

A research incubator that breeds learning algorithms and measures, honestly, whether they generalise to tasks they have never seen.

## Language

### What is bred

**Learner**:
A candidate learning algorithm: three small programs (Setup, Predict, Learn) written in the DSL.
_Avoid_: agent, model, candidate (in learner mode)

**Candidate**:
In MLP mode, one network description (layers and training settings) that gets trained and scored.
_Avoid_: learner (in MLP mode), individual

**DSL**:
The closed instruction set every Learner is written in; it has no loops, no I/O and a cost known before running.
_Avoid_: language, code

**Seed program**:
One of the hand-written Learners the search starts from.
_Avoid_: seed (alone; means a random-number seed)

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
A Cascade's weighted Bradley-Terry rating from blind pairwise judgments by unrelated Cascades; it breaks fitness ties and chooses crossover partners, and never replaces the solved rate.
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
Moving the two best finalist Cascades of a Run into `candidates/cascades/` through a human-reviewed draft pull request.
_Avoid_: deploy, release, merge (the human merges, not the Run)

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
