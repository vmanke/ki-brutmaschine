# Spec: LLMs use CNNs (CNN Worker, Architect, CNN tool)

Status: needs-triage (design of 2026-10-31, updated the same day after PR #33; nothing of
it is built). The owner approves this design before any ticket starts. Until then every
ticket stays at needs-triage.

Basis: origin/main at f8d3478 (PR #33). Every file:line below points to that commit.
Main holds cnn-basismodelle ticket 03 and a first part of ticket 04 (beaca65). PR #32
brought the CNN tests up to that part (3360436). PR #33 added the Ollama adapter
`backend-ollama` (section 8, 10.3). Facts about the unfinished CNN work (cnn-basismodelle
tickets 04 to 07) come from those tickets, not from its worktree, because that code is
still changing.

Glossary: CONTEXT.md, subsection "LLMs using CNNs (planned)". No ADR file is written now.
Section 13 holds the text that becomes an ADR when the first build pull request merges.

## 1. Goal

The breeding machine shall let language models use convolutional networks (CNNs). There
are three ways, and the owner wants all three:

- **F1, CNN Worker.** A CNN is a Worker of a Cascade. It answers a grid task first.
  Language-model Workers in later Stages see its answer on the residual path and may
  improve it. Breeding decides by itself whether a CNN Stage stays.
- **F2, Architect.** A language model writes new CNN genomes (Blueprints), next to random
  mutation.
- **F3, CNN tool.** A language-model Worker asks a CNN for an output grid while it
  answers, as often as its Tool budget allows.

All three share one groundwork. Every call to a model that is not simulated becomes a
Call record, so a Run with real models or CNNs stays reproducible.

The aim is evidence, not only mechanism. Each feature comes with a positive control, a
negative control and a paired comparison. Results of simulated models are positive
controls only; they say nothing about real models.

## 2. The owner's decision of 2026-10-31

The owner asked (translated): "The breeding machine shall let LLMs use CNNs." Asked how,
the owner chose all three ways, in this order, each as its own pull request:

1. F1, CNN as Worker.
2. F2, LLM designs CNNs.
3. F3, CNN as tool.

The owner merges every pull request himself (code-owner gate, `.github/CODEOWNERS:12-13`).
Nothing in this plan enables auto-merge or bypasses review.

## 3. Decisions

The numbers are global. They follow Q74, the last decision of step 2
(`.scratch/universum-erfinden/spec.md:67`). The first draft numbered them Q1 to Q41, in the
same order.

| # | Decision | Reason |
|---|---|---|
| Q75 | F1, then F2, then F3. Each feature is one code pull request with its tests and its container evidence. The tickets of F2 wait for the merge of F1, and those of F3 for the merge of F2. Host runbook, ARC Domain, pre-registrations and glossary updates are separate small pull requests. Every pull request is a draft that the owner merges. | The owner's order. Code-owner gate (ADR 0004:47-52). |
| Q76 | (updated 2026-10-31 after PR #33) The Call record ships first, as its own small pull request (O4). It builds on the record of PR #33. | It is unblocked. F1, F2 and F3 need it. Runs with the models of #33 gain from it at once: a failed call becomes a fault, not a committed lost call, and a replay needs no Ollama (4.1). |
| Q77 | A CNN Worker is an ordinary Worker. Its model tag is `cnn:` plus the 16-hex `Genome::id` of a CNN genome. It gets no new Worker field. | Allow-list, `ModelPool`, ModelSwitch and crossover then work unchanged (`src/cascade/genome.rs:273-280`; `src/cascade/arena.rs:45-77`; `src/cascade/variation.rs:692-704`; `src/genome.rs:325-338`). Cascade ids stay. |
| Q78 | A CNN Worker learns by Per-task training on the worked examples of its prompt, from the genome's own seed. F1 uses no pretrained weights. | Every puzzle has its own hidden rule (`src/cascade/puzzle.rs:108-127`). A task's input is one grid (`src/task.rs:258-278`), so a pretrained CNN cannot read a new task's examples. |
| Q79 | Per-task training reads no clock and no stop signal. `Genome::validate` bounds its work before it starts, under `Limits::default()`: at most 20 000 parameters, 2e10 training FLOPs and 20 s of estimated time. A task trains for e_task epochs: the genome's `epochs`, or the largest smaller count that passes `Genome::validate` on that task. The plan gives e_task before training starts. | Equal bits need that no clock stops a training (ADR 0007:123-135). Today's loop reads the clock (`src/train.rs:105, 212-216, 345`). A valid genome never reaches the FLOP stop (`src/genome.rs:284-320`). With one fixed `epochs`, the largest task would cap every task. At 30 × 30 with 10 examples, a stem and two 3 × 3 stages of 16 channels allow about 50 epochs by the time model (`src/cnn/shape.rs:359-377`). Each epoch is then one step, since 10 examples are fewer than the smallest batch (`src/genome.rs:24`; `src/train.rs:155`). |
| Q80 | (updated 2026-10-31 after PR #33) The protocol part is pure and lives in `src/cascade/predictor.rs`: a `GridPredictor` trait and a `PredictorWorker` backend. The CNN part lives in `src/cnn/worker.rs`. | Tests can drive the arena with simple predictors before the CNN parts exist. The seam is model-agnostic, so a later model kind, such as the chemical networks the owner plans, can implement `GridPredictor` too. `src/cascade/` stays free of candle, files and clock (`src/cascade/mod.rs:4`; `tests/cascade.rs:14-35`). |
| Q81 | A CNN Worker ignores its sections, its Traits, the request seed and earlier answers. It reports 0 tokens. Its compute is charged per call (Q93). | Its answer then depends only on the genome and the task. One training per task serves every Cascade, and a replay is exact. |
| Q82 | It refuses (`StopReason::Refusal`) when an example changes the grid size, when the prompt breaks the protocol or its limits (30 × 30, 10 examples), when the genome does not fit the task, or when training diverges. An internal failure is a backend error. | A refusing Stage adds nothing to the residual path (`src/cascade/arena.rs:254-256, 279-289`). The shape rule is the Stage-1 cell learner's (`src/learner/arc_cells.rs:127-135`). |
| Q83 | As a judge it predicts the test output and picks the candidate with more cells equal to its prediction. A candidate of another shape has no equal cell. A coin from the request seed breaks a tie. The coin picks a position, A or B, as SimModel's coin does. It refuses when it cannot predict. | Deterministic. The same tie rule as SimModel's judge (`src/cascade/sim.rs:296-326`). A missing verdict on a scored pair counts as wrong (`src/cascade/arena.rs:1592-1596`). |
| Q84 | The grids of one task are padded to its largest height and width, over the examples and the test input. The input has 10 colour channels and one inside channel. Padded cells carry no loss. The answer is the best colour per inside cell, ties to the lower colour, cropped to the test input. | One task spec per task. Padded cells do not dilute the loss. |
| Q85 | One memo per CNN tag and process keeps the trained CNN per (examples, padded size). A request that misses trains its own copy and never waits for another request. Equal bits make every copy the same. The first copy finished is kept, and a counter records the extra copies. The memo is never written to disk. | Stage 0 prompts are equal for every Cascade on a task (`src/cascade/arena.rs:248`). The memo serves the Worker, the judge and the tool from one training. A wait could deadlock. candle runs its matrix products as nested jobs on the caller's rayon pool (`candle-core/src/cpu_backend/mod.rs:1482-1487` at the pinned commit). A rayon thread that waits for such a job runs other jobs meanwhile, and one of them may need the key it trains. |
| Q86 | All arms start from the same seed Cascades, none with a CNN Worker. Arms differ only in their allow-list or their tool. Breeding brings CNN Workers in through ModelSwitch and stage crossover. The Mutator could too, but counted runs have no LLM rounds (`llm_every` 0). | The owner asked that breeding find out by itself. StageAdd never inserts at Stage 0 (`src/cascade/variation.rs:807`); ModelSwitch and stage crossover can (`src/cascade/variation.rs:443-484, 692-704`). A Mutator answer written by someone who knows the arms could steer them, because every request lists the tags (`src/cascade/variation.rs:1083-1095`). Step 1's counted runs had no LLM rounds either (`docs/PREREG_STEP1_CIRCUITS.md:201`). |
| Q87 | The random Trait, section, WorkerAdd and StageAdd operators skip CNN Workers. ModelSwitch, WorkerDrop, StageDrop and both crossovers treat them like any Worker. Without a `cnn:` tag the slots, and so the random draws, stay the same. | Editing or cloning a CNN Worker cannot change its answer; such children would waste evaluations. The golden runs stay byte for byte (`tests/universe.rs:182-236`). |
| Q88 | A new admission rule refuses two Workers of one `cnn:` tag in one Stage, for every new Cascade, Mutator answers included. | Both would give the same answer, which the Stage drops as a duplicate (`src/cascade/arena.rs:258-262`). |
| Q89 | The first allow-list entry must be a language model. Circuit runs refuse `cnn:` tags. | Off-list seed Workers take the first entry (`src/cascade/genome.rs:620-631`). A CNN cannot read circuit prompts. |
| Q90 | A run with a `cnn:` tag or a CNN tool uses a Mutator brief with one fixed paragraph per feature. `GRID_BRIEF` stays byte for byte. | Recorded Mutator keys stay valid (`src/cascade/variation.rs:357-365`; `tests/universe.rs:207-223`). |
| Q91 | `brut-breed --cnn-models <dir>` (default `candidates/models/`) loads `<dir>/<id>.json` for each `cnn:` tag. The file is a promotion file of cnn-basismodelle 07 or, for tests, a bare genome. This design asks cnn-basismodelle 07 for two strict entry points: one parses a bare genome, the other a promotion file, whose genome part goes through the first. brut-breed refuses a missing file, unknown fields, an MLP genome and an id that differs from the tag. The directory may change on resume. | The tag fixes the content; the path stays outside the experiment. A promotion file holds provenance and scores (`.scratch/cnn-basismodelle/issues/07-promotion-of-winners.md:19-24`), so one parser cannot read both forms. One strict genome parser serves Workers, tools and Blueprints. |
| Q92 | At load, the genome must pass the shape check on the smallest and on the largest padded task the Domain can draw, and e_task must reach `MIN_EPOCHS` on the largest. A new pure Domain method, with a default, gives the smallest and largest padded size and the most examples. Each call checks the actual task again. | The run fails at the start, not in the middle. Validity is not monotone in grid size, because kernel spans must fit (`src/cnn/shape.rs:804-815`). The Domain trait knows no task sizes today (`src/cascade/domain.rs:38-79`). |
| Q93 | The fitness cost per call of a `cnn:` tag is the main model's cost per call times (mean wall time of a call to that tag ÷ mean wall time of a call to the main model). It comes from the pilot and is fixed before the counted runs. In the container it is the cost of `sim:small`, 0.0005. These are fitness units. | Costs come from measured wall time (`.scratch/kaskaden-zucht/issues/06-model-mix-experiments.md:6-8`). Cost enters fitness (`src/cascade/arena.rs:1983-1989`). |
| Q94 | A Cascade with a CNN Worker may be promoted only if its genome is already merged in `candidates/models/`. The gate adds the merged `cnn:` tags to the pool allow-list. It reads them from the base revision, never from the pull request. | No unreviewed model enters the pool (ADR 0004:47-52; `src/hitl.rs:150-158`). CI reads the pull request's own `candidates/` as `pr/` (`.github/workflows/scope-fence.yml:47-54`). One pull request could otherwise add a genome and a Cascade that uses it. |
| Q95 | Positive control: a new Domain, grid-mix. Half of its puzzles come from the generator, half follow a Neighbour rule that no SimModel hypothesis fits. Negative control: the Silent CNN, which refuses every task. | The machinery must keep a CNN that helps and drop one that does not. On plain grid puzzles a CNN overlaps with SimModel's colour family (`src/cascade/sim.rs:112-118`). |
| Q96 | The contribution of a CNN is measured by arm comparisons and by Knock-outs: the same Cascade scored again with its CNN Workers or its CNN tool silenced, on the same tasks with the same call seeds. A silenced tag is served by the bare silent predictor, never through the Call record. Every other tag still reads the Run's Call record, so an unchanged request gets its stored reply. | Silencing keeps the Stages and the seeds, so the paired difference is the CNN's. A silent Stage adds nothing (`src/cascade/arena.rs:279-283`). A silenced Stage-0 CNN sends the same request as when bred, and the key names the tag, not the backend (4.1). A stored reply would undo the silencing. |
| Q97 | Claims about ARC-AGI-1 evaluation wait for the sealed endpoint of kaskaden-zucht 02. Until then those numbers are descriptive. A counted claim on ARC-AGI-1 training tasks is about their select panels only. | The select panel is descriptive for the sealed endpoint (`src/cascade/arena.rs:772-775`; ADR 0005:75-80). |
| Q98 | Host search runs on ARC use ARC-AGI-1 training tasks only. A fixed rule over the SHA-256 of each task id splits them into search and select pools. This Domain is its own ticket and pull request. | Contamination rules (`src/arc_data.rs:196-251`). kaskaden-zucht 02 can reuse it (issue 02:9-10). |
| Q99 | (updated 2026-10-31 after PR #33) Call record: every call to a tag that is not simulated is stored with its full request and its full reply or error, under the `call_key` of its tag and its request, as one line of `<out>/llm/worker-calls.jsonl`, as soon as it returns. This is the record of PR #33, moved into the core and extended from the replies of `ollama:` tags to `cnn:` tags and errors. A resumed Run reuses stored replies and retries stored errors. A replayed Run serves these calls from the records alone. A missing record, a failed call of a real language model or a failed record write is a fault: the process ends with exit code 1 before it commits anything of that generation. Simulated tags are not recorded. | ADR 0002:3; contract I7 (`docs/STAGE2_BACKEND_CONTRACT.md:69-73`). #33 files the answered calls of `ollama:` tags (`backend-ollama/src/lib.rs:184-290`; `backend-ollama/src/main.rs:80`). Besides them, only Mutator exchanges are stored (`src/cascade/arena.rs:710-717, 1722-1729`). One file and one key keep the records of #33 readable. A preempted VM keeps the calls already made. Simulated runs keep their files. The arena turns every backend error into a lost call (`src/cascade/arena.rs:143-150, 183-185`), so only the binary can stop on a fault. |
| Q100 | F2 lives in the MLP mode (`brut run`) as Architect rounds. A CNN genome does not become a Cascade gene. | "Instead of only random mutation" means the `RandomMutator` of `src/evolve.rs:403`. A genome is valid only for one task spec (`src/genome.rs:229`). |
| Q101 | Architect answers follow the Mutator pattern: request key, answer files, pause with exit code 3, resume. The Mutator types move to `src/backend.rs`, and `arena.rs` re-exports them. Every exchange is a record of the lineage archive. | A proven path (`src/cascade/arena.rs:436-499`; `src/bin/brut-breed.rs:574-598`). The Learner proposer stores no answers (`src/learner/llm_proposer.rs:46-58`). |
| Q102 | Blueprint check, in order: take the first balanced JSON object; refuse unknown fields at every level; require a CNN genome; the run sets the seed gene; `Genome::validate`; refuse duplicates. Nothing is repaired. A refused Blueprint is logged, and a random child takes its place. | An honest acceptance rate. `Genome` accepts unknown top-level keys (`src/genome.rs:75`). `parse_llm_cascade` repairs nothing either (`src/cascade/variation.rs:591-613`). |
| Q103 | Architect requests carry the task in words, the limits in words and the parents as JSON with their public validation scores. No data and no hidden score. | ADR 0004:37-46. Requests are built before the hidden phase (`src/evolve.rs:189-211`). |
| Q104 | (updated 2026-10-31 after PR #33) The Architect is reached only through files. In the container a Claude session writes them. On the host a file bridge asks one local model through `OllamaBackend` of PR #33. #33 has no such bridge, so ticket 14 builds it in `backend-ollama`. An optional side file names the model that answered. | The core stays free of network (ADR 0001:3; ADR 0005:30-38). The slot `EXTRA_BACKENDS` of #33 exists in `brut-breed` only, not in `brut run` (`src/bin/brut-breed.rs:920-927`). A replayed answer says only "answer-book" today (`src/cascade/arena.rs:487`). |
| Q105 | F3: a new Trait, the Tool budget (`tool_calls`), from 0 to a ceiling of 3 in code. A run names its CNN tool in `CascadeLimits.tool` and may set a lower ceiling. Both are left out of the JSON when unused. The tool's tag is not on the Workers' allow-list. | Ceilings are code (ADR 0005:40-46). Old ids stay (`src/cascade/genome.rs:314-321`). A tool arm has no CNN Worker, so F1 and F3 are measured apart. |
| Q106 | The tool protocol is plain text inside the existing request. A reply that ends with the line `TOOL CNN` and a grid asks. The next prompt carries the turns so far after `END`, each reply line quoted, and `TOOL RESULT k` with a grid or `NONE`. No native tool calling. | The contract's request shape has no tool fields (`docs/STAGE2_BACKEND_CONTRACT.md:49-66`). `parse_prompt` ignores text after `END` (`src/cascade/puzzle.rs:662-670`). Quoting stops a reply from forging a result. |
| Q107 | The tool loop runs in the pure arena, for solve calls only. Every model turn and every tool call is one counted call at its tag's cost. One answer takes at most 1 + 2 × `tool_calls` calls, and the call bounds include this. Judges never use the tool. Only solve requests carry the tool text; judge requests stay as today. | `max_calls` holds only if every turn counts (`src/cascade/arena.rs:131-156, 198-209, 1096-1106`). |
| Q108 | A tool request must have the height and width of the test input; otherwise the result is `NONE` and the tool is not called. A request after the budget is spent ends the answer without a grid. | One training per task serves every tool call (Q85). The rule is simple to check. |
| Q109 | New random draws (ToolCallsShift and the `tool_calls` crossover) happen only in runs with a tool. | The golden runs stay byte for byte. |
| Q110 | CNNs train on the CPU through candle. Language models use the GPU through Ollama. | ADR 0007:12-23. GPU training would change that ADR and the bits. |
| Q111 | (updated 2026-10-31 after PR #33) On the host, Ollama runs natively, as PR #33 assumes, and listens on 127.0.0.1:11434 only. The runbook records its version and the model digests. Docker stays an option, for Ollama or for a later sandbox of mutated code. Where Docker is installed, `tools/vm-setup.sh` makes 127.0.0.1 its default published address. | #33 calls only a loopback address and follows no redirect (contract I3, `docs/STAGE2_BACKEND_CONTRACT.md:44-45`; `backend-ollama/src/lib.rs:46-73, 111-116`). Its README section assumes a native Ollama (`ollama pull`). A native server listens on 127.0.0.1:11434 unless `OLLAMA_HOST` says otherwise, and ufw lets only SSH in (`tools/vm-setup.sh:79-81`). Docker's published ports bypass ufw. The new default keeps a plain `-p 11434:11434` on 127.0.0.1, but not a port published with an explicit address (`tools/vm-setup.sh:82-103`). This plan runs no mutated code, so it needs no Docker sandbox (contract I4). ADR 0003:3 still names Ollama inside a Docker sandbox. |
| Q112 | Arm comparisons pair runs by seed, with the same models, seeds, generations, `max_calls` and wall-time cap, and decide by the sequential rule of step 1. A proof takes at least 12 pairs. Each run is scored by its best finalist: the top bred member by search fitness, fixed before any select answer is read. Knock-outs are reported per run with an exact one-sided sign test. | The owner chose the sequential rule for step 1 (ADR 0006:158-195). (3/2)^11 is below 100 and (3/2)^12 above it (ADR 0006:175). Compute matching (ADR 0005:66-73). A finalist picked by its own select answers would do worse when scored again, which would fake a CNN effect. |
| Q113 | No ADR file now. Section 13 becomes an ADR when the first build pull request merges, numbered after the ADRs of step 2. | ADR 0008 is reserved for the CNN families; step 2 takes 0009 and up. |
| Q114 | No money figures anywhere. Reports give measured times, memory and counts, and the VM hours the Nebius console shows. | Owner rule. |
| Q115 | The planned terms go into CONTEXT.md now, in their own subsection. Existing entries change only in small glossary pull requests after each feature merges. | Q53 (`.scratch/universum-erfinden/spec.md:46`). |

## 4. Design

### 4.1 Shared base: Call records

**Why.**
- PR #33 stores the answered Worker calls of `ollama:` tags, in
  `<out>/llm/worker-calls.jsonl` (contract I7; `backend-ollama/src/lib.rs:184-290`).
  Besides them, only Mutator exchanges are stored (`src/cascade/arena.rs:710-717,
  1722-1729`).
- Grid `Born` events carry no answers (`src/cascade/domain.rs:214-216`).
- Real models are not bit-reproducible (contract I7, `docs/STAGE2_BACKEND_CONTRACT.md:69-73`).
- CNN bits differ between CPUs (ADR 0007:111-114).
- The host can be stopped at any time.

**What is recorded.**
- Every call to a tag that is not simulated. #33 records the `ollama:` tags of
  `brut-breed-ollama`. Ticket 01 adds the `cnn:` tags: CNN Workers and the CNN tool.
- Tags that start with `sim:` are not recorded. Simulated runs therefore write no record,
  and their directories and golden fixtures stay as they are.

**The store.** It is the store of #33, so its records stay readable.
- One line per call in `<out>/llm/worker-calls.jsonl` (`backend-ollama/src/main.rs:80`).
  The key is `call_key` of #33: the tag, `#` and `request_key`
  (`backend-ollama/src/lib.rs:203-206`; `src/cascade/arena.rs:443-449`). So two tags never
  share a record.
- A record holds `key`, `tag`, the full `request`, and the full `response` or, new, the
  `error` (#33: `backend-ollama/src/lib.rs:184-191`). Unknown fields are refused.
- It is appended and synced as soon as the call returns, as #33 does
  (`backend-ollama/src/lib.rs:278-285`). A later line for the same key replaces an
  earlier one. A torn last line, which a hard stop may leave, is dropped on open. Any
  other bad line is refused. Under #33 one bad line fails the open, and the resume stops
  at `ModelPool::covers`.
- Reads are capped at 8 MiB per line, above the 1 MiB answer cap of #33 (contract I6,
  `backend-ollama/src/lib.rs:39-40`). A record whose key is not the `call_key` of its tag
  and request is refused. A hit also needs the stored tag and request to equal the live
  ones. #33 checks only the stored tag (`backend-ollama/src/lib.rs:223`).
- The module is `src/call_record.rs`, outside `src/cascade/`, because it does file I/O.
  `CallRecord`, `call_key` and `Recorded` move there from
  `backend-ollama/src/lib.rs:184-290`. The `cnn:` backends live in the core, and the core
  cannot depend on the adapter (contract I1). `backend-ollama` uses them from the core.

**Live mode.** `Recorded<B>` wraps any backend.
- A stored reply is returned without a call (#33 does this).
- A stored error is retried, and its new line replaces it.
- A missing record means a live call and a new record (#33 does this).
- A failed record write is a fault (below).
- Two requests with one key are never in flight at once; the second waits for the first.
  #33 would call the model twice. Keys are unique in practice, because every seed comes
  from the Cascade id, the task id and the Worker's place (`src/cascade/arena.rs:252,
  1462`), or from the generation's random stream (`src/cascade/arena.rs:1546`).
- brut-breed wraps every backend whose tag does not start with `sim:`: the `cnn:`
  backends and every backend that the slot `EXTRA_BACKENDS` builds
  (`src/bin/brut-breed.rs:920-951`). `brut-breed-ollama` then hands over the bare
  `OllamaBackend` (`backend-ollama/src/main.rs:50-61`).

**Resume.**
- A stopped or killed generation is computed again on resume
  (`src/cascade/arena.rs:1011-1016`).
- Its answered calls come back from the store. Only the calls that were in flight, and
  stored errors, are made again.
- A served call still counts against `max_calls` and pays its cost
  (`src/cascade/arena.rs:146-147`). A resumed Run therefore makes the same decisions.

**Faults.** The arena turns every backend error into a lost call
(`src/cascade/arena.rs:143-150, 183-185`). So the recorder keeps the first fault in a slot
that brut-breed reads. #33 has no such slot: a failed Ollama call becomes a lost call that
its generation commits.
- A fault is a missing record in a replay, a failed call of a real language model, or a
  failed record write.
- Errors of `cnn:` tags are no faults. They repeat on every try, so they stay lost calls,
  as today.
- brut-breed checks the slot before every log append and every checkpoint write, the
  reservation checkpoints of `step_with` and `finish_with` included.
- On a fault it exits with 1 and names the key. Nothing of that generation or select
  member is committed, so a resume repeats it and retries the failed call.

**Replay.**
- `brut-breed --replay <old> --out <new>` runs the old Run's settings as one straight run.
- Recorded tags are served from the old store only (`RecordsOnly`). Simulated tags run
  live. No genome file is loaded.
- So a replay needs neither Ollama nor a genome file, in `brut-breed` as in
  `brut-breed-ollama`. #33 replays only inside a resume of the same Run, and there a miss
  asks the model.
- Mutator answers come from the old log's `LlmExchange` events
  (`src/cascade/arena.rs:710-717`). `--llm-answers` is refused.
- A missing record or a missing Mutator answer is a fault: exit code 1, never 3.
- For a Run that went straight to its end, the new `events.jsonl` equals the old one byte
  for byte.
- For a Run with stops or kills, the `Born`, `Compared` and `LlmExchange` events are equal
  and in order. Call counts differ by the reservations of the resumes, as in the test
  `kill_switch_then_resume_equals_a_straight_run` (`tests/cascade.rs`).

**Verify and wall time.** #33 has neither.
- `--verify` also checks every record: its size, its form, and that its key is the
  `call_key` of its tag and its request.
- The recorder appends one line `{tag, seconds}` per live call to
  `<out>/call_times.jsonl`. No hash covers it, and `--verify` ignores it, as for
  `timings.jsonl` (`src/bin/brut-breed.rs:50-52`).

**Disk.** A full disk makes a record write fail, which is a fault, so nothing is
committed. The host runbook checks free space before each run (ticket 08), and M8
measures the bytes per record.

**Why no hash chain.** Records are keyed by content, so thread timing changes only their
order in the file, never a record. The hash-chained `events.jsonl` stays the anchor: a
replay against it shows every edited reply that changes what was bred.

### 4.2 F1: CNN as Worker

#### 4.2.1 The call

1. A run lists `cnn:<id>` on its allow-list, with its fitness cost per call (Q93). The
   allow-list comes from `--config` (`src/bin/brut-breed.rs:1156-1160`).
2. brut-breed loads the genome (Q91, Q92). It registers `PredictorWorker<CnnPredictor>`
   under the tag, wrapped in the Call record. It does so where the pool is built, before
   the slot `EXTRA_BACKENDS` is asked (`sim_pool`, `src/bin/brut-breed.rs:929-951`), so
   `brut-breed` and `brut-breed-ollama` register CNN Workers alike. Then
   `ModelPool::covers` passes (`src/cascade/arena.rs:70-76`).
3. The CNN Worker gets the same solve prompt as every Worker
   (`src/cascade/arena.rs:247-253`). It reads it with `parse_prompt`
   (`src/cascade/puzzle.rs:614-671`), as SimModel does (`src/cascade/sim.rs:436`).
4. It trains (4.2.2), predicts the test output and replies with `render_answer`
   (`src/cascade/puzzle.rs:445-452`). The stop reason is `Stop`, both token counts are 0,
   and the backend name is the tag.
5. A later Stage sees the grid as `PREVIOUS ANSWER k` (`src/cascade/puzzle.rs:482-495`).
   A residual Worker may keep it or fix it (`src/cascade/variation.rs:211-231`).
6. The final answer is the last chosen answer (`src/cascade/arena.rs:286-289`). If a later
   Stage gives nothing usable, the CNN's grid stays.
7. For a judge prompt it predicts the test output and replies with `render_verdict`
   (`src/cascade/puzzle.rs:454-460`), or refuses (Q83).
8. A prompt it cannot read gets a refusal with the text "I cannot read this task."
   (`src/cascade/sim.rs:36-37`).

#### 4.2.2 Per-task training

- **Training set.** Exactly the worked example pairs of the prompt. The test output is
  never in a prompt (`src/cascade/puzzle.rs:470-480`).
- **Refusals before any tensor.** An example whose output size differs from its input;
  more than 10 examples (`src/arc_data.rs:52`); a grid that `is_valid_grid` refuses
  (`src/cascade/puzzle.rs:66-78`); a genome that fails `Genome::validate` for this task
  even at `MIN_EPOCHS`.
- **Epochs.** e_task is the genome's `epochs`, or the largest smaller count that passes
  `Genome::validate` on this task. The plan gives it before training, so no clock decides
  it and the bits stay equal. M3 reports e_task per task.
- **Encoding.** Pad to the task's largest height and width. Use 10 one-hot colour channels
  and one inside channel. Each inside cell carries its output colour as its label; padded
  cells carry no loss.
- **Task spec.** The grid output of cnn-basismodelle 04 (`OutputShape::Grid`,
  `src/task.rs:178-194`) and the per-cell classification objective of cnn-basismodelle 06,
  with `n_train` equal to the number of examples.
- **Training.** A crate-private `train::train_counted` runs the existing loop
  (`src/train.rs:145-241`) without the clock and without the stop signal. The FLOP limit
  stays. `TrainStats.wall_seconds` is 0 on this path.
- **Limits.** `Limits::default()` (`src/config.rs:54-67`), as Q79 says
  (`src/genome.rs:311-320`; `src/cnn/shape.rs:459-463`).
- **Prediction.** One row per evaluator call, a fixed chunk rule for this path
  (`src/score.rs:48-69`). Outputs come from `CnnModel::evaluator` and `Evaluator::outputs`
  (`src/cnn/model.rs:181-198, 244-263`). Q84 turns them into a grid.
- **Divergence.** A non-finite loss or value gives a refusal.
- **Memo.** As Q85 says. A counter of trainings, and one of extra copies, lets the binary
  log trainings per generation.
- **Kill switch.** A training in flight always finishes. A STOP can wait up to one
  training per thread, up to 20 s by the time model, not "about a second" (CONTEXT.md,
  "Kill switch"). We report this. Stopping a training could change an answer that is
  then committed.
- **Dependencies.** `Head::PerCell` comes with cnn-basismodelle 04. Its work in progress
  on main already builds the head (`src/cnn/shape.rs:1138`); ticket 04 waits until 04 has
  merged. The per-cell classification objective is cnn-basismodelle 06; today the loss
  takes one label per sample (`src/cnn/train.rs:115-137`). If 06 brings no mask for padded
  cells, ticket 04 adds it in `src/cnn/` after 06 merges.

#### 4.2.3 How breeding finds out by itself

- **Arms.** L: language models only. C: L plus one `cnn:` tag. Z: L plus the Silent CNN.
  The letters avoid S and N, which step 2 uses. All start from the same seed Cascades
  (`src/cascade/genome.rs:532-603`).
- **How a CNN Worker enters.** ModelSwitch, 7 % of mutations
  (`src/cascade/variation.rs:284-301`), moves a Worker to another allowed tag, the `cnn:`
  tag among them. Stage crossover may put a CNN prefix before the Stages of another
  parent (`src/cascade/variation.rs:443-484`). The Mutator sees the tag in its limits and
  its brief, but counted runs have no LLM rounds (Q86).
- **How it leaves.** ModelSwitch, WorkerDrop, StageDrop and crossover.
- **What decides.** Fitness alone: solved rate, judging skill and cost
  (`src/cascade/arena.rs:1983-1989`).
- **Operators.** Q87 and Q88.
- **What it looks like in the log.** The model-tag entropy already counts `cnn:` tags
  (`src/cascade/observe.rs:73-74, 324`).

#### 4.2.4 Positive and negative controls

- **grid-mix.** A new Domain, `DomainConfig::GridMix`, chosen with `--domain grid-mix`.
  Task i of a panel is a generator puzzle when i is even and a Neighbour rule puzzle when
  i is odd. A longer panel then extends a shorter one (`src/cascade/domain.rs:46`).
- **Neighbour rule.** A new pure module `src/cascade/neighbour.rs` draws rules that set a
  cell's colour from its own colour and those of its four neighbours. A puzzle is well
  posed only if:
  - every grid keeps its size;
  - every example changes at least one cell and keeps at least one;
  - the test output differs from the test input;
  - every colour of the test input occurs in the examples;
  - no hypothesis that SimModel can form (`src/cascade/sim.rs:109-142`) fits any single
    example: Identity and every step of `MOVES` (`src/cascade/puzzle.rs:51-59`), each with
    and without a colour map fitted to that one example;
  - no step of `MOVES` applied to the test input gives the test output, so SimModel's
    guess fails too (`src/cascade/sim.rs:293`).
- **Sizes.** Neighbour grids have 3 to 6 rows and columns, and a puzzle has `TRAIN_PAIRS`
  examples, as in the generator (`src/cascade/puzzle.rs:34, 343-359`).
- **Prompts.** The grid renderers are shared over (examples, test input), so grid prompts
  keep their bytes.
- **Calibration.** A reference genome fixture (`tests/fixtures/cnn/<id>.json`) must solve
  most Neighbour puzzles. The bound is set 0.1 below the first measurement and pinned. If
  calibration fails, the rule is redesigned before the evidence runs.
- **Silent CNN.** A `PredictorWorker` whose predictor refuses every task, under the
  reserved tag `cnn:0000000000000000`. brut-breed registers it without a genome file.
- **What we expect from SimModel.** On grid-mix, [CNN; SimModel with the right families]
  solves both halves: SimModel applies a fitting hypothesis, and where none fits it
  passes the CNN's answer through (`src/cascade/sim.rs:289-291`). Arm L can solve at most
  the generator half.

#### 4.2.5 Data model changes

- New `src/cascade/predictor.rs` (pure): `GridPredictor`, `Prediction` (a grid or a
  refusal with a reason), `PredictorWorker<P>`, `is_cnn_tag` (`cnn:` plus 16 lowercase
  hex), `SilentPredictor`, `SILENT_CNN_TAG`.
- New `src/cascade/neighbour.rs` (pure). `src/cascade/domain.rs`: `DomainConfig::GridMix`,
  the grid-mix Domain and gene pool, the admission rule of Q88, and the size method of
  Q92.
- `src/cascade/puzzle.rs`: renderers shared over (examples, test input).
- `src/cascade/variation.rs`: the skip rule of Q87; the static `GRID_CNN_BRIEF`; the pure
  `without_cnn(&BreedConfig) -> BreedConfig`, which makes arm L from arm C.
- `src/cascade/arena.rs`: `BreedConfig::check` gains the rules of Q89.
- New `src/cnn/worker.rs`: `CnnPredictor`. `src/train.rs`: `train_counted`, with an
  optional clock in `run_epochs`.
- `src/bin/brut-breed.rs`: `--cnn-models`, `--domain grid-mix`, `--replay`, `--knock-out`,
  the Call record and its fault slot, `call_times.jsonl`, `cpu.jsonl` (section 6).
  `RESUME_KEYS` gains `cnn-models` and `allow-other-cpu` (`src/bin/brut-breed.rs:92-94`).
  `--verify` and `--promote` register, for each `cnn:` tag, a backend that fails if
  called: a finished run's rebuild makes no call (`src/bin/brut-breed.rs:827-847`), and
  its genome file may sit outside `candidates/models/`.
- `src/hitl.rs` and `brut-ingest --cascades`: merged `cnn:` tags join the pool allow-list
  (Q94).
- Unchanged: `Worker`, `Traits`, `Cascade` JSON, the default `CascadeLimits`,
  `pool_limits`, `GRID_BRIEF` (`src/cascade/genome.rs:144-238, 240-321`).

#### 4.2.6 How F1 meets the Domain, the Mutator, the vote and peer judging

- **Domain.** The trait gains one pure method with a default: the sizes of its tasks
  (Q92). Nothing else changes (`src/cascade/domain.rs:38-79`). The Domain still never sees
  a Cascade (ADR 0006:11-16). The CNN reads the same protocol
  (`src/cascade/puzzle.rs:10-15`). It never sees task ids (`src/cascade/domain.rs:49-51`)
  or the truth.
- **Mutator.** It sees the tag and its cost in "Limits of this run"
  (`src/cascade/variation.rs:1081-1118`) and the CNN paragraph (Q90). It may add, move or
  drop CNN Workers. It cannot change a genome: the tag names fixed content.
- **In-stage vote.** Every Worker judges every pair of distinct answers
  (`src/cascade/arena.rs:294-328`). A CNN judges by Q83.
- **Peer judging.** Only the first Worker of the last Stage judges
  (`src/cascade/arena.rs:1565`). If that is a CNN, Q83 applies.

#### 4.2.7 Limits on time and calls

- One call per CNN answer or judgment. The call bounds stay
  (`src/cascade/arena.rs:198-209`).
- The work per call is bounded before training (Q79).
- Grids at most 30 × 30, at most 10 examples.
- The load-time check of Q92.
- Run caps stay: `max_calls`, `--max-seconds` and the reservation
  (`src/cascade/arena.rs:1096-1106`).

### 4.3 F2: LLM designs CNNs (the Architect)

#### 4.3.1 Mechanism

1. `Config` gets `architect: Option<ArchitectConfig { every, children, pause }>`. It is
   left out of the JSON when absent, like `base_genomes` (`src/config.rs:117-122`).
   `resume_conflicts` covers it as it is (`src/config.rs:175-200`).
2. Architect rounds need a run whose model kinds include CNNs (the CLI of
   cnn-basismodelle 05).
3. Every `every`-th generation is an Architect round. It plans up to `children` Blueprints,
   at most `children_per_generation`.
4. Parents are drawn as now (`src/evolve.rs:413`), from the generation's random stream
   (`src/evolve.rs:271-275`). Every second request merges two parents when a partner
   exists, as for Cascades (`src/cascade/arena.rs:1640`). The others change one parent.
5. A request (new pure module `src/architect.rs`) holds:
   - a fixed system text;
   - the task in words: kind, input shape, objective, sample counts (`src/task.rs:258-278`);
   - the limits in words (`src/config.rs:32-52`);
   - the gene ranges and the CNN parts, generated from the code's constants and from what
     the shape check accepts, so the text follows cnn-basismodelle 03 to 05;
   - each parent as pretty JSON with its public validation score;
   - the task and the rules.
6. The settings are fixed like the Mutator's: 4096 tokens, temperature 0.7, no thinking
   (`src/cascade/variation.rs:303-313, 1042-1052`).
7. Answers come from `<run>/llm/answers/<key>.txt` or from `--llm-answers <dir>`. An
   optional `<key>.json` side file names the model that answered. It holds at most 4 KiB
   and exactly `{"model": "<tag>"}`, the tag matching `^[a-z0-9][a-z0-9:._-]{0,63}$`.
   Anything else is refused before the run changes a file.
8. A missing answer with `pause` on: `brut run` writes `<run>/llm/requests/<key>.json`,
   commits nothing of that generation, appends one `RunFinished` record with the reason
   "waiting for Architect answers" and exits with 3. Today `brut` knows exit codes 0, 1
   and 2 only (`src/main.rs:71`). A resume repeats the generation with the same stream
   and the same requests. With `pause` off, a missing answer gives a random child.
9. The Blueprint check of Q102. An accepted Blueprint becomes a child with the generator
   name `architect`. A refused one is logged, and a random child takes its place.
10. Every exchange is an `ArchitectExchange { generation, key, request, response,
    accepted, reason, child }` record in the hash-chained lineage archive
    (`src/archive.rs:76`).
11. Children train and score like all others, under the same limits
    (`src/genome.rs:229-323`).
12. Winners reach `candidates/models/` only through cnn-basismodelle 07 and a human merge.
    The provenance names the Architect and the request key (ADR 0004:47-52). A merged
    genome can then be a CNN Worker of F1 (Q91).

#### 4.3.2 Data model changes

- `src/backend.rs`: `Mutator`, `request_key`, `AnswerBook` and `LiveMutator` move here from
  `src/cascade/arena.rs:436-499`. `arena.rs` re-exports them, so
  `src/bin/brut-breed.rs:66-69`, `tests/universe.rs:23-26` and
  `backend-ollama/src/lib.rs:27` keep working.
- New `src/architect.rs` (pure): request building and the Blueprint check. The strict
  parse reuses the bare-genome entry point of cnn-basismodelle 07 (Q91). The purity scan
  of `tests/cascade.rs:14-35` covers `src/cascade/` only, so it is extended to this file.
- `src/config.rs`: `ArchitectConfig`.
- `src/evolve.rs`: `propose` (`src/evolve.rs:376-441`) returns the children, or the
  requests still waiting. The `RandomMutator` makes all other children as today.
- `src/archive.rs`: `RecordKind::ArchitectExchange`.
- `src/main.rs`: `brut run --llm-answers <dir>`, exit code 3, the usage text.
- Answer files are capped at 64 KiB, in both loops. The Cascade side reads them whole today
  (`src/bin/brut-breed.rs:1149`).

#### 4.3.3 How F2 meets the rest

- F2 meets no Domain, no in-stage vote and no peer judging.
- The Architect is a role apart from the Mutator, with its own brief. The Mutator's
  request keys stay pinned (`tests/universe.rs:207-223`).
- F2 reaches Cascades only through merged genomes that F1 loads.

#### 4.3.4 Limits

- `children` is at most `children_per_generation`; `every` is at least 1.
- One request per child, no retry.
- Architect answers cost no FLOPs. They are counted as `architect_answers`.
- Blueprints face the run's FLOP and time budgets like any child (`src/evolve.rs:449`).

### 4.4 F3: CNN as tool

#### 4.4.1 Mechanism

1. The run sets `CascadeLimits.tool = ToolSpec { model: "cnn:<id>", cost_per_call,
   max_tool_calls }`, left out of the JSON when absent. `max_tool_calls` lies in 1 to 3,
   the code ceiling `TOOL_CALLS_CEILING`.
2. The tool's tag is not on the Workers' allow-list. brut-breed registers its backend like
   a CNN Worker's (Q91). `ModelPool::covers` checks it too.
3. A Worker's Tool budget is the new Trait `tool_calls`, from 0 to the run's
   `max_tool_calls`, left out of the JSON at 0. `check_worker`
   (`src/cascade/genome.rs:362-405`) refuses `tool_calls` above 0 without a tool, above
   the run's cap, or on a CNN Worker.
4. For a Worker with `tool_calls = t > 0`, a new `solve_request_with` appends `## tool`,
   the fixed `TOOL_TEXT` and the line "You may ask the tool at most t times.", when the
   run has a tool. It appends them the way `## check` is appended
   (`src/cascade/arena.rs:115-119`). Only `run_cascade_in` uses it, for solve calls
   (`src/cascade/arena.rs:253`). `worker_request_with` stays as it is, because judge calls
   use it too (`src/cascade/arena.rs:149, 183`).
5. Turn 0 is today's call. If its reply is a tool request and turns remain, the arena:
   - builds the tool prompt (4.4.2);
   - calls the tool through the pool, with a request of its own: an empty system text,
     the Worker's generation limits and the seed `derive_seed(s, "tool/k")`, where s is
     the Worker's call seed;
   - appends the turn to the prompt;
   - calls the Worker again with the seed `derive_seed(s, "turn/k")`.
6. The loop ends with the first reply that is not a tool request. That reply is the
   answer. A tool request after the budget is spent ends the answer without a grid.
7. Judge calls never use the tool, and their requests carry no tool text.

#### 4.4.2 Protocol

- **Request.** The reply's last non-blank lines are `TOOL CNN` and a grid that
  `is_valid_grid` accepts (`src/cascade/puzzle.rs:66-78`).
- **Tool prompt.** A solve prompt with the task's examples and the requested grid as
  `TEST INPUT`. The Domain builds it from the task, never from the truth
  (`src/cascade/puzzle.rs:470-480`). The grid must have the test input's height and width
  (Q108).
- **Transcript.** After `END`, for each earlier turn k: the line `REPLY k`, every line of
  that visible reply prefixed with `| `, then `TOOL RESULT k` and the grid or `NONE`.
- **Reading it.** `parse_prompt` reads these blocks into `ParsedPrompt.tool_results`. Old
  readers ignore them (`src/cascade/puzzle.rs:662-670`).
- **Result.** The tool's reply is digits or `NONE`. No other text reaches the next turn.

#### 4.4.3 Simulated tool use (positive control)

- When the system prompt holds `TOOL_TEXT` and no hypothesis fits, SimModel asks the tool
  once with the test input.
- With a `TOOL RESULT` grid it answers with that grid. With `NONE` it goes on as today:
  pass-through, then a guess (`src/cascade/sim.rs:256-294`).
- Without `TOOL_TEXT` its replies stay byte for byte, as with `SELF_CHECK_TEXT`
  (`src/cascade/sim.rs:450-454`).
- `TOOL_TEXT` names no keyword of SimModel's families (`src/cascade/sim.rs:54-69`). For
  example, "fallback" would switch on gravity.

#### 4.4.4 Data model changes

- `src/cascade/genome.rs`: `Traits.tool_calls`, `TOOL_CALLS_CEILING = 3`,
  `CascadeLimits.tool: Option<ToolSpec>`, the checks.
- `src/cascade/puzzle.rs`: `TOOL_TEXT`, the request parser, the tool prompt, the transcript
  renderer and reader, `ParsedPrompt.tool_results`.
- `src/cascade/domain.rs`: tool methods with defaults that mean "no tool". Circuits keep
  the defaults and refuse `tool_calls` above 0.
- `src/cascade/arena.rs`: `solve_request_with`; the loop for solve calls; per-Worker call
  bounds; `Usage.tool_calls`, left out at 0.
- `src/cascade/variation.rs`: ToolCallsShift and the `tool_calls` crossover, drawn only
  with a tool (Q109); tool paragraphs in the briefs; one more line in `describe_limits`
  when a tool exists.
- `src/cascade/sim.rs`: 4.4.3.
- `src/bin/brut-breed.rs`: the tool backend and the silent tool.

#### 4.4.5 How F3 meets the rest

- **Domain.** It builds tool prompts from the task's examples only.
- **Mutator.** It sees the cap and the tool paragraph. Requests of runs without a tool
  keep their keys.
- **In-stage vote and peer judging.** Judges get no tool. They judge final answers as
  today (`src/cascade/arena.rs:171-196, 1502-1618`).
- **Recording.** Every model turn and every tool call is its own request, and so its own
  Call record. A replay rebuilds the whole loop. #33 already files each model turn of an
  `ollama:` Worker, as it files any Worker call. Ticket 01 adds the tool's `cnn:` calls and
  the replay from records alone.

#### 4.4.6 Limits

- One solve answer takes at most 1 + 2t calls, with t at most 3.
- The bound of a Stage becomes the sum over its Workers of (1 + 2 t) plus the judging
  calls. Without tools it is today's bound (`src/cascade/arena.rs:200-204`).
- Three Workers at t = 3 give 3 × 7 + 9 = 30 calls per puzzle and Stage, against 12 today.
  A full Cascade gives at most 120, against the 48 of ADR 0005:82-87.
- The reservation covers this worst case (`src/cascade/arena.rs:1096-1106`).
- Each tool call keeps the limits of 4.2.2.
- The transcript grows by at most one reply and one grid per Tool turn.

## 5. Data hygiene

1. **Training sees no truth.** CNN Workers and the CNN tool train only on the worked
   examples of their prompt. Prompts never hold the test output
   (`src/cascade/puzzle.rs:470-480`). A test replaces the truth and requires equal replies.
2. **No state between tasks.** The memo is keyed by one task's examples. It lives in
   memory, is never written, and starts empty in every process. The sealed evaluator builds
   a fresh predictor per process.
3. **The ARC Domain loads ARC-AGI-1 training entries only** (role `Train`, source ARC-AGI-1
   training). It refuses everything else, `External` entries included. Today the manifest
   lists 680 evaluation files and no training file (`data/manifest.example.json`). The
   evaluation loaders stay in `brut-eval`'s path (`src/arc_data.rs:150-197`).
4. **Two training tasks never enter a panel.** 40853293 and caa06a1f repeat pairs of the
   evaluation tasks 070dd51e and 50a16a69 (`src/arc_data.rs:249-251`).
5. **Forbidden sources stay out** of brut-breed, CNN training and tool use: ARC-AGI-2
   training, which holds 376 of the 400 ARC-AGI-1 evaluation tasks; ARC-GEN; ARC-AGI-2
   evaluation; ARC-TGI; BARC (`src/arc_data.rs:27-33, 196-241`).
6. **Search and select pools are disjoint.** A fixed rule over the SHA-256 of each task id
   splits them. A test proves that they are disjoint for every seed.
7. **Genomes come from training data only, and never from the select pool.** Genomes used
   as Workers or tools come from the reference fixture, from base models, or from
   `candidates/models/`. They are bred on generated tasks or on ARC-AGI-1 training tasks
   and their re-arc generations (`.scratch/cnn-basismodelle/issues/06-arc-grids.md:16-19`).
   A genome chosen on the rules of the select tasks would favour arm C there. So this
   design asks three things of cnn-basismodelle 06 and 07, as O2 and O3 do:
   - The split rule of rule 6 is one pure function, shared by ticket 09 and
     cnn-basismodelle 06. Select-pool tasks and their re-arc generations never enter
     genome breeding, validation, hidden scoring or promotion.
   - 06 also leaves out 40853293 and caa06a1f and their re-arc generations, as rule 4
     does for panels (S7, `src/arc_data.rs:243-251`).
   - Each promotion file records the split rule's hash and its training sources.
   An ARC run refuses a `cnn:` genome whose provenance is missing, whose split rule differs
   from the run's, or whose training sources fail `check_contamination` for A1Eval
   (`src/arc_data.rs:196-241`).
8. **No hidden score reaches a proposer.** The Mutator and the Architect see public scores
   only: search fitness (`src/cascade/variation.rs:1120-1127`) and validation scores. The
   Architect never sees a promotion file with its recorded hidden scores
   (`.scratch/cnn-basismodelle/issues/07-promotion-of-winners.md:19-24`).
9. **The tool sees one task:** its examples and the grid the Worker sends.
10. **Audit.** For every host run on ARC, every CNN training set read back from the Call
    records is hashed with `canonical_pair_hash` (`src/arc_data.rs:243-284`). The run must
    show no overlap with the pair hashes of both evaluation sets.
11. **Sealed records stay sealed.** At the sealed endpoint (kaskaden-zucht 02) a CNN Worker
    trains on each evaluation task's own examples. That is the task as posed, not a leak.
    The Call records of such runs stay with the evaluator, under a separate user with mode
    0700. They never feed a Mutator, an Architect or a search.
12. **Model contamination.** Open models may have seen ARC evaluation tasks online. This
    cannot be audited. Paired arms with the same models keep the difference fair. Absolute
    scores may be inflated, and every report says so.

## 6. Reproducibility

- **Simulated runs** stay deterministic and unchanged. In every pull request these pass
  unchanged: the golden tests of `tests/universe.rs:182-236`, `tests/mlp_golden.rs`, and
  the purity scan of `tests/cascade.rs:14-35`.
- **A CNN Worker on one machine** gives the same bits at any thread count
  (ADR 0007:97-121): Per-task training reads no clock (Q79), prediction uses one row per
  call, and ties go to the lower colour.
- **Other CPUs** may give other bits (ADR 0007:111-114). A run on another machine replays
  the Call records instead.
- **A resume on another CPU.** In a Run with `cnn:` tags, every process adds one line to
  `<out>/cpu.jsonl`: the CPU model and its FMA and AVX-512 flags. No hash covers it. A
  resume refuses a CPU that differs from the first line, unless `--allow-other-cpu` is
  given; its line then says so. Otherwise one Run could mix the CNN bits of two CPUs,
  since the memo starts empty in every process.
- **Host records outlive the VM.** Call-record tarballs go to the lasting store of O13.
  Before a host report merges, `--replay` on another machine, from the downloaded
  tarball, must give the run's `Born`, `Compared` and `LlmExchange` events. A counted run
  whose records are lost, or fail this replay, is void.
- **Real models** are not bit-reproducible (contract I7). Call records make their runs
  reproducible.
- **New defaults** are left out of the JSON. New random draws happen only when a feature
  is on.
- **Tests compare within one machine only.** Threads vary through child processes with
  `RAYON_NUM_THREADS` at 1 and 3 (ADR 0007:115-121).
- **`--verify`** still rebuilds a finished run from its checkpoint without a model call
  (ADR 0006:138-156).
- **F2.** A paused run, once answered and resumed, writes the same `Candidate` and
  `ArchitectExchange` records, in the same order, as a run that had all answers from the
  start, with timestamps masked. Each process adds its own `RunStarted` and `RunFinished`
  records (`src/evolve.rs:134-139, 352-357`), so a pause adds one of each. These and
  `SafetyEvent` records are compared on their own.
- **The host** runs the CNN determinism tests once, because bits differ between CPUs.

## 7. Security

- **Data, not code.** A genome is data. It is checked before any tensor is built. No
  trained weights are loaded, so no large model file enters brut-breed.
- **Strict files.** Genome files are read with a 1 MiB cap and a strict parse, at the top
  level too; inside the CNN part this already holds (`src/cnn/genome.rs:117`). The file's
  id must equal its hash (`src/genome.rs:325-338`).
- **Tags.** A tag is `cnn:` plus 16 lowercase hex characters, so it cannot name a path
  outside the models directory.
- **Prompts.** The prompt parser is strict and capped: digits 0 to 9, at most 30 × 30
  (`src/cascade/puzzle.rs:66-78, 531-549`).
- **The tool.** Its input is one strict grid. Its output is digits or `NONE`. Quoted
  replies cannot forge a result. The only tool is one CNN; there is no file, shell or
  network tool.
- **Clean module.** Only `src/cnn/ops.rs` names candle (ADR 0007:80-95). A source scan
  keeps `src/cnn/worker.rs` free of file, network, process and clock APIs.
- **Architect text is untrusted.** It is parsed strictly and validated before training.
  Nothing in it runs. A genome has no free-text field (`src/genome.rs:75-95`), so a
  Blueprint cannot carry an instruction into a later request.
- **No secrets in requests.** Requests carry no secret and no hidden score.
- **Capped files.** Answer files are capped at 64 KiB and keyed by the request hash.
  Architect side files are capped at 4 KiB and parsed strictly. Call records are capped at
  8 MiB per line, above the 1 MiB answer cap of #33 (4.1).
- **Network.** The core gains no network code (ADR 0001:3). Only `backend-ollama` opens a
  socket, and only to a loopback address (contract I3; `backend-ollama/src/lib.rs:46-73`).
  Ollama listens on 127.0.0.1 only (Q111).
- **No address and no key in git.** Host addresses live only in the uncommitted `.env`
  (`.gitignore`). The deploy key lives in `~/.ssh` on the VM, with mode 0600, outside the
  repository.
- **Review.** `candidates/` has no code owner (`.github/CODEOWNERS:12-13`), so the gate
  must resolve every `cnn:` tag (Q94). The human merge stays the gate. Nothing auto-merges
  (ADR 0004:47-52). The evaluator never changes from inside the loop (ADR 0005:57-64).
- **No owner login on the VM.** `tools/vm-setup.sh` installs `gh` from GitHub's own
  package source, logs the owner's GitHub account into the VM and gives git its token
  (`tools/vm-setup.sh:64-77, 134-142`). An agent there could then push any branch and merge
  as the owner. So after setup the owner logs the VM out of GitHub. A later run of the
  script asks for that login again, so the owner logs out again after it. The agent
  pushes only with the deploy key, and a repository ruleset limits that key to `runs/`
  (ticket 08).

## 8. Host (Nebius H200)

The owner's VM is a Nebius AI Cloud VM with 1 NVIDIA H200 GPU, 16 vCPU and 200 GiB RAM,
on Ubuntu 24.04 with NVIDIA drivers. It is preemptible. Language models use the GPU
through Ollama. Per-task training uses the CPU through candle.

What changes against kaskaden-zucht issue 03, which was written for a Hetzner GEX44:

| Point | Issue 03 (Hetzner) | This plan (Nebius) |
|---|---|---|
| Machine | GEX44 with 20 GB, later GEX131 (`.scratch/kaskaden-zucht/spec.md:25`) | The owner's H200 VM. Record its GPU memory as nvidia-smi shows it. |
| Renting | Blocks with a setup fee per order (issue 03:6-7) | Start and stop in the Nebius console. No block logic. |
| Stopping | Not covered | Only a stop in the Nebius console or through its API ends billing. A shutdown inside the VM does not. Every runbook path ends with a console stop and a look at the console. |
| Interruptions | None planned | Nebius may stop the VM at any time with a 60 s warning; disks survive. A systemd unit writes STOP on shutdown and waits up to 50 s. Checkpoints, reservations and Call records cover a hard stop (ADR 0005:48-55). How the warning arrives is open (O6). |
| Disks | Not covered | Runs, Call records and model files on the persistent disk. Check once that they survive a console stop. Check free space before each run (ticket 08). |
| Software | Docker and Ollama (issue 03:7) | Ollama runs natively, as PR #33 assumes (Q111). `tools/vm-setup.sh` installs neither Docker nor Ollama (`tools/vm-setup.sh:7-17`). A new `tools/host-llm-setup.sh` installs Ollama at a fixed version, checks that port 11434 listens on 127.0.0.1 only, and builds `brut-breed-ollama`. Docker is optional. Where it is installed, `tools/vm-setup.sh` sets its default published address to 127.0.0.1 (`tools/vm-setup.sh:82-103`). |
| Models | gpt-oss:20b and qwen3:8b (`.scratch/kaskaden-zucht/spec.md:38`) | The same, once `backend-ollama` serves thinking models (10.3). For now it allows only qwen2.5:7b, qwen2.5:14b, qwen2.5:32b and llama3.1:8b (`backend-ollama/src/lib.rs:12-14, 33-34`), and a pilot may start with them. Verify the tags, record the digests, and measure GPU memory with each model loaded (M2). |
| Context | Not covered | ARC prompts are long. Ollama may cut a prompt longer than its context setting. #33 sends no context setting, so the server's default applies (`backend-ollama/src/lib.rs:128-140`). A follow-up in `backend-ollama` must set the context and refuse prompts that do not fit (10.3). |
| Pilot caps | `--max-seconds 7200 --max-calls 20000` (issue 03:8) | The same for the first pilot. `--max-seconds` stays below the time to the next planned stop. |
| Results | `runs/` branch with a deploy key that writes only there (issue 03:8-9) | The same. A repository ruleset makes "only there" true, since a deploy key with write access reaches every branch otherwise. Plus a Call-record tarball whose SHA-256 goes into the run manifest. The tarball stays out of git, in the lasting store of O13. |
| Access | SSH | Claude Code over Remote Control. It runs in the system service `brut-remote-control`, in `tmux -L brut` (`tools/vm-setup.sh:164-220`). The owner starts the service once after the Claude login. From then on it starts after every boot and restarts by itself. SSH for the owner. ufw lets only SSH in (`tools/vm-setup.sh:79-81`). After setup the VM holds no GitHub login of the owner (section 7). |
| Prices | Check hetzner.com (issue 03:10-11) | No price in any document. Spending is read from the console. |
| CNN bits | Not relevant | Run the CNN determinism tests once on the VM. |
| Address | Not covered | Never in the repository. It lives only in the uncommitted `.env`. |

**Who does what.**
- The owner starts and stops the VM, makes the deploy key and its ruleset, logs the VM out
  of GitHub after setup, approves Remote Control and checks billing in the console.
- Claude, over Remote Control, sets up, builds, tests, runs, measures, writes reports and
  pushes to `runs/` with the deploy key. It tells the owner when to stop the VM. It never
  reports billing as stopped.

**How brut-breed reaches real models.** brut-breed is a binary of the core package
(`Cargo.toml`), and the core gains no HTTP client (contract I1,
`docs/STAGE2_BACKEND_CONTRACT.md:37-41`). PR #33 built the seam, which answers O5:
- `src/bin/brut-breed.rs` has a slot, `EXTRA_BACKENDS`, that holds one `BackendFactory`
  (`src/bin/brut-breed.rs:920-927`).
- For every allow-listed tag that is not simulated, the pool asks that factory for a
  backend (`sim_pool`, `src/bin/brut-breed.rs:929-951`). A tag without one fails
  `ModelPool::covers`.
- The program `brut-breed-ollama` of `backend-ollama` includes that file as a module,
  fills the slot for `ollama:` tags and calls its `main`
  (`backend-ollama/src/main.rs:16-17, 65-85`).
- So every Cascade run on the host uses `brut-breed-ollama`. CNN tags need no slot:
  brut-breed registers them itself, in both programs (4.2.1).

## 9. Measurement

No result is given in money. A run's cost is its measured hours, calls and memory, plus
what the console shows.

### 9.1 Container controls (SimModel; they say nothing about real models)

| Run | Domain | Arms | Expectation |
|---|---|---|---|
| F1-pos | grid-mix | C vs L | C solves more select puzzles. Finalists keep CNN Stages. Knock-outs are positive. |
| F1-neg | grid-mix | Z vs L | No difference. The Silent CNN's share among finalists is not above its share among all children born. |
| F1-grid | grid | C vs L | Descriptive. The CNN overlaps with SimModel's colour family. |
| F2-scripted | generated image task (MLP mode) | scripted Architect vs Control | A scripted Architect that returns a known better genome is taken up. A garbage Architect changes nothing but records. |
| F2-session | the same | Claude-answered Architect vs Control | Descriptive. |
| F3-pos | grid-mix | T vs L | T solves more select puzzles. Knock-outs of the tool are positive. |
| F3-neg | grid-mix | U (silent tool) vs L | No difference. |
| F3-vs-C | grid-mix | T vs C | Descriptive. |

Each control runs one pilot seed first. Its pass criteria and its counted seeds are then
committed and pushed to the draft pull request, before the counted seeds run. The report
cites that commit. The F1 and F3 controls have no LLM rounds (`llm_every` 0, Q86).

### 9.2 What to measure on the host

- **M1.** Wall seconds per call and tag: median, 90th percentile and maximum
  (`call_times.jsonl`), with CNN trainings apart from memo hits.
- **M2.** GPU memory per loaded model and for the run's model mix, and GPU use. A script
  samples `nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv`
  every 10 s into a file outside the run. CPU load is sampled the same way.
- **M3.** CNN training time against the time model's estimate, and e_task per task. The
  model was calibrated on another machine (ADR 0007:130-135).
- **M4.** Search and select solved counts per arm and seed; the knock-out counts b and c.
- **M5.** Tool requests per answer, the share that parse, and the share of final answers
  equal to a tool result.
- **M6.** From the Call record: the share of tasks whose final answer is the CNN's; how
  often a later Stage turns a right CNN answer wrong, or a wrong one right; `Length`
  stops; the share of tasks the CNN refuses, by reason.
- **M7.** For F2: the share of Blueprints accepted, and the refusal reasons.
- **M8.** Bytes per Call record and per run.
- **M9.** Preemptions, stops on a fault (4.1), calls served from the store after a
  resume, calls made again, and wall time lost.
- **M10.** VM hours as the Nebius console shows them. No price.
- Also: seconds per generation (`timings.jsonl`); prompt tokens of ARC prompts; the share
  of finalists with CNN Workers or a Tool budget above 0; the model-tag entropy
  (`src/cascade/observe.rs:73-74`).

### 9.3 Arms, Knock-outs and the decision rule

- **Arms.** L (language models only), C (plus a `cnn:` tag), Z (plus the Silent CNN),
  T (plus the CNN tool), U (plus the silent tool). Paired runs share seeds, panels
  (`src/cascade/arena.rs:884-885`), models, generations, `max_calls`, `--max-seconds` and
  Mutator settings. We report the measured GPU and CPU seconds of each arm.
- **No LLM rounds.** Counted runs set `llm_every` to 0, as step 1 did. A Mutator arm would
  be a comparison of its own. One fixed local model, named in its pre-registration, would
  answer it through the file bridge of ticket 14. No session that knows the arms writes
  Mutator answers.
- **Primary outcome.** The select-panel solved rate of the best finalist, per run. The
  best finalist is the top bred member by search fitness: the first `Finalist` event, in
  `select_picks` order (`src/cascade/arena.rs:1380-1395`). It is fixed before any select
  answer is read. `best_finalist_select_rate` is the best rate over all finalists
  (`src/cascade/arena.rs:1342-1364`); it stays descriptive.
- **Decision.** The sequential rule of step 1 (ADR 0006:170-185). After each pair of runs,
  in a fixed seed order, the evidence value E is multiplied by 3/2 when the feature arm
  is better, by 1/2 when it is worse, and not at all on a tie. The claim is proven when E
  reaches 100. It fails when E falls to 1/20, or after the most pairs the pre-registration
  allows. A false claim then has a probability of at most 1/100. A proof takes at least 12
  pairs (ADR 0006:175).
- **Knock-outs.** For the best finalist of each run, each select puzzle is scored as bred
  and with the CNN silenced, with the same seeds. The as-bred answers are already stored
  (`src/cascade/arena.rs:796-800`). A silenced tag is served by the bare silent predictor,
  never from a Call record. Every other tag reads the run's Call records, so an unchanged
  request gets its stored reply. New records of a Knock-out go to its own
  `<out>/knockout/worker-calls.jsonl`, so the run's records stay as they were. b counts
  puzzles solved only as bred, c those solved only when silenced. We report b, c and an
  exact one-sided sign test.
- **Pilot first.** 2 seeds × 2 arms, 20 generations: generated puzzles first, then ARC
  training tasks. The pilot sets the costs of Q93, the generations, `max_calls` and the
  most pairs, as step 1 did (ADR 0006:197-215).
- **Pre-registration.** A host claim needs `docs/PREREG_LLM_CNN.md`, frozen at the commit
  that its merge creates (pattern of `docs/PREREG_STEP1_CIRCUITS.md:3-6`).
- **Confirmatory test later.** The sealed endpoint on ARC-AGI-1 evaluation
  (`.scratch/kaskaden-zucht/spec.md:33`) needs kaskaden-zucht 02.

## 10. Tickets and pull requests

### 10.1 Pull requests

| PR | Tickets | Content | Opens when |
|---|---|---|---|
| D | none | This spec, the tickets and the planned terms in CONTEXT.md (docs only) | now |
| 0 | 01 | Call records | 01 is done |
| 1 (F1) | 02 to 07 | CNN Workers, controls, Knock-outs, container evidence | 02 to 07 are done; cnn-basismodelle 04, 06 and 07 have merged |
| H1 | 08 | Nebius host runbook | 08 is done |
| H2 | 09 | ARC-AGI-1 training tasks as a Cascade Domain | after PR 1 |
| P1 | 10 | F1 pre-registration (frozen at merge), then the host report | after the pilot |
| 2 (F2) | 11 to 13 | Architect rounds, controls, container evidence | after PR 1 has merged |
| P2 | 14 | F2 pre-registration, then the host report | after its pilot |
| 3 (F3) | 15 and 16 | CNN tool, controls, container evidence | after PR 2 has merged |
| P3 | 17 | F3 pre-registration, then the host report | after its pilot |
| G1 to G3 | 18 | Glossary updates of existing entries | after PR 1, PR 2 and PR 3 |

Rules for every pull request:
- It is a draft for the owner. None enables auto-merge.
- Each code pull request updates README.md (German, `docs/SCHREIBSTIL.md`) and
  README.en.md.
- Tests are written first and seen to fail.
- The golden tests, `tests/mlp_golden.rs` and the purity scan pass unchanged.

### 10.2 Tickets

| # | File | PR | Status | Blocked by |
|---|---|---|---|---|
| 01 | `issues/01-call-records-for-model-calls.md` | 0 | needs-triage | none |
| 02 | `issues/02-grid-predictor-workers.md` | 1 | needs-triage | none |
| 03 | `issues/03-breeder-rules-for-cnn-workers.md` | 1 | needs-triage | 02; outside: cnn-basismodelle 07 (gate part) |
| 04 | `issues/04-per-task-training-of-a-cnn.md` | 1 | needs-triage | 02; outside: the merge of cnn-basismodelle 04, and 06 for the per-cell objective |
| 05 | `issues/05-cnn-workers-in-brut-breed.md` | 1 | needs-triage | 01, 03, 04, 06; outside: cnn-basismodelle 07 (both strict entry points) |
| 06 | `issues/06-grid-mix-domain-with-the-neighbour-rule.md` | 1 | needs-triage | 02 (both CNN tests also 04) |
| 07 | `issues/07-knock-outs-and-container-evidence-for-f1.md` | 1 | needs-triage | 05, 06 |
| 08 | `issues/08-nebius-host-runbook.md` | H1 | needs-triage | none |
| 09 | `issues/09-arc-training-tasks-as-a-cascade-domain.md` | H2 | needs-triage | PR 1 merged; outside: cnn-basismodelle 06 (data, shared split) and 07 (provenance) |
| 10 | `issues/10-preregistration-and-host-run-cnn-workers.md` | P1 | needs-triage | 07, 08, 09; outside: the adapter follow-ups of 10.3 (context; thinking caps for gpt-oss:20b and qwen3:8b), and kaskaden-zucht 02 for a confirmatory claim |
| 11 | `issues/11-architect-requests-and-blueprint-check.md` | 2 | needs-triage | PR 1 merged; outside: cnn-basismodelle 05 and 07 (bare-genome entry point) |
| 12 | `issues/12-architect-rounds-in-the-mlp-loop.md` | 2 | needs-triage | 01, 11 |
| 13 | `issues/13-architect-controls-and-container-evidence.md` | 2 | needs-triage | 12 |
| 14 | `issues/14-preregistration-and-host-run-architect.md` | P2 | needs-triage | 08, 13; outside: cnn-basismodelle 06; for a gpt-oss Architect (O12), the thinking caps of 10.3 |
| 15 | `issues/15-cnn-tool-protocol-budget-and-turns.md` | 3 | needs-triage | PR 2 merged; 01, 02 |
| 16 | `issues/16-cnn-tool-in-brut-breed-and-container-evidence.md` | 3 | needs-triage | 05, 07, 15 |
| 17 | `issues/17-preregistration-and-host-run-cnn-tool.md` | P3 | needs-triage | 10, 16; outside: the context follow-up of 10.3 |
| 18 | `issues/18-glossary-follow-ups.md` | G1 to G3 | needs-triage | PR 1, PR 2, PR 3 (one part each) |

Every ticket waits for the owner's approval of this design and his answers to O1 to O17
(`docs/agents/triage-labels.md:7-9`). PR #33 already answered O5. Then the unblocked
tickets become ready-for-agent. The owner's steps of ticket 08 become ready-for-human.

### 10.3 Sequencing with the work in progress

- **CNN part 2** (cnn-basismodelle 03, 04, 05, 07) is being built elsewhere. F1's CNN
  predictor (ticket 04) waits for 04 (`Head::PerCell`); ticket 05 waits for 07 (two strict
  entry points, Q91); F2 waits for 05 and 07. Tickets 01, 02, 03 (without the gate part),
  06 and 08 touch none of the CNN files and can be built once the owner approves.
- **CNN part 3** (cnn-basismodelle 06) follows 05. F1's ticket 04 waits for its per-cell
  objective. O2 and O3 ask 06 to train and score genomes the way F1 uses them, and
  section 5, rule 7 asks 06 and 07 to keep the select pool out of genome breeding. This
  design does not edit their tickets.
- **Step 2** (Soul, Coupling strength, Character) is being designed elsewhere. It will
  change Workers, votes and peer judging in `src/cascade/`. F1 adds `predictor.rs` and
  `neighbour.rs` there, and changes only the operators' slot filters, the gene pools, the
  grid-mix Domain, one size method of the Domain trait, the shared renderers and
  `BreedConfig::check`. F3 changes `Traits` and the solve path. Whichever lands second
  rebases. This design takes no ADR number and no step-2 term.
- **kaskaden-zucht 01** (Ollama adapter) is built in part, by PR #33:
  - `backend-ollama` serves `ollama:<model>` tags through `OllamaBackend`, on a loopback
    address only, with its own model list, a timeout and a byte cap (contract I3, I6).
  - `Recorded` files every answered Worker call with its full answer in
    `<out>/llm/worker-calls.jsonl`, and a resume replays it (contract I7).
  - `brut-breed-ollama` fills the slot `EXTRA_BACKENDS` of brut-breed (section 8). That
    answers O5. Every Cascade run on the host uses `brut-breed-ollama`.
- **Adapter follow-ups.** Two are still open in `backend-ollama`. This design does not edit
  kaskaden-zucht 01.
  - **Thinking caps.** #33 allows only models without a thinking mode
    (`backend-ollama/src/lib.rs:12-14, 33-34`). gpt-oss:20b and qwen3:8b wait for the caps
    of contract I5, the rest of kaskaden-zucht 01. A run with a model of its list, such as
    qwen2.5:7b, can start now.
  - **Context.** #33 sends no context setting (`backend-ollama/src/lib.rs:128-140`). ARC
    prompts (ticket 09) and Tool-turn transcripts (F3) need a context that fits, and a
    refusal of prompts that do not fit.
  - Ticket 01 moves `Recorded` from `backend-ollama` into the core, and these follow-ups
    edit `backend-ollama` too. Whichever lands second rebases.
- **kaskaden-zucht 02** is needed only for confirmatory ARC claims.

## 11. Open questions, each with a recommended answer

- **O1. Which genome is the first CNN Worker?** In tests, a hand-written per-cell
  reference genome in `tests/fixtures/cnn/`. In the host pilot, a fully convolutional base
  model of cnn-basismodelle 04 or 06 (U-Net or an FCN), promoted through cnn-basismodelle
  07. In counted runs, a promoted genome bred on ARC-AGI-1 training tasks outside the
  select pool (section 5, rule 7).
- **O2. How should cnn-basismodelle 06 score CNN genomes on ARC?** Per task: train on each
  task's examples and score on its test pairs, as F1 uses them. A genome bred by
  pretraining across families is chosen for another job. This is a suggestion to the CNN
  part 3 work.
- **O3. How should 06 handle grids of different sizes?** Pad per task, add an inside
  channel and leave padded cells out of the loss. F1 reuses it. If 06 decides otherwise,
  ticket 04 adds the mask after 06 merges.
- **O4. Should the Call record ship alone, first?** (updated 2026-10-31 after PR #33) Yes.
  It is small and unblocked. It builds on the record of PR #33 and helps its runs at once:
  a failed Ollama call becomes a fault, not a committed lost call, and a replay needs no
  Ollama. If the owner prefers, ticket 01 joins PR 1.
- **O5. How does brut-breed reach real models without an HTTP client in the core?**
  (updated 2026-10-31 after PR #33) Answered by PR #33. brut-breed has a slot,
  `EXTRA_BACKENDS`, for one `BackendFactory`. `brut-breed-ollama` includes
  `src/bin/brut-breed.rs` as a module, fills the slot and calls its `main` (section 8). The
  run loop stayed in the binary; nothing moved into the library. CNN tags need no slot,
  because CNN Workers live in the core.
- **O6. How does the VM learn of the 60 s warning?** Read the Nebius documentation, then
  test once with a console stop during a dummy run. If the warning arrives as a normal
  shutdown, the systemd unit writes STOP. If nothing arrives, rely on checkpoints,
  reservations and Call records.
- **O7. Should arm C start with a hand-written CNN Cascade?** Not in the primary
  comparison, because breeding must find it. A later descriptive arm could, but it needs a
  base-Cascade setting that this design does not build.
- **O8. Should judges use the CNN tool?** Not in F3. It would be a later comparison.
- **O9. May the Mutator change a CNN Worker's genome inside a Cascade?** No. The tag names
  fixed content. Genomes change only in the MLP mode, through random mutation or the
  Architect, and enter Cascades through reviewed promotion.
- **O10. What about ARC tasks that change the grid size?** The CNN refuses them in F1, and
  reports give the share. An output-size predictor is later work.
- **O11. Should a CNN Worker read earlier answers as an extra input, or train on rotated or
  recoloured copies of the examples?** Not in v1. Each would be a later variant with its
  own comparison. Augmentation can change the rule.
- **O12. Which model is the Architect on the host?** (updated 2026-10-31 after PR #33) One
  fixed model, named in the pre-registration and in every side file. Recommend gpt-oss:120b
  if M2 shows that it fits next to the Worker models, else gpt-oss:20b. Both are thinking
  models, which PR #33 does not serve yet (10.3). Until then, recommend qwen2.5:32b, the
  largest model on its list. In the container a Claude session answers.
- **O13. Where do the Call-record tarballs live?** They stay out of git, and real models
  are reproducible only through them (contract I7). So they need a lasting store that
  outlives the VM and its disk. Recommend a storage bucket that the owner controls, or a
  release asset that the owner uploads. The owner names it before the first counted run.
- **O14. When does section 13 become an ADR?** When the first build pull request (PR 0 or
  PR 1) merges, with the next free number after step 2's ADRs.
- **O15. Native tool calling?** Only if the pilot shows that fewer than half of the real
  models' tool requests parse.
- **O16. Does a CNN Worker carry a Soul in step 2?** Treat it like every Worker. The step-2
  design decides.
- **O17. Money ranges in the cnn-basismodelle files** (spec.md:141-145; ticket 06:36 and
  others). Remove them when those files are next edited. The owner decides.

## 12. Risks

| Risk | Effect | Response |
|---|---|---|
| Three to five examples teach a CNN too little | The positive control fails; breeding drops the CNN on ARC | Calibrate on grid-mix first (ticket 06). Report a negative ARC result as a result. |
| The time model does not fit the VM CPU | Slower runs; the bits stay equal | M3. Lower the limits if needed. |
| 16 vCPU become the bottleneck while the GPU waits | Long generations | One training per task through the memo; its extra copies are counted. M1 and M2. CNN calls pay their measured cost. |
| CNN bits differ between the container and the VM | A re-run elsewhere differs | Compare within one machine (ADR 0007:111-114). Replay from the Call records. |
| A wrong cost per call | Selection keeps or drops CNNs for the wrong reason | Measured costs (Q93). Knock-outs measure the contribution whatever the cost. |
| Simulated controls oversell | False confidence | Every report labels them. Claims come only from host runs and the sealed endpoint. |
| Open models saw ARC evaluation tasks | Absolute scores inflated | Paired arms with the same models (section 5, rule 12). |
| Long ARC prompts are cut by the model server | Wrong answers for a hidden reason | A follow-up to the adapter sets the context and refuses prompts that do not fit; PR #33 does not yet (10.3). Measure prompt tokens. |
| A 30 × 30 answer needs about 900 whitespace tokens, more for real tokenizers | Seeds with 512 answer tokens fail on large grids (`src/cascade/genome.rs:34-38`) | The ARC gene pool gives its seeds 2048 answer tokens (ticket 09). Report `Length` stops. |
| Real models ignore the tool protocol, or ask again and again | The tool goes unused or wastes calls | Budget ceiling in code. A request after the budget gives no answer. M5. O15. |
| Tool budgets inflate the reserved calls | Runs end early | Set `max_calls` from the pilot. |
| Selection exploits the CNN judge rule | Inflated judging skill | A refusal on a scored pair counts as wrong. Report the judging skill of Cascades whose last Stage starts with a CNN. |
| Call records fill the disk | A run stops | M8. The runbook checks free space before each run (ticket 08). A failed record write is a fault, so nothing is committed (4.1). Compress for transport. |
| The kill switch waits for a training | Slower stops | Bounded by the time ceiling. Reported (4.2.2). |
| Golden fixtures break by accident | Lost comparability with earlier runs | Defaults left out of the JSON. Guarded random draws. Golden tests first in every ticket. |
| Conflicts with CNN parts 2 and 3 and with step 2 | Merge work | 10.3. New fields are optional and left out when empty. |
| Mutation pull requests to `candidates/` need no code owner | A model enters unreviewed | The gate resolves every `cnn:` tag (Q94). The owner merges. No auto-merge. |
| A preemption during the select phase | Lost work | Answers are kept and reservations count (`src/cascade/arena.rs:1285-1315`). The run may need a larger `max_calls` (ADR 0005:52-53). |
| Billing goes on after a shutdown inside the VM | Spending without work | Every runbook path ends with a console or API stop and a look at the console. |
| Most Blueprints are invalid | F2 adds little | M7. Refused Blueprints become random children. The brief is generated from code. |
| Without kaskaden-zucht 02 no confirmatory ARC claim is possible | Only descriptive ARC numbers | Say so in every report until the sealed endpoint exists. |
| Remote Control is not ready | No agent on the host | Run the same steps by SSH from the runbook. |
| Records of sealed runs leak into later searches | Contamination | They never leave the evaluator (section 5, rule 11). |

## 13. Decision record

This text becomes an ADR when the first build pull request merges. Its number is assigned
then, after the ADRs of step 2.

> **LLMs use CNNs: CNN Workers, an Architect for CNN genomes, and a CNN tool**
>
> The owner asked on 2026-10-31 that the breeding machine let language models use
> convolutional networks, in three ways and in this order, each as its own pull request:
> a CNN as a Worker of a Cascade (F1), a language model that designs CNN genomes (F2), and
> a CNN as a tool that a language-model Worker calls while it answers (F3).
>
> **A CNN Worker is an ordinary Worker.** Its model tag is `cnn:` plus the content id of a
> CNN genome, so the allow-list, the model pool, ModelSwitch and crossover need no change,
> and Cascade ids stay as they were. For each task it trains a fresh CNN of that genome on
> the task's worked examples (Per-task training) and answers with the CNN's output grid.
> It ignores its prompt sections, its Traits, the request seed and earlier answers, so its
> answer depends only on the genome and the task. It refuses tasks whose examples change
> the grid size. As a judge it picks the candidate that agrees with its own prediction on
> more cells. Per-task training reads no clock: `Genome::validate` bounds its work before
> it starts, so a CNN Worker gives the same bits at any thread count on one machine
> (ADR 0007). We rejected a pretrained CNN, because a task's input is one grid and such a
> CNN cannot read a new task's examples.
>
> **Breeding finds out by itself.** The arms of an experiment start from the same seed
> Cascades, none with a CNN Worker, and differ only in their allow-list. ModelSwitch and
> stage crossover bring CNN Workers in; fitness alone keeps them. Counted runs have no LLM
> rounds, so no answerer who knows the arms can steer them. The random operators that edit
> or clone a Worker skip CNN Workers, and a Stage may not hold two Workers of one CNN tag.
> A positive control, the grid-mix Domain with its Neighbour rule that no simulated
> hypothesis fits, and a negative control, the Silent CNN, show that the machinery keeps a
> CNN that helps and drops one that does not. Knock-outs score the same Cascade again with
> its CNN silenced, on the same tasks with the same call seeds; a silenced CNN is never
> answered from a stored reply. We rejected hand-placed CNN Stages at the start, because
> the owner asked that breeding find out by itself.
>
> **Every call to a model that is not simulated is a Call record.** It stores the full
> request and the full reply, or the error, under a key of its tag and its request, as soon
> as the call returns. A resumed Run reuses stored replies, so a preempted host loses only
> the calls in flight, and a replayed Run needs no model. This extends ADR 0002 from the
> Mutator to Workers and tools (contract invariant I7). PR #33 began it for the Worker calls
> of real models, in `backend-ollama`; this design moves that record into the core and
> extends it to CNN Workers, the CNN tool and errors. Simulated models are not recorded;
> they are deterministic. (updated 2026-10-31 after PR #33)
>
> **The Architect designs CNN genomes in the MLP mode.** In fixed Architect rounds of
> `brut run`, a language model receives parents with their public validation scores and
> returns Blueprints. Its answers arrive only as files, are recorded in the lineage
> archive and replayed, and a missing answer pauses the Run with exit code 3, as for the
> Mutator (ADR 0005). A Blueprint is checked strictly, never repaired, and gets its seed
> from the run. Good genomes reach Cascades only through the reviewed promotion of models.
> We rejected CNN genomes as Cascade genes, because a genome is valid only for one task
> specification and the MLP mode is where CNNs are trained, scored and held out.
>
> **The CNN tool is a counted dialogue in the pure arena.** A Worker's Tool budget is a
> Trait from 0 to a ceiling of 3 in code. The protocol is plain text inside the existing
> request (`TOOL CNN`, `TOOL RESULT`). Every model turn and every tool call is a counted,
> recorded call, so the call bounds and the crash-safe budget still hold. Judges never use
> the tool. The tool's tag stays off the Workers' allow-list, so a tool arm has no CNN
> Worker and F1 and F3 are measured apart. We rejected a wrapping backend that hides the
> extra calls, because `max_calls` would no longer hold, and native tool calling for now,
> because the backend contract knows only `/api/generate`.
>
> **Evidence decides.** Arm comparisons pair runs by seed, with the same models, seeds and
> budgets, and decide by the sequential rule of step 1 (ADR 0006). Results of simulated
> models are positive controls only. Claims about ARC-AGI-1 evaluation wait for the sealed
> endpoint. CNNs stay on the CPU; language models use the GPU. No report gives money
> figures.
>
> Status: proposed.

## Comments
