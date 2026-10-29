# Spec: Cascade breeding (Stage 2c)

Status: ready-for-agent (container part built on branch `claude/kind-dijkstra-1bfe2f`)

Decision record: docs/adr/0005-cascades-are-bred-by-darwin-goedel-selection-and-every-cascade-is-also-a-judge.md.
Glossary: CONTEXT.md, section "Cascade breeding (Stage 2c)".

## Goal

Breed Cascades of model Workers, Darwin-Gödel style, that solve ARC-style grid puzzles.
Every Cascade is also a judge. Claude rewrites Cascades as the Mutator. The two best
Cascades of a Run are promoted through a human-reviewed pull request. Real runs use open
models on a rented GPU host.

## Decisions from the grilling session (2026-10-29)

| # | Decision |
|---|---|
| Q1 | Worker = model + sectioned system prompt + Traits. Cascade = stages of Workers; the Cascade competes. |
| Q2 | Peer judgment steers search (parents, mates); judging skill is itself bred. The objective, sealed score stays the endpoint. |
| Q3 | Simulated model here; real open models on a rented host (Hetzner GPU). |
| Q4 | Own ARC-style generator here and in tests; ARC-AGI-1 for host runs. |
| Q5 | Two speeds: sealed fast loop; only promoted Cascades enter the pool, by human-merged PR. |
| Q6 | Mutations change data first (prompts, stages, Traits); Rust candidate code later, sandboxed. |
| Q7 | Pilot on GEX44 (20 GB), then decide on GEX131 (96 GB) with measured numbers. |
| Q8 | Rent in blocks; every run is resumable from a checkpoint. |
| Q9 | The host pushes results to a `runs/` branch with a deploy key kept in a `.env` on the host. |
| Q10 | Claude may change the breeder only through code-owner-reviewed PRs; the evaluator never. |
| Q11 | Depth 1-4, width 1-3, residual path, blind in-stage vote. |
| Q12 | Blind pairwise judging across Cascades, never by relatives; Bradley-Terry; judging skill weights votes. |
| Q13 | Fitness = 0.8 solved + 0.2 judging skill - cost; peer rating breaks ties; DGM parent selection. |
| Q14 | Stage crossover, section crossover, Claude crossover (LLM rounds only, recorded). |
| Q15 | Endpoint: best Cascade vs compute-matched single Worker on ARC-AGI-1 eval, pass@2, sign test p < 0.05. |
| Q16 | Hard caps per run: wall time, Worker calls, 2048 answer tokens. Pilot: 2 h, 20 000 calls. |
| Q17 | Traits: thinking budget, temperature, answer length, self-check (vote weight is learned, not inherited). |
| Q18 | Traits fixed first; a run-time "feeling" signal later as a pre-registered comparison. |
| Q19 | Thinking ceiling in code, thinking costs fitness, per-stage ceilings. |
| Q20 | gpt-oss-20b as main Worker, Qwen3-8B as second model (to verify on the host). |
| Q21 | Single-model runs and mixed-model runs with per-call prices, as separate experiments. |
| Q22 | The baseline gets the same models and budget as the Cascade it is compared with. |
| Q23 | Claude as a Worker later, only through the Broker with a euro cap. |

## Built in this container

- `src/cascade/genome.rs`: Worker, Traits, Stage, Cascade, limits, ceilings, seeds, pool allow-list.
- `src/cascade/puzzle.rs`: ARC-style generator, prompt protocol, answer and verdict parsing.
- `src/cascade/sim.rs`: simulated Worker model (positive control).
- `src/cascade/variation.rs`: mutation, stage and section crossover, Mutator requests and parsing.
- `src/cascade/arena.rs`: Cascade runs, blind votes, peer round, Bradley-Terry, DGM selection, breeding loop, checkpoints.
- `src/bin/brut-breed.rs`: run directory, hash-chained log, kill switch, LLM request/answer files, promotion.
- `brut-ingest --cascades`, `scripts/propose-pr.sh` and the scope-fence CI accept promoted Cascades.

## Open work

See `issues/`.
