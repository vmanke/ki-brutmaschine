# Brutmaschine (English)

This page is the English version of the Stage 2c part of the German
[README.md](README.md). The rest of the project documentation is in German; the code,
its comments, `CONTEXT.md` and the ADRs are in English.

## Cascade breeding (Stage 2c)

Status: built with the simulated model (PR #8); real models follow. Described in
[docs/STUFE2C_KASKADEN_ZUCHT.md](docs/STUFE2C_KASKADEN_ZUCHT.md) and
[ADR 0005](docs/adr/0005-cascades-are-bred-by-darwin-goedel-selection-and-every-cascade-is-also-a-judge.md).

Here the Brutmaschine breeds **Cascades** of language-model **Workers**. A Worker is a
model with a system prompt and **Traits** (thinking budget, temperature, answer length,
self-check). A Cascade has one to four Stages, with one to three Workers each. Every
Cascade is also a judge. It judges the answers of other Cascades blind, never those of
a parent, child or sibling. The Brutmaschine chooses parents the way the Darwin Gödel
Machine does. Claude can write new Cascades as the **Mutator**.

The same rule holds here: candidates are data. A Cascade is JSON. Its prompt is sent to a
model and never executed. There is no language model in the container. The Workers use a
simulated model as a positive control. Real Runs need a GPU server with Ollama (ADR 0003).

```sh
cargo build --release

# Breeding Run with the simulated model
./target/release/brut-breed --out runs/k1 --seed 1 --generations 10

# Resume, optionally with more generations or more calls
./target/release/brut-breed --out runs/k1 --resume --generations 20

# Verify the hash-chained event log and prepare the two best Cascades for a PR
./target/release/brut-breed --out runs/k1 --verify
./target/release/brut-breed --out runs/k1 --promote
scripts/propose-pr.sh runs/k1/promote

# Kill switch: the Run stops before the next puzzle
touch runs/k1/STOP

# Going on after the kill switch: delete STOP first, then resume
rm runs/k1/STOP
./target/release/brut-breed --out runs/k1 --resume
```

With `--llm-every K`, every K-th generation is an LLM round. If an answer from the
Mutator is missing, the Run pauses and exits with exit code 3. The requests are then in
`runs/k1/llm/requests/`. Write each answer as `runs/k1/llm/answers/<schlüssel>.txt`, or
let a Claude session do it. Here `<schlüssel>` is the key from the request's file name.
Then continue the Run with `--resume`. If `--llm-answers <dir>` puts the answers
elsewhere, give that option on resume too. Exit codes: 0 = done or stopped cleanly,
1 = error, 2 = wrong usage (also: another process holds the run directory),
3 = waiting for LLM answers.

`--resume` may change only `--generations`, `--max-calls`, `--max-seconds` and
`--llm-answers`; everything else defines the experiment. Before each generation, the
checkpoint reserves its worst-case number of calls. If the process ends hard (kill,
crash), the resume counts that reservation as spent, so `--max-calls` holds even then.
Details are in [docs/STAGE2C_CASCADE_BREEDING.md](docs/STAGE2C_CASCADE_BREEDING.md),
section 5.
