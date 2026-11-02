# Candidate pool

This directory holds candidate learners, one JSON file per candidate, named by its
content id (`<id>.json`, where the id is the first 16 hex of the SHA-256 over the
canonical genome JSON, the same scheme as `LearnerGenome::id`).

It is the mutation sandbox of the human-in-the-loop path (ADR 0004). A mutation pull
request (branch prefix `mutate/`) may add files **here only**. The `scope-fence` CI job
rejects any such pull request that touches a path outside `candidates/`, or that changes
or removes a file the base branch holds. It revalidates every candidate (parse,
structural check, cost cap, and that each file name equals its content id). The gate
reads only regular files of at most 16 MiB (`hitl::MAX_CANDIDATE_FILE_BYTES`), so a
symlink, a FIFO or a huge file is rejected, not read.

Tools:

- `brut-propose --out <dir> --seed <u64> [--pool candidates] [--mode full|constants]`
  drafts two mutated candidates plus a provenance manifest under `<dir>`. With no pool it
  falls back to the seed programs.
- `scripts/propose-pr.sh <dir>` copies those two files in here, commits on a `mutate/`
  branch, and opens a draft pull request with the provenance in the body. It never merges.
- `brut-ingest --candidates candidates` validates the pool; it is the gate that admits
  merged candidates into a run.

## Cascades (ADR 0005)

Promoted Cascades live in `candidates/cascades/<id>.json`, where the id is the first 16
hex of the SHA-256 over the Cascade's canonical JSON (`Cascade::id`). They arrive the
same way, two per draft pull request:

- `brut-breed --out <dir> --promote` writes the two best bred finalists and a provenance
  manifest under `<dir>/promote/`.
- `scripts/propose-pr.sh <dir>/promote` opens the draft pull request.
- `brut-ingest --cascades candidates/cascades` is the gate: parse, check against the pool
  allow-list (no credential look-alikes in prompt text), file name equals content id, no
  duplicate of a seed Cascade.

## Models (ADR 0008)

Promoted models of the MLP mode, MLPs and CNNs, live in `candidates/models/<id>.json`, where
the id is the first 16 hex of the SHA-256 over the model genome's canonical JSON
(`Genome::id`). Each file holds the genome and its provenance: the run, the task, the master
seed, the commit, the scores on the validation and hidden splits, the run's best base model,
and the lineage back to generation 0. They arrive the same way, one or two per draft pull
request:

- `brut promote <run> --commit <sha>` writes the two best models of a finished run by hidden
  score that beat its best base model, and a provenance manifest, under `<run>/promote/`. A
  model already in this directory, or equal to a base model of a CNN family, adds nothing.
  Only a run that breeds CNNs or starts from promoted models can promote.
- `scripts/propose-pr.sh <run>/promote` opens the draft pull request. It refuses a file that
  the base branch holds already, and CI refuses a pull request that changes or removes one. A
  model's content id hashes its genome alone, so a rewritten provenance would keep the name.
- `brut-ingest --models candidates/models` is the gate: parse (unknown fields refused), the
  shape check and the current default limits on the task of the provenance, a consistent
  provenance, file name equals content id, no duplicate of a base model of a CNN family.
- `brut init <dir> --promoted candidates/models` copies the merged models into the config of a
  new run. Each joins its generation 0 where it fits the task, the limits and the model kinds;
  the archive names each one left out.
