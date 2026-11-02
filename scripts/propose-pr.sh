#!/usr/bin/env bash
# Draft a human-review pull request from a brut-propose output directory (ADR 0004).
#
# Thin git + gh wrapper, kept outside the core crate so the crate keeps no process or
# network dependency (ADR 0001). It works for brut-propose output (two genomes), for
# brut-breed --promote output (two Cascades under candidates/cascades/, ADR 0005) and for
# brut promote output (one or two models under candidates/models/, ADR 0008).
# It copies the candidate files into candidates/,
# commits on a `mutate/` branch, pushes, and opens a DRAFT pull request with the
# provenance manifest in the body. It never merges; a human reviews and merges.
#
# Usage: scripts/propose-pr.sh <out-dir> [base-branch]
set -euo pipefail

out="${1:?usage: propose-pr.sh <out-dir> [base-branch]}"
base="${2:-main}"

[ -d "${out}/candidates" ] || { echo "no ${out}/candidates"; exit 1; }
[ -f "${out}/proposal.json" ] || { echo "no ${out}/proposal.json"; exit 1; }

seed="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["master_seed"])' "${out}/proposal.json")"
adapter="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("adapter", "genome"))' "${out}/proposal.json")"
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
branch="mutate/${seed}-${stamp}"
count="$(find "${out}/candidates" -type f -name '*.json' | wc -l | tr -d ' ')"
noun="candidates"
[ "${count}" = "1" ] && noun="candidate"
[ "${count}" != "0" ] || { echo "no candidate files under ${out}/candidates"; exit 1; }

# Mutations are reviewed blind to held-out scores (ADR 0004). A model promotion records the
# hidden scores by which the tool chose its models (ADR 0008); review still judges only
# validity, scope, safety and provenance.
if [ "${adapter}" = "model" ]; then
  note="The tool chose these models by the hidden split of their run. Review judges validity, scope, safety and provenance. No auto-merge."
else
  note="Review judges validity, scope, safety and provenance, never a held-out score. No auto-merge."
fi

git fetch origin "${base}"
git checkout -B "${branch}" "origin/${base}"

# A candidate file is only ever added. Its name is its content id, so a file that the base
# branch holds already is a duplicate, and a duplicate adds nothing. For a model the file also
# holds provenance, so copying it would rewrite the pooled model's provenance.
while IFS= read -r -d '' file; do
  rel="${file#"${out}/"}"
  if [ -e "${rel}" ]; then
    echo "${rel} exists on ${base} already; a duplicate adds nothing"
    exit 1
  fi
done < <(find "${out}/candidates" -type f -name '*.json' -print0)

mkdir -p candidates
cp -R "${out}/candidates/." candidates/
git add candidates/
git commit -q -m "mutate: ${count} ${adapter} ${noun} for review (seed ${seed})"
git push -u origin "${branch}"

# Build the pull-request body: a short note plus the provenance manifest.
body="$(mktemp)"
{
  echo "${count} ${adapter} ${noun} for human review (ADR 0004), seed ${seed}."
  echo
  echo "Scope: candidates/ only. ${note}"
  echo
  echo '```json'
  cat "${out}/proposal.json"
  echo '```'
} > "${body}"

# REST create (GraphQL may be blocked by the proxy); draft by default.
gh api "repos/{owner}/{repo}/pulls" \
  -f title="mutate: ${count} ${adapter} ${noun} for review (seed ${seed})" \
  -f head="${branch}" \
  -f base="${base}" \
  -F body=@"${body}" \
  -F draft=true

rm -f "${body}"
