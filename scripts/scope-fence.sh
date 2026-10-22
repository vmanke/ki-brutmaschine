#!/usr/bin/env bash
# Scope fence for mutation pull requests (ADR 0004): a mutation pull request may change
# files under candidates/ only. The one scope rule lives in `brut-ingest --scope-check`
# (Rust); this script only collects the changed files from git and feeds them in.
#
# Usage: scripts/scope-fence.sh <base-ref>
#   BRUT_INGEST may point at a prebuilt brut-ingest binary; otherwise `cargo run` is used.
set -euo pipefail

base="${1:?usage: scope-fence.sh <base-ref>}"
ingest="${BRUT_INGEST:-}"

# --no-renames so a file moved out of a protected path into candidates/ shows up as the
# source deletion (outside candidates/) too, not only the destination addition.
changed="$(git diff --no-renames --name-only "${base}...HEAD")"

if [ -z "${changed}" ]; then
  echo "scope-fence: no changed files"
  exit 0
fi

if [ -n "${ingest}" ]; then
  printf '%s\n' "${changed}" | "${ingest}" --scope-check
else
  printf '%s\n' "${changed}" | cargo run --quiet --bin brut-ingest -- --scope-check
fi
