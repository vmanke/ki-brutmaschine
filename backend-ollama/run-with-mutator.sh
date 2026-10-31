#!/usr/bin/env bash
# run-with-mutator.sh: breed a run whose LLM rounds a local model answers.
#
# Usage:
#   backend-ollama/run-with-mutator.sh --tag <vllm:<model>|ollama:<model>>
#       [--max-rounds N] [--max-requests R] [--parallel P] -- <brut-breed-ollama options>
#
# It runs brut-breed-ollama with the options after "--" (a fresh run or a --resume).
# Whenever the run exits with 3 (an LLM round waits for answers), brut-mutator answers
# the pending requests with the model <tag>, and the run resumes with
#   brut-breed-ollama --out <dir> --resume
# plus those of the given options a resume accepts (--generations, --max-calls,
# --max-seconds, --llm-answers). brut-mutator writes into the same answers directory
# the run reads (--llm-answers, default <dir>/llm/answers) and records every call in
# <dir>/llm/mutator-calls.jsonl.
#
# It stops
#   - when brut-breed-ollama exits with anything but 3, and exits with that code
#     (0 done or cleanly stopped, 1 error, 2 wrong usage);
#   - when <dir>/STOP exists, the run's own kill switch (exit 0);
#   - when a mutator round fails (exit 1; 2 for wrong usage or a held lock), also when
#     it finds nothing to answer although the run waits;
#   - after --max-rounds mutator rounds (default 20) while the run still waits (exit 3).
#
# The programs come from $BRUT_BIN_DIR (default: target/release beside this script).
set -uo pipefail

prog="run-with-mutator"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bin_dir="${BRUT_BIN_DIR:-$here/target/release}"
breed="$bin_dir/brut-breed-ollama"
mutator="$bin_dir/brut-mutator"

usage() {
    sed -n '2,25p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//' >&2
    exit 2
}

fail() {
    echo "$prog: $*" >&2
    exit 2
}

tag=""
max_rounds=20
mutator_opts=()

while [ $# -gt 0 ]; do
    case "$1" in
        --tag) [ $# -ge 2 ] || fail "--tag needs a value"; tag="$2"; shift 2 ;;
        --max-rounds) [ $# -ge 2 ] || fail "--max-rounds needs a value"; max_rounds="$2"; shift 2 ;;
        --max-requests|--parallel)
            [ $# -ge 2 ] || fail "$1 needs a value"; mutator_opts+=("$1" "$2"); shift 2 ;;
        --) shift; break ;;
        -h|--help) usage ;;
        *) fail "unknown option $1 (the options of brut-breed-ollama go after --)" ;;
    esac
done

[ -n "$tag" ] || fail "--tag vllm:<model> or ollama:<model> is required"
[[ "$max_rounds" =~ ^[1-9][0-9]*$ ]] || fail "--max-rounds needs a number from 1 up, got '$max_rounds'"
[ $# -gt 0 ] || fail "the options of brut-breed-ollama are missing after --"

breed_args=("$@")
out=""
resume_opts=()
answers_opts=()
i=0

# brut-breed takes --key value pairs; only these four options take no value.
while [ $i -lt ${#breed_args[@]} ]; do
    key="${breed_args[$i]}"
    val="${breed_args[$((i + 1))]:-}"

    case "$key" in
        --resume|--promote|--verify|--no-pause) i=$((i + 1)) ;;
        --out) out="$val"; i=$((i + 2)) ;;
        --llm-answers) answers_opts=(--llm-answers "$val"); resume_opts+=("$key" "$val"); i=$((i + 2)) ;;
        --generations|--max-calls|--max-seconds) resume_opts+=("$key" "$val"); i=$((i + 2)) ;;
        *) i=$((i + 2)) ;;
    esac
done

[ -n "$out" ] || fail "the brut-breed-ollama options need --out <dir>"

for p in "$breed" "$mutator"; do
    [ -x "$p" ] || fail "$p is missing; build it with: cd backend-ollama && cargo build --release"
done

stop_file="$out/STOP"
stopped() {
    echo "$prog: $stop_file exists; stopped $1. Delete it, then run this command again with --resume." >&2
    exit 0
}

cmd=("${breed_args[@]}")
rounds=0

while :; do
    [ -e "$stop_file" ] && stopped "before brut-breed-ollama"

    "$breed" "${cmd[@]}"
    code=$?

    [ "$code" -eq 3 ] || exit "$code"
    [ -e "$stop_file" ] && stopped "while the run waits for LLM answers"

    if [ "$rounds" -ge "$max_rounds" ]; then
        echo "$prog: the run still waits after $rounds mutator round(s) (--max-rounds $max_rounds)" >&2
        exit 3
    fi

    rounds=$((rounds + 1))
    echo "$prog: mutator round $rounds of at most $max_rounds with $tag" >&2

    "$mutator" --out "$out" --tag "$tag" "${answers_opts[@]}" "${mutator_opts[@]}"
    m=$?

    case "$m" in
        0) ;;
        2) echo "$prog: brut-mutator refused (exit 2)" >&2; exit 2 ;;
        4) echo "$prog: the run waits, but brut-mutator found nothing to answer; do both read the same --llm-answers?" >&2; exit 1 ;;
        *) echo "$prog: brut-mutator failed (exit $m); the answers written so far stay" >&2; exit 1 ;;
    esac

    cmd=(--out "$out" --resume "${resume_opts[@]}")
done
