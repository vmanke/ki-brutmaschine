#!/usr/bin/env bash
# run-seed.sh: one seed of docs/PREREG_BREEDING_VS_VOTE.md, as its section 8 says.
#
# Usage, from the repository root:
#   docs/prereg-breeding-vs-vote/run-seed.sh --seed S [--runs DIR]
#
# S is a counted seed 1 to 60, run in the order of the list, or a pilot seed 901 to 905.
# DIR is the runs directory (default: runs). For seed S the script
#   1. breeds arm B: brut-breed-ollama --out DIR/B-S --config config-B.json --seed S.
#      Each time the run exits with 3, a Mutator round waits for answers (D2):
#      brut-mutator answers them with the model of D1, and the run goes on with --resume.
#   2. checks it with brut-breed-ollama --out DIR/B-S --verify.
#   3. matches the calls and runs the vote (arm V). `breeding-vs-vote --match DIR/B-S`
#      says how many samples the vote record must hold, and
#      `vote-control --run DIR/B-S --temperature 0.7 --samples N` makes them, until the
#      matching names N_up and N_down. Then vote-control runs with --samples N_down and
#      with --samples N_up; both replay the record and ask no model. After each run,
#      DIR/B-S/vote-control.json is copied to vote-down.json or vote-up.json, which the
#      analysis reads. No vote-down.json exists when no N_down exists.
#   4. breeds and checks arm L the same way, with config-L.json and --selection lexicase,
#      after arm B, so that its generation 0 comes from the seed's call cache.
#   5. writes DIR/B-S/panels.json for the recount (breeding-vs-vote --panels).
#   6. runs the analysis on every pair in DIR (breeding-vs-vote --runs DIR, with --pilot
#      for a pilot seed). It names the next seed or gives the verdicts.
#
# It never deletes anything. It resumes a run only after a pause for Mutator answers
# (exit 3). It stops with a message for a run that failed calls, failed --verify, or was
# killed, stopped or interrupted in any other way: section 8 wants such a run deleted and
# run again from scratch, and that is left to a person. It is safe to run again: a
# finished run is checked, not bred again; kept vote results are checked and kept.
#
# Each run directory gets a journal, driver.log: one line per program start and end,
# with the time. A run whose last breeding process ended with 3 waits cleanly for
# Mutator answers, and a second call of this script goes on with it. Any other
# unfinished run was interrupted.
#
# The call cache: both arms of the seed share DIR/call-cache/seed-S.jsonl (it sets
# BRUT_CALL_CACHE). Arm B files every answer there, and arm L takes the answers of
# identical requests from it: generation 0 above all, which section 4 wants equal in both
# arms. One file per seed keeps seeds apart and keeps the file small; a process loads the
# whole file on its first miss. A run deleted and run again also takes the answers of
# identical requests from it; only its failed calls go to the model again.
#
# Environment:
#   BRUT_BIN_DIR     the programs (default: backend-ollama/target/release of this checkout)
#   VLLM_URL         the vLLM server, as for the programs (default http://127.0.0.1:8000)
#   BRUT_CALL_CACHE  set by this script (see above); "off" is refused
#   PREREG_FROZEN    the directory of config-B.json and config-L.json (default: this
#                    script's directory; only the tests change it)
#   PREREG_RETRY_SECONDS  the pause before brut-mutator or vote-control asks failed
#                    calls again (default 30)
#
# Exit codes: 0 done, 1 stopped with a message, 2 wrong usage.
set -uo pipefail

prog="run-seed"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "$here/../.." && pwd)"
bin="${BRUT_BIN_DIR:-$root/backend-ollama/target/release}"
frozen="${PREREG_FROZEN:-$here}"
temperature="0.7"
# A run of 30 generations with a Mutator round in every second one pauses 15 times.
max_rounds=40
# Tries of brut-mutator and vote-control, which ask the model again for failed calls.
tries=3
pause="${PREREG_RETRY_SECONDS:-30}"

usage() {
    sed -n '2,8p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//' >&2
    exit 2
}

say() {
    echo "$prog: $*" >&2
}

stop() {
    say "$*"
    exit 1
}

seed=""
runs="runs"

while [ $# -gt 0 ]; do
    case "$1" in
        --seed) [ $# -ge 2 ] || usage; seed="$2"; shift 2 ;;
        --runs) [ $# -ge 2 ] || usage; runs="$2"; shift 2 ;;
        *) usage ;;
    esac
done

[[ "$seed" =~ ^[1-9][0-9]*$ ]] || usage

pilot=()

if [ "$seed" -ge 901 ] && [ "$seed" -le 905 ]; then
    pilot=(--pilot)
elif [ "$seed" -gt 60 ]; then
    say "seed $seed is neither a counted seed 1 to 60 nor a pilot seed 901 to 905"
    exit 2
fi

[ "${BRUT_CALL_CACHE:-}" != "off" ] || stop "BRUT_CALL_CACHE is off, but arm L must take generation 0 from the seed's call cache"

for p in brut-breed-ollama brut-mutator vote-control breeding-vs-vote; do
    [ -x "$bin/$p" ] || stop "$bin/$p is missing; build it with: cd backend-ollama && cargo build --release"
done

for f in config-B.json config-L.json; do
    [ -f "$frozen/$f" ] || stop "$frozen/$f is missing"
done

# The model of D1, which also answers the Mutator rounds (D2).
tag="$(sed -n 's/.*"tag": *"\([^"]*\)".*/\1/p' "$frozen/config-B.json" | head -n 1)"
[ -n "$tag" ] || stop "$frozen/config-B.json names no model"

mkdir -p "$runs/call-cache" || stop "cannot create $runs/call-cache"
export BRUT_CALL_CACHE="$runs/call-cache/seed-$seed.jsonl"
started=$(date +%s)

journal() {
    printf '%s %s\n' "$(date +%s)" "$2" >>"$1/driver.log"
}

# Stop when the run in $1 filed a failed Worker call: it is not counted (section 8).
no_failed_calls() {
    if [ -s "$1/llm/worker-errors.jsonl" ]; then
        stop "$1: Worker calls failed ($1/llm/worker-errors.jsonl). Section 8: the run is not counted. Delete it (rm -r $1) and run this script again."
    fi
}

# One Mutator round for the paused run $1. $2 is 1 when the run paused in this call of
# the script; then "nothing to answer" is an error, as in backend-ollama/run-with-mutator.sh.
mutator_round() {
    local attempt m

    for attempt in $(seq 1 "$tries"); do
        journal "$1" "mutator start"
        "$bin/brut-mutator" --out "$1" --tag "$tag" --parallel 8
        m=$?
        journal "$1" "mutator exit $m"

        case "$m" in
            0) return 0 ;;
            4)
                [ "$2" = 1 ] && stop "$1 waits for Mutator answers, but brut-mutator found nothing to answer"
                return 0
                ;;
            2) stop "brut-mutator refused to answer for $1 (exit 2)" ;;
            *) say "brut-mutator failed for $1 (exit $m), try $attempt of $tries; the answers written so far stay" ;;
        esac

        [ "$attempt" -lt "$tries" ] && sleep "$pause"
    done

    stop "brut-mutator failed $tries times. $1 waits cleanly for Mutator answers (see $1/llm/mutator-errors.jsonl); run this script again to go on."
}

# brut-breed-ollama --verify on the run in $1.
verify() {
    local out code

    out="$("$bin/brut-breed-ollama" --out "$1" --verify 2>&1)"
    code=$?
    journal "$1" "verify exit $code"

    if [ "$code" -ne 0 ] || ! grep -q "report recomputed and equal" <<<"$out"; then
        stop "$1 fails --verify: $out
Section 8: the run is not counted. Delete it (rm -r $1) and run this script again."
    fi

    say "$1: $(tail -n 1 <<<"$out")"
}

# Breed arm $1 into DIR/$1-S with the fresh-run options after it, through every Mutator
# round, and verify it.
breed_arm() {
    local arm="$1"
    shift
    local dir="$runs/$arm-$seed"
    local cmd=(--out "$dir" "$@")
    local mutate=0 fresh_pause=0 rounds=0 last code

    if [ -e "$dir/report.json" ]; then
        say "$dir is finished; it is checked, not bred again"
        no_failed_calls "$dir"
        verify "$dir"
        return 0
    fi

    if [ -e "$dir" ]; then
        last="$(grep -E '^[0-9]+ breed ' "$dir/driver.log" 2>/dev/null | tail -n 1)"

        case "$last" in
            *" breed exit 3")
                say "$dir waits for Mutator answers; it goes on"
                no_failed_calls "$dir"
                cmd=(--out "$dir" --resume)
                mutate=1
                ;;
            *)
                stop "$dir exists and is not finished, and its last process did not pause for Mutator answers (journal: ${last:-no breeding process}). Section 8: such a run is not resumed. Delete it (rm -r $dir) and run this script again."
                ;;
        esac
    else
        mkdir -p "$dir" || stop "cannot create $dir"
    fi

    while :; do
        if [ "$mutate" = 1 ]; then
            mutator_round "$dir" "$fresh_pause"
            cmd=(--out "$dir" --resume)
        fi

        journal "$dir" "breed start ${cmd[*]}"
        "$bin/brut-breed-ollama" "${cmd[@]}"
        code=$?
        journal "$dir" "breed exit $code"
        no_failed_calls "$dir"

        case "$code" in
            0)
                [ -e "$dir/report.json" ] && break
                stop "$dir stopped without finishing (a STOP file or a time limit). Section 8: such a run is not resumed. Delete it (rm -r $dir) and run this script again."
                ;;
            3)
                rounds=$((rounds + 1))
                [ "$rounds" -le "$max_rounds" ] || stop "$dir still waits after $max_rounds Mutator rounds"
                mutate=1
                fresh_pause=1
                ;;
            *)
                stop "brut-breed-ollama ended $dir with exit $code. Section 8: the run is not counted. Delete it (rm -r $dir) and run this script again."
                ;;
        esac
    done

    verify "$dir"
}

# vote-control on run $1 with $2 samples per puzzle, until no call failed.
vote() {
    local attempt code

    for attempt in $(seq 1 "$tries"); do
        journal "$1" "vote start --samples $2"
        "$bin/vote-control" --run "$1" --temperature "$temperature" --samples "$2" >/dev/null
        code=$?
        journal "$1" "vote exit $code"

        case "$code" in
            0) return 0 ;;
            1) say "vote-control failed on $1 (exit 1), try $attempt of $tries; a repeat asks the failed calls again" ;;
            *) stop "vote-control refused $1 (exit $code)" ;;
        esac

        [ "$attempt" -lt "$tries" ] && sleep "$pause"
    done

    stop "vote-control failed $tries times on $1; run this script again to retry"
}

# The call matching for run $1: prints "samples N" or "matched N_up N_down".
matching() {
    "$bin/breeding-vs-vote" --match "$1" --frozen "$frozen" || stop "the call matching refused $1"
}

# Keep the vote of run $1 with $3 samples as vote-$2.json; "-" keeps none.
keep() {
    local file="$1/vote-$2.json"

    [ "$3" = "-" ] && return 0

    if [ -e "$file" ]; then
        say "$file was kept before; the call matching checks it"
        return 0
    fi

    vote "$1" "$3"
    cp "$1/vote-control.json" "$file.part" && mv "$file.part" "$file" || stop "cannot write $file"
    journal "$1" "kept $file with $3 samples"
}

# Arm V on run B-S: sample until the matching is decided, then keep N_down and N_up.
vote_arm() {
    local dir="$runs/B-$seed"
    local asked=0 line n_up="" n_down="" round

    for round in 1 2 3 4 5 6 7 8; do
        line="$(matching "$dir")" || exit 1

        case "$line" in
            "samples "*)
                n="${line#samples }"
                [ "$n" -gt "$asked" ] || stop "the call matching asks for $n samples again on $dir"
                asked="$n"
                say "the vote of $dir needs $n samples per puzzle on file"
                vote "$dir" "$n"
                ;;
            "matched "*)
                read -r _ n_up n_down <<<"$line"
                break
                ;;
            *) stop "unexpected answer of the call matching: $line" ;;
        esac
    done

    [ -n "$n_up" ] || stop "the call matching of $dir did not settle"
    say "$dir: N_up $n_up, N_down $n_down"
    keep "$dir" down "$n_down"
    keep "$dir" up "$n_up"

    line="$(matching "$dir")" || exit 1
    [ "$line" = "matched $n_up $n_down" ] || stop "the call matching of $dir changed to: $line"
}

breed_arm B --config "$frozen/config-B.json" --seed "$seed"
vote_arm
breed_arm L --config "$frozen/config-L.json" --seed "$seed" --selection lexicase
"$bin/breeding-vs-vote" --panels "$runs/B-$seed" --frozen "$frozen" || stop "panels.json for $runs/B-$seed failed"
say "seed $seed done in $(( $(date +%s) - started )) s; the analysis of every pair in $runs follows"
"$bin/breeding-vs-vote" --runs "$runs" --frozen "$frozen" "${pilot[@]}" || stop "the analysis refused $runs"
