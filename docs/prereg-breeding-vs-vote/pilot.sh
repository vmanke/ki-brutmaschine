#!/usr/bin/env bash
# pilot.sh: the pilot of docs/PREREG_BREEDING_VS_VOTE.md, section 9, in one command, so
# that the VM runs only as long as the pilot needs.
#
# Usage, on the VM, from the repository root, after
#   (cd backend-ollama && cargo build --release):
#   docs/prereg-breeding-vs-vote/pilot.sh [--runs DIR]
#
# DIR is the runs directory (default: runs). The script
#   1. starts the model server (vllm-start.sh with PREREG_PILOT=1) in the background,
#      with its log in DIR/pilot/vllm.log, and waits until it answers on 127.0.0.1:8000,
#      at most 30 minutes (loading the weights);
#   2. runs the pilot seeds 901, 902 and 903 at the same time (run-seed.sh), so six
#      breeding runs share the card; each seed writes DIR/pilot/seed-S.log;
#   3. writes DIR/pilot/summary.txt: the weights revision, the analysis in pilot mode with
#      its gate line, and per run its minutes, Worker calls, failed Worker calls, the
#      longest Mutator prompt and the Mutator requests vLLM refused as too long;
#   4. stops the model server and says that the VM can be stopped.
#
# It never deletes anything and never stops the VM: stop the VM in the Nebius console.
# A seed that stops with a message (run-seed.sh exit 1) is named in the summary; the other
# seeds still finish. Run again, it goes on where the runs stand (run-seed.sh checks
# finished runs and continues paused ones).
#
# Environment: as for run-seed.sh (BRUT_BIN_DIR, VLLM_URL). PILOT_SERVER=external skips
# steps 1 and 4, for a server that already runs; the tests use it with stand-ins.
#
# Exit codes: 0 all three seeds finished, 1 a seed or the server stopped, 2 wrong usage.
set -uo pipefail

prog="pilot"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "$here/../.." && pwd)"
bin="${BRUT_BIN_DIR:-$root/backend-ollama/target/release}"
url="${VLLM_URL:-http://127.0.0.1:8000}"
seeds=(901 902 903)
runs="runs"

say() {
    echo "$prog: $*" >&2
}

while [ $# -gt 0 ]; do
    case "$1" in
        --runs) [ $# -ge 2 ] || { sed -n '2,8p' "$0" >&2; exit 2; }; runs="$2"; shift 2 ;;
        *) sed -n '2,8p' "$0" >&2; exit 2 ;;
    esac
done

out="$runs/pilot"
mkdir -p "$out" || { say "cannot create $out"; exit 1; }
start=$(date +%s)
server=""

stop_server() {
    if [ -n "$server" ] && kill -0 "$server" 2>/dev/null; then
        say "stopping the model server (pid $server)"
        kill "$server" 2>/dev/null
        wait "$server" 2>/dev/null
    fi
    server=""
}

trap stop_server EXIT

if [ "${PILOT_SERVER:-}" != external ]; then
    command -v curl >/dev/null || { say "curl is missing"; exit 1; }
    say "starting the model server; its log is $out/vllm.log"
    PREREG_PILOT=1 "$here/vllm-start.sh" >"$out/vllm.log" 2>&1 &
    server=$!
    waited=0

    until curl -sf "$url/v1/models" >/dev/null 2>&1; do
        if ! kill -0 "$server" 2>/dev/null; then
            server=""
            say "the model server ended before it answered; see $out/vllm.log"
            exit 1
        fi

        if [ "$waited" -ge 1800 ]; then
            say "the model server did not answer within 30 minutes; see $out/vllm.log"
            exit 1
        fi

        sleep 10
        waited=$((waited + 10))
    done

    say "the model server answers after ${waited} s"
fi

pids=()

for s in "${seeds[@]}"; do
    say "seed $s starts; its log is $out/seed-$s.log"
    "$here/run-seed.sh" --seed "$s" --runs "$runs" >"$out/seed-$s.log" 2>&1 &
    pids+=($!)
done

failed=()

for i in "${!pids[@]}"; do
    wait "${pids[$i]}"
    code=$?
    [ "$code" = 0 ] || failed+=("${seeds[$i]} (exit $code)")
done

stop_server
minutes=$(( ($(date +%s) - start + 59) / 60 ))

{
    echo "Pilot of docs/PREREG_BREEDING_VS_VOTE.md, section 9"
    echo "Finished: $(date -u +%Y-%m-%dT%H:%M:%SZ); wall time ${minutes} min for seeds ${seeds[*]}"

    if [ -f "$out/vllm.log" ]; then
        grep -h 'PILOT: the weights in the cache are revision' "$out/vllm.log" | head -n 1
    fi

    if [ "${#failed[@]}" -gt 0 ]; then
        echo "Seeds that stopped with a message: ${failed[*]} (see their logs)"
    else
        echo "All three seeds finished."
    fi

    echo
    echo "== The analysis in pilot mode"
    "$bin/breeding-vs-vote" --runs "$runs" --pilot 2>&1
    echo
    echo "== Per run"
    python3 -I -B - "$runs" "${seeds[@]}" <<'PY'
import json
import sys
from pathlib import Path

runs = Path(sys.argv[1])
seeds = sys.argv[2:]


def lines(path):
    try:
        with open(path, encoding="utf-8") as f:
            return [line for line in f if line.strip()]
    except OSError:
        return []


def minutes(run):
    stamps = []
    for line in lines(run / "driver.log"):
        head = line.split(" ", 1)[0]
        if head.isdigit():
            stamps.append(int(head))
    return f"{(max(stamps) - min(stamps)) / 60:.1f}" if len(stamps) >= 2 else "n/a"


def calls(run):
    try:
        with open(run / "report.json", encoding="utf-8") as f:
            return str(json.load(f).get("calls_used", "n/a"))
    except (OSError, ValueError, AttributeError):
        return "n/a"


def longest_prompt(run):
    best = None
    for line in lines(run / "llm" / "mutator-calls.jsonl"):
        try:
            tokens = json.loads(line)["response"]["prompt_tokens"]
        except (ValueError, KeyError, TypeError):
            continue
        if isinstance(tokens, int) and (best is None or tokens > best):
            best = tokens
    return "n/a" if best is None else str(best)


def too_long(run):
    return sum("maximum context length" in line for line in lines(run / "llm" / "mutator-errors.jsonl"))


print("run     minutes  worker calls  failed worker calls  longest Mutator prompt  too long")
for seed in seeds:
    for arm in ("B", "L"):
        run = runs / f"{arm}-{seed}"
        failed = len(lines(run / "llm" / "worker-errors.jsonl"))
        print(
            f"{arm}-{seed:<5} {minutes(run):>7}  {calls(run):>12}  {failed:>19}"
            f"  {longest_prompt(run):>22}  {too_long(run):>8}"
        )
PY
} >"$out/summary.txt" 2>&1

cat "$out/summary.txt"
echo
say "done after ${minutes} min. The summary is $out/summary.txt."
say "The VM is no longer needed for the pilot: stop it in the Nebius console."

[ "${#failed[@]}" -eq 0 ]
