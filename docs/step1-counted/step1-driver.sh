#!/usr/bin/env bash
# The counted runs of step 1 at the frozen commit: docs/PREREG_STEP1_CIRCUITS.md,
# section 6, steps 3 to 7. One process at a time. It stops at the stop pair of
# section 8, or at the first check that fails, and never runs a pair after the stop.
set -u

SP=${SP:?set SP to the folder that holds frozen/ and step1-log/}
F=$SP/frozen
LOGS=$SP/step1-log
BIN=$F/target/release/brut-breed
ANA=$F/target/debug/examples/circuit_positive_control
STATUS=$LOGS/status.txt

cd "$F" || exit 1
mkdir -p runs runs-threads "$LOGS"

say() {
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "$LOGS/driver.log"
}

stop() {
    say "STOPPED: $*"
    echo "stopped: $*" > "$STATUS"
    exit 1
}

# The counted seeds in their fixed order.
seeds() {
    seq 1 100
    seq 111 130
}

# One Run into a fresh directory. An unfinished directory is deleted and run again
# from scratch (step 3); a finished one is kept.
run_arm() {
    local arm=$1 seed=$2 out=$3 threads=${4:-}
    local selection=()

    if [ -d "$out" ] && [ ! -f "$out/report.json" ]; then
        say "removing the unfinished run $out"
        rm -rf "$out"
    fi

    if [ -f "$out/report.json" ]; then
        say "keeping the finished run $out"
        return 0
    fi

    if [ "$arm" = R ]; then
        selection=(--selection uniform)
    fi

    say "start $out${threads:+ with RAYON_NUM_THREADS=$threads}"

    if [ -n "$threads" ]; then
        RAYON_NUM_THREADS=$threads "$BIN" --out "$out" --domain circuits --search 33 \
            --select 22 --children 8 --generations 1000 --max-calls 13000000 \
            --seed "$seed" "${selection[@]}" > "$LOGS/$(basename "$out").stdout" 2>&1
    else
        "$BIN" --out "$out" --domain circuits --search 33 --select 22 --children 8 \
            --generations 1000 --max-calls 13000000 --seed "$seed" "${selection[@]}" \
            > "$LOGS/$(basename "$out").stdout" 2>&1
    fi

    local code=$?

    [ "$code" -eq 0 ] || stop "$out exited with code $code"
    [ -f "$out/report.json" ] || stop "$out has no report.json"
    say "done $out"
}

# Step 4: --verify must end with the pre-registered line.
verify() {
    local out=$1 last

    last=$("$BIN" --out "$out" --verify 2>&1 | tail -n 1)

    case "$last" in
        *"report recomputed and equal; rediscovery recomputed and equal"*)
            say "verify ok $out"
            ;;
        *)
            stop "verify failed on $out: $last"
            ;;
    esac
}

hashes() {
    (cd "$1" && sha256sum config.json events.jsonl snapshot.json report.json rediscovery.json) \
        | sed "s#  #  $1/#" >> "$LOGS/hashes.txt"
}

# Step 4: the frozen analysis on the finished pairs, the first n seeds of the list.
analyze() {
    local a=() r=() s

    for s in $(seeds); do
        if [ -f "runs/A-$s/report.json" ] && [ -f "runs/R-$s/report.json" ]; then
            a+=("runs/A-$s")
            r+=("runs/R-$s")
        else
            break
        fi
    done

    "$ANA" --a "${a[@]}" --r "${r[@]}" > "$LOGS/analysis.txt" 2>&1 \
        || stop "the analysis failed: $(tail -n 1 "$LOGS/analysis.txt")"
    cp "$LOGS/analysis.txt" "$LOGS/analysis-after-${#a[@]}-pairs.txt"
}

# Step 5: seed 1 of each arm twice more, at 1 and 4 threads; five files byte-equal.
thread_check() {
    local arm t f

    for arm in A R; do
        for t in 1 4; do
            run_arm "$arm" 1 "runs-threads/$arm-1-t$t" "$t"
            verify "runs-threads/$arm-1-t$t"
        done

        for f in config.json events.jsonl snapshot.json report.json rediscovery.json; do
            for t in 1 4; do
                cmp -s "runs/$arm-1/$f" "runs-threads/$arm-1-t$t/$f" \
                    || stop "thread check: runs/$arm-1/$f differs at $t thread(s)"
            done
        done
    done

    say "thread check ok: five files byte-equal at the default, 1 and 4 threads, both arms"
}

say "driver started at commit $(git -C "$F" rev-parse HEAD)"
echo "running" > "$STATUS"

next=$(seeds | head -n 1)

while :; do
    for arm in A R; do
        run_arm "$arm" "$next" "runs/$arm-$next"
    done

    verify "runs/A-$next"
    verify "runs/R-$next"
    hashes "runs/A-$next"
    hashes "runs/R-$next"
    analyze

    if [ "$next" -eq 1 ] && [ ! -f "$LOGS/thread-check.ok" ]; then
        thread_check
        touch "$LOGS/thread-check.ok"
    fi

    line=$(grep -o 'OPEN: run seed [0-9]* next' "$LOGS/analysis.txt" || true)

    if [ -z "$line" ]; then
        break
    fi

    next=$(echo "$line" | grep -o '[0-9]*')
    say "pair done; the analysis names seed $next next"
    echo "running, next seed $next" > "$STATUS"
done

say "the rule has stopped; verdicts:"
grep -E '^(Primary|Secondary)|^No claim' "$LOGS/analysis.txt" | tee -a "$LOGS/driver.log"

# Step 7: the independent recount.
python3 -I .scratch/universum-erfinden/sequential-rule/reference_rule.py --runs runs \
    > "$LOGS/recount.txt" 2>&1
say "recount exit $? (output in recount.txt)"
echo "finished" > "$STATUS"
