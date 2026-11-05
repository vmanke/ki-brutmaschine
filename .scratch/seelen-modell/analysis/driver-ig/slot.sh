#!/usr/bin/env bash
# One slot of the screen of ticket 06. The session ends each background task after 7200 s.
# driver.sh ran the queue in one such task and was stopped there, with 2 Runs unfinished.
# Now two slots run at a time, each in its own task, so at most 2 Runs run at a time, as before.
# A slot takes the next Run of the queue that has not passed --verify and that no slot has
# claimed, and runs it through run_one.sh, unchanged. Its first Run starts at once. It starts
# a further Run only if that Run can end within its task: the time since the slot started, plus
# 1.25 times the longest wall time of a finished Run, must stay within 7200 s minus 180 s.
# Otherwise the slot ends, and the session starts a new one. A Run directory without a claim
# was left by a stopped task: it is moved to killed/, and its Run starts again from generation 0.
# Usage: slot.sh NAME QUEUEFILE   (NAME: a label for driver.log, A or B; lines: ARM SEED)
set -u
ROOT=/path/to/ig-screen  # the local base folder; the only line that differs from the script that ran
log=$ROOT/logs/driver.log
slot=$1
queue=$2
limit=7200
margin=180
t0=$(date +%s)
started=0
mkdir -p "$ROOT/claims" "$ROOT/killed"
echo "$(date -u +%FT%TZ) slot $slot start queue=$queue" >>"$log"

while read -r arm seed; do
  [ -z "$arm" ] && continue
  name=$arm-$seed

  if grep -qF " verify $name exit=0 " "$log"; then
    continue
  fi

  if [ -e "$ROOT/claims/$name" ]; then
    continue
  fi

  if [ -e "$ROOT/STOP" ]; then
    echo "$(date -u +%FT%TZ) slot $slot STOPPED: STOP file before $name" >>"$log"
    exit 0
  fi

  free=$(df --output=avail -B1 / | tail -n 1)

  if [ "$free" -lt 1610612736 ]; then
    echo "$(date -u +%FT%TZ) slot $slot STOPPED: only $free bytes free before $name" >>"$log"
    exit 0
  fi

  longest=$(cat "$ROOT"/runs/*.wall | sort -g | tail -n 1)
  need=$(python3 -I -c "import math, sys; print(math.ceil(1.25 * float(sys.argv[1])))" \
    "$longest" </dev/null)
  left=$((limit - margin - ($(date +%s) - t0)))

  if [ "$started" -gt 0 ] && [ "$need" -gt "$left" ]; then
    echo "$(date -u +%FT%TZ) slot $slot end: $name may need $need s, $left s left" >>"$log"
    exit 0
  fi

  if ! mkdir "$ROOT/claims/$name" 2>/dev/null; then
    continue
  fi

  if [ -e "$ROOT/runs/$name" ]; then
    stamp=$(date -u +%Y%m%dT%H%M%SZ)
    mv "$ROOT/runs/$name" "$ROOT/killed/$name-$stamp"

    if [ -e "$ROOT/runs/$name.out" ]; then
      mv "$ROOT/runs/$name.out" "$ROOT/killed/$name-$stamp.out"
    fi

    echo "$(date -u +%FT%TZ) slot $slot moved $name to killed/$name-$stamp: a stopped task" \
      "left it unfinished; it runs again from the start" >>"$log"
  fi

  echo "$(date -u +%FT%TZ) slot $slot runs $name" >>"$log"
  started=$((started + 1))
  "$ROOT/driver/run_one.sh" "$arm" "$seed" </dev/null
done <"$queue"

echo "$(date -u +%FT%TZ) slot $slot end: no Run left to start" >>"$log"
