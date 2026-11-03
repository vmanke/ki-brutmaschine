#!/usr/bin/env bash
# Run the queue of the screen: at most 2 Runs at a time, each through run_one.sh. Before each
# start it checks the free disk space and stops starting Runs below 1.5 GB, or on a STOP file.
# Usage: driver.sh QUEUEFILE   (lines: ARM SEED)
set -u
ROOT=/path/to/w-screen   # the local base folder; the only line that differs from the script that ran
log=$ROOT/logs/driver.log
queue=$1
echo "$(date -u +%FT%TZ) driver start queue=$queue" >>"$log"
while read -r arm seed; do
  [ -z "$arm" ] && continue
  while [ "$(jobs -rp | wc -l)" -ge 2 ]; do sleep 5; done
  if [ -e "$ROOT/STOP" ]; then
    echo "$(date -u +%FT%TZ) STOPPED: STOP file before $arm-$seed" >>"$log"
    break
  fi
  free=$(df --output=avail -B1 / | tail -n 1)
  if [ "$free" -lt 1610612736 ]; then
    echo "$(date -u +%FT%TZ) STOPPED: only $free bytes free before $arm-$seed" >>"$log"
    break
  fi
  "$ROOT/driver/run_one.sh" "$arm" "$seed" &
  sleep 2
done <"$queue"
wait
echo "$(date -u +%FT%TZ) driver end" >>"$log"
