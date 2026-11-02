#!/usr/bin/env bash
# After the batch: the byte-equality reruns of seed 901 (one process at a time), then the
# final data steps.
ROOT=/path/to/pilot   # the local base folder; the only line that differs from the script that ran
cd "$ROOT" || exit 1
until grep -q 'driver end' logs/driver.log; do sleep 30; done
if grep -q 'STOPPED' logs/driver.log; then echo "batch stopped early; no reruns"; exit 1; fi
free=$(df --output=avail -B1 / | tail -n 1)
if [ "$free" -lt 1610612736 ]; then echo "only $free bytes free; no reruns"; exit 1; fi
./rerun.sh 901
./final.sh
