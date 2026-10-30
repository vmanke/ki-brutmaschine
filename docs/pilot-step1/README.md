# Step 1 pilot runs

These are pilot runs of the step 1 circuit experiment, not counted runs. Pilot seeds never count: 101 to 110, 201 to 210, 301 to 310, 401 and 402 (docs/PREREG_STEP1_CIRCUITS.md, section 10). The counted runs use the seeds 1 to 100, then 111 to 130, in this order, and have not been run; the list skips the pilot seeds 101 to 110. This folder holds the first pilot, on seeds 101 and 102, with its data in `pilot.json`. The later batches, which chose the design of the counted experiment, are described below with the pass time of every run.

## First pilot: seeds 101 and 102

We ran arm A (Darwin Gödel selection) and arm R (`--selection uniform`) on both seeds for 60 generations. The default cap of 20 000 calls stopped arm A on seed 101 after generation 20, so every run uses `--max-calls 200000`. A horizon probe repeats each run with 200 generations. `pilot.json` holds the table per run (calls, wall time, R_seed, R_bred, D, the calibration gate, the first generation per target, the `--verify` result, the observables of every 10th generation) and explains its fields in `notes`.

To reproduce, build the commit that holds this file with `cargo build --release --bin brut-breed` and run the commands below one at a time, with `brut-breed` standing for `target/release/brut-breed`. The capped run is the arm A command for seed 101 without `--max-calls`, and the probe uses `--generations 200 --max-calls 1000000` in the directories `probe200-A-101` and so on. Then `brut-breed --out <dir> --verify` checks each run, and `cargo run --example circuit_positive_control -- --a A-101 A-102 --r R-101 R-102` prints the per-seed numbers and makes no claim, since pilot seeds never count. We built with rustc 1.97.0 on x86_64 Linux (Ubuntu 24.04), where each event log ends in the head hash that `pilot.json` records; on another platform a floating-point function may round its last bit differently, and then the hashes differ. The thread count changes only the wall time. The runs were first made with an earlier state of the step 1 work, before the review fixes. With the program of the commit that holds this file we ran all nine runs again: their `config.json`, `events.jsonl`, `snapshot.json`, `report.json` and `rediscovery.json` are byte-equal to the pilot's, and all nine pass its `--verify`.

```sh
for s in 101 102; do
  brut-breed --out A-$s --domain circuits --search 33 --select 22 --children 8 \
    --generations 60 --seed $s --max-calls 200000
  brut-breed --out R-$s --domain circuits --search 33 --select 22 --children 8 \
    --generations 60 --seed $s --max-calls 200000 --selection uniform
done
```

## Later pilot batches (uncounted)

After the first pilot we ran four more batches. They are uncounted. They are the data that chose the design of the counted experiment: the ceiling of 1000 generations, the call cap of 13 000 000 and the planning figures of the sequential rule (docs/PREREG_STEP1_CIRCUITS.md, sections 6, 8 and 10). Because they chose the design, their seeds must never be counted. Every run has the settings of the counted runs except the seed, `--generations` and `--max-calls`, for example arm R on seed 301:

```sh
brut-breed --out R-301 --domain circuits --search 33 --select 22 --children 8 \
  --generations 1000 --max-calls 13000000 --seed 301 --selection uniform
```

The run directories are not in the repository. `.scratch/universum-erfinden/sequential-rule/pilot.json` holds the values that `pilot.py` there read from their `config.json`, `report.json`, `rediscovery.json` and `timings.jsonl`, and `pilot.txt` there is its summary.

| Directory | Seeds | `--generations` | `--max-calls` | Runs |
|---|---|---|---|---|
| `pilot10` | 101 to 110 | 200 | 2 600 000 | both arms on every seed |
| `pilot-base-201` | 201 to 210 | 200 | 2 600 000 | both arms on every seed |
| `pilot-1000` | 301 to 310 | 1000 | 13 000 000 | arm A on every seed, arm R on 301 to 309 |
| `pilot-2000` | 401 and 402 | 2000 | 26 000 000 | arm A on both seeds, arm R on 401 |

The pass time T of a run is the first generation in which its bred Cascades solve more than half of the targets its seed Cascades miss (PREREG section 7). The seed Cascades solve swap and cs in every run, so T is the first generation with 5 of the other 9 targets. In the tables, "–" means no pass by the run's last generation (200, 1000 or 2000); for the 200-generation batches a later pass is unknown.

Seeds 101 to 110, 200 generations (`pilot10`):

| Seed | 101 | 102 | 103 | 104 | 105 | 106 | 107 | 108 | 109 | 110 |
|---|---|---|---|---|---|---|---|---|---|---|
| T, arm A | 32 | – | – | – | – | – | – | – | – | – |
| T, arm R | – | – | – | – | – | – | – | – | – | 48 |

Seeds 201 to 210, 200 generations (`pilot-base-201`):

| Seed | 201 | 202 | 203 | 204 | 205 | 206 | 207 | 208 | 209 | 210 |
|---|---|---|---|---|---|---|---|---|---|---|
| T, arm A | 172 | – | 78 | – | – | 97 | – | 177 | – | 105 |
| T, arm R | – | – | – | – | – | – | – | – | 149 | 118 |

Seeds 301 to 310, 1000 generations (`pilot-1000`):

| Seed | 301 | 302 | 303 | 304 | 305 | 306 | 307 | 308 | 309 | 310 |
|---|---|---|---|---|---|---|---|---|---|---|
| T, arm A | – | 7 | 75 | 97 | 349 | 143 | 33 | 180 | 810 | incomplete |
| T, arm R | – | 380 | – | – | – | – | – | – | – | not run |

Seeds 401 and 402, 2000 generations (`pilot-2000`):

| Seed | 401 | 402 |
|---|---|---|
| T, arm A | – | 10 |
| T, arm R | 720 | not run |

- **Calibration.** In all 61 complete runs the seed Cascades solve exactly swap and cs, the calibration gate holds, and the run ended with all generations done. No run found an invention candidate or solved ch or fredkin.
- **A-310** was interrupted after generation 922. It is incomplete and left out of every figure.
- **Seeds 101 and 102 in `pilot10`** give the same calls and D as the first pilot's 200-generation runs; only the cap differs, and it does not bind.

What the batches set:

- **The ceiling.** By generation 200, arm A passed on 6 of the 20 seeds 101 to 110 and 201 to 210. At that rate the sequential rule fails with probability 1.000, so the ceiling could not stay at 200. 1000 is the longest horizon with data in both arms: 11 runs of arm A and 10 of arm R reached it.
- **The pass rates at 1000 generations.** Arm A passed in 9 of the 11 runs that reached generation 1000; a Kaplan-Meier estimate over all 31 complete runs of arm A gives 22/31 = 0.710. Arm R passed in 2 of 10 (Kaplan-Meier: 0.280). In the 10 pairs with both arms run to generation 1000, arm A passed first 8 times, arm R once (seed 401), and once neither passed (seed 301).
- **Calls.** The most calls by generation 1000 were 1 161 738 (A-401), below the bound of 12 723 412 for a counted run.
- **Wall time** at 2 threads, from `timings.jsonl`. A run to generation 1000 took 6.01 to 8.57 min in arm A (mean 7.38 min, 11 runs) and 4.54 to 6.12 min in arm R (mean 5.38 min, 10 runs). Both arms of a pair together took 10.57 to 14.31 min (mean 12.68 min):

| Pair | 301 | 302 | 303 | 304 | 305 | 306 | 307 | 308 | 309 | 401 |
|---|---|---|---|---|---|---|---|---|---|---|
| Minutes to generation 1000, A + R | 12.08 | 14.28 | 12.80 | 12.57 | 12.46 | 13.67 | 13.46 | 10.57 | 10.64 | 14.31 |

`python3 -I .scratch/universum-erfinden/sequential-rule/reference_rule.py --pilots <dir>`, with `<dir>` holding `pilot-1000` and `pilot10`, applies the sequential rule to these pilot pairs as an illustration; its output is `reference_rule.txt` beside it. On the pairs 301 to 309 the experiment would still be open after the ninth pair: E1 = 6561/512 and E2 = 6561/256 are both open, since neither has reached 100 or fallen to 1/20. On the pairs 101 to 110, where a pass after generation 200 is unknown, the primary claim would fail at the sixth pair. Neither is a result: pilot seeds never count.
