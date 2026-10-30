# Step 1 pilot runs

These are pilot runs of the step 1 circuit experiment, not counted runs. Seeds 101 and 102 are the pilot seeds of the pre-registration (docs/PREREG_STEP1_CIRCUITS.md): we do not count them, and they may set only `--generations` and `--max-calls` of the counted runs on seeds 1 to 10, which have not been run. We ran arm A (Darwin Gödel selection) and arm R (`--selection uniform`) on both seeds for 60 generations. The default cap of 20 000 calls stopped arm A on seed 101 after generation 20, so every run uses `--max-calls 200000`. A horizon probe repeats each run with 200 generations. `pilot.json` holds the table per run (calls, wall time, R_seed, R_bred, D, the calibration gate, the first generation per target, the `--verify` result, the observables of every 10th generation) and explains its fields in `notes`.

To reproduce, we build at commit d4a650e with `cargo build --release --bin brut-breed` and run the commands below one at a time, with `brut-breed` standing for `target/release/brut-breed`. The capped run is the arm A command for seed 101 without `--max-calls`, and the probe uses `--generations 200 --max-calls 1000000` in the directories `probe200-A-101` and so on. Then `brut-breed --out <dir> --verify` checks each run, and `cargo run --example circuit_positive_control -- --a A-101 A-102 --r R-101 R-102` prints the per-seed lines, with no claim for fewer than the 10 pre-registered seeds. We built with rustc 1.97.0 on x86_64 Linux (Ubuntu 24.04), where each event log ends in the head hash that `pilot.json` records; on another platform a floating-point function may round its last bit differently, and then the hashes differ. The thread count changes only the wall time. With the program of the later commit b3a99e1 we ran all nine runs again: their `config.json`, `events.jsonl`, `snapshot.json`, `report.json` and `rediscovery.json` are byte-equal to the pilot's. The main branch receives step 1 as new commits, so it need not hold d4a650e and b3a99e1; the tag `pilot-step1` of the repository keeps both.

```sh
for s in 101 102; do
  brut-breed --out A-$s --domain circuits --search 33 --select 22 --children 8 \
    --generations 60 --seed $s --max-calls 200000
  brut-breed --out R-$s --domain circuits --search 33 --select 22 --children 8 \
    --generations 60 --seed $s --max-calls 200000 --selection uniform
done
```
