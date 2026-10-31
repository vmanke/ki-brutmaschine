# Data of the step 1 counted Runs

This folder holds the data of the counted Runs of the step 1 circuit experiment, which ran by its frozen pre-registration [docs/PREREG_STEP1_CIRCUITS.md](../PREREG_STEP1_CIRCUITS.md); the run directories themselves are not in the repository. The experiment is a positive control of the breeder with the simulated designer; it says nothing about real language models. The report is [docs/STEP1_CIRCUITS.md](../STEP1_CIRCUITS.md), section 7 (German: [docs/SCHRITT1_SCHALTUNGEN.md](../SCHRITT1_SCHALTUNGEN.md), Abschnitt 7). The data of the pilot Runs, which never count, are in [docs/pilot-step1/](../pilot-step1/README.md).

## Result

The seeds 1 to 27 ran in both arms, from 2026-10-30T22:32:35Z to 2026-10-31T12:44:46Z. The rule stopped after pair 27, by stop rule 2 of PREREG section 8: the primary claim is proven, and E2 has crossed a threshold.

- **Primary claim (ticket 03): proven at pair 15.** E1 first reached 100 at pair 15, with 3^14/2^15 = 145.9646. At the stop, E1 = 3^24/2^27 = 282429536481/134217728 = 2104.2640; arm A passed in 24 of the 27 seeds.
- **Secondary claim (selection): proven at pair 27.** At the stop, E2 = 3^20/2^25 = 3486784401/33554432 = 103.9143. Among the 25 untied pairs, arm A passed first 20 times and arm R 5 times; pairs 10 and 19 were ties, in which neither Run passed.

## What the folder holds

| File | What it is |
|---|---|
| `counted.json` | One entry per counted Run and one per pair, built by `make_counted.py`. Its `notes` explain every field. |
| `make_counted.py` | Builds `counted.json` from the files of the Runs and the logs in this folder, and checks them against each other. |
| `step1-driver.sh` | The driver that ran PREREG section 6, steps 3 to 7, at both of its starts. Only line 7, the local base folder, differs from the script that ran. It writes "done" only after `brut-breed` exits with code 0 and `report.json` exists. It writes "verify ok" only when the last line of `--verify` contains "report recomputed and equal; rediscovery recomputed and equal", which `brut-breed` prints at the end of that line. At any failed check it writes a line that starts with "STOPPED:" and ends; `driver.log` has no such line. It sets `RAYON_NUM_THREADS` only for the thread check. |
| `analysis.txt` | The output of the frozen analysis `examples/circuit_positive_control.rs` on the 27 pairs, after pair 27. |
| `recount.txt` | The output of the independent recount `reference_rule.py --runs runs` (PREREG section 6, step 7). |
| `hashes.txt` | The SHA-256 of `config.json`, `events.jsonl`, `snapshot.json`, `report.json` and `rediscovery.json` of every counted Run, appended by the driver after each pair. |
| `deviations.txt` | Deviation 1, the container restart. |
| `acceptance.txt` | The acceptance check before seed 1 (PREREG section 6, step 2): the frozen analysis on the pilot pairs 301 to 309. |
| `acceptance-recount.txt` | The same check by `reference_rule.py --pilots`. |
| `driver.log` | The driver's log: the start and end of every Run and the result of its `--verify`, the thread check, the restart, the verdicts and the recount. |
| `step1-build-test.txt` | The build and the three test commands of PREREG section 6, step 1, at the frozen commit. |

## Frozen commit and toolchain

- **Commit.** `15cb15100a39db9d815674b0778ab6be30932e56`, the squash merge of pull request #19 on 2026-10-30 at 21:50:56 UTC. It freezes the pre-registration. The driver logged this commit at both of its starts, and seed 1 started at 22:32:35 UTC.
- **Toolchain.** rustc 1.97.0 (2d8144b78 2026-07-07) on x86_64 Linux (Ubuntu 24.04.5 LTS), as in the pilot. The program is `cargo build --release --bin brut-breed` at that commit. There, `cargo test --lib` passed 610 tests (none of the 7 ignored tests is a calibration test), `cargo test --test circuits` 18 and `cargo test --example circuit_positive_control` 31 (`step1-build-test.txt`).
- **Threads.** The driver did not set `RAYON_NUM_THREADS` for the counted Runs, so rayon used one thread per CPU; the container reported 4 CPUs after the experiment. The thread check after pair 1 ran seed 1 of both arms again at 1 and at 4 threads, and the five hashed files were byte-equal to those of the counted Runs. We compared the 20 files again after the experiment, with the same result. For the other seeds we assume that the files do not depend on the thread count either. The wall time does depend on it, and the pilot ran at 2 threads.
- **One process at a time.** The driver ran the Runs of a pair one after the other, then `--verify` on both, then the frozen analysis on all finished pairs, which named the next seed. It ran the analysis as the debug build of the example, the binary that `cargo run --example circuit_positive_control` builds.

## Reproduce one pair

Build the frozen commit in a worktree of its own, so that this folder stays as it is. Then run both arms of a seed in that worktree, one process at a time, here seed 1. Start from the root of the repository:

```sh
REPO=$PWD
git worktree add ../brut-frozen 15cb15100a39db9d815674b0778ab6be30932e56
cd ../brut-frozen
cargo build --release --bin brut-breed
B=target/release/brut-breed
$B --out runs/A-1 --domain circuits --search 33 --select 22 --children 8 \
  --generations 1000 --max-calls 13000000 --seed 1
$B --out runs/R-1 --domain circuits --search 33 --select 22 --children 8 \
  --generations 1000 --max-calls 13000000 --seed 1 --selection uniform
$B --out runs/A-1 --verify
$B --out runs/R-1 --verify
grep ' runs/[AR]-1/' "$REPO/docs/step1-counted/hashes.txt" | sort -u | sha256sum -c -
```

Each `--verify` must end with "report recomputed and equal; rediscovery recomputed and equal", and `sha256sum` must print OK for all ten files. In the experiment a Run took 4.87 to 11.18 min in arm A and 4.33 to 5.92 min in arm R. On another toolchain or platform a floating-point function may round its last bit differently, and then the files may differ (PREREG section 2); such Runs are a deviation. A Run that is interrupted is not resumed: delete its directory and run it again from scratch (PREREG section 6, step 3).

## Recount

PREREG section 2 freezes the analysis and `reference_rule.py` at the frozen commit, so the recount runs in its worktree. In `../brut-frozen`, with all 54 Run directories in its `runs/`:

```sh
python3 -I .scratch/universum-erfinden/sequential-rule/reference_rule.py --runs runs
cargo run --example circuit_positive_control -- \
  --a $(printf 'runs/A-%d ' $(seq 1 27)) --r $(printf 'runs/R-%d ' $(seq 1 27))
```

The first prints `recount.txt`, the second `analysis.txt`. Both give the same 27 rows, the same stop after pair 27 and the same verdicts.

## Rebuild counted.json

From the root of the repository, with the Runs in `../brut-frozen/runs`:

```sh
python3 -I -B docs/step1-counted/make_counted.py --runs ../brut-frozen/runs \
  --logs docs/step1-counted --out /tmp/counted.json
cmp /tmp/counted.json docs/step1-counted/counted.json
```

The script reads only the files of the Runs and the logs in this folder; it needs no build. It recomputes R_seed, D, the pass times, X, Y, E1 and E2 from `rediscovery.json` in exact integers, and stops with an error when a number differs from `analysis.txt` or `recount.txt`, a file's hash from `hashes.txt`, or a setting from PREREG section 6. The wall times and the times of the driver's "verify ok" lines come from `driver.log`, so Runs repeated at the frozen commit whose files match `hashes.txt` give the same `counted.json`. On the original Runs it took 13 s.

## The run directories

The run directories are not in the repository. The files of the 54 counted Runs hold 3 440 979 110 bytes, about 3.4 GB (3.2 GiB; 99.9 % of it `events.jsonl` and `snapshot.json`), and those of the four Runs of the thread check another 235 484 118 bytes. `hashes.txt` allows checking a rerun, as above.

`hashes.txt` has 280 lines: one for each of the 270 files (27 pairs, 2 arms, 5 files), and pair 1's ten lines a second time. After the container restart (deviation 1), the driver started again with pair 1, found it finished, verified it again and appended its hashes again; both copies are equal. After the experiment we checked all 270 hashes against the files again, and all match.

## Deviation 1

A container restart between 2026-10-31T01:18:49Z and about 09:11Z interrupted Run A-12, which had started at 01:18:49Z and had no `report.json`. Pairs 1 to 11 were finished and verified. The driver started again at 09:20:01Z. Following PREREG section 6, step 3, it deleted the unfinished Run at 09:20:09Z and ran it again from scratch, on the same seed, into a fresh directory; the Run ended at 09:27:40Z. No finished pair changed or ran again. `deviations.txt` is the note written at the restart. It gives 09:20:04Z for the restart, which is the time of R-1's second check in `driver.log`; the times above come from `driver.log`.
