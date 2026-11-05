# Screen of arm W: the result

Status: done on 2026-11-03. Verdict by the fixed rule: **stop**. Criteria 1, 2 and 3 fail.

This is the screen of `issues/02-screen-arm-w.md`, run as that ticket says at commit
`65ec9e39f31daaa4b463030985cec90260eba716`. Its criteria were fixed before the first Run, and
they are unchanged here. Seeds 913 to 920 never count, so no claim follows from these numbers.

## How it ran

- **Program.** `cargo build --release --bin brut-breed` at commit `65ec9e3` (the build of ticket
  01, pull request #52, on main `4d3f536`), with rustc 1.97.0 (2d8144b78 2026-07-07) on x86_64
  Linux. The binary was copied out of the shared target directory before the first Run. Every
  Run and every `--verify` used that copy. Its SHA-256 is
  `d967532077671e8f9a6309cefc8308c9fb5d5d3b71daed0557b2742e39dca2d3`.
- **Settings.** `--domain circuits --search 33 --select 22 --children 8 --generations 1000
  --max-calls 13000000 --seed <s>`, the settings of the pilot. Arms W, W0 and WF take
  `--config`; arm N has no Soul. `config-W.json` is the `config.json` of a pilot Run of arm S with
  `"variation": "prefer"` in its `soul` object. The binary wrote that `config.json` for seed 901
  with the pilot's flags, and it equals `docs/pilot-step2/driver/config-E.json` except for
  `"strengths"`. `config-W0.json` adds `"copy": false` and `"imprint": false`; `config-WF.json`
  has `"variation": "fixed"`. All three are in `analysis/driver/`.
- **Order and threads.** The queue of the ticket: for each seed from 913 to 920, W, W0, N and
  WF. Two Runs ran at a time, each with `RAYON_NUM_THREADS=2` and `nice -n 5`, from
  2026-11-02T23:49Z to 2026-11-03T02:29Z.
- **After each Run.** `brut-breed --out <dir> --verify`, then the SHA-256 of the five record
  files, then `xz -6 -T1` on `events.jsonl` and `snapshot.json`. The scripts are in
  `analysis/driver/`, with the driver's log and the hashes.
- **Analysis.** `analysis/screen_w.py` (Python standard library, `python3 -I`) ran once, after
  all 32 Runs had passed `--verify`. It reads each `events.jsonl.xz` once through `lzma`, and
  each Run's `config.json`, `report.json` and `rediscovery.json`. It takes the pass time and
  the divergence generation from `.scratch/seele-und-kappa/rule/reference_rule.py`, as
  `diagnose.py` does. Its docstring fixes every definition the ticket leaves open. Its output
  is `analysis/screen_w.txt` and `analysis/screen_w.json`.

## Every Run passed `--verify`

All 32 Runs ran 1000 generations to the end ("all generations done"), with the settings of their
arm. All 32 `--verify` lines say "hash chain intact; checkpoint ... matches; settings match;
report recomputed and equal; rediscovery recomputed and equal". No Run came near the call cap.
The full SHA-256 of the five record files of every Run are in `analysis/screen_w.txt` and
`analysis/driver/hashes.txt`.

| Run | Pass time | Calls | Minutes | Events | Head of the log |
|---|---|---|---|---|---|
| W-913 | never | 1 029 854 | 9.7 | 33 949 | `17a6d7685fa27e9e…` |
| W0-913 | 70 | 972 517 | 9.4 | 28 869 | `172552a391fd7428…` |
| N-913 | 41 | 1 017 648 | 8.4 | 45 893 | `83a6d591fdf3f06b…` |
| WF-913 | 54 | 1 015 339 | 10.8 | 47 319 | `00a468e003f85183…` |
| W-914 | 980 | 1 048 499 | 8.4 | 23 307 | `0428bb6baa75f31d…` |
| W0-914 | 638 | 999 504 | 9.1 | 27 915 | `c86ddf21509427da…` |
| N-914 | 126 | 1 037 795 | 6.7 | 37 937 | `548e253c88d89021…` |
| WF-914 | 192 | 1 018 559 | 9.1 | 28 591 | `4a10c6e63fe19eb3…` |
| W-915 | 150 | 1 027 908 | 8.7 | 39 829 | `1577f46736b22101…` |
| W0-915 | 171 | 1 045 975 | 9.2 | 24 284 | `e0e4553b07b9c1be…` |
| N-915 | 485 | 1 072 670 | 6.0 | 24 904 | `293fce011123074d…` |
| WF-915 | 13 | 1 301 846 | 12.2 | 51 569 | `34c98c4196d0b6fc…` |
| W-916 | 47 | 1 060 627 | 10.4 | 40 568 | `c139d174e789d9b7…` |
| W0-916 | 210 | 990 707 | 9.9 | 35 692 | `891a304ef61f2de5…` |
| N-916 | 913 | 1 042 427 | 7.9 | 38 583 | `07f91b692d7989e8…` |
| WF-916 | 22 | 1 017 531 | 11.0 | 47 201 | `688d0129f93c5940…` |
| W-917 | 387 | 1 101 006 | 8.9 | 33 982 | `f58f8a1318b438f6…` |
| W0-917 | 46 | 1 028 544 | 10.2 | 39 949 | `a83d2e934ee7f6ac…` |
| N-917 | 137 | 1 080 766 | 7.0 | 31 092 | `3dc23226ad1aa5f8…` |
| WF-917 | 147 | 1 013 810 | 9.6 | 41 867 | `332147eb76159f33…` |
| W-918 | never | 1 006 357 | 9.6 | 39 068 | `b25d8f447999ca15…` |
| W0-918 | 926 | 1 036 885 | 9.0 | 29 251 | `91a42676c34f6ecc…` |
| N-918 | 396 | 970 571 | 5.4 | 26 523 | `d15947b577af888d…` |
| WF-918 | 90 | 1 052 290 | 10.4 | 47 408 | `2cb7b5d57868e3fd…` |
| W-919 | 166 | 1 251 283 | 11.1 | 43 095 | `7c0fcc32dcaa34b2…` |
| W0-919 | 133 | 1 234 969 | 12.3 | 41 264 | `a4e9f04a156f0200…` |
| N-919 | 84 | 1 037 584 | 7.2 | 35 388 | `e8d667576651196f…` |
| WF-919 | 5 | 1 162 177 | 11.8 | 35 902 | `bb37dbf70511add2…` |
| W-920 | never | 999 122 | 11.3 | 47 601 | `df12612b470dc5e8…` |
| W0-920 | 206 | 1 023 931 | 9.4 | 37 182 | `fe79f28a08b983f9…` |
| N-920 | 235 | 1 098 778 | 6.6 | 27 258 | `758ef7ea8ed938dc…` |
| WF-920 | 60 | 1 026 438 | 10.4 | 44 192 | `a575fb16543a8e1d…` |

## Per seed

T is the pass time of step 1; "never" means no pass by generation 1000. The divergence
generation of two arms is the first generation whose newborns differ. "First change" is the
first generation whose `variation_changed` is above 0. The preference is the mean of
1 + cos(2π p / 65536) for Linear, Phase and Hadamard over the Workers of the parents chosen in
generations 41 to 50.

| Seed | T_W | T_W0 | T_N | T_WF | W-N | W-W0 | WF-W | WF-N | First change W | W0 | WF | Preference W | W0 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 913 | never | 70 | 41 | 54 | 1 | 2 | 1 | 1 | 1 | 1 | 1 | 0.800 | 0.974 |
| 914 | 980 | 638 | 126 | 192 | 5 | 4 | 5 | 5 | 5 | 4 | 5 | 0.589 | 1.005 |
| 915 | 150 | 171 | 485 | 13 | 2 | 2 | 1 | 1 | 1 | 1 | 1 | 1.354 | 1.022 |
| 916 | 47 | 210 | 913 | 22 | 1 | 2 | 1 | 1 | 1 | 1 | 1 | 0.707 | 1.053 |
| 917 | 387 | 46 | 137 | 147 | 1 | 2 | 1 | 1 | 1 | 1 | 1 | 0.778 | 1.034 |
| 918 | never | 926 | 396 | 90 | 1 | 2 | 1 | 1 | 1 | 1 | 1 | 0.689 | 0.970 |
| 919 | 166 | 133 | 84 | 5 | 1 | 2 | 1 | 2 | 1 | 1 | 2 | 1.282 | 0.995 |
| 920 | never | 206 | 235 | 60 | 1 | 2 | 1 | 1 | 1 | 1 | 2 | 1.264 | 0.998 |

Arm W passed by generation 1000 on 5 of 8 seeds; arms W0, N and WF passed on all 8.

## The criteria

1. **Early: failed.** W parts from N before either Run passes on all 8 seeds; 7 were needed.
   The weights change a choice in generation 1 or 2 on 7 of 8 seeds; all 8 were needed. On
   seed 914 arm W made no choice of unequal weights in generations 1 to 4. Its first choices
   came in generation 5 (6 choices, 2 changed).
2. **Self-adaptation: failed.** The mean preference is above 1.2 in W on 3 of 8 seeds (915,
   919, 920); 6 were needed. It is below 1.1 in W0 on all 8 seeds; 6 were needed. On the other
   5 seeds W's mean lies below 1, between 0.589 and 0.800.
3. **Protection: failed.** Pooled before the first Worker with all three families, W lost a
   family in 133 of the 305 draws that changed a promising Worker, 43.6 %; at most 40 % was
   needed. W0 lost one in 146 of 345 (42.3 %) and N in 61 of 119 (51.3 %). W lies 1.3 points
   above W0 and 7.7 points below N; it needed to lie at least 10 points below both. Every arm
   had more than 30 such draws.
4. **Direction, reported.** W passed before N on 2 seeds (915 and 916) and N before W on 6. W
   passed before W0 on 2 seeds (915 and 916) and W0 before W on 6. There was no tie.

**Verdict.** Criteria 1 and 2 fail, so the fixed rule says stop. Criterion 3 fails too, and N
passes before W on 6 seeds, more than the 4 that a go allows.

The draws on a parent with a promising Worker, pooled over the seeds, for description: of
1140 such draws in W, 305 changed a promising Worker (26.8 %) and none made the Worker with all
three. W0 had 1062 draws, 345 changes (32.5 %) and 4 Workers with all three; N had 419 draws,
119 changes (28.4 %) and 2. Each seed's counts are in `analysis/screen_w.txt`.

## Arm WF, for description only

These numbers are no part of the criteria.

- **Pass times.** WF passed before W on all 8 seeds. It passed before N on 5 seeds (915, 916,
  918, 919, 920) and after N on 3 (913, 914, 917).
- **Protection.** WF lost a family in 10 of the 167 draws that changed a promising Worker,
  6.0 %, against 43.6 % in W and 51.3 % in N. Of its 316 draws on a parent with a promising
  Worker, 167 changed a promising Worker (52.8 %), and 3 made the Worker with all three.
- **Divergence.** WF parted from W in generation 1 on 7 seeds and in generation 5 on seed
  914. It parted from N in generation 1 on 6 seeds, in generation 2 on seed 919 and in
  generation 5 on seed 914.

A reading, not a test: on 5 of 8 seeds W's parents preferred Linear, Phase and Hadamard less
than a neutral Soul does, so there its weights helped to overwrite these families. Spec 4.1
names this risk: new preferences arise slowly, and seed Souls that disfavour a family can make
W slower than N. WF, whose fixed preference favours the three families, protected them.

## Deviations

- **Shared processor.** During part of the screen the grid screen of ticket 03 ran on the
  same 4 cores, two Runs at a time. So the wall times are longer than alone,
  and they cannot be compared with the pilot's. The record files do not depend on the load or
  on the thread count (spec 15).
- **Seed 913 was seen before.** For ticket 01, four smoke Runs of 60 generations ran on seed 913
  in arms W, W0, WF and N, with the same settings otherwise. Their generations 1 to 60 equal
  those of the screen's seed-913 Runs. From them only the per-generation `variation_changed` of
  W and WF was read and reported, before this screen was ordered. They showed a change in
  generation 1 on seed 913 in both arms.
- **Definitions fixed by the script.** The ticket leaves some definitions open. The script's
  docstring fixes them, written while the Runs were running and before any of their output was
  read. A parent counts once for each child it has in generations 41 to 50, and both parents
  of a crossover count. A promising Worker names exactly two of the three families. The draw
  that makes the first Worker with all three counts. "Before either Run passes" means a
  divergence generation strictly below both pass times.
- **Tool test.** Before the screen's output was read, the script ran on four throwaway Runs of
  80 generations on seed 4242, which is on no list. An independent recount there matched its
  draw counts and its mean preference exactly. Those Runs are no part of the screen.
- **Checks after the analysis.** After its single run, two of its numbers were recomputed by
  separate code, as checks: the state lines of W-914 in generations 0 to 6 and the mean
  preference of W-919 (1.2821). Both agree.
- **`xz -T1`.** The compression used `xz -6 -T1`, one thread, as the pilot's `after_run.sh`
  does; the ticket says `xz -6`.
