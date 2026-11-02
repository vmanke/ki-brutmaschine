# Selection study: results

Seeds 1..30 (30 per arm). Per run: up to 1000 generations, 8 children per generation, 40 search and 40 held-out tasks, finalists 3, call budget 12000 (the budget ends every run). Lexicase epsilon 0, tournament size 3. Simulated models (positive control).

## Part A: arms at one call budget

**Held-out rate of the best bred finalist (select phase)**

| arm | median [p25, p75] (min..max) |
|---|---|
| dgm | 0.625 [0.575, 0.700] (0.450..0.875) |
| lexicase | 0.700 [0.650, 0.750] (0.500..0.875) |
| tournament | 0.650 [0.575, 0.700] (0.350..0.925) |
| uniform | 0.650 [0.575, 0.700] (0.425..0.850) |

**Held-out rate of the best seed Cascade (same in every arm)**

| arm | median [p25, p75] (min..max) |
|---|---|
| dgm | 0.325 [0.300, 0.375] (0.225..0.450) |
| lexicase | 0.325 [0.300, 0.375] (0.225..0.450) |
| tournament | 0.325 [0.300, 0.375] (0.225..0.450) |
| uniform | 0.325 [0.300, 0.375] (0.225..0.450) |

**Best training rate in the archive**

| arm | median [p25, p75] (min..max) |
|---|---|
| dgm | 0.675 [0.600, 0.725] (0.400..0.825) |
| lexicase | 0.750 [0.675, 0.775] (0.550..0.900) |
| tournament | 0.675 [0.575, 0.750] (0.425..0.875) |
| uniform | 0.650 [0.625, 0.700] (0.425..0.825) |

**Training rate of the final incumbent**

| arm | median [p25, p75] (min..max) |
|---|---|
| dgm | 0.650 [0.600, 0.700] (0.400..0.825) |
| lexicase | 0.725 [0.675, 0.775] (0.550..0.900) |
| tournament | 0.650 [0.575, 0.725] (0.400..0.875) |
| uniform | 0.650 [0.575, 0.700] (0.425..0.825) |

**Held-out rate of the final incumbent**

| arm | median [p25, p75] (min..max) |
|---|---|
| dgm | 0.625 [0.550, 0.675] (0.400..0.875) |
| lexicase | 0.700 [0.625, 0.750] (0.500..0.875) |
| tournament | 0.625 [0.500, 0.700] (0.300..0.925) |
| uniform | 0.625 [0.575, 0.675] (0.425..0.850) |

**Generalisation gap of the final incumbent (train - held-out)**

| arm | median [p25, p75] (min..max) |
|---|---|
| dgm | 0.000 [-0.050, 0.100] (-0.200..0.300) |
| lexicase | 0.000 [-0.050, 0.125] (-0.225..0.300) |
| tournament | 0.050 [-0.050, 0.100] (-0.200..0.275) |
| uniform | 0.025 [-0.100, 0.100] (-0.175..0.300) |

**Slope of the gap per generation**

| arm | median [p25, p75] (min..max) |
|---|---|
| dgm | -0.001 [-0.004, 0.005] (-0.017..0.020) |
| lexicase | -0.002 [-0.007, 0.004] (-0.024..0.022) |
| tournament | 0.001 [-0.001, 0.005] (-0.025..0.027) |
| uniform | -0.000 [-0.006, 0.003] (-0.017..0.016) |

**Distinct behaviours in the final archive**

| arm | median [p25, p75] (min..max) |
|---|---|
| dgm | 60.000 [45.000, 70.000] (39.000..88.000) |
| lexicase | 47.000 [41.000, 54.000] (24.000..66.000) |
| tournament | 56.000 [48.000, 65.000] (28.000..94.000) |
| uniform | 68.000 [61.000, 73.000] (41.000..94.000) |

**Behaviours per archive member**

| arm | median [p25, p75] (min..max) |
|---|---|
| dgm | 0.630 [0.500, 0.685] (0.361..0.881) |
| lexicase | 0.605 [0.488, 0.667] (0.316..0.789) |
| tournament | 0.598 [0.457, 0.652] (0.259..0.810) |
| uniform | 0.565 [0.519, 0.612] (0.331..0.702) |

**Mean entropy of the parent draw (bits)**

| arm | median [p25, p75] (min..max) |
|---|---|
| dgm | 4.766 [4.678, 4.884] (4.226..5.172) |
| lexicase | 2.945 [2.728, 3.170] (1.743..3.711) |
| tournament | 4.677 [4.565, 4.781] (4.311..5.211) |
| uniform | 5.588 [5.497, 5.588] (5.296..5.831) |

**Generations completed**

| arm | median [p25, p75] (min..max) |
|---|---|
| dgm | 13.000 [12.000, 14.000] (9.000..16.000) |
| lexicase | 10.000 [10.000, 11.000] (9.000..12.000) |
| tournament | 13.000 [12.000, 14.000] (10.000..19.000) |
| uniform | 16.000 [15.000, 16.000] (13.000..19.000) |

**Cascades evaluated on the search panel**

| arm | median [p25, p75] (min..max) |
|---|---|
| dgm | 4000.000 [3680.000, 4320.000] (2720.000..4960.000) |
| lexicase | 3040.000 [3040.000, 3360.000] (2720.000..3680.000) |
| tournament | 4000.000 [3680.000, 4320.000] (3040.000..5920.000) |
| uniform | 4960.000 [4640.000, 4960.000] (4000.000..5920.000) |

**Worker calls (budgeted)**

| arm | median [p25, p75] (min..max) |
|---|---|
| dgm | 10889.000 [10620.000, 11322.000] (9156.000..11681.000) |
| lexicase | 10880.000 [10576.000, 11172.000] (9942.000..11845.000) |
| tournament | 11082.000 [10717.000, 11273.000] (9764.000..11656.000) |
| uniform | 11153.000 [10788.000, 11327.000] (9903.000..11608.000) |

**Completion tokens**

| arm | median [p25, p75] (min..max) |
|---|---|
| dgm | 1109322.000 [921412.000, 1286471.000] (582316.000..1425210.000) |
| lexicase | 1226808.000 [1140598.000, 1317006.000] (1011043.000..2234114.000) |
| tournament | 1199440.000 [1021590.000, 1341806.000] (722274.000..2346148.000) |
| uniform | 910638.000 [774124.000, 955076.000] (643200.000..1189729.000) |

**Model cost**

| arm | median [p25, p75] (min..max) |
|---|---|
| dgm | 6.404 [5.807, 7.194] (4.297..10.304) |
| lexicase | 5.719 [5.419, 6.059] (4.696..9.717) |
| tournament | 5.986 [5.647, 6.880] (4.671..9.080) |
| uniform | 6.555 [5.934, 7.097] (5.412..10.892) |

**Wall seconds (including the held-out observer)**

| arm | median [p25, p75] (min..max) |
|---|---|
| dgm | 0.174 [0.161, 0.186] (0.140..0.218) |
| lexicase | 0.176 [0.165, 0.185] (0.143..0.232) |
| tournament | 0.179 [0.172, 0.189] (0.148..0.218) |
| uniform | 0.197 [0.182, 0.209] (0.168..0.233) |

**Champions frozen**

| arm | median [p25, p75] (min..max) |
|---|---|
| dgm | 6.000 [5.000, 6.000] (3.000..7.000) |
| lexicase | 7.000 [6.000, 7.000] (4.000..9.000) |
| tournament | 6.000 [5.000, 7.000] (4.000..9.000) |
| uniform | 5.000 [4.000, 5.000] (3.000..7.000) |

**Share of champions in the lineage of the last generation**

| arm | median [p25, p75] (min..max) |
|---|---|
| dgm | 0.833 [0.800, 1.000] (0.500..1.000) |
| lexicase | 0.750 [0.667, 0.833] (0.429..1.000) |
| tournament | 0.833 [0.750, 1.000] (0.571..1.000) |
| uniform | 0.600 [0.500, 0.750] (0.143..1.000) |

**Share of earlier champions surpassed by the final one**

| arm | median [p25, p75] (min..max) |
|---|---|
| dgm | 0.800 [0.600, 1.000] (0.333..1.000) |
| lexicase | 0.667 [0.571, 0.800] (0.400..1.000) |
| tournament | 0.800 [0.600, 1.000] (0.125..1.000) |
| uniform | 1.000 [0.667, 1.000] (0.333..1.000) |

**Rediscoveries (newborn behaves like an earlier member)**

| arm | median [p25, p75] (min..max) |
|---|---|
| dgm | 35.000 [30.000, 43.000] (10.000..68.000) |
| lexicase | 31.000 [25.000, 39.000] (12.000..52.000) |
| tournament | 36.000 [32.000, 50.000] (16.000..91.000) |
| uniform | 52.000 [47.000, 58.000] (37.000..79.000) |

**Champion cycles**

| arm | median [p25, p75] (min..max) |
|---|---|
| dgm | 0.000 [0.000, 0.000] (0.000..1.000) |
| lexicase | 0.000 [0.000, 1.000] (0.000..2.000) |
| tournament | 0.000 [0.000, 1.000] (0.000..2.000) |
| uniform | 0.000 [0.000, 0.000] (0.000..2.000) |

**Held-out rate of the best finalist** (arm minus dgm, per seed)

| arm | median diff | larger / equal / smaller |
|---|---|---|
| uniform | -0.025 | 9 / 6 / 15 |
| lexicase | +0.050 | 19 / 7 / 4 |
| tournament | +0.000 | 10 / 8 / 12 |

**Generalisation gap** (arm minus dgm, per seed)

| arm | median diff | larger / equal / smaller |
|---|---|---|
| uniform | -0.025 | 11 / 4 / 15 |
| lexicase | -0.025 | 8 / 7 / 15 |
| tournament | +0.000 | 14 / 8 / 8 |

**Distinct behaviours** (arm minus dgm, per seed)

| arm | median diff | larger / equal / smaller |
|---|---|---|
| uniform | +11.000 | 22 / 1 / 7 |
| lexicase | -13.000 | 5 / 1 / 24 |
| tournament | -1.000 | 14 / 1 / 15 |

## Part B: evaluation subsets

**Offline: ranking on a random subset of the search tasks against all of them** (final dgm archives, 200 random subsets per archive; top-3 overlap)

| subset | cost share | Kendall tau (median over seeds) | top-1 hit rate | top-1 regret | top-3 overlap |
|---|---|---|---|---|---|
| 5 of 40 | 0.12 | 0.655 | 0.175 | 0.1223 | 0.303 |
| 10 of 40 | 0.25 | 0.781 | 0.480 | 0.0505 | 0.525 |
| 20 of 40 | 0.50 | 0.879 | 0.725 | 0.0125 | 0.750 |

**End to end: dgm on a subset of the search panel at the same call budget** (reference: the full-panel dgm arm)

**Held-out rate of the best bred finalist, as the subset ranking picks it**

| arm | median [p25, p75] (min..max) |
|---|---|
| dgm | 0.625 [0.575, 0.700] (0.450..0.875) |
| dgm-subset-10 | 0.650 [0.575, 0.750] (0.300..0.900) |
| dgm-subset-20 | 0.650 [0.575, 0.700] (0.325..0.825) |

**Generations completed**

| arm | median [p25, p75] (min..max) |
|---|---|
| dgm | 13.000 [12.000, 14.000] (9.000..16.000) |
| dgm-subset-10 | 35.000 [33.000, 37.000] (17.000..44.000) |
| dgm-subset-20 | 21.000 [20.000, 22.000] (17.000..25.000) |

**Cascades evaluated on the search panel**

| arm | median [p25, p75] (min..max) |
|---|---|
| dgm | 4000.000 [3680.000, 4320.000] (2720.000..4960.000) |
| dgm-subset-10 | 2760.000 [2600.000, 2920.000] (1320.000..3480.000) |
| dgm-subset-20 | 3280.000 [3120.000, 3440.000] (2640.000..3920.000) |

| arm | seeds | finalist pick changed by full re-evaluation | held-out of subset pick (median) | held-out of full-panel pick (median) | tau subset vs full (median) |
|---|---|---|---|---|---|
| dgm-subset-10 | 30 | 7 | 0.625 | 0.650 | 0.000 |
| dgm-subset-20 | 30 | 4 | 0.600 | 0.650 | 0.000 |
