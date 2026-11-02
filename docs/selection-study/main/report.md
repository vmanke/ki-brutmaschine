# Selection study: results

Seeds 1..30 (30 per arm). Per run: up to 1000 generations, 8 children per generation, 40 search and 40 held-out tasks, finalists 3, call budget 40000 (the budget ends every run). Lexicase epsilon 0, tournament size 3. Simulated models (positive control).

## Part A: arms at one call budget

**Held-out rate of the best bred finalist (select phase)**

| arm | median [p25, p75] (min..max) |
|---|---|
| dgm | 0.675 [0.625, 0.750] (0.375..0.925) |
| lexicase | 0.750 [0.700, 0.850] (0.500..1.000) |
| tournament | 0.700 [0.650, 0.750] (0.500..0.925) |
| uniform | 0.700 [0.625, 0.725] (0.500..0.875) |

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
| dgm | 0.750 [0.675, 0.775] (0.475..0.925) |
| lexicase | 0.775 [0.750, 0.875] (0.550..1.000) |
| tournament | 0.750 [0.675, 0.800] (0.475..0.925) |
| uniform | 0.700 [0.650, 0.750] (0.500..0.900) |

**Training rate of the final incumbent**

| arm | median [p25, p75] (min..max) |
|---|---|
| dgm | 0.725 [0.675, 0.775] (0.475..0.925) |
| lexicase | 0.775 [0.750, 0.875] (0.550..1.000) |
| tournament | 0.750 [0.675, 0.800] (0.475..0.925) |
| uniform | 0.675 [0.650, 0.750] (0.475..0.900) |

**Held-out rate of the final incumbent**

| arm | median [p25, p75] (min..max) |
|---|---|
| dgm | 0.675 [0.625, 0.750] (0.375..0.925) |
| lexicase | 0.750 [0.700, 0.825] (0.500..1.000) |
| tournament | 0.700 [0.625, 0.750] (0.475..0.925) |
| uniform | 0.675 [0.600, 0.700] (0.500..0.875) |

**Generalisation gap of the final incumbent (train - held-out)**

| arm | median [p25, p75] (min..max) |
|---|---|
| dgm | 0.025 [-0.025, 0.100] (-0.225..0.300) |
| lexicase | 0.025 [-0.025, 0.100] (-0.225..0.300) |
| tournament | 0.025 [-0.050, 0.100] (-0.200..0.275) |
| uniform | 0.025 [-0.025, 0.125] (-0.175..0.300) |

**Slope of the gap per generation**

| arm | median [p25, p75] (min..max) |
|---|---|
| dgm | 0.001 [-0.001, 0.001] (-0.005..0.007) |
| lexicase | -0.000 [-0.001, 0.001] (-0.005..0.005) |
| tournament | -0.000 [-0.002, 0.002] (-0.011..0.005) |
| uniform | 0.000 [-0.000, 0.001] (-0.005..0.004) |

**Distinct behaviours in the final archive**

| arm | median [p25, p75] (min..max) |
|---|---|
| dgm | 143.000 [116.000, 178.000] (83.000..214.000) |
| lexicase | 94.000 [81.000, 109.000] (38.000..146.000) |
| tournament | 144.000 [117.000, 169.000] (61.000..204.000) |
| uniform | 177.000 [158.000, 198.000] (123.000..236.000) |

**Behaviours per archive member**

| arm | median [p25, p75] (min..max) |
|---|---|
| dgm | 0.510 [0.403, 0.579] (0.301..0.764) |
| lexicase | 0.377 [0.336, 0.455] (0.156..0.670) |
| tournament | 0.478 [0.363, 0.579] (0.209..0.699) |
| uniform | 0.486 [0.441, 0.525] (0.317..0.618) |

**Mean entropy of the parent draw (bits)**

| arm | median [p25, p75] (min..max) |
|---|---|
| dgm | 6.365 [6.334, 6.421] (6.008..6.614) |
| lexicase | 4.510 [3.851, 5.110] (2.850..5.324) |
| tournament | 6.159 [6.119, 6.234] (5.761..6.560) |
| uniform | 7.123 [7.028, 7.211] (6.892..7.348) |

**Generations completed**

| arm | median [p25, p75] (min..max) |
|---|---|
| dgm | 37.000 [36.000, 39.000] (30.000..43.000) |
| lexicase | 31.000 [31.000, 32.000] (27.000..34.000) |
| tournament | 37.000 [36.000, 39.000] (28.000..49.000) |
| uniform | 47.000 [44.000, 50.000] (40.000..55.000) |

**Cascades evaluated on the search panel**

| arm | median [p25, p75] (min..max) |
|---|---|
| dgm | 11680.000 [11360.000, 12320.000] (9440.000..13600.000) |
| lexicase | 9760.000 [9760.000, 10080.000] (8480.000..10720.000) |
| tournament | 11680.000 [11360.000, 12320.000] (8800.000..15520.000) |
| uniform | 14880.000 [13920.000, 15840.000] (12640.000..17440.000) |

**Worker calls (budgeted)**

| arm | median [p25, p75] (min..max) |
|---|---|
| dgm | 38467.000 [38223.000, 38681.000] (36609.000..39519.000) |
| lexicase | 38360.000 [37707.000, 38681.000] (37152.000..39255.000) |
| tournament | 38626.000 [38246.000, 38840.000] (37230.000..39389.000) |
| uniform | 38481.000 [38053.000, 39021.000] (37053.000..39597.000) |

**Completion tokens**

| arm | median [p25, p75] (min..max) |
|---|---|
| dgm | 4368926.000 [3750736.000, 4775564.000] (2400764.000..5584050.000) |
| lexicase | 4720174.000 [4393364.000, 5060395.000] (3840194.000..8681012.000) |
| tournament | 4604292.000 [4250459.000, 5127940.000] (2651760.000..8718846.000) |
| uniform | 3183768.000 [2786584.000, 3470607.000] (2224170.000..4173492.000) |

**Model cost**

| arm | median [p25, p75] (min..max) |
|---|---|
| dgm | 25.511 [22.101, 27.262] (19.467..40.226) |
| lexicase | 22.705 [20.611, 26.696] (19.320..41.860) |
| tournament | 23.389 [21.755, 25.378] (18.984..34.760) |
| uniform | 24.604 [22.855, 27.413] (21.457..40.322) |

**Wall seconds (including the held-out observer)**

| arm | median [p25, p75] (min..max) |
|---|---|
| dgm | 0.754 [0.721, 0.799] (0.568..0.900) |
| lexicase | 0.739 [0.724, 0.765] (0.590..1.041) |
| tournament | 0.748 [0.710, 0.789] (0.595..1.011) |
| uniform | 0.908 [0.838, 0.983] (0.741..1.297) |

**Champions frozen**

| arm | median [p25, p75] (min..max) |
|---|---|
| dgm | 8.000 [7.000, 9.000] (6.000..12.000) |
| lexicase | 9.000 [7.000, 12.000] (4.000..16.000) |
| tournament | 9.000 [7.000, 10.000] (5.000..12.000) |
| uniform | 6.000 [6.000, 7.000] (4.000..9.000) |

**Share of champions in the lineage of the last generation**

| arm | median [p25, p75] (min..max) |
|---|---|
| dgm | 0.778 [0.700, 0.889] (0.417..1.000) |
| lexicase | 0.667 [0.545, 0.750] (0.400..1.000) |
| tournament | 0.778 [0.667, 0.857] (0.500..1.000) |
| uniform | 0.429 [0.375, 0.556] (0.286..1.000) |

**Share of earlier champions surpassed by the final one**

| arm | median [p25, p75] (min..max) |
|---|---|
| dgm | 0.857 [0.750, 1.000] (0.333..1.000) |
| lexicase | 0.778 [0.636, 0.900] (0.333..1.000) |
| tournament | 0.875 [0.750, 1.000] (0.250..1.000) |
| uniform | 0.875 [0.667, 1.000] (0.250..1.000) |

**Rediscoveries (newborn behaves like an earlier member)**

| arm | median [p25, p75] (min..max) |
|---|---|
| dgm | 140.000 [113.000, 162.000] (65.000..216.000) |
| lexicase | 150.000 [132.000, 163.000] (67.000..204.000) |
| tournament | 149.000 [122.000, 181.000] (74.000..254.000) |
| uniform | 193.000 [173.000, 204.000] (133.000..261.000) |

**Champion cycles**

| arm | median [p25, p75] (min..max) |
|---|---|
| dgm | 0.000 [0.000, 1.000] (0.000..1.000) |
| lexicase | 1.000 [0.000, 2.000] (0.000..3.000) |
| tournament | 1.000 [0.000, 1.000] (0.000..4.000) |
| uniform | 0.000 [0.000, 0.000] (0.000..3.000) |

**Held-out rate of the best finalist (picked on the held-out panel, so optimistic)** (arm minus dgm, per seed)

| arm | median diff | larger / equal / smaller |
|---|---|---|
| uniform | -0.025 | 9 / 6 / 15 |
| lexicase | +0.025 | 18 / 7 / 5 |
| tournament | +0.000 | 12 / 7 / 11 |

**Held-out rate of the final incumbent (picked on training fitness alone)** (arm minus dgm, per seed)

| arm | median diff | larger / equal / smaller |
|---|---|---|
| uniform | +0.000 | 7 / 9 / 14 |
| lexicase | +0.025 | 17 / 8 / 5 |
| tournament | +0.000 | 13 / 7 / 10 |

**Generalisation gap** (arm minus dgm, per seed)

| arm | median diff | larger / equal / smaller |
|---|---|---|
| uniform | +0.000 | 12 / 9 / 9 |
| lexicase | +0.000 | 10 / 10 / 10 |
| tournament | +0.000 | 14 / 8 / 8 |

**Distinct behaviours** (arm minus dgm, per seed)

| arm | median diff | larger / equal / smaller |
|---|---|---|
| uniform | +31.000 | 25 / 0 / 5 |
| lexicase | -54.000 | 2 / 0 / 28 |
| tournament | -16.000 | 13 / 0 / 17 |

## Part B: evaluation subsets

**Offline: ranking on a random subset of the search tasks against all of them** (final dgm archives, 200 random subsets per archive; top-3 overlap)

| subset | cost share | Kendall tau (median over seeds) | top-1 hit rate | top-1 regret | top-3 overlap |
|---|---|---|---|---|---|
| 5 of 40 | 0.12 | 0.629 | 0.190 | 0.1427 | 0.175 |
| 10 of 40 | 0.25 | 0.760 | 0.530 | 0.0527 | 0.430 |
| 20 of 40 | 0.50 | 0.874 | 0.795 | 0.0125 | 0.717 |

**End to end: dgm on a subset of the search panel at the same call budget** (reference: the full-panel dgm arm)

**Held-out rate of the best bred finalist (select phase: best of the finalists on the held-out panel)**

| arm | median [p25, p75] (min..max) |
|---|---|
| dgm | 0.675 [0.625, 0.750] (0.375..0.925) |
| dgm-subset-10 | 0.700 [0.650, 0.750] (0.350..0.900) |
| dgm-subset-20 | 0.675 [0.625, 0.725] (0.375..0.875) |

**Generations completed**

| arm | median [p25, p75] (min..max) |
|---|---|
| dgm | 37.000 [36.000, 39.000] (30.000..43.000) |
| dgm-subset-10 | 118.000 [106.000, 122.000] (57.000..136.000) |
| dgm-subset-20 | 65.000 [63.000, 67.000] (59.000..75.000) |

**Cascades evaluated on the search panel**

| arm | median [p25, p75] (min..max) |
|---|---|
| dgm | 11680.000 [11360.000, 12320.000] (9440.000..13600.000) |
| dgm-subset-10 | 9400.000 [8440.000, 9720.000] (4520.000..10840.000) |
| dgm-subset-20 | 10320.000 [10000.000, 10640.000] (9360.000..11920.000) |

| arm | seeds | finalist pick changed by full re-evaluation | held-out of subset pick (median) | held-out of full-panel pick (median) | tau subset vs full (median) |
|---|---|---|---|---|---|
| dgm-subset-10 | 30 | 8 | 0.625 | 0.675 | 0.000 |
| dgm-subset-20 | 30 | 10 | 0.650 | 0.675 | 0.000 |
