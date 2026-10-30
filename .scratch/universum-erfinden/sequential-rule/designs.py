#!/usr/bin/env python3
"""The 16 candidate designs of the sequential rule for step 1, side by side.

The grid is the proof threshold U in {20, 40, 1024/11, 100} times the cap in {60, 80, 100,
120} pairs. The lower threshold is 1/20 in every design. Per design it prints:
- alpha, the exact worst-case type I error of the primary, P(proven | p = 1/2);
- the primary at the Kaplan-Meier rate p = 22/31: power, P(failed) and P(undecided);
- 4 alpha + beta at p = 22/31, with beta = 1 - power (Cohen's weighting of 4 to 1);
- the joint rule in the Kaplan-Meier scenario of oc.py: P(claim 1), P(both), E[N], N90;
- the hours: expected E[N] x 12.68 / 60 + 0.42, worst case cap x 14.31 / 60 + 0.48.
The engine is oc.py. Its primary DP runs in exact fractions, its joint DP in floats.

Usage: python3 -I -B designs.py <pilot.json>
"""
import json
import os
import sys
from fractions import Fraction as F

# python3 -I does not put this folder on sys.path, so add it for the oc import. The import
# writes no bytecode into the repository, even without -B.
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from oc import joint_oc, km_with, primary_oc

UPPERS = [("20", F(20)), ("40", F(40)), ("1024/11", F(1024, 11)), ("100", F(100))]
CAPS = [60, 80, 100, 120]
CHOSEN = ("100", 120)

# Minutes per pair: the mean and the slowest of the 10 pilot pairs run to 1000 generations
# in both arms. Hours of the thread check: expected and worst case.
PAIR_MEAN = 12.68
PAIR_MAX = 14.31
CHECK_MEAN = 0.42
CHECK_MAX = 0.48

pilot = json.load(open(sys.argv[1]))
p_km = km_with(pilot)
assert p_km == F(22, 31), p_km
km = pilot["km_categories_1000"]
cats = [km["X1W"], km["X1L"], km["X0L"], km["X0T"]]

print("16 designs; the lower threshold is 1/20 in each.")
print("alpha: exact P(proven | p = 1/2), the worst-case type I error of the primary.")
print("power, failed, undecided: the primary at the Kaplan-Meier rate p = 22/31.")
print("4a+b: 4 alpha + beta at p = 22/31, with beta = 1 - power.")
print("claim1, both, E[N], N90: the joint rule in the Kaplan-Meier scenario of oc.py.")
print(f"hours: E[N] x {PAIR_MEAN} / 60 + {CHECK_MEAN}. worst: cap x {PAIR_MAX} / 60 + {CHECK_MAX}.")
print()
print(f"{'design':18s} {'alpha':>8s} {'power':>6s} {'failed':>6s} {'undec.':>6s} {'4a+b':>6s} "
      f"{'claim1':>6s} {'both':>6s} {'E[N]':>6s} {'N90':>4s} {'hours':>6s} {'worst':>6s}")

exact = []
cost = {}

for name, U in UPPERS:
    for cap in CAPS:
        alpha = primary_oc(F(1, 2), U, cap)[0]
        power, failed, undecided, _, _ = primary_oc(p_km, U, cap)
        joint = joint_oc(cats, U, cap)
        weighted = 4 * alpha + (1 - power)
        cost[f"U={name}, cap {cap}"] = weighted
        hours = joint["EN"] * PAIR_MEAN / 60 + CHECK_MEAN
        worst = cap * PAIR_MAX / 60 + CHECK_MAX
        design = f"U={name:8s} cap {cap:3d}"
        mark = "  <- chosen" if (name, cap) == CHOSEN else ""
        print(f"{design} {float(alpha):8.6f} {float(power):6.4f} {float(failed):6.4f} "
              f"{float(undecided):6.4f} {float(weighted):6.4f} {joint['c1']:6.4f} {joint['c2']:6.4f} "
              f"{joint['EN']:6.2f} {joint['N90']:4d} {hours:6.2f} {worst:6.2f}{mark}")
        exact.append(f"  {design}: {alpha.numerator}/2^{alpha.denominator.bit_length() - 1}{mark}")

print()
print("smallest 4 alpha + beta:", min(cost, key = cost.get))
print()
print("exact alpha:")
print("\n".join(exact))
