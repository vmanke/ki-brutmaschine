#!/usr/bin/env python3
"""The 16 candidate designs of the sequential rule for step 1, side by side.

The grid is the proof threshold U in {20, 40, 1024/11, 100} times the cap in {60, 80, 100,
120} pairs. The lower threshold is 1/20 in every design. Per design it prints:
- alpha, the exact worst-case type I error of the primary, P(proven | p = 1/2);
- the primary at the Kaplan-Meier rate p = 22/31: power, P(failed) and P(undecided);
- 4 alpha + beta at p = 22/31, with beta = 1 - power (the weight 4 is the beta:alpha ratio
  of 4 to 1 in Cohen 1992);
- the joint rule in the Kaplan-Meier scenario of oc.py: P(claim 1), P(both), E[N], N90;
- the hours: expected E[N] x 12.68 / 60 + 0.42, worst case cap x 14.31 / 60 + 0.48.
Then, for the chosen design and the first approved one (U = 20, cap 60), it prints the
figures the documents cite beside the table: the joint rule in the four null scenarios of
oc.py and on the 10 long pilot pairs as observed, the primary at p = 0.3, the pass rate
above which the chosen design proves the primary more often (by bisection), the chance of
a true proof averaged over the uncertainty of the pilot rate, and the worst-case hours.
The engine is oc.py. Its primary DP runs in exact fractions, its joint DP in floats, and
so does the bisection.

Usage: python3 -I -B designs.py <pilot.json>
"""
import json
import os
import sys
from fractions import Fraction as F
from math import factorial

# python3 -I does not put this folder on sys.path, so add it for the oc import. The import
# writes no bytecode into the repository, even without -B.
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import oc
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

# The figures beside the table, for the chosen design and the first approved one.
COMPARED = [("U=100, cap 120", F(100), 120), ("U=20, cap 60", F(20), 60)]


def model(pA, pR, s):
    """The categories (X=1,W), (X=1,L), (X=0,L), (X=0,T) of a pair when the arms pass
    independently with pA and pR and, when both pass, arm A is first with probability s.
    The same model as in oc.py."""
    return [pA * (1 - pR) + pA * pR * s, pA * pR * (1 - s), (1 - pA) * pR, (1 - pA) * (1 - pR)]


# The four joint null scenarios of oc.py, each with the claim that is false in it: claim 1
# when the primary null holds (claim 2 is made only with claim 1), else claim 2.
NULLS = [
    ("both nulls (0.5, 0.5, 0.5)", model(0.5, 0.5, 0.5), "c1"),
    ("primary null, R much worse (0.5, 0.1, 0.8)", model(0.5, 0.1, 0.8), "c1"),
    ("selection irrelevant (0.75, 0.75, 0.5)", model(0.75, 0.75, 0.5), "c2"),
    ("selection irrelevant (0.85, 0.85, 0.5)", model(0.85, 0.85, 0.5), "c2"),
]

# The 10 pilot pairs with both arms run to generation 1000, as observed.
long_pairs = pilot["empirical_pairs_1000"]
observed = [sum(1 for p in long_pairs if p[3] == c) / len(long_pairs)
            for c in ("X1W", "X1L", "X0L", "X0T")]

print()
print("The joint rule of oc.py: P(claim 1), P(both), E[N], N90, hours as above, and the")
print("probability of the claim that is false in a null scenario.")
print(f"{'design':15s} {'scenario':46s} {'claim1':>8s} {'both':>8s} {'E[N]':>6s} {'N90':>4s} "
      f"{'hours':>6s} {'false':>8s}")

for design, U, cap in COMPARED:
    largest = 0.0

    for name, cats, false in NULLS:
        r = joint_oc(cats, U, cap)
        largest = max(largest, r[false])
        print(f"{design:15s} {name:46s} {r['c1']:8.6f} {r['c2']:8.6f} {r['EN']:6.2f} {r['N90']:4d} "
              f"{r['EN'] * PAIR_MEAN / 60 + CHECK_MEAN:6.2f} {r[false]:8.6f}")

    r = joint_oc(observed, U, cap)
    name = "the 10 long pilot pairs as observed"
    print(f"{design:15s} {name:46s} {r['c1']:8.6f} {r['c2']:8.6f} {r['EN']:6.2f} {r['N90']:4d} "
          f"{r['EN'] * PAIR_MEAN / 60 + CHECK_MEAN:6.2f}")
    print(f"{design:15s} largest probability of a false claim over the four null scenarios: "
          f"{largest:.6f}")

print()
print("The primary at p = 0.3, the arm A rate by generation 200:")

for design, U, cap in COMPARED:
    P, Fl, Und, EN, n90 = primary_oc(F(3, 10), U, cap)
    print(f"  {design}: P(proven) {float(P):.7f}, P(failed) {float(Fl):.7f}, E[N] {float(EN):.2f}, "
          f"N90 {n90}")


def more_often(p):
    """P(proven) of the chosen design minus that of U = 20, cap 60, at the pass rate p."""
    (_, U, cap), (_, U0, cap0) = COMPARED

    return primary_oc(p, U, cap, exact = False)[0] - primary_oc(p, U0, cap0, exact = False)[0]


lo, hi = 0.6, 0.7
assert more_often(lo) < 0 < more_often(hi)

for _ in range(40):
    mid = (lo + hi) / 2
    lo, hi = (mid, hi) if more_often(mid) < 0 else (lo, mid)

print()
print(f"The chosen design proves the primary more often than U=20, cap 60 above p = {hi:.4f} "
      f"(bisection).")


def proof_paths(U, cap):
    """The number of sequences of passes and failures that prove the primary at (k, n)."""
    alive = {0: 1}
    hits = {}

    for n in range(1, cap + 1):
        new = {}

        for k, c in alive.items():
            new[k + 1] = new.get(k + 1, 0) + c
            new[k] = new.get(k, 0) + c

        alive = {}

        for k, c in new.items():
            if oc.proven(k, n, U):
                hits[(k, n)] = c
            elif not oc.failed(k, n):
                alive[k] = c

    return hits


def beta(x, y):
    return F(factorial(x - 1) * factorial(y - 1), factorial(x + y - 1))


print()
print("The chance of a true proof averaged over the uncertainty of the pilot rate, exact:")

for label, a, b in (("Beta(10, 3), 9 of 11 passes with a uniform prior", 10, 3),
                    ("Beta(9, 2), 8 of 9 passes with a uniform prior", 9, 2)):
    averages = []

    for design, U, cap in COMPARED:
        hits = proof_paths(U, cap)
        at_3_4 = sum(c * F(3, 4)**k * F(1, 4)**(n - k) for (k, n), c in hits.items())
        assert at_3_4 == primary_oc(F(3, 4), U, cap)[0]
        mean = sum(c * beta(a + k, b + n - k) for (k, n), c in hits.items()) / beta(a, b)
        averages.append(f"{design} {float(mean):.4f}")

    print(f"  {label}: " + ", ".join(averages))

slowest_runs = sum(max(r["minutes_to_1000"] for r in pilot["runs"]
                       if r["arm"] == arm and r["minutes_to_1000"] is not None)
                   for arm in "AR")

print()
print(f"Worst case without the thread check, at {PAIR_MAX} min per pair (the slowest pilot pair) "
      f"and at {slowest_runs:.2f} min (the slowest run of each arm):")

for design, U, cap in COMPARED:
    print(f"  {design}: {cap * PAIR_MAX / 60:.2f} h and {cap * slowest_runs / 60:.2f} h")
