#!/usr/bin/env python3
"""s: extra numbers for the synthesis (scenarios with A-310, caps 50/60/80, assurance,
old 9-of-10 rule, wall time by Wald's identity). Uses the validated engine of oc.py.
Usage: python3 -I extra.py <pilot.json> <a310-rediscovery.json>
"""
import json
import os
import sys
from fractions import Fraction as F
from math import comb, factorial

# python3 -I does not put this folder on sys.path, so add it for the oc import.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from oc import primary_oc, joint_oc, proven, failed

pilot = json.load(open(sys.argv[1]))
a310 = json.load(open(sys.argv[2]))
U = F(20)


def km_curve(arm, extra = None):
    data = []
    for r in pilot["runs"]:
        if r["arm"] != arm or not r["complete"]:
            continue
        data.append((r["T"], 1) if r["T"] is not None else (r["planned"], 0))
    if extra is not None:
        data.append((extra, 1))
    S, curve = F(1), []
    for t in sorted({t for t, e in data if e and t <= 1000}):
        n = sum(1 for u, e in data if u >= t)
        d = sum(1 for u, e in data if u == t and e)
        S *= F(n - d, n)
        curve.append((t, S))
    return curve


def S_at(curve, t):
    S = F(1)
    for u, s in curve:
        if u <= t:
            S = s
    return S


def categories(cA, cR):
    prev, a1 = F(1), F(0)
    for t, S in cA:
        a1 += (prev - S) * S_at(cR, t)
        prev = S
    SA, SR = S_at(cA, 1000), S_at(cR, 1000)
    return [a1, (1 - SA) - a1, SA * (1 - SR), SA * SR]


firsts = sorted(t["first_generation"] for t in a310["targets"]
                if t["reached_by_bred"] and not t["reached_by_seed"])
T310 = firsts[(a310["library"] - a310["reached_by_seeds"]) // 2]
cR = km_curve("R")
scen = {
    "KM": categories(km_curve("A"), cR),
    "KM+A310": categories(km_curve("A", T310), cR),
    "empirical 10 long pairs": [F(8, 10), F(0), F(1, 10), F(1, 10)],
}
pairmin_mean = sum(m for s, m in pilot["pair_minutes_1000"]) / len(pilot["pair_minutes_1000"])
pairmin_max = max(m for s, m in pilot["pair_minutes_1000"])
print(f"pair minutes with both arms to 1000: mean {pairmin_mean:.2f}, max {pairmin_max:.2f}")
for cap in (50, 60, 80):
    print(f"\nJOINT U=20 cap {cap}: scenario | pA | P(claim1) P(claim2) | E[N] N90 | E[hours] = E[N]*mean/60 | worst hours = cap*max/60")
    for name, c in scen.items():
        r = joint_oc([float(x) for x in c], U, cap)
        pA = float(c[0] + c[1])
        print(f"  {name:24s} pA={pA:.4f} | {r['c1']:.3f} {r['c2']:.3f} | {r['EN']:5.1f} {r['N90']} | "
              f"{r['EN'] * pairmin_mean / 60:4.1f} h | {cap * pairmin_max / 60:4.1f} h")

for cap in (50, 60, 80):
    print(f"\nPRIMARY U=20 cap {cap}: p | P(proven) P(failed) P(undecided) E[N] N90")
    for p in [F(1, 2), F(6, 10), F(7, 10), F(22, 31), F(3, 4), F(8, 10)]:
        P, Fl, Und, EN, n90 = primary_oc(p, U, cap)
        print(f"  p={float(p):.4f}: {float(P):.3f} {float(Fl):.3f} {float(Und):.3f} {float(EN):5.1f} {n90}")


def proven_paths(cap):
    """Number of paths that first hit 'proven' at (k, n)."""
    open_ = {0: 1}
    hits = {}
    for n in range(1, cap + 1):
        new = {}
        for k, c in open_.items():
            new[k + 1] = new.get(k + 1, 0) + c
            new[k] = new.get(k, 0) + c
        open_ = {}
        for k, c in new.items():
            if proven(k, n, U):
                hits[(k, n)] = c
            elif not failed(k, n):
                open_[k] = c
    return hits


def beta_fn(x, y):  # integer arguments
    return F(factorial(x - 1) * factorial(y - 1), factorial(x + y - 1))


for cap in (50, 60):
    hits = proven_paths(cap)
    # check against the DP at p = 3/4
    chk = sum(c * F(3, 4)**k * F(1, 4)**(n - k) for (k, n), c in hits.items())
    assert chk == primary_oc(F(3, 4), U, cap)[0]
    for a, b, label in ((10, 3, "Beta(10,3): 9/11 with a uniform prior"),
                        (9, 2, "Beta(9,2): 8/9 with a uniform prior")):
        ass = sum(c * beta_fn(a + k, b + n - k) for (k, n), c in hits.items()) / beta_fn(a, b)
        print(f"assurance cap {cap}, {label}: {float(ass):.4f}")

# the old fixed rule '9 of 10' at the pilot estimates
for p, label in ((F(22, 31), "KM 22/31"), (F(9, 11), "9/11"), (F(3, 10), "6/20 at 200 gens")):
    old = sum(comb(10, j) * p**j * (1 - p)**(10 - j) for j in (9, 10))
    print(f"old rule P(>= 9 of 10) at p = {label} = {float(old):.3f}")

# min pairs to prove / to fail (count form) for U=20
print("U=20: proven needs >= 8 passes with 0 failures:", proven(8, 8, U), proven(7, 7, U))
print("U=20: failed after 5 straight failures:", failed(0, 5), failed(0, 4))
print("pilot 101.. sequence (X = 1,0,0,0,0,0): failed at 6:", failed(1, 6), "not before:", failed(1, 5))
