#!/usr/bin/env python3
"""Independent check of the sequential rule for step 1.

Written from the rule text alone, without the code of oc.py or extra.py.
All probabilities are exact fractions. Only the printout rounds them.

The rule: after n pairs with k passes of arm A, the evidence is E1 = 3^k / 2^n.
The primary claim is proven at the first pair with E1 >= upper.
It fails at the first pair with E1 <= lower.
After cap pairs without a crossing it is undecided.
The chosen design has upper 100, lower 1/20 and cap 120.
Two other designs follow for comparison.

Usage: python3 -I own_check.py
"""

from fractions import Fraction as Q


def verdict(k, n, upper, lower):
    e = Q(3 ** k, 2 ** n)

    if e >= upper:
        return "proven"

    if e <= lower:
        return "failed"

    return None


def run(p, upper, lower, cap):
    alive = {0: Q(1)}
    proven = Q(0)
    failed = Q(0)
    expected_n = Q(0)

    for n in range(1, cap + 1):
        nxt = {}

        for k, w in alive.items():
            for k2, w2 in ((k + 1, w * p), (k, w * (1 - p))):
                v = verdict(k2, n, upper, lower)

                if v == "proven":
                    proven += w2
                    expected_n += w2 * n
                elif v == "failed":
                    failed += w2
                    expected_n += w2 * n
                else:
                    nxt[k2] = nxt.get(k2, Q(0)) + w2

        alive = nxt

    undecided = sum(alive.values(), Q(0))
    expected_n += undecided * cap

    return proven, failed, undecided, expected_n


def count_form(upper, lower, max_other = 7):
    # Passes needed to prove with 0..max_other failures,
    # and failures needed to fail with 0..max_other passes.
    prove_at = []
    fail_at = []

    for f in range(max_other + 1):
        k = 0

        while verdict(k, k + f, upper, lower) != "proven":
            k += 1

        prove_at.append(k)

    for s in range(max_other + 1):
        f = 0

        while verdict(s, s + f, upper, lower) != "failed":
            f += 1

        fail_at.append(f)

    return prove_at, fail_at


def old_rule_passes(p):
    # The old rule: at least 9 passes in 10 seeds.
    return p ** 10 + 10 * p ** 9 * (1 - p)


RATES = [
    ("p = 1/2", Q(1, 2)),
    ("p = 0.6", Q(6, 10)),
    ("p = 0.7", Q(7, 10)),
    ("Kaplan-Meier 22/31", Q(22, 31)),
    ("p = 3/4", Q(3, 4)),
    ("Kaplan-Meier with A-310, 61/80", Q(61, 80)),
    ("p = 0.8", Q(8, 10)),
    ("exact pilot count 9/11", Q(9, 11)),
]

DESIGNS = [
    ("chosen", Q(100), Q(1, 20), 120),
    ("previous design", Q(20), Q(1, 20), 60),
    ("statistical twin", Q(1024, 11), Q(1, 20), 120),
]

for title, upper, lower, cap in DESIGNS:
    print(f"== {title}: upper {upper}, lower {lower}, cap {cap}")
    print("count form (passes to prove, failures to fail):", count_form(upper, lower))
    results = {}

    for name, p in RATES:
        pr, fa, un, en = run(p, upper, lower, cap)
        results[p] = (pr, fa)
        print(f"  {name:32s} proven {float(pr):.4f}  failed {float(fa):.4f}  "
              f"undecided {float(un):.4f}  E[pairs] {float(en):.1f}")

    # The worst cases: p = 1/2 for a false proof, p = 3/4 for a false failure.
    alpha = results[Q(1, 2)][0]
    miss = results[Q(3, 4)][1]
    print(f"  exact P(proven | p = 1/2) = {alpha.numerator}/2^{alpha.denominator.bit_length() - 1}"
          f" = {float(alpha):.6f}; at most 1/upper = {1 / upper}: {alpha <= 1 / upper}")
    print(f"  P(failed | p = 3/4) = {float(miss):.6f}; at most lower = {lower}: {miss <= lower}")

print("== old rule, at least 9 of 10")

for name, p in [("p = 1/2", Q(1, 2)), ("Kaplan-Meier 22/31", Q(22, 31)), ("exact pilot count 9/11", Q(9, 11))]:
    print(f"  {name:32s} P(pass) = {float(old_rule_passes(p)):.4f}")

print(f"  exact P(pass | p = 1/2) = {old_rule_passes(Q(1, 2))}")
