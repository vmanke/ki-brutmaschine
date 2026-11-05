#!/usr/bin/env python3
"""Operating characteristics of the step 2 rule (ticket 11), by dynamic programming.

One claim, V or K: each triple is "S first" with probability qs, "control first" with qc and
a tie with t = 1 - qs - qc. The evidence E starts at 1 and is multiplied by a = 2 p1 on
"S first" and by b = 2 (1 - p1) on "control first"; a tie leaves it. The claim is proven at
the first triple with E >= 200, fails at the first with E <= 1/20, and is undecided after
the cap of N_max triples. Every boundary test compares exact fractions. The rule stops after
the first triple at which both claims are decided, at the latest after N_max triples.

It prints, per design (p1, N_max):
- alpha: the exact worst case of a false proof of one claim, P(proven) at qs = qc = 1/2 with
  no ties; ties and a share of "S first" below 1/2 only lower it;
- the power of each claim at the pilot's estimate (qs, qc, t from the pilot counts), exactly;
- the power averaged over the uncertainty of that estimate: over a Dirichlet(w + 1, l + 1,
  u + 1) distribution of (qs, qc, t), the posterior of a uniform prior after w triples
  "S first", l "control first" and u ties; exactly, by counting the sequences that prove;
- the mean and the 90th percentile of the triples the experiment needs, and the probability
  of each verdict, under the joint distribution of (Y_V, Y_K) observed in the pilot, in
  floats;
- the hours: the mean triples times the mean minutes of a triple, and N_max times the slowest
  triple, both for Runs one at a time: the pilot's minutes (two Runs at a time) scaled by the
  ratio of its reruns, which ran alone, to the same Runs in the pilot.
Then, for the proposed design, the power of one claim against the share of "S first" among
untied triples, at the pilot's tie rate and without ties.

The engine for the single claim runs in exact fractions; the joint distribution in floats.
own_check.py recounts the figures of the proposed design with other methods.

Usage: python3 -I -B oc.py <pilot.json>
"""

import json
import sys
from fractions import Fraction as F

PROOF = F(200)
LOWER = F(1, 20)
PROPOSED = (F(3, 4), 120)
P1S = [F(3, 5), F(2, 3), F(7, 10), F(3, 4), F(4, 5)]
CAPS = [60, 80, 100, 120]


class Claim:
    """The boundaries of one claim for a given p1, with a cache of each state's verdict."""

    def __init__(self, p1):
        self.a = 2 * p1
        self.b = 2 * (1 - p1)
        self.cache = {}

    def verdict(self, w, l):
        key = (w, l)

        if key not in self.cache:
            e = self.a ** w * self.b ** l
            self.cache[key] = "P" if e >= PROOF else "F" if e <= LOWER else None

        return self.cache[key]


def single(claim, qs, qc, cap):
    """P(proven), P(failed), P(undecided) of one claim within cap triples."""
    t = 1 - qs - qc
    zero = qs * 0
    alive = {(0, 0): zero + 1}
    proven = failed = zero

    for _ in range(cap):
        new = {}

        for (w, l), p in alive.items():
            for (w2, l2), q in (((w + 1, l), qs), ((w, l + 1), qc), ((w, l), t)):
                if q == 0:
                    continue

                v = claim.verdict(w2, l2)

                if v == "P":
                    proven += p * q
                elif v == "F":
                    failed += p * q
                else:
                    new[(w2, l2)] = new.get((w2, l2), zero) + p * q

        alive = new

    return proven, failed, sum(alive.values(), zero)


def rising(x, n):
    """x (x + 1) ... (x + n - 1), an integer for integer x."""
    r = 1

    for i in range(n):
        r *= x + i

    return r


def average_power(claim, counts, cap):
    """P(proven) averaged over Dirichlet(w + 1, l + 1, u + 1) of (qs, qc, t), exactly.

    The DP counts the sequences of n triples with w "S first", l "control first" and
    n - w - l ties that stay open, and those that prove at triple n. The Dirichlet mean of
    qs^w qc^l t^r is a ratio of rising factorials.
    """
    a_s, a_c, a_t = counts[0] + 1, counts[1] + 1, counts[2] + 1
    alive = {(0, 0): 1}
    total = F(0)

    for n in range(1, cap + 1):
        new = {}

        for (w, l), c in alive.items():
            for w2, l2 in ((w + 1, l), (w, l + 1), (w, l)):
                v = claim.verdict(w2, l2)

                if v == "P":
                    r = n - w2 - l2
                    total += F(c * rising(a_s, w2) * rising(a_c, l2) * rising(a_t, r),
                               rising(a_s + a_c + a_t, n))
                elif v is None:
                    new[(w2, l2)] = new.get((w2, l2), 0) + c

        alive = new

    return total


def step(claim, state, y):
    """The state of one claim after one triple: (w, l) while open, else "P" or "F"."""
    if state in ("P", "F") or y == "tie":
        return state

    w, l = state
    w, l = (w + 1, l) if y == "S" else (w, l + 1)

    return claim.verdict(w, l) or (w, l)


def joint(claim, cells, cap):
    """The experiment under the joint distribution `cells` {(y_v, y_k): probability}, with
    y in "S", "C" (the control first) and "tie". Floats. Returns P(V proven), P(K proven),
    P(both proven), P(no claim proven), E[N] and N90."""
    alive = {((0, 0), (0, 0)): 1.0}
    stop_at = {}
    finals = {}

    for n in range(1, cap + 1):
        new = {}

        for (sv, sk), p in alive.items():
            for (yv, yk), q in cells.items():
                key = (step(claim, sv, yv), step(claim, sk, yk))
                new[key] = new.get(key, 0.0) + p * q

        alive = {}

        for (sv, sk), p in new.items():
            if (sv in ("P", "F") and sk in ("P", "F")) or n == cap:
                stop_at[n] = stop_at.get(n, 0.0) + p
                end = (sv if sv in ("P", "F") else "U", sk if sk in ("P", "F") else "U")
                finals[end] = finals.get(end, 0.0) + p
            else:
                alive[(sv, sk)] = p

    mean = sum(n * p for n, p in stop_at.items())
    acc, n90 = 0.0, None

    for n in sorted(stop_at):
        acc += stop_at[n]

        if acc >= 0.9 - 1e-12:
            n90 = n
            break

    pv = sum(p for (v, _), p in finals.items() if v == "P")
    pk = sum(p for (_, k), p in finals.items() if k == "P")
    both = finals.get(("P", "P"), 0.0)
    none = sum(p for (v, k), p in finals.items() if v != "P" and k != "P")

    return pv, pk, both, none, mean, n90


def main():
    pilot = json.load(open(sys.argv[1]))
    seeds = len(pilot["per_seed"])
    counts = {arm: (c["S_first"], c["other_first"], c["tie"])
              for arm, c in pilot["counts"].items() if arm in ("V", "K")}
    cells = {}

    for key, k in pilot["joint_Y_V_Y_K"].items():
        yv, yk = key.split("|")
        code = {"S first": "S", "tie": "tie"}
        cells[(code.get(yv, "C"), code.get(yk, "C"))] = k / seeds

    # Minutes per triple: the counted Runs run one process at a time. The pilot ran two at a
    # time; its reruns ran alone. The hours scale the pilot's mean and slowest triple by the
    # ratio of the reruns' minutes alone to the same Runs' minutes in the pilot.
    def mean_minutes(arm):
        values = [r["minutes"] for r in pilot["runs"] if r["arm"] == arm]

        return sum(values) / len(values)

    paired = sum(mean_minutes(arm) for arm in ("S", "V", "K"))
    slowest_paired = max(sum(r["minutes"] for r in pilot["runs"]
                             if r["seed"] == e["seed"] and r["arm"] in ("S", "V", "K"))
                         for e in pilot["per_seed"])
    ratio = 1.0
    rerun = pilot.get("rerun") or {}
    alone = [(name, x["minutes"]) for name, x in rerun.items() if name[0] in "SVK"]

    if alone:
        before = sum(r["minutes"] for r in pilot["runs"]
                     for name, _ in alone if r["dir"] == name)
        ratio = sum(m for _, m in alone) / before

    minutes = paired * ratio
    slowest = slowest_paired * ratio

    print(f"pilot: {seeds} seeds; minutes per triple (S + V + K), two Runs at a time: mean "
          f"{paired:.2f}, slowest {slowest_paired:.2f}; one at a time (factor {ratio:.4f} from "
          f"{len(alone)} reruns): mean {minutes:.2f}, slowest {slowest:.2f}")

    for arm, (w, l, u) in counts.items():
        q = F(w, w + l) if w + l else None
        print(f"claim {arm}: S first {w}, {arm} first {l}, ties {u}; "
              f"P(S first | untied) = {q if q is None else f'{q} = {float(q):.4f}'}; "
              f"tie rate {u}/{seeds} = {u / seeds:.4f}")

    print("joint cells (Y_V, Y_K): " + ", ".join(
        f"{yv}|{yk} {p:.4f}" for (yv, yk), p in sorted(cells.items())))
    print()
    print("design: p1, N_max | alpha (exact) | power V, K at the estimate | average power V, K | "
          "P(V), P(K), P(both), P(none) joint | E[N] N90 | hours mean, worst")

    for p1 in P1S:
        claim = Claim(p1)

        for cap in CAPS:
            alpha = single(claim, F(1, 2), F(1, 2), cap)[0]
            powers, averages = [], []

            for arm in ("V", "K"):
                w, l, u = counts[arm]
                powers.append(single(claim, F(w, seeds), F(l, seeds), cap)[0])
                averages.append(average_power(claim, (w, l, u), cap))

            pv, pk, both, none, mean, n90 = joint(claim, cells, cap)
            mark = "  <- proposed" if (p1, cap) == PROPOSED else ""
            print(f"p1={str(p1):5s} N={cap:3d} | {float(alpha):.6f} | "
                  f"{float(powers[0]):.4f} {float(powers[1]):.4f} | "
                  f"{float(averages[0]):.4f} {float(averages[1]):.4f} | "
                  f"{pv:.4f} {pk:.4f} {both:.4f} {none:.4f} | {mean:6.2f} {n90:3d} | "
                  f"{mean * minutes / 60:6.2f} {cap * slowest / 60:6.2f}{mark}")

    p1, cap = PROPOSED
    claim = Claim(p1)
    alpha = single(claim, F(1, 2), F(1, 2), cap)[0]
    print()
    print(f"proposed p1 = {p1}, N_max = {cap}: exact alpha per claim = {alpha.numerator}/"
          f"{alpha.denominator} = {float(alpha):.6f}; for some claim at most "
          f"{float(2 * alpha):.6f}")
    print("P(failed) of one claim at P(S first | untied) = p1 without ties: "
          f"{float(single(claim, p1, 1 - p1, cap)[1]):.6f}")

    for arm in ("V", "K"):
        w, l, u = counts[arm]
        print(f"claim {arm} exactly: power at the estimate "
              f"{single(claim, F(w, seeds), F(l, seeds), cap)[0]}, "
              f"average {average_power(claim, (w, l, u), cap)}")

    tie = F(sum(counts[arm][2] for arm in counts), 2 * seeds)
    print()
    print(f"power of one claim against q = P(S first | untied), proposed design; tie rate "
          f"{tie} = {float(tie):.4f} (the mean of both claims in the pilot) and 0:")

    for q in [F(1, 2), F(3, 5), F(13, 20), F(7, 10), F(3, 4), F(4, 5), F(9, 10), F(1)]:
        with_ties = single(claim, q * (1 - tie), (1 - q) * (1 - tie), cap)[0]
        without = single(claim, q, 1 - q, cap)[0]
        print(f"  q = {str(q):5s} = {float(q):.2f}: {float(with_ties):.4f} with ties, "
              f"{float(without):.4f} without")


if __name__ == "__main__":
    main()
