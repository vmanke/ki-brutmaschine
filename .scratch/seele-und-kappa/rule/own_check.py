#!/usr/bin/env python3
"""Independent recount of the operating characteristics of the proposed step 2 rule.

Written from the rule text alone, without the code of oc.py, and with other methods:

- One claim. Its verdict depends only on the sequence of untied triples, the "untied walk":
  each untied triple is "S first" with probability q and "control first" with 1 - q. The
  walk's evidence is E = a^w b^l with a = 2 p1 and b = 2 (1 - p1). Let f(m) be the
  probability that the walk first reaches E >= 200 at its m-th step without reaching
  E <= 1/20 before. Within N triples, of which each is untied with probability 1 - t
  independently of the walk, the claim is proven with probability
  sum over m of f(m) P(Binomial(N, 1 - t) >= m).
- The same averaged over Dirichlet(w + 1, l + 1, u + 1) of (qs, qc, t): under it q = qs /
  (qs + qc) follows Beta(w + 1, l + 1) and 1 - t follows Beta(w + l + 2, u + 1),
  independently. So the mean is a sum over first-crossing paths of a Beta mean of q^k (1 -
  q)^(m - k) times a Beta-binomial tail. Exact fractions.
- Both claims and the length of the experiment. A triple with a tie in both claims changes
  nothing. The other triples drive a chain of the two claims' states, its steps the
  "active" triples, each with the joint distribution of the pilot without the double tie.
  The experiment stops at the first triple at which both claims are decided, at the latest
  after N triples. With s(k) the probability that the chain decides both claims at its step
  k, the experiment stops at triple n < N with probability
  sum over k of s(k) C(n - 1, k - 1) (1 - z)^k z^(n - k), z the probability of a double tie.
  A claim is proven when its crossing step comes within Binomial(N, 1 - z) active triples.
  Floats.

Usage: python3 -I -B own_check.py <pilot.json>
"""

import json
import sys
from fractions import Fraction as Q
from math import comb

P1 = Q(3, 4)
N = 120
UPPER = Q(200)
LOWER = Q(1, 20)


def first_crossings(q, steps):
    """f(m) for m = 1..steps: the probability that the untied walk first proves at step m;
    and g(m), that it first fails at step m. The walk's state is its evidence value."""
    a, b = 2 * P1, 2 * (1 - P1)
    alive = {Q(1): q * 0 + 1}
    f, g = [], []

    for _ in range(steps):
        nxt = {}
        up = down = q * 0

        for e, p in alive.items():
            for e2, w in ((e * a, q), (e * b, 1 - q)):
                if e2 >= UPPER:
                    up += p * w
                elif e2 <= LOWER:
                    down += p * w
                else:
                    nxt[e2] = nxt.get(e2, q * 0) + p * w

        alive = nxt
        f.append(up)
        g.append(down)

    return f, g


def crossing_paths(steps):
    """For m = 1..steps, {k: number of paths of m untied steps with k wins that first prove
    at step m}."""
    a, b = 2 * P1, 2 * (1 - P1)
    alive = {(Q(1), 0): 1}
    paths = []

    for _ in range(steps):
        nxt = {}
        hits = {}

        for (e, k), c in alive.items():
            for e2, k2 in ((e * a, k + 1), (e * b, k)):
                if e2 >= UPPER:
                    hits[k2] = hits.get(k2, 0) + c
                elif e2 > LOWER:
                    nxt[(e2, k2)] = nxt.get((e2, k2), 0) + c

        alive = nxt
        paths.append(hits)

    return paths


def binomial_tail(n, p, m):
    """P(Binomial(n, p) >= m)."""
    return sum((comb(n, j) * p ** j * (1 - p) ** (n - j) for j in range(m, n + 1)), p * 0)


def beta(x, y):
    """B(x, y) for positive integers, as a fraction."""
    num = 1
    for i in range(1, x):
        num *= i
    for i in range(1, y):
        num *= i
    den = 1
    for i in range(1, x + y):
        den *= i
    return Q(num, den)


def power(qs, qc):
    """P(proven within N triples) at the per-triple probabilities qs and qc."""
    untied = qs + qc

    if untied == 0:
        return Q(0)

    f, _ = first_crossings(qs / untied, N)

    return sum((f[m - 1] * binomial_tail(N, untied, m) for m in range(1, N + 1)), Q(0))


def averaged(w, l, u):
    """The power averaged over Dirichlet(w + 1, l + 1, u + 1)."""
    a_s, a_c, a_t = w + 1, l + 1, u + 1
    a_u = a_s + a_c
    paths = crossing_paths(N)
    total = Q(0)

    for m in range(1, N + 1):
        if not paths[m - 1]:
            continue

        walk = sum((c * beta(a_s + k, a_c + m - k) for k, c in paths[m - 1].items()), Q(0))
        walk /= beta(a_s, a_c)
        tail = sum((comb(N, j) * beta(a_u + j, a_t + N - j) for j in range(m, N + 1)), Q(0))
        total += walk * tail / beta(a_u, a_t)

    return total


def chain(cells):
    """The joint chain over the active triples, as floats. cells: {(yv, yk): probability},
    y in "S", "C", "tie", without the double tie and summing to 1. Returns, per chain step
    k = 1..N: the probability that both claims are decided first at k, and the probabilities
    that claim V, claim K, and both are proven first at k (for both: the later crossing)."""
    a, b = 2 * P1, 2 * (1 - P1)

    def move(state, y):
        if not isinstance(state, Q) or y == "tie":
            return state

        e = state * (a if y == "S" else b)

        return "P" if e >= UPPER else "F" if e <= LOWER else e

    alive = {(Q(1), Q(1)): 1.0}
    stop, prove_v, prove_k, prove_both = [], [], [], []

    for _ in range(N):
        nxt = {}
        done = pv = pk = pb = 0.0

        for (sv, sk), p in alive.items():
            for (yv, yk), w in cells.items():
                v2, k2 = move(sv, yv), move(sk, yk)
                pr = p * w

                if v2 == "P" and sv != "P":
                    pv += pr
                if k2 == "P" and sk != "P":
                    pk += pr
                if v2 == "P" and k2 == "P" and (sv != "P" or sk != "P"):
                    pb += pr

                if isinstance(v2, Q) or isinstance(k2, Q):
                    nxt[(v2, k2)] = nxt.get((v2, k2), 0.0) + pr
                else:
                    done += pr

        alive = nxt
        stop.append(done)
        prove_v.append(pv)
        prove_k.append(pk)
        prove_both.append(pb)

    return stop, prove_v, prove_k, prove_both


def main():
    pilot = json.load(open(sys.argv[1]))
    seeds = len(pilot["per_seed"])
    counts = {arm: pilot["counts"][arm] for arm in ("V", "K")}

    # The worst case of a false proof: q = 1/2 without ties, every triple untied.
    f, _ = first_crossings(Q(1, 2), N)
    alpha = sum(f, Q(0))
    print(f"design p1 = {P1}, N_max = {N}")
    print(f"alpha per claim, exact: {alpha.numerator}/{alpha.denominator} = {float(alpha):.6f}")

    _, g = first_crossings(P1, N)
    print(f"P(failed) of one claim at q = p1 without ties: {float(sum(g, Q(0))):.6f}")

    for arm in ("V", "K"):
        c = counts[arm]
        w, l, u = c["S_first"], c["other_first"], c["tie"]
        at = power(Q(w, seeds), Q(l, seeds))
        avg = averaged(w, l, u)
        print(f"claim {arm}: power at the estimate {at} = {float(at):.4f}; "
              f"averaged {avg} = {float(avg):.4f}")

    # The joint distribution of the pilot, without the double tie.
    code = {"S first": "S", "tie": "tie"}
    joint = {}

    for key, k in pilot["joint_Y_V_Y_K"].items():
        yv, yk = key.split("|")
        joint[(code.get(yv, "C"), code.get(yk, "C"))] = k

    z = joint.pop(("tie", "tie"), 0) / seeds
    active = sum(joint.values())
    cells = {key: k / active for key, k in joint.items()}
    stop, prove_v, prove_k, prove_both = chain(cells)

    def tail(k):
        """P(at least k active triples among N)."""
        return sum(comb(N, j) * (1 - z) ** j * z ** (N - j) for j in range(k, N + 1))

    def at_triple(k, n):
        """P(the k-th active triple is triple n)."""
        return comb(n - 1, k - 1) * (1 - z) ** k * z ** (n - k)

    p_v = sum(prove_v[k - 1] * tail(k) for k in range(1, N + 1))
    p_k = sum(prove_k[k - 1] * tail(k) for k in range(1, N + 1))
    p_both = sum(prove_both[k - 1] * tail(k) for k in range(1, N + 1))
    p_none = 1 - p_v - p_k + p_both
    stops = [sum(stop[k - 1] * at_triple(k, n) for k in range(1, n + 1)) for n in range(1, N)]
    stops.append(1 - sum(stops))
    mean = sum(n * p for n, p in zip(range(1, N + 1), stops))
    acc, n90 = 0.0, None

    for n, p in zip(range(1, N + 1), stops):
        acc += p

        if acc >= 0.9 - 1e-12:
            n90 = n
            break

    print(f"joint: P(V) {p_v:.4f}, P(K) {p_k:.4f}, P(both) {p_both:.4f}, P(none) {p_none:.4f}; "
          f"E[N] {mean:.2f}, N90 {n90}")

    # Hours, for Runs one at a time. Per arm the mean minutes of the pilot, which ran two Runs
    # at a time; a triple is arms S, V and K. The reruns ran alone, so their minutes over the
    # same Runs' minutes in the pilot convert the one to the other.
    by_dir = {r["dir"]: r["minutes"] for r in pilot["runs"]}
    triple_runs = [r for r in pilot["runs"] if r["arm"] in ("S", "V", "K")]
    per_arm = {}

    for r in triple_runs:
        per_arm.setdefault(r["arm"], []).append(r["minutes"])

    mean_triple = sum(sum(v) / len(v) for v in per_arm.values())
    per_seed = {}

    for r in triple_runs:
        per_seed[r["seed"]] = per_seed.get(r["seed"], 0.0) + r["minutes"]

    alone = {name: x["minutes"] for name, x in (pilot.get("rerun") or {}).items()
             if name[0] in ("S", "V", "K")}
    factor = sum(alone.values()) / sum(by_dir[name] for name in alone) if alone else 1.0
    print(f"hours, one Run at a time: mean {mean * mean_triple * factor / 60:.2f}, "
          f"worst {N * max(per_seed.values()) * factor / 60:.2f}")


if __name__ == "__main__":
    main()
