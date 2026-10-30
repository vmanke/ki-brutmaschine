#!/usr/bin/env python3
"""s: a small reference implementation of the recommended rule, for cross-checks.

Reads run directories (rediscovery.json only), computes T, X, Y, E1, E2 in seed order and
the stop/verdicts. Used here on uncounted pilot batches as an illustration (never counted).
Usage: python3 -I reference_rule.py <scratchpad>
"""
import json
import os
import sys
from fractions import Fraction as F

CAP, CEIL = 60, 1000


def pass_time(red):
    open_ = red["library"] - red["reached_by_seeds"]
    firsts = sorted(t["first_generation"] for t in red["targets"]
                    if t["reached_by_bred"] and not t["reached_by_seed"])
    k = open_ // 2 + 1
    T = firsts[k - 1] if open_ > 0 and len(firsts) >= k else None
    assert red["passes"] == (T is not None)
    return T


def proven(w, n):
    return 3**w >= 20 * 2**n


def failed(w, n):
    return 20 * 3**w <= 2**n


def run_rule(pairs):
    """pairs: list of (seed, T_A, T_R) in seed order; T None = never (within CEIL)."""
    k = n = w = l = 0
    st1 = st2 = "open"
    lines, stop = [], None
    for i, (seed, ta, tr) in enumerate(pairs, 1):
        ta = ta if ta is not None and ta <= CEIL else None
        tr = tr if tr is not None and tr <= CEIL else None
        x = ta is not None
        if ta is not None and (tr is None or ta < tr):
            y = "A"
        elif tr is not None and (ta is None or tr < ta):
            y = "R"
        else:
            y = "tie"
        if stop is None:
            n += 1
            k += x
            if st1 == "open":
                st1 = "proven" if proven(k, n) else ("failed" if failed(k, n) else "open")
                st1_at = i
            if y != "tie":
                w, l = (w + 1, l) if y == "A" else (w, l + 1)
                if st2 == "open":
                    st2 = "proven" if proven(w, w + l) else ("failed" if failed(w, w + l) else "open")
            E1, E2 = F(3**k, 2**n), F(3**w, 2**(w + l))
            lines.append(f"pair {i} seed {seed}: T_A={ta} T_R={tr} X={int(x)} first={y} "
                         f"E1={E1} ({float(E1):.3f}) E2={E2} ({float(E2):.3f}) [{st1}/{st2}]")
            if st1 == "failed" or (st1 == "proven" and st2 != "open") or i == CAP:
                stop = i
        else:
            lines.append(f"pair {i} seed {seed}: after the stop, not counted")
    p1 = "undecided" if (stop and st1 == "open") else st1
    sec = (f"not claimed (primary not proven; E2 itself: {st2})" if st1 != "proven"
           else ("undecided" if st2 == "open" else st2)) if stop else st2
    verdict = (f"stop at pair {stop}: primary {p1}, secondary {sec}" if stop
               else f"open after {len(pairs)} pairs: run the next seed")
    return lines, verdict


def batch(sp, name, seeds, ceil_note):
    pairs = []
    for s in seeds:
        ra = json.load(open(os.path.join(sp, name, f"A-{s}", "rediscovery.json")))
        rr = json.load(open(os.path.join(sp, name, f"R-{s}", "rediscovery.json")))
        pairs.append((s, pass_time(ra), pass_time(rr)))
    lines, verdict = run_rule(pairs)
    print(f"== {name} ({ceil_note}); illustration only, pilot seeds never count")
    print("\n".join(lines))
    print(verdict)


sp = sys.argv[1]
batch(sp, "pilot-1000", range(301, 310), "1000 generations")
batch(sp, "pilot10", range(101, 111), "200 generations, so T > 200 is unknown")

# synthetic: X = 1 exactly when E1 < 1, Y alternating, never crosses in 60 pairs
k = n = 0
pairs = []
for i in range(1, 61):
    x = F(3**k, 2**n) < 1
    k, n = k + x, n + 1
    pairs.append((i, 5 if x else None, 4 if (i % 2 == 0 and x) else (None if x else (6 if i % 2 else None))))
lines, verdict = run_rule(pairs)
print("== synthetic path that keeps E1 between 1/2 and 3/2:", verdict)
print(lines[-1])
