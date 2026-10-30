#!/usr/bin/env python3
"""s: operating characteristics of the combined design, by exact dynamic programming.

Primary: E1 = 3^k / 2^n after n pairs with k passes of arm A (factor 3/2 on a pass, 1/2 on
a fail). Proven at the first n with E1 >= U, failed at the first n with E1 <= 1/20, cap
CAP pairs. Every boundary test is an exact integer comparison; the primary DP runs in
exact fractions. The joint (primary + secondary, hierarchical) DP runs in floats.
Usage: python3 -I oc.py <pilot.json> <a310-rediscovery.json>
"""
import json
import sys
from fractions import Fraction as F

LOW = F(1, 20)


def proven(k, n, U):
    return 3**k * U.denominator >= U.numerator * 2**n


def failed(k, n):
    return 3**k * LOW.denominator <= LOW.numerator * 2**n


def boundaries(U, upto = 10):
    """Count form: passes needed to prove with f failures; failures needed to fail with k passes."""
    need_pass = []
    for f in range(upto):
        k = 0
        while not proven(k, k + f, U):
            k += 1
        need_pass.append(k)
    need_fail = []
    for k in range(upto):
        f = 0
        while not failed(k, k + f):
            f += 1
        need_fail.append(f)
    return need_pass, need_fail


def primary_oc(p, U, cap, exact = True):
    """P(proven), P(failed), P(undecided), E[N], N90 (N = cap when undecided)."""
    one = F(1) if exact else 1.0
    p = F(p) if exact else float(p)
    dist = {0: one}
    P = Fl = 0 * one
    EN = 0 * one
    stopN = {}
    for n in range(1, cap + 1):
        new = {}
        for k, pr in dist.items():
            for k2, q in ((k + 1, p), (k, 1 - p)):
                new[k2] = new.get(k2, 0 * one) + pr * q
        dist = {}
        for k, pr in new.items():
            if proven(k, n, U):
                P += pr
                EN += n * pr
                stopN[n] = stopN.get(n, 0 * one) + pr
            elif failed(k, n):
                Fl += pr
                EN += n * pr
                stopN[n] = stopN.get(n, 0 * one) + pr
            else:
                dist[k] = pr
    Und = sum(dist.values(), 0 * one)
    EN += cap * Und
    stopN[cap] = stopN.get(cap, 0 * one) + Und
    acc, n90 = 0 * one, None
    for n in sorted(stopN):
        acc += stopN[n]
        if acc >= F(9, 10) if exact else acc >= 0.9 - 1e-12:
            n90 = n
            break
    return P, Fl, Und, EN, n90


def joint_oc(cats, U, cap):
    """Hierarchical design. cats: probabilities of (X=1,W), (X=1,L), (X=0,L), (X=0,T).

    E1 on X; E2 on Y with ties skipped; claim 2 only if claim 1. Stop at the first of:
    claim 1 failed; claim 1 proven and claim 2 decided; cap. Floats.
    """
    outcomes = [(1, "W", cats[0]), (1, "L", cats[1]), (0, "L", cats[2]), (0, "T", cats[3])]
    # state: (k or None, w, l or None, st1, st2) ; st in 'o','p','f'
    dist = {(0, 0, 0, "o", "o"): 1.0}
    res = {"c1": 0.0, "c2": 0.0, "EN": 0.0, "stopN": {}}
    for n in range(1, cap + 1):
        new = {}
        for (k, w, l, st1, st2), pr in dist.items():
            for x, y, q in outcomes:
                if q == 0:
                    continue
                k2, w2, l2, s1, s2 = k, w, l, st1, st2
                if s1 == "o":
                    k2 = k + x
                    if proven(k2, n, U):
                        s1, k2 = "p", None
                    elif failed(k2, n):
                        s1, k2 = "f", None
                if s2 == "o" and y != "T":
                    w2, l2 = (w + 1, l) if y == "W" else (w, l + 1)
                    if proven(w2, w2 + l2, U):
                        s2, w2, l2 = "p", None, None
                    elif failed(w2, w2 + l2):
                        s2, w2, l2 = "f", None, None
                key = (k2, w2, l2, s1, s2)
                new[key] = new.get(key, 0.0) + pr * q
        dist = {}
        for key, pr in new.items():
            k2, w2, l2, s1, s2 = key
            stop = s1 == "f" or (s1 == "p" and s2 != "o") or n == cap
            if stop:
                c1 = s1 == "p"
                res["c1"] += pr * c1
                res["c2"] += pr * (c1 and s2 == "p")
                res["EN"] += n * pr
                res["stopN"][n] = res["stopN"].get(n, 0.0) + pr
            else:
                dist[key] = pr
    acc = 0.0
    for n in sorted(res["stopN"]):
        acc += res["stopN"][n]
        if acc >= 0.9 - 1e-12:
            res["N90"] = n
            break
    return res


def km_with(pilot, extra_event = None):
    """KM p_A(1000), optionally with one extra arm-A run that passes at extra_event."""
    data = []
    for r in pilot["runs"]:
        if r["arm"] != "A" or not r["complete"]:
            continue
        data.append((r["T"], 1) if r["T"] is not None else (r["planned"], 0))
    if extra_event is not None:
        data.append((extra_event, 1))
    S = F(1)
    for t in sorted({t for t, e in data if e and t <= 1000}):
        n = sum(1 for u, e in data if u >= t)
        d = sum(1 for u, e in data if u == t and e)
        S *= F(n - d, n)
    return 1 - S


def main():
    pilot = json.load(open(sys.argv[1]))
    a310 = json.load(open(sys.argv[2]))
    firsts = sorted(t["first_generation"] for t in a310["targets"]
                    if t["reached_by_bred"] and not t["reached_by_seed"])
    T310 = firsts[(a310["library"] - a310["reached_by_seeds"]) // 2]
    pA_km = km_with(pilot)
    pA_km310 = km_with(pilot, T310)
    print(f"A-310 pass time from the finished copy: {T310}")
    print(f"pilot p_A(1000): KM {float(pA_km):.4f} ({pA_km}), KM with A-310 {float(pA_km310):.4f}, exact 9/11 = {9/11:.4f}")

    U20, U93 = F(20), F(1024, 11)
    for name, U in (("U=20", U20), ("U=1024/11", U93)):
        npass, nfail = boundaries(U)
        print(f"{name}: passes needed with 0..9 failures {npass}; failures needed with 0..9 passes {nfail}")

    # exact type I at p = 1/2 (the worst case of the composite null p <= 1/2)
    for name, U in (("U=20", U20), ("U=1024/11", U93)):
        for cap in (50, 60):
            P, Fl, Und, EN, n90 = primary_oc(F(1, 2), U, cap)
            print(f"{name} cap {cap}: exact P(proven | p=1/2) = {P} = {float(P):.6f}; "
                  f"P(failed) {float(Fl):.4f} P(undecided) {float(Und):.4f} E[N] {float(EN):.2f} N90 {n90}")
        for cap in (50, 60):
            P, Fl, Und, EN, n90 = primary_oc(F(3, 4), U, cap)
            print(f"{name} cap {cap}: exact P(failed | p=3/4) = {float(Fl):.6f}")

    grid = [F(3, 10), F(1, 2), F(6, 10), F(7, 10), pA_km, F(3, 4), pA_km310, F(8, 10), F(9, 11), F(85, 100), F(9, 10)]
    for name, U in (("U=20", U20), ("U=1024/11", U93)):
        for cap in (50, 60):
            print(f"\nPRIMARY {name}, cap {cap}: p | P(proven) P(failed) P(undecided) | E[N] N90")
            for p in grid:
                P, Fl, Und, EN, n90 = primary_oc(p, U, cap)
                print(f"  p={float(p):.4f} | {float(P):.3f} {float(Fl):.3f} {float(Und):.3f} | {float(EN):5.1f} {n90}")

    # joint hierarchical scenarios
    km = pilot["km_categories_1000"]
    km_c = [km["X1W"], km["X1L"], km["X0L"], km["X0T"]]

    def model(pA, pR, s):
        return [pA * (1 - pR) + pA * pR * s, pA * pR * (1 - s), (1 - pA) * pR, (1 - pA) * (1 - pR)]

    scen = [
        ("Both nulls pA=pR=0.5, s=0.5", model(0.5, 0.5, 0.5)),
        ("Primary null, R much worse (0.5, 0.1, 0.8)", model(0.5, 0.1, 0.8)),
        ("Selection irrelevant pA=pR=0.75", model(0.75, 0.75, 0.5)),
        ("Selection irrelevant pA=pR=0.85", model(0.85, 0.85, 0.5)),
        ("Pilot-like 1000 (0.85, 0.2, 0.8)", model(0.85, 0.2, 0.8)),
        ("Pessimistic (0.75, 0.4, 0.6)", model(0.75, 0.4, 0.6)),
        ("Pilot-like 200 (0.3, 0.15, 0.7)", model(0.3, 0.15, 0.7)),
        ("KM pilot at 1000 (independent T_A, T_R)", km_c),
        ("Empirical 10 long pairs (8 W, 0, 1 L, 1 T)/10", [0.8, 0.0, 0.1, 0.1]),
    ]
    for name, U in (("U=20", U20), ("U=1024/11", U93)):
        for cap in (50, 60):
            print(f"\nJOINT hierarchical {name}, cap {cap}: P(claim1) P(claim2) E[N] N90 | q tie")
            for sname, c in scen:
                r = joint_oc(c, U, cap)
                tie = c[3]
                q = c[0] / (c[0] + c[1] + c[2])
                print(f"  {sname:48s} {r['c1']:.4f} {r['c2']:.4f} {r['EN']:5.1f} {r['N90']} | q={q:.3f} tie={tie:.3f}")


if __name__ == "__main__":
    main()
