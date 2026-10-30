#!/usr/bin/env python3
"""s: recompute pilot pass times, KM curves, pair categories and per-run wall times.

Reads only config.json, report.json, rediscovery.json and timings.jsonl of the pilot
batches (never events.jsonl or snapshot.json). Writes pilot.json and prints a summary.
Usage: python3 -I pilot.py <scratchpad> <out.json>
"""
import json
import os
import sys
from fractions import Fraction as F

SP = sys.argv[1]
OUT = sys.argv[2]
BATCHES = ["pilot10", "pilot-base-201", "pilot-1000", "pilot-2000"]


def pass_time(red):
    """k-th smallest first_generation among bred-only targets, k = open//2 + 1."""
    seeds = [t for t in red["targets"] if t["reached_by_seed"]]
    open_ = red["library"] - len(seeds)
    k = open_ // 2 + 1  # smallest k with 2k > open
    firsts = sorted(t["first_generation"] for t in red["targets"]
                    if t["reached_by_bred"] and not t["reached_by_seed"])
    T = firsts[k - 1] if len(firsts) >= k else None
    return T, open_, len(firsts), sorted(t["target"] for t in seeds), firsts


runs = []
for b in BATCHES:
    d = os.path.join(SP, b)
    for name in sorted(os.listdir(d)):
        p = os.path.join(d, name)
        if not os.path.isdir(p):
            continue
        cfg = json.load(open(os.path.join(p, "config.json")))
        arm = "R" if cfg.get("selection") == "uniform" else "A"
        assert name.startswith(arm + "-"), (name, cfg.get("selection"))
        row = {"batch": b, "run": name, "arm": arm, "seed": cfg["seed"],
               "planned": cfg["generations"], "max_calls": cfg["max_calls"]}
        tim = os.path.join(p, "timings.jsonl")
        secs = {}
        if os.path.exists(tim):
            for line in open(tim):
                line = line.strip()
                if line:
                    r = json.loads(line)
                    secs[r["generation"]] = r["seconds"]
        row["last_timed_generation"] = max(secs) if secs else None
        row["minutes_to_1000"] = (sum(s for g, s in secs.items() if g <= 1000) / 60
                                  if secs and max(secs) >= 1000 else None)
        row["minutes_total"] = sum(secs.values()) / 60 if secs else None
        rp = os.path.join(p, "rediscovery.json")
        if not os.path.exists(rp):
            row["complete"] = False
            runs.append(row)
            continue
        rep = json.load(open(os.path.join(p, "report.json")))
        red = json.load(open(rp))
        T, open_, hits, seeds, firsts = pass_time(red)
        row.update({"complete": True, "stop_reason": rep["stop_reason"],
                    "generations_completed": rep["generations_completed"],
                    "calls_used": rep["calls_used"], "R_seed": seeds, "open": open_,
                    "final_hits": hits, "final_D": f"{hits}/{open_}", "T": T,
                    "firsts": firsts, "file_passes": red["passes"]})
        # consistency: the file's own pass flag equals 'T defined' (T <= planned)
        assert red["passes"] == (T is not None), name
        runs.append(row)

complete = [r for r in runs if r["complete"]]
print(f"run directories: {len(runs)}; complete (with rediscovery.json): {len(complete)}")
print("incomplete:", [(r['batch'], r['run'], r['last_timed_generation']) for r in runs if not r['complete']])
print("all R_seed == ['cs','swap']:", all(r["R_seed"] == ["cs", "swap"] for r in complete))
print("stop reasons:", sorted({r["stop_reason"] for r in complete}))
print("final D values:", sorted({r["final_D"] for r in complete}))
for arm in "AR":
    xs = [(r["seed"], r["T"]) for r in complete if r["arm"] == arm]
    print(f"arm {arm} pass times (seed:T):",
          ", ".join(f"{s}:{t if t is not None else '-'}" for s, t in sorted(xs)))


def km(arm, horizon):
    """Kaplan-Meier S(t) from (T or censoring at planned generations)."""
    data = []
    for r in complete:
        if r["arm"] != arm:
            continue
        if r["T"] is not None:
            data.append((r["T"], 1))
        else:
            data.append((r["planned"], 0))
    times = sorted({t for t, e in data if e == 1 and t <= horizon})
    S = F(1)
    curve = []
    for t in times:
        at_risk = sum(1 for u, e in data if u >= t)
        d = sum(1 for u, e in data if u == t and e == 1)
        S *= F(at_risk - d, at_risk)
        curve.append((t, at_risk, d, S))
    return curve


out = {"runs": runs}
for arm in "AR":
    c = km(arm, 1000)
    out[f"km_{arm}"] = [(t, n, d, float(S)) for t, n, d, S in c]
    S1000 = c[-1][3] if c else F(1)
    print(f"KM arm {arm}: p(1000) = 1 - S = {float(1 - S1000):.4f}; jumps:",
          " ".join(f"{t}[{n}]" for t, n, d, S in c))
    out[f"p_{arm}_km_1000"] = float(1 - S1000)

# exact counts on runs planned to at least 1000 generations (pass by 1000)
for arm in "AR":
    long_ = [r for r in complete if r["arm"] == arm and r["planned"] >= 1000]
    k = sum(1 for r in long_ if r["T"] is not None and r["T"] <= 1000)
    print(f"arm {arm}: passes by 1000 among runs planned >= 1000: {k}/{len(long_)}")
    out[f"exact_{arm}_1000"] = (k, len(long_))

# pair categories under KM, T_A and T_R independent, horizon 1000
cA, cR = km("A", 1000), km("R", 1000)


def S_of(curve, t):
    S = F(1)
    for u, n, d, s in curve:
        if u <= t:
            S = s
    return S


jumpsA, prev = [], F(1)
for t, n, d, S in cA:
    jumpsA.append((t, prev - S))
    prev = S
SA, SR = S_of(cA, 1000), S_of(cR, 1000)
tA = {t for t, _ in jumpsA}
assert not tA & {t for t, n, d, S in cR}, "A and R share a jump time: ties possible"
a1 = sum(m * S_of(cR, t) for t, m in jumpsA)          # A passes, R not by T_A  -> W
a2 = (1 - SA) - a1                                     # A passes, R earlier    -> L
a3 = SA * (1 - SR)                                     # A fails, R passes      -> L
a4 = SA * SR                                           # neither                -> tie
cats = {"X1W": float(a1), "X1L": float(a2), "X0L": float(a3), "X0T": float(a4)}
print("KM pair categories at 1000 (A pass & first, A pass & R earlier, A fail & R pass, neither):",
      {k: round(v, 4) for k, v in cats.items()},
      f"P(A first | untied) = {float(a1 / (a1 + a2 + a3)):.4f}")
out["km_categories_1000"] = cats

# empirical pair categories on the pairs run to >= 1000 in both arms (truncated at 1000)
byseed = {}
for r in complete:
    byseed.setdefault(r["seed"], {})[r["arm"]] = r
emp = {"X1W": 0, "X1L": 0, "X0L": 0, "X0T": 0}
pairs_long = []
for s, d in sorted(byseed.items()):
    if "A" in d and "R" in d and d["A"]["planned"] >= 1000 and d["R"]["planned"] >= 1000:
        TA = d["A"]["T"] if d["A"]["T"] is not None and d["A"]["T"] <= 1000 else None
        TR = d["R"]["T"] if d["R"]["T"] is not None and d["R"]["T"] <= 1000 else None
        if TA is not None and (TR is None or TR > TA):
            c = "X1W"
        elif TA is not None:
            c = "X1L"  # R earlier (a same-generation tie would be T; none occur)
            assert TR < TA
        elif TR is not None:
            c = "X0L"
        else:
            c = "X0T"
        emp[c] += 1
        pairs_long.append((s, TA, TR, c))
print("pairs with both arms run to >= 1000 (seed, T_A, T_R, category):", pairs_long)
print("empirical categories:", emp)
out["empirical_pairs_1000"] = pairs_long

# wall time per run to generation 1000 and per pair with both arms to 1000
mins = {(r["seed"], r["arm"]): r["minutes_to_1000"] for r in runs if r["minutes_to_1000"] is not None}
for arm in "AR":
    v = sorted(m for (s, a), m in mins.items() if a == arm)
    print(f"arm {arm}: minutes to generation 1000 (timings sum), n={len(v)}: "
          f"min {min(v):.2f} max {max(v):.2f} mean {sum(v)/len(v):.2f}")
pair_m = sorted((s, mins[(s, 'A')] + mins[(s, 'R')]) for s in {s for s, a in mins}
                if (s, 'A') in mins and (s, 'R') in mins)
print("pair minutes (both arms to 1000):", [(s, round(m, 2)) for s, m in pair_m])
pm = [m for s, m in pair_m]
print(f"pair minutes: min {min(pm):.2f} max {max(pm):.2f} mean {sum(pm)/len(pm):.2f}")
out["pair_minutes_1000"] = pair_m
mx = max((r["calls_used"] for r in complete), default = None)
print("largest report calls_used over complete runs:", mx)
json.dump(out, open(OUT, "w"), indent = 1, default = str)
