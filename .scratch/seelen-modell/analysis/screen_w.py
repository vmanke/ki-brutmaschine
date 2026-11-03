#!/usr/bin/env python3
"""The screen of arm W (ticket 02 of `.scratch/seelen-modell/`): every number of its criteria.

Usage: python3 -I -B screen_w.py --runs DIR --logs DIR [--seeds 913-920] [--generations 1000]
       [--json PATH]

Per Run <arm>-<seed> in --runs (arms W, W0, N and WF) it reads config.json, report.json and
rediscovery.json, and events.jsonl.xz (or events.jsonl) once from start to end through
Python's lzma module; it never writes a decompressed file. From --logs it reads hashes.txt, the
SHA-256 of the five record files that the driver took after --verify and before xz. Beside
each Run it reads the driver's <run>.wall (wall seconds) and <run>.verify (the --verify line).
It prints the report and, with --json, writes the per-Run records.

Every definition below was fixed before the first Run of the screen.

- The pass time T and the divergence generation of two Runs are those of
  `.scratch/seele-und-kappa/rule/reference_rule.py`, which `diagnose.py` uses. T is the k-th
  smallest first_generation among the targets that only bred members solve, k = floor(open / 2)
  + 1; it is "never" when fewer exist or when T > 1000. The divergence generation is the first
  generation whose newborns differ, member by member in birth order, in gene id, operator,
  parents (by gene id), answers, solved count or graded value; None when none differs.
- "A passes before B": T_A < T_B, with "never" later than every generation (reference_rule's
  `first`). Equal pass times, and two Runs that never pass, are a tie.
- "W parts from N before either Run passes": the divergence generation g of W and N exists, and
  g < min(T_W, T_N), with "never" as infinite.
- The first generation whose variation_changed is above 0: read from the soul group of the
  state lines; None when no generation has one.
- The mean preference for Linear, Phase and Hadamard over the Workers of the parents chosen in
  generations 41 to 50: for every newborn of generations 41 to 50, for each id in its parents
  (a parent chosen for several children counts each time, and both parents of a crossover
  count), for each Worker of that parent, and for each of Linear, Phase and Hadamard (phases 0,
  1 and 2 of the Worker's birth Soul): w = 1 + cos(2 pi p / 65536), computed as
  `soul::preference` computes it. The mean runs over all these values. Arm WF has a fixed
  preference, so it is not measured there; arm N has no Soul.
- A Worker names a family when one of its section texts, lower-cased, holds one of the family's
  stems in `designer::FAMILY_STEMS`, read from src/cascade/designer.rs. A promising Worker names
  exactly two of Linear, Phase and Hadamard. A Worker with all three names all three.
- The draws on a parent with a promising Worker: the newborns made by SectionReplace or
  SectionAdd whose parent holds a promising Worker, in log order, up to and including the
  newborn that holds the Run's first Worker with all three; every such newborn when the Run has
  none. A draw changes a promising Worker when the sections of the child's Worker at the place
  of a promising Worker of the parent differ from that Worker's (these operators keep places).
  Such a draw loses one of its families when that child Worker no longer names both families of
  the parent's Worker among the three; it brings the missing family when it names the third; it
  makes the Worker with all three when it names all three.
- Criterion 3 pools the counts of the eight seeds per arm. A share is lost / changed, compared
  as an exact fraction. An arm with fewer than 30 draws that change a promising Worker fails it.
- Go if criteria 1, 2 and 3 hold and N passes before W on at most 4 of the 8 seeds; stop if 1 or
  2 fails; in every other case the owner decides.
"""

import argparse
import json
import math
import os
import re
import sys
from fractions import Fraction as F

sys.dont_write_bytecode = True

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, ".scratch", "seele-und-kappa", "rule"))

import reference_rule as rr  # noqa: E402

ARMS = ("W", "W0", "N", "WF")
THREE = frozenset((0, 1, 2))
PHRASE_OPS = ("section_replace", "section_add")
MAX_CALLS = 13_000_000
PARENT_WINDOW = range(41, 51)

SOULS = {
    "W": {**rr.SOUL_S, "variation": "prefer"},
    "W0": {**rr.SOUL_S, "variation": "prefer", "copy": False, "imprint": False},
    "WF": {**rr.SOUL_S, "variation": "fixed"},
    "N": None,
}


def stem_lists():
    """The stems of Linear, Phase and Hadamard, in this order, from designer::FAMILY_STEMS."""
    path = os.path.join(REPO, "src", "cascade", "designer.rs")

    with open(path, encoding="utf-8") as f:
        text = f.read()

    start = text.index("pub const FAMILY_STEMS")
    block = text[start:text.index("];", start)]
    families = re.findall(r'\(\s*"(\w+)",\s*&\[(.*?)\]', block, re.S)
    names = [name for name, _ in families]

    if names[:3] != ["Linear", "Phase", "Hadamard"] or len(names) != 8:
        sys.exit(f"screen_w: unexpected FAMILY_STEMS in {path}: {names}")

    return [re.findall(r'"([^"]*)"', inner) for _, inner in families[:3]]


def named(worker, stems):
    """The families among Linear (0), Phase (1) and Hadamard (2) that a Worker names."""
    texts = [s["text"].lower() for s in worker["sections"]]

    return frozenset(f for f, family in enumerate(stems)
                     if any(stem in t for t in texts for stem in family))


def preference(p):
    """1 + Re u(p) with the complex readout, reduced to a quarter turn as soul::phasor does."""
    q, r = divmod(p, 16384)
    rest = 2 * math.pi * r / 65536.0
    c, s = math.cos(rest), math.sin(rest)

    return 1.0 + (c, -s, -c, s)[q]


def expected_config(arm, seed, generations):
    cfg = dict(rr.COMMON)
    cfg["seed"] = seed
    cfg["generations"] = generations
    cfg["max_calls"] = MAX_CALLS

    if SOULS[arm] is not None:
        cfg["soul"] = SOULS[arm]

    return cfg


def read_text(path):
    try:
        with open(path, encoding="utf-8") as f:
            return f.read().strip()
    except OSError:
        return None


def read_run(runs, arm, seed, generations, stems):
    path = os.path.join(runs, f"{arm}-{seed}")
    cfg = rr.load(os.path.join(path, "config.json"))
    report = rr.load(os.path.join(path, "report.json"))
    red = rr.load(os.path.join(path, "rediscovery.json"))
    deviations = []

    if cfg != expected_config(arm, seed, generations):
        expected = expected_config(arm, seed, generations)
        keys = sorted(k for k in set(cfg) | set(expected) if cfg.get(k) != expected.get(k))
        deviations.append(f"settings differ from arm {arm} in {keys}")

    if (report["stop_reason"] != rr.ALL_GENERATIONS
            or report["generations_completed"] != generations + 1):
        deviations.append(f"stopped with {report['stop_reason']!r} after "
                          f"{report['generations_completed']} generations")

    t = rr.pass_time(red, path)
    t = t if t is not None and t <= generations else None

    gene = {}
    born = {}
    members = {}
    counts = {}
    pref = [0.0, 0]
    draws = {"draws": 0, "changed": 0, "lost": 0, "brought": 0, "made": 0, "multiple": 0}
    first_three = None

    with rr.open_events(path) as f:
        for line in f:
            event = json.loads(line)["event"]
            kind = event["kind"]

            if kind == "state":
                soul = event.get("soul") or {}

                if "variation_changed" in soul:
                    counts[event["generation"]] = (soul["variation_choices"],
                                                   soul["variation_changed"])
                continue

            if kind != "born":
                continue

            g = event["generation"]
            cascade = event["cascade"]
            parents = event["parents"]
            gid = rr.gene_id(cascade)
            signature = (gid, event["op"], tuple(gene[p] for p in parents),
                         json.dumps(event.get("answers")), event["solved"], event.get("quality"))
            born.setdefault(g, []).append(signature)
            gene[event["id"]] = gid

            workers = [w for stage in cascade["stages"] for w in stage["workers"]]
            info = []

            for w in workers:
                phases = w["traits"].get("phase_state")
                info.append((json.dumps(w["sections"], sort_keys=True), named(w, stems),
                             None if phases is None else [preference(p) for p in phases[:3]]))

            if g in PARENT_WINDOW:
                for p in parents:
                    for _, _, prefs in members[p]:
                        if prefs is not None:
                            for w in prefs:
                                pref[0] += w
                                pref[1] += 1

            if first_three is None:
                if event["op"] in PHRASE_OPS:
                    parent = members[parents[0]]
                    promising = [k for k, (_, fams, _) in enumerate(parent)
                                 if len(fams & THREE) == 2]

                    if promising:
                        draws["draws"] += 1
                        changed = [k for k in promising
                                   if k < len(info) and info[k][0] != parent[k][0]]

                        if len(changed) > 1:
                            draws["multiple"] += 1

                        if changed:
                            k = changed[0]
                            before, after = parent[k][1] & THREE, info[k][1]
                            draws["changed"] += 1
                            draws["lost"] += not before <= after
                            draws["brought"] += (THREE - before) <= after
                            draws["made"] += THREE <= after

                if any(THREE <= fams for _, fams, _ in info):
                    first_three = {"generation": g, "op": event["op"]}

            members[event["id"]] = info

    if len(born) != report["generations_completed"]:
        deviations.append(f"newborns in {len(born)} generations, report "
                          f"{report['generations_completed']}")

    first_change = min((g for g, (_, c) in counts.items() if c > 0), default=None)

    return {
        "arm": arm,
        "seed": seed,
        "T": t,
        "passes": red["passes"],
        "calls_used": report["calls_used"],
        "wall_seconds": float(read_text(path + ".wall") or "nan"),
        "verify": read_text(path + ".verify"),
        "first_change": first_change,
        "variation_choices": sum(c for c, _ in counts.values()) if counts else None,
        "variation_changed": sum(c for _, c in counts.values()) if counts else None,
        # Only W and W0 steer by their Souls; WF's preference is fixed, and N has no Soul.
        "mean_preference": pref[0] / pref[1] if pref[1] and arm in ("W", "W0") else None,
        "preference_values": pref[1] if arm in ("W", "W0") else 0,
        "draws": draws,
        "first_three": first_three,
        "deviations": deviations,
        "born": born,
    }


def read_hashes(logs):
    out = {}

    try:
        with open(os.path.join(logs, "hashes.txt"), encoding="utf-8") as f:
            for line in f:
                digest, name = line.split()
                run, _, file = name.partition("/")
                out.setdefault(run, {})[file] = digest
    except OSError:
        pass

    return out


def before(a, b):
    """True when a passes before b; None is never."""
    return a is not None and (b is None or a < b)


def show(t):
    return "never" if t is None else str(t)


def share(lost, changed):
    return F(lost, changed) if changed else None


def pct(x):
    return "-" if x is None else f"{100 * float(x):.1f} %"


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--logs", required=True)
    ap.add_argument("--seeds", default="913-920")
    ap.add_argument("--generations", type=int, default=1000)
    ap.add_argument("--json")
    args = ap.parse_args(argv[1:])
    lo, _, hi = args.seeds.partition("-")
    seeds = list(range(int(lo), int(hi or lo) + 1))
    stems = stem_lists()
    hashes = read_hashes(args.logs)
    runs = {}
    rows = []

    # One seed at a time, so that only four Runs' newborns are held at once.
    for s in seeds:
        r = {arm: read_run(args.runs, arm, s, args.generations, stems) for arm in ARMS}
        div = {pair: rr.divergence(r[pair[0]]["born"], r[pair[1]]["born"])
               for pair in (("W", "N"), ("W", "W0"), ("WF", "W"), ("WF", "N"))}

        for arm in ARMS:
            del r[arm]["born"]
            runs[(arm, s)] = r[arm]

        rows.append((s, r, div))

    out = []
    p = out.append

    p(f"== screen of arm W: seeds {seeds[0]} to {seeds[-1]}, arms {', '.join(ARMS)}, "
      f"{args.generations} generations")
    p("")
    p("Per seed: pass times T; divergence generations (reference_rule); first generation with "
      "variation_changed > 0; mean preference for Linear, Phase and Hadamard over the parents' "
      "Workers in generations 41 to 50.")
    p(f"{'seed':>5} {'T_W':>5} {'T_W0':>5} {'T_N':>5} {'T_WF':>5} | {'W-N':>4} {'W-W0':>4} "
      f"{'WF-W':>4} {'WF-N':>4} | {'chg W':>5} {'W0':>3} {'WF':>3} | {'pref W':>6} {'W0':>6}")
    for s, r, div in rows:
        fmt = lambda x: "-" if x is None else f"{x:.3f}"  # noqa: E731
        p(f"{s:>5} {show(r['W']['T']):>5} {show(r['W0']['T']):>5} {show(r['N']['T']):>5} "
          f"{show(r['WF']['T']):>5} | {show(div[('W', 'N')]):>4} {show(div[('W', 'W0')]):>4} "
          f"{show(div[('WF', 'W')]):>4} {show(div[('WF', 'N')]):>4} | "
          f"{show(r['W']['first_change']):>5} {show(r['W0']['first_change']):>3} "
          f"{show(r['WF']['first_change']):>3} | {fmt(r['W']['mean_preference']):>6} "
          f"{fmt(r['W0']['mean_preference']):>6}")

    p("")
    p("Draws on a parent with a promising Worker, before the first Worker with all three "
      "(draws / changing a promising Worker / losing a family / bringing the missing one / "
      "making all three), and the first Worker with all three:")

    for s, r, _ in rows:
        cells = []

        for arm in ARMS:
            d = r[arm]["draws"]
            ft = r[arm]["first_three"]
            where = "none" if ft is None else f"g{ft['generation']} {ft['op']}"
            cells.append(f"{arm} {d['draws']}/{d['changed']}/{d['lost']}/{d['brought']}/"
                         f"{d['made']} ({where})")

        p(f"{s:>5}  " + "; ".join(cells))

    pooled = {arm: {k: sum(runs[(arm, s)]["draws"][k] for s in seeds)
                    for k in ("draws", "changed", "lost", "brought", "made", "multiple")}
              for arm in ARMS}
    p("Pooled: " + "; ".join(
        f"{arm} {d['draws']} draws, {d['changed']} change a promising Worker "
        f"({pct(share(d['changed'], d['draws']))}), {d['lost']} of those lose a family "
        f"({pct(share(d['lost'], d['changed']))}), {d['brought']} bring the missing one, "
        f"{d['made']} make all three ({pct(share(d['made'], d['changed']))} of the changing "
        f"draws, {pct(share(d['made'], d['draws']))} of all)"
        for arm, d in pooled.items()))

    if any(d["multiple"] for d in pooled.values()):
        p("ANOMALY: a draw changed more than one promising Worker: "
          + ", ".join(f"{a} {d['multiple']}" for a, d in pooled.items()))

    # Criterion 1.
    p("")
    p("== criteria (fixed before the first Run)")
    def until(t):
        return math.inf if t is None else t

    parted = [s for s, r, div in rows
              if div[("W", "N")] is not None
              and div[("W", "N")] < min(until(r["W"]["T"]), until(r["N"]["T"]))]
    early = [s for s, r, _ in rows
             if r["W"]["first_change"] is not None and r["W"]["first_change"] <= 2]
    c1 = len(parted) >= 7 and len(early) == len(seeds)
    p(f"1. Early: W parts from N before either Run passes on {len(parted)} of {len(seeds)} seeds "
      f"(needs 7) {parted}; the weights change a choice in generation 1 or 2 in W on "
      f"{len(early)} of {len(seeds)} (needs all) {early}. {'MET' if c1 else 'FAILED'}.")

    # Criterion 2.
    high = [s for s, r, _ in rows
            if r["W"]["mean_preference"] is not None and r["W"]["mean_preference"] > 1.2]
    low = [s for s, r, _ in rows
           if r["W0"]["mean_preference"] is not None and r["W0"]["mean_preference"] < 1.1]
    c2 = len(high) >= 6 and len(low) >= 6
    p(f"2. Self-adaptation: mean preference above 1.2 in W on {len(high)} of {len(seeds)} seeds "
      f"(needs 6) {high}; below 1.1 in W0 on {len(low)} of {len(seeds)} (needs 6) {low}. "
      f"{'MET' if c2 else 'FAILED'}.")

    # Criterion 3.
    sh = {arm: share(pooled[arm]["lost"], pooled[arm]["changed"]) for arm in ("W", "W0", "N")}
    enough = all(pooled[arm]["changed"] >= 30 for arm in ("W", "W0", "N"))
    tenth = F(1, 10)
    c3 = (enough and sh["W"] <= F(2, 5) and sh["W"] <= sh["W0"] - tenth
          and sh["W"] <= sh["N"] - tenth)
    p(f"3. Protection: of the draws that change a promising Worker, W loses a family in "
      f"{pooled['W']['lost']} of {pooled['W']['changed']} ({pct(sh['W'])}; needs at most 40 %), "
      f"W0 in {pooled['W0']['lost']} of {pooled['W0']['changed']} ({pct(sh['W0'])}), N in "
      f"{pooled['N']['lost']} of {pooled['N']['changed']} ({pct(sh['N'])}); W must lie 10 points "
      f"below W0 and below N; every arm needs 30 such draws "
      f"({'all have' if enough else 'NOT ALL HAVE'}). {'MET' if c3 else 'FAILED'}.")

    # Criterion 4 and the go rule.
    w_n = [s for s, r, _ in rows if before(r["W"]["T"], r["N"]["T"])]
    n_w = [s for s, r, _ in rows if before(r["N"]["T"], r["W"]["T"])]
    w_w0 = [s for s, r, _ in rows if before(r["W"]["T"], r["W0"]["T"])]
    w0_w = [s for s, r, _ in rows if before(r["W0"]["T"], r["W"]["T"])]
    p(f"4. Direction (reported): W passes before N on {len(w_n)} seeds {w_n}, N before W on "
      f"{len(n_w)} {n_w}, ties {len(seeds) - len(w_n) - len(n_w)}; W before W0 on {len(w_w0)} "
      f"{w_w0}, W0 before W on {len(w0_w)} {w0_w}, ties {len(seeds) - len(w_w0) - len(w0_w)}.")

    if c1 and c2 and c3 and len(n_w) <= 4:
        verdict = "GO"
    elif not c1 or not c2:
        verdict = "STOP"
    else:
        verdict = "THE OWNER DECIDES"

    p(f"Verdict by the fixed rule: {verdict} (criteria 1/2/3: {c1}/{c2}/{c3}; N passes before W "
      f"on {len(n_w)} of {len(seeds)}, at most 4 for a go).")

    # Arm WF, for description.
    wf_w = [s for s, r, _ in rows if before(r["WF"]["T"], r["W"]["T"])]
    w_wf = [s for s, r, _ in rows if before(r["W"]["T"], r["WF"]["T"])]
    wf_n = [s for s, r, _ in rows if before(r["WF"]["T"], r["N"]["T"])]
    n_wf = [s for s, r, _ in rows if before(r["N"]["T"], r["WF"]["T"])]
    p("")
    p("== arm WF, for description only")
    p(f"WF passes before W on {len(wf_w)} seeds {wf_w}, W before WF on {len(w_wf)} {w_wf}, ties "
      f"{len(seeds) - len(wf_w) - len(w_wf)}; WF before N on {len(wf_n)} {wf_n}, N before WF on "
      f"{len(n_wf)} {n_wf}, ties {len(seeds) - len(wf_n) - len(n_wf)}.")
    p(f"WF loses a family in {pooled['WF']['lost']} of {pooled['WF']['changed']} draws that change "
      f"a promising Worker ({pct(share(pooled['WF']['lost'], pooled['WF']['changed']))}).")

    p("")
    p("== pass rates within the generations, calls and wall minutes per arm")

    for arm in ARMS:
        rs = [runs[(arm, s)] for s in seeds]
        passed = sum(1 for x in rs if x["T"] is not None)
        minutes = [x["wall_seconds"] / 60 for x in rs]
        p(f"{arm}: passes {passed}/{len(rs)}; calls at most {max(x['calls_used'] for x in rs)}; "
          f"minutes {min(minutes):.1f} to {max(minutes):.1f}, mean {sum(minutes) / len(rs):.1f}")

    p("")
    p("== every Run: verify, calls, wall seconds, deviations, SHA-256 of the record files")

    for s in seeds:
        for arm in ARMS:
            x = runs[(arm, s)]
            name = f"{arm}-{s}"
            h = hashes.get(name, {})
            p(f"{name}: calls {x['calls_used']}, wall {x['wall_seconds']:.1f} s, "
              f"variation choices/changed {x['variation_choices']}/{x['variation_changed']}; "
              f"deviations: {x['deviations'] or 'none'}")
            p(f"  {x['verify']}")

            for file in ("config.json", "events.jsonl", "snapshot.json", "report.json",
                         "rediscovery.json"):
                p(f"  {h.get(file, 'MISSING')}  {file}")

    print("\n".join(out))

    if args.json:
        records = list(runs.values())
        summary = {
            "seeds": seeds,
            "criteria": {"1": c1, "2": c2, "3": c3, "N_before_W": len(n_w), "verdict": verdict},
            "pooled_draws": pooled,
            "divergence": {str(s): {f"{a}-{b}": g for (a, b), g in div.items()}
                           for s, _, div in rows},
            "runs": records,
            "hashes": hashes,
        }

        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=1, sort_keys=True)
            f.write("\n")


if __name__ == "__main__":
    main(sys.argv)
