#!/usr/bin/env python3
"""Build docs/pilot-step2/pilot.json from the pilot Runs of step 2 (ticket 11).

It reads, per Run, config.json, report.json, rediscovery.json, timings.jsonl and events.jsonl
(or events.jsonl.xz), and beside each Run directory the files that the pilot driver wrote:
<run>.wall (wall seconds of the brut-breed process, measured by the driver) and <run>.verify
(the output of brut-breed --verify). The SHA-256 of the five record files of each Run come
from the driver's hashes.txt, written before the two large files were compressed.

It recounts the pass times, the outcomes and the divergence generations with the functions of
reference_rule.py, and it checks them against the rule and divergence tables that
examples/soul_comparison.rs printed for the same Runs (--analysis). Any difference stops it
with an error.

With --rerun it also records the Runs run again for the check of byte-equal files: the
driver's log of that check, which compares the five record files of each rerun with the hashes
of the pilot Run, and the wall seconds of each rerun, which ran one process at a time.

Usage:
  python3 -I -B pilot.py --batch <name>=<dir> [--batch ...] --hashes <hashes.txt>
      --analysis <name>=<soul_comparison output> [--analysis ...] [--rerun <dir>=<rerun.log>]
      --out <pilot.json>
"""

import json
import os
import re
import sys
from fractions import Fraction as F

# python3 -I does not put this folder on sys.path, so add it for the import of
# reference_rule. -B keeps the import from writing bytecode into the repository.
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import reference_rule as rr  # noqa: E402

ARMS = ("S", "V", "K", "E", "N")
FILES = ("config.json", "events.jsonl", "snapshot.json", "report.json", "rediscovery.json")
SOUL_COLUMNS = [
    "generation", "souls", "souls_born", "souls_new", "origins", "origin_entropy_bits",
    "character_norm_mean", "contrast", "contrast_independent", "order_mean", "moved_mean",
    "kappa_min", "kappa_median", "kappa_max", "observations_vote", "observations_peer",
    "shift_mean", "votes", "votes_flipped", "votes_tied", "coherence_mean", "sim_decided",
    "order",
]


def fail(message):
    sys.exit(f"pilot: {message}")


def r4(x):
    return None if x is None else round(x, 4)


def soul_row(generation, s):
    kappa = s.get("kappa") or {}

    return [
        generation, s["souls"], s["souls_born"], s["souls_new"], s["origins"],
        r4(s["origin_entropy_bits"]), r4(s.get("character_norm_mean")), r4(s.get("contrast")),
        r4(s.get("contrast_independent")), r4(s.get("order_mean")), r4(s.get("moved_mean")),
        r4(kappa.get("min")), r4(kappa.get("median")), r4(kappa.get("max")),
        s["observations_vote"], s["observations_peer"], r4(s.get("shift_mean")), s["votes"],
        s["votes_flipped"], s["votes_tied"], r4(s.get("coherence_mean")), s["sim_decided"],
        [r4(x) for x in s.get("order") or []],
    ]


def read_states(path):
    """The soul group of every state line, by generation, and the numbers of `souls` events
    and of state lines with a soul group."""
    states = {}
    souls_events = 0

    with rr.open_events(path) as f:
        for line in f:
            head = line[:200]

            if '"kind":"souls"' in head:
                souls_events += 1

            if '"kind":"state"' not in head:
                continue

            event = json.loads(line)["event"]

            if event["kind"] == "state" and event.get("soul") is not None:
                states[event["generation"]] = event["soul"]

    return states, souls_events


def read_verify(path):
    with open(path + ".verify") as f:
        last = f.read().strip().splitlines()[-1]

    m = re.match(r"brut-breed: (\d+) events, hash chain intact, head ([0-9a-f]{64}); (.*)$", last)

    if not m or not last.endswith("report recomputed and equal; rediscovery recomputed and equal"):
        fail(f"{path}: --verify did not end as required: {last}")

    return {"events": int(m.group(1)), "head": m.group(2), "checks": m.group(3)}


def read_hashes(path):
    hashes = {}

    with open(path) as f:
        for line in f:
            digest, name = line.split()
            run, _, file = name.partition("/")
            hashes.setdefault(run, {})[file] = digest

    return hashes


def run_record(batch, path, arm, seed, hashes):
    run = rr.read_run(path, arm, seed)
    red = rr.load(os.path.join(path, "rediscovery.json"))
    name = f"{arm}-{seed}"

    if set(hashes.get(name, {})) != set(FILES):
        fail(f"{name}: hashes.txt holds {sorted(hashes.get(name, {}))}")

    with open(path + ".wall") as f:
        wall = float(f.read())

    seconds = 0.0

    with open(os.path.join(path, "timings.jsonl")) as f:
        for line in f:
            if line.strip():
                seconds += json.loads(line)["seconds"]

    planned = run["report"]["generations_completed"]
    open_targets = [t for t in red["targets"] if not t["reached_by_seed"]]
    hit = sum(1 for t in open_targets if t["reached_by_bred"])
    states, souls_events = read_states(path)

    # Arm N is step 1 without a Soul: no soul field, no souls event and no soul group.
    if arm == "N" and ("soul" in run["cfg"] or souls_events or states):
        fail(f"{name}: a Run without Souls with a soul field, {souls_events} souls events and "
             f"{len(states)} soul groups")

    if arm != "N" and (souls_events != planned or len(states) != planned):
        fail(f"{name}: {souls_events} souls events and {len(states)} soul groups for "
             f"{planned} generations")
    first_flip = first_sim = None

    for g in sorted(states):
        if first_flip is None and states[g]["votes_flipped"] > 0:
            first_flip = g

        if first_sim is None and states[g]["sim_decided"] > 0:
            first_sim = g

    record = {
        "arm": arm,
        "seed": seed,
        "batch": batch,
        "dir": name,
        "generations": run["cfg"]["generations"],
        "max_calls": run["cfg"]["max_calls"],
        "soul": run["cfg"].get("soul"),
        "generations_completed": run["report"]["generations_completed"],
        "stop_reason": run["report"]["stop_reason"],
        "calls_used": run["report"]["calls_used"],
        "wall_seconds": wall,
        "minutes": round(wall / 60, 4),
        "minutes_timings": round(seconds / 60, 4),
        "r_seed": sorted(run["r_seed"]),
        "d": f"{hit}/{len(open_targets)}",
        "d_value": round(hit / len(open_targets), 4),
        "T": run["T"],
        "passes": run["T"] is not None,
        "first_generation": {t["target"]: t["first_generation"] for t in red["targets"]},
        "votes": run["votes"],
        "souls_events": souls_events,
        "soul_groups": len(states),
        "first_flip": first_flip,
        "first_sim_decided": first_sim,
        "verify": read_verify(path),
        "sha256": hashes[name],
        "soul_every_10th": [soul_row(g, states[g]) for g in sorted(states) if g % 10 == 0],
    }

    return record, run


def parse_analysis(path):
    """The rule rows and the divergence rows of a soul_comparison output, by seed."""
    rule, div = {}, {}
    section = None

    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")

            if line.startswith(" seed    T_S"):
                section = "rule"
            elif line.startswith(" seed    S-V"):
                section = "div"
            elif not line.strip() or not line.strip()[0].isdigit():
                if section and not line.startswith(" "):
                    section = None
            elif section == "rule" and "after the stop" not in line:
                rule[int(line.split()[0])] = line
            elif section == "div":
                div[int(line.split()[0])] = line

    return rule, div


def read_rerun(rerun, hashes):
    """The reruns: per Run, whether all five record files equal the pilot's, and the wall
    seconds of the rerun, which ran alone."""
    d, log = rerun
    equal = {}

    with open(log) as f:
        for line in f:
            m = re.match(r"(\S+)/(\S+) (equal|DIFFERS) ", line)

            if m:
                equal.setdefault(m.group(1), []).append(m.group(3) == "equal")

    out = {}

    for name, flags in sorted(equal.items()):
        if len(flags) != len(FILES) or set(hashes.get(name, {})) != set(FILES):
            fail(f"rerun {name}: {len(flags)} files compared")

        with open(os.path.join(d, name + ".wall")) as f:
            wall = float(f.read())

        out[name] = {"byte_equal": all(flags), "wall_seconds": wall,
                     "minutes": round(wall / 60, 4), "verify": read_verify(os.path.join(d, name))}

    return out


def main(argv):
    batches, analyses, hashes_path, out, rerun = [], [], None, None, None
    args = iter(argv[1:])

    for a in args:
        if a == "--rerun":
            d, _, log = next(args).partition("=")
            rerun = (d, log)
        elif a == "--batch":
            name, _, d = next(args).partition("=")
            batches.append((name, d))
        elif a == "--analysis":
            name, _, p = next(args).partition("=")
            analyses.append((name, p))
        elif a == "--hashes":
            hashes_path = next(args)
        elif a == "--out":
            out = next(args)
        else:
            fail(f"unknown argument {a}")

    hashes = read_hashes(hashes_path)
    runs, seeds = [], {}

    for name, d in batches:
        found = sorted({int(n.split("-")[1]) for n in os.listdir(d)
                        if re.fullmatch(r"[SVKEN]-\d+", n) and os.path.isdir(os.path.join(d, n))})

        for seed in found:
            per_arm = {}

            for arm in ARMS:
                path = os.path.join(d, f"{arm}-{seed}")

                if os.path.isdir(path):
                    record, run = run_record(name, path, arm, seed, hashes)
                    runs.append(record)
                    per_arm[arm] = run

            seeds[seed] = (name, per_arm)

    per_seed = []

    for seed, (batch, r) in sorted(seeds.items()):
        entry = {"seed": seed, "batch": batch,
                 "T": {arm: r[arm]["T"] for arm in ARMS if arm in r}}

        for arm in ("V", "K", "E", "N"):
            if arm in r:
                entry[f"Y_{arm}"] = rr.first(r["S"]["T"], r[arm]["T"], arm)

        entry["divergence"] = {}

        for x, y in rr.PAIRS:
            if x in r and y in r:
                entry["divergence"][f"{x}-{y}"] = rr.divergence(r[x]["born"], r[y]["born"])

        per_seed.append(entry)

    # The rule and divergence tables of soul_comparison must match the recount, row by row.
    for name, path in analyses:
        rule, div = parse_analysis(path)
        batch_seeds = sorted(s for s, (b, _) in seeds.items() if b == name)
        lines = rr.rule([(s, seeds[s][1]) for s in batch_seeds],
                        (False, "the seeds are pilot seeds of step 2 (901 to 923), which never count"))
        mine = {int(line.split()[0]): line for line in lines
                if line[:5].strip().isdigit() and "after the stop" not in line}
        theirs_div = {}

        for line in rr.describe([(s, seeds[s][1]) for s in batch_seeds]):
            if line[:5].strip().isdigit() and ":" not in line:
                theirs_div[int(line.split()[0])] = line

        if mine != rule:
            fail(f"{name}: the rule rows of {path} differ from the recount:\n"
                 + "\n".join(f"{s}: {rule.get(s)!r} / {mine.get(s)!r}" for s in sorted(set(mine) | set(rule))
                             if rule.get(s) != mine.get(s)))

        if theirs_div != div:
            fail(f"{name}: the divergence rows of {path} differ from the recount")

    def votes_of(arm):
        v = {"runs": 0, "complex": 0, "flipped": 0, "tied": 0, "sim_decided": 0}

        for rec in runs:
            if rec["arm"] == arm:
                v["runs"] += 1

                for k in ("complex", "flipped", "tied", "sim_decided"):
                    v[k] += rec["votes"][k]

        v["flipped_share"] = round(v["flipped"] / v["complex"], 4) if v["complex"] else None
        v["tied_share"] = round(v["tied"] / v["complex"], 4) if v["complex"] else None

        return v

    per_arm = {}

    for arm in ARMS:
        recs = [rec for rec in runs if rec["arm"] == arm]

        if not recs:
            continue

        minutes = [rec["minutes"] for rec in recs]
        per_arm[arm] = {
            "runs": len(recs),
            "passes": sum(rec["passes"] for rec in recs),
            "votes": votes_of(arm),
            "minutes_mean": round(sum(minutes) / len(minutes), 4),
            "minutes_min": min(minutes),
            "minutes_max": max(minutes),
            "calls_max": max(rec["calls_used"] for rec in recs),
        }

    counts = {}

    for arm in ("V", "K", "E", "N"):
        ys = [e[f"Y_{arm}"] for e in per_seed if f"Y_{arm}" in e]
        counts[arm] = {"S_first": ys.count("S first"), "other_first": ys.count(f"{arm} first"),
                       "tie": ys.count("tie")}

    joint = {}

    for e in per_seed:
        key = f"{e['Y_V']}|{e['Y_K']}"
        joint[key] = joint.get(key, 0) + 1

    pilot = {
        "what": "Pilot Runs of the step 2 comparison (ticket 11 of .scratch/seele-und-kappa/). "
                "Pilot seeds 901 to 923 never count.",
        "counted": False,
        "notes": [
            "T is the pass time of step 1 within generation 1000, null for never.",
            "minutes: wall time of the brut-breed process as the pilot driver measured it, two "
            "Runs at a time on 4 cores, RAYON_NUM_THREADS=2 each; minutes_timings: the sum of "
            "timings.jsonl, which leaves out the start and the select phase.",
            "divergence: the first generation whose newborns differ, member by member in birth "
            "order, in gene id, operator, parents (by gene id), answers, solved count or graded "
            "value; null when no generation differs.",
            "votes: the sums over the born events of the vote tallies (complex, flipped, tied, "
            "sim_decided); first_flip and first_sim_decided: the first generation whose state "
            "line counts one.",
            "soul_every_10th: the soul group of the state line of every 10th generation, one row "
            "per line in the order of soul_columns; floats rounded to 4 decimal places.",
            "sha256: the five record files, hashed by the driver before events.jsonl and "
            "snapshot.json were compressed with xz.",
        ],
        "soul_columns": SOUL_COLUMNS,
        "per_arm": per_arm,
        "counts": counts,
        "joint_Y_V_Y_K": joint,
        "rerun": read_rerun(rerun, hashes) if rerun else None,
        "per_seed": per_seed,
        "runs": runs,
    }

    # One line per row of soul_every_10th, so that the file stays readable and small.
    rows = []

    for rec in runs:
        for i, row in enumerate(rec["soul_every_10th"]):
            rows.append(json.dumps(row, separators=(",", ":")))
            rec["soul_every_10th"][i] = f"@row{len(rows) - 1}@"

    text = json.dumps(pilot, indent=1)
    text = re.sub(r'"@row(\d+)@"', lambda m: rows[int(m.group(1))], text)

    with open(out, "w") as f:
        f.write(text + "\n")

    print(f"{len(runs)} runs, {len(per_seed)} seeds")

    for arm, a in per_arm.items():
        v = a["votes"]
        print(f"arm {arm}: passes {a['passes']}/{a['runs']}; votes {v['complex']}, flipped "
              f"{v['flipped']} ({v['flipped_share']}), tied {v['tied']} ({v['tied_share']}), "
              f"sim_decided {v['sim_decided']}; minutes {a['minutes_mean']} "
              f"({a['minutes_min']} to {a['minutes_max']}); calls at most {a['calls_max']}")

    for arm, c in counts.items():
        print(f"S against {arm}: S first {c['S_first']}, {arm} first {c['other_first']}, "
              f"tie {c['tie']}")

    print("joint (Y_V|Y_K):", json.dumps(joint, sort_keys=True))

    for e in per_seed:
        print(e["seed"], e["T"], e.get("Y_V"), e.get("Y_K"), e["divergence"])


if __name__ == "__main__":
    main(sys.argv)
