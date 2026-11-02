#!/usr/bin/env python3
"""Why and when two arms of a pilot seed part (ticket 11, the diagnosis of the pilot).

For each seed and pair of arms it finds the divergence generation of section 12 of the
draft (as reference_rule.py does), the first newborn that differs there and the fields in
which it differs, and the vote tallies of that newborn in both Runs: complex votes, flipped
votes (the winner differs from the count's), tied votes (the tie rule decided) and replies
in which the phases decided a simulated model's choice (sim_decided). By spec 15 the arms
share every random number until the Soul changes an outcome, so a first difference needs a
flipped vote or a phase-decided reply in that newborn, in one of the two Runs. The script
checks this, and it reports the first generation with a flipped vote, with a phase-decided
reply and with a complex vote of any kind in each Run.

Usage: python3 -I -B diagnose.py <first>-<last> <dir>...
"""

import json
import os
import sys

FIELDS = ("gene id", "operator", "parents", "answers", "solved", "graded value")
PAIRS = [("S", "V"), ("S", "K"), ("V", "K"), ("S", "E"), ("S", "N")]

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import reference_rule as rr  # noqa: E402


def read(path):
    """Per generation the newborns as (signature, tally), and the first generations with a
    complex vote, a flipped vote and a phase-decided reply."""
    genes = {}
    born = {}
    firsts = {"vote": None, "flip": None, "sim": None}

    with rr.open_events(path) as f:
        for line in f:
            event = json.loads(line)["event"]

            if event["kind"] != "born":
                continue

            gid = rr.gene_id(event["cascade"])
            signature = (gid, event["op"], tuple(genes[p] for p in event["parents"]),
                         json.dumps(event.get("answers")), event["solved"], event.get("quality"))
            tally = event.get("votes") or {}
            born.setdefault(event["generation"], []).append((signature, tally))
            genes[event["id"]] = gid
            g = event["generation"]

            for key, field in (("vote", "complex"), ("flip", "flipped"), ("sim", "sim_decided")):
                if firsts[key] is None and tally.get(field, 0) > 0:
                    firsts[key] = g

    return born, firsts


def main(argv):
    lo, _, hi = argv[1].partition("-")
    seeds = range(int(lo), int(hi or lo) + 1)
    dirs = argv[2:]
    explained = unexplained = 0

    for seed in seeds:
        runs = {}

        for arm in ("S", "V", "K", "E", "N"):
            path = rr.find(dirs, f"{arm}-{seed}")

            if path is not None:
                runs[arm] = read(path)

        line = [f"seed {seed}:"]

        for arm, (_, firsts) in runs.items():
            line.append(f"{arm} vote/flip/sim {firsts['vote']}/{firsts['flip']}/{firsts['sim']}")

        print(" ".join(line))

        for x, y in PAIRS:
            if x not in runs or y not in runs:
                continue

            a, b = runs[x][0], runs[y][0]
            g = rr.divergence({k: [s for s, _ in v] for k, v in a.items()},
                              {k: [s for s, _ in v] for k, v in b.items()})

            if g is None:
                print(f"  {x}-{y}: no divergence")
                continue

            members = list(zip(a.get(g, []), b.get(g, [])))
            index = next((i for i, (u, v) in enumerate(members) if u[0] != v[0]), len(members))

            if index == len(members):
                print(f"  {x}-{y}: diverge at {g}: another number of newborns")
                continue

            (sa, ta), (sb, tb) = members[index]
            fields = [name for name, u, v in zip(FIELDS, sa, sb) if u != v]
            moved = any(t.get("flipped", 0) or t.get("sim_decided", 0) for t in (ta, tb))
            explained += moved
            unexplained += not moved
            show = lambda t: (f"{t.get('complex', 0)}/{t.get('flipped', 0)}/{t.get('tied', 0)}/"
                              f"{t.get('sim_decided', 0)}")
            print(f"  {x}-{y}: diverge at {g}, newborn {index}, in {', '.join(fields)}; "
                  f"complex/flipped/tied/sim {x} {show(ta)}, {y} {show(tb)}; "
                  f"{'a flipped vote or a phase-decided reply' if moved else 'NOT EXPLAINED'}")

    print(f"first differences with a flipped vote or a phase-decided reply in that newborn: "
          f"{explained}; without: {unexplained}")


if __name__ == "__main__":
    main(sys.argv)
