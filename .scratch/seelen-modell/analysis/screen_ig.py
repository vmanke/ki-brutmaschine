#!/usr/bin/env python3
"""The screen of arm IG (ticket 06 of `.scratch/seelen-modell/`): every number of its criteria.

Usage: python3 -I -B screen_ig.py --runs DIR --logs DIR --configs DIR [--seeds 5001-5008]
       [--generations 1000] [--json PATH]

Per Run <arm>-<seed> in --runs (arms N, G, I and IG) it reads config.json, report.json and
rediscovery.json, and events.jsonl.xz (or events.jsonl) once from start to end through
Python's lzma module; it never writes a decompressed file. From --logs it reads hashes.txt, the
SHA-256 of the five record files that the driver took after --verify and before xz, and
driver.log, the exit codes of each Run and of its --verify. Beside each Run it reads the
driver's <run>.wall (wall seconds) and <run>.verify (the --verify output). From --configs it
reads config-N.json, config-G.json, config-I.json and config-IG.json. It prints the report and,
with --json, writes the per-Run records.

Every definition below was fixed before the first Run of the screen.

- Settings. config-N.json is the config.json that brut-breed writes for arm N with the flags of
  ticket 06; config-G.json, config-I.json and config-IG.json are config-N.json with
  "goals": {"length": 20}, with "imagination": {"draws": 4, "rate": 250}, and with both. A Run
  of arm A on seed s must hold config-A.json with "seed" set to s. config-N.json must hold the
  selection lexicase, --generations generations, 13 000 000 calls, 33 search tasks, 22 select
  tasks and 8 children, and no soul, imagination or goals. A Run must stop with "all
  generations done" after --generations + 1 generations and log newborns in each of them, and
  the driver must log exit 0 for the Run and for its --verify. Every deviation is printed.
- The pass time T, the divergence generation of two Runs and "A passes before B" are those of
  `analysis/screen_w.py`: T is `reference_rule.pass_time`, "never" when the Run does not pass or
  T > --generations; the divergence generation is the first generation whose newborns differ,
  member by member in birth order, in gene id, operator, parents (by gene id), answers, solved
  count or graded value; "A passes before B" is T_A < T_B with "never" later than every
  generation, and equal pass times, or two Runs that never pass, are a tie.
- Families. A Worker names a family when one of its section texts, lower-cased, holds one of
  the family's stems in `designer::FAMILY_STEMS` (all eight, read from src/cascade/designer.rs),
  as `designer::named_families` reads them. Linear, Phase and Hadamard are families 0, 1 and
  2; the other five are families 3 to 7.
- Learned values. Each member's current values start as the "values" of its Born event and
  become, at each "learned" event in log order, the values that event lists for one of its
  Workers. A generation's "learned" event follows all its Born events, so at a Born event of
  generation g every member holds its values after the commits of the generations before g:
  the values with which it was chosen as a parent.
- Criterion 1, early. The first changed pick of a Run is the smallest generation of a newborn
  whose Born event holds an effect with "draw" 2 or more; None when there is none. It holds
  when the first changed pick of IG is at most 5 on all seeds. Check: in every generation the
  number of such newborns equals the state line's mind.imagination_changed; a difference is an
  ANOMALY.
- Criterion 2, learning. For every newborn of generations 41 to 50, for each id in its parents
  (a parent chosen for several children counts each time, and both parents of a crossover
  count), for each Worker of that parent: its current values of Linear, Phase and Hadamard go
  into one mean, its five other values into the other. Both means are exact fractions of
  thousandths. It holds when the first mean is higher than the second on at least 6 of the 8
  seeds of IG. Check: the same sums over the newborns of each generation give the state line's
  mind.linear_phase_hadamard_mean and mind.other_families_mean of that generation, as the same
  floats; a difference is an ANOMALY.
- Criterion 3, imagination measures the effect. Among the newborns of a Run whose Born event
  holds an effect with p other than 0 (every newborn with p other than 0 holds one), D is the
  mean s of those with p > 0 minus the mean s of those with p < 0, an exact fraction. It holds
  when D > 0 on at least 7 of the 8 seeds of IG; a seed with fewer than 5 newborns in either
  group counts as not above. The pooled numbers over the seeds of an arm are reported: the
  size and the mean s of each group, and their difference.
- Criterion 4, protection, with the definitions of `analysis/screen_w.py`. A promising Worker
  names exactly two of Linear, Phase and Hadamard. The draws on a parent with a promising
  Worker are the newborns made by SectionReplace or SectionAdd whose parent holds a promising
  Worker, in log order, up to and including the newborn that holds the Run's first Worker with
  all three; every such newborn when the Run has none. A draw changes a promising Worker when
  the sections of the child's Worker at the place of a promising Worker of the parent differ
  from that Worker's (these operators keep places; the first such place counts). Such a draw
  loses one of its families when that child Worker no longer names both families of the
  parent's Worker among the three. The counts are pooled over the seeds of an arm, and a share
  is lost / changed, an exact fraction. It holds when IG and N each have at least 30 draws that
  change a promising Worker, IG's share is at most 2/5, and IG's share is at most N's share
  minus 1/10.
- Criterion 5, direction, reported: the seeds on which IG passes before N, I before N, G before
  N, IG before I, and IG before G, and those of the reverse and of the ties.
- Verdict: GO if criteria 1, 2, 3 and 4 hold and IG passes before N on at least 5 of the 8
  seeds; STOP if 1, 2 or 3 fails; in every other case the owner decides.
- Description, not tested: the numbers of criteria 1 to 4 for arm I, of criterion 4 for arm
  G, the pass times of all arms, the divergence generations of IG, I and G from N, the
  imagination choices and changed picks of arms I and IG, and the goals of arms G and IG: the
  goal of each block of generations, which must last 20 generations from generation 1 on,
  alternate between all tasks and a goal without one family, and be the same in G and IG on a
  seed.
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

ARMS = ("N", "G", "I", "IG")
FILES = ("config.json", "events.jsonl", "snapshot.json", "report.json", "rediscovery.json")
THREE = frozenset((0, 1, 2))
PHRASE_OPS = ("section_replace", "section_add")
MAX_CALLS = 13_000_000
PARENT_WINDOW = range(41, 51)
EARLY = 5
GOAL_LENGTH = 20
ADDED = {
    "N": {},
    "G": {"goals": {"length": GOAL_LENGTH}},
    "I": {"imagination": {"draws": 4, "rate": 250}},
    "IG": {"imagination": {"draws": 4, "rate": 250}, "goals": {"length": GOAL_LENGTH}},
}


def stem_lists():
    """The stems of the eight families, in the order of designer::FAMILY_STEMS."""
    path = os.path.join(REPO, "src", "cascade", "designer.rs")

    with open(path, encoding="utf-8") as f:
        text = f.read()

    start = text.index("pub const FAMILY_STEMS")
    block = text[start:text.index("];", start)]
    families = re.findall(r'\(\s*"(\w+)",\s*&\[(.*?)\]', block, re.S)
    names = [name for name, _ in families]

    if names[:3] != ["Linear", "Phase", "Hadamard"] or len(names) != 8:
        sys.exit(f"screen_ig: unexpected FAMILY_STEMS in {path}: {names}")

    return names, [re.findall(r'"([^"]*)"', inner) for _, inner in families]


def named(worker, stems):
    """The families a Worker names, by their places in FAMILY_STEMS."""
    texts = [s["text"].lower() for s in worker["sections"]]

    return frozenset(f for f, family in enumerate(stems)
                     if any(stem in t for t in texts for stem in family))


def read_text(path):
    try:
        with open(path, encoding="utf-8") as f:
            return f.read().strip()
    except OSError:
        return None


def load_configs(configs, generations):
    """config-<arm>.json of every arm, after the checks of the docstring."""
    out = {}
    problems = []

    for arm in ARMS:
        out[arm] = rr.load(os.path.join(configs, f"config-{arm}.json"))

    base = out["N"]
    wanted = {"selection": "lexicase", "generations": generations, "max_calls": MAX_CALLS,
              "search_puzzles": 33, "select_puzzles": 22, "children_per_generation": 8}

    for key, value in wanted.items():
        if base.get(key) != value:
            problems.append(f"config-N.json: {key} is {base.get(key)!r}, not {value!r}")

    for key in ("soul", "imagination", "goals"):
        if key in base:
            problems.append(f"config-N.json holds {key}")

    for arm in ARMS:
        if out[arm] != {**base, **ADDED[arm]}:
            problems.append(f"config-{arm}.json is not config-N.json with {ADDED[arm]}")

    return out, problems


def read_driver_log(logs):
    """Per Run the exit codes the driver logged for the Run and for its --verify."""
    done, verify = {}, {}

    try:
        with open(os.path.join(logs, "driver.log"), encoding="utf-8") as f:
            for line in f:
                m = re.search(r" done (\S+) exit=(\d+)", line)

                if m:
                    done[m.group(1)] = int(m.group(2))

                m = re.search(r" verify (\S+) exit=(\d+)", line)

                if m:
                    verify[m.group(1)] = int(m.group(2))
    except OSError:
        pass

    return done, verify


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


def goal_blocks(goals, generations):
    """The goals as blocks [name, first generation, last generation] from generation 1 on."""
    blocks = []

    for g in range(1, generations + 1):
        name = goals.get(g)

        if blocks and blocks[-1][0] == name:
            blocks[-1][2] = g
        else:
            blocks.append([name, g, g])

    return blocks


def goal_problems(goals, generations):
    """What breaks the rule of the goals (docstring, description)."""
    problems = []

    if goals.get(0) != "all":
        problems.append(f"generation 0 has the goal {goals.get(0)!r}")

    missing = [g for g in range(generations + 1) if g not in goals]

    if missing:
        problems.append(f"no goal in {len(missing)} generations, first {missing[0]}")
        return problems

    # Without a goal that leaves out a family, every block is all tasks.
    variants = any(name != "all" for name in goals.values())

    for g in range(1, generations + 1):
        k = (g - 1) // GOAL_LENGTH
        name = goals[g]
        start = goals[k * GOAL_LENGTH + 1]

        if name != start:
            problems.append(f"generation {g} has {name!r} inside a block of {start!r}")
            break

        if variants and k % 2 == 1:
            right = isinstance(name, str) and name.startswith("-")
        else:
            right = name == "all"

        if not right:
            problems.append(f"block {k + 1} (generation {g}) has {name!r}")
            break

    return problems


def read_run(runs, arm, seed, generations, stems, expected, done, verify):
    label = f"{arm}-{seed}"
    path = os.path.join(runs, label)
    cfg = rr.load(os.path.join(path, "config.json"))
    report = rr.load(os.path.join(path, "report.json"))
    red = rr.load(os.path.join(path, "rediscovery.json"))
    deviations = []

    want = {**expected, "seed": seed}

    if cfg != want:
        keys = sorted(k for k in set(cfg) | set(want) if cfg.get(k) != want.get(k))
        deviations.append(f"settings differ from arm {arm} in {keys}")

    if (report["stop_reason"] != rr.ALL_GENERATIONS
            or report["generations_completed"] != generations + 1):
        deviations.append(f"stopped with {report['stop_reason']!r} after "
                          f"{report['generations_completed']} generations")

    if done.get(label) != 0:
        deviations.append(f"the driver logged the Run's exit as {done.get(label)}")

    if verify.get(label) != 0:
        deviations.append(f"the driver logged --verify's exit as {verify.get(label)}")

    t = rr.pass_time(red, path)
    t = t if t is not None and t <= generations else None

    imagines = "imagination" in cfg
    gene = {}
    born = {}
    members = {}
    values = {}
    anomalies = []
    goals = {}
    state = {}
    # Criterion 1.
    changed_per_g = {}
    # Criterion 2: per generation the sums and counts of the parents' values.
    sums = {}
    # Criterion 3: p and s of every newborn with p other than 0.
    effects = []
    draws = {"draws": 0, "changed": 0, "lost": 0, "brought": 0, "made": 0, "multiple": 0}
    first_three = None

    with rr.open_events(path) as f:
        for line in f:
            event = json.loads(line)["event"]
            kind = event["kind"]

            if kind == "state":
                mind = event.get("mind")

                if mind is not None:
                    goals[event["generation"]] = mind.get("goal")
                    state[event["generation"]] = mind
                continue

            if kind == "learned":
                for c in event["changed"]:
                    if c["member"] not in values or c["worker"] >= len(values[c["member"]]):
                        anomalies.append(f"generation {event['generation']}: learned names "
                                         f"Worker {c['worker']} of {c['member']}")
                        continue

                    values[c["member"]][c["worker"]] = list(c["values"])
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
            info = [(json.dumps(w["sections"], sort_keys=True), named(w, stems)) for w in workers]

            if imagines:
                effect = event.get("effect")

                if event.get("values") is None or len(event["values"]) != len(workers):
                    anomalies.append(f"generation {g}: {event['id']} logs no values for each "
                                     f"Worker")

                if effect is not None:
                    if effect.get("draw", 1) >= 2:
                        changed_per_g[g] = changed_per_g.get(g, 0) + 1

                    if effect["p"] != 0:
                        effects.append((effect["p"], effect["s"]))

                total = sums.setdefault(g, [0, 0, 0, 0])

                for p in parents:
                    for v in values.get(p, []):
                        total[0] += sum(v[:3])
                        total[1] += len(v[:3])
                        total[2] += sum(v[3:])
                        total[3] += len(v[3:])

                values[event["id"]] = [list(v) for v in event.get("values") or []]

            if first_three is None:
                if event["op"] in PHRASE_OPS:
                    parent = members[parents[0]]
                    promising = [k for k, (_, fams) in enumerate(parent)
                                 if len(fams & THREE) == 2]

                    if promising:
                        draws["draws"] += 1
                        changed = [k for k in promising
                                   if k < len(info) and info[k][0] != parent[k][0]]

                        if len(changed) > 1:
                            draws["multiple"] += 1

                        if changed:
                            k = changed[0]
                            before, after = parent[k][1] & THREE, info[k][1] & THREE
                            draws["changed"] += 1
                            draws["lost"] += not before <= after
                            draws["brought"] += (THREE - before) <= after
                            draws["made"] += THREE <= after

                if any(THREE <= fams for _, fams in info):
                    first_three = {"generation": g, "op": event["op"]}

            members[event["id"]] = info

    if len(born) != report["generations_completed"]:
        deviations.append(f"newborns in {len(born)} generations, report "
                          f"{report['generations_completed']}")

    first_change = min(changed_per_g, default=None)
    means = None
    choices = changed_total = None

    if imagines:
        for g in range(generations + 1):
            mind = state.get(g, {})

            if mind.get("imagination_changed") != changed_per_g.get(g, 0):
                anomalies.append(f"generation {g}: {changed_per_g.get(g, 0)} newborns with a "
                                 f"changed pick, mind says {mind.get('imagination_changed')}")

            a, na, b, nb = sums.get(g, [0, 0, 0, 0])
            lph = a / na if na else None
            other = b / nb if nb else None

            if (mind.get("linear_phase_hadamard_mean") != lph
                    or mind.get("other_families_mean") != other):
                anomalies.append(f"generation {g}: parents' means {lph}, {other}; mind says "
                                 f"{mind.get('linear_phase_hadamard_mean')}, "
                                 f"{mind.get('other_families_mean')}")

        window = [sums.get(g, [0, 0, 0, 0]) for g in PARENT_WINDOW]
        a, na, b, nb = (sum(w[i] for w in window) for i in range(4))
        means = {"lph": F(a, na) if na else None, "other": F(b, nb) if nb else None,
                 "lph_values": na, "other_values": nb}
        choices = sum(m.get("imagination_choices", 0) for m in state.values())
        changed_total = sum(changed_per_g.values())

    plus = [s for p, s in effects if p > 0]
    minus = [s for p, s in effects if p < 0]

    if "goals" in cfg:
        problems = goal_problems(goals, generations)
        deviations.extend(f"goals: {why}" for why in problems)

    return {
        "arm": arm,
        "seed": seed,
        "T": t,
        "passes": red["passes"],
        "calls_used": report["calls_used"],
        "wall_seconds": float(read_text(path + ".wall") or "nan"),
        "verify": read_text(path + ".verify"),
        "first_change": first_change,
        "imagination_choices": choices,
        "imagination_changed": changed_total,
        "means": means,
        "plus": (len(plus), sum(plus)),
        "minus": (len(minus), sum(minus)),
        "draws": draws,
        "first_three": first_three,
        "goal_blocks": goal_blocks(goals, generations) if "goals" in cfg else None,
        "deviations": deviations,
        "anomalies": anomalies,
        "born": born,
    }


def before(a, b):
    """True when a passes before b; None is never."""
    return a is not None and (b is None or a < b)


def show(t):
    return "never" if t is None else str(t)


def share(lost, changed):
    return F(lost, changed) if changed else None


def pct(x):
    return "-" if x is None else f"{100 * float(x):.1f} %"


def frac(x):
    return "-" if x is None else f"{float(x):.2f}"


def last_line(text):
    lines = (text or "").splitlines()

    return lines[-1][:60] if lines else "missing"


def difference(plus, minus, floor=5):
    """D = mean s(p > 0) − mean s(p < 0), and whether it counts as above 0."""
    (n_plus, s_plus), (n_minus, s_minus) = plus, minus

    if n_plus == 0 or n_minus == 0:
        return None, False

    d = F(s_plus, n_plus) - F(s_minus, n_minus)

    return d, n_plus >= floor and n_minus >= floor and d > 0


def to_json(r):
    out = dict(r)
    m = out.get("means")

    if m is not None:
        out["means"] = {k: (None if v is None else str(v)) if isinstance(v, F) or v is None
                        else v for k, v in m.items()}

    return out


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--logs", required=True)
    ap.add_argument("--configs", required=True)
    ap.add_argument("--seeds", default="5001-5008")
    ap.add_argument("--generations", type=int, default=1000)
    ap.add_argument("--json")
    args = ap.parse_args(argv[1:])
    lo, _, hi = args.seeds.partition("-")
    seeds = list(range(int(lo), int(hi or lo) + 1))
    names, stems = stem_lists()
    configs, config_problems = load_configs(args.configs, args.generations)
    hashes = read_hashes(args.logs)
    done, verify = read_driver_log(args.logs)
    runs = {}
    rows = []

    # One seed at a time, so that only four Runs' newborns are held at once.
    for s in seeds:
        r = {arm: read_run(args.runs, arm, s, args.generations, stems, configs[arm], done,
                           verify) for arm in ARMS}
        div = {pair: rr.divergence(r[pair[0]]["born"], r[pair[1]]["born"])
               for pair in (("IG", "N"), ("I", "N"), ("G", "N"), ("IG", "I"), ("IG", "G"))}

        for arm in ARMS:
            del r[arm]["born"]
            runs[(arm, s)] = r[arm]

        rows.append((s, r, div))

    out = []
    p = out.append

    p(f"== screen of arm IG: seeds {seeds[0]} to {seeds[-1]}, arms {', '.join(ARMS)}, "
      f"{args.generations} generations")
    p("")
    p("Runs: pass time T, calls used, wall seconds, five hashes recorded, --verify.")

    for s, r, _ in rows:
        for arm in ARMS:
            x = r[arm]
            label = f"{arm}-{s}"
            got = hashes.get(label, {})
            recorded = all(f in got for f in FILES)
            p(f"  {label:>8}  T {show(x['T']):>5}  calls {x['calls_used']:>9}  wall "
              f"{x['wall_seconds']:8.1f} s  hashes {'5/5' if recorded else f'{len(got)}/5'}  "
              f"verify: {last_line(x['verify'])}")

    problems = list(config_problems)

    for (arm, s), x in runs.items():
        problems.extend(f"{arm}-{s}: {d}" for d in x["deviations"])

        if not all(f in hashes.get(f"{arm}-{s}", {}) for f in FILES):
            problems.append(f"{arm}-{s}: not all five hashes recorded")

    for s, r, _ in rows:
        if r["G"]["goal_blocks"] != r["IG"]["goal_blocks"]:
            problems.append(f"seed {s}: the goals of G and IG differ")

    anomalies = [f"{arm}-{s}: {a}" for (arm, s), x in runs.items() for a in x["anomalies"]]

    p("")
    p("Deviations: " + ("none" if not problems else ""))

    for d in problems:
        p(f"  DEVIATION {d}")

    p("Anomalies: " + ("none" if not anomalies else ""))

    for a in anomalies[:40]:
        p(f"  ANOMALY {a}")

    if len(anomalies) > 40:
        p(f"  ... and {len(anomalies) - 40} more")

    p("")
    p("Per seed: pass times; divergence generations; first changed pick (I, IG); mean values "
      "of the parents' Workers in generations 41 to 50, Linear/Phase/Hadamard against the "
      "other five, in thousandths (I, IG).")
    p(f"{'seed':>5} {'T_N':>5} {'T_G':>5} {'T_I':>5} {'T_IG':>5} | {'IG-N':>4} {'I-N':>4} "
      f"{'G-N':>4} | {'pick I':>6} {'IG':>4} | {'I lph':>7} {'other':>7} | {'IG lph':>7} "
      f"{'other':>7}")

    for s, r, div in rows:
        mi, mig = r["I"]["means"] or {}, r["IG"]["means"] or {}
        p(f"{s:>5} {show(r['N']['T']):>5} {show(r['G']['T']):>5} {show(r['I']['T']):>5} "
          f"{show(r['IG']['T']):>5} | {show(div[('IG', 'N')]):>4} {show(div[('I', 'N')]):>4} "
          f"{show(div[('G', 'N')]):>4} | {show(r['I']['first_change']):>6} "
          f"{show(r['IG']['first_change']):>4} | {frac(mi.get('lph')):>7} "
          f"{frac(mi.get('other')):>7} | {frac(mig.get('lph')):>7} {frac(mig.get('other')):>7}")

    p("")
    p("Newborns with p other than 0 (count and mean s, p > 0 | p < 0; D = the difference):")

    for s, r, _ in rows:
        cells = []

        for arm in ("I", "IG"):
            x = r[arm]
            d, _ = difference(x["plus"], x["minus"])
            (np_, sp), (nm, sm) = x["plus"], x["minus"]
            cells.append(f"{arm} {np_} {frac(F(sp, np_) if np_ else None)} | {nm} "
                         f"{frac(F(sm, nm) if nm else None)}; D {frac(d)}")

        p(f"{s:>5}  " + ";   ".join(cells))

    pooled_effects = {}

    for arm in ("I", "IG"):
        n1 = sum(runs[(arm, s)]["plus"][0] for s in seeds)
        s1 = sum(runs[(arm, s)]["plus"][1] for s in seeds)
        n2 = sum(runs[(arm, s)]["minus"][0] for s in seeds)
        s2 = sum(runs[(arm, s)]["minus"][1] for s in seeds)
        d, _ = difference((n1, s1), (n2, s2))
        pooled_effects[arm] = {"plus": n1, "mean_s_plus": None if not n1 else str(F(s1, n1)),
                               "minus": n2, "mean_s_minus": None if not n2 else str(F(s2, n2)),
                               "D": None if d is None else str(d)}
        p(f"Pooled {arm}: p > 0 in {n1} newborns, mean s {frac(F(s1, n1) if n1 else None)}; "
          f"p < 0 in {n2}, mean s {frac(F(s2, n2) if n2 else None)}; D {frac(d)}")

    p("")
    p("Imagination choices and changed picks over the Run (I, IG):")
    p("  " + "; ".join(f"{s}: I {r['I']['imagination_choices']}/{r['I']['imagination_changed']}"
                       f", IG {r['IG']['imagination_choices']}/{r['IG']['imagination_changed']}"
                       for s, r, _ in rows))

    p("")
    p("Goals of G and IG (blocks of generations from generation 1 on; the first 15):")

    for s, r, _ in rows:
        bg, big = r["G"]["goal_blocks"], r["IG"]["goal_blocks"]
        same = "same in G and IG" if bg == big else "DIFFERENT IN G AND IG"
        names_g = [b[0] for b in (bg or [])[:15]]
        p(f"{s:>5}  {len(bg or [])} blocks, {same}: {' '.join(str(n) for n in names_g)}")

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
        f"{arm} {d['draws']} draws, {d['changed']} change a promising Worker, {d['lost']} of "
        f"those lose a family ({pct(share(d['lost'], d['changed']))}), {d['made']} make all "
        f"three" for arm, d in pooled.items()))

    if any(d["multiple"] for d in pooled.values()):
        p("NOTE: a draw changed more than one promising Worker (the first counts): "
          + ", ".join(f"{a} {d['multiple']}" for a, d in pooled.items()))

    p("")
    p("== criteria (fixed before the first Run)")

    early = [s for s, r, _ in rows
             if r["IG"]["first_change"] is not None and r["IG"]["first_change"] <= EARLY]
    early_i = [s for s, r, _ in rows
               if r["I"]["first_change"] is not None and r["I"]["first_change"] <= EARLY]
    c1 = len(early) == len(seeds)
    p(f"1. Early: imagination changes a pick in generation {EARLY} or earlier in IG on "
      f"{len(early)} of {len(seeds)} seeds (needs all) {early}; first changed picks "
      f"{[show(r['IG']['first_change']) for _, r, _ in rows]}. {'MET' if c1 else 'FAILED'}. "
      f"Arm I (description): {len(early_i)} of {len(seeds)}, first changed picks "
      f"{[show(r['I']['first_change']) for _, r, _ in rows]}.")

    def higher(arm):
        return [s for s, r, _ in rows if r[arm]["means"] and r[arm]["means"]["lph"] is not None
                and r[arm]["means"]["other"] is not None
                and r[arm]["means"]["lph"] > r[arm]["means"]["other"]]

    high, high_i = higher("IG"), higher("I")
    c2 = len(high) >= 6
    p(f"2. Learning: the parents' mean value for Linear, Phase and Hadamard in generations 41 "
      f"to 50 is above that of the other five families in IG on {len(high)} of {len(seeds)} "
      f"seeds (needs 6) {high}. {'MET' if c2 else 'FAILED'}. Arm I (description): "
      f"{len(high_i)} of {len(seeds)} {high_i}.")

    above = [s for s, r, _ in rows if difference(r["IG"]["plus"], r["IG"]["minus"])[1]]
    above_i = [s for s, r, _ in rows if difference(r["I"]["plus"], r["I"]["minus"])[1]]
    thin = [s for s, r, _ in rows if min(r["IG"]["plus"][0], r["IG"]["minus"][0]) < 5]
    c3 = len(above) >= 7
    p(f"3. Imagination measures the effect: D > 0 in IG on {len(above)} of {len(seeds)} seeds "
      f"(needs 7) {above}; seeds with fewer than 5 newborns in a group {thin}. "
      f"{'MET' if c3 else 'FAILED'}. Pooled IG: D {frac(_d(pooled_effects['IG']))}. Arm I "
      f"(description): {len(above_i)} of {len(seeds)} {above_i}, pooled D "
      f"{frac(_d(pooled_effects['I']))}.")

    sh = {arm: share(pooled[arm]["lost"], pooled[arm]["changed"]) for arm in ARMS}
    enough = pooled["IG"]["changed"] >= 30 and pooled["N"]["changed"] >= 30
    tenth = F(1, 10)
    c4 = (enough and sh["IG"] <= F(2, 5) and sh["IG"] <= sh["N"] - tenth)
    p(f"4. Protection: of the draws that change a promising Worker, IG loses a family in "
      f"{pooled['IG']['lost']} of {pooled['IG']['changed']} ({pct(sh['IG'])}; needs at most "
      f"40 %), N in {pooled['N']['lost']} of {pooled['N']['changed']} ({pct(sh['N'])}); IG must "
      f"lie 10 points below N; IG and N need 30 such draws "
      f"({'both have' if enough else 'NOT BOTH HAVE'}). {'MET' if c4 else 'FAILED'}. "
      f"Description: I {pooled['I']['lost']} of {pooled['I']['changed']} ({pct(sh['I'])}), "
      f"G {pooled['G']['lost']} of {pooled['G']['changed']} ({pct(sh['G'])}).")

    def order(a, b):
        ab = [s for s, r, _ in rows if before(r[a]["T"], r[b]["T"])]
        ba = [s for s, r, _ in rows if before(r[b]["T"], r[a]["T"])]

        return ab, ba, len(seeds) - len(ab) - len(ba)

    p("5. Direction (reported):")
    direction = {}

    for a, b in (("IG", "N"), ("I", "N"), ("G", "N"), ("IG", "I"), ("IG", "G")):
        ab, ba, ties = order(a, b)
        direction[f"{a}-{b}"] = {"first": ab, "second": ba, "ties": ties}
        p(f"   {a} before {b} on {len(ab)} seeds {ab}; {b} before {a} on {len(ba)} {ba}; "
          f"ties {ties}.")

    ig_n = direction["IG-N"]["first"]

    if c1 and c2 and c3 and c4 and len(ig_n) >= 5:
        verdict = "GO"
    elif not (c1 and c2 and c3):
        verdict = "STOP"
    else:
        verdict = "THE OWNER DECIDES"

    p("")
    p(f"Verdict: {verdict} (criteria 1 {'met' if c1 else 'failed'}, 2 "
      f"{'met' if c2 else 'failed'}, 3 {'met' if c3 else 'failed'}, 4 "
      f"{'met' if c4 else 'failed'}; IG passes before N on {len(ig_n)} of {len(seeds)}).")

    if problems or anomalies:
        p(f"Integrity: {len(problems)} deviations and {len(anomalies)} anomalies above.")

    print("\n".join(out))

    if args.json:
        record = {
            "seeds": seeds,
            "families": names,
            "runs": [to_json(x) for x in runs.values()],
            "pooled_draws": pooled,
            "pooled_effects": pooled_effects,
            "direction": direction,
            "criteria": {"1": c1, "2": c2, "3": c3, "4": c4},
            "verdict": verdict,
            "deviations": problems,
            "anomalies": anomalies,
        }

        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=1, default=str)
            f.write("\n")


def _d(pooled):
    return None if pooled["D"] is None else F(pooled["D"])


if __name__ == "__main__":
    main(sys.argv)
