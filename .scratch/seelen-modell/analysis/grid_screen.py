#!/usr/bin/env python3
"""The pre-specified grid screen of ticket 03 (`../issues/03-grid-lever-endpoint.md`).

Written together with the pre-specification, before any Run of the screen, and tested only on
the synthetic Runs of `--self-test`. It reads the Runs of arms S, V, K and N on seeds 3001 to
3040, each in a directory `<arm>-<seed>` under the given root, with `config.json`,
`report.json`, `events.jsonl` or `events.jsonl.xz`, and the driver's file `<arm>-<seed>.verify`
beside the directory.

Per Run it computes the endpoint A = B(1) + ... + B(300), where B(g) is the largest `solved`
count among the Cascades born in generations 1 to g, and the first generation with a complex
vote and with a flipped vote. Per seed it compares arm S with each of N, V and K: "S higher",
"X higher" or a tie, and the divergence generation by the rule of `diagnose.py` (genes without
Souls, operator, parents by genes, answers, solved count, graded value; grid `born` events carry
no answers and no graded value). Then it applies the gates and the go/stop rule of the ticket.

Python standard library only. Run it with `python3 -I -B`.

Usage:
  python3 -I -B grid_screen.py <run root>
  python3 -I -B grid_screen.py --self-test
"""

import hashlib
import json
import lzma
import os
import statistics
import sys
from math import comb

SEEDS = list(range(3001, 3041))
ARMS = ("S", "V", "K", "N")
CONTROLS = ("N", "V", "K")
GENERATIONS = 300
ALL_GENERATIONS = "all generations done"

# Go for a comparison when the one-sided exact binomial tail P(W >= w | u, 1/2) is at most
# 1 / ALPHA_INVERSE.
ALPHA_INVERSE = 20

# More ties than this in a comparison make the endpoint too coarse there: no go.
MAX_TIES = 10

# More void seeds than this void the screen.
MAX_VOID = 2

# Gates G2 and G3: a flipped vote in arm S by generation GATE_FLIP, and S parting from each
# control before generation GATE_PART, each on at least 9 in 10 of the valid seeds.
GATE_FLIP = 50
GATE_PART = 100
GATE_SHARE = (9, 10)

# The settings every Run must have (ticket 03); the domain (grid) and the selection (Darwin
# Gödel) are the defaults, so their keys are absent.
PLANNED = {
    "generations": GENERATIONS,
    "children_per_generation": 8,
    "search_puzzles": 33,
    "select_puzzles": 22,
    "crossover_rate": 0.5,
    "peer_opponents": 3,
    "peer_puzzles_per_pair": 2,
    "judge_weight": 0.2,
    "token_cost_per_1k": 0.01,
    "dgm_lambda": 10.0,
    "dgm_alpha0": 0.5,
    "llm_every": 0,
    "max_calls": 13_000_000,
    "finalists": 3,
    "limits": {
        "models": [
            {"tag": "sim:small", "cost_per_call": 0.0005},
            {"tag": "sim:large", "cost_per_call": 0.002},
        ],
        "max_stages": 4,
        "max_width": 3,
        "think_ceiling_per_stage": [],
    },
}

SOUL_S = {
    "readout": "complex",
    "strengths": "domain",
    "vote": True,
    "simulator": True,
    "observe_vote": True,
    "observe_peer": True,
    "kappa_seed": 200,
    "kappa_fixed": False,
    "kappa_drift": 20,
    "copy": True,
    "mix": True,
    "phase_drift": 512,
    "new_soul_theta": 1.0,
    "imprint": True,
    "teach": False,
}

PRESETS = {
    "S": SOUL_S,
    "V": {**SOUL_S, "readout": "real"},
    "K": {**SOUL_S, "kappa_seed": 0, "kappa_fixed": True},
    "N": None,
}


def genes(cascade):
    """A hash of the Cascade without the Souls and Coupling strengths of its Workers."""
    stages = []

    for stage in cascade["stages"]:
        workers = []

        for worker in stage["workers"]:
            worker = json.loads(json.dumps(worker))
            worker["traits"].pop("phase_state", None)
            worker["traits"].pop("coupling", None)
            workers.append(worker)

        stages.append({"workers": workers})

    text = json.dumps({"stages": stages}, sort_keys = True)

    return hashlib.sha256(text.encode()).hexdigest()


def scan(events):
    """Per Run from its events: the endpoint, the running best per generation, the first
    complex and flipped vote, the generations with a flip, and the newborns' signatures."""
    gene_of = {}
    born = {}
    best = {}
    first = {"vote": None, "flip": None}
    flip_generations = set()

    for event in events:
        if event["kind"] != "born":
            continue

        g = event["generation"]
        gid = genes(event["cascade"])
        signature = (gid, event["op"], tuple(gene_of[p] for p in event["parents"]),
                     json.dumps(event.get("answers")), event["solved"], event.get("quality"))
        gene_of[event["id"]] = gid
        born.setdefault(g, []).append(signature)
        tally = event.get("votes") or {}

        if g >= 1:
            best[g] = max(best.get(g, 0), event["solved"])

            if first["vote"] is None and tally.get("complex", 0) > 0:
                first["vote"] = g

            if tally.get("flipped", 0) > 0:
                flip_generations.add(g)

                if first["flip"] is None:
                    first["flip"] = g

    running = []
    b = 0

    for g in range(1, GENERATIONS + 1):
        b = max(b, best.get(g, 0))
        running.append(b)

    return {"A": sum(running), "B": running, "first_vote": first["vote"],
            "first_flip": first["flip"], "flip_generations": flip_generations, "born": born}


def divergence(a, b):
    """The first generation whose newborns differ, member by member in birth order, or None."""
    for g in sorted(set(a) | set(b)):
        if a.get(g) != b.get(g):
            return g

    return None


def events_of(path):
    for name, opener in (("events.jsonl.xz", lzma.open), ("events.jsonl", open)):
        full = os.path.join(path, name)

        if os.path.exists(full):
            with opener(full, "rt") as f:
                for line in f:
                    yield json.loads(line)["event"]

            return

    raise FileNotFoundError(f"{path}: no events")


def read_run(root, arm, seed):
    """One Run: its problems (empty when it is sound) and, when it has events, its scan."""
    path = os.path.join(root, f"{arm}-{seed}")
    problems = []

    try:
        with open(os.path.join(path, "config.json")) as f:
            config = json.load(f)
    except OSError as e:
        return {"problems": [f"no config.json ({e})"]}

    if config.get("seed") != seed:
        problems.append(f"seed {config.get('seed')}")

    for key, value in PLANNED.items():
        if config.get(key) != value:
            problems.append(f"{key} = {config.get(key)!r}")

    for key in ("domain", "selection", "selection_params"):
        if key in config:
            problems.append(f"unplanned {key}")

    if config.get("soul") != PRESETS[arm]:
        problems.append(f"soul {config.get('soul')!r}")

    try:
        with open(os.path.join(path, "report.json")) as f:
            report = json.load(f)

        if report.get("generations_completed") != GENERATIONS + 1:
            problems.append(f"generations_completed {report.get('generations_completed')}")

        if report.get("stop_reason") != ALL_GENERATIONS:
            problems.append(f"stop_reason {report.get('stop_reason')!r}")
    except OSError:
        problems.append("no report.json")
        report = {}

    try:
        with open(f"{path}.verify") as f:
            if "verify exit 0" not in f.read():
                problems.append("--verify failed")
    except OSError:
        problems.append("no .verify file")

    try:
        run = scan(events_of(path))
    except (OSError, KeyError, ValueError, EOFError, lzma.LZMAError) as e:
        problems.append(f"events: {e}")

        return {"problems": problems}

    run["problems"] = problems
    run["calls"] = report.get("calls_used")
    run["select"] = report.get("best_finalist_select_rate")
    run["seconds"] = None

    # The driver's wall time, "<seconds> seconds, exit <code>"; descriptive only.
    try:
        with open(f"{path}.wall") as f:
            run["seconds"] = int(f.read().split()[0])
    except (OSError, ValueError, IndexError):
        pass

    return run


def go_count(u):
    """The smallest w with P(W >= w | u, 1/2) <= 1 / ALPHA_INVERSE, in integers; None when
    no w qualifies."""
    for w in range(u + 1):
        if ALPHA_INVERSE * sum(comb(u, j) for j in range(w, u + 1)) <= 2 ** u:
            return w

    return None


def verdict(w, l, ties):
    """The go/stop rule of ticket 03 for one comparison."""
    u = w + l
    k = go_count(u)

    if ties > MAX_TIES:
        decision = "no go: too many ties"
    elif k is not None and w >= k:
        decision = "go"
    elif 2 * w <= u:
        decision = "stop"
    else:
        decision = "open: the owner decides"

    harm = k is not None and l >= k

    return {"u": u, "w": w, "l": l, "ties": ties, "k": k, "decision": decision, "harm": harm}


def at_least_share(count, n):
    num, den = GATE_SHARE

    return den * count >= num * n


def spread(values):
    values = [v for v in values if v is not None]

    if not values:
        return "none"

    return f"{min(values)} to {max(values)}, median {statistics.median(values)}"


def analyse(runs):
    """Gates and verdicts from {(arm, seed): run}; returns the lines of the report."""
    lines = []
    void = sorted({seed for (arm, seed), run in runs.items() if run["problems"]})

    for (arm, seed), run in sorted(runs.items(), key = lambda x: (x[0][1], x[0][0])):
        if run["problems"]:
            lines.append(f"problem {arm}-{seed}: {'; '.join(run['problems'])}")

    valid = [s for s in SEEDS if s not in void]
    g1 = len(void) <= MAX_VOID
    lines.append(f"G1 integrity: {len(void)} void seeds {void}: {'holds' if g1 else 'FAILS'}")

    if not g1:
        lines.append("verdict: the screen is void")

        return lines

    flips = [s for s in valid if (runs[("S", s)]["first_flip"] or GENERATIONS + 1) <= GATE_FLIP]
    g2 = at_least_share(len(flips), len(valid))
    lines.append(f"G2 a flipped vote in arm S by generation {GATE_FLIP}: {len(flips)} of "
                 f"{len(valid)} seeds: {'holds' if g2 else 'FAILS'}")
    parts = {}
    g3 = True

    for x in CONTROLS:
        parts[x] = {s: divergence(runs[("S", s)]["born"], runs[(x, s)]["born"]) for s in valid}
        early = [s for s, d in parts[x].items() if d is not None and d < GATE_PART]
        holds = at_least_share(len(early), len(valid))
        g3 = g3 and holds
        lines.append(f"G3 S parts from {x} before generation {GATE_PART}: {len(early)} of "
                     f"{len(valid)} seeds: {'holds' if holds else 'FAILS'}")

    for x in CONTROLS:
        w = l = ties = built = 0

        for s in valid:
            a_s, a_x = runs[("S", s)]["A"], runs[(x, s)]["A"]

            if a_s > a_x:
                w += 1
            elif a_x > a_s:
                l += 1
            else:
                ties += 1
                built += parts[x][s] is None

        v = verdict(w, l, ties)
        differences = [runs[("S", s)]["A"] - runs[(x, s)]["A"] for s in valid]
        lines.append(f"S against {x}: S higher {w}, {x} higher {l}, ties {ties} ({built} never "
                     f"parted); go needs {v['k']} of {v['u']}: {v['decision']}"
                     f"{'; ' + x + ' ahead of S by the same rule' if v['harm'] else ''}; "
                     f"median A_S - A_{x} = {statistics.median(differences)}")

    if not (g2 and g3):
        lines.append("verdict: stop, the Soul does not act early on these seeds (G2 or G3)")

    for arm in ARMS:
        rs = [runs[(arm, s)] for s in valid]
        early_flips = [len([g for g in r["flip_generations"] if g <= 50]) for r in rs]
        lines.append(f"arm {arm}: A {spread([r['A'] for r in rs])}; B(300) "
                     f"{spread([r['B'][-1] for r in rs])}; first complex vote "
                     f"{spread([r['first_vote'] for r in rs])}; first flipped vote "
                     f"{spread([r['first_flip'] for r in rs])}; generations with a flip in 1 to "
                     f"50 {spread(early_flips)}; calls {spread([r['calls'] for r in rs])}; "
                     f"seconds {spread([r.get('seconds') for r in rs])}; best finalist's "
                     f"select rate {spread([r.get('select') for r in rs])}")

    for x in CONTROLS:
        never = sum(d is None for d in parts[x].values())
        lines.append(f"divergence S-{x}: {spread(list(parts[x].values()))}; never parted {never}")

    return lines


def synthetic_born(generation, cid, parents, solved, flipped = 0, gene = "g"):
    worker = {"model": "sim:small", "sections": [{"name": "role", "text": gene}],
              "traits": {"phase_state": [1, 2], "coupling": 200}}

    return {"kind": "born", "generation": generation, "id": cid, "parents": parents,
            "op": "seed" if generation == 0 else "section_add",
            "cascade": {"stages": [{"workers": [worker]}]}, "solved": solved,
            "votes": {"complex": flipped, "flipped": flipped}}


def self_test():
    # Two Runs that agree up to generation 2 and part in generation 3.
    common = [synthetic_born(0, "s0", [], 4), synthetic_born(1, "a1", ["s0"], 6, gene = "x"),
              synthetic_born(2, "a2", ["a1"], 9, flipped = 1, gene = "y")]
    one = scan(common + [synthetic_born(3, "a3", ["a2"], 12, gene = "z")])
    two = scan(common + [synthetic_born(3, "b3", ["a2"], 7, gene = "w")])
    assert one["B"][:4] == [6, 9, 12, 12] and two["B"][:4] == [6, 9, 9, 9]
    assert one["A"] == 6 + 9 + 12 * 298 and two["A"] == 6 + 9 * 299
    assert one["first_flip"] == 2 and one["first_vote"] == 2
    assert divergence(one["born"], two["born"]) == 3
    assert divergence(one["born"], one["born"]) is None

    # Souls do not change the genes.
    with_soul = synthetic_born(1, "c1", ["s0"], 6, gene = "x")
    without = json.loads(json.dumps(with_soul))
    del without["cascade"]["stages"][0]["workers"][0]["traits"]["phase_state"]
    del without["cascade"]["stages"][0]["workers"][0]["traits"]["coupling"]
    assert genes(with_soul["cascade"]) == genes(without["cascade"])

    # The go counts of the ticket's table.
    table = {28: 19, 29: 20, 30: 20, 31: 21, 32: 22, 33: 22, 34: 23, 35: 23, 36: 24, 37: 24,
             38: 25, 39: 26, 40: 26}
    assert all(go_count(u) == k for u, k in table.items())

    # The rule.
    assert verdict(26, 14, 0)["decision"] == "go"
    assert verdict(25, 15, 0)["decision"] == "open: the owner decides"
    assert verdict(20, 20, 0)["decision"] == "stop"
    assert verdict(26, 3, 11)["decision"] == "no go: too many ties"
    assert verdict(10, 26, 4)["harm"]

    # The whole report on 40 synthetic seeds on which S is ahead of every control.
    runs = {}

    for seed in SEEDS:
        for arm in ARMS:
            run = dict(one if arm == "S" else two)
            run["problems"] = []
            run["calls"] = 100
            runs[(arm, seed)] = run

    text = "\n".join(analyse(runs))
    assert text.count(" holds") == 5
    assert all(f"S against {x}: S higher 40, {x} higher 0" in text for x in CONTROLS)
    assert "verdict" not in text
    print("self-test passed")


def main(argv):
    if argv[1:] == ["--self-test"]:
        self_test()

        return

    if len(argv) != 2:
        sys.exit(__doc__)

    runs = {(arm, seed): read_run(argv[1], arm, seed) for seed in SEEDS for arm in ARMS}
    print("\n".join(analyse(runs)))


if __name__ == "__main__":
    main(sys.argv)
