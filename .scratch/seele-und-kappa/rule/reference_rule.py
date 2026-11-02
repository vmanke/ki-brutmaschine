#!/usr/bin/env python3
"""Independent recount of the step 2 comparison, for cross-checks.

It recounts what examples/soul_comparison.rs computes, written apart from that code: the
pass times, the outcomes Y_V and Y_K, the evidence values E_V and E_K, the stop triple and
the verdicts (docs/PREREG_STEP2_SOUL.md, sections 5 to 8), and the divergence generation of
section 12. Evidence values are exact fractions; every decision compares fractions exactly.

It reads config.json, report.json and rediscovery.json of each run, and its events.jsonl for
the divergence generation and the vote tallies. When only events.jsonl.xz is there, it reads
that file instead.

Definitions, as in section 5 of the draft:
- T, the pass time: the k-th smallest first_generation among the targets that only bred
  members solve, k = floor(|L \\ R_seed| / 2) + 1; never when fewer exist, or when T > G.
- Y: "S first" when T_S < T_control, "<control> first" when T_control < T_S, else a tie.
- E starts at 1 and is multiplied by 2 p1 on "S first" and by 2 (1 - p1) on "<control>
  first"; a tie leaves it. A claim is proven at the first triple with E >= 200 and fails at
  the first with E <= 1/20; its first crossing is final. The rule stops after the first
  triple at which both claims are decided, at the latest after N_MAX triples.
- The divergence generation of two runs: the first generation whose newborns differ, member
  by member in birth order, in gene id, operator, parents (by gene id), answers, solved count
  or graded value. The gene id is the member id of the genome with every Worker's
  phase_state and coupling removed: the first 16 hex digits of the SHA-256 of its compact
  JSON.

Two modes:

  python3 -I reference_rule.py --runs <dir>
      The counted experiment. <dir> holds S-1001, V-1001, K-1001, S-1002, ... The counted
      seeds are 1001, 1002, ... in this order, at most N_MAX. A triple counts once its three
      runs are finished. A seed off the list or a gap is an error.

  python3 -I reference_rule.py --pilot <first>-<last> <dir>...
      Pilot triples on the seeds first to last (pilot seeds 901 to 923), read from S-<seed>,
      V-<seed> and K-<seed>, and E-<seed> and N-<seed> when present, in the given
      directories. Pilot seeds never count: no claim is made.
"""

import hashlib
import json
import lzma
import os
import sys
from fractions import Fraction as F

# The pre-registered numbers of the draft. G, M, N_MAX and P1 are proposals of the pilot.
G = 1000
M = 13_000_000
N_MAX = 120
P1 = F(3, 4)
PROOF = F(200)
LOWER = F(1, 20)
FIRST_SEED = 1001
PILOT = range(901, 924)
SEED_TARGETS = {"swap", "cs"}
ALL_GENERATIONS = "all generations done"

# The settings of every run, written out from section 6 of docs/PREREG_STEP1_CIRCUITS.md
# (arm A of step 1) and the table of spec 12.2; the seed, the generations and the call cap
# vary.
COMMON = {
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
    "llm_children": 2,
    "pause_for_llm": True,
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
    "domain": {"circuits": {"targets": [], "dress": True, "max_gates": 64}},
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

SOULS = {
    "S": SOUL_S,
    "V": {**SOUL_S, "readout": "real"},
    "K": {**SOUL_S, "kappa_seed": 0, "kappa_fixed": True},
    "E": {**SOUL_S, "strengths": "equal"},
    "N": None,
}

TRIPLE = ("S", "V", "K")
PAIRS = [("S", "V"), ("S", "K"), ("V", "K"), ("S", "E"), ("S", "N")]


def fail(message):
    sys.exit(f"reference_rule: {message}")


def load(path):
    with open(path) as f:
        return json.load(f)


def check_config(path, cfg, arm, seed):
    expected = dict(COMMON)
    expected["seed"] = seed
    expected["generations"] = cfg.get("generations")
    expected["max_calls"] = cfg.get("max_calls")

    if SOULS[arm] is not None:
        expected["soul"] = SOULS[arm]

    if cfg != expected:
        keys = sorted(k for k in set(cfg) | set(expected) if cfg.get(k) != expected.get(k))
        fail(f"{path}: its settings are not those of arm {arm}: {keys}")


def pass_time(red, label):
    """T without the ceiling; None when the run never passes."""
    open_targets = sum(1 for t in red["targets"] if not t["reached_by_seed"])
    firsts = sorted(t["first_generation"] for t in red["targets"]
                    if t["reached_by_bred"] and not t["reached_by_seed"])
    k = open_targets // 2 + 1
    t = firsts[k - 1] if open_targets > 0 and len(firsts) >= k else None

    if red["passes"] != (red["calibrated"] and t is not None):
        fail(f"{label}: rediscovery.json says passes = {red['passes']}, but T = {t}")

    return t


def within(t):
    return t if t is not None and t <= G else None


def first(ts, tc, control):
    if ts is not None and (tc is None or ts < tc):
        return "S first"

    if tc is not None and (ts is None or tc < ts):
        return f"{control} first"

    return "tie"


def gene_id(cascade):
    genes = json.loads(json.dumps(cascade))

    for stage in genes["stages"]:
        for worker in stage["workers"]:
            worker["traits"].pop("phase_state", None)
            worker["traits"].pop("coupling", None)

    text = json.dumps(genes, separators=(",", ":"), ensure_ascii=False)

    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def open_events(path):
    plain = os.path.join(path, "events.jsonl")

    if os.path.exists(plain):
        return open(plain, encoding="utf-8")

    return lzma.open(plain + ".xz", "rt", encoding="utf-8")


def read_events(path):
    """Per generation the newborns' signatures, and the summed vote tallies."""
    genes = {}
    born = {}
    votes = {"complex": 0, "flipped": 0, "tied": 0, "sim_decided": 0}

    with open_events(path) as f:
        for line in f:
            event = json.loads(line)["event"]

            if event["kind"] != "born":
                continue

            gid = gene_id(event["cascade"])
            parents = tuple(genes[p] for p in event["parents"])
            signature = (gid, event["op"], parents, json.dumps(event.get("answers")),
                         event["solved"], event.get("quality"))
            born.setdefault(event["generation"], []).append(signature)
            genes[event["id"]] = gid

            for key in votes:
                votes[key] += (event.get("votes") or {}).get(key, 0)

    return born, votes


def divergence(a, b):
    for g in range(max(len(a), len(b))):
        if a.get(g) != b.get(g):
            return g

    return None


def read_run(path, arm, seed):
    cfg = load(os.path.join(path, "config.json"))
    report = load(os.path.join(path, "report.json"))
    red = load(os.path.join(path, "rediscovery.json"))

    check_config(path, cfg, arm, seed)

    planned = cfg["generations"] + 1

    if report["stop_reason"] != ALL_GENERATIONS or report["generations_completed"] != planned:
        fail(f"{path}: stopped with {report['stop_reason']!r} after "
             f"{report['generations_completed']} generations (generation 0 included)")

    born, votes = read_events(path)

    if len(born) != report["generations_completed"]:
        fail(f"{path}: newborns in {len(born)} generations, report {planned}")

    r_seed = {t["target"] for t in red["targets"] if t["reached_by_seed"]}

    return {
        "cfg": cfg,
        "report": report,
        "T": within(pass_time(red, path)),
        "r_seed": r_seed,
        "born": born,
        "votes": votes,
    }


def show_e(e):
    return f"{e.numerator}/{e.denominator} = {float(e):.4f}"


def show_t(t):
    return "never" if t is None else str(t)


def rule(triples, claim_allowed):
    """triples: list of (seed, {arm: run}) in seed order."""
    a, b = 2 * P1, 2 * (1 - P1)
    e = {"V": F(1), "K": F(1)}
    verdict = {"V": None, "K": None}
    lines = []
    stop = None

    for i, (seed, runs) in enumerate(triples, 1):
        if stop is not None:
            lines.append(f"{seed:>5}  after the stop, not counted")
            continue

        ts = runs["S"]["T"]
        ys = {}

        for control in ("V", "K"):
            y = first(ts, runs[control]["T"], control)
            ys[control] = y

            if y == "S first":
                e[control] *= a
            elif y != "tie":
                e[control] *= b

            if verdict[control] is None:
                if e[control] >= PROOF:
                    verdict[control] = ("PROVEN", i)
                elif e[control] <= LOWER:
                    verdict[control] = ("FAILED", i)

        lines.append(f"{seed:>5}  {show_t(ts):>5}  {show_t(runs['V']['T']):>5}  "
                     f"{show_t(runs['K']['T']):>5}  {ys['V']:<7}  {ys['K']:<7}  "
                     f"{show_e(e['V']):<24}  {show_e(e['K'])}")

        if (verdict["V"] is not None and verdict["K"] is not None) or i == N_MAX:
            stop = i

    counted = stop if stop is not None else len(triples)

    for control in ("V", "K"):
        v = verdict[control]
        where = (f"; it reached 200 at triple {v[1]}" if v and v[0] == "PROVEN" else
                 f"; it fell to 1/20 at triple {v[1]}" if v else
                 "; it reached neither 200 nor 1/20" if stop is not None else
                 "; open, it has reached neither 200 nor 1/20")
        lines.append(f"E_{control} = {show_e(e[control])} after {counted} triple(s){where}.")

    lines.append(f"The rule stops after triple {stop}." if stop is not None
                 else f"The rule goes on after triple {counted}.")

    # Gate G1 for the counted triples up to the stop.
    void = [seed for seed, runs in triples[:counted]
            if any(runs[arm]["r_seed"] != SEED_TARGETS for arm in TRIPLE)]

    if not claim_allowed[0]:
        lines.append(f"No claim: {claim_allowed[1]}.")
    elif void:
        lines.append(f"VOID: gate G1 fails for seed(s) {void}.")
    elif stop is None:
        lines.append(f"OPEN: run seed {FIRST_SEED + counted} next.")
    else:
        for control in ("V", "K"):
            v = verdict[control]
            word = f"{v[0]} at triple {v[1]}" if v else f"UNDECIDED after {N_MAX} triples"
            lines.append(f"Claim {control}: {word}.")

    return lines


def describe(triples):
    lines = []
    arms = [arm for arm in ("S", "V", "K", "E", "N") if any(arm in r for _, r in triples)]
    rates = []

    for arm in arms:
        runs = [r[arm] for _, r in triples if arm in r]
        rates.append(f"{arm} {sum(1 for x in runs if x['T'] is not None)}/{len(runs)}")

    lines.append("Pass rates within generation %d: %s." % (G, ", ".join(rates)))
    against = []

    for arm in arms[1:]:
        tally = {"S first": 0, f"{arm} first": 0, "tie": 0}

        for _, r in triples:
            if arm in r:
                tally[first(r["S"]["T"], r[arm]["T"], arm)] += 1

        against.append(f"{arm} {tally['S first']}/{tally[arm + ' first']}/{tally['tie']}")

    lines.append("Against arm S by the pass time (S first / the other arm first / tie): "
                 + ", ".join(against) + ".")
    pairs = [(x, y) for x, y in PAIRS if x in arms and y in arms]
    lines.append(f"{'seed':>5}" + "".join(f"  {x + '-' + y:>5}" for x, y in pairs))

    for seed, r in triples:
        cells = []

        for x, y in pairs:
            if x in r and y in r:
                g = divergence(r[x]["born"], r[y]["born"])
                cells.append("none" if g is None else str(g))
            else:
                cells.append("-")

        lines.append(f"{seed:>5}" + "".join(f"  {c:>5}" for c in cells))

    lines.append("Votes per run: complex, flipped, tied, sim_decided.")

    for arm in arms:
        for seed, r in triples:
            if arm in r:
                v = r[arm]["votes"]
                lines.append(f"{arm}-{seed}: {v['complex']} {v['flipped']} {v['tied']} "
                             f"{v['sim_decided']}")

    return lines


def find(dirs, name):
    for d in dirs:
        path = os.path.join(d, name)

        if os.path.isdir(path):
            return path

    return None


def finished(path):
    return path is not None and all(
        os.path.isfile(os.path.join(path, f)) for f in ("report.json", "rediscovery.json"))


def shared_length(triples):
    shared = {(r["cfg"]["generations"], r["cfg"]["max_calls"])
              for _, runs in triples for r in runs.values()}

    if len(shared) != 1:
        fail(f"the runs differ in --generations or --max-calls: {sorted(shared)}")

    return shared.pop()


def pilot(seeds, dirs):
    if not all(s in PILOT for s in seeds):
        fail("--pilot takes pilot seeds 901 to 923 only")

    triples = []

    for s in seeds:
        runs = {}

        for arm in ("S", "V", "K", "E", "N"):
            path = find(dirs, f"{arm}-{s}")

            if path is None:
                if arm in TRIPLE:
                    fail(f"seed {s} has no run in arm {arm}")
                continue

            runs[arm] = read_run(path, arm, s)

        triples.append((s, runs))

    generations, max_calls = shared_length(triples)
    print(f"== pilot triples {seeds[0]} to {seeds[-1]}; every run: --generations {generations}, "
          f"--max-calls {max_calls}; pilot seeds never count")
    print("\n".join(rule(triples, (False, "the seeds are pilot seeds of step 2 (901 to 923), "
                                          "which never count"))))
    print("\n".join(describe(triples)))


def counted(runs_dir):
    # Run directories are S-<seed>, V-<seed> and K-<seed>; other entries are not read.
    seeds = set()

    for n in os.listdir(runs_dir):
        arm, _, seed = n.partition("-")

        if arm in TRIPLE and seed.isdigit() and os.path.isdir(os.path.join(runs_dir, n)):
            seeds.add(int(seed))

    if not seeds:
        fail(f"{runs_dir}: no run directories S-<seed>, V-<seed> or K-<seed>")

    listed = list(range(FIRST_SEED, FIRST_SEED + N_MAX))
    off = sorted(seeds - set(listed))

    if off:
        fail(f"seeds {off} are not on the list of the counted seeds, {listed[0]} to {listed[-1]}")

    n = 0

    while n < N_MAX and all(finished(find([runs_dir], f"{arm}-{listed[n]}")) for arm in TRIPLE):
        n += 1

    rest = sorted(seeds - set(listed[:n]))

    if rest and rest != [listed[n]]:
        fail(f"seeds must run in the order of the list; after {n} complete triples found "
             f"seeds {rest}, but only seed {listed[n]} may be running")

    if n == 0:
        fail("no complete triple")

    triples = [(s, {arm: read_run(os.path.join(runs_dir, f"{arm}-{s}"), arm, s) for arm in TRIPLE})
               for s in listed[:n]]
    generations, max_calls = shared_length(triples)

    # read_run refuses a run that ended at the call cap, so only the length and the cap can
    # still withhold the claim here.
    if (generations, max_calls) != (G, M):
        allowed = (False, f"the runs have --generations {generations} and --max-calls "
                          f"{max_calls}, not the pre-registered {G} and {M}")
    else:
        allowed = (True, "")

    print(f"== counted experiment in {runs_dir}: {n} complete triples")
    print("\n".join(rule(triples, allowed)))
    print("\n".join(describe(triples)))

    if rest:
        print(f"seed {rest[0]}: not finished, not counted yet")


def main(argv):
    if len(argv) == 3 and argv[1] == "--runs":
        counted(argv[2])
    elif len(argv) >= 4 and argv[1] == "--pilot":
        lo, _, hi = argv[2].partition("-")
        pilot(list(range(int(lo), int(hi or lo) + 1)), argv[3:])
    else:
        fail("usage: reference_rule.py --runs <dir> | --pilot <first>-<last> <dir>...")


if __name__ == "__main__":
    main(sys.argv)
