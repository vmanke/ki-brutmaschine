#!/usr/bin/env python3
"""Build counted.json, the table of the counted Runs of step 1.

The script reads only the files of the Runs and the logs of the driver:

- per Run (<runs>/A-<seed> and <runs>/R-<seed>): config.json, events.jsonl, report.json
  and rediscovery.json, and the SHA-256 of these four and of snapshot.json;
- the logs (<logs>): driver.log (wall time, the 'verify ok' lines, the thread check, the
  restart), hashes.txt (the hashes the driver recorded after each pair), analysis.txt (the
  frozen analysis), recount.txt (reference_rule.py) and step1-build-test.txt (toolchain and
  tests).

It recomputes R_seed, D, the pass time T, X, Y, E1 and E2 from rediscovery.json alone, in
exact integers, as docs/PREREG_STEP1_CIRCUITS.md (sections 7 and 8) defines them. It stops
with an error when a number differs from analysis.txt or recount.txt, when a file's hash
differs from hashes.txt, or when a setting differs from section 6. Its output depends only
on these files, so Runs repeated at the frozen commit with the frozen toolchain, whose
files match hashes.txt, give the same counted.json.

Usage, from the root of the repository:

    python3 -I -B docs/step1-counted/make_counted.py --runs <dir> \
        --logs docs/step1-counted --out docs/step1-counted/counted.json

<dir> holds the Run directories A-1, R-1, ..., A-27, R-27.
"""

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import datetime
from fractions import Fraction

# The fixed list of counted seeds (PREREG section 2).
SEEDS = list(range(1, 101)) + list(range(111, 131))
GENERATIONS = 1000
MAX_CALLS = 13_000_000
PROOF = 100
FAILURE = 20
LIBRARY = ["cz", "swap", "cs", "ch", "qft2", "ccz", "toffoli", "peres", "fredkin", "rtof",
           "ccix"]
HASHED = ["config.json", "events.jsonl", "snapshot.json", "report.json", "rediscovery.json"]
KINDS = {"run_start", "born", "compared", "generation", "state", "finalist"}

# Every setting of PREREG section 6 except the seed; arm R adds "selection": "uniform".
SETTINGS = {
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
    "llm_children": 2,
    "pause_for_llm": True,
    "max_calls": MAX_CALLS,
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

COLUMNS = [
    "generation",
    "performance.solved_best",
    "performance.solved_median",
    "performance.solved_best_born",
    "performance.fitness_best",
    "performance.fitness_median",
    "performance.quality_best",
    "performance.quality_median",
    "diversity.shapes",
    "diversity.shape_entropy_bits",
    "diversity.blocks",
    "diversity.block_entropy_bits",
    "diversity.model_entropy_bits",
    "diversity.behaviours",
    "diversity.parent_entropy_bits",
]

COMMAND_A = ("brut-breed --out runs/A-S --domain circuits --search 33 --select 22 "
             "--children 8 --generations 1000 --max-calls 13000000 --seed S")

NOTES = [
    "Every number below is recomputed by make_counted.py from the files of the Runs and the "
    "logs of the driver; the script stops with an error if one differs from analysis.txt, "
    "recount.txt or hashes.txt. Floats are rounded to 4 decimal places.",
    "frozen_commit: the commit at which the driver ran, from its 'driver started at commit' "
    "lines; it is the squash merge of PR #19, at which docs/PREREG_STEP1_CIRCUITS.md is frozen.",
    "toolchain and tests: from step1-build-test.txt, the build and the three test commands of "
    "PREREG section 6 step 1 at the frozen commit.",
    "threads: the driver set no RAYON_NUM_THREADS for the counted Runs, so rayon used one "
    "thread per CPU; the thread check found the files of seed 1 byte-equal at the default, "
    "1 and 4 threads (thread_check); the wall time depends on the thread count.",
    "stop: the pair after which the rule of PREREG section 8 stopped, and which of its three "
    "stop rules applied.",
    "verdicts: each claim with the pair at which its evidence value first crossed a threshold, "
    "and its evidence value at the stop, as a fraction and as a decimal.",
    "error_bound: PREREG section 8; it holds however the experiment stops, under the conduct "
    "and the assumptions stated there.",
    "descriptive: results with no decision attached (PREREG section 8): the median of D_A, the "
    "T-only variant, the Runs that ended at the call cap and the invention candidates.",
    "checks: what the script checked; hash_lines counts the lines of hashes.txt, "
    "distinct_hash_lines its distinct lines. Pair 1's ten lines appear twice, because the "
    "driver, started again after the container restart, verified pair 1 again and appended "
    "its hashes again; both copies are equal.",
    "pairs: one entry per counted pair, in the order of the seed list (PREREG section 8, "
    "'Reported per pair').",
    "pairs.r_seed: the targets the seed Cascades solve; it is the same in both arms.",
    "pairs.d_a, pairs.d_r: D of the arm A and the arm R Run at generation 1000, as hits/open "
    "where open = |L \\ R_seed| (not reduced, as the analysis prints it).",
    "pairs.t_a, pairs.t_r: the pass times T (PREREG section 7), the first generation with "
    "2 * |R_bred(<= g) \\ R_seed| > |L \\ R_seed|; null means never by generation 1000.",
    "pairs.x: 1 when the arm A Run passes (T_A <= 1000), else 0.",
    "pairs.y: 'A first' when T_A < T_R, 'R first' when T_R < T_A, 'tie' when they are equal, "
    "including when neither Run passes; never counts as later than every generation.",
    "pairs.e1, pairs.e2: the evidence values after the pair, E1 = 3^passes / 2^pairs and "
    "E2 = 3^(A first) / 2^(untied pairs), as fractions; e1_value and e2_value are decimals.",
    "pairs.primary, pairs.secondary: 'open', 'proven' or 'failed' after the pair, by the "
    "integer tests 3^k >= 100 * 2^n (proven) and 20 * 3^k <= 2^n (failed); a verdict is final "
    "at its first crossing.",
    "pairs.d_a_t_only, pairs.d_r_t_only: D under the T-only variant (descriptive): a target "
    "counts when the cheapest exact answer has T <= T_ref, and also CNOT <= CNOT_ref where "
    "T_ref = 0.",
    "runs: one entry per counted Run, in the order A-1, R-1, A-2, R-2, ...",
    "runs.generations, runs.max_calls: --generations and --max-calls from config.json.",
    "runs.generations_completed counts generation 0 (the seed Cascades) and the bred "
    "generations 1 to 1000.",
    "runs.stop_reason, runs.calls_used, runs.archive_size, runs.comparisons: from report.json; "
    "calls_used counts every Worker call of the Run, the select phase included.",
    "runs.calls_by_generation_1000: calls_used of the generation event of generation 1000 in "
    "events.jsonl, before the select phase; PREREG section 6 bounds it by 12 723 412.",
    "runs.wall_seconds: wall time of the Run's process, the seconds between its last 'start' "
    "line and its 'done' line in driver.log (1 s resolution, one process at a time); minutes "
    "is the same in minutes.",
    "runs.attempts: how many times the driver started the Run; interrupted_attempt gives the "
    "start of an attempt that never finished and the time the driver removed it (PREREG "
    "section 6 step 3; deviation 1).",
    "runs.library: |L|; r_seed, r_bred: the targets the seed and the bred Cascades solve "
    "(PREREG section 7); r_bred_only: R_bred \\ R_seed; d: D as hits/open; d_value: D as a "
    "decimal; calibrated: the gate 3 * |R_seed| <= |L|; d_above_half: D > 1/2.",
    "runs.pass_time: T of the Run, null for never.",
    "runs.r_seed_t_only, runs.d_t_only: R_seed and D under the T-only variant.",
    "runs.targets: per target of L, in library order: reference, best_seed and best_bred are "
    "[T, CNOT] of the library reference and of the cheapest exact answer of the seed and of "
    "the bred Cascades (null when none was exact); seed_hit: the seed Cascades solve it; "
    "first_bred_generation: the earliest generation of a bred member that solved one of the "
    "target's dressed search tasks (null when none did); calls_to_first_hit: calls_used of "
    "the generation event of that generation, the calls through it (descriptive).",
    "runs.invention_candidates: targets with an exact answer below a BestKnown count "
    "(below_best_known or candidate in rediscovery.json).",
    "runs.integrity_alarm: false when the Run finished with exit code 0 and wrote "
    "rediscovery.json; a Run with an alarm exits with code 1 and has no rediscovery.json, and "
    "the driver stops at any other exit code (step1-driver.sh).",
    "runs.verify: events is the number of events in events.jsonl and head the hash of its "
    "last event, the head that brut-breed --verify reports; chain_linked: every event's prev "
    "equals the hash of the event before it (the script does not recompute the hashes; "
    "--verify does). logged: the times of the driver's 'verify ok' lines; the driver writes "
    "one only when the last line of --verify contains 'report recomputed and equal; "
    "rediscovery recomputed and equal', which brut-breed prints at the end of that line "
    "(PREREG section 6 step 4; step1-driver.sh).",
    "runs.sha256: the SHA-256 of the five files that hashes.txt records; each equals its line "
    "there.",
    "runs.best_single_cascade_solved: the most search tasks one member of the archive solves, "
    "the largest 'solved' of the born events; best_by_fitness_solved: 'solved' of the best "
    "member by fitness (report.json, top[0]); they need not be the same Cascade "
    "(descriptive).",
    "runs.select: select-panel hits of the finalists and of the seed Cascades, "
    "select_solved of the 22 select tasks (report.json, finalists), and the best rates "
    "(descriptive).",
    "runs.trajectory: the state line of every 10th generation, one row per line in the order "
    "of trajectory_columns (descriptive).",
    "thread_check: PREREG section 6 step 5, from driver.log: seed 1 of each arm again at "
    "RAYON_NUM_THREADS 1 and 4, with the wall time and the time of the driver's 'verify ok' "
    "line of each Run.",
    "deviations: the deviations from the pre-registration, with deviations.txt as the source.",
]


def fail(msg):
    sys.exit(f"make_counted: {msg}")


def r4(x):
    return round(x, 4) if isinstance(x, float) else x


def frac(f):
    return f"{f.numerator}/{f.denominator}"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def stamp(s):
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ")


def state(e):
    """The state of an evidence value: proven, failed or open, in integers."""
    if e.numerator >= PROOF * e.denominator:
        return "proven"
    if FAILURE * e.numerator <= e.denominator:
        return "failed"
    return "open"


def cost(c):
    return None if c is None else [c["t"], c["cnot"]]


def t_only(target, best):
    ref = target["reference"]
    return best is not None and best["t"] <= ref["t"] and (ref["t"] > 0 or best["cnot"] <= ref["cnot"])


def count(found, rule):
    """R_seed, hits and open of one Run under one rule."""
    seeds, bred = [], []
    for t in found["targets"]:
        if rule == "full":
            by_seed, by_bred = t["reached_by_seed"], t["reached_by_bred"]
        else:
            by_seed, by_bred = t_only(t, t["best_seed"]), t_only(t, t["best_bred"])
        if by_seed:
            seeds.append(t["target"])
        if by_bred:
            bred.append(t["target"])
    open_ = len(found["targets"]) - len(seeds)
    hits = len([t for t in bred if t not in seeds])
    return seeds, bred, hits, open_


def pass_time(found):
    open_ = len([t for t in found["targets"] if not t["reached_by_seed"]])
    firsts = sorted(t["first_generation"] for t in found["targets"]
                    if t["reached_by_bred"] and not t["reached_by_seed"])
    k = open_ // 2
    return firsts[k] if k < len(firsts) else None


def first(t_a, t_r):
    a = float("inf") if t_a is None else t_a
    r = float("inf") if t_r is None else t_r
    return "A first" if a < r else "R first" if r < a else "tie"


def read_driver(path):
    """The timestamped lines of driver.log, and the per-run start, done and verify times."""
    lines, commits, runs, thread_ok, restarts = [], set(), {}, None, 0
    for raw in open(path, encoding="utf-8"):
        m = re.match(r"^(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ) (.*)$", raw.rstrip("\n"))
        if not m:
            continue
        t, msg = m.groups()
        lines.append((t, msg))
        if msg.startswith("driver started at commit "):
            commits.add(msg.split()[-1])
            restarts += 1
            continue
        m2 = re.match(r"^(start|done|verify ok|removing the unfinished run|keeping the finished run) "
                      r"(runs(?:-threads)?/[AR]-\d+(?:-t\d)?)(?: with RAYON_NUM_THREADS=(\d))?$", msg)
        if m2:
            what, run, threads = m2.groups()
            r = runs.setdefault(run, {"starts": [], "done": [], "verify": [], "removed": [],
                                      "kept": [], "threads": threads})
            {"start": r["starts"], "done": r["done"], "verify ok": r["verify"],
             "removing the unfinished run": r["removed"],
             "keeping the finished run": r["kept"]}[what].append(t)
        elif msg.startswith("thread check ok"):
            thread_ok = f"{t} {msg}"
    return lines, commits, runs, thread_ok, restarts - 1


def wall(r, name):
    """Wall seconds of the last attempt, and the attempts that never finished."""
    if len(r["done"]) != 1:
        fail(f"{name}: {len(r['done'])} 'done' lines in driver.log")
    done = r["done"][0]
    starts = [s for s in r["starts"] if s <= done]
    last = starts[-1]
    lost = []
    for s in starts[:-1]:
        removed = [x for x in r["removed"] if x > s]
        lost.append({"start": s, "removed": removed[0] if removed else None})
    return int((stamp(done) - stamp(last)).total_seconds()), last, done, lost


def read_hashes(path):
    rec, n = {}, 0
    for line in open(path, encoding="utf-8"):
        n += 1
        h, p = line.rstrip("\n").split("  ", 1)
        rec.setdefault(p, set()).add(h)
    return rec, n


def read_analysis(path):
    """Rows of the pre-registered table, the T-only table, the sequential table, the first
    generations and the verdict lines of the frozen analysis."""
    text = open(path, encoding="utf-8").read()
    part = None
    full, tonly, seq, firsts, verdicts = {}, {}, {}, {}, []
    median = above = cap_line = inv_line = None
    row_re = re.compile(r"^\s*(\d+)\s+(\d+)\s+(\d+): ([a-z, ]+?)\s+(\d+/\d+) = [0-9.]+\s+"
                        r"(\d+/\d+) = [0-9.]+\s+(yes|no)$")
    seq_re = re.compile(r"^\s*(\d+)\s+(never|\d+)\s+(never|\d+)\s+([01])\s+(A|R|tie)\s+"
                        r"(\d+/\d+) = [0-9.]+\s+(\d+/\d+) = [0-9.]+$")
    for line in text.splitlines():
        if line.startswith("Pre-registered rule:"):
            part = "full"
        elif line.startswith("Sequential rule"):
            part = "seq"
        elif line.startswith("T-only variant, descriptive"):
            part = "tonly"
        elif line.startswith("Descriptive: the first generation"):
            part = "firsts"
        m = row_re.match(line)
        if m and part in ("full", "tonly"):
            seed, lib, _, names, d_a, d_r, cal = m.groups()
            (full if part == "full" else tonly)[int(seed)] = (
                int(lib), sorted(names.split(", ")), d_a, d_r, cal == "yes")
            continue
        m = seq_re.match(line)
        if m and part == "seq":
            seed, ta, tr, x, y, e1, e2 = m.groups()
            seq[int(seed)] = (None if ta == "never" else int(ta), None if tr == "never" else int(tr),
                              int(x), {"A": "A first", "R": "R first", "tie": "tie"}[y], e1, e2)
            continue
        if part == "firsts" and line and not line.startswith(("Descriptive", "target", "Runs",
                                                             "Invention")):
            cells = line.split()
            firsts[cells[0]] = cells[1:]
        if line.startswith("Median D_A"):
            median = re.search(r"descriptive: (\d+/\d+)", line).group(1)
        if line.startswith("T-only variant: D_A > 1/2 in"):
            above = re.search(r"in (\d+) of (\d+)", line).groups()
        if re.match(r"^(Primary|Secondary) \(.*\): (PROVEN|FAILED|UNDECIDED|NOT)", line) or \
                line.startswith("The rule stops"):
            verdicts.append(line)
        if line.startswith("Runs that ended at the call cap:"):
            cap_line = line
        if line.startswith("Invention candidates"):
            inv_line = line
    return full, tonly, seq, firsts, verdicts, median, above, cap_line, inv_line


def read_recount(path):
    rows, stop = {}, None
    r = re.compile(r"^pair (\d+) seed (\d+): T_A=(None|\d+) T_R=(None|\d+) X=([01]) "
                   r"first=(A|R|tie) E1=(\d+(?:/\d+)?) \([0-9.]+\) E2=(\d+(?:/\d+)?) \([0-9.]+\) "
                   r"\[(\w+)/(\w+)\]$")
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        m = r.match(line)
        if m:
            i, seed, ta, tr, x, y, e1, e2, s1, s2 = m.groups()
            rows[int(i)] = (int(seed), None if ta == "None" else int(ta),
                            None if tr == "None" else int(tr), int(x),
                            {"A": "A first", "R": "R first", "tie": "tie"}[y],
                            Fraction(e1), Fraction(e2), s1, s2)
        elif line.startswith("stop at pair"):
            stop = line
    return rows, stop


def read_events(path):
    """Events count, head, linkage, calls by generation, best born solved, trajectory."""
    n, prev, linked, kinds = 0, "0" * 64, True, set()
    calls, best_born, born, trajectory = {}, 0, 0, []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec["seq"] != n or rec["prev"] != prev:
                linked = False
            prev = rec["hash"]
            n += 1
            e = rec["event"]
            kinds.add(e["kind"])
            if e["kind"] == "generation":
                calls[e["generation"]] = e["calls_used"]
            elif e["kind"] == "born":
                born += 1
                best_born = max(best_born, e["solved"])
            elif e["kind"] == "state" and e["generation"] % 10 == 0:
                row = []
                for c in COLUMNS:
                    v = e
                    for k in c.split("."):
                        v = v.get(k) if isinstance(v, dict) else None
                    row.append(r4(v))
                trajectory.append(row)
    return {"events": n, "head": prev, "linked": linked, "kinds": kinds, "calls": calls,
            "best_born": best_born, "born": born, "trajectory": trajectory}


def read_build(path):
    text = open(path, encoding="utf-8").read()
    toolchain = re.search(r"^(rustc .*)$", text, re.M).group(1)
    commit = re.search(r"^commit ([0-9a-f]{40})$", text, re.M).group(1)
    results = re.findall(r"^(cargo test [^:]+):\ntest result: (.*?); finished", text, re.M)
    if "build exit 0" not in text or len(results) != 3 or \
            not all(res.startswith("ok. ") and " 0 failed;" in res for _, res in results):
        fail("step1-build-test.txt: the build or a test command did not pass")
    return toolchain, commit, {cmd: res for cmd, res in results}


def run_entry(runs_dir, arm, seed, driver_runs, hashes):
    name = f"{arm}-{seed}"
    d = os.path.join(runs_dir, name)
    config = json.load(open(os.path.join(d, "config.json"), encoding="utf-8"))
    report = json.load(open(os.path.join(d, "report.json"), encoding="utf-8"))
    found = json.load(open(os.path.join(d, "rediscovery.json"), encoding="utf-8"))

    want = dict(SETTINGS, seed=seed)
    if arm == "R":
        want["selection"] = "uniform"
    if config != want:
        fail(f"{name}: config.json differs from the settings of PREREG section 6")
    if [t["target"] for t in found["targets"]] != LIBRARY:
        fail(f"{name}: the targets are not L in library order")
    if report["stop_reason"] != "all generations done" or report["generations_completed"] != GENERATIONS + 1:
        fail(f"{name}: the Run did not complete all generations")

    ev = read_events(os.path.join(d, "events.jsonl"))
    if not ev["linked"] or not ev["kinds"] <= KINDS:
        fail(f"{name}: events.jsonl is not linked or has an unknown event kind")
    if ev["born"] != report["archive_size"]:
        fail(f"{name}: born events and archive size differ")

    digests = {}
    for f in HASHED:
        h = sha256(os.path.join(d, f))
        if hashes.get(f"runs/{name}/{f}") != {h}:
            fail(f"{name}/{f}: the hash differs from hashes.txt")
        digests[f] = h

    r_seed, r_bred, hits, open_ = count(found, "full")
    ts, _, th, to = count(found, "tonly")
    if (found["library"], found["reached_by_seeds"], found["reached_by_bred_only"]) != (len(LIBRARY), len(r_seed), hits):
        fail(f"{name}: rediscovery.json disagrees with its own targets")
    cal = 3 * len(r_seed) <= len(LIBRARY)
    if found["calibrated"] != cal or found["passes"] != (cal and 2 * hits > open_):
        fail(f"{name}: calibrated or passes disagrees with the targets")

    targets, candidates = [], []
    for t in found["targets"]:
        g = t["first_generation"]
        targets.append({
            "target": t["target"],
            "reference": [t["reference"]["t"], t["reference"]["cnot"]],
            "best_seed": cost(t["best_seed"]),
            "best_bred": cost(t["best_bred"]),
            "seed_hit": t["reached_by_seed"],
            "first_bred_generation": g,
            "calls_to_first_hit": None if g is None else ev["calls"][g],
        })
        if t["below_best_known"] or t["candidate"] is not None:
            candidates.append({"target": t["target"], "candidate": t["candidate"]})

    r = driver_runs.get(f"runs/{name}")
    if r is None or not r["verify"]:
        fail(f"{name}: driver.log has no 'verify ok' line")
    secs, start, done, lost = wall(r, name)
    t = pass_time(found)
    finalists = report["finalists"]

    entry = {
        "arm": arm,
        "seed": seed,
        "dir": name,
        "generations": config["generations"],
        "max_calls": config["max_calls"],
        "generations_completed": report["generations_completed"],
        "stop_reason": report["stop_reason"],
        "calls_used": report["calls_used"],
        "calls_by_generation_1000": ev["calls"][GENERATIONS],
        "archive_size": report["archive_size"],
        "comparisons": report["comparisons"],
        "start": start,
        "done": done,
        "wall_seconds": secs,
        "minutes": round(secs / 60, 4),
        "attempts": len(lost) + 1,
    }
    if lost:
        entry["interrupted_attempt"] = lost
    entry.update({
        "library": found["library"],
        "r_seed": r_seed,
        "r_bred": r_bred,
        "r_bred_only": [x for x in r_bred if x not in r_seed],
        "d": f"{hits}/{open_}",
        "d_value": round(hits / open_, 4),
        "calibrated": cal,
        "d_above_half": 2 * hits > open_,
        "pass_time": t,
        "r_seed_t_only": ts,
        "d_t_only": f"{th}/{to}",
        "targets": targets,
        "invention_candidates": candidates,
        "integrity_alarm": False,
        "verify": {
            "events": ev["events"],
            "head": ev["head"],
            "chain_linked": ev["linked"],
            "logged": r["verify"],
        },
        "sha256": digests,
        "best_single_cascade_solved": ev["best_born"],
        "best_by_fitness_solved": report["top"][0]["solved"],
        "select": {
            "finalists_select_solved": [x["select_solved"] for x in finalists if not x["is_seed"]],
            "seeds_select_solved": [x["select_solved"] for x in finalists if x["is_seed"]],
            "select_tasks": finalists[0]["select_puzzles"],
            "best_finalist_select_rate": r4(report["best_finalist_select_rate"]),
            "best_seed_select_rate": r4(report["best_seed_select_rate"]),
        },
        "trajectory": ev["trajectory"],
    })
    return entry, found, (ts, th, to)


def dumps(value, indent=0, in_list=False):
    """JSON with two-space indent; scalar lists, and flat dicts inside lists, on one line."""
    pad, inner = "  " * indent, "  " * (indent + 1)

    def scalar(v):
        return not isinstance(v, (dict, list))

    def flat(v):
        return scalar(v) or (isinstance(v, list) and all(scalar(x) for x in v))

    if scalar(value):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, list):
        line = json.dumps(value, ensure_ascii=False, separators=(", ", ": "))
        if all(scalar(x) for x in value) and \
                (len(line) <= 100 or not any(isinstance(x, str) for x in value)):
            return line
        if all(scalar(x) for x in value):
            body = [inner + json.dumps(x, ensure_ascii=False) for x in value]
            return "[\n" + ",\n".join(body) + "\n" + pad + "]"
        body = [inner + dumps(x, indent + 1, True) for x in value]
        return "[\n" + ",\n".join(body) + "\n" + pad + "]"
    if in_list and all(flat(v) for v in value.values()):
        return json.dumps(value, ensure_ascii=False, separators=(", ", ": "))
    body = [inner + json.dumps(k) + ": " + dumps(v, indent + 1) for k, v in value.items()]
    return "{\n" + ",\n".join(body) + "\n" + pad + "}"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--runs", required=True, help="the folder with A-1, R-1, ...")
    ap.add_argument("--logs", required=True, help="the folder with driver.log and the other logs")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    log = lambda f: os.path.join(args.logs, f)
    toolchain, build_commit, tests = read_build(log("step1-build-test.txt"))
    lines, commits, druns, thread_ok, restarts = read_driver(log("driver.log"))
    hashes, hash_lines = read_hashes(log("hashes.txt"))
    a_full, a_tonly, a_seq, a_firsts, a_verdicts, a_median, a_above, cap_line, inv_line = \
        read_analysis(log("analysis.txt"))
    rc_rows, rc_stop = read_recount(log("recount.txt"))

    if len(commits) != 1 or commits != {build_commit}:
        fail("the driver and the build ran at different commits")
    commit = build_commit

    pairs, runs = [], []
    e1, e2 = Fraction(1), Fraction(1)
    primary = secondary = None
    stop = None
    passes = a_first = r_first = ties = 0
    for i, seed in enumerate(SEEDS, start=1):
        if not all(os.path.isfile(os.path.join(args.runs, f"{arm}-{seed}", "report.json"))
                   for arm in "AR"):
            break
        if stop is not None:
            fail(f"pair {i} (seed {seed}) was run after the stop")
        ea, fa, ta_only = run_entry(args.runs, "A", seed, druns, hashes)
        er, fr, tr_only = run_entry(args.runs, "R", seed, druns, hashes)
        if ea["r_seed"] != er["r_seed"]:
            fail(f"seed {seed}: R_seed differs between the arms")
        runs += [ea, er]
        t_a, t_r = ea["pass_time"], er["pass_time"]
        x = 1 if t_a is not None else 0
        y = first(t_a, t_r)
        passes += x
        e1 *= Fraction(3, 2) if x else Fraction(1, 2)
        if y == "A first":
            e2 *= Fraction(3, 2)
            a_first += 1
        elif y == "R first":
            e2 *= Fraction(1, 2)
            r_first += 1
        else:
            ties += 1
        if primary is None and state(e1) != "open":
            primary = (state(e1), i)
        if secondary is None and state(e2) != "open":
            secondary = (state(e2), i)
        pairs.append({
            "pair": i,
            "seed": seed,
            "r_seed": ea["r_seed"],
            "d_a": ea["d"],
            "d_r": er["d"],
            "t_a": t_a,
            "t_r": t_r,
            "x": x,
            "y": y,
            "e1": frac(e1),
            "e1_value": r4(float(e1)),
            "e2": frac(e2),
            "e2_value": r4(float(e2)),
            "primary": primary[0] if primary else "open",
            "secondary": secondary[0] if secondary else "open",
            "d_a_t_only": ea["d_t_only"],
            "d_r_t_only": er["d_t_only"],
        })
        if primary and primary[0] == "failed":
            stop = (i, 1)
        elif primary and primary[0] == "proven" and secondary:
            stop = (i, 2)
        elif i == len(SEEDS):
            stop = (i, 3)

        # The frozen analysis and the recount must give the same numbers.
        row = a_full.get(seed)
        if row != (len(LIBRARY), sorted(ea["r_seed"]), ea["d"], er["d"], ea["calibrated"]):
            fail(f"seed {seed}: the pre-registered row differs from analysis.txt")
        if a_tonly.get(seed) != (len(LIBRARY), sorted(ea["r_seed_t_only"]), ea["d_t_only"],
                                 er["d_t_only"], 3 * len(ea["r_seed_t_only"]) <= len(LIBRARY)):
            fail(f"seed {seed}: the T-only row differs from analysis.txt")
        if a_seq.get(seed) != (t_a, t_r, x, y, frac(e1), frac(e2)):
            fail(f"seed {seed}: the sequential row differs from analysis.txt")
        rc = rc_rows.get(i)
        if rc is None or rc[:7] != (seed, t_a, t_r, x, y, e1, e2) or \
                rc[7:] != (pairs[-1]["primary"], pairs[-1]["secondary"]):
            fail(f"pair {i}: the row differs from recount.txt")
        for k, t in enumerate(LIBRARY):
            cells = []
            for entry in (ea, er):
                g = entry["targets"][k]["first_bred_generation"]
                star = "*" if entry["targets"][k]["seed_hit"] else ""
                cells.append("-" if g is None else f"{g}{star}")
            if a_firsts[t][i - 1] != "|".join(cells):
                fail(f"seed {seed}, {t}: the first generations differ from analysis.txt")

    if stop is None:
        fail("the rule has not stopped on these pairs")
    n = len(pairs)
    if len(rc_rows) != n or len(a_seq) != n or len(a_full) != n:
        fail("analysis.txt or recount.txt holds another number of pairs")

    d_a = sorted((Fraction(*map(int, p["d_a"].split("/"))) for p in pairs))
    median = d_a[n // 2] if n % 2 else (d_a[n // 2 - 1] + d_a[n // 2]) / 2
    if a_median is None or Fraction(a_median) != median:
        fail("the median of D_A differs from analysis.txt")
    above = sum(1 for e in runs if e["arm"] == "A"
                and 2 * int(e["d_t_only"].split("/")[0]) > int(e["d_t_only"].split("/")[1]))
    if a_above != (str(above), str(n)):
        fail("the T-only count differs from analysis.txt")
    at_cap = [e["dir"] for e in runs if e["stop_reason"] != "all generations done"
              or e["calls_used"] >= e["max_calls"]]
    candidates = [{"dir": e["dir"], **c} for e in runs for c in e["invention_candidates"]]
    if (cap_line != "Runs that ended at the call cap: none.") != bool(at_cap) or \
            (inv_line != "Invention candidates (below a BestKnown count): none.") != bool(candidates):
        fail("the call cap or invention lines differ from analysis.txt")

    verdict_text = {
        "proven": "PROVEN at pair {}",
        "failed": "FAILED at pair {}",
    }
    want = [f"Primary (ticket 03): {verdict_text[primary[0]].format(primary[1])}.",
            f"Secondary (selection): {verdict_text[secondary[0]].format(secondary[1])}."]
    if [v.split(". ")[0] + "." for v in a_verdicts if not v.startswith("The rule")] != want:
        fail("the verdict lines differ from analysis.txt")
    if rc_stop != f"stop at pair {stop[0]}: primary {primary[0]}, secondary {secondary[0]}":
        fail("the stop line differs from recount.txt")

    # The thread check of PREREG section 6 step 5, from driver.log.
    threads = []
    for name in ["A-1-t1", "A-1-t4", "R-1-t1", "R-1-t4"]:
        r = druns.get(f"runs-threads/{name}")
        if r is None or not r["verify"]:
            fail(f"{name}: the thread check Run is missing from driver.log")
        secs, start, done, _ = wall(r, name)
        threads.append({"dir": f"runs-threads/{name}", "rayon_num_threads": int(r["threads"]),
                        "start": start, "done": done, "wall_seconds": secs,
                        "minutes": round(secs / 60, 4), "verify_logged": r["verify"]})
    if thread_ok is None:
        fail("driver.log has no 'thread check ok' line")

    a_runs = [e for e in runs if e["arm"] == "A"]
    out = {
        "what": "The counted Runs of the step 1 circuit experiment, the positive control of the "
                "breeder with the simulated designer (docs/PREREG_STEP1_CIRCUITS.md). It says "
                "nothing about real language models.",
        "counted": True,
        "frozen_commit": commit,
        "toolchain": toolchain,
        "tests": tests,
        "binary": "cargo build --release --bin brut-breed at the frozen commit",
        "threads": "RAYON_NUM_THREADS unset for the counted Runs (rayon's default, one thread "
                   "per CPU); 1 and 4 for the thread check",
        "started": lines[0][0],
        "stopped": [t for t, m in lines if m.startswith("the rule has stopped")][0],
        "driver_restarts": restarts,
        "arms": {"A": COMMAND_A, "R": "the same with --out runs/R-S --selection uniform"},
        "stop": {
            "pair": stop[0],
            "seed": pairs[stop[0] - 1]["seed"],
            "rule": stop[1],
            "why": {1: "the primary claim failed",
                    2: "the primary claim is proven and E2 has crossed a threshold",
                    3: "120 pairs are done"}[stop[1]],
        },
        "verdicts": {
            "primary": {"claim": "ticket 03: more than half of all seeds pass within 1000 "
                                 "generations (rejects the pass rate <= 1/2)",
                        "verdict": primary[0], "at_pair": primary[1],
                        "e1_at_crossing": pairs[primary[1] - 1]["e1"],
                        "e1_at_stop": frac(e1), "e1_at_stop_value": r4(float(e1)),
                        "passes": passes, "pairs": n},
            "secondary": {"claim": "selection: among untied pairs, arm A passes first more "
                                   "often than arm R (rejects P(A first | not tied) <= 1/2)",
                          "verdict": secondary[0], "at_pair": secondary[1],
                          "e2_at_crossing": pairs[secondary[1] - 1]["e2"],
                          "e2_at_stop": frac(e2), "e2_at_stop_value": r4(float(e2)),
                          "a_first": a_first, "r_first": r_first, "ties": ties,
                          "tied_pairs": [p["pair"] for p in pairs if p["y"] == "tie"]},
        },
        "error_bound": "each claim is made falsely with probability at most 1/100, and some "
                       "claim is made falsely with probability at most 1/100 (Ville's "
                       "inequality and the fixed sequence of the two claims)",
        "descriptive": {
            "median_d_a": frac(median),
            "t_only_d_a_above_half": above,
            "runs_at_call_cap": at_cap,
            "invention_candidates": candidates,
            "max_calls_by_generation_1000": max(e["calls_by_generation_1000"] for e in runs),
            "arm_a_passes": sum(1 for e in a_runs if e["pass_time"] is not None),
        },
        "checks": {
            "hash_lines": hash_lines,
            "distinct_hash_lines": sum(len(v) for v in hashes.values()),
            "files_hashed_and_equal": len(runs) * len(HASHED),
            "settings_equal_to_section_6": len(runs),
            "verify_ok": len(runs),
            "analysis_agrees": True,
            "recount_agrees": True,
            "thread_check": thread_ok,
        },
        "notes": NOTES,
        "trajectory_columns": COLUMNS,
        "pairs": pairs,
        "runs": runs,
        "thread_check": threads,
        "deviations": [
            "Deviation 1 (deviations.txt): a container restart interrupted Run A-12, which had "
            "started at 2026-10-31T01:18:49Z. Following PREREG section 6 step 3, the driver, "
            "started again, deleted the unfinished Run and ran it again from scratch on the "
            "same seed into a fresh directory; no finished pair changed or ran again.",
        ],
    }
    if any(e.get("interrupted_attempt") for e in runs) != bool(out["deviations"]):
        fail("the interrupted attempts and the deviations disagree")
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(dumps(out) + "\n")
    print(f"make_counted: {n} pairs, stop at pair {stop[0]} by rule {stop[1]}; primary "
          f"{primary[0]} at pair {primary[1]}, secondary {secondary[0]} at pair {secondary[1]}; "
          f"wrote {args.out}")


if __name__ == "__main__":
    main()
