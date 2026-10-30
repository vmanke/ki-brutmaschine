#!/usr/bin/env python3
"""Reference implementation of the sequential rule of issue 09, for cross-checks.

It reads only config.json, report.json and rediscovery.json of each run. In seed order it
computes the pass time T of each run, the pair outcome, the evidence values E1 and E2, the
stop pair and the two verdicts. A claim is proven at the first pair with E >= 100 and fails
at the first pair with E <= 1/20. Both tests are exact integer comparisons.

Two modes:

  python3 -I reference_rule.py --runs <dir>
      The counted experiment. <dir> holds A-1, R-1, A-2, R-2, ... The counted seeds are a
      fixed list of 120: 1 to 100, then 111 to 130. The list skips the pilot seeds 101 to
      110. Pair i is the i-th seed of the list. The seeds must run in this order, one pair
      at a time. A run directory on any other seed is an error. A pair counts once both of
      its runs are finished. Every counted run must have the pre-registered settings and
      pass the calibration gate. Recount the counted result with this mode before it is
      reported.

  python3 -I reference_rule.py --pilots <scratchpad>
      An illustration on uncounted pilot batches. Pilot seeds never count.
"""

import json
import os
import re
import sys
from fractions import Fraction as F

# A claim is proven at the first pair with E >= PROOF. It fails at the first pair with
# E <= 1 / LOWER.
PROOF = 100
LOWER = 20
CAP = 120
CEIL = 1000
ALL_GENERATIONS = "all generations done"

# The counted seeds in their fixed order. Pair i is the i-th seed of the list. The list
# skips the pilot seeds 101 to 110. The other pilot seeds (201-210, 301-310, 401 and 402)
# lie outside it.
SEEDS = list(range(1, 101)) + list(range(111, 131))

assert len(SEEDS) == CAP

# The pre-registered settings every counted run must have.
PLANNED = {
    "generations": CEIL,
    "max_calls": 13_000_000,
    "children_per_generation": 8,
    "search_puzzles": 33,
    "select_puzzles": 22,
}

RUN_DIR = re.compile(r"^([AR])-([0-9]+)$")


def fail(message):
    sys.exit(f"reference_rule: {message}")


def load(path):
    with open(path) as f:
        return json.load(f)


def pass_time(red, label):
    # T: the k-th smallest first generation among the targets only bred members reach.
    open_targets = red["library"] - red["reached_by_seeds"]
    firsts = sorted(
        t["first_generation"]
        for t in red["targets"]
        if t["reached_by_bred"] and not t["reached_by_seed"]
    )
    k = open_targets // 2 + 1

    if open_targets > 0 and len(firsts) >= k:
        t = firsts[k - 1]
    else:
        t = None

    if red["passes"] != (red["calibrated"] and t is not None):
        fail(f"{label}: rediscovery.json says passes = {red['passes']}, but T = {t}")

    return t


def proven(w, n):
    # E = 3^w / 2^n >= PROOF, in integers.
    return 3 ** w >= PROOF * 2 ** n


def failed(w, n):
    # E = 3^w / 2^n <= 1 / LOWER, in integers.
    return LOWER * 3 ** w <= 2 ** n


def state(w, n):
    if proven(w, n):
        return "proven"

    if failed(w, n):
        return "failed"

    return "open"


def within_ceiling(t):
    if t is not None and t <= CEIL:
        return t

    return None


def first_to_pass(ta, tr):
    if ta is not None and (tr is None or ta < tr):
        return "A"

    if tr is not None and (ta is None or tr < ta):
        return "R"

    return "tie"


def run_rule(pairs):
    """pairs: list of (seed, T_A, T_R) in seed order; T None = never (within CEIL)."""
    k = 0
    n = 0
    w = 0
    l = 0
    st1 = "open"
    st2 = "open"
    lines = []
    stop = None

    for i, (seed, ta, tr) in enumerate(pairs, 1):
        if stop is not None:
            lines.append(f"pair {i} seed {seed}: after the stop, not counted")
            continue

        ta = within_ceiling(ta)
        tr = within_ceiling(tr)
        x = ta is not None
        y = first_to_pass(ta, tr)

        n += 1
        k += x

        if st1 == "open":
            st1 = state(k, n)

        if y != "tie":
            if y == "A":
                w += 1
            else:
                l += 1

            if st2 == "open":
                st2 = state(w, w + l)

        E1 = F(3 ** k, 2 ** n)
        E2 = F(3 ** w, 2 ** (w + l))
        lines.append(f"pair {i} seed {seed}: T_A={ta} T_R={tr} X={int(x)} first={y} "
                     f"E1={E1} ({float(E1):.3f}) E2={E2} ({float(E2):.3f}) [{st1}/{st2}]")

        if st1 == "failed" or (st1 == "proven" and st2 != "open") or i == CAP:
            stop = i

    if stop is None:
        verdict = f"open after {len(pairs)} pairs: run the next seed"
    else:
        primary = "undecided" if st1 == "open" else st1

        if st1 != "proven":
            secondary = f"not claimed (primary not proven; E2 itself: {st2})"
        elif st2 == "open":
            secondary = "undecided"
        else:
            secondary = st2

        verdict = f"stop at pair {stop}: primary {primary}, secondary {secondary}"

    return lines, verdict, stop


def check_run(path, arm, seed):
    cfg = load(os.path.join(path, "config.json"))
    report = load(os.path.join(path, "report.json"))

    if cfg["seed"] != seed:
        fail(f"{path}: config.json has seed {cfg['seed']}, expected {seed}")

    for key, value in PLANNED.items():
        if cfg.get(key) != value:
            fail(f"{path}: {key} is {cfg.get(key)}, pre-registered {value}")

    if "circuits" not in cfg.get("domain", {}):
        fail(f"{path}: not a circuit run")

    uniform = cfg.get("selection") == "uniform"

    if uniform != (arm == "R"):
        fail(f"{path}: arm {arm} runs with selection {cfg.get('selection', 'dgm (default)')}")

    if report["stop_reason"] != ALL_GENERATIONS or report["generations_completed"] != CEIL + 1:
        fail(f"{path}: stopped with {report['stop_reason']!r} after "
             f"{report['generations_completed']} generations (generation 0 included)")


def finished(runs_dir, arm, seed):
    path = os.path.join(runs_dir, f"{arm}-{seed}")

    return all(os.path.isfile(os.path.join(path, f)) for f in ("rediscovery.json", "report.json"))


def counted_pairs(runs_dir):
    """The number n of complete pairs, which hold the first n seeds of SEEDS, and the seed
    of a pair still running, if any."""
    seeds = set()

    for name in os.listdir(runs_dir):
        m = RUN_DIR.match(name)

        if m:
            seeds.add(int(m.group(2)))

    if not seeds:
        fail(f"{runs_dir}: no run directories A-<seed> or R-<seed>")

    # This also refuses more than CAP pairs, since the next pair would need a seed off the
    # list.
    off = sorted(seeds - set(SEEDS))

    if off:
        fail(f"{runs_dir}: seeds {off} are not on the list of the {CAP} counted seeds, "
             f"1 to 100, then 111 to 130")

    n = 0

    while n < CAP and finished(runs_dir, "A", SEEDS[n]) and finished(runs_dir, "R", SEEDS[n]):
        n += 1

    # After the complete pairs there may only be the next pair of the list, still running.
    rest = sorted(seeds - set(SEEDS[:n]))

    if rest and rest != [SEEDS[n]]:
        fail(f"seeds must run in the order of the list, one pair at a time; after {n} "
             f"complete pairs found seeds {rest}, but only seed {SEEDS[n]} may be running")

    return n, (rest[0] if rest else None)


def recount(runs_dir):
    n, running = counted_pairs(runs_dir)
    pairs = []
    calibrated = []

    for s in SEEDS[:n]:
        reds = []

        for arm in ("A", "R"):
            path = os.path.join(runs_dir, f"{arm}-{s}")
            check_run(path, arm, s)
            reds.append(load(os.path.join(path, "rediscovery.json")))

        calibrated.append(reds[0]["calibrated"] and reds[1]["calibrated"])
        pairs.append((s, pass_time(reds[0], f"A-{s}"), pass_time(reds[1], f"R-{s}")))

    lines, verdict, stop = run_rule(pairs)
    print(f"== counted experiment in {runs_dir}: {n} complete pairs")
    print("\n".join(lines))

    # A counted pair up to the stop that fails the calibration gate makes the experiment void.
    counted = stop if stop is not None else n

    for i in range(counted):
        if not calibrated[i]:
            print(f"VOID: seed {SEEDS[i]} fails the calibration gate |R_seed| <= |L| / 3")
            sys.exit(1)

    print(verdict)

    # Without a stop, fewer than CAP pairs are done, so the list has a next seed.
    if stop is None:
        print(f"next: run seed {SEEDS[n]}")

    if running is not None:
        print(f"seed {running}: not finished, not counted yet")


def pilot_batch(sp, name, seeds, ceil_note):
    pairs = []

    for s in seeds:
        red_a = load(os.path.join(sp, name, f"A-{s}", "rediscovery.json"))
        red_r = load(os.path.join(sp, name, f"R-{s}", "rediscovery.json"))
        pairs.append((s, pass_time(red_a, f"{name}/A-{s}"), pass_time(red_r, f"{name}/R-{s}")))

    lines, verdict, _ = run_rule(pairs)
    print(f"== {name} ({ceil_note}); illustration only, pilot seeds never count")
    print("\n".join(lines))
    print(verdict)


def pilots(sp):
    pilot_batch(sp, "pilot-1000", range(301, 310), "1000 generations")
    pilot_batch(sp, "pilot10", range(101, 111), "200 generations, so T > 200 is unknown")

    # Synthetic: X = 1 exactly when E1 < 1, so E1 stays in [1/2, 3/2) and never crosses.
    # It runs CAP pairs, and pair i carries the i-th seed of the list, as in a counted run.
    # Its pass times are synthetic.
    k = 0
    n = 0
    pairs = []

    for i in range(1, CAP + 1):
        x = F(3 ** k, 2 ** n) < 1
        k += x
        n += 1

        if x:
            tr = 4 if i % 2 == 0 else None
        else:
            tr = 6 if i % 2 == 1 else None

        pairs.append((SEEDS[i - 1], 5 if x else None, tr))

    lines, verdict, _ = run_rule(pairs)
    print("== synthetic path that keeps E1 between 1/2 and 3/2:", verdict)
    print(lines[-1])


def main(argv):
    if len(argv) == 3 and argv[1] == "--runs":
        recount(argv[2])
    elif len(argv) == 3 and argv[1] == "--pilots":
        pilots(argv[2])
    else:
        fail("usage: reference_rule.py --runs <dir> | --pilots <scratchpad>")


if __name__ == "__main__":
    main(sys.argv)
