#!/usr/bin/env python3
"""Copy the live results of step 1 into the Brutmaschine Atlas.

Usage: python3 -I -B sync_step1.py <atlas.html> [<analysis.txt>]

It reads the analysis that the step 1 driver rewrites after every pair (by default
docs/step1-counted/analysis.txt next to this folder), parses the sequential table and the verdict lines,
recomputes E1 and E2 with exact fractions by section 8 of docs/PREREG_STEP1_CIRCUITS.md,
and replaces only the text between <!-- step1-data:start --> and <!-- step1-data:end -->
in <atlas.html> with a JSON block. It writes no other file.

It exits 1 with a message when the analysis has an unexpected format, when a recomputed
value, pass, first arm, crossing or stop differs from the printed one, when the file
changes while it is read, or when the analysis makes no claim (void or withheld), a state
the page does not show. Run on the same analysis twice, it changes nothing the second time.
"""

import json
import os
import re
import sys
from datetime import datetime, timezone
from fractions import Fraction

DEFAULT_ANALYSIS = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "step1-counted", "analysis.txt"
)
START = "<!-- step1-data:start -->"
END = "<!-- step1-data:end -->"

# Section 8 of the pre-registration: proven at E >= 100, failed at E <= 1/20, at most
# 120 pairs, on the fixed seed list 1 to 100, then 111 to 130.
PROOF = Fraction(100)
FAILURE = Fraction(1, 20)
MAX_PAIRS = 120
SEEDS = list(range(1, 101)) + list(range(111, 131))
UP, DOWN = Fraction(3, 2), Fraction(1, 2)

TITLE = "Step 1 circuit positive control: the frozen analysis of docs/PREREG_STEP1_CIRCUITS.md"
COUNT = re.compile(r"^(\d+) seed pair\(s\); every run: --generations (\d+), --max-calls (\d+)\.$")
HEADER = re.compile(r"^\s*seed\s+T_A\s+T_R\s+X\s+first\s+E1\s+E2\s*$")
FRACTION = r"(\d+)/(\d+) = (\d+\.\d{4})"
ROW = re.compile(
    r"^\s*(\d+)\s+(never|\d+)\s+(never|\d+)\s+([01])\s+(A|R|tie)\s+"
    + FRACTION + r"\s+" + FRACTION + r"$"
)
AFTER = re.compile(r"^\s*(\d+)\s+after the stop, not counted$")
EVIDENCE = {
    "primary": re.compile(r"^Primary \(ticket 03\): E1 = " + FRACTION + r" after (\d+) pair\(s\)(.*)\.$"),
    "secondary": re.compile(r"^Secondary \(selection\): E2 = " + FRACTION + r" after (\d+) pair\(s\)(.*)\.$"),
}
RULE = re.compile(r"^The rule (goes on|stops) after pair (\d+)\.$")
NEXT = re.compile(r"^OPEN: run seed (\d+) next\.$")
VERDICT = {
    "primary": re.compile(r"^Primary \(ticket 03\): (.+?)\.(?: .*)?$"),
    "secondary": re.compile(r"^Secondary \(selection\): (.+?)\.(?: .*)?$"),
}


class Bad(Exception):
    """The analysis or the page is not what this script expects."""


def crossing_tail(tail):
    """The state that the end of an evidence line names, as (state, pair)."""
    if tail == "":
        return ("not tested", None)
    m = re.fullmatch(r"; it reached 100 at pair (\d+)", tail)
    if m:
        return ("proven", int(m.group(1)))
    m = re.fullmatch(r"; it fell to 1/20 at pair (\d+)", tail)
    if m:
        return ("failed", int(m.group(1)))
    if tail == "; open, it has reached neither 100 nor 1/20":
        return ("open", None)
    if tail == "; it reached neither 100 nor 1/20":
        return ("undecided", None)
    raise Bad(f"unexpected end of an evidence line: {tail!r}")


def verdict_text(text):
    """The state that a verdict line at the stop names, as (state, pair)."""
    m = re.fullmatch(r"PROVEN at pair (\d+)", text)
    if m:
        return ("proven", int(m.group(1)))
    m = re.fullmatch(r"FAILED at pair (\d+)", text)
    if m:
        return ("failed", int(m.group(1)))
    if text == f"UNDECIDED after {MAX_PAIRS} pairs, no claim":
        return ("undecided", None)
    if text == "NOT TESTED, since the primary is not proven":
        return ("not tested", None)
    if text == "open":
        return ("open", None)
    raise Bad(f"unexpected verdict: {text!r}")


def check_value(name, pair, printed, exact):
    """The printed fraction must equal the exact one and its decimal must round it."""
    num, den, dec = printed
    if (num, den) != (exact.numerator, exact.denominator):
        raise Bad(f"{name} after pair {pair}: printed {num}/{den}, recomputed {exact}")
    # The analysis prints a float rounded to 4 places; allow that rounding and the
    # float's own relative error, which matters only for very large values.
    if abs(Fraction(dec) - exact) > Fraction(1, 20000) + exact * Fraction(1, 10**12):
        raise Bad(f"{name} after pair {pair}: printed {dec}, exact value {float(exact)!r}")


def recompute(rows):
    """Section 8, pair by pair in seed order: E1, E2, both verdicts and the stop pair."""
    e1 = e2 = Fraction(1)
    primary, secondary = ["open", None], ["open", None]
    evidence, stop = [], None

    for i, row in enumerate(rows, 1):
        e1 *= UP if row["x"] == 1 else DOWN
        if row["first"] == "A":
            e2 *= UP
        elif row["first"] == "R":
            e2 *= DOWN
        evidence.append((e1, e2))

        # A verdict is final at its first crossing.
        for verdict, e in ((primary, e1), (secondary, e2)):
            if verdict[0] == "open":
                if e >= PROOF:
                    verdict[:] = ["proven", i]
                elif e <= FAILURE:
                    verdict[:] = ["failed", i]

        if primary[0] == "failed" or (primary[0] == "proven" and secondary[0] != "open") or i == MAX_PAIRS:
            stop = i
            if primary[0] == "open":
                primary = ["undecided", None]
            # A fixed sequence: the secondary is tested only when the primary is proven.
            if primary[0] == "proven":
                if secondary[0] == "open":
                    secondary = ["undecided", None]
            else:
                secondary = ["not tested", None]
            break

    return evidence, tuple(primary), tuple(secondary), stop


def parse(text):
    """The counted rows, the rows after the stop and the printed verdicts of the analysis."""
    lines = text.split("\n")
    if not lines or lines[0] != TITLE:
        raise Bad("the first line is not the title of the frozen step 1 analysis")
    m = COUNT.match(lines[1]) if len(lines) > 1 else None
    if not m:
        raise Bad("the second line does not give the number of seed pairs")
    total = int(m.group(1))

    for line in lines:
        if line.startswith("No claim:") or line.startswith("VOID:"):
            raise Bad(f"the analysis makes no claim ({line.strip()}); the page has no state for this")

    heads = [i for i, line in enumerate(lines) if HEADER.match(line)]
    if len(heads) != 1:
        raise Bad(f"expected one header of the sequential table, found {len(heads)}")

    rows, after, i = [], [], heads[0] + 1
    while i < len(lines):
        m = ROW.match(lines[i])
        if m and not after:
            g = m.groups()
            rows.append({
                "seed": int(g[0]),
                "t_a": None if g[1] == "never" else int(g[1]),
                "t_r": None if g[2] == "never" else int(g[2]),
                "x": int(g[3]),
                "first": g[4],
                "e1_printed": (int(g[5]), int(g[6]), g[7]),
                "e2_printed": (int(g[8]), int(g[9]), g[10]),
            })
        elif AFTER.match(lines[i]):
            after.append(int(AFTER.match(lines[i]).group(1)))
        else:
            break
        i += 1

    if not rows:
        raise Bad("the sequential table has no rows")
    if total != len(rows) + len(after):
        raise Bad(f"the analysis names {total} seed pair(s), the table has {len(rows) + len(after)}")

    printed = {}
    for name in ("primary", "secondary"):
        m = EVIDENCE[name].match(lines[i]) if i < len(lines) else None
        if not m:
            raise Bad(f"expected the {name} evidence line, found {lines[i] if i < len(lines) else 'the end'!r}")
        g = m.groups()
        printed[name] = {
            "e": (int(g[0]), int(g[1]), g[2]),
            "pairs": int(g[3]),
            "crossing": crossing_tail(g[4]),
        }
        i += 1

    m = RULE.match(lines[i]) if i < len(lines) else None
    if not m:
        raise Bad("expected the line that says whether the rule goes on or stops")
    goes_on, rule_pair = m.group(1) == "goes on", int(m.group(2))
    i += 1

    verdicts, next_seed = None, None
    if goes_on:
        m = NEXT.match(lines[i]) if i < len(lines) else None
        if not m:
            raise Bad("the rule goes on, but the analysis names no next seed")
        next_seed = int(m.group(1))
    else:
        verdicts = {}
        for name in ("primary", "secondary"):
            m = VERDICT[name].match(lines[i]) if i < len(lines) else None
            if not m:
                raise Bad(f"the rule stops, but the {name} verdict line is missing")
            verdicts[name] = verdict_text(m.group(1))
            i += 1

    return rows, after, printed, goes_on, rule_pair, next_seed, verdicts


def build(rows, after, printed, goes_on, rule_pair, next_seed, verdicts, as_of):
    """Check the analysis against the recomputation and return the data for the page."""
    n = len(rows)
    if n > MAX_PAIRS:
        raise Bad(f"{n} counted pairs, more than {MAX_PAIRS}")

    for i, row in enumerate(rows, 1):
        if row["seed"] != SEEDS[i - 1]:
            raise Bad(f"pair {i} has seed {row['seed']}, the list has {SEEDS[i - 1]}")
        if row["x"] != int(row["t_a"] is not None):
            raise Bad(f"pair {i}: X = {row['x']} does not match T_A")
        for t in (row["t_a"], row["t_r"]):
            if t is not None and not 1 <= t <= 1000:
                raise Bad(f"pair {i}: pass time {t} outside generations 1 to 1000")
        never = 10**9  # never counts as later than every generation
        a = never if row["t_a"] is None else row["t_a"]
        r = never if row["t_r"] is None else row["t_r"]
        first = "A" if a < r else ("R" if r < a else "tie")
        if row["first"] != first:
            raise Bad(f"pair {i}: first is {row['first']}, T_A and T_R give {first}")

    evidence, primary, secondary, stop = recompute(rows)
    if stop is not None and stop != n:
        raise Bad(f"the rule stops after pair {stop}, but {n} pairs are counted")
    for i, (row, (e1, e2)) in enumerate(zip(rows, evidence), 1):
        check_value("E1", i, row["e1_printed"], e1)
        check_value("E2", i, row["e2_printed"], e2)

    e1, e2 = evidence[-1]
    for name, e, state in (("primary", e1, primary), ("secondary", e2, secondary)):
        p = printed[name]
        if p["pairs"] != n:
            raise Bad(f"the {name} evidence line counts {p['pairs']} pairs, the table {n}")
        check_value(name, n, p["e"], e)
        if p["crossing"] != state:
            raise Bad(f"{name}: the analysis says {p['crossing']}, the recount gives {state}")

    if rule_pair != n or goes_on != (stop is None):
        raise Bad("the line on whether the rule goes on does not match the recount")
    if goes_on:
        if after:
            raise Bad("pairs are listed after the stop, but the rule goes on")
        if next_seed != SEEDS[n]:
            raise Bad(f"the analysis names seed {next_seed} next, the list has {SEEDS[n]}")
    elif verdicts != {"primary": primary, "secondary": secondary}:
        raise Bad(f"verdicts {verdicts} differ from the recount {primary}, {secondary}")

    return {
        "as_of": as_of,
        "stopped": stop is not None,
        "pairs_max": MAX_PAIRS,
        "pairs": [
            {
                "pair": i,
                "seed": row["seed"],
                "t_a": row["t_a"],
                "t_r": row["t_r"],
                "x": row["x"],
                "first": row["first"],
                "e1": [e1.numerator, e1.denominator],
                "e2": [e2.numerator, e2.denominator],
            }
            for i, (row, (e1, e2)) in enumerate(zip(rows, evidence), 1)
        ],
        "primary": {"state": primary[0], "at_pair": primary[1]},
        "secondary": {"state": secondary[0], "at_pair": secondary[1]},
    }


def block(data):
    """The JSON block between the markers: one pair per line, exact integers."""
    head = ", ".join(json.dumps(k) + ": " + json.dumps(data[k]) for k in ("as_of", "stopped", "pairs_max"))
    pairs = ",\n".join(json.dumps(p) for p in data["pairs"])
    tail = ", ".join(json.dumps(k) + ": " + json.dumps(data[k]) for k in ("primary", "secondary"))
    return (
        '\n<script type="application/json" id="step1-data">\n'
        + "{" + head + ', "pairs": [\n' + pairs + "\n], " + tail + "}\n"
        + "</script>\n"
    )



def log_time(analysis_path):
    """The time of the last timestamped line of driver.log next to the analysis.

    In a git checkout the file times say when the files were checked out, not when the
    experiment wrote them. The driver log records the real times, so the page shows the
    time of its last line. Returns None when there is no such log.
    """
    log = os.path.join(os.path.dirname(analysis_path), "driver.log")
    try:
        with open(log, encoding="utf-8") as f:
            stamps = [m.group(0) for m in (re.match(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", line) for line in f) if m]
    except OSError:
        return None
    return stamps[-1] if stamps else None

def main(argv):
    if len(argv) not in (2, 3):
        raise Bad("usage: python3 -I -B sync_step1.py <atlas.html> [<analysis.txt>]")
    page_path = argv[1]
    analysis_path = argv[2] if len(argv) == 3 else DEFAULT_ANALYSIS

    before = os.stat(analysis_path)
    with open(analysis_path, encoding="utf-8") as f:
        text = f.read()
    now = os.stat(analysis_path)
    if (before.st_mtime_ns, before.st_size) != (now.st_mtime_ns, now.st_size):
        raise Bad("analysis.txt changed while it was read; run again")
    as_of = log_time(analysis_path) or datetime.fromtimestamp(
        now.st_mtime, tz=timezone.utc
    ).strftime("%Y-%m-%dT%H:%M:%SZ")

    parsed = parse(text)
    data = build(*parsed, as_of)
    after = parsed[1]

    with open(page_path, encoding="utf-8", newline="") as f:
        page = f.read()
    if page.count(START) != 1 or page.count(END) != 1:
        raise Bad(f"{page_path} must hold each marker exactly once")
    a, b = page.index(START) + len(START), page.index(END)
    if a > b:
        raise Bad(f"in {page_path} the end marker comes before the start marker")
    new = page[:a] + block(data) + page[b:]

    changed = new != page
    if changed:
        with open(page_path, "w", encoding="utf-8", newline="") as f:
            f.write(new)

    last = data["pairs"][-1]
    show = lambda e: f"{e[0] / e[1]:.4f}"
    verdict = lambda v: v["state"] + (f" at pair {v['at_pair']}" if v["at_pair"] else "")
    print(
        f"sync_step1: {len(data['pairs'])} pair(s) as of {as_of}; "
        f"E1 = {show(last['e1'])} ({verdict(data['primary'])}), "
        f"E2 = {show(last['e2'])} ({verdict(data['secondary'])}); "
        f"{'stopped' if data['stopped'] else 'running'}"
        f"{f', {len(after)} pair(s) after the stop not counted' if after else ''}; "
        f"{os.path.basename(page_path)} {'updated' if changed else 'unchanged'}"
    )


if __name__ == "__main__":
    try:
        main(sys.argv)
    except (Bad, OSError, ValueError) as e:
        print(f"sync_step1: {e}", file=sys.stderr)
        sys.exit(1)
