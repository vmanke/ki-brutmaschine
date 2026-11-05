#!/usr/bin/env python3
"""Independent recount of the analysis of docs/PREREG_BREEDING_VS_VOTE.md (section 8, step 7).

It shares no code with the analysis program (backend-ollama/src/breeding_vs_vote.rs, binary
breeding-vs-vote). It reads the same files and must give the same outcomes, evidence values,
stop pair and verdicts. It needs the Python standard library only.

How it differs from the analysis, on purpose:

- It does not regenerate the puzzles. It reads them from panels.json beside run B-s and
  checks each one against its content id, the first 16 hex digits of the SHA-256 of
  (train, test input, test output), and those ids against both runs' hash-chained logs.
  So the answers it grades with are the logged puzzles' own answers.
- It does not replay Cascades. It counts Worker calls by their request seeds: it derives
  every seed a run of a Cascade can carry, as the arena draws them, and counts the
  records of the call record that carry one.
- It grades every answer itself, as exact equality with the test output.

Usage:

  python3 -I recount.py --runs <dir> [--pilot] [--frozen <dir>] [--json <file>]
      The counted experiment (with --pilot: the pilot seeds 901 to 905, no claim). <dir>
      holds B-1, L-1, B-2, L-2, ... --frozen names the directory of config-B.json and
      config-L.json (default: the directory of this script).

  python3 -I recount.py --rule <file> [--json <file>]
      The sequential rule alone, for the tests, on each case of <file>:
      {"cases": [{"cap": 60, "outcomes": [["breeding first", "seed first", "tie"], ...]}]}.
"""

import hashlib
import json
import os
import re
import struct
import sys
from fractions import Fraction

PROOF = 100
FUTILITY = 20
SECONDARY = 200
CAP = 60
PILOT = [901, 902, 903, 904, 905]
TEMPERATURE = 0.7
ALL_GENERATIONS = "all generations done"
GENESIS = "0" * 64

Y = ("breeding first", "vote first", "tie")
Y1 = ("champion first", "seed first", "tie")
Y2 = ("lexicase first", "dgm first", "tie")

LOG_LINE = re.compile(r'^\{"seq":(\d+),"prev":"([0-9a-f]{64})","hash":"([0-9a-f]{64})","event":(.*)\}$')
RUN_NAME = re.compile(r"^([BL])-([0-9]+)$")


class Refused(Exception):
    pass


def refuse(message):
    raise Refused(message)


def load(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError) as e:
        refuse(f"{path}: {e}")


def non_empty_lines(path):
    if not os.path.exists(path):
        return 0

    with open(path, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def sha256_hex(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical(value):
    """JSON as serde_json writes it: no spaces, keys in the order read, text as UTF-8."""
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


def derive_seed(master, tag):
    """The first 8 bytes, little-endian, of SHA-256(master as 8 little-endian bytes, tag)."""
    digest = hashlib.sha256(master.to_bytes(8, "little") + tag.encode("utf-8")).digest()

    return int.from_bytes(digest[:8], "little")


def wire(seed):
    """The low 63 bits, as the sample seeds of the vote are masked."""
    return seed & ((1 << 63) - 1)


def call_seeds(cascade, base):
    """Every request seed a run of `cascade` from the seed `base` can carry: Worker j of
    stage k answers with derive(base, "stage{k}/worker{j}"); when a stage's Workers disagree,
    Worker j judges the pair (x, y) of distinct answers with derive(v, "{x}/{y}/{j}"), where
    v = derive(base, "vote{k}")."""
    seeds = []

    for k, stage in enumerate(cascade["stages"]):
        w = len(stage["workers"])
        vote = derive_seed(base, f"vote{k}")
        seeds += [derive_seed(base, f"stage{k}/worker{j}") for j in range(w)]

        for x in range(w):
            for y in range(x + 1, w):
                seeds += [derive_seed(vote, f"{x}/{y}/{j}") for j in range(w)]

    return seeds


def workers(cascade):
    return sum(len(s["workers"]) for s in cascade["stages"])


def cascade_id(cascade):
    return sha256_hex(canonical(cascade))[:16]


def total_key(x):
    """An integer in the order of IEEE 754 totalOrder, as Rust's f64::total_cmp has it."""
    bits = struct.unpack("<q", struct.pack("<d", x))[0]

    return bits ^ 0x7FFFFFFFFFFFFFFF if bits < 0 else bits


def ranked(archive):
    """Archive indices best first: fitness, then peer rating, both high first; then id."""
    return sorted(
        range(len(archive)),
        key=lambda i: (-total_key(archive[i]["fitness"]), -total_key(archive[i]["rating"]), archive[i]["id"]),
    )


def champion_of(archive, finalists):
    """The bred member with the best search standing, if the select phase scored one."""
    if finalists < 1:
        return None

    return next((i for i in ranked(archive) if archive[i]["op"] != "seed"), None)


def voted_seed_of(archive):
    """The seed member with the best search standing."""
    return next(i for i in ranked(archive) if archive[i]["op"] == "seed")


def fixed_hash(cfg):
    """The hash of the settings that define the run: all but generations and max_calls."""
    fixed = dict(cfg)
    fixed["generations"] = 0
    fixed["max_calls"] = 0

    return sha256_hex(canonical(fixed))


def puzzle_hash(p, salt):
    material = [[{"input": t["input"], "output": t["output"]} for t in p["train"]], p["test_input"], p["test_output"]]

    if salt is not None:
        material.append(salt)

    return sha256_hex(canonical(material))[:16]


def read_log(path):
    """The events of a hash-chained log and its head, every line checked."""
    with open(path, encoding="utf-8") as f:
        text = f.read()

    if text and not text.endswith("\n"):
        refuse(f"{path}: the last line is incomplete")

    head = GENESIS
    events = []

    for i, line in enumerate(text.split("\n")[:-1]):
        m = LOG_LINE.match(line)

        if not m:
            refuse(f"{path}: entry {i} is not a log entry")

        seq, prev, digest, event = m.groups()

        if int(seq) != i or prev != head or digest != sha256_hex(f"{prev}\n{event}"):
            refuse(f"{path}: entry {i} breaks the hash chain")

        head = digest
        events.append(json.loads(event))

    return events, head


def check_started(path, arm, seed, frozen):
    config = os.path.join(path, "config.json")

    if os.path.exists(config):
        cfg = load(config)

        if cfg.get("seed") != seed:
            refuse(f"{path}: the run has seed {cfg.get('seed')}, its name says {seed}")

        ours = {k: v for k, v in cfg.items() if k != "seed"}
        theirs = {k: v for k, v in frozen[arm].items() if k != "seed"}

        if ours != theirs:
            differ = sorted(k for k in set(ours) | set(theirs) if ours.get(k) != theirs.get(k))
            refuse(f"{path}: the settings differ from config-{arm}.json in {differ}")

    failed = non_empty_lines(os.path.join(path, "llm", "worker-errors.jsonl"))

    if failed:
        refuse(f"{path}: {failed} Worker call(s) failed; such a run is not counted (section 8)")


def load_run(path, arm, seed, frozen):
    """A finished run, checked: settings, failed calls, the whole generations, the log."""
    check_started(path, arm, seed, frozen)
    cfg = load(os.path.join(path, "config.json"))
    checkpoint = load(os.path.join(path, "snapshot.json"))
    snap = checkpoint["snapshot"]
    report = load(os.path.join(path, "report.json"))
    planned = cfg["generations"] + 1

    if (not snap.get("finished") or snap["next_generation"] != planned or report["stop_reason"] != ALL_GENERATIONS
            or report["generations_completed"] != planned):
        refuse(f"{path}: the run ended with {report['stop_reason']!r} after "
               f"{report['generations_completed']} of {planned} generations")

    events, head = read_log(os.path.join(path, "events.jsonl"))

    if checkpoint["log_len"] != len(events) or checkpoint["log_head"] != head:
        refuse(f"{path}: the checkpoint does not stand at the end of the log")

    start = events[0] if events else {}

    if start.get("kind") != "run_start":
        refuse(f"{path}: the first event is not run_start")

    if not (start["config_hash"] == snap["config_hash"] == fixed_hash(cfg)):
        refuse(f"{path}: the settings, the snapshot and the log disagree")

    last_generation = max((i for i, e in enumerate(events) if e["kind"] == "generation"), default=-1)
    finalists = [{k: v for k, v in e.items() if k != "kind"}
                 for e in events[last_generation + 1:] if e["kind"] == "finalist"]

    if finalists != report["finalists"]:
        refuse(f"{path}: the report's select results differ from the logged ones")

    return {
        "path": path,
        "cfg": cfg,
        "snap": snap,
        "search": start["search"],
        "select": start["select"],
        "logged": {f["id"]: f["select_solved"] for f in finalists},
    }


def check_panels(b, l):
    """The puzzles of panels.json, each checked against its logged content id: their answers."""
    panels = load(os.path.join(b["path"], "panels.json"))

    if panels["seed"] != b["cfg"]["seed"]:
        refuse(f"{b['path']}: panels.json names seed {panels['seed']}")

    for name in ("search", "select"):
        listed = [p["id"] for p in panels[name]]

        if listed != b[name] or listed != l[name]:
            refuse(f"{b['path']}: the {name} panel of panels.json is not the logged one")

        seen = set()

        for i, p in enumerate(panels[name]):
            plain = puzzle_hash(p, None)
            salted = puzzle_hash(p, i)

            if p["id"] != (plain if plain not in seen else salted):
                refuse(f"{b['path']}: puzzle {i} of the {name} panel does not hash to its id")

            seen.add(plain)
            seen.add(p["id"])

    return {p["id"]: p["test_output"] for p in panels["select"]}


def solved(run, member, pid, truth):
    answer = run["snap"]["select_answers"].get(member, {}).get(pid)

    return answer is not None and answer == truth[pid]


def solved_on(run, member, counted, truth):
    return sum(1 for pid in counted if solved(run, member, pid, truth))


def check_logged(run, member, truth):
    every = solved_on(run, member, run["select"], truth)

    if run["logged"].get(member) != every:
        refuse(f"{run['path']}: member {member} solves {every} select puzzles, the log says "
               f"{run['logged'].get(member)}")


def seed_counts(path, wanted, in_progress=False):
    """How many records of the call record `path` carry each seed of `wanted`. With
    `in_progress` (a vote that may still run), a last line without its newline is a call
    being filed and is not read; otherwise it is an error."""
    counts = {}

    if not os.path.exists(path):
        return counts

    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.endswith("\n"):
                if in_progress:
                    break

                refuse(f"{path}: the last line is incomplete")

            if not line.strip():
                continue

            seed = json.loads(line)["request"]["seed"]

            if seed in wanted:
                counts[seed] = counts.get(seed, 0) + 1

    return counts


def champion_calls(b, member, counted):
    """K_c: the records of the run's own record that carry a seed of the champion's select
    runs on the counted puzzles. Every select puzzle must have at least one call per Worker."""
    seed = b["cfg"]["seed"]
    cascade = member["cascade"]
    per_puzzle = {pid: call_seeds(cascade, derive_seed(seed, f"select/{member['id']}/{pid}")) for pid in b["select"]}
    wanted = {s for seeds in per_puzzle.values() for s in seeds}
    counts = seed_counts(os.path.join(b["path"], "llm", "worker-calls.jsonl"), wanted)
    calls = {pid: sum(counts.get(s, 0) for s in seeds) for pid, seeds in per_puzzle.items()}

    if any(c < workers(cascade) for c in calls.values()):
        refuse(f"{b['path']}: the record lacks select-phase calls of the champion")

    return sum(calls[pid] for pid in counted)


def vote_samples(b, voted, voted_id, counted):
    """The counted calls of each vote sample on file, in sample order. A sample is on file
    when every select puzzle has at least one record per Worker for it."""
    seed = b["cfg"]["seed"]
    path = os.path.join(b["path"], "vote-control", "worker-calls.jsonl")
    lines = non_empty_lines(path)
    most = lines // max(1, len(b["select"]))
    plan = {(k, pid): call_seeds(voted, wire(derive_seed(seed, f"vote/{voted_id}/{k}/{pid}")))
            for k in range(most + 1) for pid in b["select"]}
    counts = seed_counts(path, {s for seeds in plan.values() for s in seeds}, in_progress=True)
    samples = []

    for k in range(most + 1):
        calls = {pid: sum(counts.get(s, 0) for s in plan[(k, pid)]) for pid in b["select"]}

        if any(c < workers(voted) for c in calls.values()):
            break

        samples.append(sum(calls[pid] for pid in counted))

    return samples


def match(k_c, samples, floor):
    """N_up, K_v(N_up), N_down, K_v(N_down), or the samples still needed (section 4)."""
    if k_c <= 0 or floor <= 0 or any(c < floor for c in samples):
        refuse("the call matching has impossible inputs")

    k_v = 0

    for i, calls in enumerate(samples):
        before = k_v
        k_v += calls

        if k_v >= k_c:
            n_up = i + 1

            if k_v == k_c:
                return {"n_up": n_up, "k_v_up": k_v, "n_down": n_up, "k_v_down": k_v}

            if n_up >= 2:
                return {"n_up": n_up, "k_v_up": k_v, "n_down": n_up - 1, "k_v_down": before}

            return {"n_up": n_up, "k_v_up": k_v, "n_down": None, "k_v_down": None}

    return {"need": len(samples) + -(-(k_c - k_v) // floor)}


def kept_vote(path, n, b, voted_member, voted_id, champion):
    """A kept vote result, checked against the run and the call matching."""
    kept = load(path)
    plan = kept["cascade"]

    if not kept["complete"] or kept["partial"] or kept["failed_calls"]:
        refuse(f"{path}: not a complete vote")

    if kept["samples_per_puzzle"] != n:
        refuse(f"{path}: {kept['samples_per_puzzle']} samples per puzzle, the call matching gives {n}")

    if plan["member"] != voted_member or plan["id"] != voted_id or plan["temperature"] != TEMPERATURE:
        refuse(f"{path}: it does not sample the voted seed at temperature {TEMPERATURE}")

    if (kept.get("champion") or {}).get("id") != champion or kept["run_seed"] != b["cfg"]["seed"]:
        refuse(f"{path}: it belongs to another champion or run")

    if [p["id"] for p in kept["puzzles"]] != b["select"]:
        refuse(f"{path}: its puzzles are not the select panel")

    return kept


def vote_solved(kept, path, b, voted_member, champion, truth, counted):
    """The counted puzzles a kept vote solves, graded here against the answers of the
    puzzles; its own grades must agree."""
    right = 0

    for p in kept["puzzles"]:
        mine = p["majority"] is not None and p["majority"] == truth[p["id"]]

        if mine != p["correct"]:
            refuse(f"{path}: puzzle {p['id']} is graded otherwise there")

        if p["champion_correct"] != solved(b, champion, p["id"], truth):
            refuse(f"{path}: the champion's answer on {p['id']} is graded otherwise there")

        if p["single_correct"] != solved(b, voted_member, p["id"], truth):
            refuse(f"{path}: the voted seed's answer on {p['id']} is graded otherwise there")

        right += mine and p["id"] in counted

    return right


def outcome(a, b, names):
    return names[0] if a > b else names[1] if a < b else names[2]


def arm_state(runs, arm, seed, frozen):
    path = os.path.join(runs, f"{arm}-{seed}")

    if not os.path.isdir(path):
        return "missing", None

    if not os.path.exists(os.path.join(path, "report.json")):
        check_started(path, arm, seed, frozen)
        return "unfinished", None

    return "finished", load_run(path, arm, seed, frozen)


def pair(runs, seed, frozen):
    """("complete", values), ("waiting", why) or ("missing", None) for one seed."""
    sb, b = arm_state(runs, "B", seed, frozen)
    sl, l = arm_state(runs, "L", seed, frozen)

    if sb == sl == "missing":
        return "missing", None

    if not (sb == sl == "finished"):
        word = {"missing": "not started", "unfinished": "not finished", "finished": "finished"}
        return "waiting", f"arm B {word[sb]}, arm L {word[sl]}"

    if b["search"] != l["search"] or b["select"] != l["select"]:
        refuse(f"seed {seed}: the arms have other panels")

    def seeds(run):
        return [(m["id"], m["answers"]) for m in run["snap"]["archive"] if m["op"] == "seed"]

    if seeds(b) != seeds(l):
        refuse(f"seed {seed}: the seed Cascades' search answers differ between arms B and L (section 4)")

    search = set(b["search"])
    overlap = [pid for pid in b["select"] if pid in search]
    counted = [pid for pid in b["select"] if pid not in search]
    archive = b["snap"]["archive"]
    c_index = champion_of(archive, b["cfg"]["finalists"])

    if c_index is None:
        refuse(f"{b['path']}: no bred finalist, so no champion")

    champion = archive[c_index]
    k_c = champion_calls(b, champion, counted)
    voted_member = archive[voted_seed_of(archive)]

    if cascade_id(voted_member["cascade"]) != voted_member["id"]:
        refuse(f"{b['path']}: the voted seed's Cascade does not hash to its id")

    voted = json.loads(json.dumps(voted_member["cascade"]))

    for stage in voted["stages"]:
        for w in stage["workers"]:
            w["traits"]["temperature"] = TEMPERATURE

    voted_id = cascade_id(voted)
    samples = vote_samples(b, voted, voted_id, counted)
    m = match(k_c, samples, workers(voted) * len(counted))
    up = os.path.join(b["path"], "vote-up.json")
    down = os.path.join(b["path"], "vote-down.json")

    if "need" in m:
        if os.path.exists(up) or os.path.exists(down):
            refuse(f"{b['path']}: kept vote results exist, but the vote record holds too few samples")

        return "waiting", f"the vote record holds {len(samples)} of at least {m['need']} samples"

    if not os.path.exists(up):
        return "waiting", "no vote-up.json yet"

    args = (b, voted_member["id"], voted_id, champion["id"])
    kept_up = kept_vote(up, m["n_up"], *args)

    if m["n_down"] is not None and not os.path.exists(down):
        return "waiting", "no vote-down.json yet"

    if m["n_down"] is None and os.path.exists(down):
        refuse(f"{b['path']}: vote-down.json exists, but no N_down exists")

    kept_down = kept_vote(down, m["n_down"], *args) if m["n_down"] is not None else None

    if not os.path.exists(os.path.join(b["path"], "panels.json")):
        return "waiting", "no panels.json yet"

    truth = check_panels(b, l)
    l_archive = l["snap"]["archive"]
    l_index = champion_of(l_archive, l["cfg"]["finalists"])

    if l_index is None:
        refuse(f"{l['path']}: no bred finalist, so no champion")

    champion_l = l_archive[l_index]["id"]

    for run, member in ((b, champion["id"]), (b, voted_member["id"]), (l, champion_l)):
        check_logged(run, member, truth)

    grading = (b, voted_member["id"], champion["id"], truth, set(counted))
    v = vote_solved(kept_up, up, *grading)
    v_down = vote_solved(kept_down, down, *grading) if kept_down is not None else None
    c = solved_on(b, champion["id"], counted, truth)
    plain = solved_on(b, voted_member["id"], counted, truth)
    c_l = solved_on(l, champion_l, counted, truth)

    return "complete", {
        "seed": seed,
        "overlap": overlap,
        "counted_puzzles": len(counted),
        "champion": champion["id"],
        "voted_member": voted_member["id"],
        "voted_cascade": voted_id,
        "c": c,
        "k_c": k_c,
        "n_up": m["n_up"],
        "k_v_up": m["k_v_up"],
        "v": v,
        "y": outcome(c, v, Y),
        "n_down": m["n_down"],
        "k_v_down": m["k_v_down"],
        "v_down": v_down,
        "y_down": None if v_down is None else outcome(c, v_down, Y),
        "seed_plain": plain,
        "y1": outcome(c, plain, Y1),
        "champion_l": champion_l,
        "c_l": c_l,
        "y2": outcome(c_l, c, Y2),
    }


def fraction(w, l):
    return f"{3 ** w}/{2 ** (w + l)}"


def sequential(outcomes, cap):
    """The rule of section 6: outcomes are (Y, Y1, Y2) per pair in list order."""
    counts = {"b": [0, 0], "s1": [0, 0], "s2": [0, 0]}
    steps = []

    for i, (y, y1, y2) in enumerate(outcomes, 1):
        for key, o, names in (("b", y, Y), ("s1", y1, Y1), ("s2", y2, Y2)):
            if o == names[0]:
                counts[key][0] += 1
            elif o == names[1]:
                counts[key][1] += 1
            elif o != names[2]:
                refuse(f"unknown outcome {o!r}")

        w, l = counts["b"]
        steps.append({
            "e_b": fraction(w, l),
            "e_v": fraction(l, w),
            "e_s1": fraction(*counts["s1"]),
            "e_s2": fraction(*counts["s2"]),
            "counts": {k: list(v) for k, v in counts.items()},
        })

        if 3 ** w >= PROOF * 2 ** (w + l):
            return steps, i, "proof"

        if FUTILITY * 3 ** w <= 2 ** (w + l):
            return steps, i, "futility"

        if i >= cap:
            return steps, i, "open"

    return steps, None, "running"


def run_seeds(runs):
    seeds = set()

    for name in os.listdir(runs):
        if not os.path.isdir(os.path.join(runs, name)):
            continue

        m = RUN_NAME.match(name)

        if m:
            if str(int(m.group(2))) != m.group(2):
                refuse(f"{runs}/{name}: write the seed in plain decimal")

            seeds.add(int(m.group(2)))

    return seeds


def recount(runs, frozen, pilot):
    seeds = PILOT if pilot else list(range(1, CAP + 1))
    found = run_seeds(runs)
    off = sorted(found - set(seeds))

    if off:
        if pilot:
            refuse(f"seeds {off} are not pilot seeds")

        if any(s in PILOT for s in off):
            refuse(f"seeds {off}: pilot seeds never count")

        if any(s > CAP for s in off):
            refuse(f"seeds {off}: the experiment counts at most {CAP} pairs (D6)")

        refuse(f"seeds {off} are not on the list 1 to {CAP}")

    for i, s in enumerate(seeds):
        if s not in found and any(t in found for t in seeds[i + 1:]):
            refuse(f"seed {s} has no run, but later seeds of the list do: pairs enter in list order")

    started = [s for s in seeds if s in found]

    states = [(s,) + pair(runs, s, frozen) for s in started]
    complete = []

    for s, state, value in states:
        if state != "complete":
            break

        complete.append(value)

    steps, stop, primary = sequential([(p["y"], p["y1"], p["y2"]) for p in complete], len(seeds))
    rows = []

    for i, p in enumerate(complete):
        row = {"pair": i + 1, "counted": i < len(steps)}
        row.update(p)

        for key in ("e_b", "e_v", "e_s1", "e_s2"):
            row[key] = steps[i][key] if i < len(steps) else None

        rows.append(row)

    last = steps[-1] if steps else {"e_b": "1/1", "e_v": "1/1", "e_s1": "1/1", "e_s2": "1/1",
                                     "counts": {"b": [0, 0], "s1": [0, 0], "s2": [0, 0]}}
    claims = None

    if not pilot and stop is not None:
        claims = tuple(3 ** last["counts"][k][0] >= SECONDARY * 2 ** sum(last["counts"][k]) for k in ("s1", "s2"))

    return {
        "list": "pilot" if pilot else "counted",
        "cap": len(seeds),
        "pairs": rows,
        "waiting": [{"seed": s, "why": v if state == "waiting" else f"complete; it enters after seed "
                     f"{seeds[len(complete)]}"} for s, state, v in states[len(complete):]],
        "counted_pairs": len(steps),
        "stop": None if stop is None else {"pair": stop, "seed": seeds[stop - 1]},
        "primary": primary,
        "e_b": last["e_b"],
        "e_v": last["e_v"],
        "e_s1": last["e_s1"],
        "e_s2": last["e_s2"],
        "s1_claim": None if claims is None else claims[0],
        "s2_claim": None if claims is None else claims[1],
        "next_seed": seeds[len(started)] if stop is None and len(started) < len(seeds) else None,
    }


def show(fr):
    return f"{fr} = {float(Fraction(fr)):.4f}"


def print_result(r):
    print(f"recount: {r['list']} seeds, {len(r['pairs'])} complete pair(s), {r['counted_pairs']} counted")
    print(f"{'pair':>4} {'seed':>5} {'ovl':>3} {'C':>4} {'V':>4} {'N_up':>4} {'K_c':>6} {'K_v_up':>6} "
          f"{'Y':<14} {'N_down':>6} {'V_down':>6} {'plain':>5} {'Y1':<14} {'C_L':>4} {'Y2':<14} E_B")

    for p in r["pairs"]:
        print(f"{p['pair']:>4} {p['seed']:>5} {len(p['overlap']):>3} {p['c']:>4} {p['v']:>4} {p['n_up']:>4} "
              f"{p['k_c']:>6} {p['k_v_up']:>6} {p['y']:<14} {str(p['n_down']):>6} {str(p['v_down']):>6} "
              f"{p['seed_plain']:>5} {p['y1']:<14} {p['c_l']:>4} {p['y2']:<14} "
              f"{p['e_b'] if p['counted'] else 'after the stop, not counted'}")

    for w in r["waiting"]:
        print(f"seed {w['seed']}: not counted yet: {w['why']}")

    if r["list"] == "pilot":
        print("Pilot seeds: shown for illustration only; no claim.")

    stop = r["stop"]
    where = f" at pair {stop['pair']} (seed {stop['seed']})" if stop else ""
    print(f"Primary: {r['primary']}{where}. E_B = {show(r['e_b'])}. E_V = {show(r['e_v'])}, without a claim.")

    for name, key in (("S1", "s1_claim"), ("S2", "s2_claim")):
        claim = r[key]
        word = "read at the stop" if claim is None else ("claim (E >= 200)" if claim else "no claim")
        print(f"{name}: E = {show(r['e_' + name.lower()])}; {word}.")

    if r["next_seed"] is not None:
        print(f"Next: run seed {r['next_seed']}.")


def main(argv):
    args = argv[1:]
    opts = {}

    while args:
        key = args.pop(0)

        if key == "--pilot":
            opts["pilot"] = True
        elif key in ("--runs", "--frozen", "--json", "--rule") and args:
            opts[key[2:]] = args.pop(0)
        else:
            sys.exit("recount: usage: recount.py --runs <dir> [--pilot] [--frozen <dir>] [--json <file>] "
                     "| --rule <file>")

    try:
        if "rule" in opts:
            result = {"results": []}

            for case in load(opts["rule"])["cases"]:
                steps, stop, primary = sequential([tuple(o) for o in case["outcomes"]], case["cap"])
                result["results"].append({
                    "steps": [{k: s[k] for k in ("e_b", "e_v", "e_s1", "e_s2")} for s in steps],
                    "stop": stop,
                    "primary": primary,
                })
        elif "runs" in opts:
            frozen_dir = opts.get("frozen", os.path.dirname(os.path.abspath(__file__)))
            frozen = {arm: load(os.path.join(frozen_dir, f"config-{arm}.json")) for arm in ("B", "L")}
            result = recount(opts["runs"], frozen, opts.get("pilot", False))
            print_result(result)
        else:
            sys.exit("recount: give --runs <dir> or --rule <file>")
    except Refused as e:
        sys.exit(f"recount: {e}")

    if "json" in opts:
        with open(opts["json"], "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
            f.write("\n")
    elif "rule" in opts:
        print(json.dumps(result))


if __name__ == "__main__":
    main(sys.argv)
