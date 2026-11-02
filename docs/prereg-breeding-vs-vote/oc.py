"""Operating characteristics of the sequential rule of docs/PREREG_BREEDING_VS_VOTE.md.

Each counted pair gives one outcome: "breeding first" (the evidence value E is multiplied
by 3/2), "vote first" (E is multiplied by 1/2) or a tie (E stays). The rule proves its
claim when E >= 100 and stops for futility when E <= 1/20; it stops at the cap n_max
otherwise. The decisions are made in integers, as in step 1: after w wins and l losses,
E = 3^w / 2^(w + l).

The script computes, by exact dynamic programming over the pairs, the chance of each
ending and the expected number of pairs, for a given chance of a win among untied pairs
and a given tie rate. It reads no file and needs only the Python standard library.

Usage: python3 -I -B oc.py
"""


def decide(w, l):
    """The ending after w wins and l losses, or None while the rule goes on."""
    if 3**w >= 100 * 2 ** (w + l):
        return "proof"

    if 20 * 3**w <= 2 ** (w + l):
        return "futile"

    return None


def operating_characteristics(p_win, p_tie, n_max):
    """Chances of proof, futility and an open end, and the expected number of pairs."""
    p_loss = 1.0 - p_win - p_tie
    alive = {(0, 0): 1.0}
    ends = {"proof": 0.0, "futile": 0.0, "open": 0.0}
    expected_pairs = 0.0

    for n in range(1, n_max + 1):
        step = {}

        for (w, l), pr in alive.items():
            for dw, dl, q in ((1, 0, p_win), (0, 1, p_loss), (0, 0, p_tie)):
                if q > 0.0:
                    key = (w + dw, l + dl)
                    step[key] = step.get(key, 0.0) + pr * q

        alive = {}

        for (w, l), pr in step.items():
            ending = decide(w, l)

            if ending is None:
                alive[(w, l)] = pr
            else:
                ends[ending] += pr
                expected_pairs += pr * n

    rest = sum(alive.values())
    ends["open"] = rest
    expected_pairs += rest * n_max

    return ends, expected_pairs


def main():
    p_tie = 0.2

    print(f"tie rate {p_tie}; win = chance of 'breeding first' among untied pairs")
    print("n_max  win   P(proof)  P(futile)  P(open)  E[pairs]")

    for n_max in (30, 60, 120):
        for win in (0.5, 0.6, 0.7, 0.75, 0.8, 0.9):
            ends, pairs = operating_characteristics(win * (1.0 - p_tie), p_tie, n_max)

            print(
                f"{n_max:5d}  {win:.2f}  {ends['proof']:8.3f}  {ends['futile']:9.3f}"
                f"  {ends['open']:7.3f}  {pairs:8.1f}"
            )


if __name__ == "__main__":
    main()
