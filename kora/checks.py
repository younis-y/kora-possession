"""Label-free plausibility checks on a submission file (no ground truth is read).

    python -m kora.checks "Kora Analytics.csv"

1. Possession count, mean possession length and dead-ball share per half, next to the same numbers
   in the seven training matches' official labels.
2. Agreement with Metrica's automatic Smart Tagging: inside phases whose code names an attacking
   team (BUILD UP, PROGRESSION, ...) how often do we give that team the ball; inside DEFENDING
   phases, how often the other team; inside SET PIECES, how often dead ball. Smart Tagging is itself
   automatic, so this is a consistency check, not a score.
"""
import sys

import numpy as np
import pandas as pd

from .data import FPS, smart_tagging, teams
from .external import MATCHES, load
from .target import load_target

ATTACKING = {"BUILD UP", "PROGRESSION", "CREATING CHANCES", "BALL IN FINAL THIRD", "BALL IN THE BOX",
             "PLAYERS IN THE BOX", "ATTACKING TRANSITION", "LONG BALL"}
DEFENDING = {"DEFENDING IN DEFENSIVE THIRD", "DEFENDING IN MIDDLE THIRD", "DEFENDING IN ATTACKING THIRD",
             "DEFENSIVE TRANSITION"}


def spells(codes):
    """(number of possessions, mean length s, dead share) for one half of codes in {0,1,2}."""
    change = np.flatnonzero(codes[1:] != codes[:-1]) + 1
    starts = np.r_[0, change]
    n_poss = int(np.isin(codes[starts], [0, 1]).sum())
    live = int(np.isin(codes, [0, 1]).sum())
    return n_poss, live / max(n_poss, 1) / FPS, float((codes == 2).mean())


def main(path):
    lab = pd.read_csv(path)["label"].to_numpy()
    code = np.select([lab == "H", lab == "A", lab == "D"], [0, 1, 2], 3)
    _, _, _, _, graded = load_target()
    print("per half                  possessions  mean s  dead share  per 45 min")
    for i, (a, b) in enumerate(graded, 1):
        n, m, dsh = spells(code[a:b])
        print(f"  challenge half {i}        {n:6d}   {m:6.1f}   {dsh:8.2f}   {n / ((b - a) / FPS / 60) * 45:7.0f}")
    rows = []
    for mt in MATCHES:
        d = load(mt)
        for h in (1, 2):
            c = d["label"][d["half"] == h]
            n, m, dsh = spells(c)
            rows.append((n / (len(c) / FPS / 60) * 45, m, dsh))
    r = np.array(rows)
    print(f"  training halves (14)  range per 45 min {r[:, 0].min():.0f}-{r[:, 0].max():.0f}, "
          f"mean length {r[:, 1].min():.1f}-{r[:, 1].max():.1f} s, dead share {r[:, 2].min():.2f}-{r[:, 2].max():.2f}")

    home, away, _ = teams()
    st = smart_tagging()
    agree = {"attacking": [0, 0], "defending": [0, 0], "set piece -> D": [0, 0]}
    for t in st.itertuples():
        seg = code[t.start_frame:t.end_frame]
        seg = seg[seg != 3]
        if not len(seg):
            continue
        code_u = (t.code or "").upper()
        side = 0 if t.team == home else 1 if t.team == away else None
        if code_u in ATTACKING and side is not None:
            live = seg[seg < 2]
            agree["attacking"][0] += int((live == side).sum()); agree["attacking"][1] += len(live)
        elif code_u in DEFENDING and side is not None:
            live = seg[seg < 2]
            agree["defending"][0] += int((live == 1 - side).sum()); agree["defending"][1] += len(live)
        elif code_u == "SET PIECES":
            agree["set piece -> D"][0] += int((seg == 2).sum()); agree["set piece -> D"][1] += len(seg)
    print("agreement with Smart Tagging (automatic, so a consistency check only):")
    for k, (a, n) in agree.items():
        print(f"  {k:<16} {a / max(n, 1):.2f} of {n:,} frames")


if __name__ == "__main__":
    main(sys.argv[1])
