"""Y27: snap each decoded H<->A change to the change detector's evidence nearby.

The Y13 ceiling puts boundary timing at +0.015 of spellF1, and the HSMM moves in 3-frame blocks. Here
every H<->A change of the decoded path (dead-ball changes are never moved) moves to the detector's
team-change probability within +-r frames: its peak ("peak", a tie keeps the decoded frame) or its centre
of mass ("mass"). Each window stops at the midpoint to the neighbouring changes and inside the half, so
no run disappears and no two moves overlap. Labels are unchanged; only where each run starts moves.
"""
import numpy as np


def snap(path, change, half, r, mode):
    """`path` (0 H, 1 A, 2 D) with every H<->A change moved to the detector's evidence within +-r."""
    out = path.copy()
    b = np.unique(np.r_[0, np.flatnonzero((np.diff(path) != 0) | (np.diff(half) != 0)) + 1, len(path)])
    for i in range(1, len(b) - 1):
        t = b[i]
        if path[t - 1] == 2 or path[t] == 2 or half[t - 1] != half[t]:
            continue
        lo, hi = max(t - r, (b[i - 1] + t) // 2 + 1), min(t + r, (t + b[i + 1]) // 2)
        w = np.arange(lo, hi + 1)
        c = change[w]
        if mode == "peak":
            new = t if change[t] >= c.max() else int(w[c.argmax()])
        else:
            new = int(round(float((w * c).sum() / c.sum()))) if c.sum() > 0 else t
        if new < t:
            out[new:t] = path[t]
        elif new > t:
            out[t:new] = path[t - 1]
    return out
