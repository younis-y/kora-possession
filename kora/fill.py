"""Short ball-gap repair (variant switch "fill", RESEARCH.md Y15).

The challenge match loses the ball on 41% of frames; features then carry the last-seen values and
the touch features cannot fire. Most live gaps are short, and offline both ends are known, so a gap
of fewer than MAX_GAP frames inside one half, seen on both sides, is filled with a straight line
between its ends. Longer gaps stay NaN. `b_filled` marks the filled frames, so the classifier can
still tell a measured ball from an interpolated one. No labels are read.
"""
import numpy as np

MAX_GAP = 25                                   # frames (1 s)


def fill(ball, half):
    """(ball with short gaps filled linearly, mask of filled frames); ball is [n, 2], NaN = not seen."""
    out = ball.copy()
    mask = np.zeros(len(ball), bool)
    miss = ~np.isfinite(ball[:, 0])
    edges = np.diff(np.r_[0, miss.astype(np.int8), 0])
    for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if a == 0 or b == len(ball) or b - a >= MAX_GAP or half[a - 1] != half[b]:
            continue
        w = (np.arange(a, b) - (a - 1)) / (b - (a - 1))
        out[a:b] = ball[a - 1] + w[:, None] * (ball[b] - ball[a - 1])
        mask[a:b] = True
    return out, mask


def wrap(build):
    """features.build with short ball gaps filled first and the `b_filled` column appended."""
    def built(players, team, ball, half):
        b, mask = fill(ball, half)
        X = build(players, team, b, half)
        X["b_filled"] = mask.astype(np.float32)
        return X
    return built
