"""Boundary-cut relabelling (RESEARCH.md Y10; ASRF, Ishikawa et al., WACV 2021).

Every change of possession costs the decoder a switching penalty, so a short possession inside the
other team's spell is cheaper to swallow than to cut out. ASRF recovers such segments by cutting the
decoded path where a boundary detector peaks and relabelling each piece by its mean frame
probabilities. Here:

1. peaks: local maxima of the detector's team-change probability at or above theta, at least `gap`
   frames apart (strongest first);
2. every H/A run of the decoded path is cut at the peaks inside it (dead-ball runs are never touched);
3. each piece takes H or A by its summed stage-1 probabilities (a tie keeps the decoded label).
"""
import numpy as np


def peaks(change, theta, gap):
    """Frames of local maxima of `change` >= theta, greedily kept strongest first, >= gap apart."""
    left = np.r_[-np.inf, change[:-1]]
    right = np.r_[change[1:], -np.inf]
    cand = np.flatnonzero((change >= theta) & (change >= left) & (change > right))
    taken = np.zeros(len(change), bool)
    keep = []
    for i in cand[np.argsort(-change[cand], kind="stable")]:
        if not taken[i]:
            keep.append(i)
            taken[max(i - gap + 1, 0):i + gap] = True
    return np.sort(np.array(keep, dtype=np.int64))


def relabel(path, proba, change, half, theta, gap):
    """`path` (0 H, 1 A, 2 D) with its H/A runs cut at detector peaks and relabelled by stage 1."""
    n = len(path)
    bounds = np.unique(np.r_[0, np.flatnonzero(np.diff(path) != 0) + 1,
                             np.flatnonzero(np.diff(half) != 0) + 1, peaks(change, theta, gap), n])
    cum = np.vstack([np.zeros((1, 2)), np.cumsum(proba[:, :2], 0)])
    out = path.copy()
    for a, b in zip(bounds[:-1], bounds[1:]):
        if path[a] == 2:
            continue
        h, aw = cum[b] - cum[a]
        if h != aw:
            out[a:b] = 0 if h > aw else 1
    return out
