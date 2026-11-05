"""Where the held-out score is lost (RESEARCH.md Y13): spellF1 if one error type were fixed perfectly.

    python -m kora.ceiling mixed mixed_rand      # reads models/nested_<variant>_predictions.npz

Each oracle copies the truth into the held-out prediction for one kind of error only:

- swallowed   true H/A spells the prediction covers on less than half their frames
- short       true H/A spells under 2 s
- edges       frames within 12 (0.5 s) of a true change that also has a predicted change within 12

Plus, inside the swallowed spells, how well stage 1 (the cached outer fold) sees them, against how
often the other team spikes inside long spells the prediction holds correctly. External matches only.
"""
import json
import sys

import numpy as np

from .data import cache_dir
from .nested import scores
from .train import MODELS
from .variant import VARIANTS, stage1_version

EDGE = 12


def spells(c):
    b = np.r_[0, np.flatnonzero(np.diff(c) != 0) + 1, len(c)]
    return [(a, e, int(c[a])) for a, e in zip(b[:-1], b[1:])]


def oracles(y, p):
    out = {}
    q = p.copy()
    for a, e, k in spells(y):
        if k < 2 and (p[a:e] == k).mean() < 0.5:
            q[a:e] = k
    out["swallowed"] = q
    q = p.copy()
    for a, e, k in spells(y):
        if k < 2 and e - a < 50:
            q[a:e] = k
    out["short"] = q
    q, pc = p.copy(), np.flatnonzero(np.diff(p) != 0)
    for t in np.flatnonzero(np.diff(y) != 0):
        j = np.searchsorted(pc, t)
        near = [pc[i] for i in (j - 1, j) if 0 <= i < len(pc) and abs(pc[i] - t) <= EDGE]
        if near:
            lo = max(min(t, near[0]) - EDGE, 0)
            q[lo:t + EDGE + 1] = y[lo:t + EDGE + 1]
    out["edges"] = q
    return out


def report(variant):
    z = np.load(MODELS / f"nested_{variant}_predictions.npz")
    ms = json.loads(str(z["meta"]))["matches"]
    f1 = {"today": [], "swallowed": [], "short": [], "edges": []}
    seen, spike, held = [], [], []
    for m in ms:
        y, p, h = (z[f"{m}__{k}"].astype(int) for k in ("y_true", "y_pred", "half"))
        f1["today"].append(scores(y, p, h)["mean_f1"])
        for k, q in oracles(y, p).items():
            f1[k].append(scores(y, q, h)["mean_f1"])
        path = cache_dir() / f"oof_{stage1_version(VARIANTS[variant])}_{m}.npy"
        if path.exists():
            P = np.load(path)
            for a, e, k in spells(y):
                if k < 2 and (p[a:e] == k).mean() < 0.5:
                    seen.append(((P[a:e, k] > P[a:e, 1 - k]).mean(), P[a:e, k].max()))
                elif k < 2 and e - a >= 100 and (p[a:e] == k).mean() > 0.9:
                    held.append(P[a:e, 1 - k].max())
    print(f"{variant}: " + "  ".join(f"{k} {np.mean(v):.4f}" for k, v in f1.items()))
    if seen:
        s, held = np.array(seen), np.array(held)
        print(f"  swallowed spells {len(s)}: stage 1 prefers the true team on >= half their frames in "
              f"{(s[:, 0] >= 0.5).mean():.2f}; true team reaches P >= 0.5 in {(s[:, 1] >= 0.5).mean():.2f}, "
              f"other team inside long held spells in {(held >= 0.5).mean():.2f}")


if __name__ == "__main__":
    for v in sys.argv[1:] or ["mixed"]:
        report(v)
