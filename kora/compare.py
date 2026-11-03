"""Paired comparison of two nested-CV variants, with the keep rule (RESEARCH.md Y2).

    python -m kora.compare b0 touch

Reads models/nested_<variant>.json and, when present, models/nested_<variant>_predictions.npz.
Keep rule: the candidate's mean spellF1 is higher AND it is better on at least 5 of the 7 matches.
"""
import json
import sys

import numpy as np

from .train import MODELS
from .vendor import possession_metrics as pm

KEEP_WINS = 5
SHORT_S = (1.0, 2.0)          # true-possession length buckets for recall: <1 s, 1-2 s


def load(variant):
    summary = json.loads((MODELS / f"nested_{variant}.json").read_text())
    path = MODELS / f"nested_{variant}_predictions.npz"
    bundle = np.load(path, allow_pickle=False) if path.exists() else None
    return summary, bundle


def short_recall(bundle, match, iou=0.5):
    """Recall of true possessions shorter than each SHORT_S bound (and of all), at one IoU."""
    y, p, h = (bundle[f"{match}__{k}"].astype(np.int64) for k in ("y_true", "y_pred", "half"))
    cut = np.flatnonzero(np.diff(h) != 0) + 1
    gt, pr = np.insert(y, cut, pm.X), np.insert(p, cut, pm.X)
    runs = pm.valid_runs(gt)
    ts, ps = pm.possession_spells(gt, runs), pm.possession_spells(pr, runs)
    hit = pm.matched_gt_flags(pm.spell_overlaps(ts, ps), len(ts), iou)
    length = np.array([(b - a) / 25 for a, b, _ in ts])
    out, lo = [], 0.0
    for hi in SHORT_S:
        m = (length >= lo) & (length < hi)
        out.append((int(hit[m].sum()), int(m.sum())))
        lo = hi
    out.append((int(hit.sum()), len(hit)))
    return out


def report(base, cand):
    (sb, bb), (sc, bc) = load(base), load(cand)
    matches = list(sb["folds"])
    if list(sc["folds"]) != matches:
        raise SystemExit("variants were scored on different matches")
    a = np.array([sb["folds"][m]["spell_f1"] for m in matches])
    b = np.array([sc["folds"][m]["spell_f1"] for m in matches])
    d = b - a
    print(f"{'match':8s} {base:>10s} {cand:>10s}   delta   n_true  n_pred {base}/{cand}")
    for m, x, z, dd in zip(matches, a, b, d):
        fb, fc = sb["folds"][m], sc["folds"][m]
        print(f"{m:8s} {x:10.4f} {z:10.4f} {dd:+.4f}   {fb['n_true']:6d}  {fb['n_pred']:5d}/{fc['n_pred']:<5d}")
    wins = int((d > 0).sum())
    keep = bool(d.mean() > 0 and wins >= KEEP_WINS)
    print(f"\nmean {a.mean():.4f} -> {b.mean():.4f}  delta {d.mean():+.4f}  median {np.median(d):+.4f}  "
          f"min {d.min():+.4f}  max {d.max():+.4f}  better on {wins}/{len(d)}")
    for name, s in ((base, sb), (cand, sc)):
        ious = np.mean([[v for v in s["folds"][m]["spell_f1_per_iou"].values()] for m in matches], 0)
        ratio = np.mean([s["folds"][m]["n_pred"] / s["folds"][m]["n_true"] for m in matches])
        print(f"{name:>12s}: spellF1 per IoU 0.3-0.7 " + " ".join(f"{v:.3f}" for v in ious) +
              f"   count ratio {ratio:.3f}")
    for name, bundle in ((base, bb), (cand, bc)):
        if bundle is None:
            print(f"{name:>12s}: no prediction bundle, short-possession recall skipped")
            continue
        tot = np.sum([short_recall(bundle, m) for m in matches], 0)
        print(f"{name:>12s}: recall at IoU 0.5  <1 s {tot[0, 0] / tot[0, 1]:.3f}  1-2 s {tot[1, 0] / tot[1, 1]:.3f}"
              f"  all {tot[2, 0] / tot[2, 1]:.3f}  (n {tot[0, 1]}, {tot[1, 1]}, {tot[2, 1]})")
    print(f"\nKEEP {cand}" if keep else f"\nDROP {cand} (keep rule: mean up and better on >= {KEEP_WINS}/{len(d)})")
    return keep


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    report(sys.argv[1], sys.argv[2])
