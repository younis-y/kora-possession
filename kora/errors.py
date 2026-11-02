"""Where does the score go? Error analysis on the training matches, each decoded by models that never
saw it (the same out-of-fold predictions the decoder was chosen on). No challenge labels are used.

    python -m kora.errors       # after kora.train and kora.boundary

Caveat: ball blackouts in these copies are simulated (mostly during dead ball, see kora/degrade.py),
so the "ball missing" rows describe the simulation, not the challenge tracking.
"""
import json
import warnings

import numpy as np

from .boundary import runs_of
from .decode import decode, decode_hsmm, log_duration
from .external import MATCHES
from .train import MODELS, copies, segments_of
from .vendor import possession_metrics as pm

warnings.filterwarnings("ignore", category=RuntimeWarning)
IOU = 0.5
BUCKETS = ((0, 1), (1, 2), (2, 4), (4, 8), (8, 15), (15, 1e9))


def spells(y, half):
    """(start, stop, team) of every H/A run, halves apart."""
    ch = np.flatnonzero((y[1:] != y[:-1]) | (half[1:] != half[:-1])) + 1
    st, en = np.r_[0, ch], np.r_[ch, len(y)]
    return [(a, b, int(y[a])) for a, b in zip(st, en) if y[a] < 2]


def cause(sp, other):
    """Why a spell has no partner at IoU 0.5, judged by what the other sequence says over its frames."""
    a, b, k = sp
    seg = other[a:b]
    same, dead = (seg == k).mean(), (seg == 2).mean()
    if same >= 0.5:
        return "same team, boundaries off"
    return "dead ball instead" if dead >= 1 - same - dead else "wrong team"


def main():
    cfg = json.loads((MODELS / "decoder.json").read_text())
    oof = np.load(MODELS / "oof.npz")
    h = cfg.get("hsmm")
    change = np.load(MODELS / "change_oof.npz") if h else None
    prior = np.array(cfg["prior"])
    data = {m: copies(m)[1] for m in MATCHES}

    missed, spurious, gt_len, gt_hit = {}, {}, [], []
    frames = {"all": [0, 0], "within 1 s of a change": [0, 0], "dead ball": [0, 0], "ball missing (simulated)": [0, 0]}
    n_gt = n_pred = 0
    for m in MATCHES:
        Xe, y, half = data[m]
        if h:
            ld = log_duration([r for k in MATCHES if k != m for r in runs_of(*data[k][1:])])
            pred = decode_hsmm(oof[m], change[m], segments_of(half), prior, ld,
                               h["cost_ha"], h["cost_d"], h["w_dur"], h["beta"], h["rate"])
        else:
            pred = decode(oof[m], segments_of(half), prior, cfg["alpha"], cfg["cost_ha"], cfg["cost_d"])
        G, P = spells(y, half), spells(pred, half)
        ov = pm.spell_overlaps(G, P)
        hit = pm.matched_gt_flags(ov, len(G), IOU)
        used = {j for iou, i, j in ov if iou >= IOU and hit[i]}          # approximate: greedy partner set
        n_gt += len(G); n_pred += len(P)
        for i, g in enumerate(G):
            gt_len.append((g[1] - g[0]) / 25); gt_hit.append(hit[i])
            if not hit[i]:
                c = cause(g, pred); missed[c] = missed.get(c, 0) + 1
        for j, p in enumerate(P):
            if j not in used:
                c = cause(p, y); spurious[c] = spurious.get(c, 0) + 1
        ch = np.flatnonzero((y[1:] != y[:-1]) & (half[1:] == half[:-1])) + 1
        near = np.zeros(len(y), bool)
        for c in ch:
            near[max(c - 25, 0):c + 25] = True
        wrong = pred != y
        for name, mask in (("all", np.ones(len(y), bool)), ("within 1 s of a change", near),
                           ("dead ball", y == 2), ("ball missing (simulated)", Xe["b_seen"].to_numpy() == 0)):
            frames[name][0] += int(wrong[mask].sum()); frames[name][1] += int(mask.sum())

    L, H = np.array(gt_len), np.array(gt_hit)
    tot_wrong = frames["all"][0]
    print(f"## Error analysis, 7 training matches out of fold ({'detector + duration decoder' if h else 'plain decoder'})\n")
    print(f"{n_gt} true possessions, {n_pred} predicted; matched at IoU {IOU}: {H.sum()} "
          f"(recall {H.mean():.2f}, precision {H.sum() / n_pred:.2f})\n")
    print("| true possession length | count | share | found at IoU 0.5 |\n|---|---|---|---|")
    for lo, hi in BUCKETS:
        k = (L >= lo) & (L < hi)
        print(f"| {lo:g}-{hi:g} s | {k.sum()} | {k.mean():.0%} | {H[k].mean():.0%} |".replace("-1e+09", "+"))
    print("\n| missed true possessions, what we said instead | count |\n|---|---|")
    for c, n in sorted(missed.items(), key=lambda x: -x[1]):
        print(f"| {c} | {n} |")
    print("\n| invented possessions, what was really there | count |\n|---|---|")
    for c, n in sorted(spurious.items(), key=lambda x: -x[1]):
        print(f"| {c} | {n} |")
    print("\n| wrong frames | error rate | share of all wrong frames |\n|---|---|---|")
    for name, (w, n) in frames.items():
        print(f"| {name} ({n / frames['all'][1]:.0%} of frames) | {w / max(n, 1):.1%} | {w / tot_wrong:.0%} |")


if __name__ == "__main__":
    main()
