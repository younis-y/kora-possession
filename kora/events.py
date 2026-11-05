"""Touch-event model (RESEARCH.md Y13): who changed the ball's path, and when.

The frame classifier asks every frame "who has it?"; the possessions it misses are short ones whose
frames look like noise inside the other team's spell. ballradar (Kim et al., KDD 2023) and ELASTIC
(arXiv 2508.09238) instead find the moments the ball's path changes and classify those. Here:

- candidates  local maxima of b_acc (|v(t+2) - v(t-2)|, kora/touch.py) over +-2 frames at or above
              0.2 x TOUCH_DV, the weak-touch floor of kora/peaks.py
- table       every feature column at the candidate, the change of key kinematics 3 and 6 frames
              either side, and the gap to and strength of the previous and next candidate
- label       1 / 2 when an H<->A change of possession into H / A lies within RADIUS frames (the
              change detector's own window), else 0; changes to or from dead ball are not events

    python -m kora.events     # the Y13 gate on the production variant's held-out matches

The gate fits, for each held-out match m, the event model on the other six matches' training copies
and the current change detector exactly as kora.nested fits m's outer detector (same cached stage-1
folds), then compares how many of the possessions production swallowed on m each one would cut out,
at equal numbers of false peaks. No challenge data is read.
"""
import json

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingClassifier

from .boundary import RADIUS
from .features import CAP_S, FPS
from .touch import TOUCH_DV

KIN = ["d1H", "d1A", "d2H", "d2A", "relvH", "relvA", "spnH", "spnA", "b_speed", "b_turn", "b_dspeed",
       "b_x", "b_absy", "touchH", "touchA"]
SHIFTS = (-6, -3, 3, 6)
FLOOR = 0.2 * TOUCH_DV


def candidates(X):
    """Frames where the ball's velocity change peaks (+-2 frames) at or above FLOOR."""
    a = np.nan_to_num(X["b_acc"].to_numpy(np.float64), nan=-np.inf)
    local = np.ones(len(a), bool)
    for s in (1, 2):
        local &= a >= np.r_[np.full(s, -np.inf), a[:-s]]
        local &= a > np.r_[a[s:], np.full(s, -np.inf)]
    return np.flatnonzero(local & (a >= FLOOR))


def table(X, idx):
    """One row per candidate frame."""
    n = len(X)
    f = {c: X[c].to_numpy()[idx] for c in X.columns}
    for c in KIN:
        v = X[c].to_numpy()
        for s in SHIFTS:
            f[f"{c}_s{s}"] = v[np.clip(idx + s, 0, n - 1)] - v[idx]
    acc, side = X["b_acc"].to_numpy()[idx], X["touch_diff"].to_numpy()[idx]
    gap = np.diff(idx) / FPS
    f["ev_prev_t"] = np.minimum(np.r_[CAP_S, gap], CAP_S)
    f["ev_next_t"] = np.minimum(np.r_[gap, CAP_S], CAP_S)
    f["ev_prev_acc"], f["ev_next_acc"] = np.r_[np.nan, acc[:-1]], np.r_[acc[1:], np.nan]
    f["ev_prev_side"], f["ev_next_side"] = np.r_[np.nan, side[:-1]], np.r_[side[1:], np.nan]
    return pd.DataFrame({k: np.asarray(v, np.float32) for k, v in f.items()})


def handovers(y, half):
    """Frames c where possession passes between H and A (y[c-1], y[c] in {0, 1}, same half)."""
    return np.flatnonzero((y[1:] != y[:-1]) & (half[1:] == half[:-1]) & (y[1:] < 2) & (y[:-1] < 2)) + 1


def labels(y, half, idx):
    """0 none, 1 / 2 a handover into H / A within RADIUS frames of the candidate."""
    out = np.zeros(len(idx), np.int8)
    for c in handovers(y, half):
        lo, hi = np.searchsorted(idx, [c - RADIUS, c + RADIUS + 1])
        out[lo:hi] = y[c] + 1
    return out


def model(seed):
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, max_leaf_nodes=31, min_samples_leaf=200,
                                          l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
                                          n_iter_no_change=25, random_state=seed)


def fit(copies, seed):
    """Event model from [(X, y, half)] copies with labels."""
    rows, labs = [], []
    for X, y, half in copies:
        idx = candidates(X)
        rows.append(table(X, idx))
        labs.append(labels(y, half, idx))
    return model(seed).fit(pd.concat(rows, ignore_index=True), np.concatenate(labs))


def frames(clf, X):
    """(n, 2) per-frame probabilities of a handover into H / A at candidate frames, 0 elsewhere."""
    idx = candidates(X)
    out = np.zeros((len(X), 2), np.float32)
    if len(idx):
        p = clf.predict_proba(table(X, idx))
        cls = list(clf.classes_)
        for j, k in enumerate((1, 2)):
            if k in cls:
                out[idx, j] = p[:, cls.index(k)]
    return out


# -- the Y13 gate --------------------------------------------------------------------------------------

def swallowed(y, pred):
    """(start, stop) of true H/A spells production covers on less than half their frames."""
    b = np.r_[0, np.flatnonzero(np.diff(y) != 0) + 1, len(y)]
    return [(a, e) for a, e in zip(b[:-1], b[1:]) if y[a] < 2 and (pred[a:e] == y[a]).mean() < 0.5]


def hits(score, theta, truth, sw):
    """(false peaks, swallowed spells with a peak within RADIUS of both their start and their stop)."""
    from .cuts import peaks
    pk = peaks(score, theta, 2 * RADIUS + 2)
    if not len(pk):
        return 0, 0
    false = int((np.abs(truth[:, None] - pk[None, :]).min(0) > 2 * RADIUS + 2).sum())
    near = lambda c: np.abs(pk - c).min() <= RADIUS
    return false, sum(near(a) and near(e) for a, e in sw)


def gate():
    from . import nested
    from .external import MATCHES
    from .train import STAGE1_VERSION, SEED, copies
    from .data import cache_dir
    from .boundary import context, target

    data = {m: copies(m) for m in MATCHES}
    y = {m: data[m][1][1] for m in MATCHES}
    half = {m: data[m][1][2] for m in MATCHES}
    raw = {m: data[m][1][0][nested.RAW] for m in MATCHES}
    oof = lambda k, ex: np.load(cache_dir() / f"oof_{STAGE1_VERSION}_{k if ex == {k} else k + '_x' + ''.join(sorted(ex - {k}))}.npy")
    ctx = {m: {**{j: context(oof(j, {m, j}), raw[j]) for j in MATCHES if j != m}, m: context(oof(m, {m}), raw[m])}
           for m in MATCHES}
    tgt = {m: target(y[m], half[m], **nested.DETECTOR_CFG) for m in MATCHES}
    det = Parallel(n_jobs=len(MATCHES))(delayed(nested.fit_detector)(ctx[m], tgt, [j for j in MATCHES if j != m], m)
                                         for m in MATCHES)
    train_sets = {m: [(X, lab, half[j]) for j in MATCHES if j != m for X, lab in data[j][0]] for m in MATCHES}
    clfs = Parallel(n_jobs=len(MATCHES))(delayed(fit)(train_sets[m], SEED) for m in MATCHES)
    ev = {m: frames(c, data[m][1][0]) for m, c in zip(MATCHES, clfs)}
    det = dict(zip(MATCHES, det))

    bundle = np.load(nested.MODELS / f"nested_{nested.NAME}_predictions.npz")
    sw = {m: swallowed(y[m], bundle[f"{m}__y_pred"].astype(int)) for m in MATCHES}
    truth = {m: handovers(y[m], half[m]) for m in MATCHES}
    n_sw = sum(len(v) for v in sw.values())

    def pooled(scores, theta):
        r = [hits(scores[m], theta, truth[m], sw[m]) for m in MATCHES]
        return sum(a for a, _ in r), sum(b for _, b in r)

    evs = {m: ev[m].sum(1) for m in MATCHES}
    grid = np.round(np.arange(0.05, 0.96, 0.01), 2)
    curve = [(t, *pooled(evs, t)) for t in grid]
    out = {"swallowed": n_sw, "levels": []}
    print(f"swallowed spells on held-out matches: {n_sw}")
    print("detector theta  false peaks  cut out (both edges) | event theta  false peaks  cut out")
    for theta in (0.3, 0.4, 0.5):
        f_d, b_d = pooled(det, theta)
        ok = [(t, f, b) for t, f, b in curve if f <= f_d]
        t_e, f_e, b_e = max(ok, key=lambda r: (r[2], -r[1])) if ok else (None, 0, 0)
        print(f"  {theta:.2f}          {f_d:6d}       {b_d:4d} ({b_d / n_sw:.2f})      |  {t_e}        {f_e:6d}     "
              f"{b_e:4d} ({b_e / n_sw:.2f})")
        out["levels"].append({"detector_theta": theta, "detector_false": f_d, "detector_cut": b_d,
                              "event_theta": t_e, "event_false": f_e, "event_cut": b_e})
    team = []
    for m in MATCHES:
        for a, _ in sw[m]:
            w = ev[m][max(a - RADIUS, 0):a + RADIUS + 1]
            if w.max() > 0:
                team.append(int(np.unravel_index(w.argmax(), w.shape)[1]) == int(y[m][a]))
    out["event_team_right"] = float(np.mean(team)) if team else None
    out["passed"] = all(l["event_cut"] > l["detector_cut"] for l in out["levels"])
    print(f"event model names the right team at {out['event_team_right']:.2f} of swallowed starts it scores")
    print("GATE " + ("PASSED" if out["passed"] else "FAILED") + " (event model cuts out more swallowed spells at all three levels)")
    (nested.MODELS / f"gate_y13_{nested.NAME}.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    gate()
