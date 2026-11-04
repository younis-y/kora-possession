"""Change-of-possession detector plus a duration-aware decoder.

The frame classifier recognises most short possessions, but a plain HMM decoder pays one flat cost
per switch and so smooths 1-4 s possessions away; they are a third of all possessions. Two fixes,
both trained only on the stage-1 out-of-fold probabilities of the training matches:

1. A detector for "the team in possession changes within 0.2 s of this frame", from windows of the
   stage-1 probabilities and a few raw signals. Only its H<->A output is used: alive/dead edges in
   the training copies line up with the simulated ball blackouts, so that output is not trusted.
2. A semi-Markov (segment) decoder that scores each possession by its summed log-probability, a
   learned duration prior and a bonus where the detector fires (see kora.decode.hsmm).

Kept only if leave-one-match-out spellF1 beats the plain decoder on average and on at least 5 of 7
matches. A nested estimate (settings chosen without the held-out match) is reported as well.

    python -m kora.boundary     # after kora.train; writes models/boundary.joblib and cfg["hsmm"]
"""
import itertools
import json
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingClassifier

from .decode import decode_hsmm, log_duration
from .external import MATCHES
from .features import nanroll
from .train import MODELS, SEED, copies, segments_of, spell_f1
from .variant import detector

warnings.filterwarnings("ignore", category=RuntimeWarning)
RAW = ["b_seen", "t_since", "t_until", "sp_med_c101", "b_out", "d_diff_c25", "b_turn", "b_dspeed", "b_speed",
       "d_diff", "d_min", "d_diff_j25", "last_close", "next_close", "t_last_close", "t_next_close",
       "b_speed_j25", "sp_med_j25"]
RADIUS = 5                                  # frames either side of a change that count as "near"
GRID = list(itertools.product((2, 4, 8), (8, 16), (2, 3, 4), (2, 3, 4)))   # cost_ha, cost_d, w_dur, beta


def context(proba, raw):
    """Detector features from stage-1 probabilities (columns H, A, D) and raw signals."""
    f = {}
    team = proba[:, 0] - proba[:, 1]
    dead = proba[:, 2]
    f["team"], f["dead"] = team, dead
    for w in (13, 25, 51, 101, 201, 401, 801):
        f[f"team_c{w}"] = nanroll(team, w, "c")
        f[f"dead_c{w}"] = nanroll(dead, w, "c")
    for w in (25, 75, 250):
        for k, v in (("team", team), ("dead", dead)):
            p, q = nanroll(v, w, "p"), nanroll(v, w, "f")
            f[f"{k}_p{w}"], f[f"{k}_f{w}"], f[f"{k}_j{w}"] = p, q, q - p
    for k in RAW:
        f[k] = raw[k].to_numpy()
    return pd.DataFrame({k: np.asarray(v, np.float32) for k, v in f.items()}).to_numpy()


def target(y, half, ha_priority=False):
    """0 no nearby change, 1 H<->A within RADIUS frames, 2 alive<->dead.

    Class 2 wins overlaps unless ha_priority gives class 1 priority (V7).
    Cross-half changes are excluded; windows are not clipped at half boundaries.
    """
    t = np.zeros(len(y), np.int8)
    handover = np.zeros(len(y), bool) if ha_priority else None
    for c in np.flatnonzero((y[1:] != y[:-1]) & (half[1:] == half[:-1])) + 1:
        k = 2 if 2 in (y[c], y[c - 1]) else 1
        t[max(c - RADIUS, 0):c + RADIUS + 1] = np.maximum(t[max(c - RADIUS, 0):c + RADIUS + 1], k)
        if ha_priority and k == 1:
            handover[max(c - RADIUS, 0):c + RADIUS + 1] = True
    if ha_priority:
        t[handover] = 1
    return t


def model():
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, max_leaf_nodes=31, min_samples_leaf=400,
                                          l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
                                          n_iter_no_change=25, random_state=SEED)


def runs_of(y, half):
    """(label, length in frames) of every maximal run, halves kept apart."""
    ch = np.flatnonzero((y[1:] != y[:-1]) | (half[1:] != half[:-1])) + 1
    st, en = np.r_[0, ch], np.r_[ch, len(y)]
    return [(int(y[a]), int(b - a)) for a, b in zip(st, en)]


def main():
    t0 = time.time()
    detector_cfg = detector()
    oof = np.load(MODELS / "oof.npz")
    cfg = json.loads((MODELS / "decoder.json").read_text())
    prior = np.array(cfg["prior"])
    data = {}
    for m in MATCHES:
        _, (Xe, ye, he) = copies(m)
        data[m] = (oof[m], context(oof[m], Xe[RAW]), ye, he)

    def xy(ms):
        return (np.concatenate([data[k][1][::2] for k in ms]),
                np.concatenate([target(data[k][2], data[k][3], **detector_cfg)[::2] for k in ms]))

    def fold(m):
        X, y = xy([k for k in MATCHES if k != m])
        return model().fit(X, y).predict_proba(data[m][1])[:, 1]

    change = dict(zip(MATCHES, Parallel(n_jobs=len(MATCHES))(delayed(fold)(m) for m in MATCHES)))
    rate = float(np.mean([(target(data[m][2], data[m][3], **detector_cfg) == 1).mean() for m in MATCHES]))
    np.savez_compressed(MODELS / "change_oof.npz", **change)          # for kora.errors; not committed
    print(f"detector folds done ({time.time() - t0:.0f} s)", flush=True)

    def score(m, cha, cd, w, beta):
        p, _, ye, he = data[m]
        ld = log_duration([r for k in MATCHES if k != m for r in runs_of(data[k][2], data[k][3])])
        return spell_f1(ye, decode_hsmm(p, change[m], segments_of(he), prior, ld, cha, cd, w, beta, rate), he)[0]

    res = np.array(Parallel(n_jobs=-1)(delayed(score)(m, *g) for g in GRID for m in MATCHES)).reshape(len(GRID), -1)
    best = int(res.mean(1).argmax())
    nested = [res[int(np.delete(res, i, 1).mean(1).argmax()), i] for i in range(len(MATCHES))]
    base = list(cfg["cv_spell_f1_per_match"].values())
    gain = res[best] - np.array(base)
    print(f"plain decoder:         spellF1 {np.mean(base):.4f} +- {np.std(base):.4f}")
    print(f"detector + durations:  spellF1 {res[best].mean():.4f} +- {res[best].std():.4f}  "
          f"(cost_HA, cost_D, w_dur, beta) = {GRID[best]}")
    print(f"  nested (settings chosen without the held-out match): {np.mean(nested):.4f}")
    print("  per-match gain:", " ".join(f"{g:+.3f}" for g in gain))

    cfg.pop("hsmm", None)
    (MODELS / "boundary.joblib").unlink(missing_ok=True)
    if res[best].mean() > np.mean(base) and (gain > 0).sum() >= 5:
        X, y = xy(list(MATCHES))
        joblib.dump({"model": model().fit(X, y), "raw": RAW}, MODELS / "boundary.joblib", compress=3)
        ld = log_duration([r for k in MATCHES for r in runs_of(data[k][2], data[k][3])])
        cha, cd, w, beta = GRID[best]
        cfg["hsmm"] = {"cost_ha": cha, "cost_d": cd, "w_dur": w, "beta": beta, "rate": rate, "log_duration": ld.tolist(),
                       "cv_spell_f1_mean": float(res[best].mean()), "cv_spell_f1_std": float(res[best].std()),
                       "cv_spell_f1_nested": float(np.mean(nested)),
                       "cv_spell_f1_per_match": dict(zip(MATCHES, res[best].tolist()))}
        print("detector + duration decoder kept")
    else:
        print("detector + duration decoder dropped")
    (MODELS / "decoder.json").write_text(json.dumps(cfg, indent=2))
    print(f"({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
