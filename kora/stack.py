"""Stage 2 (auto-context / stacked sequential learning) on top of the frame classifier.

Stage 1 sees one frame plus hand-built context. Stage 2 sees stage 1's own probabilities over many
windows, past and future, and learns what a real possession looks like in that signal (a dip that is
too short to be a change of possession, a slow fade into a stoppage). It is trained only on the
stage-1 out-of-fold probabilities of the training matches, so it never sees probabilities that were
fitted to the same frames, and it is kept only if leave-one-match-out spellF1 improves.

    python -m kora.stack        # after kora.train; writes models/stage2.joblib if it helps
"""
import itertools
import json
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from .decode import decode
from .external import MATCHES
from .features import nanroll
from .train import MODELS, SEED, copies, segments_of, spell_f1

warnings.filterwarnings("ignore", category=RuntimeWarning)
RAW = ["b_seen", "t_since", "t_until", "sp_med_c101", "b_out", "d_diff_c25"]
GRID_HA = (2, 4, 8, 16, 32, 64, 128)
GRID_D = (2, 4, 8, 16, 32, 64, 128)


def context(proba, raw):
    """Stage-2 features from stage-1 probabilities (columns H, A, D) and a few raw signals."""
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
    return pd.DataFrame({k: np.asarray(v, np.float32) for k, v in f.items()})


def mirrored(proba, raw, y):
    """Swap the teams: H and A probabilities exchange, team-signed raw signals flip."""
    p = proba[:, [1, 0, 2]]
    r = raw.copy()
    r["d_diff_c25"] = -r["d_diff_c25"]
    yy = y.copy(); yy[yy < 2] = 1 - yy[yy < 2]
    return p, r, yy


def model():
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                          min_samples_leaf=400, l2_regularization=1.0, early_stopping=True,
                                          validation_fraction=0.1, n_iter_no_change=25, random_state=SEED)


def main():
    t0 = time.time()
    oof = np.load(MODELS / "oof.npz")
    data = {}
    for m in MATCHES:
        _, (Xe, ye, he) = copies(m)
        data[m] = (oof[m], Xe[RAW], ye, he)
        print(f"{m}: eval copy rebuilt ({time.time() - t0:.0f} s)", flush=True)

    def xy(ms, step=2):
        X, Y = [], []
        for m in ms:
            p, r, y, _ = data[m]
            for pp, rr, yy in ((p, r, y), mirrored(p, r, y)):
                X.append(context(pp, rr).to_numpy()[::step]); Y.append(yy[::step])
        return np.concatenate(X), np.concatenate(Y)

    oof2 = {}
    for m in MATCHES:
        X, y = xy([k for k in MATCHES if k != m])
        clf = model().fit(X, y)
        p, r, ye, _ = data[m]
        oof2[m] = clf.predict_proba(context(p, r).to_numpy())
        print(f"stage-2 fold {m}: frame acc {(oof2[m].argmax(1) == ye).mean():.3f} vs stage 1 {(p.argmax(1) == ye).mean():.3f}", flush=True)

    cfg = json.loads((MODELS / "decoder.json").read_text())
    prior = np.array(cfg["prior"])
    best = None
    for alpha, cha, cd in itertools.product((0.0, 1.0), GRID_HA, GRID_D):
        s = [spell_f1(data[m][2], decode(oof2[m], segments_of(data[m][3]), prior, alpha, cha, cd), data[m][3])[0]
             for m in MATCHES]
        if best is None or np.mean(s) > best[0]:
            best = (float(np.mean(s)), float(np.std(s)), alpha, cha, cd, s)
    print(f"\nstage 1 + decoder: spellF1 {cfg['cv_spell_f1_mean']:.4f} +- {cfg['cv_spell_f1_std']:.4f}")
    print(f"stage 2 + decoder: spellF1 {best[0]:.4f} +- {best[1]:.4f}  (alpha {best[2]}, cost_HA {best[3]}, cost_D {best[4]})")
    gain = [b - a for a, b in zip(cfg["cv_spell_f1_per_match"].values(), best[5])]
    print("per-match gain:", " ".join(f"{g:+.3f}" for g in gain))
    if best[0] > cfg["cv_spell_f1_mean"] and sum(g > 0 for g in gain) >= 5:
        X, y = xy(list(MATCHES))
        joblib.dump({"model": model().fit(X, y), "raw": RAW}, MODELS / "stage2.joblib", compress=3)
        cfg["stage2"] = {"alpha": best[2], "cost_ha": best[3], "cost_d": best[4], "cv_spell_f1_mean": best[0],
                         "cv_spell_f1_std": best[1], "cv_spell_f1_per_match": dict(zip(MATCHES, best[5]))}
        (MODELS / "decoder.json").write_text(json.dumps(cfg, indent=2))
        print("stage 2 kept (better on average and on at least 5 of 7 matches)")
    else:
        (MODELS / "stage2.joblib").unlink(missing_ok=True)
        cfg.pop("stage2", None)
        (MODELS / "decoder.json").write_text(json.dumps(cfg, indent=2))
        print("stage 2 dropped")
    print(f"({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
