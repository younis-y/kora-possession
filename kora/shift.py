"""How different do the degraded training matches look from the challenge match? (no labels used)

Adversarial validation: train a classifier to tell training frames from challenge frames using the
model's own features. AUC near 0.5 = indistinguishable. The features it leans on most are where the
degradation does not yet resemble the challenge tracking.

    python -m kora.shift
"""
import warnings

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score

from .degrade import degrade
from .external import MATCHES, load
from .features import build
from .target import load_target

warnings.filterwarnings("ignore", category=RuntimeWarning)


def main(step=10):
    players, team, ball, section, graded = load_target()
    Xt = build(players, team, ball, section)
    g = np.zeros(len(Xt), bool)
    for a, b in graded:
        g[a:b] = True
    Xt = Xt[g].iloc[::step]
    rng = np.random.default_rng(7)
    parts = []
    for m in list(MATCHES)[:3]:
        d = load(m)
        p, b = degrade(d["players"], d["ball"], d["label"], rng)
        parts.append(build(p, d["team"], b, d["half"]).iloc[::step * 2])
    import pandas as pd
    Xs = pd.concat(parts)
    X = pd.concat([Xs, Xt]).to_numpy()
    y = np.r_[np.zeros(len(Xs)), np.ones(len(Xt))]
    idx = np.random.default_rng(0).permutation(len(y))
    X, y = X[idx], y[idx]
    k = int(0.7 * len(y))
    clf = HistGradientBoostingClassifier(max_iter=200, random_state=0).fit(X[:k], y[:k])
    auc = roc_auc_score(y[k:], clf.predict_proba(X[k:])[:, 1])
    print(f"domain AUC {auc:.3f} (0.5 = training copies look like the challenge tracking)")
    imp = permutation_importance(clf, X[k:k + 4000], y[k:k + 4000], n_repeats=3, random_state=0, scoring="roc_auc")
    order = np.argsort(-imp.importances_mean)[:15]
    cols = list(Xs.columns)
    print("features that give the domain away:")
    for j in order:
        c = cols[j]
        print(f"  {c:<18} drop {imp.importances_mean[j]:.3f}   train median {np.nanmedian(Xs[c]):8.3f}   challenge median {np.nanmedian(Xt[c]):8.3f}")


if __name__ == "__main__":
    main()
