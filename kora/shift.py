"""Compare simulated and challenge tracking without challenge labels.

The separated comparison holds out a whole source match and a challenge half. Features are built
separately for challenge halves, so rolling windows and carried ball context cannot cross the split.
The random-frame comparison uses the same features and is optimistic when nearby frames correlate.

    python -m kora.shift
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from .degrade import degrade
from .external import MATCHES, load
from .features import build
from .target import load_target

warnings.filterwarnings("ignore", category=RuntimeWarning)


def build_halves(players, team, ball, half):
    edges = np.r_[0, np.flatnonzero(half[1:] != half[:-1]) + 1, len(half)]
    return pd.concat([build(players[a:b], team, ball[a:b], half[a:b])
                      for a, b in zip(edges[:-1], edges[1:])], ignore_index=True)


def grouped_split(source_match, target_half, held_match, held_half):
    """Indices into source rows followed by target rows, with both domains in each partition."""
    source_test = source_match == held_match
    target_test = target_half == held_half
    if not (source_test.any() and (~source_test).any() and target_test.any() and (~target_test).any()):
        raise ValueError("each partition must contain source and target frames")
    test_mask = np.concatenate([source_test, target_test])
    return np.flatnonzero(~test_mask), np.flatnonzero(test_mask)


def score_split(X, y, train, test, columns, name):
    clf = HistGradientBoostingClassifier(max_iter=200, random_state=0).fit(X[train], y[train])
    auc = roc_auc_score(y[test], clf.predict_proba(X[test])[:, 1])
    rng = np.random.default_rng(0)
    sampled = np.concatenate([
        rng.choice(test[y[test] == label], size=min(500, (y[test] == label).sum()), replace=False)
        for label in (0, 1)
    ])
    imp = permutation_importance(clf, X[sampled], y[sampled], n_repeats=2,
                                 random_state=0, scoring="roc_auc")
    order = np.argsort(-imp.importances_mean)[:10]
    print(f"{name}: AUC {auc:.3f}, train {len(train):,}, test {len(test):,}", flush=True)
    print("  held-out permutation AUC drop: " + ", ".join(
        f"{columns[j]} {imp.importances_mean[j]:.3f}" for j in order), flush=True)


def main(step=10):
    players, team, ball, section, graded = load_target()
    target_parts = [build(players[a:b], team, ball[a:b], section[a:b]).iloc[::step]
                    for a, b in graded]
    target_half = np.concatenate([np.full(len(part), i) for i, part in enumerate(target_parts)])
    target = pd.concat(target_parts, ignore_index=True)
    del players, ball

    rng = np.random.default_rng(7)
    source_parts, source_match = [], []
    for match in list(MATCHES)[:3]:
        data = load(match)
        p, b = degrade(data["players"], data["ball"], data["label"], rng)
        features = build_halves(p, data["team"], b, data["half"]).iloc[::step * 2]
        source_parts.append(features)
        source_match.extend([match] * len(features))
        print(f"source {match}: {len(features):,} frames", flush=True)
    source = pd.concat(source_parts, ignore_index=True)
    X = pd.concat([source, target], ignore_index=True).to_numpy()
    y = np.r_[np.zeros(len(source), dtype=np.int8), np.ones(len(target), dtype=np.int8)]
    source_match = np.asarray(source_match)
    columns = source.columns
    del source_parts, target_parts, source, target

    train, test = train_test_split(np.arange(len(y)), test_size=0.3, random_state=0, stratify=y)
    score_split(X, y, train, test, columns, "random frames")
    for match in np.unique(source_match):
        for half in np.unique(target_half):
            train, test = grouped_split(source_match, target_half, match, half)
            score_split(X, y, train, test, columns, f"held out {match}, challenge half {half + 1}")


if __name__ == "__main__":
    main()
