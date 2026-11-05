"""Train the frame classifier on the open Bundesliga matches and choose the decoder by
leave-one-match-out cross-validation, scored with the challenge's own spellF1 code.

    python -m kora.train            # writes models/classifier.joblib and models/decoder.json

No data about the challenge match is labelled, scored or measured here. The degradation takes every
number from external data (kora/missingness.py -> models/degrade_external.json).
"""
import itertools
import json
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from .data import cache_dir
from .decode import decode
from .degrade import degrade, random_ball
from .external import MATCHES, load
from .features import build, mirror
from .variant import CONFIG, feature_version, stage1_version
from .vendor import possession_metrics as pm

warnings.filterwarnings("ignore", category=RuntimeWarning)
ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
SEED = 2027
TRAIN_COPIES = 2            # degraded copies per match, each also mirrored
SUBSAMPLE = 6               # every k-th frame of a training copy goes to the classifier (neighbours are near-duplicates)
GRID_HA = (2, 4, 8, 16, 32, 64, 128)
GRID_D = (2, 4, 8, 16, 32, 64, 128)
GRID_ALPHA = (0.0, 1.0)


def classifier():
    return HistGradientBoostingClassifier(
        max_iter=250, learning_rate=0.1, max_leaf_nodes=63, min_samples_leaf=200,
        l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
        n_iter_no_change=25, random_state=SEED)


def segments_of(half):
    edges = np.flatnonzero(np.diff(half) != 0) + 1
    starts = np.r_[0, edges]
    stops = np.r_[edges, len(half)]
    return list(zip(starts.tolist(), stops.tolist()))


FEATURE_VERSION = feature_version()      # hash of features/degradation code + external params
STAGE1_VERSION = stage1_version()        # + training code and switches


def copies(match):
    """Training copies (degraded, plus mirrored) and one evaluation copy, cached in data/cache."""
    path = cache_dir() / f"feat_{FEATURE_VERSION}_{match}.npz"
    if path.exists():
        z = np.load(path, allow_pickle=False)
        cols = list(z["columns"])
        train = [(pd.DataFrame(z[f"X{i}"], columns=cols), z[f"y{i}"]) for i in range(2 * TRAIN_COPIES)]
        return train, (pd.DataFrame(z["Xe"], columns=cols), z["ye"], z["he"])
    train, ev = _copies(match)
    arrays = {f"X{i}": x.to_numpy() for i, (x, _) in enumerate(train)}
    arrays.update({f"y{i}": y for i, (_, y) in enumerate(train)})
    np.savez(path, columns=np.array(train[0][0].columns, dtype="U64"), Xe=ev[0].to_numpy(), ye=ev[1], he=ev[2], **arrays)
    return train, ev


def _copies(match):
    if CONFIG.get("eval_stress") and CONFIG.get("eval_rand"):
        raise ValueError("eval_stress and eval_rand are mutually exclusive")
    d = load(match)
    rng = np.random.default_rng([SEED, list(MATCHES).index(match)])
    train = []
    for i in range(TRAIN_COPIES):
        p, b = degrade(d["players"], d["ball"], d["label"], rng)
        if CONFIG.get("mixed") and i % 2:       # own rng, so the b0 draws stay identical
            b = random_ball(d["ball"], d["half"], np.random.default_rng([SEED, list(MATCHES).index(match), i]))
        train.append((build(p, d["team"], b, d["half"]), d["label"]))
        pm_, tm_, bm_ = mirror(p, d["team"], b)
        y = d["label"].copy(); y[y < 2] = 1 - y[y < 2]
        train.append((build(pm_, tm_, bm_, d["half"]), y))
    if CONFIG.get("eval_stress"):
        from .degrade import RANDOM_GAPS
        p, b = degrade(d["players"], d["ball"], d["label"], rng, severity=1.0, params=RANDOM_GAPS)
    else:
        p, b = degrade(d["players"], d["ball"], d["label"], rng)
    if CONFIG.get("eval_rand"):
        b = random_ball(d["ball"], d["half"], np.random.default_rng([SEED, list(MATCHES).index(match), TRAIN_COPIES]))
    evaluation = (build(p, d["team"], b, d["half"]), d["label"], d["half"])
    return train, evaluation


def stack(train_sets):
    X = np.concatenate([x.to_numpy()[::SUBSAMPLE] for x, _ in train_sets])
    y = np.concatenate([lab[::SUBSAMPLE] for _, lab in train_sets])
    return X, y


def balance_weights(label, lam):
    """Per-frame weight (1 - lam) + lam * mean_len / len(own possession), normalised to mean 1.

    spellF1 counts every possession once, however long; lam > 0 moves the training loss toward that.
    Dead-ball frames keep weight 1 before normalisation."""
    edges = np.flatnonzero(np.diff(label) != 0) + 1
    starts, stops = np.r_[0, edges], np.r_[edges, len(label)]
    lengths = np.repeat(stops - starts, stops - starts).astype(np.float64)
    live = label < 2
    w = np.ones(len(label))
    if live.any():
        w[live] = (1 - lam) + lam * lengths[live].mean() / lengths[live]
    return w / w.mean()


def fit_classifier(train_sets):
    """Stage-1 classifier on stacked training copies, with balanced weights when the variant asks."""
    X, y = stack(train_sets)
    lam = CONFIG.get("balance")
    if not lam:
        return classifier().fit(X, y)
    w = np.concatenate([balance_weights(lab, lam)[::SUBSAMPLE] for _, lab in train_sets])
    return classifier().fit(X, y, sample_weight=w)


def spell_f1(label, pred, half):
    """Official spellF1 with an X frame inserted between the halves, so they are scored apart."""
    cut = np.flatnonzero(np.diff(half) != 0) + 1
    gt = np.insert(label.astype(np.int64), cut, pm.X)
    pr = np.insert(pred.astype(np.int64), cut, pm.X)
    runs = pm.valid_runs(gt)
    return pm.spell_scores(gt, pr, runs)["mean_f1"], pm.macro_f1(pm.scored_confusion(pm.confusion(gt, pr)))


def main():
    t0 = time.time()

    data = {}
    for m in MATCHES:
        data[m] = copies(m)
        print(f"{m}: features ready ({time.time() - t0:.0f} s)", flush=True)
    columns = list(data[next(iter(MATCHES))][1][0].columns)

    # leave-one-match-out: out-of-fold probabilities on each held-out match
    oof = {}
    for m in MATCHES:
        Xe, ye, he = data[m][1]
        fold_cache = cache_dir() / f"oof_{STAGE1_VERSION}_{m}.npy"      # resumable: finished folds are kept
        if fold_cache.exists():
            oof[m] = np.load(fold_cache)
            print(f"fold {m}: cached", flush=True)
            continue
        clf = fit_classifier([s for k in MATCHES if k != m for s in data[k][0]])
        oof[m] = clf.predict_proba(Xe.to_numpy())
        np.save(fold_cache, oof[m])
        acc = (oof[m].argmax(1) == ye).mean()
        print(f"fold {m}: {clf.n_iter_} trees, frame accuracy {acc:.3f} ({time.time() - t0:.0f} s)", flush=True)

    X, y = stack([s for k in MATCHES for s in data[k][0]])
    prior = np.bincount(y, minlength=3) / len(y)

    # decoder: pick switching costs and prior correction by mean held-out spellF1
    results = []
    for alpha, cha, cd in itertools.product(GRID_ALPHA, GRID_HA, GRID_D):
        f1s, mfs = [], []
        for m in MATCHES:
            _, ye, he = data[m][1]
            pred = decode(oof[m], segments_of(he), prior, alpha, cha, cd)
            s, mf = spell_f1(ye, pred, he)
            f1s.append(s); mfs.append(mf)
        results.append((float(np.mean(f1s)), float(np.std(f1s)), float(np.mean(mfs)), alpha, cha, cd, f1s))
    results.sort(key=lambda r: -r[0])
    print("\nbest decoders (mean held-out spellF1 over 7 matches):")
    for r in results[:8]:
        print(f"  spellF1 {r[0]:.4f} +- {r[1]:.4f}  macroF1 {r[2]:.4f}  alpha {r[3]}  cost_HA {r[4]}  cost_D {r[5]}")
    raw = []
    for m in MATCHES:
        _, ye, he = data[m][1]
        raw.append(spell_f1(ye, oof[m].argmax(1), he)[0])
    print(f"  (per-frame argmax without decoding: spellF1 {np.mean(raw):.4f})")
    best = results[0]

    # final classifier on all seven matches
    clf = fit_classifier([s for k in MATCHES for s in data[k][0]])
    MODELS.mkdir(exist_ok=True)
    joblib.dump({"model": clf, "columns": columns}, MODELS / "classifier.joblib", compress=3)
    cfg = {"alpha": best[3], "cost_ha": best[4], "cost_d": best[5], "prior": prior.tolist(),
           "cv_spell_f1_mean": best[0], "cv_spell_f1_std": best[1], "cv_spell_f1_per_match": dict(zip(MATCHES, best[6])),
           "cv_macro_f1_mean": best[2], "train_copies": TRAIN_COPIES, "subsample": SUBSAMPLE, "seed": SEED,
           "feature_version": FEATURE_VERSION, "stage1_version": STAGE1_VERSION}
    (MODELS / "decoder.json").write_text(json.dumps(cfg, indent=2))
    np.savez_compressed(MODELS / "oof.npz", **{m: oof[m] for m in MATCHES})
    print(f"\nsaved models/ ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
