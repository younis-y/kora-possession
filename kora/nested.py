"""Clean held-out estimate (RESEARCH.md Y1): for each match m, every fitted stage and every choice
is made on the other six matches only, then scored once on m.

kora.train + kora.boundary leak a little: the detector for held-out m learns from stage-1 predictions
on the other matches, and those came from classifiers that had seen m; the prior, the detector rate
and the plain decoder settings were fit on all seven. Here, per outer match m:

1. stage 1: inner out-of-fold predictions on each k != m from a classifier trained on the five
   matches that are neither m nor k (42 fits, cached); predictions on m from the classifier trained
   on all six (the kora.train fold, reused from its cache).
2. prior, plain decoder grid, detector (inner leave-one-out for selection, all six for m), rate,
   duration prior and the HSMM grid, all on the six inner matches.
3. the HSMM is used on m only if it beats the plain decoder on the inner mean and on at least
   `inner_wins` of the 6 inner matches (default 5; grids and the rule come from kora.variant.decoder).

With decoder "refine" (Y7a) a stage-2 model re-reads windows of the stage-1 probabilities and its H/A/D
probabilities replace stage 1's in both decoders; it is fitted on the same folds as the detector.

    python -m kora.nested --variant b0     # a few hours on first run (stage-1 folds are cached)

Writes models/nested_<variant>.json (settings and scores per fold, committed) and
models/nested_<variant>_predictions.npz (schema-1 bundle for `kora.errors --predictions`, gitignored).
Compare two variants with `python -m kora.compare b0 <variant>`.
"""
import argparse
import json
import subprocess
import time
import warnings

import numpy as np
from joblib import Parallel, delayed

from .boundary import GRID, RAW, context, model as detector, runs_of, target
from . import dwell, fill, peaks
from .cuts import relabel
from . import train as _train
from .data import cache_dir
from .decode import decode, decode_hsmm, log_duration
from .external import MATCHES
from .train import (FEATURE_VERSION, GRID_ALPHA, GRID_D, GRID_HA, MODELS, STAGE1_VERSION, SUBSAMPLE,
                    TRAIN_COPIES, copies, fit_classifier, segments_of)
from .variant import CONFIG, NAME, VARIANTS, decoder, detector as detector_config
from .vendor import possession_metrics as pm

warnings.filterwarnings("ignore", category=RuntimeWarning)
DECODER = decoder()
DETECTOR_CFG = detector_config()
PLAIN = [tuple(s) for s in DECODER["plain_grid"] or
         [(a, h, d) for a in GRID_ALPHA for h in GRID_HA for d in GRID_D]]
HSMM = [tuple(s) for s in DECODER["hsmm_grid"] or GRID]
INNER_WINS = DECODER["inner_wins"]   # HSMM kept if it beats plain on the inner mean and on >= this many of 6


def scores(y, pred, half):
    """Official spell scores with an X frame between the halves, so they are scored apart."""
    cut = np.flatnonzero(np.diff(half) != 0) + 1
    gt, pr = np.insert(y.astype(np.int64), cut, pm.X), np.insert(pred.astype(np.int64), cut, pm.X)
    return pm.spell_scores(gt, pr, pm.valid_runs(gt))


def stage1(data, excluded, k, mirrored=None):
    """Stage-1 probabilities on k from a classifier trained without any match in `excluded` (k included).

    With `mirrored` (Y11) they are averaged with the same classifier's H/A-swapped probabilities on k's
    mirrored evaluation copy; those are cached beside the plain ones under the same stage-1 key."""
    name = k if excluded == {k} else f"{k}_x{''.join(sorted(excluded - {k}))}"
    path = cache_dir() / f"oof_{STAGE1_VERSION}_{name}.npy"           # {k} alone is kora.train's own fold
    mpath = path.with_name(f"{path.stem}_mirror.npy")
    if path.exists() and (mirrored is None or mpath.exists()):
        p = np.load(path)
    else:
        clf = fit_classifier([s for j in MATCHES if j not in excluded for s in data[j][0]])
        p = clf.predict_proba(data[k][1][0].to_numpy())
        if not path.exists():
            np.save(path, p)
        if mirrored is not None:
            np.save(mpath, clf.predict_proba(mirrored[k].to_numpy()))
    return p if mirrored is None else (p + np.load(mpath)[:, [1, 0, 2]]) / 2


def mirrored_eval(match, X):
    """The evaluation copy of `match` with the teams swapped (Y11), checked against its cached copy X.

    Replays kora.train._copies: the shared generator is consumed only by degrade(), once per training
    copy and then once for the evaluation copy."""
    d = _train.load(match)
    i = list(MATCHES).index(match)
    rng = np.random.default_rng([_train.SEED, i])
    for _ in range(TRAIN_COPIES):
        _train.degrade(d["players"], d["ball"], d["label"], rng)
    p, b = _train.degrade(d["players"], d["ball"], d["label"], rng)
    if CONFIG.get("eval_rand"):
        b = _train.random_ball(d["ball"], d["half"], np.random.default_rng([_train.SEED, i, TRAIN_COPIES]))
    if not np.array_equal(_train.build(p, d["team"], b, d["half"]).to_numpy(), X.to_numpy(), equal_nan=True):
        raise RuntimeError(f"{match}: replayed evaluation copy differs from the cache")
    return _train.build(*_train.mirror(p, d["team"], b), d["half"])


def fit_detector(ctx, tgt, train, test):
    X = np.concatenate([ctx[j][::2] for j in train])
    y = np.concatenate([tgt[j][::2] for j in train])
    return detector().fit(X, y).predict_proba(ctx[test])[:, 1]


def fit_refiner(ctx, labels, train, test):
    """Stage 2 (Y7a, the MS-TCN idea): H/A/D probabilities on test from the detector's context windows."""
    X = np.concatenate([ctx[j][::2] for j in train])
    y = np.concatenate([labels[j][::2] for j in train])
    return detector().fit(X, y).predict_proba(ctx[test])


def plain_scores(P, y, half, prior, setting):
    return [scores(y[k], decode(P[k], segments_of(half[k]), prior, *setting), half[k])["mean_f1"] for k in P]


def hsmm_paths(P, C, y, half, prior, rate, setting):
    """HSMM path on each inner match, its duration prior from the other inner matches."""
    out = {}
    for k in P:
        ld = log_duration([r for j in P if j != k for r in runs_of(y[j], half[j])])
        out[k] = decode_hsmm(P[k], C[k], segments_of(half[k]), prior, ld,
                             *setting[:4], rate, *setting[4:])
    return out


def hsmm_scores(P, C, y, half, prior, rate, setting):
    paths = hsmm_paths(P, C, y, half, prior, rate, setting)
    return [scores(y[k], paths[k], half[k])["mean_f1"] for k in P]


def cut_scores(paths, P, C, y, half, theta, gap):
    return [scores(y[k], relabel(paths[k], P[k], C[k], half[k], theta, gap), half[k])["mean_f1"] for k in paths]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", default=NAME, choices=VARIANTS, help="read at import by kora.variant")
    parser.parse_args()
    if CONFIG.get("seed"):             # read at call time by kora.train (copies, classifier)
        _train.SEED = CONFIG["seed"]
    if CONFIG.get("fill"):             # Y15: kora.train builds every copy through this (train.py unchanged, keys kept)
        _train.build = fill.wrap(_train.build)
    t0 = time.time()
    print(f"variant {NAME} {CONFIG}  features {FEATURE_VERSION}  stage 1 {STAGE1_VERSION}", flush=True)
    data = {m: copies(m) for m in MATCHES}
    mirrored = {m: mirrored_eval(m, data[m][1][0]) for m in MATCHES} if CONFIG.get("tta") else None
    if mirrored is not None:
        print(f"mirrored evaluation copies built and checked ({time.time() - t0:.0f} s)", flush=True)
    for key, extra in (("dwell", dwell), ("peaks", peaks)):           # Y8a, Y12: columns derived from the cache
        if CONFIG.get(key):
            data = {m: ([(extra.add(X), lab) for X, lab in tr], (extra.add(ev[0]), *ev[1:]))
                    for m, (tr, ev) in data.items()}
            mirrored = mirrored and {m: extra.add(X) for m, X in mirrored.items()}
    y = {m: data[m][1][1] for m in MATCHES}
    half = {m: data[m][1][2] for m in MATCHES}
    raw = {m: data[m][1][0][RAW] for m in MATCHES}
    seen = {m: data[m][1][0]["b_seen"].to_numpy().astype(np.int8) for m in MATCHES}
    labels = {m: np.concatenate([lab[::SUBSAMPLE] for _, lab in data[m][0]]) for m in MATCHES}
    tgt = {m: target(y[m], half[m], **DETECTOR_CFG) for m in MATCHES}

    # 1. stage-1 predictions: outer (kora.train folds) and inner (without m and k)
    outer = {m: stage1(data, {m}, m, mirrored) for m in MATCHES}
    inner = {}
    for m in MATCHES:
        for k in MATCHES:
            if k != m:
                inner[m, k] = stage1(data, {m, k}, k, mirrored)
                print(f"stage 1 without {m}, {k} ({time.time() - t0:.0f} s)", flush=True)
    del data, mirrored

    # 2. per outer match: detector folds on the inner matches, then the detector for m
    ctx = {(m, k): context(inner[m, k], raw[k]) for m, k in inner}
    ctx.update({(m, m): context(outer[m], raw[m]) for m in MATCHES})
    jobs = [(m, k) for m in MATCHES for k in MATCHES]                 # k == m is the outer detector
    preds = Parallel(n_jobs=len(MATCHES))(
        delayed(fit_detector)({j: ctx[m, j] for j in MATCHES if j != m or k == m},
                              tgt, [j for j in MATCHES if j not in (m, k)], k) for m, k in jobs)
    change = dict(zip(jobs, preds))
    print(f"detectors done ({time.time() - t0:.0f} s)", flush=True)
    if DECODER["refine"]:                                              # Y7a: same folds as the detector
        preds = Parallel(n_jobs=len(MATCHES))(
            delayed(fit_refiner)({j: ctx[m, j] for j in MATCHES if j != m or k == m},
                                 y, [j for j in MATCHES if j not in (m, k)], k) for m, k in jobs)
        refined = dict(zip(jobs, preds))
        inner = {(m, k): refined[m, k] for m, k in inner}
        outer = {m: refined[m, m] for m in MATCHES}
        print(f"refiners done ({time.time() - t0:.0f} s)", flush=True)

    folds, bundle = {}, {}
    for m in MATCHES:
        ins = [k for k in MATCHES if k != m]
        P = {k: inner[m, k] for k in ins}
        C = {k: change[m, k] for k in ins}
        yi, hi = {k: y[k] for k in ins}, {k: half[k] for k in ins}
        prior = np.bincount(np.concatenate([labels[k] for k in ins]), minlength=3) / sum(len(labels[k]) for k in ins)
        rate = float(np.mean([(tgt[k] == 1).mean() for k in ins]))

        plain = np.array(Parallel(n_jobs=-1)(delayed(plain_scores)(P, yi, hi, prior, s) for s in PLAIN))
        bp = int(plain.mean(1).argmax())
        hs = np.array(Parallel(n_jobs=-1)(delayed(hsmm_scores)(P, C, yi, hi, prior, rate, g) for g in HSMM))
        bh = int(hs.mean(1).argmax())
        keep = bool(hs[bh].mean() > plain[bp].mean() and (hs[bh] > plain[bp]).sum() >= INNER_WINS)

        ld = log_duration([r for k in ins for r in runs_of(y[k], half[k])])
        seg = segments_of(half[m])
        pred_plain = decode(outer[m], seg, prior, *PLAIN[bp])
        pred_hsmm = decode_hsmm(outer[m], change[m, m], seg, prior, ld, *HSMM[bh][:4], rate, *HSMM[bh][4:])
        pred = pred_hsmm if keep else pred_plain
        s_nocut, cut = scores(y[m], pred, half[m]), {"theta": None, "gap": None}
        if DECODER["cuts"]:                                            # Y10: cut at detector peaks (kora/cuts.py)
            paths = (hsmm_paths(P, C, yi, hi, prior, rate, HSMM[bh]) if keep else
                     {k: decode(P[k], segments_of(half[k]), prior, *PLAIN[bp]) for k in ins})
            off = np.array([scores(y[k], paths[k], half[k])["mean_f1"] for k in ins])
            grid = [(t, g) for t in DECODER["cuts"]["theta"] for g in DECODER["cuts"]["gap"]]
            cs = np.array(Parallel(n_jobs=-1)(delayed(cut_scores)(paths, P, C, yi, hi, t, g) for t, g in grid))
            bc = int(cs.mean(1).argmax())
            cut.update(inner_mean=float(cs[bc].mean()), inner_mean_off=float(off.mean()))
            if cs[bc].mean() > off.mean() and (cs[bc] > off).sum() >= INNER_WINS:   # same rule as the HSMM
                cut.update(theta=grid[bc][0], gap=grid[bc][1])
                pred = relabel(pred, outer[m], change[m, m], half[m], *grid[bc])
        s, s_plain, s_hsmm = scores(y[m], pred, half[m]), scores(y[m], pred_plain, half[m]), scores(y[m], pred_hsmm, half[m])
        folds[m] = {
            "inner_matches": ins, "decoder": "hsmm" if keep else "plain",
            "plain": dict(zip(("alpha", "cost_ha", "cost_d"), PLAIN[bp]), inner_mean=float(plain[bp].mean())),
            "hsmm": dict(zip(("cost_ha", "cost_d", "w_dur", "beta", "ha_bonus_only"),
                             (HSMM[bh] + (False,))[:5]), inner_mean=float(hs[bh].mean()), rate=rate),
            "prior": prior.tolist(),
            "spell_f1": s["mean_f1"], "spell_f1_per_iou": {str(t): v["f1"] for t, v in s["per_iou"].items()},
            "n_true": s["n_gt"], "n_pred": s["n_pred"],
            "spell_f1_plain": s_plain["mean_f1"], "spell_f1_hsmm": s_hsmm["mean_f1"],
            "spell_f1_nocut": s_nocut["mean_f1"], "cuts": cut,
        }
        bundle.update({f"{m}__y_true": y[m].astype(np.int8), f"{m}__y_pred": pred.astype(np.int8),
                       f"{m}__half": half[m], f"{m}__ball_seen": seen[m], f"{m}__frame": np.arange(len(pred))})
        print(f"{m}: {folds[m]['decoder']:5s} spellF1 {s['mean_f1']:.4f}  (plain {s_plain['mean_f1']:.4f}, "
              f"hsmm {s_hsmm['mean_f1']:.4f})  ({time.time() - t0:.0f} s)", flush=True)

    f1 = np.array([folds[m]["spell_f1"] for m in MATCHES])
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "kora"], capture_output=True, text=True).stdout.strip())
    meta = {"schema": 1, "fps": 25, "matches": list(MATCHES),
            "provenance": "kora.nested: each held-out match excluded from stage 1, the detector, the prior, "
                          "the rate, the duration prior and decoder selection; scored once. "
                          "Ball and player gaps are simulated from external measurements (kora/degrade.py), "
                          "not challenge tracking.",
            "commit": commit, "dirty": dirty, "seed": _train.SEED, "variant": NAME, "config": CONFIG,
            "feature_version": FEATURE_VERSION, "stage1_version": STAGE1_VERSION, "inner_wins": INNER_WINS, "decoder": DECODER,
            "detector": DETECTOR_CFG,
            "train_copies": TRAIN_COPIES, "subsample": SUBSAMPLE}
    summary = {**meta, "spell_f1_mean": float(f1.mean()), "spell_f1_std": float(f1.std()),
               "spell_f1_plain_mean": float(np.mean([folds[m]["spell_f1_plain"] for m in MATCHES])),
               "spell_f1_hsmm_mean": float(np.mean([folds[m]["spell_f1_hsmm"] for m in MATCHES])),
               "spell_f1_nocut_mean": float(np.mean([folds[m]["spell_f1_nocut"] for m in MATCHES])),
               "folds": folds}
    (MODELS / f"nested_{NAME}.json").write_text(json.dumps(summary, indent=2))
    np.savez_compressed(MODELS / f"nested_{NAME}_predictions.npz", meta=json.dumps(meta), **bundle)
    print(f"\nclean spellF1 {f1.mean():.4f} +- {f1.std():.4f}  (plain {summary['spell_f1_plain_mean']:.4f}, "
          f"hsmm {summary['spell_f1_hsmm_mean']:.4f})  per match: " + " ".join(f"{v:.3f}" for v in f1))
    print(f"({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
