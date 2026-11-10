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
    python -m kora.nested --variant y20_lgbm --screen   # Y20: outer folds only, vs the base's cached ones

Writes models/nested_<variant>.json (settings and scores per fold, committed) and
models/nested_<variant>_predictions.npz (schema-1 bundle for `kora.errors --predictions`, gitignored).
Compare two variants with `python -m kora.compare b0 <variant>`.
"""
import argparse
import itertools
import json
import subprocess
import time
import warnings
from contextlib import contextmanager

import numpy as np
from joblib import Parallel, delayed

from .boundary import GRID, RAW, context, model as detector, runs_of, target
from . import dwell, engine, events, fill, grf, mix, peaks, seq, visdecode
from .cuts import relabel
from .snap import snap
from . import train as _train
from .data import cache_dir
from .decode import decode, decode_hsmm, log_duration
from .external import MATCHES
from .train import (FEATURE_VERSION, GRID_ALPHA, GRID_D, GRID_HA, MODELS, STAGE1_VERSION, SUBSAMPLE,
                    TRAIN_COPIES, copies, fit_classifier, segments_of)
from .variant import (CONFIG, NAME, VARIANTS, _hash, decoder, detector as detector_config, feature_version,
                      stage1_version)
from .vendor import possession_metrics as pm

warnings.filterwarnings("ignore", category=RuntimeWarning)
DECODER = decoder()
DETECTOR_CFG = detector_config()
PLAIN = [tuple(s) for s in DECODER["plain_grid"] or
         [(a, h, d) for a in GRID_ALPHA for h in GRID_HA for d in GRID_D]]
HSMM = [tuple(s) for s in DECODER["hsmm_grid"] or GRID]
INNER_WINS = DECODER["inner_wins"]   # HSMM kept if it beats plain on the inner mean and on >= this many of 6
if DECODER["vis_factor"]:            # Y26: plain settings gain a 4th entry, cost_d's factor where the ball is missing
    PLAIN = [s + (f,) for s in PLAIN for f in DECODER["vis_factor"]]


def scores(y, pred, half):
    """Official spell scores with an X frame between the halves, so they are scored apart."""
    cut = np.flatnonzero(np.diff(half) != 0) + 1
    gt, pr = np.insert(y.astype(np.int64), cut, pm.X), np.insert(pred.astype(np.int64), cut, pm.X)
    return pm.spell_scores(gt, pr, pm.valid_runs(gt))


def stage1(data, excluded, k, mirrored=None, twin=None):
    """Stage-1 probabilities on k from a classifier trained without any match in `excluded` (k included).

    With `mirrored` (Y11) they are averaged with the same classifier's H/A-swapped probabilities on k's
    mirrored evaluation copy; those are cached beside the plain ones under the same stage-1 key.
    With `twin` (Y20: (stage-1 key, {match: X})) the same classifier also predicts k's heavy-gap evaluation
    copy, cached under the _rand twin's key: its training copies are this variant's (kora.train._copies)."""
    name = k if excluded == {k} else f"{k}_x{''.join(sorted(excluded - {k}))}"
    path = cache_dir() / f"oof_{STAGE1_VERSION}_{name}.npy"           # {k} alone is kora.train's own fold
    mpath = path.with_name(f"{path.stem}_mirror.npy")
    tpath = twin and cache_dir() / f"oof_{twin[0]}_{name}.npy"
    if path.exists() and (mirrored is None or mpath.exists()) and (twin is None or tpath.exists()):
        p = np.load(path)
    else:
        clf = fit_classifier([s for j in MATCHES if j not in excluded for s in data[j][0]])
        p = clf.predict_proba(data[k][1][0].to_numpy())
        if not path.exists():
            np.save(path, p)
        if mirrored is not None:
            np.save(mpath, clf.predict_proba(mirrored[k].to_numpy()))
        if twin is not None:
            np.save(tpath, clf.predict_proba(twin[1][k].to_numpy()))
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


def event_probs(data, y):
    """Y14: touch-event probabilities on each evaluation copy, keyed (m, k) like stage 1: (m, m) from a
    model trained without m, (m, k) from one trained without m and k. One model per excluded set
    serves every match in it. Cached by feature key and events.py."""
    from pathlib import Path
    key = FEATURE_VERSION + "e" + _hash([Path(events.__file__).read_bytes(), str(_train.SEED)])[:10]
    path = lambda m, k: cache_dir() / f"events_{key}_{k if m == k else k + '_x' + m}.npy"
    sets = [(m,) for m in MATCHES] + list(itertools.combinations(MATCHES, 2))
    todo = [ex for ex in sets if not all(path(m, k).exists() for m in ex for k in ex if len(ex) == 1 or m != k)]
    if todo:
        parts = {j: [events.rows(X, lab, data[j][1][2]) for X, lab in data[j][0]] for j in MATCHES}
        ev = {k: (events.candidates(data[k][1][0]), len(y[k])) for k in MATCHES}
        tabs = {k: events.table(data[k][1][0], ev[k][0]) for k in MATCHES}

        def job(ex):
            clf = events.fit_rows([r for j in MATCHES if j not in ex for r in parts[j]], _train.SEED)
            return {k: events.place(clf, ev[k][1], ev[k][0], tabs[k]) for k in ex}
        for ex, out in zip(todo, Parallel(n_jobs=len(MATCHES))(delayed(job)(ex) for ex in todo)):
            for k, p in out.items():
                for m in ex:
                    if len(ex) == 1 or m != k:
                        np.save(path(m, k), p)
    return {(m, k): np.load(path(m, k)) for m in MATCHES for k in MATCHES}


def eval_config(rand):
    """Y16: the variant's evaluation setup, with or without random ball gaps (mixed / mixed_rand)."""
    return {**{k: v for k, v in CONFIG.items() if k not in ("ensemble", "eval_rand")}, **({"eval_rand": True} if rand else {})}


def eval_copies(rand):
    """(feature key, {match: evaluation copy}) read from the cache of mixed or mixed_rand."""
    import pandas as pd
    fv, out = feature_version(eval_config(rand)), {}
    for m in MATCHES:
        path = cache_dir() / f"feat_{fv}_{m}.npz"
        if not path.exists():
            raise SystemExit(f"{path.name} missing: run kora.nested --variant {'mixed_rand' if rand else 'mixed'} first")
        z = np.load(path, allow_pickle=False)
        out[m] = pd.DataFrame(z["Xe"], columns=list(z["columns"]))
    return fv, out


@contextmanager
def member(seed):
    """Y16: kora.train set up for one ensemble member (training copies and classifier on `seed`, no
    evaluation gaps); yields the member's stage-1 key. kora.train reads these globals at call time."""
    cfg = {**eval_config(False), "seed": seed}
    saved = _train.SEED, _train.CONFIG, _train.FEATURE_VERSION
    _train.SEED, _train.CONFIG, _train.FEATURE_VERSION = seed, cfg, feature_version(cfg)
    try:
        yield stage1_version(cfg)
    finally:
        _train.SEED, _train.CONFIG, _train.FEATURE_VERSION = saved


def member_stage1(seed, evs, pairs=True):
    """Y16: one member's stage-1 probabilities {fv: {(excluded, k): P}} on every evaluation set in `evs`
    ({fv: {match: X}}). Excluded sets are the 7 singles and (with `pairs`) 21 pairs; a pair's classifier
    predicts on both its matches, so 28 fits serve all 49 predictions. Cached per member key and
    evaluation key; kora.ensemble needs the singles only."""
    sets = [frozenset([m]) for m in MATCHES] + [frozenset(p) for p in itertools.combinations(MATCHES, 2) if pairs]
    with member(seed) as key:
        name = lambda ex, k: k if ex == {k} else f"{k}_x{''.join(sorted(ex - {k}))}"
        path = lambda fv, ex, k: cache_dir() / f"oof_{key}_e{fv}_{name(ex, k)}.npy"
        todo = [ex for ex in sets if not all(path(fv, ex, k).exists() for fv in evs for k in ex)]
        if todo:
            train = {m: copies(m)[0] for m in MATCHES}
            for ex in todo:
                clf = fit_classifier([s for j in MATCHES if j not in ex for s in train[j]])
                for fv, X in evs.items():
                    for k in ex:
                        np.save(path(fv, ex, k), clf.predict_proba(X[k].to_numpy()))
                print(f"  member {seed}: stage 1 without {', '.join(sorted(ex))}", flush=True)
    return {fv: {(ex, k): np.load(path(fv, ex, k)) for ex in sets for k in ex} for fv in evs}


def seq_stage1(data, evs, cfg, pairs=True):
    """Y22: the temporal network's probabilities {fv: {(excluded, k): P}} on every evaluation set in `evs`,
    one fit per excluded set (the 7 singles and, with `pairs`, the 21 pairs, each predicting both its
    matches) on this variant's training copies. Cached by the training copies' stage-1 key, kora/seq.py,
    the settings and the seed, so a _rand twin reuses the fits. The tree classifier ("clf", Y20) is left out
    of the key: it does not change the training copies the network sees (Y24 reuses Y23's fits)."""
    from pathlib import Path
    key = stage1_version({k: v for k, v in eval_config(False).items() if k not in ("clf", "grf", "mix")}) + "q" + _hash(
        [Path(seq.__file__).read_bytes(), json.dumps(cfg.get("settings", {}), sort_keys=True), str(cfg["seed"])])
    sets = [frozenset([m]) for m in MATCHES] + [frozenset(p) for p in itertools.combinations(MATCHES, 2) if pairs]
    name = lambda ex, k: k if ex == {k} else f"{k}_x{''.join(sorted(ex - {k}))}"
    path = lambda fv, ex, k: cache_dir() / f"seq_{key}_e{fv}_{name(ex, k)}.npy"
    half = {m: data[m][1][2] for m in MATCHES}
    for ex in sets:
        if all(path(fv, ex, k).exists() for fv in evs for k in ex):
            continue
        train = [(s, half[j]) for j in MATCHES if j not in ex for s in data[j][0]]
        model = seq.fit([s for s, _ in train], [h for _, h in train], cfg["seed"], CONFIG.get("balance") or 0.0,
                        cfg.get("settings"))
        for fv, X in evs.items():
            for k in ex:
                np.save(path(fv, ex, k), seq.predict(model, X[k], half[k]))
        print(f"  seq {cfg['seed']}: without {', '.join(sorted(ex))}", flush=True)
    return {fv: {(ex, k): np.load(path(fv, ex, k)) for ex in sets for k in ex} for fv in evs}


def event_scores(paths, E, y, half, theta, max_len):
    return [scores(y[k], events.relabel(paths[k], E[k], half[k], theta, max_len), half[k])["mean_f1"] for k in paths]


def fit_detector(ctx, tgt, train, test):
    X = np.concatenate([ctx[j][::2] for j in train])
    y = np.concatenate([tgt[j][::2] for j in train])
    return detector().fit(X, y).predict_proba(ctx[test])[:, 1]


def fit_refiner(ctx, labels, train, test):
    """Stage 2 (Y7a, the MS-TCN idea): H/A/D probabilities on test from the detector's context windows."""
    X = np.concatenate([ctx[j][::2] for j in train])
    y = np.concatenate([labels[j][::2] for j in train])
    return detector().fit(X, y).predict_proba(ctx[test])


def plain_path(P, half, prior, setting, missing=None):
    """Plain decoder path; a 4-entry setting (Y26) is kora/visdecode.py's, with `missing` the ball-gap frames."""
    if len(setting) == 4:
        return visdecode.decode(P, segments_of(half), prior, *setting, missing)
    return decode(P, segments_of(half), prior, *setting)


def plain_scores(P, y, half, prior, setting, missing=None):
    return [scores(y[k], plain_path(P[k], half[k], prior, setting, missing and missing[k]), half[k])["mean_f1"]
            for k in P]


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


def snap_scores(paths, C, y, half, r, mode):
    return [scores(y[k], snap(paths[k], C[k], half[k], r, mode), half[k])["mean_f1"] for k in paths]


def screen(outer, y, half, labels, P0=None):
    """Y20 screen: kora.train's plain-decoder pick (best mean over the seven outer folds, so leaky like
    kora.train) for this variant and for its base, mixed or mixed_rand, from the base's cached outer folds on
    the same evaluation copies. Y22 passes P0, the same run's outer folds before the temporal network is
    blended in. Writes models/screen_<variant>.json."""
    base = f"{NAME} without seq" if P0 is not None else "mixed_rand" if CONFIG.get("eval_rand") else "mixed"
    if P0 is None:
        if feature_version(VARIANTS[base]) != FEATURE_VERSION:
            raise SystemExit(f"{NAME} and {base} must share their evaluation copies")
        key = stage1_version(VARIANTS[base])
        P0 = {m: np.load(cache_dir() / f"oof_{key}_{m}.npy") for m in MATCHES}
    prior = np.bincount(np.concatenate([labels[m] for m in MATCHES]), minlength=3) / sum(map(len, labels.values()))

    def best(P):
        s = np.array(Parallel(n_jobs=-1)(delayed(plain_scores)(P, y, half, prior, g) for g in PLAIN))
        i = int(s.mean(1).argmax())
        return s[i], PLAIN[i]
    (b, gb), (c, gc) = best(P0), best(outer)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    out = {"variant": NAME, "base": base, "commit": commit, "stage1_version": STAGE1_VERSION,
           "base_mean": float(b.mean()), "mean": float(c.mean()), "delta": float((c - b).mean()),
           "wins": int((c > b).sum()), "base_decoder": gb, "decoder": gc,
           "per_match": {m: [float(x), float(z)] for m, x, z in zip(MATCHES, b, c)}}
    (MODELS / f"screen_{NAME}.json").write_text(json.dumps(out, indent=2))
    print(f"screen {base} {b.mean():.4f} -> {NAME} {c.mean():.4f}  delta {out['delta']:+.4f}  better on "
          f"{out['wins']}/7  decoders {gb} / {gc}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", default=NAME, choices=VARIANTS, help="read at import by kora.variant")
    parser.add_argument("--screen", action="store_true", help="Y20: outer folds only, scored against the base")
    args = parser.parse_args()
    if CONFIG.get("seed"):             # read at call time by kora.train (copies, classifier)
        _train.SEED = CONFIG["seed"]
    if CONFIG.get("fill"):             # Y15: kora.train builds every copy through this (train.py unchanged, keys kept)
        _train.build = fill.wrap(_train.build)
    engine.use(CONFIG)                 # Y20: the variant's stage-1 classifier, same way
    if CONFIG.get("grf"):              # Y28: synthetic GRF rows in every stage-1 fit, same way
        global fit_classifier
        fit_classifier = grf.fit_classifier(CONFIG["grf"])
    t0 = time.time()
    print(f"variant {NAME} {CONFIG}  features {FEATURE_VERSION}  stage 1 {STAGE1_VERSION}", flush=True)
    data = {m: copies(m) for m in MATCHES}
    if CONFIG.get("mix"):              # Y29: more copies, or more with random ball gaps; evaluation copies unchanged
        data = mix.apply(data, CONFIG["mix"])
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
    E = event_probs(data, y) if DECODER["events"] else None
    if E is not None:
        print(f"event models done ({time.time() - t0:.0f} s)", flush=True)
    tgt = {m: target(y[m], half[m], **DETECTOR_CFG) for m in MATCHES}

    # 1. stage-1 predictions: outer (kora.train folds) and inner (without m and k)
    twin = None
    if (CONFIG.get("clf") or CONFIG.get("grf") or CONFIG.get("mix")) and not CONFIG.get("eval_rand") and not CONFIG.get("ensemble"):   # Y20, Y28, Y29
        twin = (stage1_version({**CONFIG, "eval_rand": True}), eval_copies(True)[1])
    outer = {m: stage1(data, {m}, m, mirrored, twin) for m in MATCHES}
    if args.screen and not CONFIG.get("seq"):
        return screen(outer, y, half, labels)
    inner = {}
    for m in MATCHES:
        for k in MATCHES:
            if k != m and not args.screen:
                inner[m, k] = stage1(data, {m, k}, k, mirrored, twin)
                print(f"stage 1 without {m}, {k} ({time.time() - t0:.0f} s)", flush=True)
    cross = None
    if CONFIG.get("cross"):            # Y26: detector, prior and decoder chosen on the other gap condition's inner folds
        if args.screen or CONFIG.get("tta") or DECODER["refine"] or DECODER["cuts"] or DECODER["events"] or DECODER["snap"] or DECODER["vis_factor"] \
                or "weight_missing" in CONFIG.get("seq", {}):
            raise SystemExit("cross is registered on the plain and HSMM decoders only")
        fv_o, X_o = eval_copies(not CONFIG.get("eval_rand"))
        key_o = stage1_version(eval_config(not CONFIG.get("eval_rand")))
        cross = {"fv": fv_o, "raw": {m: X_o[m][RAW] for m in MATCHES},
                 "inner": {mk: np.load(cache_dir() / f"oof_{key_o}_{mk[1]}_x{mk[0]}.npy") for mk in inner}}
    if CONFIG.get("seq"):              # Y22: fitted while the training copies are loaded, blended in below
        evs, sc = dict(eval_copies(r) for r in (False, True)), CONFIG["seq"]
        parts = [seq_stage1(data, evs, {**sc, "seed": s}, pairs=not args.screen)
                 for s in sc.get("seeds") or [sc["seed"]]]                   # Y23: averaged over seeds
        seqP = {key: np.mean([p[FEATURE_VERSION][key] for p in parts], axis=0) for key in parts[0][FEATURE_VERSION]}
        if cross:
            cross["seq"] = {key: np.mean([p[cross["fv"]][key] for p in parts], axis=0) for key in seqP}
        print(f"temporal network done ({time.time() - t0:.0f} s)", flush=True)
    del data, mirrored
    if CONFIG.get("ensemble"):         # Y16: equal-weight average with the other members' stage 1
        base, *others = CONFIG["ensemble"]
        evs = dict(eval_copies(r) for r in (False, True))
        if base != _train.SEED or FEATURE_VERSION not in evs:
            raise SystemExit("the first ensemble seed must be production's and the evaluation copies its own")
        for s in others:
            pp = member_stage1(s, evs, pairs=not args.screen)
            p = pp[FEATURE_VERSION]
            outer = {m: outer[m] + p[frozenset([m]), m] for m in MATCHES}
            inner = {(m, k): v + p[frozenset([m, k]), k] for (m, k), v in inner.items()}
            if cross:
                cross["inner"] = {(m, k): v + pp[cross["fv"]][frozenset([m, k]), k] for (m, k), v in cross["inner"].items()}
            print(f"member {s} added ({time.time() - t0:.0f} s)", flush=True)
        outer = {m: v / len(CONFIG["ensemble"]) for m, v in outer.items()}
        inner = {mk: v / len(CONFIG["ensemble"]) for mk, v in inner.items()}
        if cross:
            cross["inner"] = {mk: v / len(CONFIG["ensemble"]) for mk, v in cross["inner"].items()}
    if CONFIG.get("seq"):              # Y22: (1 - w) x trees + w x temporal network
        w, trees = CONFIG["seq"]["weight"], outer
        if "weight_missing" in CONFIG["seq"]:   # Y26: the network's weight on frames without the ball
            wm = CONFIG["seq"]["weight_missing"]
            wk = {k: np.where(seen[k] == 1, w, wm)[:, None] for k in MATCHES}
        else:
            wk = {k: w for k in MATCHES}
        outer = {m: (1 - wk[m]) * v + wk[m] * seqP[frozenset([m]), m] for m, v in outer.items()}
        inner = {(m, k): (1 - wk[k]) * v + wk[k] * seqP[frozenset([m, k]), k] for (m, k), v in inner.items()}
        if cross:
            cross["inner"] = {(m, k): (1 - w) * v + w * cross["seq"][frozenset([m, k]), k]
                              for (m, k), v in cross["inner"].items()}
        if args.screen:
            return screen(outer, y, half, labels, P0=trees)

    # 2. per outer match: detector folds on the inner matches, then the detector for m
    if cross:                          # Y26: from here on the inner folds are the other condition's; outer[m] is unchanged
        inner, raw_in = cross["inner"], cross["raw"]
    else:
        raw_in = raw
    ctx = {(m, k): context(inner[m, k], raw_in[k]) for m, k in inner}
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

        miss = {k: seen[k] == 0 for k in ins} if DECODER["vis_factor"] else None
        plain = np.array(Parallel(n_jobs=-1)(delayed(plain_scores)(P, yi, hi, prior, s, miss) for s in PLAIN))
        bp = int(plain.mean(1).argmax())
        hs = np.array(Parallel(n_jobs=-1)(delayed(hsmm_scores)(P, C, yi, hi, prior, rate, g) for g in HSMM))
        bh = int(hs.mean(1).argmax())
        keep = bool(hs[bh].mean() > plain[bp].mean() and (hs[bh] > plain[bp]).sum() >= INNER_WINS)

        ld = log_duration([r for k in ins for r in runs_of(y[k], half[k])])
        seg = segments_of(half[m])
        pred_plain = plain_path(outer[m], half[m], prior, PLAIN[bp], seen[m] == 0)
        pred_hsmm = decode_hsmm(outer[m], change[m, m], seg, prior, ld, *HSMM[bh][:4], rate, *HSMM[bh][4:])
        pred = pred_hsmm if keep else pred_plain
        s_nocut, cut = scores(y[m], pred, half[m]), {"theta": None, "gap": None}
        if DECODER["cuts"] or DECODER["events"] or DECODER["snap"]:
            paths = (hsmm_paths(P, C, yi, hi, prior, rate, HSMM[bh]) if keep else
                     {k: plain_path(P[k], half[k], prior, PLAIN[bp], miss and miss[k]) for k in ins})
            off = np.array([scores(y[k], paths[k], half[k])["mean_f1"] for k in ins])
        if DECODER["cuts"]:                                            # Y10: cut at detector peaks (kora/cuts.py)
            grid = [(t, g) for t in DECODER["cuts"]["theta"] for g in DECODER["cuts"]["gap"]]
            cs = np.array(Parallel(n_jobs=-1)(delayed(cut_scores)(paths, P, C, yi, hi, t, g) for t, g in grid))
            bc = int(cs.mean(1).argmax())
            cut.update(inner_mean=float(cs[bc].mean()), inner_mean_off=float(off.mean()))
            if cs[bc].mean() > off.mean() and (cs[bc] > off).sum() >= INNER_WINS:   # same rule as the HSMM
                cut.update(theta=grid[bc][0], gap=grid[bc][1])
                pred = relabel(pred, outer[m], change[m, m], half[m], *grid[bc])
        s_noevent, evsel = scores(y[m], pred, half[m]), {"theta": None, "max_len": None}
        if DECODER["events"]:                                          # Y14: closed event pairs (kora/events.py)
            grid = [(t, n) for t in DECODER["events"]["theta"] for n in DECODER["events"]["max_len"]]
            es = np.array(Parallel(n_jobs=-1)(delayed(event_scores)(paths, {k: E[m, k] for k in ins}, yi, hi, t, n)
                                              for t, n in grid))
            be = int(es.mean(1).argmax())
            evsel.update(inner_mean=float(es[be].mean()), inner_mean_off=float(off.mean()))
            if es[be].mean() > off.mean() and (es[be] > off).sum() >= INNER_WINS:   # same rule as the HSMM
                evsel.update(theta=grid[be][0], max_len=grid[be][1])
                pred = events.relabel(pred, E[m, m], half[m], *grid[be])
        s_nosnap, snapsel = scores(y[m], pred, half[m]), {"r": None, "mode": None}
        if DECODER["snap"]:                                            # Y27: move H<->A changes to the detector (kora/snap.py)
            grid = [(r, md) for r in DECODER["snap"]["r"] for md in DECODER["snap"]["mode"]]
            ss = np.array(Parallel(n_jobs=-1)(delayed(snap_scores)(paths, C, yi, hi, r, md) for r, md in grid))
            bs = int(ss.mean(1).argmax())
            snapsel.update(inner_mean=float(ss[bs].mean()), inner_mean_off=float(off.mean()))
            if ss[bs].mean() > off.mean() and (ss[bs] > off).sum() >= INNER_WINS:   # same rule as the HSMM
                snapsel.update(r=grid[bs][0], mode=grid[bs][1])
                pred = snap(pred, change[m, m], half[m], *grid[bs])
        s, s_plain, s_hsmm = scores(y[m], pred, half[m]), scores(y[m], pred_plain, half[m]), scores(y[m], pred_hsmm, half[m])
        folds[m] = {
            "inner_matches": ins, "decoder": "hsmm" if keep else "plain",
            "plain": dict(zip(("alpha", "cost_ha", "cost_d", "factor"), PLAIN[bp]), inner_mean=float(plain[bp].mean())),
            "hsmm": dict(zip(("cost_ha", "cost_d", "w_dur", "beta", "ha_bonus_only"),
                             (HSMM[bh] + (False,))[:5]), inner_mean=float(hs[bh].mean()), rate=rate),
            "prior": prior.tolist(),
            "spell_f1": s["mean_f1"], "spell_f1_per_iou": {str(t): v["f1"] for t, v in s["per_iou"].items()},
            "n_true": s["n_gt"], "n_pred": s["n_pred"],
            "spell_f1_plain": s_plain["mean_f1"], "spell_f1_hsmm": s_hsmm["mean_f1"],
            "spell_f1_nocut": s_nocut["mean_f1"], "cuts": cut,
            "spell_f1_noevent": s_noevent["mean_f1"], "events": evsel,
            "spell_f1_nosnap": s_nosnap["mean_f1"], "snap": snapsel,
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
