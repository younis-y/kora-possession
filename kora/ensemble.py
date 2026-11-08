"""Y16 in the submission: stage 1 averaged over the production variant's seed members.

    python -m kora.ensemble      # after kora.train, before kora.boundary (build.sh)

kora.train fits the first member (production's own seed). This fits the others on all seven matches,
averages their leave-one-match-out probabilities with kora.train's (the cache of the Y16 nested run, or
7 fits per member), re-picks the plain decoder on the average exactly as kora.train does, and rewrites
models/classifier.joblib (every member), oof.npz and decoder.json, so kora.boundary fits the detector on
the average, as kora.nested did. With "seq", it also blends the cached network folds and saves one
final network per seed. A variant without "ensemble" or "seq" leaves models/ untouched. Kept out of
kora/train.py so its stage-1 caches keep their keys.
"""
import itertools
import json
import time

import joblib
import numpy as np

from . import nested, train as _train
from .decode import decode
from .external import MATCHES
from .train import GRID_ALPHA, GRID_D, GRID_HA, MODELS, copies, fit_classifier, segments_of, spell_f1
from .variant import CONFIG


def pick_decoder(oof, ev, prior):
    """kora.train's plain-decoder choice: best mean held-out spellF1, first in grid order on ties."""
    results = []
    for alpha, cha, cd in itertools.product(GRID_ALPHA, GRID_HA, GRID_D):
        f1s, mfs = zip(*(spell_f1(ev[m][1], decode(oof[m], segments_of(ev[m][2]), prior, alpha, cha, cd), ev[m][2])
                         for m in MATCHES))
        results.append((float(np.mean(f1s)), float(np.std(f1s)), float(np.mean(mfs)), alpha, cha, cd, list(f1s)))
    return max(results, key=lambda r: r[0])


def add_sequence(oof, data):
    """Blend cached single-fold networks, save all-match fits, and return decoder metadata."""
    import torch
    from . import seqdeploy
    sc = CONFIG["seq"]
    seeds = sc.get("seeds") or [sc["seed"]]
    evs = {_train.FEATURE_VERSION: {m: data[m][1][0] for m in MATCHES}}
    parts = [nested.seq_stage1(data, evs, {**sc, "seed": s}, pairs=False)[_train.FEATURE_VERSION]
             for s in seeds]
    w = sc["weight"]
    blended = {m: (1 - w) * v + w * np.mean([p[frozenset([m]), m] for p in parts], axis=0)
               for m, v in oof.items()}
    files = []
    for s in seeds:
        model = seqdeploy.fit_final(data, sc, s)
        name = f"seq_{s}.pt"
        seqdeploy.save(model, MODELS / name)
        files.append(name)
        print(f"network {s} fitted", flush=True)
    return blended, {"weight": w, "seeds": seeds, "settings": model["settings"],
                     "files": files, "torch_version": str(torch.__version__)}


def main():
    seeds = CONFIG.get("ensemble")
    if not seeds and not CONFIG.get("seq"):
        print("variant has no ensemble: models/ unchanged")
        return
    seeds = seeds or [_train.SEED]
    if seeds[0] != _train.SEED:
        raise SystemExit("the first ensemble seed must be kora.train's")
    t0 = time.time()
    if CONFIG.get("seq"):
        data = {m: copies(m) for m in MATCHES}
        ev = {m: data[m][1] for m in MATCHES}
    else:
        ev = {m: copies(m)[1] for m in MATCHES}
    bundle = joblib.load(MODELS / "classifier.joblib")
    cfg = json.loads((MODELS / "decoder.json").read_text())
    if cfg.get("ensemble"):
        raise SystemExit("models/ already holds an ensemble: run kora.train first")
    with np.load(MODELS / "oof.npz") as z:
        oof = {m: z[m] for m in MATCHES}
    models = [bundle["model"]]
    for s in seeds[1:]:
        p = nested.member_stage1(s, {_train.FEATURE_VERSION: {m: ev[m][0] for m in MATCHES}}, pairs=False)
        oof = {m: oof[m] + p[_train.FEATURE_VERSION][frozenset([m]), m] for m in MATCHES}
        with nested.member(s):
            models.append(fit_classifier([c for m in MATCHES for c in copies(m)[0]]))
        print(f"member {s} fitted ({time.time() - t0:.0f} s)", flush=True)
    oof = {m: v / len(seeds) for m, v in oof.items()}
    if CONFIG.get("seq"):
        oof, cfg["seq"] = add_sequence(oof, data)
    best = pick_decoder(oof, ev, np.array(cfg["prior"]))
    print(f"plain decoder on the average: spellF1 {best[0]:.4f} +- {best[1]:.4f}  alpha {best[3]}  "
          f"cost_HA {best[4]}  cost_D {best[5]}  (one model: {cfg['cv_spell_f1_mean']:.4f})")
    cfg.update(alpha=best[3], cost_ha=best[4], cost_d=best[5], cv_spell_f1_mean=best[0], cv_spell_f1_std=best[1],
               cv_spell_f1_per_match=dict(zip(MATCHES, best[6])), cv_macro_f1_mean=best[2], ensemble=seeds)
    joblib.dump({"models": models, "columns": bundle["columns"]}, MODELS / "classifier.joblib", compress=3)
    (MODELS / "decoder.json").write_text(json.dumps(cfg, indent=2))
    np.savez_compressed(MODELS / "oof.npz", **oof)
    print(f"saved models/ ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
