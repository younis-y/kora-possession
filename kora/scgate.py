"""Y19 gate: do SkillCorner's event-derived possession labels look like the possession the challenge scores?

    python -m kora.scgate        # writes models/gate_y19.json; reads only skillcorner.GATE

Rules and thresholds are pre-registered in RESEARCH.md Y19, before this ran. Nothing is fitted on the
SkillCorner labels: G1 compares label statistics with IDSSE's, G2 checks the frame alignment against
SkillCorner's own per-frame possession group, G3 scores the shipped models (trained on IDSSE only) against
the labels. G4 is reported, not gated: does the real-gap benchmark rank mixed above touch_bal, as heavy
simulated gaps did (V8, J6)?
"""
import json
import time

import joblib
import numpy as np

from . import skillcorner as sc
from . import train as _train
from .data import cache_dir
from .decode import decode
from .external import MATCHES, load as idsse
from .features import build
from .nested import scores
from .run import MODELS, stage1
from .train import segments_of
from .variant import NAME, PRODUCTION, VARIANTS, feature_version, stage1_version
from .vendor import possession_metrics as pm

FPS = 25
G1_DEAD = (0.30, 0.60)       # dead share of labelled frames
G1_RATIO = (0.67, 1.5)       # spells per 45 min and median spell length, relative to IDSSE
G1_UNKNOWN = 0.15            # largest unknown share of frames
G2_AGREE = 0.90              # event label vs possession.group, frames both call H or A
G3_SPELL = 0.45              # production mean spellF1
G3_TEAM = 0.80               # production team accuracy, frames both call H or A


def stats(label, half):
    """Dead share, H/A spells per 45 scored minutes, median spell length (s), unknown share."""
    cut = np.flatnonzero(np.diff(half) != 0) + 1
    gt = np.insert(label.astype(np.int64), cut, pm.X)
    spells = pm.possession_spells(gt, pm.valid_runs(gt))
    known = label != pm.X
    minutes = known.sum() / FPS / 60
    return {"dead": float((label == pm.D).sum() / known.sum()), "spells_per_45": len(spells) / minutes * 45,
            "median_s": float(np.median([(b - a) / FPS for a, b, _ in spells])),
            "unknown": float(1 - known.mean())}


def touch_bal_model():
    """touch_bal on all seven IDSSE matches (one fit, cached), for G4."""
    cfg = VARIANTS["touch_bal"]
    path = cache_dir() / f"full_{stage1_version(cfg)}.joblib"
    if path.exists():
        return joblib.load(path)
    saved = _train.CONFIG, _train.FEATURE_VERSION
    _train.CONFIG, _train.FEATURE_VERSION = cfg, feature_version(cfg)
    try:
        clf = _train.fit_classifier([s for m in MATCHES for s in _train.copies(m)[0]])
    finally:
        _train.CONFIG, _train.FEATURE_VERSION = saved
    joblib.dump(clf, path)
    return clf


def main():
    if NAME != PRODUCTION:
        raise SystemExit("run with the production variant: the features must be production's")
    t0 = time.time()
    bundle = joblib.load(MODELS / "classifier.joblib")
    cfg = json.loads((MODELS / "decoder.json").read_text())
    if "hsmm" in cfg:
        raise SystemExit("the gate decodes with the shipped plain decoder; models/ holds an HSMM")
    prior, setting = np.array(cfg["prior"]), (cfg["alpha"], cfg["cost_ha"], cfg["cost_d"])
    ref = [stats(d["label"], d["half"]) for d in map(idsse, MATCHES)]
    ref = {k: float(np.mean([r[k] for r in ref])) for k in ref[0]}
    models = {"y16_seeds": bundle, "mixed": {"models": bundle["models"][:1], "columns": bundle["columns"]},
              "touch_bal": {"model": touch_bal_model(), "columns": bundle["columns"]}}
    print(f"IDSSE reference {ref}  ({time.time() - t0:.0f} s)", flush=True)

    per = {}
    for m in sc.GATE:
        lab10, grp = sc.events_label(m), sc.raw(m)["label"]
        both = (lab10 < 2) & (grp < 2)
        d = sc.load(m, events=True)
        y, half = d["label"], d["half"]
        X = build(d["players"], d["team"], d["ball"], half)
        out = {**stats(y, half), "group_agree": float((lab10[both] == grp[both]).mean()), "models": {}}
        for name, b in models.items():
            pred = decode(stage1(b, X), segments_of(half), prior, *setting)
            s = scores(y, pred, half)
            ha = (y < 2) & (pred < 2)
            out["models"][name] = {"spell_f1": s["mean_f1"], "team_acc": float((pred[ha] == y[ha]).mean()),
                                   "n_true": s["n_gt"], "n_pred": s["n_pred"]}
        per[m] = out
        print(m, json.dumps({k: v for k, v in out.items() if k != "models"}),
              {k: round(v["spell_f1"], 4) for k, v in out["models"].items()}, f"({time.time() - t0:.0f} s)", flush=True)

    mean = lambda key: float(np.mean([per[m][key] for m in sc.GATE]))
    prod = lambda key: float(np.mean([per[m]["models"]["y16_seeds"][key] for m in sc.GATE]))
    g1 = (G1_DEAD[0] <= mean("dead") <= G1_DEAD[1] and mean("unknown") <= G1_UNKNOWN and
          all(G1_RATIO[0] <= mean(k) / ref[k] <= G1_RATIO[1] for k in ("spells_per_45", "median_s")))
    g2 = mean("group_agree") >= G2_AGREE
    g3 = prod("spell_f1") >= G3_SPELL and prod("team_acc") >= G3_TEAM
    g4 = {n: float(np.mean([per[m]["models"][n]["spell_f1"] for m in sc.GATE])) for n in models}
    verdict = {"G1_labels": g1, "G2_alignment": g2, "G3_production": g3, "pass": bool(g1 and g2 and g3)}
    result = {"gate": list(sc.GATE), "rule": {"FLIGHT_S": sc.FLIGHT_S, "OUT_S": sc.OUT_S, "RESTART_S": sc.RESTART_S},
              "thresholds": {"G1_DEAD": G1_DEAD, "G1_RATIO": G1_RATIO, "G1_UNKNOWN": G1_UNKNOWN, "G2_AGREE": G2_AGREE,
                             "G3_SPELL": G3_SPELL, "G3_TEAM": G3_TEAM},
              "idsse": ref, "skillcorner_mean": {k: mean(k) for k in ("dead", "spells_per_45", "median_s", "unknown",
                                                                        "group_agree")},
              "production": {"spell_f1": prod("spell_f1"), "team_acc": prod("team_acc")}, "G4_spell_f1": g4,
              "per_match": per, "verdict": verdict}
    (MODELS / "gate_y19.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: result[k] for k in ("idsse", "skillcorner_mean", "production", "G4_spell_f1", "verdict")}, indent=1))
    print(f"({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
