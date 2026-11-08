"""Label every frame of the challenge match.

    python -m kora.run --out "Kora Analytics.csv"

Needs models/ from `python -m kora.train` (trained on the open Bundesliga matches only).
"""
import argparse
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .boundary import context
from .decode import decode, decode_hsmm
from .features import build
from .target import load_target

warnings.filterwarnings("ignore", category=RuntimeWarning)
MODELS = Path(__file__).resolve().parent.parent / "models"


def stage1(bundle, X, cfg=None, half=None):
    """Average the trees, then optionally blend the saved CPU networks, each half separately."""
    x = X[bundle["columns"]].to_numpy()
    proba = np.mean([m.predict_proba(x) for m in bundle.get("models") or [bundle["model"]]], axis=0)
    if cfg is not None and "seq" in cfg:
        from . import seqdeploy
        sc = cfg["seq"]
        net = np.mean([seqdeploy.predict_cpu(seqdeploy.load(MODELS / name), X, half)
                       for name in sc["files"]], axis=0)
        proba = (1 - sc["weight"]) * proba + sc["weight"] * net
    return proba


def predict():
    players, team, ball, section, graded = load_target()
    X = build(players, team, ball, section)
    bundle = joblib.load(MODELS / "classifier.joblib")
    cfg = json.loads((MODELS / "decoder.json").read_text())
    proba = stage1(bundle, X, cfg, section)       # section identifies the halves, including ungraded frames
    prior = np.array(cfg["prior"])
    if "hsmm" in cfg:                         # kept only if it won leave-one-match-out CV
        h = cfg["hsmm"]
        det = joblib.load(MODELS / "boundary.joblib")
        change = det["model"].predict_proba(context(proba, X[det["raw"]]))[:, 1]
        path = decode_hsmm(proba, change, graded, prior, np.array(h["log_duration"]),
                           h["cost_ha"], h["cost_d"], h["w_dur"], h["beta"], h["rate"])
    else:
        path = decode(proba, graded, prior, cfg["alpha"], cfg["cost_ha"], cfg["cost_d"])
    labels = np.full(len(path), "X", dtype="<U1")
    for a, b in graded:
        labels[a:b] = np.array(["H", "A", "D"])[path[a:b]]
    return labels, proba


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    labels, _ = predict()
    pd.DataFrame({"frame": np.arange(len(labels)), "label": labels}).to_csv(a.out, index=False)
    g = labels != "X"
    codes = labels[g]
    change = np.flatnonzero(codes[1:] != codes[:-1]) + 1
    starts = np.r_[0, change]
    spells = [c for c in codes[starts] if c in "HA"]
    live = np.isin(codes, ["H", "A"]).sum()
    print(f"wrote {a.out}: " + ", ".join(f"{k} {int((codes == k).sum()):,}" for k in "HAD")
          + f"; {len(spells)} possessions, mean {live / max(len(spells), 1) / 25:.1f} s")


if __name__ == "__main__":
    main()
