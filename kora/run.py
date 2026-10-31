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

from .decode import decode
from .features import build
from .stack import context
from .target import load_target

warnings.filterwarnings("ignore", category=RuntimeWarning)
MODELS = Path(__file__).resolve().parent.parent / "models"


def predict():
    players, team, ball, section, graded = load_target()
    X = build(players, team, ball, section)
    bundle = joblib.load(MODELS / "classifier.joblib")
    cfg = json.loads((MODELS / "decoder.json").read_text())
    proba = bundle["model"].predict_proba(X[bundle["columns"]].to_numpy())
    dec = cfg
    if "stage2" in cfg:                       # kept only if it won leave-one-match-out CV
        s2 = joblib.load(MODELS / "stage2.joblib")
        proba = s2["model"].predict_proba(context(proba, X[s2["raw"]]).to_numpy())
        dec = cfg["stage2"]
    path = decode(proba, graded, np.array(cfg["prior"]), dec["alpha"], dec["cost_ha"], dec["cost_d"])
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
