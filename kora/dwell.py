"""Possession dwell and pressure features (variant switch "dwell", RESEARCH.md Y8a).

Link & Hoernig (PLOS ONE 2017): a player possesses the ball while it stays in their sphere of
influence, and the opponent's distance grades how secure that control is. Built only from columns
features.py already makes, so the feature cache is reused and only stage 1 is refitted:

- dwell_p/f        seconds the close side (last_close / next_close: the team nearest the ball when
                   it was last / is next within 1.5 m of a player) has held up to / will hold from
                   this frame, capped at CAP_S
- flips_p/f{w}     changes of that side in the past / next 25 and 50 frames (passes inside a team
                   keep it, interceptions flip it)
- press            distance from the ball to the nearest player of the team farther from it
- press_c13, press_j12   its 0.5 s mean and its change from the 0.5 s before to the 0.5 s after

Nothing is tuned on labels: 1.5 m is the radius features.py already uses for last_close.
"""
import numpy as np
import pandas as pd

from .features import CAP_S, FPS, nanroll


def since_change(side):
    """Frames since `side` last took a new value; NaN where side is NaN."""
    n = len(side)
    v = np.where(np.isfinite(side), side, 9.0)
    start = np.maximum.accumulate(np.where(np.r_[True, v[1:] != v[:-1]], np.arange(n), 0))
    return np.where(np.isfinite(side), np.arange(n) - start, np.nan)


def add(X):
    """X with the dwell columns appended (X is a features.build frame)."""
    last, nxt = X["last_close"].to_numpy(), X["next_close"].to_numpy()
    f = {"dwell_p": np.minimum(since_change(last) / FPS, CAP_S),
         "dwell_f": np.minimum(since_change(nxt[::-1])[::-1] / FPS, CAP_S)}
    flip = np.r_[False, np.isfinite(last[1:]) & np.isfinite(last[:-1]) & (last[1:] != last[:-1])]
    for w in (25, 50):
        f[f"flips_p{w}"] = nanroll(flip, w, "p") * w
        f[f"flips_f{w}"] = nanroll(flip, w, "f") * w
    d1H, d1A = X["d1H"].to_numpy(), X["d1A"].to_numpy()
    press = np.where(np.isfinite(d1H) & np.isfinite(d1A), np.maximum(d1H, d1A), np.nan)
    f["press"] = press
    f["press_c13"] = nanroll(press, 13, "c")
    f["press_j12"] = nanroll(press, 12, "f") - nanroll(press, 12, "p")
    return pd.concat([X, pd.DataFrame({k: np.asarray(v, np.float32) for k, v in f.items()}, index=X.index)],
                     axis=1)
