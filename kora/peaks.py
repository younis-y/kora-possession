"""Weak-touch peaks (variant switch "peaks", RESEARCH.md Y12).

touch.py marks a touch where the ball's velocity change near a player passes TOUCH_DV (2.7 m/s), which
already covers the gain rule of Vidal-Codina et al. (2022). One-touch deflections are often weaker:
ballradar (Kim et al., KDD 2023) takes local maxima of a touch score between 0.2 and 0.5 of its strong
threshold as one-touch passes, and Link & Hoernig (2017) detect touches from acceleration peaks. From
touch.py's columns:

- weak touch   max(touchH, touchA) is a local maximum over +-2 frames and in [0.2, 1) x TOUCH_DV;
               its side is the stronger team (0 if tied)
- wlast/wnext, t_wlast/t_wnext   side of and seconds since / until the nearest weak touch, capped
- wtouch_p/f{25,50}   home minus away weak touches in the past / next 1 s and 2 s
"""
import numpy as np
import pandas as pd

from .features import CAP_S, FPS, carry, nanroll
from .touch import TOUCH_DV


def weak_touches(th, ta):
    t = np.maximum(th, ta)
    local = np.ones(len(t), bool)
    for s in (1, 2):
        local &= t >= np.r_[np.full(s, -np.inf), t[:-s]]
        local &= t > np.r_[t[s:], np.full(s, -np.inf)]
    return local & (t >= 0.2 * TOUCH_DV) & (t < TOUCH_DV), np.sign(th - ta).astype(np.float32)


def add(X):
    """X with the weak-touch columns appended (needs touchH/touchA from kora/touch.py)."""
    weak, side = weak_touches(np.nan_to_num(X["touchH"].to_numpy()), np.nan_to_num(X["touchA"].to_numpy()))
    f = {}
    f["wlast"], dl = carry(weak, side, "f")
    f["wnext"], dn = carry(weak, side, "b")
    f["t_wlast"] = np.minimum(dl / FPS, CAP_S)
    f["t_wnext"] = np.minimum(dn / FPS, CAP_S)
    eh, ea = (weak & (side > 0)).astype(np.float32), (weak & (side < 0)).astype(np.float32)
    for w in (25, 50):
        for mode in ("p", "f"):
            f[f"wtouch_{mode}{w}"] = (nanroll(eh, w, mode) - nanroll(ea, w, mode)) * w
    return pd.concat([X, pd.DataFrame({k: np.asarray(v, np.float32) for k, v in f.items()}, index=X.index)],
                     axis=1)
