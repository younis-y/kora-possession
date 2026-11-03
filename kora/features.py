"""Per-frame features, identical for the training matches and the challenge match.

Inputs are in metres with the origin on the centre spot: players[n, P, 2] (NaN = not tracked),
team[P] (0 home, 1 away), ball[n, 2] (NaN = not seen), section[n] (orientation block, e.g. half).

Design rules:
- Permutation-invariant: track slots carry no identity, so everything is a team-level aggregate.
- Orientation-canonical: within each section the home team attacks +x.
- Missing-aware: nothing assumes the ball is visible; ball-less stretches get their own context
  (time since / until the ball is seen, what it looked like at either end of the gap).
- Bidirectional context: the submission is offline, so windows look both ways in time.
"""
import numpy as np
import pandas as pd

from .variant import CONFIG

FPS = 25
HALF_L, HALF_W = 52.5, 34.0
CAP_S = 120.0


# ---------- helpers ----------

def nanroll(x, w, mode="c"):
    """NaN-aware moving mean over w frames: centred ('c'), past-only ('p') or future-only ('f')."""
    x = np.asarray(x, dtype=np.float64)
    v = np.where(np.isfinite(x), x, 0.0)
    c = np.isfinite(x).astype(np.float64)
    cv = np.concatenate([[0.0], np.cumsum(v)])
    cc = np.concatenate([[0.0], np.cumsum(c)])
    n = len(x)
    i = np.arange(n)
    if mode == "c":
        lo, hi = i - w // 2, i + w // 2 + 1
    elif mode == "p":
        lo, hi = i - w, i
    else:
        lo, hi = i + 1, i + w + 1
    lo, hi = np.clip(lo, 0, n), np.clip(hi, 0, n)
    s, k = cv[hi] - cv[lo], cc[hi] - cc[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(k > 0, s / k, np.nan).astype(np.float32)


def carry(valid, values, direction="f"):
    """Last (f) or next (b) value where valid, and the distance in frames to it."""
    n = len(valid)
    idx = np.where(valid, np.arange(n), -1 if direction == "f" else n)
    if direction == "f":
        idx = np.maximum.accumulate(idx)
        ok = idx >= 0
        dist = np.where(ok, np.arange(n) - idx, np.nan)
    else:
        idx = np.minimum.accumulate(idx[::-1])[::-1]
        ok = idx < n
        dist = np.where(ok, idx - np.arange(n), np.nan)
    out = np.full(n, np.nan, np.float32)
    out[ok] = values[idx[ok]]
    return out, dist.astype(np.float32)


def velocity(pos, k):
    """Central difference over +-k frames, m/s. pos[n, ..., 2]."""
    v = np.full(pos.shape, np.nan, np.float32)
    v[k:-k] = (pos[2 * k:] - pos[:-2 * k]) * (FPS / (2.0 * k))
    return v


def orientation(players, team, section):
    """Per-frame sign so that home attacks +x: +1 if home defends the -x goal in that section."""
    sign = np.ones(len(section), np.float32)
    x = players[:, :, 0]
    for s in np.unique(section):
        m = section == s
        with np.errstate(all="ignore"):
            hx = x[m][:, team == 0]
            ax = x[m][:, team == 1]
            h = np.nanmedian(np.nanmin(hx, 1) + np.nanmax(hx, 1))   # GK end dominates the sum
            a = np.nanmedian(np.nanmin(ax, 1) + np.nanmax(ax, 1))
        sign[m] = 1.0 if (h - a) < 0 else -1.0
    return sign


# ---------- features ----------

def build(players, team, ball, section):
    n = len(ball)
    o = orientation(players, team, section)
    P = players.copy()
    B = ball[:, :2].copy()
    P[..., 0] *= o[:, None]
    P[..., 1] *= o[:, None]
    B[:, 0] *= o
    B[:, 1] *= o
    H, A = team == 0, team == 1
    f = {}

    seen = np.isfinite(B[:, 0])
    f["b_seen"] = seen.astype(np.float32)
    f["b_x"] = B[:, 0]
    f["b_absy"] = np.abs(B[:, 1])
    out = np.maximum(np.abs(B[:, 0]) - HALF_L, np.abs(B[:, 1]) - HALF_W)
    f["b_out"] = np.where(seen, np.maximum(out, 0), np.nan).astype(np.float32)
    f["b_edge"] = np.where(seen, out, np.nan).astype(np.float32)        # <0 inside, distance to nearest line
    vb = velocity(B, 2)
    sb = np.linalg.norm(vb, axis=1)
    f["b_speed"] = sb
    vb_pre, vb_post = np.full_like(vb, np.nan), np.full_like(vb, np.nan)
    vb_pre[4:], vb_post[:-4] = vb[:-4], vb[4:]
    dot = (vb_pre * vb_post).sum(1)
    with np.errstate(all="ignore"):
        f["b_turn"] = (dot / (np.linalg.norm(vb_pre, axis=1) * np.linalg.norm(vb_post, axis=1))).astype(np.float32)
    f["b_dspeed"] = np.linalg.norm(vb_post, axis=1) - np.linalg.norm(vb_pre, axis=1)
    f["b_vx"] = vb[:, 0]

    # player-ball
    d = np.linalg.norm(P - B[:, None, :], axis=2)
    d = np.where(np.isfinite(d), d, np.inf)
    vp = velocity(P, 3)
    for name, m in (("H", H), ("A", A)):
        dm = np.where(m[None, :], d, np.inf)
        part = np.sort(dm, axis=1)[:, :2]
        d1 = np.where(np.isfinite(part[:, 0]), part[:, 0], np.nan)
        d2 = np.where(np.isfinite(part[:, 1]), part[:, 1], np.nan)
        f[f"d1{name}"], f[f"d2{name}"] = d1.astype(np.float32), d2.astype(np.float32)
        f[f"n2{name}"] = np.where(seen, (dm < 2).sum(1), np.nan).astype(np.float32)
        f[f"n5{name}"] = np.where(seen, (dm < 5).sum(1), np.nan).astype(np.float32)
        j = dm.argmin(1)
        vj = vp[np.arange(n), j]
        rel = np.linalg.norm(vb - vj, axis=1)
        f[f"relv{name}"] = np.where(np.isfinite(d1), rel, np.nan).astype(np.float32)
        f[f"spn{name}"] = np.where(np.isfinite(d1), np.linalg.norm(vj, axis=1), np.nan).astype(np.float32)
    f["d_diff"] = np.clip(f["d1H"] - f["d1A"], -20, 20)
    f["n2_diff"] = f["n2H"] - f["n2A"]
    f["relv_diff"] = f["relvH"] - f["relvA"]
    dmin = np.fmin(f["d1H"], f["d1A"])
    f["d_min"] = dmin

    # team shape and movement
    sp = np.linalg.norm(vp, axis=2)
    with np.errstate(all="ignore"):
        for name, m in (("H", H), ("A", A)):
            x, y = P[:, m, 0], P[:, m, 1]
            f[f"n{name}"] = np.isfinite(x).sum(1).astype(np.float32)
            f[f"cx{name}"] = np.nanmean(x, 1)
            f[f"sx{name}"] = np.nanstd(x, 1)
            f[f"sy{name}"] = np.nanstd(y, 1)
            q = np.nanpercentile(x, [10, 90], axis=1)
            f[f"len{name}"] = q[1] - q[0]
            f[f"vx{name}"] = np.nanmean(vp[:, m, 0], 1)
            f[f"sp{name}"] = np.nanmean(sp[:, m], 1)
            f[f"run{name}"] = np.nanmean(np.where(np.isfinite(sp[:, m]), sp[:, m] > 4.0, np.nan), 1)
            f[f"bx_rel{name}"] = B[:, 0] - f[f"cx{name}"]
        f["sp_med"] = np.nanmedian(sp, 1)
        f["sp_max"] = np.nanmax(sp, 1)
    f["cx_diff"] = f["cxH"] - f["cxA"]
    f["vx_diff"] = f["vxH"] + f["vxA"]          # both in home-attacks-+x frame: sum = net push toward away goal
    f["sp_diff"] = f["spH"] - f["spA"]
    f["sx_diff"] = f["sxH"] - f["sxA"]
    f["sy_diff"] = f["syH"] - f["syA"]

    # blackout context
    _, since = carry(seen, f["b_seen"], "f")
    _, until = carry(seen, f["b_seen"], "b")
    f["t_since"] = np.minimum(since / FPS, CAP_S)
    f["t_until"] = np.minimum(until / FPS, CAP_S)
    for k in ("d_diff", "b_x", "b_out"):
        f[f"last_{k}"], _ = carry(seen, f[k], "f")
        f[f"next_{k}"], _ = carry(seen, f[k], "b")
    close = seen & (dmin < 1.5)
    side = np.sign(f["d_diff"])
    f["last_close"], dl = carry(close, side, "f")
    f["next_close"], dn = carry(close, side, "b")
    f["t_last_close"] = np.minimum(dl / FPS, CAP_S)
    f["t_next_close"] = np.minimum(dn / FPS, CAP_S)

    # temporal context
    ctx = ["b_seen", "d_diff", "b_speed", "sp_med", "vx_diff", "cx_diff", "n2_diff", "b_out",
           "relv_diff", "sp_diff", "b_x", "sx_diff"]
    for k in ctx:
        for w in (13, 25, 51, 101, 201, 401):
            f[f"{k}_c{w}"] = nanroll(f[k], w, "c")
    for k in ("d_diff", "sp_med", "b_seen", "b_out", "vx_diff", "b_speed", "cx_diff"):
        for w in (25, 75):
            p, q = nanroll(f[k], w, "p"), nanroll(f[k], w, "f")
            f[f"{k}_p{w}"], f[f"{k}_f{w}"] = p, q
            f[f"{k}_j{w}"] = q - p
    if CONFIG.get("touch"):              # optional touch/handover features (kora/touch.py)
        from .touch import add
        add(f, vb)
    return pd.DataFrame({k: np.asarray(v, dtype=np.float32) for k, v in f.items()})


def mirror(players, team, ball):
    """Swap the roles of the two teams: rotate the pitch 180 degrees and exchange team ids."""
    P = players.copy(); P[..., :2] *= -1
    B = ball.copy(); B[:, :2] *= -1
    return P, (1 - team).astype(team.dtype), B
