"""Touch and handover features (variant switch "touch", RESEARCH.md R1).

spellF1 counts every possession once, and the possessions we miss are mostly short ones that start
with a touch. A touch is a sudden change of ball velocity next to a player, so:

- b_acc     |v(t+2) - v(t-2)| of the ball, m/s over 0.16 s (velocity itself is over +-2 frames)
- b_turn2   cosine between the ball's velocity 2 frames before and 2 frames after
- touchH/A  b_acc * exp(-distance of that team's nearest player / 1 m)
- events    frames where max(touchH, touchA) first exceeds TOUCH_DV or the stronger team changes
            (a tie is a contested touch, side 0)
- last/next touch team and time, counts of each team's touches in past and future windows, and the
  signed length of the current same-team touch chain (the "two consecutive passes" control proxy).

TOUCH_DV = 2.7 m/s is the 95th percentile of the ball's velocity change in free flight (no player
within 5 m) during live play in three IDSSE matches: external data, no possession labels used.
"""
import numpy as np

from .features import CAP_S, FPS, carry, nanroll

TOUCH_DV = 2.7
REACH_M = 1.0


def chain(events, side):
    """Signed count of consecutive same-team touches up to and including each frame (0 before any)."""
    out = np.zeros(len(events), np.float32)
    count, last = 0, 0.0
    idx = np.flatnonzero(events)
    for a, b in zip(idx, np.r_[idx[1:], len(events)]):
        count = count + 1 if side[a] == last else 1
        last = side[a]
        out[a:b] = count * last
    return out


def add(f, vb):
    """Add touch features in place to the feature dict f (needs d1H/d1A); vb is ball velocity, m/s."""
    n = len(vb)
    dv = np.full(n, np.nan, np.float32)
    dv[2:-2] = np.linalg.norm(vb[4:] - vb[:-4], axis=1)
    f["b_acc"] = dv
    pre, post = np.full_like(vb, np.nan), np.full_like(vb, np.nan)
    pre[2:], post[:-2] = vb[:-2], vb[2:]
    with np.errstate(all="ignore"):
        f["b_turn2"] = ((pre * post).sum(1) / (np.linalg.norm(pre, axis=1) * np.linalg.norm(post, axis=1))
                        ).astype(np.float32)
    for name in ("H", "A"):
        f[f"touch{name}"] = (dv * np.exp(-f[f"d1{name}"] / REACH_M)).astype(np.float32)
    f["touch_diff"] = f["touchH"] - f["touchA"]

    th, ta = np.nan_to_num(f["touchH"]), np.nan_to_num(f["touchA"])
    strong = np.maximum(th, ta) > TOUCH_DV
    side = np.sign(th - ta).astype(np.float32)                        # +1 home, -1 away, 0 contested
    events = strong & (~np.r_[False, strong[:-1]] | (side != np.r_[0, side[:-1]]))
    f["last_touch"], dl = carry(events, side, "f")
    f["next_touch"], dn = carry(events, side, "b")
    f["t_last_touch"] = np.minimum(dl / FPS, CAP_S)
    f["t_next_touch"] = np.minimum(dn / FPS, CAP_S)
    f["touch_chain"] = chain(events, side)
    eh = (events & (side > 0)).astype(np.float32)
    ea = (events & (side < 0)).astype(np.float32)
    for w in (25, 50, 100):
        for mode in ("p", "f"):
            nh, na = nanroll(eh, w, mode) * w, nanroll(ea, w, mode) * w
            f[f"touches_{mode}{w}"] = nh + na
            f[f"touches_diff_{mode}{w}"] = nh - na
    for k in ("d_diff", "touchH", "touchA"):
        for w in (5, 9):
            f[f"{k}_c{w}"] = nanroll(f[k], w, "c")
