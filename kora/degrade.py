"""Make clean, corrected tracking look like automatic tracking, using external measurements only.

The training matches (IDSSE) are corrected tracking: every player and the ball on every frame. The
challenge match is automatic tracking, which loses the ball and players. Every number below comes from
models/degrade_external.json, written by kora/missingness.py from the SkillCorner open data (automatic
broadcast tracking with real detection gaps) and the IDSSE labels. No challenge data is used; see that
module for the method and the provenance of each value.

Each call draws one severity s ~ U(0, 1) and scales every process by it, so training copies span the
two external extremes: clean corrected tracking (s = 0) and broadcast automatic tracking (s = 1).
"""
import json
from pathlib import Path

import numpy as np

from .features import FPS

MODELS = Path(__file__).resolve().parent.parent / "models"
PARAMS = json.loads((MODELS / "degrade_external.json").read_text())
# J6 was registered on the every-frame estimate (40% missing); camera on (Y5) would leave it 5.7%.
RANDOM_GAPS = json.loads((MODELS / "degrade_external_all.json").read_text())


def stretches(mask):
    m = np.concatenate([[False], mask, [False]]).astype(np.int8)
    d = np.diff(m)
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def draw(quantiles, rng):
    """One value from an empirical distribution stored as evenly spaced quantiles."""
    q = np.asarray(quantiles, float)
    return float(np.interp(rng.random(), np.linspace(0, 1, len(q)), q))


def random_ball(ball_xyz, half, rng):
    """Ball hidden by external-length gaps placed without labels, up to the external missing share
    of each half (RESEARCH.md J6), both from the every-frame estimate it was registered with. Overlaps are allowed, so the realised share is a little lower."""
    out = ball_xyz[:, :2].copy()
    for h in np.unique(half):
        idx = np.flatnonzero(half == h)
        budget = RANDOM_GAPS["missing_share"] * len(idx)
        while budget > 0:
            start = idx[rng.integers(len(idx))]
            length = max(int(draw(RANDOM_GAPS["gap_s_quantiles"], rng) * FPS), 1)
            out[start:min(start + length, idx[-1] + 1)] = np.nan
            budget -= length
    return out


def degrade(players, ball_xyz, label, rng, severity=None):
    p = PARAMS
    n, P, _ = players.shape
    s = rng.random() if severity is None else severity
    out_p = players.copy()
    out_b = ball_xyz[:, :2].copy()
    dead = label == 2
    missing = np.zeros(n, bool)

    # ball during stoppages: one blackout per dead stretch, offsets from the external stoppage proxy
    for a, b in stretches(dead):
        if rng.random() < s * p["p_miss_dead"]:
            lo = a + int(draw(p["dead_start_offset_s_quantiles"], rng) * FPS)
            hi = b - int(draw(p["dead_end_offset_s_quantiles"], rng) * FPS)
            missing[lo:max(hi, lo)] = True
    # ball in play: gaps with external live-gap lengths until the external share is reached
    alive_idx = np.flatnonzero(~dead)
    budget = s * p["p_miss_alive"] * len(alive_idx)
    while budget > 0 and len(alive_idx):
        start = alive_idx[rng.integers(len(alive_idx))]
        length = max(int(draw(p["live_gap_s_quantiles"], rng) * FPS), 1)
        missing[start:start + length] = True
        budget -= length
    out_b[missing] = np.nan

    # players: alternating visible and lost stretches; each lost stretch kept with probability s
    vis_mean = p["player_visible_s_mean"] * FPS
    for j in range(P):
        t = -int(rng.integers(0, max(int(vis_mean), 1)))
        while t < n:
            t += int(rng.exponential(vis_mean)) + 1
            g = int(draw(p["player_gap_s_quantiles"], rng) * FPS) + 1
            if rng.random() < s:
                out_p[max(t, 0):max(t + g, 0), j] = np.nan
            t += g
    # camera close-ups in some long stoppages: players far from the ball vanish
    radii = p["closeup_radius_m_quantiles"]
    for a, b in stretches(dead):
        if radii and b - a > p["closeup_min_s"] * FPS and rng.random() < s * p["closeup_prob"]:
            c = np.nanmean(ball_xyz[a:b, :2], 0)
            far = np.linalg.norm(players[a:b] - c[None, None, :], axis=2) > draw(radii, rng)
            out_p[a:b][far] = np.nan
    # measurement noise on observed positions
    sd = s * p["position_noise_m"]
    if sd > 0:
        out_p += rng.normal(0, sd, out_p.shape).astype(out_p.dtype)
    return out_p, out_b
