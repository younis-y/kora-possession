"""Make clean, corrected tracking look like the challenge's automatic tracking.

The training matches are official corrected tracking: every player, the ball on every frame. The
challenge match is automatic tracking, and its gaps are not random. Measured on the challenge match
with no labels (notes/2026-10-31-missingness.md): within each band of median player speed, the
share of frames with the ball missing tracks the share of dead-ball frames in the training matches
at the same speed. A two-component mixture fit gives

    P(ball missing | dead) ~ 0.80        P(ball missing | alive) ~ 0.04

and frames with fewer than 15 tracked players sit almost entirely at walking pace (camera close-ups
during stoppages). Positions themselves are as smooth as the corrected data (jitter 0.05 m vs 0.09 m,
same median player speed), so no positional noise is added.

Each training copy draws its knobs from ranges around those estimates (domain randomisation), so the
model learns how much a missing ball means without being tied to one exact value.
"""
import numpy as np

FPS = 25


def stretches(mask):
    m = np.concatenate([[False], mask, [False]]).astype(np.int8)
    d = np.diff(m)
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def degrade(players, ball_xyz, label, rng):
    n, P, _ = players.shape
    out_p = players.copy()
    out_b = ball_xyz[:, :2].copy()
    dead = label == 2
    missing = np.zeros(n, bool)

    # ball during stoppages: one blackout per dead stretch, mostly covering it
    q = rng.uniform(0.80, 0.97)
    for s, e in stretches(dead):
        if rng.random() < q:
            a = s + int(rng.uniform(-1.0, 2.0) * FPS)
            b = e - int(rng.uniform(0.0, 3.0) * FPS)
            missing[max(a, 0):max(b, 0)] = True
    # ball in play: short occasional blackouts
    alive_idx = np.flatnonzero(~dead)
    share = rng.uniform(0.02, 0.06)
    mean_len = rng.uniform(1.0, 4.0) * FPS
    k = int(share * len(alive_idx) / mean_len)
    for start in rng.choice(alive_idx, size=k, replace=False):
        missing[start:start + int(rng.exponential(mean_len)) + 1] = True
    out_b[missing] = np.nan

    # players: tracks drop out now and then
    vis_mean = rng.uniform(30, 150) * FPS
    gap_mean = rng.uniform(1, 6) * FPS
    for j in range(P):
        t = -int(rng.integers(0, int(vis_mean)))
        while t < n:
            t += int(rng.exponential(vis_mean)) + 1
            g = int(rng.exponential(gap_mean)) + 1
            out_p[max(t, 0):max(t + g, 0), j] = np.nan
            t += g
    # camera close-ups in some stoppages: players far from the ball's last position vanish
    zoom = rng.uniform(0.15, 0.4)
    for s, e in stretches(dead):
        if e - s > 4 * FPS and rng.random() < zoom:
            c = np.nanmean(ball_xyz[s:e, :2], 0)
            r = rng.uniform(15, 30)
            far = np.linalg.norm(players[s:e] - c[None, None, :], axis=2) > r
            seg = out_p[s:e]
            seg[far] = np.nan
    return out_p, out_b
