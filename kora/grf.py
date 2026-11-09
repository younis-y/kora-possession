"""Y28: synthetic training games from Google Research Football (Radke and Tilbury, AAMAS 2025).

3,000 simulated GRF games of 3,000 steps at 10 Hz (5 minutes of play each), 22 players and the ball on every
step, and an ON_BALL flag per player (github.com/Dtradke/grf_tracking_data, pinned below). The repository
names no licence; the owner chose to use it (RESEARCH.md Y28). No data about the challenge match is used.

Conversion: steps are interpolated to 25 fps; x * 52.5 and y * 34 / 0.42 give metres from the centre spot.
Labels: the team of the last player on the ball (0 = team 0 = H); dead (D) from the ball leaving the pitch,
or entering a goal, until the next touch on the pitch, and before the kick-off touch. GRF has few stoppages,
so D comes almost only from IDSSE. Each game becomes one section and gets the same degraded and mirrored
training copies as an IDSSE match (kora.train._copies); stage 1 sees every `stride`-th frame of them.

    python -m kora.grf 150      # download the first 150 games (3 per file) if missing
"""
import csv
import io
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from .data import cache_dir, data_dir
from .variant import _hash

COMMIT = "fe823ab08aa6f93047bdda85031d4065743cc862"
INDEX = f"https://raw.githubusercontent.com/Dtradke/grf_tracking_data/{COMMIT}/scripts/names_ids_links.csv"
STEP_FPS, FPS = 10, 25
SX, SY = 52.5, 34.0 / 0.42


def grf_dir():
    d = data_dir() / "external" / "grf"
    d.mkdir(parents=True, exist_ok=True)
    return d


def file_of(game):
    return grf_dir() / f"gfootball_results_games_{game // 3 * 3}-{game // 3 * 3 + 2}.csv"


def download(games):
    rows = list(csv.DictReader(io.StringIO(urllib.request.urlopen(INDEX).read().decode())))
    for r in rows[:(games + 2) // 3]:
        p = grf_dir() / r["Filename"]
        if not p.exists():
            urllib.request.urlretrieve(r["Download Link"], p.with_suffix(".part"))
            p.with_suffix(".part").rename(p)


def labels(ball, owner, goals):
    """Per-step 0/1 (team of the last touch) or 2 (dead). ball[T, 2] normalised, owner[T] team or -1."""
    out = (np.abs(ball[:, 0]) > 1.0) | (np.abs(ball[:, 1]) > 0.42)
    for g in goals:                        # the ball sits in the net (|x| <= 1) before a goal's restart
        j = g
        while j > 0 and abs(ball[j - 1, 0]) > 0.9 and owner[j - 1] < 0:
            j -= 1
        out[j:g + 1] = True
    lab = np.full(len(ball), 2, np.int8)
    team, dead = -1, True
    for t in range(len(ball)):
        if out[t]:
            dead = True
        elif owner[t] >= 0:
            team, dead = owner[t], False
        lab[t] = 2 if dead or team < 0 else team
    return lab


def load(game):
    """The same dict as kora.external.load, at 25 fps, one section (half = 1)."""
    d = pd.read_csv(file_of(game))
    d = d[d.GAME_ID == game].sort_values(["PLAYER_ID", "TIMESTEP"])
    T = d.TIMESTEP.nunique()
    b = d[d.PLAYER_ID == -1]
    p = d[d.PLAYER_ID != -1]
    ids = np.sort(p.PLAYER_ID.unique())
    xy = p[["LOCATION_X", "LOCATION_Y"]].to_numpy().reshape(len(ids), T, 2).transpose(1, 0, 2) * [SX, SY]
    team = p.groupby("PLAYER_ID").TEAM_ID.first().loc[ids].to_numpy().astype(np.int8)
    bn = b[["LOCATION_X", "LOCATION_Y"]].to_numpy()
    on = p[p.ON_BALL].groupby("TIMESTEP").TEAM_ID.first()
    owner = np.full(T, -1); owner[on.index.to_numpy()] = on.to_numpy()
    score = d.groupby("TIMESTEP")[["TEAM_0_SCORE", "TEAM_1_SCORE"]].first().sum(axis=1).to_numpy()
    lab = labels(bn, owner, np.flatnonzero(np.diff(score) > 0) + 1)
    t = np.arange((T - 1) * FPS // STEP_FPS + 1) * STEP_FPS / FPS          # step units
    interp = lambda a: np.stack([np.interp(t, np.arange(T), a[:, k]) for k in range(a.shape[1])], 1)
    ball = interp(np.c_[bn * [SX, SY], b.LOCATION_Z.to_numpy()]).astype(np.float32)
    players = np.stack([interp(xy[:, j]) for j in range(len(ids))], 1).astype(np.float32)
    return {"half": np.ones(len(t), np.int8), "players": players, "team": team, "ball": ball,
            "label": lab[np.floor(t + 1e-9).astype(int)]}


def rows(cfg, lam):
    """Stage-1 rows (X, y, weight) of the first cfg["games"] games, every cfg["stride"]-th frame of each
    training copy; weights as kora.train.fit_classifier (balance lam, or ones). Cached per game."""
    from . import train
    key = train.FEATURE_VERSION + _hash([Path(__file__).read_bytes(), str(cfg["stride"]), str(train.SEED)])
    X, y, w = [], [], []
    for g in range(cfg["games"]):
        path = cache_dir() / f"grf_{key}_{g}.npz"
        if not path.exists():
            sets = copies(g, train)
            np.savez(path, X=np.concatenate([x.to_numpy()[::cfg["stride"]] for x, _ in sets]),
                     y=np.concatenate([lab[::cfg["stride"]] for _, lab in sets]),
                     w=np.concatenate([train.balance_weights(lab, lam or 0)[::cfg["stride"]] for _, lab in sets]))
        z = np.load(path)
        X.append(z["X"]); y.append(z["y"]); w.append(z["w"])
    return np.concatenate(X), np.concatenate(y), np.concatenate(w)


def fit_classifier(cfg):
    """kora.train.fit_classifier with the GRF rows appended; kora.nested swaps it in, so train.py and every
    other stage-1 cache key stay unchanged."""
    from . import train

    def fit(train_sets):
        X, y = train.stack(train_sets)
        lam = train.CONFIG.get("balance")
        Xg, yg, wg = rows(cfg, lam)
        w = np.concatenate([train.balance_weights(lab, lam)[::train.SUBSAMPLE] for _, lab in train_sets]) if lam else None
        return train.classifier().fit(np.concatenate([X, Xg]), np.concatenate([y, yg]),
                                      sample_weight=None if w is None else np.concatenate([w, wg]))
    return fit


def copies(game, train):
    """kora.train._copies for a GRF game: degraded copies, odd ones with label-free gaps (mixed), each mirrored."""
    d = load(game)
    rng = np.random.default_rng([train.SEED, 1000 + game])
    out = []
    for i in range(train.TRAIN_COPIES):
        p, b = train.degrade(d["players"], d["ball"], d["label"], rng)
        if train.CONFIG.get("mixed") and i % 2:
            b = train.random_ball(d["ball"], d["half"], np.random.default_rng([train.SEED, 1000 + game, i]))
        out.append((train.build(p, d["team"], b, d["half"]), d["label"]))
        pm_, tm_, bm_ = train.mirror(p, d["team"], b)
        yy = d["label"].copy(); yy[yy < 2] = 1 - yy[yy < 2]
        out.append((train.build(pm_, tm_, bm_, d["half"]), yy))
    return out


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 150
    download(n)
    for g in (0, n - 1):
        d = load(g)
        lab = d["label"]
        print(g, d["players"].shape, f"H {np.mean(lab == 0):.2f} A {np.mean(lab == 1):.2f} D {np.mean(lab == 2):.3f}",
              f"runs {np.sum(np.diff(lab) != 0) + 1}")
