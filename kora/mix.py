"""Y29: more training copies, or a larger share of them with label-free ball gaps.

`mixed` trains on two degraded copies per match, each mirrored: copy 0 keeps the camera-on ball mask
(the ball goes missing mostly in stoppages), copy 1 loses the ball at random (kora.degrade.random_ball).
A Y29 variant keeps those cached copies, optionally drops copy 0, and appends extra copies drawn with
their own generators, so the evaluation copies and every other cache stay exactly as in `mixed`:

    {"extra": ["cam", "rand"]}                   4 copies, half with random gaps
    {"extra": ["rand", "rand"]}                  4 copies, three quarters
    {"drop_cam": True, "extra": ["rand"]}        2 copies, all

kora.nested swaps the training copies in after loading them (train.py unchanged, keys kept).
"""
from pathlib import Path

import numpy as np
import pandas as pd

from .data import cache_dir
from .variant import _hash


def extra_copy(match, j, kind, train):
    """Extra copy j of `match` and its mirror: degraded with its own generator, the ball mask replaced by
    random gaps when kind is "rand". Cached in data/cache under the feature version."""
    key = train.FEATURE_VERSION + _hash([Path(__file__).read_bytes(), str(train.SEED)])
    path = cache_dir() / f"mix_{key}_{match}_{j}_{kind}.npz"
    if path.exists():
        z = np.load(path, allow_pickle=False)
        cols = list(z["columns"])
        return [(pd.DataFrame(z["X0"], columns=cols), z["y0"]), (pd.DataFrame(z["X1"], columns=cols), z["y1"])]
    d = train.load(match)
    i = list(train.MATCHES).index(match)
    p, b = train.degrade(d["players"], d["ball"], d["label"], np.random.default_rng([train.SEED, i, 500 + j]))
    if kind == "rand":
        b = train.random_ball(d["ball"], d["half"], np.random.default_rng([train.SEED, i, 600 + j]))
    X0 = train.build(p, d["team"], b, d["half"])
    pm_, tm_, bm_ = train.mirror(p, d["team"], b)
    y1 = d["label"].copy(); y1[y1 < 2] = 1 - y1[y1 < 2]
    X1 = train.build(pm_, tm_, bm_, d["half"])
    np.savez(path, columns=np.array(X0.columns, dtype="U64"), X0=X0.to_numpy(), y0=d["label"], X1=X1.to_numpy(), y1=y1)
    return [(X0, d["label"]), (X1, y1)]


def training_sets(match, base, cfg, make=None):
    """`base` is `mixed`'s training list for `match` (copy 0, its mirror, copy 1, its mirror)."""
    from . import train
    make = make or (lambda j, kind: extra_copy(match, j, kind, train))
    out = list(base[2:] if cfg.get("drop_cam") else base)
    for j, kind in enumerate(cfg.get("extra", [])):
        out += make(j, kind)
    return out


def apply(data, cfg):
    return {m: (training_sets(m, tr, cfg), ev) for m, (tr, ev) in data.items()}
