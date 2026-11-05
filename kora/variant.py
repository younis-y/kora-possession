"""Named pipeline variants and the cache fingerprints that keep their files apart.

A variant is a small dict of switches. Pick one with `--variant NAME` on any `python -m kora.*` command,
or KORA_VARIANT=NAME. PRODUCTION is what build.sh and kora.run use; it changes only when a variant
passes the keep rule in RESEARCH.md (paired nested CV, mean up and at least 5 of 7 matches better).

Feature caches are keyed by a hash of the code and data that shape features (features.py, degrade.py,
models/degrade_external.json, and the feature switches), so an edited file can never be scored from a
stale cache. Stage-1 caches add the training switches.
"""
import hashlib
import itertools
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# RESEARCH.md V5 round 1: preserve B0 grid order before appending wider settings.
V5_HSMM_GRID = list(itertools.product((2, 4, 8), (8, 16), (2, 3, 4), (2, 3, 4)))
V5_HSMM_GRID += [
    setting for setting in itertools.product((1, 2, 4, 8), (8, 16, 32), (2, 3, 4), (2, 3, 4, 6, 8))
    if setting not in V5_HSMM_GRID
]

VARIANTS = {
    "b0": {},                                         # compliant baseline: rebased degradation, v3 features
    "b0_w4": {"decoder": {"inner_wins": 4}},          # exploratory: HSMM kept on 4 of 6 inner wins (Y5)
    "touch": {"touch": True},                         # R1: touch and handover features
    "touch_bal": {"touch": True, "balance": 0.5},     # R2: R1 + possession-balanced sample weights
    # RESEARCH.md V7 round 1: give handover detector targets priority over live/dead windows.
    "v7_ha_priority": {"touch": True, "balance": 0.5, "detector": {"ha_priority": True}},
    # J6 on top of production (its registration: "if R1/R2 change PRODUCTION, the same switches go on top").
    # b0_rand keeps its registered name; it is production + eval_rand.
    "mixed": {"touch": True, "balance": 0.5, "mixed": True},          # odd training copies get label-free gaps
    "b0_rand": {"touch": True, "balance": 0.5, "eval_rand": True},    # random-gap test: evaluation copies too
    "mixed_rand": {"touch": True, "balance": 0.5, "mixed": True, "eval_rand": True},
    "v5_wide": {"decoder": {"hsmm_grid": V5_HSMM_GRID}},
    # RESEARCH.md V6 round 1: B0 settings with handover-only boundary bonuses.
    "v6_ha_only": {"decoder": {"hsmm_grid": [s + (True,) for s in V5_HSMM_GRID[:54]]}},
    # RESEARCH.md Y7/Y8 (pre-registered 2026-11-05), each alone on top of touch_bal.
    "y7a_refine": {"touch": True, "balance": 0.5, "decoder": {"refine": True}},       # stage-2 refiner
    "y7c_wide": {"touch": True, "balance": 0.5, "decoder": {"hsmm_grid": V5_HSMM_GRID}},
    "y8a_dwell": {"touch": True, "balance": 0.5, "dwell": True},                    # kora/dwell.py
    "y8b_bal1": {"touch": True, "balance": 1.0},
    # Y7 confirmation: the same pipeline with seed 2028 for the degraded copies and stage 1.
    "mixed_s2028": {"touch": True, "balance": 0.5, "mixed": True, "seed": 2028},
}
PRODUCTION = "touch_bal"                              # Y6: R2 passed against B0 and R1
FEATURE_KEYS = ("touch", "mixed", "eval_rand")
TRAIN_KEYS = ("balance",)

# Decoder selection in kora.nested. A variant overrides any key under "decoder"; nothing here touches the
# stage-1 caches. Grids are lists of tuples: plain (alpha, cost_ha, cost_d), hsmm (cost_ha, cost_d, w_dur,
# beta); None means the default grids (kora.train GRID_*, kora.boundary GRID). refine: decode stage-2
# probabilities (kora.nested.fit_refiner, Y7a) instead of stage 1's.
DECODER = {"inner_wins": 5, "plain_grid": None, "hsmm_grid": None, "refine": False}
DETECTOR = {"ha_priority": False}


def _name():
    if "--variant" in sys.argv:
        i = sys.argv.index("--variant")
        if i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return os.environ.get("KORA_VARIANT", PRODUCTION)


NAME = _name()
if NAME not in VARIANTS:
    raise SystemExit(f"unknown variant {NAME!r}; known: {', '.join(VARIANTS)}")
CONFIG = VARIANTS[NAME]


def _hash(parts):
    h = hashlib.sha256()
    for part in parts:
        h.update(part if isinstance(part, bytes) else part.encode())
    return h.hexdigest()[:10]


def feature_version(config=CONFIG):
    files = [ROOT / "kora" / "features.py", ROOT / "kora" / "degrade.py", ROOT / "models" / "degrade_external.json"]
    if config.get("touch"):
        files.append(ROOT / "kora" / "touch.py")
    if config.get("mixed") or config.get("eval_rand"):
        files.append(ROOT / "models" / "degrade_external_all.json")
    switches = json.dumps({k: config.get(k) for k in FEATURE_KEYS}, sort_keys=True)
    if config.get("seed"):             # added later: keyed only when set, so older caches keep their names
        switches += f"seed{config['seed']}"
    return "f" + _hash([f.read_bytes() for f in files] + [switches])


def stage1_version(config=CONFIG):
    switches = json.dumps({k: config.get(k) for k in TRAIN_KEYS}, sort_keys=True)
    parts = [(ROOT / "kora" / "train.py").read_bytes(), switches]
    if config.get("dwell"):            # Y8a, added later: keyed only when on, so older caches keep their names
        parts.append((ROOT / "kora" / "dwell.py").read_bytes())
    return feature_version(config) + "s" + _hash(parts)


def decoder(config=CONFIG):
    return {**DECODER, **config.get("decoder", {})}


def detector(config=CONFIG):
    return {**DETECTOR, **config.get("detector", {})}
