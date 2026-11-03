"""Named pipeline variants and the cache fingerprints that keep their files apart.

A variant is a small dict of switches. Pick one with `--variant NAME` on any `python -m kora.*` command,
or KORA_VARIANT=NAME. PRODUCTION is what build.sh and kora.run use; it changes only when a variant
passes the keep rule in RESEARCH.md (paired nested CV, mean up and at least 5 of 7 matches better).

Feature caches are keyed by a hash of the code and data that shape features (features.py, degrade.py,
models/degrade_external.json, and the feature switches), so an edited file can never be scored from a
stale cache. Stage-1 caches add the training switches.
"""
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

VARIANTS = {
    "b0": {},                                         # compliant baseline: rebased degradation, v3 features
    "b0_w4": {"decoder": {"inner_wins": 4}},          # exploratory: HSMM kept on 4 of 6 inner wins (Y5)
    "touch": {"touch": True},                         # R1: touch and handover features
    "touch_bal": {"touch": True, "balance": 0.5},     # R2: R1 + possession-balanced sample weights
}
PRODUCTION = "b0"
FEATURE_KEYS = ("touch",)
TRAIN_KEYS = ("balance",)

# Decoder selection in kora.nested. A variant overrides any key under "decoder"; nothing here touches the
# stage-1 caches. Grids are lists of tuples: plain (alpha, cost_ha, cost_d), hsmm (cost_ha, cost_d, w_dur,
# beta); None means the default grids (kora.train GRID_*, kora.boundary GRID).
DECODER = {"inner_wins": 5, "plain_grid": None, "hsmm_grid": None}


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
    switches = json.dumps({k: config.get(k) for k in FEATURE_KEYS}, sort_keys=True)
    return "f" + _hash([f.read_bytes() for f in files] + [switches])


def stage1_version(config=CONFIG):
    switches = json.dumps({k: config.get(k) for k in TRAIN_KEYS}, sort_keys=True)
    return feature_version(config) + "s" + _hash([(ROOT / "kora" / "train.py").read_bytes(), switches])


def decoder(config=CONFIG):
    return {**DECODER, **config.get("decoder", {})}
