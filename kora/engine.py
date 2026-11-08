"""Y20: other stage-1 classifiers as a variant switch ("clf"), kept out of kora/train.py so its stage-1
caches keep their keys (train.py's bytes are part of every stage-1 key).

kora.nested calls use(CONFIG) before any fit; kora.train.fit_classifier then builds the variant's
classifier, seeded with kora.train.SEED at call time, so ensemble members keep their own seeds.
"""
from . import train as _train


def make(settings, seed):
    s = dict(settings)
    kind = s.pop("kind")
    if kind != "lgbm":
        raise ValueError(f"unknown classifier kind {kind!r}")
    import lightgbm
    return lightgbm.LGBMClassifier(random_state=seed, deterministic=True, force_col_wise=True, verbose=-1, **s)


def use(config):
    """Point kora.train at the variant's classifier; a variant without "clf" keeps HistGradientBoosting."""
    if config.get("clf"):
        _train.classifier = lambda: make(config["clf"], _train.SEED)
