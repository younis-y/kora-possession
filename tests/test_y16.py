"""Checks for Y16 (stage 1 averaged over seed members): cache keys, kora.train globals, fits per member."""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from kora import nested, train, variant


class KeyTests(unittest.TestCase):
    def test_base_member_is_production_cache_and_members_differ(self):
        V = variant.VARIANTS
        self.assertEqual(variant.stage1_version(V["y16_seeds"]), variant.stage1_version(V["mixed"]))
        self.assertEqual(variant.stage1_version(V["y16_seeds_rand"]), variant.stage1_version(V["mixed_rand"]))
        keys = {variant.stage1_version({**V["mixed"], "seed": s}) for s in (2028, 2029)}
        self.assertEqual(len(keys), 2)
        self.assertNotIn(variant.stage1_version(V["mixed"]), keys)
        self.assertEqual(variant.feature_version({**V["mixed"], "seed": 2028}), variant.feature_version(V["mixed_s2028"]))

    def test_member_sets_and_restores_train_globals(self):
        before = train.SEED, train.CONFIG, train.FEATURE_VERSION
        with mock.patch.dict(nested.CONFIG, {"eval_rand": True, "ensemble": [2027, 2028]}):
            with nested.member(2028) as key:
                self.assertEqual(train.SEED, 2028)
                self.assertNotIn("eval_rand", train.CONFIG)
                self.assertNotIn("ensemble", train.CONFIG)
                self.assertTrue(key.startswith(train.FEATURE_VERSION))
        self.assertEqual((train.SEED, train.CONFIG, train.FEATURE_VERSION), before)


class MemberTests(unittest.TestCase):
    def test_pairs_predict_both_matches_and_cache(self):
        ms = ("a", "b", "c")
        X = {m: pd.DataFrame({"x": np.full(4, i, np.float32)}) for i, m in enumerate(ms)}
        fits = []

        class Stub:                                    # P(H) = number of training copies, so each fit is told apart
            def __init__(self, n): self.n = n
            def predict_proba(self, x): return np.tile([self.n, 0.0, 0.0], (len(x), 1))

        def fit(sets):
            fits.append(len(sets))
            return Stub(len(sets))

        with tempfile.TemporaryDirectory() as d, \
                mock.patch.object(nested, "MATCHES", ms), mock.patch.object(nested, "cache_dir", lambda: Path(d)), \
                mock.patch.object(nested, "copies", lambda m: ([m], None)), mock.patch.object(nested, "fit_classifier", fit):
            p = nested.member_stage1(2028, {"fv": X})["fv"]
            self.assertEqual(len(fits), 6)                                   # 3 singles + 3 pairs
            self.assertEqual(p[frozenset("a"), "a"][0, 0], 2)               # trained on b and c
            self.assertEqual(p[frozenset("ab"), "a"][0, 0], 1)              # trained on c only
            self.assertEqual(p[frozenset("ab"), "b"][0, 0], 1)
            self.assertEqual(len(p), 3 + 6)
            nested.member_stage1(2028, {"fv": X})
            self.assertEqual(len(fits), 6)                                   # second call reads the cache


if __name__ == "__main__":
    unittest.main()
