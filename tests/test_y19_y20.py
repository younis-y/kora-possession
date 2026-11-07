"""Checks for Y20 (stage-1 classifier as a variant switch): cache keys, the kora.train hook, and the
heavy-gap twin key. Y19's labeller is tested in test_skillcorner.py."""
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from kora import engine, train, variant


class Y20Tests(unittest.TestCase):
    def test_keys(self):
        V = variant.VARIANTS
        names = [f"y20_{k}" for k in variant.Y20]
        keys = {variant.stage1_version(V[n]) for n in names}
        self.assertEqual(len(keys), 4)
        self.assertNotIn(variant.stage1_version(V["mixed"]), keys)
        for n in names:   # same evaluation copies as the base, and the twin is the _rand variant's own key
            self.assertEqual(variant.feature_version(V[n]), variant.feature_version(V["mixed"]))
            self.assertEqual(variant.feature_version(V[n + "_rand"]), variant.feature_version(V["mixed_rand"]))
            self.assertEqual(variant.stage1_version({**V[n], "eval_rand": True}), variant.stage1_version(V[n + "_rand"]))
        self.assertEqual(variant.stage1_version({**V["y20_bag"], "ensemble": [2027, 2028]}),
                         variant.stage1_version(V["y20_bag"]))

    def test_use_points_train_at_lightgbm_with_the_seed_at_call_time(self):
        with mock.patch.object(train, "classifier", train.classifier), mock.patch.object(train, "SEED", 2027):
            engine.use({})
            self.assertEqual(type(train.classifier()).__name__, "HistGradientBoostingClassifier")
            engine.use(variant.VARIANTS["y20_bag"])
            train.SEED = 2029
            clf = train.classifier()
            self.assertEqual(type(clf).__name__, "LGBMClassifier")
            self.assertEqual(clf.get_params()["random_state"], 2029)
            self.assertEqual(clf.get_params()["colsample_bytree"], 0.5)

    def test_fit_classifier_with_balance_weights(self):
        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.normal(size=(600, 3)), columns=list("abc"))
        y = np.repeat([0, 1, 2], 200).astype(np.int64)
        small = {**variant.LGBM, "n_estimators": 5, "min_child_samples": 5}
        with mock.patch.object(train, "classifier", train.classifier), \
                mock.patch.object(train, "CONFIG", {"balance": 0.5, "clf": small}):
            engine.use(train.CONFIG)
            p = train.fit_classifier([(X, y)]).predict_proba(X.to_numpy())
        self.assertEqual(p.shape, (600, 3))
        np.testing.assert_allclose(p.sum(1), 1, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
