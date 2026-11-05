"""Synthetic checks for the report-only heavy-gap evaluation (RESEARCH.md V8)."""
import copy
import hashlib
import unittest
from unittest import mock

import numpy as np

from kora import degrade as dg, train as tr, variant


STRESS_PAIRS = (("touch_bal", "touch_bal_stress"), ("mixed", "mixed_stress"))

def scene(n=1200):
    rng = np.random.default_rng(11)
    return {
        "players": rng.uniform(-30, 30, (n, 4, 2)).astype(np.float32),
        "ball": rng.uniform(-10, 10, (n, 3)).astype(np.float32),
        "label": np.resize(np.repeat([0, 1, 2], 200), n).astype(np.int8),
        "team": np.array([0, 0, 1, 1], np.int8),
        "half": np.repeat([1, 2], n // 2).astype(np.int8),
    }


def raw_features(players, team, ball, half):
    """Keep every degraded coordinate visible to the assertions, without feature computation."""
    return np.concatenate([players.reshape(len(ball), -1), ball], axis=1)


class StressTests(unittest.TestCase):
    def assert_bytes_equal(self, actual, expected):
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertEqual(actual.tobytes(), expected.tobytes())

    def copies(self, config, seed):
        d = scene()
        draws = []

        def degrade(players, ball, label, rng, **kwargs):
            draws.append((copy.deepcopy(rng.bit_generator.state), kwargs))
            return dg.degrade(players, ball, label, rng, **kwargs)

        with mock.patch.multiple(tr, CONFIG=config, SEED=seed, MATCHES={"synthetic": None}), \
                mock.patch.object(tr, "load", return_value=d) as load, \
                mock.patch.object(tr, "build", side_effect=raw_features), \
                mock.patch.object(tr, "degrade", side_effect=degrade):
            result = tr._copies("synthetic")
        load.assert_called_once_with("synthetic")
        return result, draws

    def test_default_parameters_preserve_outputs_and_rng_state(self):
        d = scene()
        for seed in (0, 1, 7, 31, 2027):
            for severity in (None, 0.0, 1.0):
                with self.subTest(seed=seed, severity=severity):
                    rng, explicit_rng = np.random.default_rng(seed), np.random.default_rng(seed)
                    args = d["players"], d["ball"], d["label"]
                    actual = dg.degrade(*args, rng, severity=severity)
                    expected = dg.degrade(*args, explicit_rng, severity=severity, params=dg.PARAMS)
                    for a, b in zip(actual, expected):
                        self.assert_bytes_equal(a, b)
                    self.assertEqual(rng.bit_generator.state, explicit_rng.bit_generator.state)

    def test_stress_preserves_training_copies_and_uses_full_external_evaluation(self):
        for base_name, stress_name in STRESS_PAIRS:
            base = variant.VARIANTS[base_name]
            stress = variant.VARIANTS[stress_name]
            for seed in (1, 7, 2027):
                with self.subTest(variant=stress_name, seed=seed):
                    (train, _), draws = self.copies(base, seed)
                    (stress_train, evaluation), stress_draws = self.copies(stress, seed)
                    self.assertEqual(len(train), 2 * tr.TRAIN_COPIES)
                    for actual, expected in zip(stress_train, train):
                        for a, b in zip(actual, expected):
                            self.assert_bytes_equal(a, b)
                    # Training draw order and the RNG entering evaluation stay exactly the same.
                    self.assertEqual(stress_draws[:-1], draws[:-1])
                    self.assertTrue(all(kwargs == {} for _, kwargs in stress_draws[:-1]))
                    self.assertEqual(stress_draws[-1][0], draws[-1][0])
                    self.assertEqual(stress_draws[-1][1]["severity"], 1.0)
                    self.assertIs(stress_draws[-1][1]["params"], dg.RANDOM_GAPS)

                    d = scene()
                    rng = np.random.default_rng()
                    rng.bit_generator.state = stress_draws[-1][0]
                    p, b = dg.degrade(d["players"], d["ball"], d["label"], rng,
                                      severity=1.0, params=dg.RANDOM_GAPS)
                    self.assert_bytes_equal(evaluation[0], raw_features(p, d["team"], b, d["half"]))
                    self.assert_bytes_equal(evaluation[1], d["label"])
                    self.assert_bytes_equal(evaluation[2], d["half"])

    def test_evaluation_switches_are_mutually_exclusive_before_loading(self):
        with mock.patch.object(tr, "CONFIG", {"eval_stress": True, "eval_rand": True}), \
                mock.patch.object(tr, "load") as load:
            with self.assertRaisesRegex(ValueError, "mutually exclusive"):
                tr._copies("synthetic")
        load.assert_not_called()

    def test_existing_variants_keep_byte_identical_copies(self):
        # Captured from the original _copies on this synthetic scene, seed 2027.
        baseline = "f06403d24f75c5cbf4e7bbfc707dc63d8981c9bd3b1eb252207d541eb8726a1f"
        switched = {
            "mixed": "49cff84d35823bf37a4c405736c997e1b6a8539e090edd4bc9bf7639de3f3ad7",
            "b0_rand": "fbfea030a43717fde8291f33410c222ae44ffd64ccaebc7be6f7e1bc722420be",
            "mixed_rand": "9d8534dc2398c835eff557c5fa7bfba85f57f577a85c5fd4c0a5760fcb7d5d48",
        }
        for name, config in variant.VARIANTS.items():
            if name in dict(STRESS_PAIRS).values() or "seed" in config:
                continue  # stress variants change evaluation copies, seed variants change every draw
            with self.subTest(variant=name):
                (train, evaluation), _ = self.copies(config, 2027)
                digest = hashlib.sha256()
                for pair in train:
                    for array in pair:
                        digest.update(array.tobytes())
                for array in evaluation:
                    digest.update(array.tobytes())
                self.assertEqual(digest.hexdigest(), switched.get(name, baseline))

    def test_stress_is_report_only_and_has_separate_features(self):
        for base_name, stress_name in STRESS_PAIRS:
            with self.subTest(variant=stress_name):
                base, stress = variant.VARIANTS[base_name], variant.VARIANTS[stress_name]
                self.assertEqual(stress, {**base, "eval_stress": True})
                self.assertNotEqual(variant.PRODUCTION, stress_name)
                self.assertNotEqual(variant.feature_version(base), variant.feature_version(stress))
                self.assertNotEqual(variant.stage1_version(base), variant.stage1_version(stress))

    def test_both_stress_variants_share_identical_evaluation_copies(self):
        for seed in (1, 2027):
            with self.subTest(seed=seed):
                (_, a), _ = self.copies(variant.VARIANTS["touch_bal_stress"], seed)
                (_, b), _ = self.copies(variant.VARIANTS["mixed_stress"], seed)
                for x, y in zip(a, b):
                    self.assert_bytes_equal(x, y)

    def test_full_every_frame_parameters_hide_more_ball_than_camera_on(self):
        d = scene(n=12000)
        for seed in (1, 7, 2027):
            with self.subTest(seed=seed):
                missing = []
                for params in (dg.PARAMS, dg.RANDOM_GAPS):
                    _, ball = dg.degrade(d["players"], d["ball"], d["label"],
                                         np.random.default_rng(seed), severity=1.0, params=params)
                    missing.append(np.isnan(ball[:, 0]).mean())
                self.assertGreater(missing[1], missing[0])


if __name__ == "__main__":
    unittest.main()
