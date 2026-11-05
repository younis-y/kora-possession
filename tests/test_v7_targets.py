"""Synthetic target compatibility and configuration checks for RESEARCH.md V7 round 1."""
import ast
import json
from pathlib import Path
import unittest
from unittest import mock

import numpy as np

from kora.boundary import RADIUS, target
from kora import variant


def legacy_target(y, half):
    """Frozen pre-V7 implementation, including class-2 overwrite and half handling."""
    t = np.zeros(len(y), np.int8)
    for c in np.flatnonzero((y[1:] != y[:-1]) & (half[1:] == half[:-1])) + 1:
        k = 2 if 2 in (y[c], y[c - 1]) else 1
        t[max(c - 5, 0):c + 6] = np.maximum(t[max(c - 5, 0):c + 6], k)
    return t


class DefaultTargetTests(unittest.TestCase):
    def check_legacy(self, y, half):
        originals = y.copy(), half.copy()
        expected = legacy_target(y, half)
        for actual in (target(y, half), target(y, half, ha_priority=False)):
            self.assertEqual(actual.dtype, expected.dtype)
            self.assertEqual(actual.shape, expected.shape)
            self.assertEqual(actual.tobytes(), expected.tobytes())
        for actual, original in zip((y, half), originals):
            np.testing.assert_array_equal(actual, original)

    def test_random_sequences_with_two_halves_match_frozen_default(self):
        self.assertEqual(RADIUS, 5)
        for seed in (0, 17, 2027, 73):
            rng = np.random.default_rng(seed)
            for length in (0, 1, 2, 17, 61, 127):
                for max_run in (1, 20):
                    with self.subTest(seed=seed, length=length, max_run=max_run):
                        y = np.repeat(rng.integers(0, 3, length, dtype=np.int8),
                                      rng.integers(1, max_run + 1, length))[:length]
                        half = np.where(np.arange(length) < length // 2, 1, 2)
                        self.check_legacy(y, half)

    def test_edge_changes_and_close_changes_match_frozen_default(self):
        cases = [[], [0], [2] * 20, [0] + [1] * 19, [0] * 19 + [1],
                 [2] + [0] * 19, [0] * 19 + [2],
                 [2] * 20 + [0] * 3 + [1] * 3 + [2] * 20,
                 [0, 1, 2, 1, 0, 2, 0, 1]]
        for labels in cases:
            y = np.array(labels, np.int8)
            for split in (len(y), len(y) // 2):
                with self.subTest(labels=labels, split=split):
                    self.check_legacy(y, np.where(np.arange(len(y)) < split, 1, 2))

    def test_cross_half_changes_are_excluded_but_nearby_windows_spill(self):
        half = np.repeat([1, 2], 20)
        for priority in (False, True):
            for state in (1, 2):
                with self.subTest(priority=priority, state=state):
                    y = np.repeat([0, state], 20).astype(np.int8)
                    np.testing.assert_array_equal(target(y, half, priority), np.zeros(40, np.int8))
                    y = np.array([0] * 18 + [state] * 2 + [0] * 20, np.int8)
                    expected = np.zeros(40, np.int8)
                    expected[13:24] = 1 if state == 1 else 2
                    np.testing.assert_array_equal(target(y, half, priority), expected)


class HandoverPriorityTests(unittest.TestCase):
    def test_handover_after_restart_keeps_all_11_frames(self):
        for home in (0, 1):
            for distance in range(1, 11):
                with self.subTest(home=home, distance=distance):
                    y = np.repeat([2, home, 1 - home], [20, distance, 40 - distance]).astype(np.int8)
                    half = np.ones(len(y), np.int8)
                    expected = np.zeros(len(y), np.int8)
                    expected[15:26] = 2
                    expected[15 + distance:26 + distance] = 1
                    actual = target(y, half, ha_priority=True)
                    np.testing.assert_array_equal(actual, expected)
                    self.assertEqual(int((actual == 1).sum()), 11)
                    self.assertEqual(int((target(y, half) == 1).sum()), distance)

    def test_handover_before_stoppage_keeps_all_11_frames(self):
        for home in (0, 1):
            for distance in range(1, 11):
                with self.subTest(home=home, distance=distance):
                    y = np.repeat([home, 1 - home, 2], [20, distance, 40 - distance]).astype(np.int8)
                    half = np.ones(len(y), np.int8)
                    expected = np.zeros(len(y), np.int8)
                    expected[15 + distance:26 + distance] = 2
                    expected[15:26] = 1
                    actual = target(y, half, ha_priority=True)
                    np.testing.assert_array_equal(actual, expected)
                    self.assertEqual(int((actual == 1).sum()), 11)
                    self.assertEqual(int((target(y, half) == 1).sum()), distance)

    def test_stoppages_bracketing_handover_no_longer_erase_it(self):
        y = np.array([2] * 20 + [0] * 3 + [1] * 3 + [2] * 20, np.int8)
        half = np.ones(len(y), np.int8)
        expected = np.zeros(len(y), np.int8)
        expected[15:32] = 2
        expected[18:29] = 1
        np.testing.assert_array_equal(target(y, half, True), expected)
        self.assertEqual(int((target(y, half) == 1).sum()), 0)
        self.assertEqual(int((expected == 1).sum()), 11)

    def test_isolated_and_dead_only_windows_equal_default(self):
        for labels in ([0, 1, 2, 0], [2, 1, 0, 2], [0, 2, 1], [2], []):
            with self.subTest(labels=labels):
                y = np.repeat(np.array(labels, np.int8), 20)
                half = np.ones(len(y), np.int8)
                np.testing.assert_array_equal(target(y, half, True), legacy_target(y, half))
                if 0 not in labels or 1 not in labels:
                    self.assertEqual(int((target(y, half, True) == 1).sum()), 0)

    def test_priority_never_removes_existing_class_1_frames(self):
        for seed in (0, 17, 2027, 73):
            rng = np.random.default_rng(seed)
            for length in (1, 17, 121, 501):
                with self.subTest(seed=seed, length=length):
                    y = np.repeat(rng.integers(0, 3, length, dtype=np.int8),
                                  rng.integers(1, 21, length))[:length]
                    half = np.where(np.arange(length) < length // 2, 1, 2)
                    originals = y.copy(), half.copy()
                    default, priority = target(y, half), target(y, half, True)
                    self.assertTrue(np.all(priority[default == 1] == 1))
                    self.assertGreaterEqual(int((priority == 1).sum()), int((default == 1).sum()))
                    for actual, original in zip((y, half), originals):
                        np.testing.assert_array_equal(actual, original)


class V7VariantTests(unittest.TestCase):
    def test_candidate_is_touch_bal_plus_detector_key(self):
        baseline, candidate = (variant.VARIANTS[k] for k in ("touch_bal", "v7_ha_priority"))
        self.assertEqual(variant.PRODUCTION, "touch_bal")
        self.assertEqual(candidate, {**baseline, "detector": {"ha_priority": True}})
        self.assertEqual(variant.decoder(candidate), variant.decoder(baseline))

    def test_all_existing_variants_resolve_to_default_detector(self):
        self.assertEqual(variant.DETECTOR, {"ha_priority": False})
        self.assertEqual(variant.detector(), variant.detector(variant.CONFIG))
        for name, config in variant.VARIANTS.items():
            with self.subTest(variant=name):
                self.assertEqual(variant.detector(config), {"ha_priority": name == "v7_ha_priority"})

    def test_detector_config_round_trips_through_json(self):
        for name in ("touch_bal", "v7_ha_priority"):
            with self.subTest(variant=name):
                config = variant.VARIANTS[name]
                restored = json.loads(json.dumps(config))
                self.assertEqual(restored, config)
                resolved = variant.detector(restored)
                self.assertEqual(json.loads(json.dumps(resolved)), resolved)
                self.assertIs(resolved["ha_priority"], name == "v7_ha_priority")

    def test_feature_and_stage1_versions_reuse_touch_bal_caches(self):
        baseline, candidate = (variant.VARIANTS[k] for k in ("touch_bal", "v7_ha_priority"))
        with mock.patch.object(Path, "read_bytes", autospec=True, side_effect=Path.read_bytes) as reads:
            self.assertEqual(variant.feature_version(candidate), variant.feature_version(baseline))
            self.assertEqual(variant.stage1_version(candidate), variant.stage1_version(baseline))
        self.assertNotIn(Path(variant.__file__).resolve(), [call.args[0] for call in reads.call_args_list])


class DetectorRoutingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Inspect source only: importing/running kora.nested or either main is unnecessary.
        root = Path(__file__).resolve().parents[1] / "kora"
        cls.nested = ast.parse((root / "nested.py").read_text())
        cls.boundary = ast.parse((root / "boundary.py").read_text())

    def test_nested_targets_use_resolved_detector_config(self):
        assignments = {node.targets[0].id: node.value for node in ast.walk(self.nested)
                       if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)}
        y = np.array([2] * 20 + [0] * 3 + [1] * 37, np.int8)
        half = np.ones(len(y), np.int8)
        for priority in (False, True):
            config = {"detector": {"ha_priority": priority}}
            resolved = eval(compile(ast.Expression(assignments["DETECTOR_CFG"]), "nested-config", "eval"),
                            {"detector_config": lambda: variant.detector(config)})
            context = {"MATCHES": ["synthetic"], "y": {"synthetic": y}, "half": {"synthetic": half},
                       "target": target, "DETECTOR_CFG": resolved}
            actual = eval(compile(ast.Expression(assignments["tgt"]), "nested-targets", "eval"), context)
            np.testing.assert_array_equal(actual["synthetic"], target(y, half, priority))

    def test_nested_summary_records_resolved_detector_config(self):
        values = [value for node in ast.walk(self.nested) if isinstance(node, ast.Dict)
                  for key, value in zip(node.keys, node.values)
                  if isinstance(key, ast.Constant) and key.value == "detector"]
        self.assertEqual(len(values), 1)
        for priority in (False, True):
            config = {"ha_priority": priority}
            actual = eval(compile(ast.Expression(values[0]), "nested-summary", "eval"), {"DETECTOR_CFG": config})
            self.assertEqual(json.loads(json.dumps(actual)), config)

    def test_final_fit_targets_and_rate_use_same_detector_config(self):
        main = next(node for node in self.boundary.body if isinstance(node, ast.FunctionDef) and node.name == "main")
        assignments = {node.targets[0].id: node.value for node in main.body
                       if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)}
        calls = [node for node in ast.walk(main) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Name) and node.func.id == "target"]
        self.assertEqual(len(calls), 2)   # xy() fitting targets and the base rate
        y = np.array([2] * 20 + [0] * 3 + [1] * 37, np.int8)
        half = np.ones(len(y), np.int8)
        for priority in (False, True):
            config = {"detector": {"ha_priority": priority}}
            resolved = eval(compile(ast.Expression(assignments["detector_cfg"]), "boundary-config", "eval"),
                            {"detector": lambda: variant.detector(config)})
            context = {"data": {"synthetic": (None, None, y, half)}, "k": "synthetic", "m": "synthetic",
                       "target": target, "detector_cfg": resolved}
            for call in calls:
                with self.subTest(priority=priority, line=call.lineno):
                    actual = eval(compile(ast.Expression(call), "boundary-target", "eval"), context)
                    np.testing.assert_array_equal(actual, target(y, half, priority))


class WithDetectorTests(unittest.TestCase):
    def test_copies_base_and_adds_only_detector_settings(self):
        pinned = variant.with_detector("touch_bal", ha_priority=True)
        self.assertEqual(pinned, variant.VARIANTS["v7_ha_priority"])
        self.assertNotIn("detector", variant.VARIANTS["touch_bal"])       # base untouched
        pinned["touch"] = False
        self.assertTrue(variant.VARIANTS["touch_bal"]["touch"])           # deep copy

    def test_cache_versions_follow_the_base(self):
        for base in ("touch_bal", "y8a_dwell", "b0"):
            pinned = variant.with_detector(base, ha_priority=True)
            self.assertEqual(variant.feature_version(pinned), variant.feature_version(variant.VARIANTS[base]))
            self.assertEqual(variant.stage1_version(pinned), variant.stage1_version(variant.VARIANTS[base]))
            self.assertEqual(variant.detector(pinned)["ha_priority"], True)
            json.dumps(pinned)


if __name__ == "__main__":
    unittest.main()
