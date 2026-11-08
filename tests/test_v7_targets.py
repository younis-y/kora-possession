"""Synthetic V7 target compatibility, priority and configuration checks."""
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


class TargetTests(unittest.TestCase):
    def test_frozen_default_and_priority_invariants(self):
        self.assertEqual(RADIUS, 5)
        # Random and edge cases protect byte-for-byte defaults and input immutability.
        edges = [[], [0], [2] * 20, [0] + [1] * 19, [0] * 19 + [1], [2] + [0] * 19,
                 [0] * 19 + [2], [2] * 20 + [0] * 3 + [1] * 3 + [2] * 20, [0, 1, 2, 1, 0, 2, 0, 1]]
        cases = [(np.array(y, np.int8), split) for y in edges for split in (len(y), len(y) // 2)]
        for seed in (0, 17, 2027, 73):
            rng = np.random.default_rng(seed)
            for length in (0, 1, 2, 17, 61, 127, 121, 501):
                for max_run in (1, 20):
                    y = np.repeat(rng.integers(0, 3, length, dtype=np.int8),
                                  rng.integers(1, max_run + 1, length))[:length]
                    cases.append((y, length // 2))
        for i, (y, split) in enumerate(cases):
            with self.subTest(case=i):
                half = np.where(np.arange(len(y)) < split, 1, 2)
                originals = y.copy(), half.copy()
                expected = legacy_target(y, half)
                for actual in (target(y, half), target(y, half, ha_priority=False)):
                    self.assertEqual((actual.dtype, actual.shape, actual.tobytes()),
                                     (expected.dtype, expected.shape, expected.tobytes()))
                self.assertTrue(np.all(target(y, half, True)[expected == 1] == 1))
                for actual, original in zip((y, half), originals):
                    np.testing.assert_array_equal(actual, original)

    def test_priority_windows(self):
        # Near restarts/stoppages every handover keeps 11 frames in either direction.
        for home in (0, 1):
            for distance in range(1, 11):
                for restart in (False, True):
                    with self.subTest(home=home, distance=distance, restart=restart):
                        labels = [2, home, 1 - home] if restart else [home, 1 - home, 2]
                        y = np.repeat(labels, [20, distance, 40 - distance]).astype(np.int8)
                        half, expected = np.ones(len(y), np.int8), np.zeros(len(y), np.int8)
                        dead, live = (15, 15 + distance) if restart else (15 + distance, 15)
                        expected[dead:dead + 11] = 2
                        expected[live:live + 11] = 1
                        np.testing.assert_array_equal(target(y, half, True), expected)
                        self.assertEqual(int((target(y, half) == 1).sum()), distance)
        y = np.array([2] * 20 + [0] * 3 + [1] * 3 + [2] * 20, np.int8)
        expected = np.zeros(len(y), np.int8)
        expected[15:32], expected[18:29] = 2, 1
        np.testing.assert_array_equal(target(y, np.ones(len(y)), True), expected)
        self.assertEqual(int((target(y, np.ones(len(y))) == 1).sum()), 0)
        # Isolated/dead-only windows remain identical; half changes are excluded, nearby windows spill.
        for labels in ([0, 1, 2, 0], [2, 1, 0, 2], [0, 2, 1], [2], []):
            y = np.repeat(np.array(labels, np.int8), 20)
            np.testing.assert_array_equal(target(y, np.ones(len(y)), True), legacy_target(y, np.ones(len(y))))
        for priority in (False, True):
            for state in (1, 2):
                half = np.repeat([1, 2], 20)
                np.testing.assert_array_equal(target(np.repeat([0, state], 20), half, priority), np.zeros(40))
                expected = np.zeros(40, np.int8)
                expected[13:24] = 1 if state == 1 else 2
                np.testing.assert_array_equal(target(np.array([0] * 18 + [state] * 2 + [0] * 20), half, priority), expected)

    def test_variant_defaults_and_cache_versions(self):
        baseline, candidate = (variant.VARIANTS[k] for k in ("touch_bal", "v7_ha_priority"))
        self.assertEqual(candidate, {**baseline, "detector": {"ha_priority": True}})
        self.assertEqual(variant.decoder(candidate), variant.decoder(baseline))
        self.assertEqual(variant.DETECTOR, {"ha_priority": False})
        self.assertEqual(variant.detector(), variant.detector(variant.CONFIG))
        # Every existing default and pinned base keeps its own feature/stage-1 cache keys.
        for name, config in variant.VARIANTS.items():
            with self.subTest(variant=name):
                self.assertEqual(variant.detector(json.loads(json.dumps(config))), {"ha_priority": name == "v7_ha_priority"})
        for base in ("touch_bal", "y8a_dwell", "b0"):
            with self.subTest(base=base):
                pinned = variant.with_detector(base, ha_priority=True)
                self.assertTrue(variant.detector(pinned)["ha_priority"])
                with mock.patch.object(Path, "read_bytes", autospec=True, side_effect=Path.read_bytes) as reads:
                    for version in (variant.feature_version, variant.stage1_version):
                        self.assertEqual(version(pinned), version(variant.VARIANTS[base]))
                self.assertNotIn(Path(variant.__file__).resolve(), [c.args[0] for c in reads.call_args_list])
                self.assertEqual(json.loads(json.dumps(pinned)), pinned)
        pinned = variant.with_detector("touch_bal", ha_priority=True)
        self.assertEqual(pinned, candidate)
        self.assertNotIn("detector", baseline)
        pinned["touch"] = False
        self.assertTrue(baseline["touch"])

    def test_detector_routing(self):
        # Evaluate source expressions only: no nested fitting, main calls, or match loading.
        root = Path(__file__).resolve().parents[1] / "kora"
        nested, boundary = (ast.parse((root / name).read_text()) for name in ("nested.py", "boundary.py"))
        main = next(n for n in boundary.body if isinstance(n, ast.FunctionDef) and n.name == "main")
        assignments = [{n.targets[0].id: n.value for n in nodes if isinstance(n, ast.Assign)
                        and isinstance(n.targets[0], ast.Name)} for nodes in (ast.walk(nested), main.body)]
        calls = [n for n in ast.walk(main) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "target"]
        summaries = [v for n in ast.walk(nested) if isinstance(n, ast.Dict) for k, v in zip(n.keys, n.values)
                     if isinstance(k, ast.Constant) and k.value == "detector"]
        self.assertEqual((len(calls), len(summaries)), (2, 1))
        y, half = np.array([2] * 20 + [0] * 3 + [1] * 37, np.int8), np.ones(60, np.int8)
        for priority in (False, True):
            with self.subTest(priority=priority):
                config = {"detector": {"ha_priority": priority}}
                resolver = lambda: variant.detector(config)
                resolved = eval(compile(ast.Expression(assignments[0]["DETECTOR_CFG"]), "config", "eval"), {"detector_config": resolver})
                final = eval(compile(ast.Expression(assignments[1]["detector_cfg"]), "config", "eval"), {"detector": resolver})
                context = dict(MATCHES=["S"], y={"S": y}, half={"S": half}, target=target, DETECTOR_CFG=resolved,
                               data={"S": (None, None, y, half)}, k="S", m="S", detector_cfg=final)
                actual = eval(compile(ast.Expression(assignments[0]["tgt"]), "targets", "eval"), context)
                np.testing.assert_array_equal(actual["S"], target(y, half, priority))
                for call in calls:
                    np.testing.assert_array_equal(eval(compile(ast.Expression(call), "final-target", "eval"), context), target(y, half, priority))
                summary = eval(compile(ast.Expression(summaries[0]), "summary", "eval"), context)
                self.assertEqual(json.loads(json.dumps(summary)), {"ha_priority": priority})
