"""Synthetic compatibility and transition-bonus checks for RESEARCH.md V6."""
import ast
import json
import unittest
from pathlib import Path

import numpy as np

from kora.boundary import GRID
from kora.decode import DMAX, STRIDE, decode_hsmm, emissions, hsmm
from kora.variant import VARIANTS, decoder, feature_version, stage1_version

LEGACY_DMAX = 600
LEGACY_STRIDE = 3


def legacy_hsmm(e, ld, cost_ha, cost_d, w, bonus):
    """Frozen pre-V6 implementation, including arithmetic and tie order."""
    T = len(e)
    S = np.vstack([np.zeros(3), np.cumsum(e, 0)])
    C = np.array([[cost_ha, cost_ha, cost_d], [cost_ha, cost_ha, cost_d], [cost_d, cost_d, cost_d]], float)
    ldw = (w * ld).T
    Vs = np.full((T + 1, 3), -np.inf); Vs[0] = 0
    prev = np.zeros((T + 1, 3), np.int64)
    dur = np.zeros((T + 1, 3), np.int64)
    V = np.full((T + 1, 3), -np.inf)
    for t in range(1, T + 1):
        D = min(LEGACY_DMAX, t)
        j = t - np.arange(1, D + 1)
        cand = Vs[j] - S[j] + ldw[:D] + bonus[j]
        i = cand.argmax(0)
        V[t] = cand[i, (0, 1, 2)] + S[t]
        dur[t] = i + 1
        if t < T:
            M = V[t][:, None] - C
            pk = M.argmax(0)
            Vs[t] = M[pk, (0, 1, 2)]
            prev[t] = pk
    path = np.empty(T, np.int8)
    t, k = T, int(V[T].argmax())
    while t > 0:
        d = dur[t, k]
        path[t - d:t] = k
        t -= d
        if t > 0:
            k = prev[t, k]
    return path


def legacy_decode_hsmm(proba, change, segments, prior, ld, cost_ha, cost_d, w, beta, rate):
    """Frozen pre-V6 wrapper; emissions() is unchanged by V6."""
    out = np.full(len(proba), -1, np.int8)
    e = emissions(proba, prior, 0.0)
    for a, z in segments:
        n = (z - a) // LEGACY_STRIDE * LEGACY_STRIDE
        eb = e[a:a + n].reshape(-1, LEGACY_STRIDE, 3).sum(1)
        cb = change[a:a + n].reshape(-1, LEGACY_STRIDE).max(1)
        bonus = np.zeros((len(eb) + 1, 3))
        bonus[:-1, :2] = beta * np.log(np.clip(cb, 1e-4, 1.0) / rate)[:, None]
        p = legacy_hsmm(eb, ld, cost_ha, cost_d, w, bonus)
        out[a:a + n] = np.repeat(p, LEGACY_STRIDE)
        out[a + n:z] = p[-1]
    return out


class DecoderTests(unittest.TestCase):
    def test_default_hsmm_matches_frozen_reference_on_random_inputs(self):
        for seed in (0, 17, 2027):
            rng = np.random.default_rng(seed)
            ld = rng.normal(-5, 2, (3, DMAX))
            for length in (1, 2, 17, 43):
                with self.subTest(seed=seed, length=length):
                    e = rng.normal(-2, 3, (length, 3))
                    bonus = rng.normal(0, 4, (length + 1, 3))
                    args = (e, ld, 4, 8, 2, bonus)
                    expected = legacy_hsmm(*args)
                    np.testing.assert_array_equal(hsmm(*args), expected)
                    np.testing.assert_array_equal(hsmm(*args, ha_bonus_only=False), expected)

    def test_default_wrapper_matches_frozen_reference_with_two_halves_and_tails(self):
        for seed in (0, 17, 2027):
            with self.subTest(seed=seed):
                rng = np.random.default_rng(seed)
                proba = rng.dirichlet(np.ones(3), size=127)
                change = rng.uniform(0, 1, len(proba))
                change[[3, 4, 69]] = (0, 1e-8, 1)
                segments = [(3, 64), (69, 127)]
                args = (proba, change, segments, np.array([0.4, 0.4, 0.2]),
                        rng.normal(-5, 2, (3, DMAX)), 4, 8, 2, 3, 0.037)
                expected = legacy_decode_hsmm(*args)
                np.testing.assert_array_equal(decode_hsmm(*args), expected)
                np.testing.assert_array_equal(decode_hsmm(*args, ha_bonus_only=False), expected)

    def test_default_ties_keep_legacy_order(self):
        args = (np.zeros((7, 3)), np.zeros((3, DMAX)), 0, 0, 0, np.zeros((8, 3)))
        np.testing.assert_array_equal(hsmm(*args), legacy_hsmm(*args))
        np.testing.assert_array_equal(hsmm(*args), np.zeros(7, np.int8))
        args = (np.full((25, 3), 1 / 3), np.full(25, 0.1), [(0, 10), (10, 25)],
                np.full(3, 1 / 3), np.zeros((3, DMAX)), 0, 0, 0, 0, 0.1)
        np.testing.assert_array_equal(decode_hsmm(*args), legacy_decode_hsmm(*args))

    def test_dmax_chaining_keeps_long_possessions_possible(self):
        self.assertEqual((DMAX, STRIDE), (LEGACY_DMAX, LEGACY_STRIDE))
        length = (DMAX + 3) * STRIDE + 1
        for state in range(3):
            with self.subTest(state=state):
                proba = np.full((length, 3), 0.005)
                proba[:, state] = 0.99
                args = (proba, np.full(length, 0.1), [(0, length)], np.full(3, 1 / 3),
                        np.zeros((3, DMAX)), 4, 8, 2, 3, 0.1)
                expected = legacy_decode_hsmm(*args)
                np.testing.assert_array_equal(decode_hsmm(*args), expected)
                np.testing.assert_array_equal(expected, np.full(length, state, np.int8))
                np.testing.assert_array_equal(decode_hsmm(*args, ha_bonus_only=True), expected)

    def test_transition_edge_optima(self):
        # Large bonuses on forbidden edges would change these optima if accidentally used.
        cases = [
            ("H to A reward", [[5, -100, -100], [0, -1, -100]], 1, 1, 4, (0, 1)),
            ("A to H reward", [[-100, 5, -100], [-1, 0, -100]], 1, 0, 4, (1, 0)),
            ("H to A penalty", [[5, -100, -100], [0, 3, -100]], 1, 1, -4, (0, 0)),
            ("A to H penalty", [[-100, 5, -100], [3, 0, -100]], 1, 0, -4, (1, 1)),
            ("no restart reward", [[-100, -100, 5], [-1, -100, 0]], 1, 0, 100, (2, 2)),
            ("no restart penalty", [[-100, -100, 5], [3, -100, 0]], 1, 0, -100, (2, 0)),
            ("no stoppage reward", [[5, -100, -100], [0, -100, -1]], 1, 2, 100, (0, 0)),
            ("no initial reward", [[0, -1, -2]], 0, 1, 100, (0,)),
            ("no initial penalty", [[0, -1, -2]], 0, 0, -100, (0,)),
            ("no H continuation reward", [[5, -100, -100], [0, 3, -100]], 1, 0, 100, (0, 1)),
            ("no A continuation reward", [[-100, 5, -100], [3, 0, -100]], 1, 1, 100, (1, 0)),
        ]
        ld = np.zeros((3, DMAX))
        for name, rows, start, destination, value, expected in cases:
            with self.subTest(case=name):
                e = np.array(rows, float)
                bonus = np.zeros((len(e) + 1, 3))
                bonus[start, destination] = value
                np.testing.assert_array_equal(hsmm(e, ld, 2, 2, 1, bonus, True), expected)

    def test_wrapper_excludes_restart_bonus_and_repeats_stride_tail(self):
        e = np.array([[-100, -100, 5], [-1, -100, 0]], float)
        proba = np.exp(e / STRIDE)
        proba /= proba.sum(1, keepdims=True)
        proba = np.vstack([np.repeat(proba, STRIDE, axis=0), proba[-1:]])
        change = np.r_[np.full(STRIDE, 0.01), np.full(STRIDE + 1, 0.01 * np.exp(4))]
        args = (proba, change, [(0, len(proba))], np.full(3, 1 / 3), np.zeros((3, DMAX)),
                2, 2, 1, 1, 0.01)
        np.testing.assert_array_equal(decode_hsmm(*args), [2, 2, 2, 0, 0, 0, 0])
        np.testing.assert_array_equal(decode_hsmm(*args, ha_bonus_only=True), np.full(7, 2))

    def test_h_a_swap_symmetry(self):
        swap = np.array([1, 0, 2])
        for seed in (0, 17, 2027):
            with self.subTest(seed=seed):
                rng = np.random.default_rng(seed)
                ld = rng.normal(-3, 1, (3, DMAX))
                ld[1] = ld[0]
                proba = rng.dirichlet(np.ones(3), size=61)
                change = rng.uniform(0, 1, len(proba))
                prior = np.array([0.4, 0.35, 0.25])
                args = (change, [(0, 31), (31, 61)])
                path = decode_hsmm(proba, *args, prior, ld, 2, 4, 2, 3, 0.037, True)
                mirrored = decode_hsmm(proba[:, swap], *args, prior[swap], ld[swap], 2, 4, 2, 3, 0.037, True)
                np.testing.assert_array_equal(mirrored, swap[path])

    def test_variant_has_54_b0_settings_with_only_bonus_mode_changed(self):
        config = VARIANTS["v6_ha_only"]
        grid = config["decoder"]["hsmm_grid"]
        self.assertEqual(len(grid), 54)
        self.assertEqual(len(set(grid)), 54)
        self.assertEqual(grid, [s + (True,) for s in GRID])
        self.assertEqual(set(config), {"decoder"})
        self.assertEqual(set(config["decoder"]), {"hsmm_grid"})
        baseline, resolved = decoder(VARIANTS["b0"]), decoder(config)
        self.assertEqual({k: v for k, v in resolved.items() if k != "hsmm_grid"},
                         {k: v for k, v in baseline.items() if k != "hsmm_grid"})
        self.assertEqual((resolved["plain_grid"], resolved["inner_wins"]), (None, 5))

    def test_nested_routing_and_serialization(self):
        # Evaluate source expressions only: no imports of nested or match data.
        tree = ast.parse((Path(__file__).resolve().parents[1] / "kora/nested.py").read_text())
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                 and n.func.id == "decode_hsmm"]
        values = [v for n in ast.walk(tree) if isinstance(n, ast.Dict) for k, v in zip(n.keys, n.values)
                  if isinstance(k, ast.Constant) and k.value == "hsmm"]
        self.assertEqual((len(calls), len(values)), (2, 1))
        for setting in ((2, 8, 3, 4), (2, 8, 3, 4, True)):
            with self.subTest(setting=setting):
                def stub(proba, change, segments, prior, ld, cost_ha, cost_d, w, beta, rate, ha_bonus_only=False):
                    return (cost_ha, cost_d, w, beta), rate, ha_bonus_only
                context = dict(P={0: None}, C={0: None}, half={0: None}, k=0, segments_of=lambda _: None,
                               prior=None, ld=None, setting=setting, rate=0.037, outer={0: None}, change={(0, 0): None},
                               m=0, seg=None, HSMM=[setting], bh=0, hs=np.array([[0.1, 0.3]]))
                for call in calls:
                    result = eval(compile(ast.Expression(call), "nested-call", "eval"), {"decode_hsmm": stub}, context)
                    self.assertEqual(result, (setting[:4], 0.037, len(setting) == 5))
                result = eval(compile(ast.Expression(values[0]), "nested-setting", "eval"), context)
                self.assertEqual(result, dict(cost_ha=2, cost_d=8, w_dur=3, beta=4, ha_bonus_only=len(setting) == 5,
                                              inner_mean=0.2, rate=0.037))
                self.assertEqual(json.loads(json.dumps(result)), result)
        for version in (feature_version, stage1_version):
            self.assertEqual(version(VARIANTS["v6_ha_only"]), version(VARIANTS["b0"]))
