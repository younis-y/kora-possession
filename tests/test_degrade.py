import unittest
from unittest import mock

import numpy as np

from kora import degrade as dg

PARAMS = {"p_miss_dead": 1.0, "p_miss_alive": 0.1, "live_gap_s_quantiles": [0.4, 0.4],
          "dead_start_offset_s_quantiles": [0.0, 0.0], "dead_end_offset_s_quantiles": [0.0, 0.0],
          "player_visible_s_mean": 4.0, "player_gap_s_quantiles": [1.0, 1.0], "closeup_prob": 1.0,
          "closeup_min_s": 1.0, "closeup_radius_m_quantiles": [5.0, 5.0], "position_noise_m": 0.1}


def match(n=2000, P=6):
    rng = np.random.default_rng(0)
    players = rng.uniform(-30, 30, (n, P, 2)).astype(np.float32)
    ball = np.zeros((n, 3), np.float32)
    label = np.zeros(n, np.int8)
    label[500:700] = 2
    return players, ball, label


class DegradeTests(unittest.TestCase):
    def test_severity_zero_returns_clean_tracking(self):
        players, ball, label = match()
        with mock.patch.dict(dg.PARAMS, PARAMS, clear=True):
            p, b = dg.degrade(players, ball, label, np.random.default_rng(1), severity=0.0)
        np.testing.assert_array_equal(p, players)
        np.testing.assert_array_equal(b, ball[:, :2])

    def test_full_severity_hides_the_whole_stoppage_and_some_live_play(self):
        players, ball, label = match()
        with mock.patch.dict(dg.PARAMS, PARAMS, clear=True):
            p, b = dg.degrade(players, ball, label, np.random.default_rng(1), severity=1.0)
        miss = np.isnan(b[:, 0])
        self.assertTrue(miss[500:700].all())
        live = np.r_[miss[:500], miss[700:]]
        self.assertAlmostEqual(live.mean(), 0.1, delta=0.03)
        self.assertTrue(np.isnan(p).any())
        self.assertFalse(np.allclose(np.nan_to_num(p), np.nan_to_num(players)))

    def test_same_seed_same_copy(self):
        players, ball, label = match()
        with mock.patch.dict(dg.PARAMS, PARAMS, clear=True):
            a = dg.degrade(players, ball, label, np.random.default_rng(7))
            b = dg.degrade(players, ball, label, np.random.default_rng(7))
        for x, y in zip(a, b):
            np.testing.assert_array_equal(x, y)

    def test_random_ball_is_repeatable_and_spends_the_share_per_half(self):
        _, ball, _ = match(n=20000)
        half = np.repeat([1, 2], 10000)
        ext = {"missing_share": 0.4, "gap_s_quantiles": [1.0, 3.0]}
        with mock.patch.dict(dg.RANDOM_GAPS, ext, clear=True):
            b = dg.random_ball(ball, half, np.random.default_rng(3))
            again = dg.random_ball(ball, half, np.random.default_rng(3))
        np.testing.assert_array_equal(b, again)
        for h in (1, 2):
            share = np.isnan(b[half == h, 0]).mean()
            self.assertTrue(0.25 < share <= 0.4, share)
        self.assertTrue(np.isfinite(b[~np.isnan(b[:, 0])]).all())


if __name__ == "__main__":
    unittest.main()
