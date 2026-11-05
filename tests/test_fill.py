"""Synthetic checks for RESEARCH.md Y15 (short ball-gap repair) and its cache key."""
import unittest

import numpy as np
import pandas as pd

from kora import fill, variant


class FillTests(unittest.TestCase):
    def test_short_gap_filled_on_a_straight_line(self):
        ball = np.c_[np.arange(10.0), np.zeros(10)]
        ball[3:6] = np.nan
        out, mask = fill.fill(ball, np.ones(10, int))
        np.testing.assert_allclose(out[:, 0], np.arange(10.0))
        np.testing.assert_array_equal(np.flatnonzero(mask), [3, 4, 5])

    def test_long_edge_and_cross_half_gaps_stay_missing(self):
        ball = np.c_[np.arange(60.0), np.zeros(60)]
        ball[:2] = np.nan                               # no seen frame before
        ball[10:10 + fill.MAX_GAP] = np.nan              # too long
        ball[48:52] = np.nan                             # spans the half change at 50
        half = np.r_[np.ones(50, int), np.full(10, 2)]
        out, mask = fill.fill(ball, half)
        self.assertFalse(mask.any())
        np.testing.assert_array_equal(np.isnan(out), np.isnan(ball))

    def test_wrap_appends_the_flag(self):
        build = lambda p, t, b, h: pd.DataFrame({"bx": b[:, 0]})
        ball = np.c_[np.arange(6.0), np.zeros(6)]
        ball[2] = np.nan
        X = fill.wrap(build)(None, None, ball, np.ones(6, int))
        self.assertEqual(list(X["b_filled"]), [0, 0, 1, 0, 0, 0])
        self.assertEqual(X["bx"][2], 2.0)

    def test_fill_is_keyed_only_when_on(self):
        v = variant.VARIANTS
        self.assertNotEqual(variant.feature_version(v["y15_fill"]), variant.feature_version(v["mixed"]))
        self.assertEqual(variant.feature_version({**v["y15_fill"], "fill": None}), variant.feature_version(v["mixed"]))


if __name__ == "__main__":
    unittest.main()
