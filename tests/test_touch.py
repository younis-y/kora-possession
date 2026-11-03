import unittest

import numpy as np

from kora.features import velocity
from kora.touch import TOUCH_DV, add, chain


def scene():
    """Home player 1 dribbles, passes to home player 2, away player 3 intercepts. 25 fps, metres."""
    n = 150
    ball = np.zeros((n, 2), np.float32)
    ball[:40, 0] = np.linspace(0, 4, 40)                 # dribble with home 1
    ball[40:80, 0] = np.linspace(4, 24, 40)              # pass: fast, straight
    ball[80:110] = np.c_[np.linspace(24, 26, 30), np.linspace(0, 12, 30)]     # home 2 turns it sideways
    ball[110:, 0] = np.linspace(26, 10, 40)              # away 3 wins it and plays it back
    ball[110:, 1] = 12
    d1H = np.full(n, 8.0, np.float32); d1A = np.full(n, 8.0, np.float32)
    d1H[:42] = 0.3; d1H[78:110] = 0.3; d1A[108:] = 0.3
    return ball, d1H, d1A


class TouchTests(unittest.TestCase):
    def test_pass_and_interception_give_touches_for_the_right_teams(self):
        ball, d1H, d1A = scene()
        f = {"d1H": d1H, "d1A": d1A, "d_diff": d1H - d1A}
        add(f, velocity(ball, 2))
        self.assertTrue(np.nanmax(f["touchH"][35:45]) > TOUCH_DV)          # the pass
        self.assertTrue(np.nanmax(f["touchA"][105:115]) > TOUCH_DV)        # the interception
        self.assertEqual(f["last_touch"][120], -1.0)
        self.assertEqual(f["last_touch"][100], 1.0)
        self.assertGreaterEqual(f["touch_chain"][100], 2.0)                # two home touches in a row
        self.assertLess(f["touch_chain"][140], 0.0)

    def test_missing_ball_gives_no_touches(self):
        ball, d1H, d1A = scene()
        ball[:] = np.nan
        f = {"d1H": d1H, "d1A": d1A, "d_diff": d1H - d1A}
        add(f, velocity(ball, 2))
        self.assertTrue(np.isnan(f["touchH"]).all())
        self.assertTrue(np.isnan(f["last_touch"]).all())
        self.assertEqual(np.nansum(f["touches_p25"]), 0.0)

    def test_chain_counts_consecutive_same_team_touches(self):
        events = np.zeros(10, bool); events[[1, 3, 5, 7]] = True
        side = np.array([1, 1, 1, 1, 1, 1, -1, -1, 1, 1], np.float32)
        np.testing.assert_array_equal(chain(events, side), [0, 1, 1, 2, 2, 3, 3, -1, -1, -1])


if __name__ == "__main__":
    unittest.main()
