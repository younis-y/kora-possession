"""Synthetic checks for Y11 (mirror averaging) and Y12 (weak-touch peaks), and their cache keys."""
import unittest

import numpy as np
import pandas as pd

from kora import variant
from kora.peaks import add, weak_touches
from kora.touch import TOUCH_DV


class PeakTests(unittest.TestCase):
    def test_only_weak_local_maxima_count(self):
        th = np.zeros(60); ta = np.zeros(60)
        th[10] = 0.5 * TOUCH_DV                      # weak home touch
        ta[30] = 0.3 * TOUCH_DV; ta[31] = 0.4 * TOUCH_DV   # one weak away peak, at 31
        th[50] = 1.5 * TOUCH_DV                      # strong: touch.py's job, not a weak touch
        th[55] = 0.1 * TOUCH_DV                      # too weak
        weak, side = weak_touches(th, ta)
        np.testing.assert_array_equal(np.flatnonzero(weak), [10, 31])
        self.assertEqual((side[10], side[31]), (1, -1))

    def test_columns_follow_the_last_and_next_weak_touch(self):
        th = np.zeros(100, np.float32); ta = np.zeros(100, np.float32)
        th[20] = 0.5 * TOUCH_DV; ta[60] = 0.5 * TOUCH_DV
        f = add(pd.DataFrame({"touchH": th, "touchA": ta}))
        self.assertEqual((f["wlast"][40], f["wnext"][40]), (1, -1))
        self.assertAlmostEqual(f["t_wlast"][40], 20 / 25, places=5)
        self.assertEqual(f["wtouch_p25"][30], 1)
        self.assertEqual(f["wtouch_f25"][50], -1)
        self.assertTrue(np.isnan(f["wlast"][10]))


class KeyTests(unittest.TestCase):
    def test_new_switches_leave_other_caches_alone(self):
        v = variant.VARIANTS
        self.assertEqual(variant.stage1_version(v["y11_mirror"]), variant.stage1_version(v["mixed"]))
        self.assertEqual(variant.feature_version(v["y12_peaks"]), variant.feature_version(v["mixed"]))
        self.assertNotEqual(variant.stage1_version(v["y12_peaks"]), variant.stage1_version(v["mixed"]))
        self.assertEqual(variant.stage1_version(v["y8a_dwell"]), "ff4d49f7e6es8b99e85822")   # unchanged by Y12


if __name__ == "__main__":
    unittest.main()
