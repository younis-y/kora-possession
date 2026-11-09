"""Checks for Y27 (kora/snap.py): only H<->A changes move, within +-r and the midpoints, and the variants
reuse y23_big2's caches."""
import unittest

import numpy as np

from kora import snap, variant


class Y27Tests(unittest.TestCase):
    def test_peak_moves_the_change_to_the_detector(self):
        path = np.repeat([0, 1], [50, 50]).astype(np.int8)
        half = np.ones(100, np.int8)
        change = np.zeros(100)
        change[54] = 0.9
        out = snap.snap(path, change, half, 6, "peak")
        np.testing.assert_array_equal(out, np.repeat([0, 1], [54, 46]))
        np.testing.assert_array_equal(snap.snap(path, change, half, 3, "peak"), path)   # 54 is outside +-3

    def test_mass_and_ties(self):
        path = np.repeat([1, 0], [50, 50]).astype(np.int8)
        half = np.ones(100, np.int8)
        change = np.zeros(100)
        change[[46, 48]] = 0.5
        np.testing.assert_array_equal(snap.snap(path, change, half, 6, "mass"), np.repeat([1, 0], [47, 53]))
        np.testing.assert_array_equal(snap.snap(path, np.zeros(100), half, 6, "peak"), path)

    def test_dead_ball_halves_and_neighbours_are_respected(self):
        path = np.repeat([0, 2, 1, 0], [30, 10, 4, 56]).astype(np.int8)
        half = np.repeat([1, 2], [42, 58]).astype(np.int8)
        change = np.ones(100)
        change[[25, 41, 48]] = 5.0
        out = snap.snap(path, change, half, 12, "peak")
        np.testing.assert_array_equal(out[:44], path[:44])               # D edges and the half edge never move
        self.assertEqual(int(np.flatnonzero(np.diff(out) != 0)[-1]) + 1, 48)   # the A->H change moved 44 -> 48
        runs = np.flatnonzero(np.diff(out) != 0)
        self.assertEqual(len(runs), len(np.flatnonzero(np.diff(path) != 0)))   # no run disappears

    def test_variants_reuse_y23_big2_caches(self):
        V = variant.VARIANTS
        for name, base in (("y27_snap", "y23_big2"), ("y27_snap_rand", "y23_big2_rand"),
                           ("y27_snap_r_rand", "y23_big_s2028_rand")):
            self.assertEqual(variant.feature_version(V[name]), variant.feature_version(V[base]), name)
            self.assertEqual(variant.stage1_version(V[name]), variant.stage1_version(V[base]), name)
            self.assertEqual(V[name]["seq"], V[base]["seq"], name)
            self.assertEqual(variant.decoder(V[name])["snap"]["r"], [3, 6, 12], name)
        self.assertIsNone(variant.decoder(V["y23_big2"])["snap"])


if __name__ == "__main__":
    unittest.main()
