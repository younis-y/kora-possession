"""Synthetic checks for boundary-cut relabelling (RESEARCH.md Y10)."""
import unittest

import numpy as np

from kora.cuts import peaks, relabel


def probs(labels):
    """Stage-1 probabilities that favour each frame's label at 0.7."""
    p = np.full((len(labels), 3), 0.15)
    p[np.arange(len(labels)), labels] = 0.7
    return p


class CutTests(unittest.TestCase):
    def test_peaks_are_local_maxima_above_theta_and_apart(self):
        c = np.zeros(100); c[[10, 12, 40, 70]] = [0.6, 0.9, 0.3, 0.8]
        np.testing.assert_array_equal(peaks(c, 0.5, 6), [12, 70])      # 10 is within 6 of the stronger 12
        np.testing.assert_array_equal(peaks(c, 0.2, 1), [10, 12, 40, 70])

    def test_swallowed_short_possession_is_recovered(self):
        truth = np.r_[np.ones(40), np.zeros(10), np.ones(40)].astype(np.int8)    # A, short H, A
        path = np.ones(90, np.int8)                                              # decoder swallowed it
        change = np.zeros(90); change[[40, 50]] = 0.8
        out = relabel(path, probs(truth), change, np.ones(90, int), 0.5, 6)
        np.testing.assert_array_equal(out, truth)

    def test_no_peaks_or_dead_ball_leave_the_path_alone(self):
        truth = np.r_[np.ones(40), np.zeros(10), np.ones(40)].astype(np.int8)
        path = np.ones(90, np.int8)
        np.testing.assert_array_equal(relabel(path, probs(truth), np.zeros(90), np.ones(90, int), 0.5, 6), path)
        dead = np.full(90, 2, np.int8)
        change = np.zeros(90); change[[40, 50]] = 0.8
        np.testing.assert_array_equal(relabel(dead, probs(truth), change, np.ones(90, int), 0.5, 6), dead)

    def test_piece_with_the_same_team_stays_one_spell(self):
        truth = np.zeros(90, np.int8)
        change = np.zeros(90); change[45] = 0.9
        out = relabel(truth.copy(), probs(truth), change, np.ones(90, int), 0.5, 6)
        np.testing.assert_array_equal(out, truth)


if __name__ == "__main__":
    unittest.main()
