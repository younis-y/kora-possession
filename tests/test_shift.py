import unittest

import numpy as np

from kora.shift import grouped_split, build_halves


class DomainSplitTest(unittest.TestCase):
    def test_held_out_match_and_half_are_absent_from_training(self):
        source = np.array(['m1', 'm1', 'm2', 'm2', 'm3'])
        target = np.array([1, 1, 2, 2, 2])
        train, test = grouped_split(source, target, 'm2', 2)
        np.testing.assert_array_equal(train, [0, 1, 4, 5, 6])
        np.testing.assert_array_equal(test, [2, 3, 7, 8, 9])

    def test_other_half_does_not_change_features_at_boundary(self):
        players = np.tile(np.array([[[-30., 0.], [-10., 3.], [10., -3.], [30., 0.]]]), (100, 1, 1))
        team = np.array([0, 0, 1, 1])
        ball = np.zeros((100, 2))
        half = np.repeat([1, 2], 50)
        before = build_halves(players, team, ball, half)
        changed = players.copy()
        changed[50:, :2, 0] += 30
        after = build_halves(changed, team, ball, half)
        np.testing.assert_allclose(before.iloc[:50]['cx_diff_c401'], after.iloc[:50]['cx_diff_c401'])

    def test_rejects_empty_domain_in_either_partition(self):
        with self.assertRaises(ValueError):
            grouped_split(np.array(['m1', 'm2']), np.array([1, 2]), 'missing', 1)


if __name__ == '__main__':
    unittest.main()
