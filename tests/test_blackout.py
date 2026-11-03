import unittest

import numpy as np

from kora.blackout import apply_mask, make_mask, summarize, shuffle_gaps


class BlackoutTests(unittest.TestCase):
    def test_independent_masks_ignore_labels_and_repeat_for_seed(self):
        labels = np.tile([0, 1, 2, 2], 1000)
        half = np.repeat([1, 2], 2000)
        first = make_mask(labels, half, 'independent', 17)
        np.testing.assert_array_equal(first, make_mask(2 - labels, half, 'independent', 17))
        self.assertTrue(first.any())
        self.assertFalse(first.all())
        self.assertFalse(np.array_equal(first, make_mask(labels, half, 'independent', 18)))

    def test_live_stress_retains_baseline_and_adds_only_live_frames(self):
        labels = np.tile([0, 1, 2, 2], 1000)
        half = np.repeat([1, 2], 2000)
        baseline = np.zeros(len(labels), dtype=bool)
        baseline[2::12] = True
        result = make_mask(labels, half, 'live', 7, baseline)
        self.assertTrue(np.all(result[baseline]))
        self.assertTrue(np.any(result & ~baseline))
        self.assertFalse(np.any((result & ~baseline) & (labels == 2)))
        self.assertEqual(baseline.sum(), len(baseline[2::12]))

    def test_shuffled_gaps_preserve_counts_and_lengths_within_each_half(self):
        rng = np.random.default_rng(51)
        half = np.repeat([1, 2], 4000)
        original = rng.random(len(half)) < 0.3
        shuffled = shuffle_gaps(original, half, 17)
        self.assertTrue(np.array_equal(shuffled, shuffle_gaps(original, half, 17)))
        self.assertFalse(np.array_equal(shuffled, original))
        for h in (1, 2):
            selection = half == h
            old = summarize(original[selection], np.zeros(selection.sum()), half[selection])
            new = summarize(shuffled[selection], np.zeros(selection.sum()), half[selection])
            self.assertEqual(old['live_missing'], new['live_missing'])
            self.assertEqual(sorted(old['gap_frames']), sorted(new['gap_frames']))

    def test_shuffled_gaps_handle_empty_and_full_halves(self):
        mask = np.r_[np.zeros(10, dtype=bool), np.ones(10, dtype=bool)]
        np.testing.assert_array_equal(shuffle_gaps(mask, np.repeat([1, 2], 10), 3), mask)

    def test_live_crossing_stress_starts_in_play_and_can_span_stoppages(self):
        labels = np.tile(np.r_[np.zeros(40, dtype=int), np.full(20, 2)], 100)
        half = np.repeat([1, 2], len(labels) // 2)
        mask = make_mask(labels, half, 'live_crossing', 11, np.zeros(len(labels), bool))
        starts = np.flatnonzero(mask & ~np.r_[False, mask[:-1]])
        self.assertGreater(len(starts), 0)
        self.assertTrue(np.all(labels[starts] != 2))
        self.assertTrue(np.any(mask & (labels == 2)))
        self.assertTrue(np.array_equal(mask, make_mask(labels, half, 'live_crossing', 11, np.zeros(len(labels), bool))))

    def test_apply_does_not_modify_source_and_reports_gap_lengths_by_half(self):
        ball = np.arange(16, dtype=float).reshape(8, 2)
        original = ball.copy()
        mask = np.array([False, True, True, True, True, True, False, False])
        labels = np.array([0, 0, 2, 2, 1, 1, 0, 2])
        half = np.array([1, 1, 1, 1, 2, 2, 2, 2])
        result = apply_mask(ball, mask)
        np.testing.assert_array_equal(ball, original)
        self.assertTrue(np.isnan(result[mask]).all())
        np.testing.assert_array_equal(result[~mask], original[~mask])
        stats = summarize(mask, labels, half)
        self.assertEqual(stats['live_missing'], 3)
        self.assertEqual(stats['live_total'], 5)
        self.assertEqual(stats['dead_missing'], 2)
        self.assertEqual(stats['dead_total'], 3)
        self.assertEqual(stats['gap_frames'], [3, 2])

    def test_rejects_unknown_condition(self):
        with self.assertRaises(ValueError):
            make_mask(np.zeros(10), np.ones(10), 'unknown', 1)


if __name__ == '__main__':
    unittest.main()
