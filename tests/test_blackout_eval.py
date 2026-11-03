import unittest
import copy
import subprocess
import sys
from unittest.mock import patch

import pandas as pd

import numpy as np

from kora.blackout_eval import conditions, training_matches, scores, training_data
from kora.external import MATCHES


class BlackoutEvaluationTests(unittest.TestCase):
    def test_cli_rejects_unknown_or_unenabled_duration_condition(self):
        for arguments, message in ((['--conditions', 'unknown'], 'invalid choice'),
                                   (['--conditions', 'duration_matched'], 'requires --duration-matched')):
            result = subprocess.run([sys.executable, '-m', 'kora.blackout_eval',
                                     '--held-out', 'J03WPY', *arguments],
                                    capture_output=True, text=True, check=False)
            self.assertNotEqual(result.returncode, 0)
            self.assertNotIn('training condition:', result.stdout)
            self.assertIn(message, result.stderr)

    def test_training_partition_excludes_held_out_match(self):
        self.assertEqual(training_matches('b', ['a', 'b', 'c']), ['a', 'c'])
        with self.assertRaises(ValueError):
            training_matches('missing', ['a', 'b', 'c'])

    def test_training_data_never_loads_held_out_match(self):
        data = {'players': np.zeros((12, 2, 2)), 'ball': np.zeros((12, 2)),
                'label': np.tile([0, 1, 2], 4), 'half': np.ones(12),
                'team': np.array([0, 1])}
        def load_training(match):
            if match == 'held':
                raise AssertionError('loaded held-out labels')
            return data
        with patch('kora.blackout_eval.MATCHES', {'a': None, 'held': None, 'b': None}), \
             patch('kora.blackout_eval.load', side_effect=load_training) as loader, \
             patch('kora.blackout_eval.TRAIN_COPIES', 1), \
             patch('kora.blackout_eval.degrade', side_effect=lambda p, b, y, rng: (p, b)), \
             patch('kora.blackout_eval.build', side_effect=lambda p, t, b, h: pd.DataFrame({'feature': np.zeros(len(h))})), \
             patch('kora.blackout_eval.mirror', side_effect=lambda p, t, b: (p, t, b)):
            training_data('held')
        self.assertEqual([call.args[0] for call in loader.call_args_list], ['a', 'b'])

    def test_independent_training_changes_ball_without_changing_labels(self):
        data = {'players': np.zeros((600, 2, 2)), 'ball': np.ones((600, 2)),
                'label': np.tile([0, 1, 2], 200), 'half': np.repeat([1, 2], 300),
                'team': np.array([0, 1])}
        def build_ball(p, t, b, h):
            return pd.DataFrame({'visible': np.isfinite(b[:, 0]).astype(float)})
        with patch('kora.blackout_eval.MATCHES', {'a': None, 'held': None}), \
             patch('kora.blackout_eval.load', return_value=data), \
             patch('kora.blackout_eval.TRAIN_COPIES', 1), \
             patch('kora.blackout_eval.SUBSAMPLE', 1), \
             patch('kora.blackout_eval.degrade', side_effect=lambda p, b, y, rng: (p, b)), \
             patch('kora.blackout_eval.build', side_effect=build_ball), \
             patch('kora.blackout_eval.mirror', side_effect=lambda p, t, b: (p, t, b)):
            current_x, current_y = training_data('held', 'current')
            masked_x, masked_y = training_data('held', 'independent')
        np.testing.assert_array_equal(current_y, masked_y)
        self.assertTrue((current_x == 1).all())
        self.assertTrue((masked_x == 0).any())
        np.testing.assert_array_equal(masked_x[:600], masked_x[600:])

    def test_mixed_training_masks_only_odd_copies(self):
        data = {'players': np.zeros((600, 2, 2)), 'ball': np.ones((600, 2)),
                'label': np.tile([0, 1, 2], 200), 'half': np.repeat([1, 2], 300),
                'team': np.array([0, 1])}
        def build_ball(p, t, b, h):
            return pd.DataFrame({'visible': np.isfinite(b[:, 0]).astype(float)})
        with patch('kora.blackout_eval.MATCHES', {'a': None, 'held': None}), \
             patch('kora.blackout_eval.load', return_value=data), \
             patch('kora.blackout_eval.TRAIN_COPIES', 2), \
             patch('kora.blackout_eval.SUBSAMPLE', 1), \
             patch('kora.blackout_eval.degrade', side_effect=lambda p, b, y, rng: (p, b)), \
             patch('kora.blackout_eval.build', side_effect=build_ball), \
             patch('kora.blackout_eval.mirror', side_effect=lambda p, t, b: (p, t, b)):
            x, _ = training_data('held', 'mixed')
        self.assertTrue((x[:1200] == 1).all())
        self.assertTrue((x[1200:] == 0).any())

    def test_training_seed_is_repeatable_and_changes_blackout(self):
        data = {'players': np.zeros((600, 2, 2)), 'ball': np.ones((600, 2)),
                'label': np.tile([0, 1, 2], 200), 'half': np.repeat([1, 2], 300),
                'team': np.array([0, 1])}
        def build_ball(p, t, b, h):
            return pd.DataFrame({'visible': np.isfinite(b[:, 0]).astype(float)})
        with patch('kora.blackout_eval.MATCHES', {'a': None, 'held': None}), \
             patch('kora.blackout_eval.load', return_value=data), \
             patch('kora.blackout_eval.TRAIN_COPIES', 1), \
             patch('kora.blackout_eval.SUBSAMPLE', 1), \
             patch('kora.blackout_eval.degrade', side_effect=lambda p, b, y, rng: (p, b)), \
             patch('kora.blackout_eval.build', side_effect=build_ball), \
             patch('kora.blackout_eval.mirror', side_effect=lambda p, t, b: (p, t, b)):
            one, _ = training_data('held', 'independent', 17)
            repeat, _ = training_data('held', 'independent', 17)
            other, _ = training_data('held', 'independent', 18)
        np.testing.assert_array_equal(one, repeat)
        self.assertFalse(np.array_equal(one, other))

    def test_conditions_are_reproducible_and_do_not_mutate_input(self):
        n = 5000
        data = {'players': np.ones((n, 4, 2)), 'ball': np.ones((n, 3)),
                'label': np.tile([0, 1, 2, 2], n // 4),
                'half': np.repeat([1, 2], n // 2)}
        first = conditions(data, 13)
        second = conditions(data, 13)
        self.assertEqual(list(first), ['clean', 'clean_ball', 'current', 'independent', 'live', 'live_crossing', 'clean_players_current', 'clean_players_independent', 'shuffled_current', 'clean_players_shuffled_current'])
        for name in first:
            p, b = first[name]
            np.testing.assert_array_equal(p, second[name][0])
            np.testing.assert_array_equal(b, second[name][1])
        np.testing.assert_array_equal(first['current'][0], first['clean_ball'][0])
        np.testing.assert_array_equal(first['clean_ball'][1], data['ball'][:, :2])
        np.testing.assert_array_equal(first['current'][0], first['independent'][0])
        np.testing.assert_array_equal(first['current'][0], first['live'][0])
        np.testing.assert_array_equal(first['current'][0], first['live_crossing'][0])
        np.testing.assert_array_equal(first['clean'][0], first['clean_players_current'][0])
        np.testing.assert_array_equal(first['clean'][0], first['clean_players_independent'][0])
        np.testing.assert_array_equal(first['current'][1], first['clean_players_current'][1])
        np.testing.assert_array_equal(first['independent'][1], first['clean_players_independent'][1])
        np.testing.assert_array_equal(first['current'][0], first['shuffled_current'][0])
        np.testing.assert_array_equal(first['clean'][0], first['clean_players_shuffled_current'][0])
        np.testing.assert_array_equal(first['shuffled_current'][1], first['clean_players_shuffled_current'][1])
        self.assertEqual(np.isnan(first['current'][1][:, 0]).sum(), np.isnan(first['shuffled_current'][1][:, 0]).sum())
        self.assertFalse(np.isnan(first['clean'][0]).any())
        self.assertFalse(np.isnan(data['players']).any())
        self.assertFalse(np.isnan(data['ball']).any())

    def test_duration_scenario_is_optional_and_preserves_existing_masks(self):
        n = 5000
        data = {'players': np.ones((n, 4, 2)), 'ball': np.ones((n, 3)),
                'label': np.tile([0, 1, 2, 2], n // 4), 'half': np.repeat([1, 2], n // 2)}
        old = conditions(data, 13)
        extended = conditions(data, 13, ([250, 500], 0.4))
        for name in old:
            np.testing.assert_array_equal(old[name][1], extended[name][1])
        self.assertIs(extended['duration_matched'][0], extended['current'][0])
        self.assertIs(extended['clean_players_duration_matched'][0], data['players'])
        np.testing.assert_array_equal(extended['duration_matched'][1], extended['clean_players_duration_matched'][1])

    def test_matched_control_rejects_ball_with_preexisting_gaps(self):
        n = 200
        data = {'players': np.ones((n, 4, 2)), 'ball': np.ones((n, 3)),
                'label': np.tile([0, 1, 2, 2], n // 4),
                'half': np.repeat([1, 2], n // 2)}
        data['ball'][20:30] = np.nan
        with self.assertRaisesRegex(ValueError, 'complete ball tracking'):
            conditions(data, 13)

    def test_scores_perfect_predictions(self):
        y = np.array([0, 0, 1, 1, 2, 2, 0, 0, 1, 1, 2, 2])
        half = np.repeat([1, 2], 6)
        out = scores(y, y, half)
        self.assertAlmostEqual(out['accuracy'], 1.)
        self.assertAlmostEqual(out['macro_f1'], 1.)
        self.assertAlmostEqual(out['spell_f1'], 1.)
        self.assertEqual(out['spell_f1_by_iou'], [1., 1., 1., 1., 1.])


if __name__ == '__main__':
    unittest.main()
