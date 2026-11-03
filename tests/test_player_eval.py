import csv
import io
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import patch

import numpy as np
import pandas as pd

from kora.player_only import PLAYER_FEATURES, fit_player_model
from kora.player_eval import baseline_scores, evaluate_fold, fit_fold, gap_scores, main, pool_scores, training_data


class PlayerEvaluationTests(unittest.TestCase):
    def test_gap_bins_split_at_halves_and_exclude_dead_frames(self):
        # A 24-frame gap straddles halves: two 12-frame (<0.5s) gaps.
        labels = np.zeros(250, dtype=np.int8)
        labels[50:60] = 2
        pred = np.zeros(250, dtype=np.int8)
        pred[100:125] = 1
        half = np.r_[np.ones(12), np.full(238, 2)]
        mask = np.zeros(250, dtype=bool)
        for start, end in ((0, 24), (30, 43), (50, 75), (100, 150)):
            mask[start:end] = True
        scores = gap_scores(labels, pred, half, mask)
        self.assertEqual(scores['live'], {'correct': 215, 'total': 240})
        self.assertEqual(scores['missing_live'], {'correct': 77, 'total': 102})
        self.assertEqual(scores['gap_<0.5s'], {'correct': 24, 'total': 24})
        self.assertEqual(scores['gap_0.5-1s'], {'correct': 13, 'total': 13})
        self.assertEqual(scores['gap_1-2s'], {'correct': 15, 'total': 15})
        self.assertEqual(scores['gap_2-4s'], {'correct': 25, 'total': 50})
        self.assertEqual(scores['gap_4+s'], {'correct': 0, 'total': 0})

    def test_gap_boundary_four_seconds_and_empty_support(self):
        labels = np.zeros(100, dtype=np.int8)
        scores = gap_scores(labels, labels, np.ones(100), np.ones(100, dtype=bool))
        self.assertEqual(scores['gap_4+s'], {'correct': 100, 'total': 100})
        empty = gap_scores(np.full(100, 2), labels, np.ones(100), np.zeros(100, bool))
        self.assertEqual(empty['live'], {'correct': 0, 'total': 0})
        with self.assertRaises(ValueError):
            gap_scores(labels, labels[:-1], np.ones(100), np.ones(100, bool))

    def test_pool_uses_frame_counts_and_keeps_conditions_separate(self):
        rows = [{'predictor': 'player_only', 'condition': 'current', 'scope': 'live', 'correct': 1, 'total': 2},
                {'predictor': 'player_only', 'condition': 'current', 'scope': 'live', 'correct': 8, 'total': 8},
                {'predictor': 'always_H', 'condition': 'current', 'scope': 'live', 'correct': 0, 'total': 3}]
        pooled = pool_scores(rows)
        self.assertEqual(pooled[('player_only', 'current', 'live')], {'correct': 9, 'total': 10})
        self.assertEqual(pooled[('always_H', 'current', 'live')], {'correct': 0, 'total': 3})
        self.assertEqual(rows[0]['total'], 2)

    def test_fold_excludes_held_out_features_and_labels(self):
        def data(value, label):
            return (pd.DataFrame(np.full((500, len(PLAYER_FEATURES)), value),
                                 columns=PLAYER_FEATURES), np.full(500, label))
        training = {'a': data(0, 0), 'b': data(99, 2), 'c': data(1, 1)}
        with patch('kora.player_eval.fit_player_model', wraps=fit_player_model) as fit:
            model = fit_fold(training, 'b')
        x, y = fit.call_args.args
        self.assertEqual(len(x), 1000)
        self.assertFalse((x.to_numpy() == 99).any())
        self.assertEqual(set(y), {0, 1})
        np.testing.assert_array_equal(model.classes_, [0, 1])
        with self.assertRaises(ValueError):
            fit_fold(training, 'missing')

    def test_training_copies_are_reproducible_player_only_and_mirrored(self):
        n = 120
        players = np.tile([[[-30., 0.], [-10., 4.], [10., -4.], [30., 0.]]], (n, 1, 1))
        data = {'players': players, 'ball': np.zeros((n, 3)), 'team': np.array([0, 0, 1, 1]),
                'label': np.tile([0, 1, 2], 40), 'half': np.ones(n)}
        with patch('kora.player_eval.load', return_value=data):
            first = training_data('J03WPY')
            second = training_data('J03WPY')
        pd.testing.assert_frame_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])
        self.assertEqual(tuple(first[0].columns), PLAYER_FEATURES)
        self.assertEqual(len(first[1]), 80)
        np.testing.assert_array_equal(first[1][:20], np.zeros(20))
        np.testing.assert_array_equal(first[1][20:40], np.ones(20))
        self.assertFalse(np.isnan(players).any())

    def test_evaluation_uses_the_existing_matched_masks(self):
        n = 250
        labels = np.tile([0, 1, 2, 2, 0], n // 5)
        data = {'players': np.zeros((n, 4, 2)), 'ball': np.zeros((n, 3)),
                'team': np.array([0, 0, 1, 1]), 'label': labels, 'half': np.ones(n)}
        class Model:
            def predict(self, features):
                return np.zeros(len(features), dtype=np.int8)
        from kora.blackout_eval import conditions
        expected = conditions(data, 11)
        scores = evaluate_fold(Model(), data, 11)
        self.assertEqual(list(scores), list(expected))
        for name, (_, ball) in expected.items():
            missing_live = (~np.isfinite(ball).all(axis=1)) & (labels < 2)
            self.assertEqual(scores[name]['missing_live']['total'], int(missing_live.sum()))
            self.assertEqual(scores[name]['live'], {'correct': 100, 'total': 150})
        self.assertFalse(np.isnan(data['players']).any())

    def test_default_cli_reports_seven_folds_and_pooled_support(self):
        from kora.external import MATCHES
        n = 120
        data = {'players': np.tile([[[-30., 0.], [-10., 4.], [10., -4.], [30., 0.]]], (n, 1, 1)),
                'ball': np.zeros((n, 3)), 'team': np.array([0, 0, 1, 1]),
                'label': np.tile([0, 1, 2], 40), 'half': np.ones(n)}
        output = io.StringIO()
        with patch('kora.player_eval.load', return_value=data), \
                patch('kora.player_eval.SUBSAMPLE', 1), \
                patch('sys.argv', ['player_eval', '--seeds', '11']), \
                redirect_stdout(output), redirect_stderr(io.StringIO()):
            main()
        rows = list(csv.DictReader(io.StringIO(output.getvalue())))
        self.assertEqual({r['match'] for r in rows}, {*MATCHES, 'pooled'})
        clean = [r for r in rows if r['condition'] == 'clean' and r['scope'] == 'live'
                 and r['predictor'] == 'player_only']
        self.assertEqual(len(clean), 8)
        self.assertEqual([int(r['total']) for r in clean], [80] * 7 + [560])
        self.assertTrue(all(abs(float(r['accuracy']) - 0.5) <= 0.05 for r in clean))  # constant players: chance level
        self.assertEqual({r['predictor'] for r in rows}, {'player_only', 'always_H'})

    def test_constant_home_baseline_is_fixed_even_when_away_is_majority(self):
        labels = np.array([0, 1, 1, 2], dtype=np.int8)
        players, ball = np.zeros((4, 4, 2)), np.full((4, 2), np.nan)
        data = {'players': players, 'label': labels, 'half': np.ones(4)}
        with patch('kora.player_eval.conditions', return_value={'current': (players, ball)}):
            scores = baseline_scores(data, 11)['current']
        self.assertEqual(scores['live'], {'correct': 1, 'total': 3})
        self.assertEqual(scores['missing_live'], {'correct': 1, 'total': 3})
        self.assertEqual(scores['gap_<0.5s'], {'correct': 1, 'total': 3})

    def test_clean_player_controls_use_clean_predictions(self):
        from kora.blackout_eval import conditions
        n = 5000
        data = {'players': np.zeros((n, 4, 2)), 'ball': np.zeros((n, 3)),
                'team': np.array([0, 0, 1, 1]), 'label': np.zeros(n, np.int8),
                'half': np.ones(n)}
        class Model:
            def predict(self, features):
                return (features['nH'].to_numpy() < 2).astype(np.int8)
        scores = evaluate_fold(Model(), data, 11)
        for name, (players, _) in conditions(data, 11).items():
            home_visible = np.isfinite(players[:, :2, 0]).sum(axis=1)
            self.assertEqual(scores[name]['live']['correct'], int((home_visible == 2).sum()), name)
        self.assertLess(scores['current']['live']['correct'], n)
        self.assertEqual(scores['clean_players_current']['live']['correct'], n)

    def test_new_player_condition_cannot_silently_reuse_degraded_predictions(self):
        players = np.zeros((6, 4, 2))
        ball = np.zeros((6, 2))
        data = {'players': players, 'team': np.array([0, 0, 1, 1]),
                'half': np.ones(6), 'label': np.zeros(6, dtype=np.int8)}
        class Model:
            def predict(self, features):
                return np.zeros(len(features), dtype=np.int8)
        scenarios = {'clean': (players, ball), 'current': (players.copy(), ball),
                     'new_player_loss': (players.copy(), ball)}
        with patch('kora.player_eval.conditions', return_value=scenarios):
            with self.assertRaises(ValueError):
                evaluate_fold(Model(), data, 11)


if __name__ == '__main__':
    unittest.main()
