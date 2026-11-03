import copy
import csv
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import patch

import numpy as np
import pandas as pd

from kora.metrica_eval import (AUXILIARY_WEIGHT, auxiliary_training, evaluate_models,
                               fit_fold, load_auxiliary, main, require_cached_matches,
                               training_arrays, validation_partition)
from kora.metrica_tracking import PlayerTracking
from kora.player_only import PLAYER_FEATURES


def carry(first, last, team='FIFATMA'):
    return {'type': {'name': 'CARRY'}, 'team': {'id': team}, 'period': 1,
            'start': {'frame': first}, 'end': {'frame': last}}


def tracking(n=24):
    players = np.tile([[[-30., 0.], [-10., 4.], [10., -4.], [30., 0.]]], (n, 1, 1))
    players[:, :2, 0] += np.arange(n)[:, None] / 10
    return PlayerTracking(np.arange(1, n + 1), players, np.array([0, 0, 1, 1]),
                          np.ones(n), ('h1', 'h2', 'a1', 'a2'), ((1, n),), (105., 68.), 25)


def arrays(value, labels):
    return (pd.DataFrame(np.full((len(labels), len(PLAYER_FEATURES)), value),
                         columns=PLAYER_FEATURES), np.asarray(labels))


class MetricaEvaluationTests(unittest.TestCase):
    def test_alignment_filters_unknown_conflicts_and_then_mirrors_without_mutation(self):
        match = tracking()
        events = [carry(1, 7), carry(7, 13, 'FIFATMB'), carry(19, 19, 'unknown')]
        before, players = copy.deepcopy(events), match.players.copy()
        features, labels, counts = auxiliary_training(match, events)
        # Stride six selects source frames 1,7,13,19; 7 conflicts and 19 is unknown.
        np.testing.assert_array_equal(labels, [0, 1, 1, 0])
        np.testing.assert_allclose(features['cxH'], [-20., -18.8, -20., -20.])
        self.assertEqual(counts['conflict_frames'], 1)
        self.assertEqual(counts['sampled_rows_with_mirror'], 4)
        self.assertEqual(tuple(features.columns), PLAYER_FEATURES)
        self.assertEqual(events, before)
        np.testing.assert_array_equal(match.players, players)

    def test_alignment_rejects_offset_duplicate_frames_wrong_rate_or_half(self):
        match = tracking()
        for bad in (replace(match, frame=match.frame + 1),
                    replace(match, frame=np.ones(24)), replace(match, fps=24),
                    replace(match, section=np.full(24, 2))):
            with self.subTest(bad=bad.frame[:2]), self.assertRaises(ValueError):
                auxiliary_training(bad, [carry(1, 2)])

    def test_empty_auxiliary_does_not_make_unknown_frames_training_targets(self):
        features, labels, counts = auxiliary_training(tracking(), [])
        self.assertEqual(len(features), 0)
        self.assertEqual(len(labels), 0)
        self.assertEqual(counts['sampled_rows_with_mirror'], 0)

    def test_outer_partition_dead_exclusion_and_fixed_weights_preserve_inputs(self):
        train = {'train': arrays(1., [0, 1, 2]), 'held': arrays(999., [0, 1])}
        aux = arrays(3., [1, 0])
        before = train['train'][0].copy()
        features, labels, weights = training_arrays(train, 'held', aux, 0.25)
        self.assertEqual(features.iloc[:, 0].tolist(), [1., 1., 3., 3.])
        np.testing.assert_array_equal(labels, [0, 1, 1, 0])
        np.testing.assert_array_equal(weights, [1., 1., 0.25, 0.25])
        self.assertEqual(weights[2:].sum(), 0.5)
        pd.testing.assert_frame_equal(before, train['train'][0])
        baseline = training_arrays(train, 'held', aux, 0.)
        self.assertEqual(len(baseline[0]), 2)
        np.testing.assert_array_equal(baseline[2], [1., 1.])

    def test_invalid_partition_labels_weights_or_absent_teams_fail(self):
        train = {'train': arrays(1., [0, 1]), 'held': arrays(999., [0, 1])}
        aux = arrays(3., [0, 1])
        for weight in (-1, np.nan, np.inf):
            with self.subTest(weight=weight), self.assertRaises(ValueError):
                training_arrays(train, 'held', aux, weight)
        for invalid in ({'held': train['held']},
                        {'held': train['held'], 'train': arrays(1., [0, 9])},
                        {'held': train['held'], 'train': arrays(1., [0, 0])}):
            with self.subTest(invalid=list(invalid)), self.assertRaises(ValueError):
                training_arrays(invalid, 'held', aux, 0.25)
        with self.assertRaises(ValueError):
            training_arrays(train, 'unknown', aux, 0.25)
        with self.assertRaises(ValueError):
            training_arrays(train, 'held', arrays(3., [0, 2]), 0.25)

    def test_fixed_validation_match_excludes_outer_and_preserves_exact_membership(self):
        train = {'J03WN1': arrays(1., [0, 1, 2, 0]),
                 'J03WR9': arrays(11., [2, 1, 0, 2, 1]),
                 'J03WPY': arrays(999., [0, 1])}
        before = {name: (x.copy(), y.copy()) for name, (x, y) in train.items()}
        remaining, validation = validation_partition(train, 'J03WPY')
        repeated, second_validation = validation_partition(dict(reversed(list(train.items()))), 'J03WPY')
        self.assertEqual(set(remaining), {'J03WN1'})
        self.assertEqual(set(repeated), {'J03WN1'})
        x, y, weights = validation
        self.assertEqual(x.iloc[:, 0].tolist(), [11., 11., 11.])
        np.testing.assert_array_equal(y, [1, 0, 1])
        np.testing.assert_array_equal(weights, [1., 1., 1.])
        pd.testing.assert_frame_equal(x, second_validation[0])
        np.testing.assert_array_equal(y, second_validation[1])
        for name, (x, y) in train.items():
            pd.testing.assert_frame_equal(x, before[name][0])
            np.testing.assert_array_equal(y, before[name][1])

    def test_validation_partition_rejects_missing_validation_or_outer_validation_collision(self):
        for train, outer in (({'J03WN1': arrays(1., [0, 1])}, 'J03WPY'),
                             ({'J03WR9': arrays(1., [0, 1])}, 'J03WR9'),
                             ({'J03WR9': arrays(1., [0, 1])}, 'J03WPY')):
            with self.subTest(outer=outer, matches=list(train)), self.assertRaises(ValueError):
                validation_partition(train, outer)

    def test_both_fits_use_exact_german_validation_and_auxiliary_only_for_training(self):
        from kora.train import classifier
        labels = np.tile([0, 1], 300)
        features = arrays(0., labels)[0]
        features['cxH'] = labels * 10.
        train = {'J03WN1': (features, labels), 'J03WR9': arrays(11., np.tile([1, 0], 200)),
                 'J03WPY': arrays(999., [0, 1])}
        observed = []
        baseline_probabilities = None
        for weight in (0., AUXILIARY_WEIGHT):
            model = classifier()
            with patch('kora.metrica_eval.classifier', return_value=model), \
                    patch.object(model, 'fit', wraps=model.fit) as fit:
                fit_fold(train, 'J03WPY', arrays(777., [0, 1]), weight)
            x, y = fit.call_args.args
            observed.append(fit.call_args.kwargs)
            self.assertFalse(x.isin([11., 999.]).any().any())
            self.assertEqual(int((x.iloc[:, 0] == 777.).sum()), 2 if weight else 0)
            self.assertEqual(len(x), 602 if weight else 600)
            np.testing.assert_array_equal(y[:600], labels)
            np.testing.assert_array_equal(observed[-1]['sample_weight'][:600], np.ones(600))
            if weight:
                np.testing.assert_array_equal(observed[-1]['sample_weight'][-2:], [0.25, 0.25])
            self.assertEqual(len(model.validation_score_), model.n_iter_ + 1)
            self.assertEqual(model.random_state, 2027)
            if not weight:
                baseline_probabilities = model.predict_proba(features)
        pd.testing.assert_frame_equal(observed[0]['X_val'], observed[1]['X_val'])
        self.assertEqual(observed[0]['X_val'].iloc[:, 0].tolist(), [11.] * 400)
        np.testing.assert_array_equal(observed[0]['y_val'], np.tile([1, 0], 200))
        np.testing.assert_array_equal(observed[0]['y_val'], observed[1]['y_val'])
        np.testing.assert_array_equal(observed[0]['sample_weight_val'], np.ones(400))
        np.testing.assert_array_equal(observed[0]['sample_weight_val'], observed[1]['sample_weight_val'])
        repeated = fit_fold(train, 'J03WPY', arrays(777., [0, 1]), 0.)
        np.testing.assert_array_equal(repeated.predict_proba(features), baseline_probabilities)

    def test_paired_evaluation_keeps_masks_and_clean_player_controls_identical(self):
        match = tracking(5000)
        data = {'players': match.players, 'team': match.team, 'half': match.section,
                'ball': np.zeros((5000, 3)), 'label': np.tile([0, 0, 2, 0, 0], 1000)}
        before = {name: value.copy() for name, value in data.items()}
        class VisibleHome:
            def predict(self, features):
                return (features['nH'].to_numpy() < 2).astype(np.int8)
        class AlwaysAway:
            def predict(self, features):
                return np.ones(len(features), dtype=np.int8)
        scores = evaluate_models({'home': VisibleHome(), 'away': AlwaysAway()}, data, 11)
        for condition in scores['home']:
            for scope in scores['home'][condition]:
                self.assertEqual(scores['home'][condition][scope]['total'],
                                 scores['away'][condition][scope]['total'])
        self.assertEqual(scores['home']['clean_players_current']['live']['correct'], 4000)
        self.assertEqual(scores['away']['clean']['live']['correct'], 0)
        self.assertLess(scores['home']['current']['live']['correct'], 4000)
        for name in data:
            np.testing.assert_array_equal(data[name], before[name])

    def test_new_player_condition_fails_instead_of_reusing_wrong_features(self):
        match = tracking()
        data = {'players': match.players, 'team': match.team, 'half': match.section,
                'ball': np.zeros((24, 3)), 'label': np.zeros(24, dtype=np.int8)}
        scenarios = {'current': (match.players.copy(), data['ball']),
                     'new': (match.players.copy(), data['ball'])}
        with patch('kora.metrica_eval.conditions', return_value=scenarios):
            with self.assertRaises(ValueError):
                evaluate_models({}, data, 11)

    def test_unpinned_events_rejected_before_tracking_is_read(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as folder:
            events = Path(folder) / 'events.json'
            events.write_text(json.dumps({'data': []}))
            with self.assertRaisesRegex(ValueError, 'SHA-256'):
                load_auxiliary(events, Path(folder) / 'absent.xml', Path(folder) / 'absent.txt')

    def test_verified_events_flow_into_aligned_auxiliary_rows(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as folder:
            events = Path(folder) / 'events.json'
            raw = json.dumps({'data': [carry(1, 1), carry(13, 13, 'FIFATMB')]}).encode()
            events.write_bytes(raw)
            with patch('kora.metrica_eval.EVENTS_SHA256', hashlib.sha256(raw).hexdigest()), \
                    patch('kora.metrica_eval.load_metrica_tracking', return_value=tracking()):
                features, labels, counts = load_auxiliary(events)
            np.testing.assert_array_equal(labels, [0, 1, 1, 0])
            self.assertEqual(len(features), 4)
            self.assertEqual(counts['labeled_home_frames'], 1)

    def test_existing_german_caches_are_accepted_without_creating_files(self):
        from kora.external import MATCHES
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as folder, \
                patch.dict('os.environ', {'MTAH_DATA_DIR': folder}):
            cache = Path(folder) / 'cache'
            cache.mkdir()
            for match in MATCHES:
                (cache / f'idsse_{match}.npz').touch()
            before = sorted(cache.iterdir())
            require_cached_matches()
            self.assertEqual(sorted(cache.iterdir()), before)

    def test_cli_rejects_missing_german_cache_without_downloading(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as folder, \
                patch.dict('os.environ', {'MTAH_DATA_DIR': folder}), \
                patch('sys.argv', ['metrica_eval']), redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                main()
            self.assertEqual(list(Path(folder).iterdir()), [])

    def test_cli_rejects_validation_match_as_outer_test(self):
        errors = io.StringIO()
        with patch('sys.argv', ['metrica_eval', '--held-out', 'J03WR9']), redirect_stderr(errors):
            with self.assertRaises(SystemExit):
                main()
        self.assertIn('invalid choice', errors.getvalue())
        self.assertIn('J03WR9', errors.getvalue())

    def test_cli_emits_both_predictors_on_equal_support(self):
        match = tracking(120)
        data = {'players': match.players, 'team': match.team, 'half': match.section,
                'ball': np.zeros((120, 3)), 'label': np.tile([0, 1, 2], 40)}
        output = io.StringIO()
        with patch('kora.metrica_eval.require_cached_matches'), \
                patch('kora.metrica_eval.load_auxiliary', return_value=(*arrays(2., [0, 1]), {})), \
                patch('kora.metrica_eval.training_data', return_value=arrays(1., np.tile([0, 1], 250))), \
                patch('kora.metrica_eval.load', return_value=data), \
                patch('sys.argv', ['metrica_eval']), redirect_stdout(output), redirect_stderr(io.StringIO()):
            main()
        rows = list(csv.DictReader(io.StringIO(output.getvalue())))
        self.assertEqual({r['predictor'] for r in rows}, {'sportec_only', 'sportec_metrica_0.25'})
        clean = [r for r in rows if r['condition'] == 'clean' and r['scope'] == 'live']
        self.assertEqual([r['total'] for r in clean], ['80', '80'])
        self.assertEqual({r['match'] for r in rows}, {'J03WPY'})


if __name__ == '__main__':
    unittest.main()
