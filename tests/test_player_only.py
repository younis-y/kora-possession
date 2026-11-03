import unittest

import numpy as np
import pandas as pd

from kora.features import build
from kora.player_only import PLAYER_FEATURES, player_features, fit_player_model


class PlayerOnlyTests(unittest.TestCase):
    def test_features_do_not_depend_on_ball(self):
        n = 60
        players = np.tile(np.array([[[-30, 0], [-10, 4], [10, -4], [30, 0]]], dtype=np.float32), (n, 1, 1))
        players[:, :, 0] += np.arange(n, dtype=np.float32)[:, None] / 25
        teams = np.array([0, 0, 1, 1])
        section = np.ones(n, dtype=np.int8)
        with_ball = build(players, teams, np.zeros((n, 2), dtype=np.float32), section)
        actual = player_features(players, teams, section)
        pd.testing.assert_frame_equal(actual, with_ball[actual.columns])

    def test_model_trains_on_live_teams_only(self):
        x = pd.DataFrame(np.zeros((90, len(PLAYER_FEATURES)), dtype=np.float32), columns=PLAYER_FEATURES)
        x['b_seen'] = np.arange(90)
        y = np.repeat([0, 1, 2], 30)
        model = fit_player_model(x, y)
        np.testing.assert_array_equal(model.classes_, [0, 1])
        self.assertNotIn('b_seen', model.feature_names_in_)

    def test_model_rejects_missing_team(self):
        x = pd.DataFrame(np.zeros((20, len(PLAYER_FEATURES)), dtype=np.float32), columns=PLAYER_FEATURES)
        with self.assertRaises(ValueError):
            fit_player_model(x, np.zeros(20, dtype=np.int8))


if __name__ == '__main__':
    unittest.main()
