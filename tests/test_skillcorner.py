import json
import unittest

import numpy as np

import pandas as pd

from kora.skillcorner import UNKNOWN, label_events, parse, resample

META = {"home_team": {"id": 1}, "players": [{"id": 10, "team_id": 1}, {"id": 20, "team_id": 2}]}


def frame(k, period, px, ball_ok, group):
    return json.dumps({"frame": k, "period": period,
                       "ball_data": {"x": 1.0 * k, "y": 0.0, "z": 0.0, "is_detected": ball_ok},
                       "possession": {"player_id": None, "group": group},
                       "player_data": [{"x": px, "y": 0.0, "player_id": 10, "is_detected": True},
                                       {"x": 5.0, "y": 5.0, "player_id": 20, "is_detected": False}]})


class SkillCornerTests(unittest.TestCase):
    def test_parse_hides_undetected_objects_and_maps_groups(self):
        lines = [frame(0, None, 0, True, None), frame(1, 1, 1.0, True, "home team"),
                 frame(2, 1, 2.0, False, "away team"), frame(3, 2, 3.0, True, None)]
        d = parse(META, lines)
        np.testing.assert_array_equal(d["half"], [1, 1, 2])
        np.testing.assert_array_equal(d["team"], [0, 1])
        self.assertTrue(np.isnan(d["players"][:, 1]).all())          # extrapolated player 20
        self.assertTrue(np.isnan(d["ball"][1]).all())                  # undetected ball
        np.testing.assert_array_equal(d["label"], [0, 1, UNKNOWN])

    def test_resample_interpolates_and_keeps_gaps(self):
        n = 11
        d = {"half": np.ones(n, np.int8), "team": np.array([0], np.int8),
             "players": np.arange(n, dtype=np.float32)[:, None, None].repeat(2, 2),
             "ball": np.arange(n, dtype=np.float32)[:, None].repeat(2, 1), "label": np.zeros(n, np.int8)}
        d["ball"][5] = np.nan
        r = resample(d, 25)
        self.assertEqual(len(r["half"]), 26)                             # 1 s at 10 fps -> 26 frames at 25 fps
        self.assertAlmostEqual(float(r["players"][5, 0, 0]), 2.0)        # t = 0.2 s
        self.assertTrue(np.isnan(r["ball"][12:13]).all())                # t = 0.48 s touches the gap

    def test_label_events_gaps_interruptions_and_unknown(self):
        meta = {"home_team": {"id": 1}, "match_periods": [
            {"period": 1, "start_frame": 10, "end_frame": 159}, {"period": 2, "start_frame": 170, "end_frame": 189}]}
        ev = pd.DataFrame([   # frames inclusive; team 1 is home
            (1, 15, 19, 1, "player_possession", "unknown", None, None),
            (1, 22, 24, 2, "player_possession", "pass_reception", None, "throw_in_for"),
            (1, 50, 52, 2, "player_possession", "throw_in_reception", "throw_in_for", None),
            (1, 53, 53, 1, "passing_option", "unknown", None, None),                  # not a possession
            (1, 60, 61, 1, "player_possession", "recovery", None, None),
            (1, 70, 71, 2, "player_possession", "pass_reception", None, None),
            (1, 120, 121, 2, "player_possession", "recovery", None, None),
            (1, 150, 151, 1, "player_possession", "free_kick_interception", "free_kick_against", None),
            (2, 175, 176, 1, "player_possession", "unknown", None, None),
        ], columns=["period", "frame_start", "frame_end", "team_id", "event_type", "start_type",
                    "game_interruption_before", "game_interruption_after"])
        lab = label_events(meta, ev)
        self.assertEqual(len(lab), 150 + 20)
        f = lambda *frames: [int(lab[k - 10]) for k in frames]          # period 1 frame numbers
        g = lambda *frames: [int(lab[150 + k - 170]) for k in frames]   # period 2
        self.assertEqual(f(14, 15, 19, 20, 21, 22), [UNKNOWN, 0, 0, 0, 0, 1])   # live gap: kept by the passer
        self.assertEqual(f(25, 29, 30, 39, 40, 49, 50), [1, 1, 2, 2, 1, 1, 1])  # ball out, dead, restart
        self.assertEqual(f(53, 59, 60, 62, 69, 70), [1, 1, 0, 0, 0, 1])
        self.assertEqual(set(f(*range(72, 120))), {UNKNOWN})                   # live gap over FLIGHT_S
        self.assertEqual(f(122, 126, 127, 139, 140, 149, 150), [1, 1, 2, 2, 1, 1, 0])  # restart intercepted
        self.assertEqual(set(f(*range(152, 160))), {UNKNOWN})
        self.assertEqual(g(170, 174, 175, 176, 177), [UNKNOWN, UNKNOWN, 0, 0, UNKNOWN])


if __name__ == "__main__":
    unittest.main()
