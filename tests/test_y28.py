"""Checks for Y28 (kora/grf.py): GRF labels and conversion, and the variants' cache keys."""
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from kora import grf, variant


class Y28Tests(unittest.TestCase):
    def test_labels(self):
        ball = np.zeros((12, 2))
        ball[5:7, 1] = 0.5                                   # out over the touchline at steps 5-6
        ball[9:11, 0] = 0.95                                 # in the net before the goal at step 11
        owner = np.array([-1, 0, -1, 1, -1, -1, 0, -1, 1, -1, -1, -1])   # step 6: a throw-in held off the pitch
        np.testing.assert_array_equal(grf.labels(ball, owner, [11]), [2, 0, 0, 1, 1, 2, 2, 2, 1, 2, 2, 2])

    def test_load_converts_to_idsse_layout(self):
        rows = []
        for t in range(5):
            for pid, team, x in ((-1, -1, 0.0), (0, 0, -0.5), (100, 1, 0.5)):
                rows.append({"GAME_ID": 3, "TIMESTEP": t, "TEAM_0_SCORE": 0, "TEAM_1_SCORE": 0, "PLAYER_ID": pid,
                             "TEAM_ID": team, "PLAYER_ROLE": "x", "LOCATION_X": x + 0.1 * t * (pid == -1),
                             "LOCATION_Y": 0.42 if pid == 0 else 0.0, "LOCATION_Z": 0.1, "VELOCITY_X": 0,
                             "VELOCITY_Y": 0, "VELOCITY_Z": 0, "ON_BALL": pid == 0 and t == 1})
        with tempfile.TemporaryDirectory() as d:
            old = os.environ.get("MTAH_DATA_DIR")
            os.environ["MTAH_DATA_DIR"] = d
            try:
                pd.DataFrame(rows).to_csv(grf.file_of(3), index=False)
                out = grf.load(3)
            finally:
                os.environ.pop("MTAH_DATA_DIR") if old is None else os.environ.__setitem__("MTAH_DATA_DIR", old)
        self.assertEqual(out["players"].shape, (11, 2, 2))
        np.testing.assert_array_equal(out["team"], [0, 1])
        np.testing.assert_allclose(out["players"][0, 0], [-26.25, 34.0], atol=1e-4)
        np.testing.assert_allclose(out["ball"][5, 0], 0.2 * 52.5, atol=1e-4)    # frame 5 = step 2, interpolated
        np.testing.assert_array_equal(out["label"], [2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0])
        np.testing.assert_array_equal(out["half"], np.ones(11))

    def test_variants_keep_the_base_caches(self):
        V = variant.VARIANTS
        for name, base in (("y28_grf", "mixed"), ("y28_grf_rand", "mixed_rand")):
            self.assertEqual(variant.feature_version(V[name]), variant.feature_version(V[base]), name)
            self.assertNotEqual(variant.stage1_version(V[name]), variant.stage1_version(V[base]), name)
            no_grf = {k: v for k, v in V[name].items() if k != "grf"}
            self.assertEqual(variant.stage1_version(no_grf), variant.stage1_version(V[base]), name)


if __name__ == "__main__":
    unittest.main()
