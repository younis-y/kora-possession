"""Synthetic checks for RESEARCH.md Y13 (touch-event model) and the ceiling oracles."""
import unittest

import numpy as np
import pandas as pd

from kora import events
from kora.ceiling import oracles


def frame(n=200, kinks=(50, 120)):
    """Ball flying freely with sharp velocity changes at `kinks`; every other column constant."""
    acc = np.full(n, 0.1, np.float32)
    for k in kinks:
        acc[k - 2:k + 3] = [1.0, 2.0, 4.0, 2.0, 1.0]
    cols = {c: np.ones(n, np.float32) for c in events.KIN}
    cols.update(b_acc=acc, touch_diff=np.zeros(n, np.float32))
    return pd.DataFrame(cols)


class EventTests(unittest.TestCase):
    def test_candidates_are_kinks_and_free_flight_gives_none(self):
        np.testing.assert_array_equal(events.candidates(frame()), [50, 120])
        self.assertEqual(len(events.candidates(frame(kinks=()))), 0)

    def test_labels_mark_handovers_into_each_team_only(self):
        y = np.zeros(200, np.int8)
        y[52:118] = 1                       # A takes over at 52, H back at 118
        y[150:] = 2                         # dead ball: not an event
        half = np.ones(200, np.int8)
        np.testing.assert_array_equal(events.labels(y, half, np.array([50, 120, 152])), [2, 1, 0])

    def test_table_has_one_row_per_candidate_with_neighbour_gaps(self):
        X = frame()
        t = events.table(X, events.candidates(X))
        self.assertEqual(len(t), 2)
        self.assertAlmostEqual(t["ev_next_t"][0], 70 / 25, places=5)
        self.assertTrue(np.isnan(t["ev_prev_acc"][0]))

    def test_frames_put_probabilities_on_candidates_only(self):
        class Stub:                                       # handover into H when touch_diff > 0
            classes_ = np.array([0, 1, 2])
            def predict_proba(self, t):
                h = (t["touch_diff"].to_numpy() > 0).astype(float)
                return np.c_[np.zeros_like(h), 0.8 * h + 0.1, 0.9 - 0.8 * h]
        X = frame()
        X.loc[50, "touch_diff"] = 1.0
        p = events.frames(Stub(), X)
        self.assertEqual(p.shape, (200, 2))
        self.assertTrue((p[np.setdiff1d(np.arange(200), [50, 120])] == 0).all())
        self.assertGreater(p[50, 0], p[50, 1])
        self.assertGreater(p[120, 1], p[120, 0])


class CeilingTests(unittest.TestCase):
    def test_swallowed_oracle_restores_only_the_swallowed_spell(self):
        y = np.array([0] * 40 + [1] * 10 + [0] * 40)
        p = np.zeros(90, int)
        q = oracles(y, p)["swallowed"]
        np.testing.assert_array_equal(q, y)
        np.testing.assert_array_equal(oracles(y, y.copy())["swallowed"], y)


if __name__ == "__main__":
    unittest.main()
