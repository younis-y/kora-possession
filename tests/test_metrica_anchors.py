import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from kora.metrica_anchors import carry_anchors


def carry(start, end, team="FIFATMA", period=1):
    return {"type": {"name": "CARRY"}, "team": {"id": team},
            "start": {"frame": start}, "end": {"frame": end}, "period": period}


class MetricaAnchorTests(unittest.TestCase):
    def test_sparse_support_includes_endpoints_and_preserves_inputs(self):
        events = [carry(2, 3), carry(6, 6, "FIFATMB"),
                  {"type": {"name": "PASS"}}]
        before = copy.deepcopy(events)
        labels, counts = carry_anchors(events, n_frames=8)
        np.testing.assert_array_equal(labels, ["", "H", "H", "", "", "A", "", ""])
        self.assertEqual(events, before)
        self.assertEqual(counts["carry_events"], 2)
        self.assertEqual(counts["accepted_carries"], 2)
        self.assertEqual(counts["labeled_home_frames"], 2)
        self.assertEqual(counts["labeled_away_frames"], 1)
        self.assertEqual(counts["unknown_frames"], 5)

    def test_opposing_overlap_remains_unknown_even_after_later_carries(self):
        events = [carry(2, 5), carry(4, 7, "FIFATMB"), carry(4, 6)]
        labels, counts = carry_anchors(events, n_frames=8)
        np.testing.assert_array_equal(labels, ["", "H", "H", "", "", "", "A", ""])
        self.assertEqual(counts["conflict_frames"], 3)
        self.assertEqual(counts["conflicting_carry_pairs"], 2)
        self.assertEqual(counts["carry_frame_sum"], 11)

    def test_opposing_intervals_touching_at_endpoint_conflict(self):
        labels, counts = carry_anchors([carry(1, 3), carry(3, 5, "FIFATMB")], 5)
        np.testing.assert_array_equal(labels, ["H", "H", "", "A", "A"])
        self.assertEqual(counts["conflict_frames"], 1)

    def test_invalid_bounds_and_types_are_rejected_without_clipping(self):
        invalid = [carry(0, 2), carry(7, 9), carry(5, 4), carry(2.0, 3),
                   carry(True, 3), carry(1, 2, period=3), carry(1, 2, period=2),
                   carry(1, 2, period=True), carry("1", 2),
                   {"type": {"name": "CARRY"}, "team": {"id": "FIFATMA"}}]
        labels, counts = carry_anchors(invalid, n_frames=8)
        self.assertTrue(np.all(labels == ""))
        self.assertEqual(counts["rejected_invalid"], len(invalid))

    def test_unknown_team_does_not_default_to_away(self):
        labels, counts = carry_anchors([carry(1, 2, "other"), carry(3, 4, None)], 5)
        self.assertTrue(np.all(labels == ""))
        self.assertEqual(counts["rejected_unknown_team"], 2)

    def test_half_boundary_is_respected(self):
        events = [carry(69660, 69662), carry(69662, 69663, "FIFATMB", 2)]
        labels, counts = carry_anchors(events, n_frames=69663)
        self.assertEqual(counts["rejected_invalid"], 1)
        self.assertEqual(labels[69660], "")
        np.testing.assert_array_equal(labels[-2:], ["A", "A"])

    def test_empty_events_remain_unknown(self):
        labels, counts = carry_anchors([], 4)
        self.assertEqual(labels.tolist(), ["", "", "", ""])
        self.assertEqual(counts["unknown_frames"], 4)
        self.assertEqual(counts["conflict_frames"], 0)

    def test_invalid_frame_count_is_rejected(self):
        for n in (0, -1, 1.5, True):
            with self.subTest(n=n), self.assertRaises(ValueError):
                carry_anchors([], n)

    def test_cli_rejects_unpinned_data(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as folder:
            path = Path(folder) / "events.json"
            path.write_text(json.dumps({"data": []}))
            result = subprocess.run(
                [sys.executable, "-m", "kora.metrica_anchors", str(path)],
                capture_output=True, text=True, check=False,
            )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("SHA-256", result.stderr)
        self.assertEqual(result.stdout, "")


if __name__ == "__main__":
    unittest.main()
