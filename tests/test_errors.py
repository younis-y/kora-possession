"""Synthetic-only metric checks against the vendored official scorer.

These establish calculation correctness for the constructed inputs, not model
performance on any match. Run from the repository root with standard-library
unittest discovery; no match assets or fitted models are used.
"""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]

import numpy as np
from kora import errors
from kora.vendor import possession_metrics as pm


def codes(text):
    return np.array([{"H": pm.H, "A": pm.A, "D": pm.D, "X": pm.X}[x] for x in text], dtype=np.int8)


class ErrorChecks(unittest.TestCase):
    def analyse(self, truth, prediction, half=None, ball_seen=None):
        y, p = codes(truth), codes(prediction)
        half = np.ones(len(y), dtype=int) if half is None else np.asarray(half)
        seen = np.ones(len(y), dtype=bool) if ball_seen is None else np.asarray(ball_seen)
        originals = [x.copy() for x in (y, p, half, seen)]
        result = errors.analyse(y, p, half, seen)
        for original, after in zip(originals, (y, p, half, seen)):
            np.testing.assert_array_equal(original, after)
        filled = pm.fill_abstentions(p, result["runs"])
        for h, run in zip(result["halves"], result["runs"]):
            a, b = run
            np.testing.assert_array_equal(h["v4"].sum(axis=1), pm.confusion(y[a:b], p[a:b])[:3])
            self.assertEqual(int(h["v4"].sum()), b - a)
            gt_trans, pred_trans = pm.transitions(y, [run]), pm.transitions(filled, [run])
            for tol in pm.TAUS:
                exact = pm.transition_scores(gt_trans, pred_trans, tol, typed="pair")
                reference = pm.transition_scores(gt_trans, pred_trans, tol, typed="to")
                rows = h["v3"]["pairs"][tol]
                for key, source in (("n_true", "n_gt"), ("n_pred", "n_pred"), ("matched", "n_matched"),
                                    ("missed", "n_missed"), ("invented", "n_spurious")):
                    self.assertEqual(sum(row[key] for row in rows.values()), exact[source])
                    self.assertEqual(h["v3"]["official"][tol][key], reference[source])
                self.assertEqual(sum(row["ambiguous_true"] for row in rows.values()), exact["n_ambiguous"])
                self.assertEqual(sum(row["ambiguous_pred"] for row in rows.values()),
                                 pm._count_ambiguous(pred_trans, gt_trans, tol, typed="pair"))
                self.assertEqual(sorted(lat for row in rows.values() for lat in row["latencies"]),
                                 sorted(p[0] - g[0] for g, p in exact["pairs"]))
            windows = h["v3"]["windows"]
            self.assertEqual(sum(windows[name]["frames"] for name in errors.WINDOW_ROWS), b - a)
            self.assertEqual(sum(windows[name]["wrong"] for name in errors.WINDOW_ROWS), int(np.sum(y[a:b] != p[a:b])))
            for key in ("frames", "wrong"):
                self.assertEqual(windows[errors.WINDOW_UNION][key],
                                 sum(windows[name][key] for name in errors.WINDOW_ROWS[:-1]))
            official = pm.spell_scores(y, filled, [run])
            self.assertEqual(h["n_true"], official["n_gt"])
            self.assertEqual(h["n_pred"], official["n_pred"])
            for iou, pairs in h["matches"].items():
                reference = official["per_iou"][iou]
                self.assertEqual(len(pairs), reference["n_matched"])
                self.assertEqual(h["n_true"] - len(pairs), reference["n_missed"])
                self.assertEqual(h["n_pred"] - len(pairs), reference["n_spurious"])
                self.assertEqual(len({i for i, _, _ in pairs}), len(pairs))
                self.assertEqual(len({j for _, j, _ in pairs}), len(pairs))
                v2 = h["v2"][iou]
                for field, expected in (("support", h["n_true"]), ("predicted", h["n_pred"]),
                                        ("matched", reference["n_matched"]),
                                        ("matched_predicted", reference["n_matched"]),
                                        ("missed", reference["n_missed"]),
                                        ("false_predicted", reference["n_spurious"])):
                    self.assertEqual(sum(row[field] for row in v2.values()), expected)
            counts = h["associations"]["counts"]
            self.assertEqual(sum(x["true_spells"] for x in counts.values()), h["n_true"])
            self.assertEqual(sum(x["pred_spells"] for x in counts.values()), h["n_pred"])
        return result

    def check_subtype(self, truth, prediction, kind, subtype, ng, npred):
        h = self.analyse(truth, prediction)["halves"][0]
        self.assertEqual(h["associations"]["subtypes"][kind][subtype],
                         {"components": 1, "true_spells": ng, "pred_spells": npred})

    def test_import_isolation(self):
        script = (
            "import sys\n"
            "import kora.errors\n"
            "assert not any(m == 'sklearn' or m.startswith('sklearn.') for m in sys.modules)\n"
            "assert not any(m in sys.modules for m in "
            "('kora.train', 'kora.boundary', 'kora.decode', 'kora.external'))\n"
        )
        completed = subprocess.run([sys.executable, "-B", "-c", script], cwd=ROOT,
                                   capture_output=True, text=True, timeout=30)
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_perfect_prediction(self):
        h = self.analyse("HHHDDDAAAHH", "HHHDDDAAAHH")["halves"][0]
        self.assertEqual((h["n_true"], h["n_pred"], h["count_error"]), (3, 3, 0))
        self.assertTrue(all(len(pairs) == 3 for pairs in h["matches"].values()))
        self.assertEqual(h["associations"]["counts"]["association"]["components"], 3)
        for kind in ("split", "merge", "complex"):
            self.assertEqual(h["associations"]["counts"][kind]["components"], 0)

    def test_tied_iou_exact_partner(self):
        h = self.analyse("H" * 10, "HHHHDDHHHH")["halves"][0]
        for iou in (0.3, 0.4):
            self.assertEqual(h["matches"][iou], [(0, 0, 0.4)])
        for iou in (0.5, 0.6, 0.7):
            self.assertEqual(h["matches"][iou], [])

    def test_collision_exact_partner(self):
        h = self.analyse("HHHHDDHHHH", "H" * 10)["halves"][0]
        self.assertEqual(h["matches"][0.4], [(0, 0, 0.4)])

    def test_erased_opposite_possession_is_not_a_merge(self):
        associations = self.analyse("H" * 10 + "A" * 10, "H" * 20)["halves"][0]["associations"]
        expected = {"association": (1, 1, 1), "split": (0, 0, 0), "merge": (0, 0, 0),
                    "complex": (0, 0, 0), "isolated true": (1, 1, 0), "isolated predicted": (0, 0, 0)}
        self.assertEqual(associations["counts"],
                         {kind: dict(zip(("components", "true_spells", "pred_spells"), values))
                          for kind, values in expected.items()})
        self.assertEqual(associations["subtypes"], {"split": {}, "merge": {}})
        self.assertEqual((associations["positive_overlaps"], associations["retained_overlaps"],
                          associations["discarded_overlaps"]), (1, 1, 0))

    def test_same_team_merge_leaves_opposite_spell_isolated(self):
        associations = self.analyse("H" * 10 + "A" * 10 + "H" * 10, "H" * 30)["halves"][0]["associations"]
        expected = {"association": (0, 0, 0), "split": (0, 0, 0), "merge": (1, 2, 1),
                    "complex": (0, 0, 0), "isolated true": (1, 1, 0), "isolated predicted": (0, 0, 0)}
        self.assertEqual(associations["counts"],
                         {kind: dict(zip(("components", "true_spells", "pred_spells"), values))
                          for kind, values in expected.items()})
        self.assertEqual(associations["subtypes"], {"split": {}, "merge": {
            "missed opposite possession": {"components": 1, "true_spells": 2, "pred_spells": 1}}})
        self.assertEqual((associations["positive_overlaps"], associations["retained_overlaps"],
                          associations["discarded_overlaps"]), (2, 2, 0))

    def test_split_false_d(self):
        self.check_subtype("H" * 10, "HHHHDDHHHH", "split", "false D interruption", 1, 2)

    def test_split_wrong_team(self):
        self.check_subtype("H" * 10, "HHHHAAHHHH", "split", "wrong-team fragment", 1, 2)

    def test_split_mixed(self):
        self.check_subtype("H" * 10, "HHHDAHHHDH", "split", "mixed", 1, 3)

    def test_merge_missed_stoppage(self):
        self.check_subtype("HHHHDDHHHH", "H" * 10, "merge", "missed stoppage", 2, 1)

    def test_merge_missed_possession(self):
        self.check_subtype("HHHHAAHHHH", "H" * 10, "merge", "missed opposite possession", 2, 1)

    def test_merge_mixed(self):
        self.check_subtype("HHHDAHHHDH", "H" * 10, "merge", "mixed", 3, 1)

    def test_many_to_many(self):
        h = self.analyse("H" * 10 + "DD" + "H" * 10, "HHHHD" + "H" * 17)["halves"][0]
        self.assertEqual(h["associations"]["counts"]["complex"],
                         {"components": 1, "true_spells": 2, "pred_spells": 2})
        self.assertEqual(h["associations"]["retained_overlaps"], 3)

    def test_weak_overlap_discarded(self):
        h = self.analyse("H" * 10 + "D" * 8, "D" * 8 + "H" * 10)["halves"][0]
        a = h["associations"]
        self.assertEqual((a["positive_overlaps"], a["retained_overlaps"], a["discarded_overlaps"]), (1, 0, 1))
        for kind in ("isolated true", "isolated predicted"):
            self.assertEqual(a["counts"][kind]["components"], 1)

    def test_edge_threshold_inclusive(self):
        h = self.analyse("HHHHDD", "DDHHHH")["halves"][0]
        self.assertEqual(h["associations"]["retained_overlaps"], 1)

    def test_association_is_not_match(self):
        h = self.analyse("H" * 100, "D" * 49 + "H" + "D" * 50)["halves"][0]
        self.assertEqual(h["associations"]["counts"]["association"]["components"], 1)
        self.assertTrue(all(not pairs for pairs in h["matches"].values()))

    def test_half_boundary_same_team(self):
        r = self.analyse("H" * 10, "H" * 10, [1] * 5 + [2] * 5)
        self.assertEqual(r["runs"], [(0, 5), (5, 10)])
        self.assertEqual(sum(h["n_true"] for h in r["halves"]), 2)
        self.assertEqual(sum(h["n_pred"] for h in r["halves"]), 2)
        for h in r["halves"]:
            self.assertEqual(h["true_mean_s"], 0.2)
            self.assertTrue(all(pairs == [(0, 0, 1.0)] for pairs in h["matches"].values()))

    def test_leading_trailing_and_all_x(self):
        r = self.analyse("HHHHHDDDDD" + "HHHH", "XXHHHDDXXX" + "XXXX", [1] * 10 + [2] * 4)
        first, second = r["halves"]
        self.assertEqual(first["pred_spells"], [(0, 5, pm.H)])
        self.assertEqual(first["matches"][0.7], [(0, 0, 1.0)])
        self.assertEqual((first["pred_x_frames"], first["wrong_frames"]), (5, 5))
        self.assertEqual((first["pred_d_share"], first["pred_x_share"]), (0.2, 0.5))
        self.assertEqual(second["pred_spells"], [])
        self.assertIsNone(second["pred_mean_s"])
        self.assertIsNone(second["pred_median_s"])
        self.assertEqual((second["pred_x_share"], second["wrong_frames"]), (1.0, 4))
        self.assertTrue(all(not pairs for pairs in second["matches"].values()))

    def test_x_does_not_cross_half_boundary(self):
        first, second = self.analyse("HHHH", "XXHH", [1, 1, 2, 2])["halves"]
        self.assertEqual(first["n_pred"], 0)
        self.assertEqual(second["n_pred"], 1)

    def test_total_time_normalisation(self):
        text = "H" * 225 + "D" * 225 + "A" * 225
        h = self.analyse(text, text)["halves"][0]
        self.assertEqual((h["n_frames"], h["true_per_45"], h["pred_per_45"]), (675, 200, 200))
        self.assertEqual((h["true_mean_s"], h["true_median_s"]), (9.0, 9.0))
        self.assertEqual(h["true_d_share"], 1 / 3)

    def test_no_possession_spells(self):
        h = self.analyse("DDDD", "DDDD")["halves"][0]
        self.assertEqual((h["n_true"], h["n_pred"]), (0, 0))
        self.assertIsNone(h["true_mean_s"])
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            errors.print_report({"fps": 25, "provenance": "synthetic only"},
                                [("S", self.analyse("DDDD", "DDDD"))])
        self.assertIn("n/a", output.getvalue())
        self.assertNotIn("| nan |", output.getvalue())

    def test_v2_duration_edges(self):
        text = "D".join("H" * n for n in (24, 25, 50, 100, 200, 375))
        h = self.analyse(text, text)["halves"][0]
        for iou in pm.SPELL_IOUS:
            for group in ("<1", "1–2", "2–4", "4–8", "8–15", "15+"):
                with self.subTest(iou=iou, group=group):
                    self.assertEqual(h["v2"][iou][group],
                                     {"support": 1, "matched": 1, "missed": 0, "predicted": 1,
                                      "matched_predicted": 1, "false_predicted": 0})

    def test_v2_pair_in_different_groups(self):
        for truth, pred, true_group, pred_group in (("H" * 24 + "D" * 6, "H" * 30, "<1", "1–2"),
                                                   ("H" * 30, "H" * 24 + "D" * 6, "1–2", "<1")):
            h = self.analyse(truth, pred)["halves"][0]
            for iou in pm.SPELL_IOUS:
                with self.subTest(iou=iou, true_group=true_group):
                    true_row, pred_row = h["v2"][iou][true_group], h["v2"][iou][pred_group]
                    self.assertEqual((true_row["support"], true_row["matched"], true_row["predicted"]), (1, 1, 0))
                    self.assertEqual((pred_row["predicted"], pred_row["matched_predicted"], pred_row["support"]), (1, 1, 0))

    def duration_output(self, results):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            errors.print_report({"fps": 25, "provenance": "synthetic only"}, results)
        return output.getvalue().split("V2 duration recovery:", 1)[1]

    def test_v2_empty_groups_and_separate_denominators(self):
        text = self.duration_output([("S", self.analyse("H" * 24 + "D" * 6, "H" * 30))])
        for iou in pm.SPELL_IOUS:
            self.assertIn(f"| {iou} | 15+ | 0 | 0 | 0 | n/a | 0 | 0 | 0 | n/a |", text)
            self.assertIn(f"| {iou} | <1 | 1 | 1 | 0 | 1.000 | 0 | 0 | 0 | n/a |", text)
            self.assertIn(f"| {iou} | 1–2 | 0 | 0 | 0 | n/a | 1 | 1 | 0 | 1.000 |", text)
        self.assertIn("| 15+ | 0 | n/a | n/a | n/a | n/a | n/a |", text)

    def test_v2_extra_fragments_lower_precision(self):
        truth = "H" * 10 + "D" * 20
        baseline = self.analyse(truth, truth)["halves"][0]
        fragmented = self.analyse(truth, "H" * 10 + "D" * 5 + "H" * 5 + "D" * 10)["halves"][0]
        for iou in pm.SPELL_IOUS:
            before, after = baseline["v2"][iou]["<1"], fragmented["v2"][iou]["<1"]
            self.assertEqual(before["matched"] / before["support"], 1)
            self.assertEqual(after["matched"] / after["support"], 1)
            self.assertEqual(before["matched_predicted"] / before["predicted"], 1)
            self.assertEqual(after["matched_predicted"] / after["predicted"], 0.5)
            self.assertEqual(after["false_predicted"], 1)

    def test_v2_zero_rates_with_nonempty_denominators(self):
        text = self.duration_output([("S", self.analyse("H" * 10, "A" * 10))])
        for iou in pm.SPELL_IOUS:
            self.assertIn(f"| {iou} | <1 | 1 | 0 | 1 | 0.000 | 1 | 0 | 1 | 0.000 |", text)

    def test_v2_pools_counts_not_ratios(self):
        results = [("S", self.analyse("H" * 10 + "D" * 20, "H" * 10 + "D" * 5 + "H" * 5 + "D" * 10)),
                   ("T", self.analyse("H" * 10 + "D" * 5 + "H" * 10 + "D" * 5 + "H" * 10,
                                      "H" * 10 + "D" * 30))]
        text = self.duration_output(results)
        for iou in pm.SPELL_IOUS:
            self.assertIn(f"| {iou} | <1 | 4 | 2 | 2 | 0.500 | 3 | 2 | 1 | 0.667 |", text)
        self.assertIn("| <1 | 4 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 |", text)
        self.assertIn("| <1 | 3 | 0.667 | 0.667 | 0.667 | 0.667 | 0.667 |", text)

    def test_v2_reporting_needs_only_cached_counts(self):
        result = self.analyse("HHHHDDHHHH", "H" * 10)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            errors._print_duration_report([{"v2": h["v2"]} for h in result["halves"]])
        text = output.getvalue()
        self.assertIn("| 0.3 | <1 | 2 | 1 | 1 | 0.500 | 1 | 1 | 0 | 1.000 |", text)
        self.assertIn("| 0.5 | <1 | 2 | 0 | 2 | 0.000 | 1 | 0 | 1 | 0.000 |", text)

    def v4_case(self):
        # Six manually specified true-label/visibility strata, with all four predictions.
        strata = (("H", 1, "HHADDX"), ("H", 0, "HADDXXX"),
                  ("A", 1, "HAAADX"), ("A", 0, "HHADX"),
                  ("D", 1, "HHADDXX"), ("D", 0, "HAAADXX"))
        truth = "".join(label * len(pred) for label, _, pred in strata)
        pred = "".join(pred for _, _, pred in strata)
        seen = [visibility for _, visibility, pred in strata for _ in pred]
        return truth, pred, seen

    def frame_output(self, results):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            errors.print_report({"fps": 25, "provenance": "synthetic only"}, results)
        return output.getvalue().split("V4 false D during live play", 1)[1]

    def test_v4_mixed_visibility_hand_counts(self):
        truth, pred, seen = self.v4_case()
        result = self.analyse(truth, pred, ball_seen=seen)
        rows = errors._frame_error_rows(result["halves"][0]["v4"])
        # true, visibility, false D, denominator, X, wrong team, correct team.
        expected_live = [("H", "seen", 2, 6, 1, 1, 2), ("H", "missing", 2, 7, 3, 1, 1),
                         ("A", "seen", 1, 6, 1, 1, 3), ("A", "missing", 1, 5, 1, 2, 1),
                         ("H", "all", 4, 13, 4, 2, 3), ("A", "all", 2, 11, 2, 3, 4),
                         ("live", "seen", 3, 12, 2, 2, 5), ("live", "missing", 3, 12, 4, 3, 2),
                         ("live", "all", 6, 24, 6, 5, 7)]
        fields = ("true", "visibility", "false_d", "denominator", "pred_x", "wrong_team", "correct_team")
        self.assertEqual([tuple(row[key] for key in fields) for row in rows["live"]], expected_live)
        # visibility, predicted H/A subtotal, denominator, H, A, X, correct D.
        expected_dead = [("seen", 3, 7, 2, 1, 2, 2), ("missing", 4, 7, 1, 3, 2, 1),
                         ("all", 7, 14, 3, 4, 4, 3)]
        fields = ("visibility", "pred_live", "denominator", "pred_h", "pred_a", "pred_x", "correct_d")
        self.assertEqual([tuple(row[key] for key in fields) for row in rows["dead"]], expected_dead)
        text = self.frame_output([("S", result)])
        for label, vis, num, den, x, wrong, correct in expected_live:
            self.assertIn(f"| {label} | {vis} | {num} | {den} | {num / den:.3f} | {x} | {wrong} | {correct} |", text)
        for vis, num, den, h, a, x, d in expected_dead:
            self.assertIn(f"| {vis} | {num} | {den} | {num / den:.3f} | {h} | {a} | {x} | {d} |", text)

    def test_v4_raw_x_in_denominators(self):
        result = self.analyse("HHDD", "XDXH")
        matrix = result["halves"][0]["v4"]
        self.assertEqual(int(matrix[:, :, pm.X].sum()), 2)
        rows = errors._frame_error_rows(matrix)
        live = next(row for row in rows["live"] if row["true"] == "live" and row["visibility"] == "all")
        self.assertEqual((live["false_d"], live["pred_x"], live["denominator"]), (1, 1, 2))
        dead = rows["dead"][-1]
        self.assertEqual((dead["pred_live"], dead["pred_x"], dead["denominator"]), (1, 1, 2))
        self.assertIn("| live | all | 1 | 2 | 0.500 | 1 | 0 | 0 |", self.frame_output([("S", result)]))

    def test_v4_outcome_partitions_and_pooled_halves(self):
        truth, pred, seen = self.v4_case()
        whole = self.analyse(truth, pred, ball_seen=seen)["halves"][0]["v4"]
        split = self.analyse(truth, pred, half=[1] * 19 + [2] * 19, ball_seen=seen)
        pooled = sum((h["v4"] for h in split["halves"]), np.zeros((3, 2, 4), dtype=int))
        np.testing.assert_array_equal(pooled, whole)
        for matrix in [whole, pooled, *[h["v4"] for h in split["halves"]]]:
            rows = errors._frame_error_rows(matrix)
            for row in rows["live"]:
                self.assertEqual(row["false_d"] + row["pred_x"] + row["wrong_team"] + row["correct_team"],
                                 row["denominator"])
            for row in rows["dead"]:
                self.assertEqual(row["pred_h"] + row["pred_a"] + row["pred_x"] + row["correct_d"], row["denominator"])
                self.assertEqual(row["pred_live"], row["pred_h"] + row["pred_a"])
        self.assertIn("| live | all | 6 | 24 | 0.250 | 6 | 5 | 7 |", self.frame_output([("S", split)]))

    def test_v4_matrix_matches_confusion(self):
        truth, pred, seen = self.v4_case()
        result = self.analyse(truth, pred, ball_seen=np.array(seen, dtype=float))
        matrix = result["halves"][0]["v4"]
        self.assertEqual(matrix.shape, (3, 2, 4))
        np.testing.assert_array_equal(np.vstack((matrix.sum(axis=1), np.zeros((1, 4), dtype=int))),
                                      pm.confusion(codes(truth), codes(pred)))
        np.testing.assert_array_equal(matrix.sum(axis=(0, 2)), [19, 19])

    def test_v4_all_seen_missing_n_a(self):
        text = self.frame_output([("S", self.analyse("HHDD", "HDHA"))])
        for label in ("H", "A", "live"):
            self.assertIn(f"| {label} | missing | 0 | 0 | n/a | 0 | 0 | 0 |", text)
        self.assertIn("| missing | 0 | 0 | n/a | 0 | 0 | 0 | 0 |", text)
        self.assertIn("Ball visibility (legacy: simulated blackout)", text)
        self.assertIn("Legacy b_seen is simulated blackout, not challenge tracking", text)
        self.assertIn('"2% of errors on true dead ball" figure is a different quantity', text)
        self.assertIn("share of all wrong frames whose true label is D", text)

    def test_v4_pools_counts_not_rates(self):
        results = [("S", self.analyse("HH", "DD")), ("T", self.analyse("HHHHHH", "HHHHHH"))]
        text = self.frame_output(results)
        self.assertIn("| H | all | 2 | 8 | 0.250 | 0 | 0 | 6 |", text)
        self.assertIn("| live | all | 2 | 8 | 0.250 | 0 | 0 | 6 |", text)

    def test_v4_reporting_uses_only_matrix(self):
        truth, pred, seen = self.v4_case()
        matrix = self.analyse(truth, pred, ball_seen=seen)["halves"][0]["v4"]
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            errors._print_frame_report([{"v4": matrix}])
        self.assertIn("| all | 7 | 14 | 0.500 | 3 | 4 | 4 | 3 |", output.getvalue())

    def transition_output(self, results):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            errors.print_report({"fps": 25, "provenance": "synthetic only"}, results)
        return output.getvalue().split("V3 exact-pair transition diagnostics", 1)[1]

    def test_v3_perfect_all_pairs(self):
        text = "".join(label * 60 for label in "HADHDAH")
        result = self.analyse(text, text)
        h = result["halves"][0]
        for tol in pm.TAUS:
            for keys in errors.TRANSITION_CATEGORIES.values():
                row = errors._pool_transition_rows([h["v3"]["pairs"][tol][key] for key in keys])
                self.assertEqual((row["n_true"], row["n_pred"], row["matched"]), (2, 2, 2))
                self.assertEqual(row["latencies"], [0, 0])
                self.assertEqual((row["missed"], row["invented"]), (0, 0))
        self.assertTrue(all(row["wrong"] == 0 for row in h["v3"]["windows"].values()))
        output = self.transition_output([("S", result)])
        for tol in pm.TAUS:
            for category in ("H↔A", "H/A→D", "D→H/A"):
                self.assertIn(f"| {tol} | {category} | category subtotal | 2 | 2 | 2 | 0 | 0 | 1.000 | 1.000 |", output)
        self.assertIn("| H↔A only | 100 | 0 | 0.000 | n/a |", output)

    def test_v3_signed_latency_and_tolerances(self):
        for latency in (10, 20, -5):
            result = self.analyse("H" * 80 + "A" * 120, "H" * (80 + latency) + "A" * (120 - latency))
            for tol in pm.TAUS:
                with self.subTest(latency=latency, tol=tol):
                    row = result["halves"][0]["v3"]["pairs"][tol][(pm.H, pm.A)]
                    matched = int(abs(latency) <= tol)
                    self.assertEqual((row["matched"], row["missed"], row["invented"]), (matched, 1 - matched, 1 - matched))
                    self.assertEqual(row["latencies"], [latency] if matched else [])
                    stats = errors._latency_stats(row["latencies"])
                    self.assertEqual((stats["early"], stats["exact"], stats["late"]),
                                     (int(matched and latency < 0), 0, int(matched and latency > 0)))
                    self.assertEqual(stats["mean"], latency if matched else None)
            output = self.transition_output([("S", result)])
            self.assertIn(f"{latency:.3f} / {latency / 25:.3f}", output)
            if latency == 20:
                self.assertIn("| 12 | H↔A | H→A | 0 | 0 | 0 | n/a | n/a | n/a | n/a |", output)

    def test_v3_type_confusion_and_official_reference(self):
        result = self.analyse("D" * 80 + "H" * 120, "A" * 80 + "H" * 120)
        for tol in pm.TAUS:
            data = result["halves"][0]["v3"]
            restart = data["pairs"][tol][(pm.D, pm.H)]
            handover = data["pairs"][tol][(pm.A, pm.H)]
            self.assertEqual((restart["matched"], restart["missed"]), (0, 1))
            self.assertEqual((handover["matched"], handover["invented"]), (0, 1))
            self.assertEqual(data["official"][tol], {"n_true": 1, "n_pred": 1, "matched": 1, "missed": 0, "invented": 0})
        output = self.transition_output([("S", result)])
        self.assertIn('official default reference (typed="to", matched per half, pooled by count)', output)
        self.assertIn("diagnostic, not the official default metric", output)

    def test_v3_no_cross_half_matching(self):
        result = self.analyse("H" * 79 + "A" * 81, "H" * 81 + "A" * 79, [1] * 80 + [2] * 80)
        for tol in pm.TAUS:
            rows = [h["v3"]["pairs"][tol][(pm.H, pm.A)] for h in result["halves"]]
            pooled = errors._pool_transition_rows(rows)
            self.assertEqual((pooled["matched"], pooled["missed"], pooled["invented"]), (0, 1, 1))
            self.assertEqual(sum(h["v3"]["official"][tol]["matched"] for h in result["halves"]), 0)
        first, second = result["halves"]
        self.assertEqual(first["v3"]["windows"]["H↔A only"], {"frames": 26, "wrong": 1})
        self.assertEqual(second["v3"]["windows"]["H↔A only"], {"frames": 0, "wrong": 0})
        self.assertEqual(second["v3"]["windows"]["outside all windows"], {"frames": 80, "wrong": 1})

    def test_v3_ambiguity_both_directions(self):
        single = "H" * 50 + "A" * 50
        double = "H" * 45 + "A" * 10 + "H" * 5 + "A" * 40
        for truth, pred, expected in ((single, double, (1, 0)), (double, single, (0, 1))):
            result = self.analyse(truth, pred)
            for tol in pm.TAUS:
                row = result["halves"][0]["v3"]["pairs"][tol][(pm.H, pm.A)]
                self.assertEqual((row["ambiguous_true"], row["ambiguous_pred"]), expected)
            output = self.transition_output([("S", result)])
            self.assertIn("matching was a choice", output)
            self.assertIn("Latency figures are indicative when either count is nonzero", output)

    def test_v3_window_partition_and_raw_x(self):
        truth = "H" * 40 + "A" * 10 + "D" * 70
        pred = list(truth)
        for frame in (0, 14, 15, 24, 25, 40, 64, 65, 74, 75, 119):
            pred[frame] = "X"
        result = self.analyse(truth, "".join(pred))
        self.assertEqual(result["halves"][0]["v3"]["windows"], {
            "H↔A only": {"frames": 10, "wrong": 2}, "H/A→D only": {"frames": 10, "wrong": 2},
            "D→H/A only": {"frames": 0, "wrong": 0}, "overlapping": {"frames": 40, "wrong": 3},
            "outside all windows": {"frames": 60, "wrong": 4},
            "H↔A (inclusive)": {"frames": 50, "wrong": 5},
            "H/A→D (inclusive)": {"frames": 50, "wrong": 5},
            "D→H/A (inclusive)": {"frames": 0, "wrong": 0},
            "any window (union)": {"frames": 60, "wrong": 7}})
        output = self.transition_output([("S", result)])
        self.assertIn("| overlapping | 40 | 3 | 0.075 | 0.273 |", output)
        self.assertIn("| any window (union) — subtotal | 60 | 7 | 0.117 | 0.636 |", output)
        self.assertIn("---\n\nWindow subtotal", output)

    def test_v3_inclusive_window_repro(self):
        truth = "H" * 40 + "A" * 10 + "D" * 10 + "H" * 60
        pred = list(truth)
        pred[30] = pred[80] = "D"
        pred = "".join(pred)
        result = self.analyse(truth, pred)
        self.assertEqual(result["halves"][0]["v3"]["windows"], {
            "H↔A only": {"frames": 10, "wrong": 0},
            "H/A→D only": {"frames": 0, "wrong": 0},
            "D→H/A only": {"frames": 10, "wrong": 1},
            "overlapping": {"frames": 50, "wrong": 1},
            "outside all windows": {"frames": 50, "wrong": 0},
            "H↔A (inclusive)": {"frames": 50, "wrong": 1},
            "H/A→D (inclusive)": {"frames": 50, "wrong": 1},
            "D→H/A (inclusive)": {"frames": 50, "wrong": 1},
            "any window (union)": {"frames": 70, "wrong": 2}})
        output = self.transition_output([("S", result)])
        for name in ("H↔A (inclusive)", "H/A→D (inclusive)", "D→H/A (inclusive)"):
            self.assertIn(f"| {name} | 50 | 1 | 0.020 | 0.500 |", output)
        self.assertIn("Inclusive rows overlap, so their shares can sum above 1", output)
        self.assertIn("the exclusive rows partition all frames and wrong frames", output)
        self.assertIn("| any window (union) — subtotal | 70 | 2 | 0.029 | 1.000 |", output)
        # Unequal window exposures across halves must be pooled before division.
        short = "H" * 5 + "A" * 5 + "D" * 5 + "H" * 5
        pooled = self.analyse(truth + short, pred + short, [1] * 120 + [2] * 20)
        pooled_output = self.transition_output([("S", pooled)])
        for name in ("H↔A (inclusive)", "H/A→D (inclusive)", "D→H/A (inclusive)"):
            self.assertIn(f"| {name} | 70 | 1 | 0.014 | 0.500 |", pooled_output)

    def test_v3_same_category_window_union(self):
        truth = "H" * 40 + "A" * 10 + "H" * 70
        windows = self.analyse(truth, truth)["halves"][0]["v3"]["windows"]
        self.assertEqual(windows["H↔A only"], {"frames": 60, "wrong": 0})
        self.assertEqual(windows["overlapping"], {"frames": 0, "wrong": 0})
        self.assertEqual(windows["outside all windows"], {"frames": 60, "wrong": 0})

    def test_v3_windows_clipped_at_both_half_edges(self):
        truth = "H" + "A" * 9 + "H" * 9 + "A"
        result = self.analyse(truth, truth, [1] * 10 + [2] * 10)
        for h in result["halves"]:
            self.assertEqual(h["v3"]["windows"]["H↔A only"], {"frames": 10, "wrong": 0})
            self.assertEqual(h["v3"]["windows"]["outside all windows"], {"frames": 0, "wrong": 0})

    def test_v3_pooled_p90(self):
        truth = ("H" * 80 + "A" * 120) * 2
        pred = "H" * 80 + "A" * 120 + "H" * 100 + "A" * 100
        result = self.analyse(truth, pred, [1] * 200 + [2] * 200)
        rows = [h["v3"]["pairs"][25][(pm.H, pm.A)] for h in result["halves"]]
        pooled = errors._pool_transition_rows(rows)
        self.assertEqual(pooled["latencies"], [0, 20])
        self.assertEqual(errors._latency_stats(pooled["latencies"])["p90_abs"], np.percentile([0, 20], 90))
        self.assertEqual(errors._latency_stats(pooled["latencies"])["p90_abs"], 18)
        self.assertIn("| 25 | H↔A | H→A | 0 | 1 | 1 | 10.000 / 0.400 | 10.000 / 0.400 | 10.000 / 0.400 | 18.000 / 0.720 |",
                      self.transition_output([("S", result)]))

    def test_v3_pools_counts_not_ratios(self):
        first = self.analyse("H" * 40 + "A" * 120, "H" * 40 + "A" * 120)
        second = self.analyse("H" * 40 + "A" * 40 + "H" * 40 + "A" * 40, "H" * 160)
        output = self.transition_output([("S", first), ("T", second)])
        for tol in pm.TAUS:
            self.assertIn(f"| {tol} | H↔A | category subtotal | 4 | 1 | 1 | 3 | 0 | 1.000 | 0.250 |", output)

    def test_v3_reporting_needs_only_cached_data(self):
        result = self.analyse("H" * 60, "H" * 60)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            errors._print_transition_report([{"v3": h["v3"]} for h in result["halves"]])
        text = output.getvalue()
        self.assertIn("| 12 | H↔A | category subtotal | 0 | 0 | 0 | 0 | 0 | n/a | n/a |", text)
        self.assertIn("| outside all windows | 60 | 0 | 0.000 | n/a |", text)

    def test_validation(self):
        good = [codes("HHH"), codes("HHH"), np.ones(3, dtype=int), np.ones(3)]
        cases = [(2, np.array([1, 2, 1]), "contiguous"),
                 (0, codes("HXH"), "y_true"),
                 (1, np.array([0, 4, 0]), "y_pred"),
                 (1, np.array([-1, 0, 0]), "y_pred"),
                 (1, np.zeros(3, dtype=float), "integer"),
                 (0, np.zeros((3, 1), dtype=int), "1-D"),
                 (1, codes("HH"), "equal length"),
                 (3, np.array([0, 2, 1]), "binary"),
                 (3, np.array([0, np.nan, 1]), "binary")]
        for index, replacement, message in cases:
            with self.subTest(message=message):
                args = good.copy()
                args[index] = replacement
                with self.assertRaisesRegex(ValueError, message):
                    errors.analyse(*args)
        for fps in (24, 50, "25", True, float("nan")):
            with self.subTest(fps=fps), self.assertRaisesRegex(ValueError, "fps must equal 25"):
                errors.analyse(*good, fps=fps)

    def test_missing_artifacts_check_only(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(SystemExit) as caught:
                errors.check_legacy_artifacts(directory)
            self.assertNotEqual(caught.exception.code, 0)
            self.assertIn("oof.npz", str(caught.exception))
            self.assertIn("--predictions", str(caught.exception))
            Path(directory, "decoder.json").write_text(json.dumps({"hsmm": {"enabled": True}}))
            with self.assertRaises(SystemExit) as caught:
                errors.check_legacy_artifacts(directory)
            self.assertIn("change_oof.npz", str(caught.exception))
        self.test_import_isolation()

    def bundle_arrays(self):
        meta = {"schema": 1, "fps": 25, "matches": ["S", "T"], "provenance": "synthetic only"}
        arrays = {"meta": np.array(json.dumps(meta))}
        for m, text in (("S", "HHDDHH"), ("T", "AAADDD")):
            arrays.update({f"{m}__y_true": codes(text), f"{m}__y_pred": codes(text),
                           f"{m}__half": np.array([1, 1, 1, 2, 2, 2]),
                           f"{m}__ball_seen": np.array([1, 0, 1, 1, 0, 1]),
                           f"{m}__frame": np.arange(918273, 918279)})
        return arrays

    def run_cli(self, path):
        return subprocess.run([sys.executable, "-B", "-m", "kora.errors", "--predictions", str(path)],
                              cwd=ROOT, capture_output=True, text=True, timeout=30)

    def test_bundle_cli_end_to_end(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "synthetic.npz")
            np.savez(path, **self.bundle_arrays())
            meta, data = errors.load_bundle(path)
            self.assertEqual(meta["matches"], ["S", "T"])
            np.testing.assert_array_equal(data["S"][3], [1, 0, 1, 1, 0, 1])
            completed = self.run_cli(path)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("Provenance: synthetic only", completed.stdout)
            self.assertIn("| S | 1 |", completed.stdout)
            self.assertIn("| T | 2 |", completed.stdout)
            self.assertIn("| IoU | Duration s | True support | Matched true | Missed true | Recall |", completed.stdout)
            self.assertIn("| True duration s | True support | IoU 0.3 | IoU 0.4 | IoU 0.5 | IoU 0.6 | IoU 0.7 |", completed.stdout)
            self.assertIn("| Pred duration s | Predicted spells | IoU 0.3 | IoU 0.4 | IoU 0.5 | IoU 0.6 | IoU 0.7 |", completed.stdout)
            self.assertIn("| True label | Ball visibility (legacy: simulated blackout) | False D frames | Live frames |", completed.stdout)
            self.assertIn("| Ball visibility (legacy: simulated blackout) | Pred H/A frames (subtotal) | True D frames |", completed.stdout)
            self.assertIn("| Tol frames | Category | Pair | n_true | n_pred | Matched | Missed | Invented |", completed.stdout)
            self.assertIn("| Tol frames | Category | Pair | Early | Exact | Late | Mean signed frames / s |", completed.stdout)
            self.assertIn("| Window region | Frames | Wrong frames | Wrong / frames | Wrong / all wrong frames |", completed.stdout)
            self.assertNotIn("918273", completed.stdout)
            self.assertNotIn("[(", completed.stdout)
            self.assertEqual(completed.stderr, "")

    def test_malformed_bundle_cli(self):
        arrays = self.bundle_arrays()
        arrays["T__frame"] = np.array([0, 1, 2, 4, 5, 6])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "malformed.npz")
            np.savez(path, **arrays)
            completed = self.run_cli(path)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("match T: frame must be strictly increasing by 1", completed.stderr)
            self.assertNotIn("Traceback", completed.stderr)
            self.assertEqual(completed.stdout, "")

    def test_bundle_schema_validation(self):
        cases = [({}, "missing meta"),
                 ({"meta": np.array("not JSON")}, "not valid JSON"),
                 ({"meta": np.array(json.dumps({"schema": 2}))}, "schema"),
                 ({"S__frame": np.array([0, 1, 2, 3, 4, 4])}, "increasing by 1"),
                 ({"S__frame": np.array([0, 1])}, "aligned"),
                 ({"S__ball_seen": np.array([0, 0, 0, 0, 0, 2])}, "binary")]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "synthetic.npz")
            for overrides, message in cases:
                with self.subTest(message=message):
                    arrays = self.bundle_arrays()
                    if overrides:
                        arrays.update(overrides)
                    else:
                        arrays.pop("meta")
                    np.savez(path, **arrays)
                    with self.assertRaisesRegex(ValueError, message):
                        errors.load_bundle(path)
            arrays = self.bundle_arrays()
            arrays.pop("S__y_pred")
            np.savez(path, **arrays)
            with self.assertRaisesRegex(ValueError, "missing: S__y_pred"):
                errors.load_bundle(path)

    def test_pooled_report_uses_counts_and_durations(self):
        results = [("S", self.analyse("H" * 25, "H" * 25)),
                   ("T", self.analyse("H" * 75 + "D" * 25, "H" * 75 + "D" * 25))]
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            errors.print_report({"fps": 25, "provenance": "synthetic only"}, results)
        text = output.getvalue()
        self.assertIn("| ALL | all | 125 | 2 | 2 | 0 | 1080.000 | 1080.000 | 0.000 |", text)
        self.assertIn("| ALL | all | 2.000 | 2.000 | 2.000 | 2.000 | 25 / 0.200", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
