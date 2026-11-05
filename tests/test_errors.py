"""Synthetic metric and bundle checks; no match data or fitted models are used."""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

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

    def report(self, results, printer=None):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            if printer:
                key = {errors._print_duration_report: "v2", errors._print_transition_report: "v3",
                       errors._print_frame_report: "v4", errors._print_spell_error_report: "spell_errors"}[printer]
                printer([{key: h[key]} for _, result in results for h in result["halves"]])
            else:
                errors.print_report({"fps": 25, "provenance": "synthetic only"}, results)
        return output.getvalue()

    def test_association_components(self):
        # Association graphs distinguish erased teams, splits, merges and weak edges.
        cases = [
            ("perfect", "HHHDDDAAAHH", "HHHDDDAAAHH", {"association": (3, 3, 3)}, (3, 3, 0)),
            ("erased opponent", "H" * 10 + "A" * 10, "H" * 20,
             {"association": (1, 1, 1), "isolated true": (1, 1, 0)}, (1, 1, 0)),
            ("merge around opponent", "H" * 10 + "A" * 10 + "H" * 10, "H" * 30,
             {"merge": (1, 2, 1), "isolated true": (1, 1, 0)}, (2, 2, 0)),
            ("many to many", "H" * 10 + "DD" + "H" * 10, "HHHHD" + "H" * 17,
             {"complex": (1, 2, 2)}, (3, 3, 0)),
            ("weak overlap", "H" * 10 + "D" * 8, "D" * 8 + "H" * 10,
             {"isolated true": (1, 1, 0), "isolated predicted": (1, 0, 1)}, (1, 0, 1)),
            ("inclusive edge", "HHHHDD", "DDHHHH", {"association": (1, 1, 1)}, (1, 1, 0)),
            ("association without match", "H" * 100, "D" * 49 + "H" + "D" * 50,
             {"association": (1, 1, 1)}, (1, 1, 0)),
        ]
        for name, truth, pred, counts, overlaps in cases:
            with self.subTest(case=name):
                h = self.analyse(truth, pred)["halves"][0]
                a = h["associations"]
                expected = {kind: dict(zip(("components", "true_spells", "pred_spells"),
                                           counts.get(kind, (0, 0, 0)))) for kind in a["counts"]}
                self.assertEqual(a["counts"], expected)
                self.assertEqual(tuple(a[k] for k in ("positive_overlaps", "retained_overlaps", "discarded_overlaps")), overlaps)
                if name == "association without match":
                    self.assertTrue(all(not pairs for pairs in h["matches"].values()))
        # Gap labels, rather than the count graph alone, determine each subtype.
        for gap, split, merge in (("DD", "false D interruption", "missed stoppage"),
                                  ("AA", "wrong-team fragment", "missed opposite possession"),
                                  ("DA", "mixed", "mixed")):
            text = "HHHH" + gap + "HHHH"
            for truth, pred, kind, subtype in (("H" * 10, text, "split", split),
                                               (text, "H" * 10, "merge", merge)):
                with self.subTest(gap=gap, kind=kind):
                    a = self.analyse(truth, pred)["halves"][0]["associations"]
                    self.assertEqual(a["subtypes"][kind][subtype],
                                     dict(zip(("components", "true_spells", "pred_spells"),
                                              (1, 1, 2) if kind == "split" else (1, 2, 1))))

    def test_exact_spell_pairs(self):
        # Equal-IoU ties and competing truths retain the official greedy partner.
        for truth, pred in (("H" * 10, "HHHHDDHHHH"), ("HHHHDDHHHH", "H" * 10)):
            with self.subTest(truth=truth):
                h = self.analyse(truth, pred)["halves"][0]
                for iou in pm.SPELL_IOUS:
                    self.assertEqual(h["matches"][iou], [(0, 0, 0.4)] if iou <= 0.4 else [])

    def test_halves_x_and_time_denominators(self):
        # Temporal X fills only within halves; frame errors/shares retain raw X.
        cases = [("H" * 10, "H" * 10, [1] * 5 + [2] * 5,
                  [(1, 1, 0.2, 0.2, 0, 0, 0), (1, 1, 0.2, 0.2, 0, 0, 0)]),
                 ("HHHHHDDDDDHHHH", "XXHHHDDXXXXXXX", [1] * 10 + [2] * 4,
                  [(1, 1, 0.2, 0.2, 0.5, 0.2, 5), (1, 0, None, None, 1, 0, 4)]),
                 ("HHHH", "XXHH", [1, 1, 2, 2], [(1, 0, None, None, 1, 0, 2), (1, 1, 0.08, 0.08, 0, 0, 0)])]
        fields = ("n_true", "n_pred", "pred_mean_s", "pred_median_s", "pred_x_share", "pred_d_share", "wrong_frames")
        for truth, pred, half, expected in cases:
            with self.subTest(truth=truth, prediction=pred):
                result = self.analyse(truth, pred, half)
                self.assertEqual([tuple(h[k] for k in fields) for h in result["halves"]], expected)
        text = "H" * 225 + "D" * 225 + "A" * 225
        h = self.analyse(text, text)["halves"][0]
        self.assertEqual(tuple(h[k] for k in ("n_frames", "true_per_45", "pred_per_45", "true_mean_s", "true_median_s", "true_d_share")),
                         (675, 200, 200, 9, 9, 1 / 3))

    def test_duration_group_edges(self):
        text = "D".join("H" * n for n in (24, 25, 50, 100, 200, 375))
        h = self.analyse(text, text)["halves"][0]
        for iou in pm.SPELL_IOUS:
            for group in errors.DURATION_GROUPS:
                with self.subTest(iou=iou, group=group):
                    self.assertEqual(h["v2"][iou][group], dict(support=1, matched=1, missed=0,
                                     predicted=1, matched_predicted=1, false_predicted=0))

    def test_duration_reporting(self):
        # Separate true/predicted denominators, zero support, fragmentation and pooled counts.
        cases = [
            ([("H" * 24 + "D" * 6, "H" * 30)],
             ["15+ | 0 | 0 | 0 | n/a | 0 | 0 | 0 | n/a |",
              "<1 | 1 | 1 | 0 | 1.000 | 0 | 0 | 0 | n/a |",
              "1–2 | 0 | 0 | 0 | n/a | 1 | 1 | 0 | 1.000 |"]),
            ([("H" * 30, "H" * 24 + "D" * 6)],
             ["<1 | 0 | 0 | 0 | n/a | 1 | 1 | 0 | 1.000 |",
              "1–2 | 1 | 1 | 0 | 1.000 | 0 | 0 | 0 | n/a |"]),
            ([("H" * 10, "A" * 10)], ["<1 | 1 | 0 | 1 | 0.000 | 1 | 0 | 1 | 0.000 |"]),
            ([("H" * 10 + "D" * 20, "H" * 10 + "D" * 5 + "H" * 5 + "D" * 10),
              ("H" * 10 + "D" * 5 + "H" * 10 + "D" * 5 + "H" * 10, "H" * 10 + "D" * 30)],
             ["<1 | 4 | 2 | 2 | 0.500 | 3 | 2 | 1 | 0.667 |"]),
        ]
        for pairs, rows in cases:
            with self.subTest(pairs=pairs):
                results = [(str(i), self.analyse(y, p)) for i, (y, p) in enumerate(pairs)]
                text = self.report(results, errors._print_duration_report)
                for row in rows:
                    for iou in pm.SPELL_IOUS:
                        self.assertIn(f"| {iou} | " + row, text)
        results = [("S", self.analyse("H" * 25, "H" * 25)),
                   ("T", self.analyse("H" * 75 + "D" * 25, "H" * 75 + "D" * 25))]
        self.assertIn("| ALL | all | 125 | 2 | 2 | 0 | 1080.000 | 1080.000 | 0.000 |", self.report(results))
        self.assertIn("| ALL | all | 2.000 | 2.000 | 2.000 | 2.000 | 25 / 0.200", self.report(results))

    def test_spell_causes(self):
        # Ordered 50% rules, duration edges, half-local X and symmetric causes.
        # Rows alternate missed/invented per half; None skips the unasserted side.
        lengths = (24, 25, 49, 50, 99, 100, 101)
        cases = [("H" * 10, "A" * 10, None, [[(0, "<1", 1)], None]),
                 ("D" * 10 + "H" * 10 + "D" * 10, "A" * 30, None, [[(1, "<1", 1)], None]),
                 ("H" * 10, "D" * 10, None, [[(2, "<1", 1)], None]),
                 ("H" * 10, "HHDDHHDDHH", None, [[(3, "<1", 1)], None]),
                 ("H" * 9, "HHHAAADDD", None, [[(4, "<1", 1)], None]),
                 ("H" * 10, "A" * 5 + "D" * 5, None, [[(0, "<1", 1)], None]),
                 ("D" * 10 + "H" * 10 + "D" * 10, "A" * 15 + "D" * 15, None, [[(1, "<1", 1)], None]),
                 ("H" * 10, "HHDDDDDHHH", None, [[(2, "<1", 1)], None]),
                 ("D".join("H" * n for n in lengths), "D".join("A" * n for n in lengths), None,
                  [[(0, g, n) for g, n in zip(errors.SPELL_ERROR_GROUPS, (1, 2, 2, 2))]] * 2),
                 ("H" * 24 + "D" * 6, "A" * 30, None, [[(0, "<1", 1)], [(0, "1–2", 1)]]),
                 ("H" * 10, "XXXXXAXXXX", None, [[(0, "<1", 1)]] * 2),
                 ("H" * 10, "HHXXDDHXXX", None, [[(3, "<1", 1)], [(3, "<1", 2)]]),
                 ("H" * 10 + "D" * 20, "A" * 30, [1] * 10 + [2] * 20, [[(0, "<1", 1)], [(0, "<1", 1)], [], [(2, "<1", 1)]]),
                 ("H" * 10 + "D" * 10, "X" * 10 + "A" * 10, [1] * 10 + [2] * 10, [[(4, "<1", 1)], [], [], [(2, "<1", 1)]]),
                 ("H" * 10 + "A" * 10, "H" * 20, None, [[(0, "<1", 1)], []])]
        for truth, pred, half, expected in cases:
            for flip in (False, True):
                y, p = (s.translate(str.maketrans("HA", "AH")) if flip else s for s in (truth, pred))
                for reverse in (False, True) if "X" not in pred else (False,):
                    with self.subTest(truth=truth, prediction=pred, flip=flip, reverse=reverse):
                        halves = self.analyse(p, y, half)["halves"] if reverse else self.analyse(y, p, half)["halves"]
                        for i, h in enumerate(halves):
                            for j, side in enumerate(("missed", "invented")):
                                rows = expected[2 * i + (1 - j if reverse else j)]
                                if rows is None:
                                    continue
                                counts = {c: dict.fromkeys(errors.SPELL_ERROR_GROUPS, 0) for c in errors.SPELL_ERROR_CAUSES[side]}
                                for cause, group, n in rows:
                                    counts[errors.SPELL_ERROR_CAUSES[side][cause]][group] = n
                                self.assertEqual(h["spell_errors"][side], counts)

    def test_spell_error_totals_equal_official(self):
        cases = (("HHHHDDHHHH", "H" * 10), ("H" * 10, "HHHHDDHHHH"),
                 ("H" * 10 + "A" * 10, "H" * 20), ("H" * 9, "HHHAAADDD"),
                 ("H" * 10 + "D" * 10, "X" * 10 + "A" * 10), ("D" * 20, "D" * 20))
        for truth, pred in cases:
            halves = [1] * 10 + [2] * 10 if len(truth) == 20 else [1] * len(truth)
            result = self.analyse(truth, pred, halves)
            filled = pm.fill_abstentions(codes(pred), result["runs"])
            for h, run in zip(result["halves"], result["runs"]):
                official = pm.spell_scores(codes(truth), filled, [run])["per_iou"][0.5]
                for side, key in (("missed", "n_missed"), ("invented", "n_spurious")):
                    self.assertEqual(sum(sum(row.values()) for row in h["spell_errors"][side].values()),
                                     official[key])

    def test_cause_reporting(self):
        results = [("S", self.analyse("H" * 10, "A" * 10)),
                   ("T", self.analyse("H" * 10 + "D" * 5 + "H" * 10 + "D" * 5 + "H" * 10, "D" * 40)),
                   ("U", self.analyse("D" * 30, "H" * 30))]
        text = self.report(results, errors._print_spell_error_report)
        for row in ("team swap | 1 | 0 | 0 | 0 | 1 | 0.250", "dead ball instead | 3 | 0 | 0 | 0 | 3 | 0.750",
                    "Total | 4 | 0 | 0 | 0 | 4 | 1.000", "team swap | 1 | 0 | 0 | 0 | 1 | 0.500",
                    "on dead ball | 0 | 1 | 0 | 0 | 1 | 0.500", "Total | 1 | 1 | 0 | 0 | 2 | 1.000"):
            self.assertIn("| " + row + " |", text)
        self.assertEqual(self.report([("S", self.analyse("DD", "DD"))],
                                    errors._print_spell_error_report).count("| Total | 0 | 0 | 0 | 0 | 0 | n/a |"), 2)

    def test_frame_counts_and_denominators(self):
        # All label/visibility strata include X; pooled counts retain raw denominators.
        strata = (("H", 1, "HHADDX"), ("H", 0, "HADDXXX"), ("A", 1, "HAAADX"),
                  ("A", 0, "HHADX"), ("D", 1, "HHADDXX"), ("D", 0, "HAAADXX"))
        truth, pred = "".join(y * len(p) for y, _, p in strata), "".join(p for _, _, p in strata)
        seen = [v for _, v, p in strata for _ in p]
        whole = self.analyse(truth, pred, ball_seen=seen)["halves"][0]["v4"]
        split = self.analyse(truth, pred, [1] * 19 + [2] * 19, seen)
        np.testing.assert_array_equal(sum(h["v4"] for h in split["halves"]), whole)
        np.testing.assert_array_equal(np.vstack((whole.sum(axis=1), np.zeros((1, 4), int))),
                                      pm.confusion(codes(truth), codes(pred)))
        np.testing.assert_array_equal(whole.sum(axis=(0, 2)), [19, 19])
        rows = errors._frame_error_rows(whole)
        expected = [(2, 6, 1, 1, 2), (2, 7, 3, 1, 1), (1, 6, 1, 1, 3), (1, 5, 1, 2, 1),
                    (4, 13, 4, 2, 3), (2, 11, 2, 3, 4), (3, 12, 2, 2, 5), (3, 12, 4, 3, 2), (6, 24, 6, 5, 7)]
        self.assertEqual([tuple(r[k] for k in ("false_d", "denominator", "pred_x", "wrong_team", "correct_team"))
                          for r in rows["live"]], expected)
        self.assertEqual([tuple(r[k] for k in ("pred_live", "denominator", "pred_h", "pred_a", "pred_x", "correct_d"))
                          for r in rows["dead"]], [(3, 7, 2, 1, 2, 2), (4, 7, 1, 3, 2, 1), (7, 14, 3, 4, 4, 3)])
        for pairs, row in [([("HH", "DD"), ("HHHHHH", "HHHHHH")], "| live | all | 2 | 8 | 0.250 | 0 | 0 | 6 |"),
                           ([("HHDD", "XDXH")], "| live | all | 1 | 2 | 0.500 | 1 | 0 | 0 |"),
                           ([("HHDD", "HDHA")], "| missing | 0 | 0 | n/a | 0 | 0 | 0 | 0 |")]:
            with self.subTest(row=row):
                self.assertIn(row, self.report([(str(i), self.analyse(y, p)) for i, (y, p) in enumerate(pairs)],
                                              errors._print_frame_report))

    def test_transition_pairing(self):
        # The shared scorer checks exact counts/pairs; these cover signed stats and ambiguity.
        for latency in (10, 20, -5):
            h = self.analyse("H" * 80 + "A" * 120, "H" * (80 + latency) + "A" * (120 - latency))["halves"][0]
            for tol in pm.TAUS:
                with self.subTest(latency=latency, tolerance=tol):
                    stats = errors._latency_stats(h["v3"]["pairs"][tol][(pm.H, pm.A)]["latencies"])
                    matched = abs(latency) <= tol
                    self.assertEqual((stats["early"], stats["exact"], stats["late"], stats["mean"]),
                                     (int(matched and latency < 0), 0, int(matched and latency > 0), latency if matched else None))
        self.analyse("D" * 80 + "H" * 120, "A" * 80 + "H" * 120)  # Same destination, different origin.
        single, double = "H" * 50 + "A" * 50, "H" * 45 + "A" * 10 + "H" * 5 + "A" * 40
        for y, p in ((single, double), (double, single)):
            self.analyse(y, p)
        result = self.analyse("H" * 79 + "A" * 81, "H" * 81 + "A" * 79, [1] * 80 + [2] * 80)
        self.assertEqual([h["v3"]["windows"]["H↔A only"] for h in result["halves"]],
                         [dict(frames=26, wrong=1), dict(frames=0, wrong=0)])

    def test_transition_windows(self):
        # Half-open radius, category union/overlap, raw X, clipping and pooled exposures.
        cases = [("H" * 40 + "A" * 10 + "D" * 70, [0, 14, 15, 24, 25, 40, 64, 65, 74, 75, 119],
                  [10, 10, 0, 40, 60, 50, 50, 0, 60], [2, 2, 0, 3, 4, 5, 5, 0, 7]),
                 ("H" * 40 + "A" * 10 + "D" * 10 + "H" * 60, [30, 80],
                  [10, 0, 10, 50, 50, 50, 50, 50, 70], [0, 0, 1, 1, 0, 1, 1, 1, 2])]
        for truth, wrong, frames, counts in cases:
            with self.subTest(truth=truth):
                pred = list(truth)
                for i in wrong:
                    pred[i] = "X"
                result = self.analyse(truth, "".join(pred))
                windows = result["halves"][0]["v3"]["windows"]
                self.assertEqual(list(windows.values()), [dict(frames=n, wrong=w) for n, w in zip(frames, counts)])
                if len(wrong) == 2:
                    short = "H" * 5 + "A" * 5 + "D" * 5 + "H" * 5
                    pooled = self.analyse(truth + short, "".join(pred) + short, [1] * 120 + [2] * 20)
                    text = self.report([("S", pooled)], errors._print_transition_report)
                    for name in ("H↔A", "H/A→D", "D→H/A"):
                        self.assertIn(f"| {name} (inclusive) | 70 | 1 | 0.014 | 0.500 |", text)
        for truth, half, expected in (("H" * 40 + "A" * 10 + "H" * 70, None, (60, 60)),
                                      ("H" + "A" * 9 + "H" * 9 + "A", [1] * 10 + [2] * 10, (10, 0))):
            for h in self.analyse(truth, truth, half)["halves"]:
                self.assertEqual(tuple(h["v3"]["windows"][k]["frames"] for k in ("H↔A only", "outside all windows")), expected)

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
                      self.report([("S", result)]))

    def test_v3_pools_counts_not_ratios(self):
        first = self.analyse("H" * 40 + "A" * 120, "H" * 40 + "A" * 120)
        second = self.analyse("H" * 40 + "A" * 40 + "H" * 40 + "A" * 40, "H" * 160)
        output = self.report([("S", first), ("T", second)])
        for tol in pm.TAUS:
            self.assertIn(f"| {tol} | H↔A | category subtotal | 4 | 1 | 1 | 3 | 0 | 1.000 | 0.250 |", output)

    def test_missed_stoppage_causes(self):
        # Priority, candidate reuse, tolerance edges, X filling and half isolation.
        truth = "H" * 60 + "D" * 60
        cases = [(truth, "X" * 20 + "A" * 65 + "D" * 35, None, [(1, 0, 0)]),
                 (truth, "H" * 86 + "D" * 34, None, [(0, 1, 0)]),
                 (truth, truth, None, [(0, 0, 0)]), (truth, "X" * 120, None, [(0, 1, 0)]),
                 (truth, "H" * 30 + "X" * 30 + "D" * 60, None, [(0, 0, 0)]),
                 (truth, "A" * 30 + "X" * 30 + "D" * 60, None, [(1, 0, 0)]),
                 ("H" * 79 + "D" * 81, "A" * 81 + "D" * 79, [1] * 80 + [2] * 80, [(0, 1, 0), (0, 0, 0)])]
        double = "H" * 40 + "D" * 5 + "H" * 5 + "D" * 50
        cases += [(double, p, None, [counts]) for p, counts in
                  (("H" * 45 + "D" * 55, (0, 0, 1)), ("H" * 40 + "D" * 5 + "A" * 5 + "D" * 50, (1, 0, 0)),
                   ("A" * 45 + "D" * 55, (2, 0, 0)))]
        cases += [(truth, "A" * (60 + d) + "D" * (60 - d), None,
                   [(1, 0, 0) if abs(d) <= 25 else (0, 1, 0)]) for d in (-26, -25, 0, 25, 26)]
        for y, p, half, expected in cases:
            with self.subTest(prediction=p, half=half):
                for h, counts in zip(self.analyse(y, p, half)["halves"], expected):
                    self.assertEqual(h["v3"]["missed_stoppages"], dict(zip(errors.STOPPAGE_MISS_CAUSES, counts)))

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

    def bundle_arrays(self):
        meta, data = self.comparison_bundle(["HHDDHH", "AAADDD"], ["HHDDHH", "AAADDD"], [[1, 1, 1, 2, 2, 2]] * 2)
        for arrays in data.values():
            arrays[3][:] = [1, 0, 1, 1, 0, 1]
        return {"meta": json.dumps(meta), **{f"{m}__{f}": a for m, arrays in data.items()
                                           for f, a in zip(errors.BUNDLE_ARRAYS, arrays)}}

    def run_cli(self, path, *extra):
        return subprocess.run([sys.executable, "-B", "-m", "kora.errors", "--predictions", str(path), *map(str, extra)],
                              cwd=ROOT, capture_output=True, text=True, timeout=30)

    def comparison_bundle(self, predictions, truths=None, halves=None):
        truths = ["HHDDAA"] * len(predictions) if truths is None else truths
        meta = {"schema": 1, "fps": 25, "matches": [f"M{i}" for i in range(len(predictions))],
                "provenance": "synthetic only"}
        data = {}
        for i, (match, truth, pred) in enumerate(zip(meta["matches"], truths, predictions)):
            half = np.ones(len(truth), dtype=int) if halves is None else np.asarray(halves[i])
            data[match] = (codes(truth), codes(pred), half, np.ones(len(truth), dtype=bool),
                           np.arange(918273, 918273 + len(truth)))
        return meta, data

    def paired_example(self):
        base = self.comparison_bundle(["HHDDDD"] + ["HHDDAA"] * 6)
        candidate = self.comparison_bundle(["HHDDAA"] * 7)
        return base, candidate

    def save_comparison_bundle(self, path, bundle):
        meta, data = bundle
        arrays = {f"{match}__{field}": array for match, values in data.items()
                  for field, array in zip(errors.BUNDLE_ARRAYS, values)}
        np.savez(path, meta=json.dumps(meta), **arrays)

    def test_comparison_zero_differences(self):
        # Identical predictions and empty truth have degenerate uncertainty intervals.
        base, _ = self.paired_example()
        empty = self.comparison_bundle(["HHHH", ""], ["DDDD", ""])
        for pair in ((base, base), (empty, self.comparison_bundle(["XXXX", ""], ["DDDD", ""]))):
            with self.subTest(matches=pair[0][0]["matches"]):
                result = errors.compare_bundles(*pair, draws=7)
                self.assertTrue(all(r["difference"] == 0 for r in result["matches"]))
                self.assertEqual((result["mean_difference"], result["possession_interval"], result["match_interval"],
                                  result["positive_share"]), (0, (0, 0), (0, 0), 0))

    def test_comparison_matches_official_bootstrap(self):
        truth = "H" * 10 + "D" * 5 + "A" * 10 + "D" * 5 + "H" * 10
        prediction = "H" * 5 + "D" * 10 + "A" * 15 + "H" * 10
        for y, p in ((truth, prediction), ("HHDDAA", "HHDDDD")):
            with self.subTest(truth=y):
                base = self.comparison_bundle(["X" * len(y)], [y])
                candidate = self.comparison_bundle([p], [y])
                result = errors.compare_bundles(base, candidate, draws=317, seed=29)
                runs = [(0, len(y))]
                filled = pm.fill_abstentions(codes(p), runs)
                expected = pm.bootstrap_spell_f1(codes(y), filled, runs, 317, seed=29, percentiles=(5, 95))
                self.assertEqual(result["possession_interval"], expected)
                self.assertEqual(result["match_interval"], (result["mean_difference"],) * 2)

    def test_comparison_official_points_and_half_local_filling(self):
        truths = ["HHHHHHHH", "HHAADDAA", "DDDD", "HHAA"]
        halves = [[1] * 4 + [2] * 4, [1] * 4 + [2] * 4, [1, 1, 2, 2], [1] * 4]
        base = self.comparison_bundle(["HHXXXXXX", "HXXXDXAX", "HHHH", "HHHH"], truths, halves)
        candidate = self.comparison_bundle(["XXXXHHHH", "XXAADXXX", "XXXX", "AAAA"], truths, halves)
        result = errors.compare_bundles(base, candidate, draws=31, seed=8)
        for row in result["matches"]:
            for name, bundle in (("base", base), ("candidate", candidate)):
                y, p, half, seen, _ = bundle[1][row["match"]]
                runs = errors._validate_inputs(y, p, half, seen, 25)[-1]
                official = pm.spell_scores(y, pm.fill_abstentions(p, runs), runs)
                self.assertEqual(row[name], official["mean_f1"])
        # Each half is a distinct H spell; X must not fill across the half boundary.
        self.assertEqual(result["matches"][0]["base"], 2 / 3)
        self.assertEqual(result["matches"][0]["candidate"], 2 / 3)

    def test_comparison_pairing_failures_before_scoring(self):
        for field in ("match order", "match IDs", "y_true", "half", "frame", "ball_seen"):
            with self.subTest(field=field):
                base, candidate = self.paired_example()
                if field == "match order":
                    candidate[0]["matches"].reverse()
                elif field == "match IDs":
                    candidate[0]["matches"][0] = "OTHER"
                else:
                    index = errors.BUNDLE_ARRAYS.index(field)
                    candidate[1]["M0"][index][0] += 1 if field != "ball_seen" else -1
                message = "same order" if field.startswith("match") else f"M0: {field} arrays differ"
                with mock.patch.object(errors, "_comparison_spells", side_effect=AssertionError("scored before pairing")):
                    with self.assertRaisesRegex(ValueError, message):
                        errors.compare_bundles(base, candidate)

    def test_comparison_determinism_and_match_bootstrap(self):
        base, candidate = self.paired_example()
        first = errors.compare_bundles(base, candidate, draws=137, seed=42)
        second = errors.compare_bundles(base, candidate, draws=137, seed=42)
        self.assertEqual(first, second)
        self.assertAlmostEqual(first["mean_difference"], 1 / 21)
        self.assertEqual((first["wins"], first["ties"], first["losses"], first["possession_interval"]), (1, 6, 0, (0, 1 / 7)))
        reverse = errors.compare_bundles(candidate, base, draws=137, seed=42)
        self.assertEqual((reverse["wins"], reverse["ties"], reverse["losses"], reverse["possession_interval"]), (0, 6, 1, (-1 / 7, 0)))
        self.assertAlmostEqual(reverse["mean_difference"], -1 / 21)
        # Independent reproduction of paired draws, followed by resampling match differences.
        rng = np.random.RandomState(42)
        possession = np.zeros(137)
        for match in base[0]["matches"]:
            for draw in range(137):
                picks = rng.randint(0, 2, 2)
                if match == "M0":
                    tp = int(np.sum(picks == 0))
                    baseline = 2 * tp / (2 * tp + max(0, 1 - tp) + 2 - tp)
                    possession[draw] = (1 - baseline) / 7
        differences = [row["difference"] for row in first["matches"]]
        means = [float(np.mean([differences[i] for i in rng.randint(0, 7, 7)])) for _ in range(137)]
        self.assertEqual(first["match_interval"], tuple(np.percentile(means, [5, 95])))
        self.assertEqual(first["possession_interval"], tuple(np.percentile(possession, [5, 95])))
        self.assertEqual(first["positive_share"], float(np.mean(possession > 0)))

    def test_comparison_cli_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            base_path, path = Path(directory, "base.npz"), Path(directory, "candidate.npz")
            base, candidate = self.paired_example()
            self.save_comparison_bundle(base_path, base)
            candidate[1]["M0"][4][:] += 1
            self.save_comparison_bundle(path, candidate)
            completed = self.run_cli(path, "--compare", base_path)
            self.assertEqual(completed.returncode, 2)
            self.assertIn("pairing failed", completed.stderr)
            self.assertNotIn("Base spellF1", completed.stdout)
            self.save_comparison_bundle(path, base)
            for option, value in (("--draws", 0), ("--draws", -1), ("--seed", -1)):
                with self.subTest(option=option, value=value):
                    completed = self.run_cli(path, "--compare", base_path, option, value)
                    self.assertEqual(completed.returncode, 2)
                    self.assertNotIn("Traceback", completed.stderr)
            completed = subprocess.run([sys.executable, "-B", "-m", "kora.errors", "--compare", str(base_path)],
                                       cwd=ROOT, capture_output=True, text=True, timeout=30)
            self.assertEqual(completed.returncode, 2)
            self.assertIn("--compare requires --predictions", completed.stderr)

    def test_bundle_schema_validation(self):
        # Schema, alignment, monotone frames, binary visibility and missing arrays are refused.
        cases = [("meta", None, "missing meta"), ("meta", "not JSON", "not valid JSON"),
                 ("meta", json.dumps({"schema": 2}), "schema"), ("M0__y_pred", None, "missing: M0__y_pred"),
                 ("M0__frame", [0, 1, 2, 3, 4, 4], "increasing by 1"), ("M0__frame", [0, 1], "aligned"),
                 ("M1__frame", [0, 1, 2, 4, 5, 6], "increasing by 1"),
                 ("M0__ball_seen", [0, 0, 0, 0, 0, 2], "binary")]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "malformed.npz")
            for field, value, message in cases:
                with self.subTest(field=field, message=message):
                    arrays = self.bundle_arrays()
                    if value is None:
                        arrays.pop(field)
                    else:
                        arrays[field] = np.asarray(value)
                    np.savez(path, **arrays)
                    with self.assertRaisesRegex(ValueError, message):
                        errors.load_bundle(path)
                    completed = self.run_cli(path)
                    self.assertNotEqual(completed.returncode, 0)
                    self.assertIn(message, completed.stderr)
                    self.assertNotIn("Traceback", completed.stderr)
                    self.assertEqual(completed.stdout, "")

    def test_bundle_and_comparison_cli(self):
        with tempfile.TemporaryDirectory() as directory:
            path, base_path = Path(directory, "candidate.npz"), Path(directory, "base.npz")
            np.savez(path, **self.bundle_arrays())
            meta, data = errors.load_bundle(path)
            self.assertEqual(meta["matches"], ["M0", "M1"])
            np.testing.assert_array_equal(data["M0"][3], [1, 0, 1, 1, 0, 1])
            completed = self.run_cli(path)
            self.assertEqual((completed.returncode, completed.stderr), (0, ""))
            expected = io.StringIO()
            with contextlib.redirect_stdout(expected):
                errors.print_report(meta, [(m, errors.analyse(*arrays, fps=meta["fps"])) for m, arrays in data.items()])
            self.assertEqual(completed.stdout, expected.getvalue())
            base, candidate = self.paired_example()
            self.save_comparison_bundle(base_path, base)
            self.save_comparison_bundle(path, candidate)
            completed = self.run_cli(path, "--compare", base_path)
            self.assertEqual((completed.returncode, completed.stderr), (0, ""))
            for text in ("Pairing check: passed", "| M0 | 0.666667 | 1.000000 | +0.333333 |",
                         "| MEAN | 0.952381 | 1.000000 | +0.047619 |", "Wins / ties / losses: 1 / 6 / 0",
                         "not a significance test", "do not replace the registered keep rule", "Draws: 1000; seed: 0"):
                self.assertIn(text, completed.stdout)
            self.assertNotIn("918273", completed.stdout)
            self.assertNotIn("KEEP", completed.stdout)
            self.assertEqual(self.run_cli(path, "--compare", base_path, "--draws", 1000, "--seed", 0).stdout, completed.stdout)
