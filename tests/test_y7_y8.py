"""Synthetic checks for RESEARCH.md Y7a (stage-2 refiner) and Y8a (dwell features), and their cache keys."""
import unittest

import numpy as np
import pandas as pd

from kora import variant
from kora.dwell import add, since_change
from kora.features import carry
from kora.nested import fit_refiner


def scene():
    """Home carries (0-50), passes (ball far from everyone, 50-70), home receives (70-100), away
    intercepts (100-150). Nearest-player distances in metres; the far team stays 8 m away."""
    n = 150
    d1H = np.full(n, 8.0, np.float32)
    d1A = np.full(n, 8.0, np.float32)
    d1H[:50] = 0.3; d1H[50:70] = 6.0; d1A[50:70] = 9.0; d1H[70:100] = 0.3; d1A[100:] = 0.3
    return d1H, d1A


def frame(d1H, d1A):
    """The columns dwell.add reads, made the way features.build makes them."""
    d_diff = np.clip(d1H - d1A, -20, 20)
    close = np.isfinite(d_diff) & (np.fmin(d1H, d1A) < 1.5)
    last, _ = carry(close, np.sign(d_diff), "f")
    nxt, _ = carry(close, np.sign(d_diff), "b")
    return pd.DataFrame({"d1H": d1H, "d1A": d1A, "last_close": last, "next_close": nxt})


class DwellTests(unittest.TestCase):
    def test_pass_keeps_dwell_and_interception_resets_it(self):
        f = add(frame(*scene()))
        self.assertAlmostEqual(f["dwell_p"][99], 99 / 25, places=5)       # the pass is still home's spell
        self.assertAlmostEqual(f["dwell_p"][120], 20 / 25, places=5)      # reset by the interception
        self.assertAlmostEqual(f["dwell_f"][50], 49 / 25, places=5)       # home's last frame is 99
        self.assertEqual(f["flips_p25"][90], 0)
        self.assertEqual(f["flips_p25"][110], 1)
        self.assertEqual(f["flips_f50"][60], 1)
        self.assertEqual(f["press"][20], 8.0)                             # away is 8 m from home's ball
        self.assertEqual(f["press"][120], 8.0)

    def test_missing_ball_gives_no_dwell(self):
        d1H, d1A = scene()
        d1H[:], d1A[:] = np.nan, np.nan
        f = add(frame(d1H, d1A))
        for k in ("dwell_p", "dwell_f", "press"):
            self.assertTrue(np.isnan(f[k]).all(), k)
        self.assertEqual(np.nansum(f["flips_p25"]), 0)

    def test_since_change_counts_from_each_new_value(self):
        np.testing.assert_array_equal(since_change(np.array([np.nan, 1, 1, -1, -1, -1, 1.])),
                                      [np.nan, 0, 1, 0, 1, 2, 0])

    def test_original_columns_untouched(self):
        x = frame(*scene())
        f = add(x)
        pd.testing.assert_frame_equal(f[x.columns], x)


class RefinerTests(unittest.TestCase):
    def data(self, seed):
        rng = np.random.default_rng(seed)
        ctx, labels = {}, {}
        for j in "abcd":
            y = rng.integers(0, 3, 3000)
            ctx[j] = (np.eye(3)[y] + rng.normal(0, 0.8, (3000, 3))).astype(np.float32)
            labels[j] = y
        return ctx, labels

    def test_three_probabilities_per_test_frame(self):
        ctx, labels = self.data(0)
        p = fit_refiner(ctx, labels, ["a", "b", "c"], "d")
        self.assertEqual(p.shape, (3000, 3))
        np.testing.assert_allclose(p.sum(1), 1, rtol=1e-6)
        self.assertGreater((p.argmax(1) == labels["d"]).mean(), 0.5)    # it learns from the windows

    def test_test_labels_are_never_read(self):
        ctx, labels = self.data(1)
        p = fit_refiner(ctx, labels, ["a", "b", "c"], "d")
        labels["d"] = np.zeros_like(labels["d"])
        np.testing.assert_array_equal(p, fit_refiner(ctx, labels, ["a", "b", "c"], "d"))


class CacheKeyTests(unittest.TestCase):
    def test_y_variants_reuse_production_caches_where_they_can(self):
        v = variant.VARIANTS
        bal = v["touch_bal"]
        for name in ("y7a_refine", "y7c_wide"):                           # decoder only: stage 1 reused
            self.assertEqual(variant.stage1_version(v[name]), variant.stage1_version(bal))
        for name in ("y8a_dwell", "y8b_bal1"):                            # same features, new stage 1
            self.assertEqual(variant.feature_version(v[name]), variant.feature_version(bal))
            self.assertNotEqual(variant.stage1_version(v[name]), variant.stage1_version(bal))

    def test_seed_runs_get_their_own_caches_and_leave_others_alone(self):
        v = variant.VARIANTS
        self.assertNotEqual(variant.feature_version(v["mixed_s2028"]), variant.feature_version(v["mixed"]))
        self.assertEqual(variant.feature_version({**v["mixed_s2028"], "seed": None}),
                         variant.feature_version(v["mixed"]))

    def test_refine_is_off_unless_asked(self):
        self.assertFalse(variant.decoder(variant.VARIANTS["touch_bal"])["refine"])
        self.assertTrue(variant.decoder(variant.VARIANTS["y7a_refine"])["refine"])


if __name__ == "__main__":
    unittest.main()
