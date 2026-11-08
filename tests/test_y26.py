"""Checks for Y26 (decode-only candidates on y23_big2): the visibility-aware plain decoder, kora.nested's dispatch
to it, and that every Y26 variant reuses y23_big2's caches."""
import unittest

import numpy as np

from kora import decode, nested, variant, visdecode


class Y26Tests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.P = rng.dirichlet([1, 1, 1], size=400)
        self.prior = np.array([0.4, 0.35, 0.25])
        self.seg = [(0, 150), (150, 400)]

    def test_factor_one_is_decode(self):
        missing = np.random.default_rng(1).random(400) < 0.4
        for alpha, ha, d in ((0.0, 4, 8), (1.0, 8, 32)):
            np.testing.assert_array_equal(visdecode.decode(self.P, self.seg, self.prior, alpha, ha, d, 1, missing),
                                          decode.decode(self.P, self.seg, self.prior, alpha, ha, d))

    def test_dearer_switch_only_where_the_ball_is_missing(self):
        P = np.tile([0.8, 0.1, 0.1], (100, 1))
        P[40:60] = [0.05, 0.05, 0.9]                     # a 20-frame dip towards dead ball
        prior = np.ones(3) / 3
        none = np.zeros(100, bool)                        # no frame without the ball
        gap = np.ones(100, bool)                          # (a short gap can be stepped round: switch just outside it)
        self.assertTrue((visdecode.decode(P, [(0, 100)], prior, 0.0, 8, 8, 8, none)[40:60] == 2).all())
        self.assertTrue((visdecode.decode(P, [(0, 100)], prior, 0.0, 8, 8, 8, gap) == 0).all())

    def test_nested_dispatch(self):
        half = np.repeat([1, 2], [150, 250]).astype(np.int8)
        missing = np.zeros(400, bool)
        np.testing.assert_array_equal(nested.plain_path(self.P, half, self.prior, (0.0, 4, 8)),
                                      decode.decode(self.P, nested.segments_of(half), self.prior, 0.0, 4, 8))
        np.testing.assert_array_equal(nested.plain_path(self.P, half, self.prior, (0.0, 4, 8, 4), missing),
                                      decode.decode(self.P, nested.segments_of(half), self.prior, 0.0, 4, 8))

    def test_variants_reuse_y23_big2_caches(self):
        V = variant.VARIANTS
        for name in ("y26_heavysel", "y26_viscost", "y26_cuts", "y26_visw", "y23_big2_xn",
                     "y26_viscost_rand", "y26_cuts_rand", "y26_visw_rand"):
            base = V["y23_big2_rand" if name.endswith(("_rand", "_xn")) else "y23_big2"]
            self.assertEqual(variant.feature_version(V[name]), variant.feature_version(base), name)
            self.assertEqual(variant.stage1_version(V[name]), variant.stage1_version(base), name)
            self.assertEqual(V[name]["seq"]["settings"], base["seq"]["settings"], name)
            self.assertEqual(V[name]["ensemble"], base["ensemble"], name)
        for name in ("y26_viscost_r_rand", "y26_cuts_r_rand", "y26_visw_r_rand", "y26_heavysel_r_xn"):
            base = V["y23_big_s2028_rand"]
            self.assertEqual(variant.stage1_version(V[name]), variant.stage1_version(base), name)
            self.assertEqual(V[name]["seq"]["seed"], 2028, name)
            self.assertEqual(V[name]["seq"]["settings"], base["seq"]["settings"], name)
        self.assertNotIn("seeds", V["y26_visw_r_rand"]["seq"])
        self.assertTrue(V["y26_heavysel"]["cross"] and not V["y26_heavysel"].get("eval_rand"))
        self.assertTrue(V["y23_big2_xn"]["cross"] and V["y23_big2_xn"]["eval_rand"])


if __name__ == "__main__":
    unittest.main()
