"""Checks for Y29 (kora/mix.py): how the training copies are composed, and the variants' cache keys."""
import unittest

from kora import mix, variant


class Y29Tests(unittest.TestCase):
    def test_training_sets(self):
        base = [("c0", 0), ("c0m", 0), ("c1", 1), ("c1m", 1)]
        make = lambda j, kind: [(f"e{j}{kind}", kind), (f"e{j}{kind}m", kind)]
        x4 = mix.training_sets("m", base, variant.MIX["y29_x4"], make)
        self.assertEqual([x for x, _ in x4], ["c0", "c0m", "c1", "c1m", "e0cam", "e0camm", "e1rand", "e1randm"])
        r34 = mix.training_sets("m", base, variant.MIX["y29_r34"], make)
        self.assertEqual(sum(k in (1, "rand") for _, k in r34), 6)        # 3 of 4 copies (and mirrors) random
        r1 = mix.training_sets("m", base, variant.MIX["y29_r1"], make)
        self.assertEqual([x for x, _ in r1], ["c1", "c1m", "e0rand", "e0randm"])

    def test_variants_keep_the_base_caches(self):
        for name in variant.MIX:
            for suffix, base in (("", "mixed"), ("_rand", "mixed_rand")):
                v, b = variant.VARIANTS[name + suffix], variant.VARIANTS[base]
                self.assertEqual(variant.feature_version(v), variant.feature_version(b))
                self.assertNotEqual(variant.stage1_version(v), variant.stage1_version(b))
                self.assertEqual(variant.stage1_version({k: x for k, x in v.items() if k != "mix"}),
                                 variant.stage1_version(b))
        keys = {variant.stage1_version(variant.VARIANTS[n]) for n in variant.MIX}
        self.assertEqual(len(keys), 3)

    def test_stage_b_reuses_the_winner_and_the_networks(self):
        for suffix in ("", "_rand"):
            v = variant.VARIANTS["y29_prod" + suffix]
            self.assertEqual(variant.stage1_version(v), variant.stage1_version(variant.VARIANTS["y29_r34" + suffix]))
            self.assertEqual(v["seq"], variant.VARIANTS["y23_big2" + suffix]["seq"])


if __name__ == "__main__":
    unittest.main()
