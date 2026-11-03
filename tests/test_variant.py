import unittest

from kora import variant


class VariantTests(unittest.TestCase):
    def test_switches_change_the_right_cache_keys(self):
        b0, touch, bal = (variant.VARIANTS[k] for k in ("b0", "touch", "touch_bal"))
        self.assertNotEqual(variant.feature_version(b0), variant.feature_version(touch))
        self.assertEqual(variant.feature_version(touch), variant.feature_version(bal))
        self.assertNotEqual(variant.stage1_version(touch), variant.stage1_version(bal))

    def test_production_is_a_known_variant(self):
        self.assertIn(variant.PRODUCTION, variant.VARIANTS)



class DecoderConfigTests(unittest.TestCase):
    def test_reference_rule_is_five_of_six_and_variants_override_it(self):
        self.assertEqual(variant.decoder(variant.VARIANTS["b0"])["inner_wins"], 5)
        self.assertEqual(variant.decoder(variant.VARIANTS["b0_w4"])["inner_wins"], 4)

    def test_decoder_switches_do_not_touch_stage1_caches(self):
        self.assertEqual(variant.stage1_version(variant.VARIANTS["b0"]),
                         variant.stage1_version(variant.VARIANTS["b0_w4"]))


if __name__ == "__main__":
    unittest.main()
