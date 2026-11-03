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


if __name__ == "__main__":
    unittest.main()
