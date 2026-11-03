import itertools
import json
import unittest

from kora import variant
from kora.boundary import GRID


class V5VariantTests(unittest.TestCase):
    def test_grid_has_180_unique_settings(self):
        self.assertEqual(len(variant.V5_HSMM_GRID), 180)
        self.assertEqual(len(set(variant.V5_HSMM_GRID)), 180)

    def test_b0_grid_is_an_unchanged_prefix_and_subset(self):
        self.assertEqual(variant.V5_HSMM_GRID[:54], GRID)
        self.assertTrue(set(GRID).issubset(variant.V5_HSMM_GRID))

    def test_remaining_settings_follow_wide_product_order(self):
        base = set(GRID)
        remaining = [
            setting for setting in itertools.product((1, 2, 4, 8), (8, 16, 32), (2, 3, 4), (2, 3, 4, 6, 8))
            if setting not in base
        ]
        self.assertEqual(variant.V5_HSMM_GRID[54:], remaining)

    def test_settings_are_tuples_of_four_positive_numbers(self):
        for setting in variant.V5_HSMM_GRID:
            with self.subTest(setting=setting):
                self.assertIsInstance(setting, tuple)
                self.assertEqual(len(setting), 4)
                for value in setting:
                    self.assertIsInstance(value, (int, float))
                    self.assertGreater(value, 0)

    def test_v5_decoder_changes_only_the_hsmm_grid(self):
        self.assertEqual(variant.decoder(variant.VARIANTS["v5_wide"]), {
            "inner_wins": 5,
            "plain_grid": None,
            "hsmm_grid": variant.V5_HSMM_GRID,
        })

    def test_b0_and_decoder_defaults_are_unchanged(self):
        expected = {"inner_wins": 5, "plain_grid": None, "hsmm_grid": None}
        self.assertEqual(variant.VARIANTS["b0"], {})
        self.assertEqual(variant.DECODER, expected)
        self.assertEqual(variant.decoder(variant.VARIANTS["b0"]), expected)

    def test_variant_config_round_trips_through_json(self):
        restored = json.loads(json.dumps(variant.VARIANTS["v5_wide"]))
        self.assertEqual(restored, {
            "decoder": {"hsmm_grid": [list(setting) for setting in variant.V5_HSMM_GRID]},
        })
        self.assertEqual([tuple(setting) for setting in variant.decoder(restored)["hsmm_grid"]],
                         variant.V5_HSMM_GRID)


if __name__ == "__main__":
    unittest.main()
