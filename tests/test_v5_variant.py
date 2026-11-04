"""V5 round 1 (RESEARCH.md): v5_wide must start with B0's HSMM grid and change nothing else."""
import unittest

from kora.boundary import GRID
from kora.variant import DECODER, VARIANTS, decoder


class V5VariantTests(unittest.TestCase):
    def test_grid_keeps_b0_settings_first_and_adds_unique_settings(self):
        grid = VARIANTS["v5_wide"]["decoder"]["hsmm_grid"]
        self.assertEqual(grid[:len(GRID)], GRID)        # catches drift from kora.boundary.GRID
        self.assertEqual(len(grid), 180)
        self.assertEqual(len(set(grid)), 180)

    def test_only_the_hsmm_grid_changes(self):
        self.assertEqual(decoder(VARIANTS["b0"]), DECODER)
        resolved = decoder(VARIANTS["v5_wide"])
        self.assertEqual({k: v for k, v in resolved.items() if k != "hsmm_grid"},
                         {k: v for k, v in DECODER.items() if k != "hsmm_grid"})


if __name__ == "__main__":
    unittest.main()
