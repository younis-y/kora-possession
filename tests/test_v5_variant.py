"""V5 preserves the B0 grid order, defaults and stage-1 caches."""
import unittest
from kora.boundary import GRID
from kora.variant import DECODER, VARIANTS, decoder, feature_version, stage1_version


class V5VariantTests(unittest.TestCase):
    def test_grid_defaults_and_caches(self):
        baseline, candidate = VARIANTS["b0"], VARIANTS["v5_wide"]
        grid = candidate["decoder"]["hsmm_grid"]
        self.assertEqual((grid[:len(GRID)], len(grid), len(set(grid))), (GRID, 180, 180))
        self.assertEqual(decoder(baseline), DECODER)
        self.assertEqual(decoder(candidate), {**DECODER, "hsmm_grid": grid})
        for version in (feature_version, stage1_version):
            self.assertEqual(version(candidate), version(baseline))
