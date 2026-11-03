import unittest

import numpy as np

from kora.train import balance_weights


class BalanceWeightTests(unittest.TestCase):
    def test_lambda_zero_is_uniform(self):
        y = np.array([0] * 50 + [1] * 5 + [2] * 10 + [0] * 20, np.int8)
        np.testing.assert_allclose(balance_weights(y, 0.0), 1.0)

    def test_short_possessions_weigh_more_and_mean_is_one(self):
        y = np.array([0] * 50 + [1] * 5 + [2] * 10 + [0] * 20, np.int8)
        w = balance_weights(y, 0.5)
        self.assertAlmostEqual(w.mean(), 1.0)
        self.assertGreater(w[52], w[10])
        mean_len = (50 * 50 + 5 * 5 + 20 * 20) / 75                # live frames, by their possession length
        self.assertAlmostEqual(w[52] / w[10], (0.5 + 0.5 * mean_len / 5) / (0.5 + 0.5 * mean_len / 50))
        self.assertAlmostEqual(w[60] / w[10], 1 / (0.5 + 0.5 * mean_len / 50))


if __name__ == "__main__":
    unittest.main()
