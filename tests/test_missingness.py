import unittest

import numpy as np

from kora.missingness import runs, second_diff_var


class MissingnessTests(unittest.TestCase):
    def test_runs(self):
        s, e = runs(np.array([0, 1, 1, 0, 1], bool))
        np.testing.assert_array_equal(s, [1, 4])
        np.testing.assert_array_equal(e, [3, 5])

    def test_noise_raises_second_difference_variance(self):
        t = np.linspace(0, 10, 500)
        smooth = np.stack([t, t ** 2 / 20], 1)[:, None, :].astype(np.float32)
        noisy = smooth + np.random.default_rng(0).normal(0, 0.1, smooth.shape).astype(np.float32)
        self.assertAlmostEqual(second_diff_var(noisy) - second_diff_var(smooth), 6 * 0.01, delta=0.01)


if __name__ == "__main__":
    unittest.main()
