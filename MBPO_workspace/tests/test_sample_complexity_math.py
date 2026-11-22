import unittest
import numpy as np

from MBPO_workspace.src.sample_complexity import _threshold_steps, _auc, _bootstrap_curves


class SampleComplexityMathTests(unittest.TestCase):
    def test_threshold_requires_sustained_crossing(self):
        steps = np.asarray([100, 200, 300, 400, 500])
        curve = np.asarray([0.2, 0.8, 0.7, 0.9, 0.91])
        self.assertEqual(_threshold_steps(curve, steps, target=0.85, sustain=2), 400.0)
        self.assertIsNone(_threshold_steps(curve, steps, target=0.85, sustain=3))

    def test_auc_is_normalized_over_observed_interaction_range(self):
        steps = np.asarray([100, 200, 300])
        curve = np.asarray([1.0, 2.0, 3.0])
        self.assertAlmostEqual(_auc(curve, steps), 2.0)

    def test_two_way_bootstrap_returns_finite_ordered_bands(self):
        rng = np.random.default_rng(5)
        matrix = rng.normal(size=(5, 12, 4))
        mean, lo, hi = _bootstrap_curves(matrix, replicates=250, seed=9)
        self.assertEqual(mean.shape, (4,))
        self.assertTrue(np.isfinite(mean).all())
        self.assertTrue(np.all(lo <= mean))
        self.assertTrue(np.all(mean <= hi))


if __name__ == "__main__":
    unittest.main()
