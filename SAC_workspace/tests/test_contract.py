"""Lightweight tests; no pvlib/NSRDB dataset needed until integration checks."""
import unittest
import numpy as np
from SAC_workspace.src.config import config as config_module
from SAC_workspace.src.representation.representation import encode, decode
from SAC_workspace.src.config.config import load_config

class ContractTests(unittest.TestCase):
    def test_source_config_parses_without_legacy_tensorflow(self):
        if not config_module._BASE.is_file():
            self.skipTest('Reference repository not mounted: config AST tested separately in integration fixture')
        cfg = load_config(require_data=False)
        self.assertEqual(cfg.train_env['observation_mode'], 'compact')
        self.assertEqual(cfg.train_env['weather_source'], 'nsrdb_multiyear')
        self.assertEqual(cfg.algo['gamma'], 1.)
        self.assertEqual(cfg.algo['hidden_sizes'], (64,64))
        self.assertEqual(cfg.algo['reward_scale'], 100.)
        self.assertEqual(cfg.algo['updates_per_step'], 2)
        self.assertEqual(cfg.algo['epochs'], 2000)
    def test_compact_is_not_rescaled(self):
        s = np.array([.4,.2,.09,.3,-.6,.5], dtype=np.float32)
        np.testing.assert_array_equal(encode(s), s)
        self.assertAlmostEqual(decode(s)['power_w'], 180., places=4)
    def test_reject_inconsistent_observations(self):
        for s in (np.zeros(12), np.array([0,0,0,0,0,np.nan]), np.ones(6)*3):
            with self.assertRaises(ValueError): encode(s)

if __name__ == '__main__':
    unittest.main()
