"""Offline algebraic regression tests for read-only SAC diagnostics."""
import importlib.util
import unittest
from pathlib import Path
import numpy as np

TARGET=Path(__file__).resolve().parents[1]/'scripts'/'sac_policy_forensic.py'
spec=importlib.util.spec_from_file_location('sac_policy_forensic',TARGET)
mod=importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

class ForensicMathTests(unittest.TestCase):
    def test_horizontal_isotropic_poa(self):
        # Horizontal plane: direct horizontal + DHI = GHI when consistent.
        po=mod.isotropic_poa(np.array([30.]),np.array([180.]),
             np.array([600.]),np.array([100.]),np.array([619.615242]),
             np.array([0.]),np.array([0.]))
        self.assertAlmostEqual(float(po[0]),619.615242,places=5)

    def test_azimuth_south_azimuth_changes_direct_irradiance(self):
        kwargs=[np.array([40.]),np.array([180.]),np.array([700.]),
                np.array([120.]),np.array([700.]),np.array([45.])]
        facing=mod.isotropic_poa(*kwargs,np.array([180.]))
        away=mod.isotropic_poa(*kwargs,np.array([0.]))
        self.assertGreater(float(facing[0]),float(away[0]))

    def test_pvwatts_temperature_derate(self):
        poa=np.array([1000.]);w=np.array([2.]);t=np.array([25.])
        power,tc=mod.dc_from_poa(poa,t,w,'pvwatts_sapm',.18,-.004,'open_rack_glass_polymer')
        self.assertGreater(float(tc[0]),25.)
        self.assertLess(float(power[0]),180.)
        self.assertGreater(float(power[0]),0.)

    def test_unsupported_mount_fails_closed(self):
        with self.assertRaises(AssertionError):
            mod.dc_from_poa(np.array([700.]),np.array([25.]),np.array([2.]),
                           'pvwatts_sapm',.18,-.004,'not_a_mount')

if __name__=='__main__':unittest.main()
