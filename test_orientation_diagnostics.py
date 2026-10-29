import unittest
import numpy as np
import pandas as pd
from pv_orientation_diagnostics import expected_pose, wrap_delta, check_group
class MotionContractTest(unittest.TestCase):
    def test_wrap_and_tilt_saturation(self):
        t,a=expected_pose([89,0,40], [359,1,180], [1,-1,0], [.2,-.2,0])
        np.testing.assert_allclose(t,[90,0,40]);np.testing.assert_allclose(a,[1,359,180])
        self.assertAlmostEqual(float(wrap_delta(-358)),2.)
    def test_finite_horizon_motion_integrity(self):
        rows=[];tilt=30.;az=359.
        for step in range(1,118):
            pt,pa=tilt,az
            at,aa=(1., .2) if step < 15 else (-1., -.2)
            tilt,az=expected_pose(pt,pa,at,aa)
            rows.append(dict(step=step,power_w=100.,energy_kwh=100./12000.,tilt_deg=float(tilt),azimuth_deg=float(az),pre_tilt_deg=float(pt),pre_azimuth_deg=float(pa),action_tilt=at,action_azimuth=aa))
        d=check_group(pd.DataFrame(rows))
        self.assertLess(d['max_tilt_mapping_error_deg'],1e-9)
        self.assertLess(d['max_azimuth_mapping_error_deg'],1e-9)
    def test_direction_sign_flip_detected(self):
        rows=[];tilt=30.;az=359.
        for step in range(1,118):
            pt,pa=tilt,az;at=1. if step<10 else -1.
            tilt,az=expected_pose(pt,pa,at,0)
            rows.append(dict(step=step,power_w=100.,energy_kwh=100./12000.,tilt_deg=float(tilt),azimuth_deg=float(az),pre_tilt_deg=float(pt),pre_azimuth_deg=float(pa),action_tilt=at,action_azimuth=0.))
        self.assertGreater(check_group(pd.DataFrame(rows))['strong_adjacent_tilt_action_reversals'],0)
if __name__=='__main__':unittest.main()
