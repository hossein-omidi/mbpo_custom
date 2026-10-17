import sys
import types
import unittest
from unittest.mock import patch
import numpy as np
from MBPO_workspace.src.evaluation import ideal_instantaneous_poa


class MinimalEnv:
    horizon=3
    def __init__(self):
        self.unwrapped=self
        self.tilt=30.;self.azimuth=180.;self.area=1.;self.efficiency=.18
        self.interval_hours=1/12
        self.pos=0
    def seed(self,seed):pass
    def reset(self,scenario_id=None):self.pos=0;return np.zeros(6,dtype=np.float32)
    def step(self,action):
        self.pos+=1
        return np.zeros(6,dtype=np.float32),0.,self.pos==self.horizon,{
            'solar_zenith_deg':50.,'solar_azimuth_deg':180.,
            'dni_wm2':800.,'ghi_wm2':900.,'dhi_wm2':100.}
    def close(self):pass

class IdealReferenceTest(unittest.TestCase):
    def test_poa_optimization_uses_project_physics_function_and_no_net_claim(self):
        physics=types.ModuleType('mbpo.env.pvlib_physics')
        physics.compute_panel_power_w=lambda tilt,az,zen,saz,dni,ghi,dhi,area,efficiency: (
            1000. - (tilt-35.)**2 - ((az-saz+180.)%360.-180.)**2)
        # The counterfactual must use already recorded solar/weather trajectory.
        # It must never reseed/reset/re-step the physical environment.
        path={'scenario_id':'2020-06-21','seed':42,'initial_pose':[30.,180.],
              'energy':[0.01]*3,
              'infos':[{'timestamp_utc_iso':f'2020-06-21T12:{5*(i+1):02d}:00+00:00',
                        'solar_zenith_deg':50.,'solar_azimuth_deg':180.,
                        'dni_wm2':800.,'ghi_wm2':900.,'dhi_wm2':100.} for i in range(3)]}
        env=MinimalEnv()
        env.seed=lambda *a,**k: self.fail('Counterfactual should not seed environment')
        env.reset=lambda *a,**k: self.fail('Counterfactual should not reset environment')
        env.step=lambda *a,**k: self.fail('Counterfactual should not step environment')
        with patch.dict(sys.modules,{'mbpo':types.ModuleType('mbpo'),
             'mbpo.env':types.ModuleType('mbpo.env'),
             'mbpo.env.pvlib_physics':physics}):
            p=ideal_instantaneous_poa(None,path,env=env)
        self.assertEqual(len(p['energy']),3)
        self.assertTrue(all(abs(t-35.)<1e-3 for t in p['tilt']))
        self.assertIsNone(p['net_energy_kwh'])
        self.assertIsNone(p['movement_cost_kwh'])
        self.assertAlmostEqual(p['gross_energy_kwh'],3*1000/12000,places=7)

if __name__=='__main__':unittest.main()
