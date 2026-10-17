"""Regression tests for the missing-foundation and SAC counterfactual errors."""
import sys
import types
import unittest
from unittest.mock import Mock, patch
from pathlib import Path

import numpy as np

from SAC_workspace.src.config.config import SACConfig
from SAC_workspace.src.evaluation import evaluation as sac_evaluation
from MBPO_workspace.src.evaluation import METHODS, FEASIBLE
from MBPO_workspace.src.config import load_config as load_mbpo_config
from MBPO_workspace.src.paper_reporting import expand_paths


class RepairInterfacesTest(unittest.TestCase):
    def test_both_mbpo_and_reporting_imports(self):
        import MBPO_workspace.src.config
        import MBPO_workspace.src.evaluation
        import MBPO_workspace.src.plots
        import MBPO_workspace.src.dynamics
        import MBPO_workspace.src.replay
        import MBPO_workspace.src.trainer

    def test_reference_fallback_when_compact_sets_none(self):
        cfg = SACConfig({}, {'fixed_eval_scenarios': None}, {}, Path('mock'), tuple())
        with patch.object(sac_evaluation.reference_config, '_constant', return_value=['2018-02-15', '2024-06-21']) as ref:
            self.assertEqual(sac_evaluation.reference_scenario_ids(cfg), ['2018-02-15', '2024-06-21'])
            ref.assert_called_once()
        cfg2 = SACConfig({}, {'fixed_eval_scenarios': ['2019-05-15']}, {}, Path('mock'), tuple())
        self.assertEqual(sac_evaluation.reference_scenario_ids(cfg2), ['2019-05-15'])

    def test_sun_pointing_reuses_fixed_weather_without_any_simulator_steps(self):
        physics=types.ModuleType('mbpo.env.pvlib_physics')
        physics.dual_axis_tracking_targets=lambda z,az:(min(max(z,0),90),az)
        physics.compute_panel_power_w=lambda tilt, az, zen, saz, dni,ghi,dhi,area,efficiency: float(dni*area*efficiency)
        info={'solar_zenith_deg':42., 'solar_azimuth_deg':190.,
              'dni_wm2':800., 'ghi_wm2':900.,'dhi_wm2':100.,
              'timestamp_utc_iso':'2020-06-21T12:05:00+00:00'}
        fixed=dict(scenario_id='2020-06-21',seed=7,initial_pose=[30.,180.],
                   energy=[0.012,0.012],infos=[info,info])
        physical=types.SimpleNamespace(area=1.,efficiency=.18,interval_hours=1./12.)
        env=types.SimpleNamespace(unwrapped=physical,horizon=2)
        with patch.dict(sys.modules,{'mbpo':types.ModuleType('mbpo'),
                 'mbpo.env':types.ModuleType('mbpo.env'),
                 'mbpo.env.pvlib_physics':physics}):
            path=sac_evaluation.ideal_instantaneous_dual_axis(None,fixed,env=env)
        self.assertAlmostEqual(path['gross_energy_kwh'],2*800*.18/12000.)
        self.assertEqual(path['initial_pose'],fixed['initial_pose'])
        self.assertEqual(len(path['energy']),2)
        self.assertIsNone(path['net_energy_kwh'])
        self.assertIsNone(path['movement_cost_kwh'])

    def test_no_missing_workflow_imports(self):
        import importlib
        importlib.import_module('MBPO_workspace.workflows.01_setup_config')
        importlib.import_module('MBPO_workspace.workflows.02_train')
        importlib.import_module('MBPO_workspace.workflows.03_evaluate')
        importlib.import_module('MBPO_workspace.workflows.04_advanced_report')


if __name__=='__main__':unittest.main()
