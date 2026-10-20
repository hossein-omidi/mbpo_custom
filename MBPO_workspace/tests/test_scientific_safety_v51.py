"""Targeted regression tests: do not publish partial rewards or invalid oracle upper bounds."""
import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import sys
import types
import numpy as np
from MBPO_workspace.src.advanced_suite import _read_eval, export_detail
from MBPO_workspace.src.evaluation import ideal_instantaneous_poa
from MBPO_workspace.tests.test_advanced_suite_v5 import sample_paths, setup_dir
from MBPO_workspace.tests.test_y_poa_reference import MinimalEnv

class ScientificSafetyTests(unittest.TestCase):
    def test_partial_reward_column_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            setup_dir(root,sample_paths(),rich=True)
            fp=root/'trajectories_detailed.csv'
            with fp.open(newline='') as f:
                r=csv.DictReader(f); fields=r.fieldnames;rows=list(r)
            row=next(x for x in rows if x['method']=='sac' and x['step']=='3')
            row['net_step_kwh']=''
            with fp.open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
            with self.assertRaisesRegex(ValueError,'Partly missing'):
                _read_eval(root)

    def test_oracle_rejects_failure_to_dominate_fixed(self):
        physics=types.ModuleType('mbpo.env.pvlib_physics')
        physics.compute_panel_power_w=lambda tilt,az,zen,saz,dni,ghi,dhi,area,efficiency: 5.
        infos=[{'solar_zenith_deg':45.,'solar_azimuth_deg':180.,
                'dni_wm2':800.,'ghi_wm2':850.,'dhi_wm2':50.}]
        fixed={'scenario_id':'2020-06-21','seed':42,'initial_pose':[20.,180.],
               'infos':infos,'energy':[10./12000.],'power':[10.] }
        with patch.dict(sys.modules,{'mbpo.env.pvlib_physics':physics}):
            with self.assertRaisesRegex(RuntimeError,'below paired fixed power'):
                ideal_instantaneous_poa(None,fixed,env=MinimalEnv())

if __name__=='__main__':unittest.main()
