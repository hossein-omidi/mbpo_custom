"""End-to-end reporting tests: offline reconstruction and scientific invariants."""
import csv
import copy
import tempfile
import unittest
from datetime import datetime,timedelta,timezone
from pathlib import Path
import numpy as np
from MBPO_workspace.src.paper_reporting import (
    expand_paths, write_rich_csv, render_from_csv, create_advanced_report,
)
from MBPO_workspace.src.evaluation import METHODS


class ScientificReportingTests(unittest.TestCase):
    @staticmethod
    def make_paths():
        paths=[]
        for day in (1,2):
            for method in METHODS:
                ideal=method in ('dual_axis_instant_ideal','poa_instant_ideal')
                tilt=30.;az=180.
                path=dict(method=method,scenario_id=f'2020-06-{day:02d}',seed=42,
                          initial_pose=[tilt,az],infos=[],reward=[],energy=[],power=[],
                          cost=[],tilt=[],azimuth=[],delta_tilt_deg=[],delta_azimuth_deg=[],
                          action_tilt=[],action_azimuth=[])
                for i in range(117):
                    if ideal:
                        tilt=40.;az=190.
                        dt=da=at=aa=float('nan')
                    else:
                        at=0.01 if method in ('mbpo','sac') else 0.
                        aa=0.005 if method=='dual_axis_slew_limited' else 0.
                        dt=at*5.;da=aa*10.
                        tilt=np.clip(tilt+dt,0,90);az=(az+da)%360.
                    power=dict(mbpo=320,sac=310,fixed_no_motion=260,
                               dual_axis_slew_limited=330,dual_axis_instant_ideal=350,
                               poa_instant_ideal=360)[method]+day
                    energy=power/12000.
                    ts=(datetime(2020,6,day,12,tzinfo=timezone.utc)+timedelta(minutes=5*(i+1))).isoformat()
                    path['infos'].append(dict(timestamp_utc_iso=ts,season='summer',
                        solar_zenith_deg=40.,solar_azimuth_deg=190.,
                        solar_altitude_deg=50.,dni_wm2=820.,ghi_wm2=850.,dhi_wm2=110.,
                        weather_condition='clear'))
                    for key,value in [('reward',energy),('energy',energy),('power',power),
                                      ('cost',0.),('tilt',tilt),('azimuth',az),
                                      ('delta_tilt_deg',dt),('delta_azimuth_deg',da),
                                      ('action_tilt',at),('action_azimuth',aa)]:
                        path[key].append(value)
                path['gross_energy_kwh']=sum(path['energy'])
                path['net_energy_kwh']=None if ideal else sum(path['reward'])
                path['movement_cost_kwh']=None if ideal else 0.
                paths.append(path)
        return paths

    def test_csv_2d_3d_pdf_and_offline_recreation(self):
        with tempfile.TemporaryDirectory() as d:
            data=self.make_paths()
            report=create_advanced_report(data,d)
            root=Path(d)/'advanced'
            self.assertEqual(report['episode_pairs'],2)
            self.assertTrue((root/'trajectories_rich.csv').is_file())
            self.assertTrue((root/'advanced_paper_figures.pdf').is_file())
            self.assertTrue((root/'04_hourly_power_gap.png').is_file())
            self.assertTrue((root/'08_actions_3d.png').is_file())
            self.assertTrue((root/'09_orientation_vs_sun_3d.png').is_file())
            self.assertGreaterEqual(len(report['plots']),10)
            regenerated=render_from_csv(root/'trajectories_rich.csv',Path(d)/'offline')
            self.assertEqual(regenerated['methods'],report['methods'])
            with (root/'trajectories_rich.csv').open(newline='') as f:
                rows=list(csv.DictReader(f))
            self.assertEqual(len(rows),2*len(METHODS)*117)

    def test_reject_nonchronological_or_bad_action(self):
        p=self.make_paths()
        p[0]['infos'][1]['timestamp_utc_iso']=p[0]['infos'][0]['timestamp_utc_iso']
        with self.assertRaisesRegex(ValueError,'Nonchronological'):
            expand_paths(p)
        p=self.make_paths();p[0]['action_tilt'][1]=2.0
        with self.assertRaisesRegex(ValueError,'Invalid executed'):
            expand_paths(p)

    def test_reject_false_feasible_ideal_or_unpaired_scenario(self):
        p=self.make_paths()
        p[-1]['net_energy_kwh']=0.
        with self.assertRaisesRegex(ValueError,'feasible net'):
            expand_paths(p)
        p=self.make_paths();p.pop()
        with self.assertRaisesRegex(ValueError,'Missing paired'):
            expand_paths(p)

if __name__=='__main__':unittest.main()
