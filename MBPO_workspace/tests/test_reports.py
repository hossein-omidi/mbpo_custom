import csv
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from MBPO_workspace.src.evaluation import METHODS,paired_summary,write_outputs
from MBPO_workspace.src.plots import make_figures


class ReportTests(unittest.TestCase):
    def _path(self,method,day):
        from datetime import datetime,timedelta,timezone
        energy={'mbpo':.02,'sac':.019,'fixed_no_motion':.014,
                'dual_axis_slew_limited':.018,'dual_axis_instant_ideal':.022,
                'poa_instant_ideal':.023}[method]
        n=117
        ts=datetime(2020,6,day,12,0,tzinfo=timezone.utc)
        infos=[{'timestamp_utc_iso':(ts+timedelta(minutes=5*(i+1))).isoformat(),
                'season':'summer'} for i in range(n)]
        return dict(method=method,scenario_id=f'2020-06-{day:02d}',seed=42,
            initial_pose=[30,180],infos=infos,
            energy=[energy]*n,power=[energy*12000]*n,cost=[0.]*n,
            reward=[energy]*n,tilt=[30.+i/120. for i in range(n)],
            azimuth=[180.+i/200. for i in range(n)],
            gross_energy_kwh=energy*n,
            net_energy_kwh=energy*n if method not in ('dual_axis_instant_ideal','poa_instant_ideal') else None,
            movement_cost_kwh=0. if method not in ('dual_axis_instant_ideal','poa_instant_ideal') else None)

    def test_paired_comparison_and_2d_3d_plots(self):
        paths=[self._path(m,day) for day in (1,2) for m in METHODS]
        stats=paired_summary(paths)
        self.assertAlmostEqual(stats['paired_net_kwh']['mbpo_minus_sac']['mean_difference_kwh'],.117)
        with tempfile.TemporaryDirectory() as d:
            write_outputs(paths,d)
            plots=make_figures(paths,d)
            self.assertTrue((Path(d)/'comparison_report.pdf').is_file())
            self.assertTrue((Path(d)/'sac_reference'/'matched_trajectory.png').is_file())
            self.assertTrue((Path(d)/'05_orientation_3d.png').is_file())
            self.assertGreaterEqual(len(plots),7)
            j=json.loads((Path(d)/'summary.json').read_text())
            self.assertEqual(j['scenario_count'],2)
            with (Path(d)/'episodes.csv').open() as f:
                self.assertEqual(len(list(csv.DictReader(f))),12)

if __name__=='__main__':unittest.main()
