"""Regressions for the POA-versus-feasible metric contract."""
import csv
import copy
import json
import tempfile
import unittest
from pathlib import Path

from MBPO_workspace.src.advanced_suite import render_suite, _read_eval
from MBPO_workspace.tests.test_advanced_suite_v5 import sample_paths, setup_dir


class ScientificPresentationV52Tests(unittest.TestCase):
    def _features(self, root, days):
        out=root/'stratified';out.mkdir(exist_ok=True)
        with (out/'episode_features.csv').open('w',newline='') as fp:
            cols=['scenario_id','method','gross_energy_kwh','net_energy_kwh','movement_total_deg']
            w=csv.DictWriter(fp,fieldnames=cols);w.writeheader()
            for ix,day in enumerate(days):
                for method in ('mbpo','sac','fixed_no_motion','poa_instant_ideal'):
                    gross=10.+ix*4
                    w.writerow({'scenario_id':day,'method':method,
                                'gross_energy_kwh':gross,
                                'net_energy_kwh':'' if method=='poa_instant_ideal' else gross-2,
                                'movement_total_deg':0})

    def test_separate_gross_and_net_seasonal_panels(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            paths=[]
            for month in (1,4,7,10):
                for original in sample_paths(include_two_days=False):
                    item=copy.deepcopy(original)
                    item['scenario_id']=f'2020-{month:02d}-02'
                    for info in item['infos']:
                        info['timestamp_utc_iso']=info['timestamp_utc_iso'].replace('2020-01-02',f'2020-{month:02d}-02')
                    paths.append(item)
            setup_dir(root,paths)
            days=sorted({p['scenario_id'] for p in paths})
            (root/'scenario_selection.json').write_text(json.dumps({
                'protocol':'calendar_stratified_utc_daytime','population_count':8,
                'strata':[{'selected_scenario_id':day,'population_days':2} for day in days]}))
            self._features(root,days)
            result=render_suite(root)
            self.assertIn('21_seasonal_weighted_energy.png',result['figures'])
            self.assertIn('27_seasonal_weighted_net_energy.png',result['figures'])

    def test_partial_solar_weather_data_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            setup_dir(root,sample_paths(),rich=True)
            path=root/'trajectories_detailed.csv'
            with path.open(newline='') as fp:
                rows=list(csv.DictReader(fp));fields=list(rows[0])
            rows[1]['dni_wm2']=''
            with path.open('w',newline='') as fp:
                w=csv.DictWriter(fp,fieldnames=fields);w.writeheader();w.writerows(rows)
            with self.assertRaisesRegex(ValueError,'Partially missing weather'):
                _read_eval(root)

if __name__=='__main__':unittest.main()
