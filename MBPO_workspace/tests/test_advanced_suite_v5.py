import csv
import json
import tempfile
import unittest
from datetime import datetime,timedelta,timezone
from pathlib import Path
import numpy as np
from MBPO_workspace.src.advanced_suite import (weighted_curve,render_suite,export_detail,
                                                  _read_eval, _weights, VISIBLE)


def sample_paths(include_two_days=True):
    base=datetime(2020,1,2,12,0,tzinfo=timezone.utc)
    dates=('2020-01-02','2020-01-22') if include_two_days else ('2020-01-02',)
    paths=[]
    for dindex,sid in enumerate(dates):
        for mi,m in enumerate(VISIBLE):
            factor=(dindex+1)*(.2+mi*.1)
            vals=[float(factor*100) for _ in range(117)]
            infos=[]
            stamp=base.replace(day=int(sid[-2:]))
            for j in range(117):
                infos.append({'timestamp_utc_iso':(stamp+timedelta(minutes=5*(j+1))).isoformat(),
                              'solar_zenith_deg':65.,'solar_azimuth_deg':185.,
                              'dni_wm2':400.,'dhi_wm2':90.,'ghi_wm2':450.,
                              'weather_condition':'sunny','season':'winter'})
            energy=[v/12000. for v in vals];costs=[0.]*117
            path=dict(method=m,scenario_id=sid,seed=42,initial_pose=[10.,180.],
                infos=infos,power=vals,energy=energy,cost=costs,
                reward=energy,tilt=[10.]*117,azimuth=[180.]*117,
                action_tilt=[float('nan') if m=='poa_instant_ideal' else 0.]*117,
                action_azimuth=[float('nan') if m=='poa_instant_ideal' else 0.]*117,
                delta_tilt_deg=[float('nan') if m=='poa_instant_ideal' else 0.]*117,
                delta_azimuth_deg=[float('nan') if m=='poa_instant_ideal' else 0.]*117,
                gross_energy_kwh=sum(energy),movement_cost_kwh=None if m=='poa_instant_ideal' else 0.,
                net_energy_kwh=None if m=='poa_instant_ideal' else sum(energy))
            paths.append(path)
    return paths


def setup_dir(folder,paths,rich=False):
    with (folder/'trajectories.csv').open('w',newline='') as f:
        fields=['scenario_id','seed','method','step','utc_iso','power_w','energy_kwh',
                'cumulative_gross_kwh','tilt_deg','azimuth_deg','movement_cost_kwh']
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for p in paths:
            cum=0.
            for j in range(117):
                cum+=p['energy'][j]
                writer.writerow(dict(scenario_id=p['scenario_id'],seed=p['seed'],method=p['method'],
                                     step=j+1,utc_iso=p['infos'][j]['timestamp_utc_iso'],
                                     power_w=p['power'][j],energy_kwh=p['energy'][j],
                                     cumulative_gross_kwh=cum,tilt_deg=p['tilt'][j],
                                     azimuth_deg=p['azimuth'][j],movement_cost_kwh=p['cost'][j]))
    if rich:export_detail(paths,folder)
    unique=sorted({p['scenario_id'] for p in paths})
    weights=[1,3][:len(unique)]
    plan={'protocol':'calendar_stratified_utc_daytime', 'population_count':sum(weights),
          'strata':[{'selected_scenario_id':d,'population_days':w} for d,w in zip(unique,weights)]}
    (folder/'scenario_selection.json').write_text(json.dumps(plan))


class AdvancedSuiteTests(unittest.TestCase):
    def test_weighted_mean_band_is_observed_day_variability(self):
        means,spread=weighted_curve({'a':np.array([2.,1.]),'b':np.array([6.,5.])},{'a':1,'b':3})
        self.assertTrue(np.allclose(means,[5.,4.]))
        self.assertTrue(np.allclose(spread,np.sqrt(3.)*np.ones(2)))
        mean,sd=weighted_curve({'a':[2.,7.]},{'a':100})
        self.assertIsNone(sd)
        self.assertTrue(np.allclose(mean,[2.,7.]))
        with self.assertRaises(ValueError):
            weighted_curve({'a':[1.]},{'b':1})

    def test_old_basic_csv_runs_no_fabricated_actions(self):
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d);setup_dir(folder,sample_paths())
            info=render_suite(folder)
            self.assertGreaterEqual(len(info['figures']),9)
            self.assertFalse(info['rich_metadata_used'])
            self.assertIn('06_normalized_tilt_actions_band.png: historical CSV lacks action_tilt',info['skipped'])
            self.assertTrue((folder/'advanced_suite'/'advanced_suite.pdf').exists())
            with (folder/'advanced_suite'/'curve_band_data.csv').open() as f:
                rows=list(csv.DictReader(f))
            r=next(x for x in rows if x['figure']=='01_power_weighted_mean_spread.png' and x['method']=='mbpo')
            self.assertAlmostEqual(float(r['weighted_mean']),35.)
            self.assertAlmostEqual(float(r['observed_day_rms_sd']),np.sqrt(3.)*5.)
            self.assertEqual(int(r['n_distinct_days']),2)
            self.assertEqual(int(r['represented_manifest_days']),4)
            self.assertFalse(any('ci_95' in row for row in rows))
            self.assertNotIn('poa_instant_ideal',set(x['method'] for x in rows if x['figure']=='03_cumulative_net_weighted_mean_spread.png'))

    def test_rich_export_retains_physical_measured_controls(self):
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d);setup_dir(folder,sample_paths(),rich=True)
            info=render_suite(folder)
            self.assertTrue(info['rich_metadata_used'])
            self.assertIn('06_normalized_tilt_actions_band.png',info['figures'])
            self.assertIn('18_tilt_vs_solar_geometry.png',info['figures'])

    def test_legacy_rich_csv_is_reused_offline(self):
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d);setup_dir(folder,sample_paths(),rich=True)
            new=folder/'trajectories_detailed.csv';sub=folder/'advanced';sub.mkdir()
            with new.open(newline='') as f:rows=list(csv.DictReader(f))
            for row in rows:
                row['timestamp_utc_iso']=row.pop('utc_iso')
                row['reward_kwh']=row.pop('net_step_kwh')
            with (sub/'trajectories_rich.csv').open('w',newline='') as f:
                wr=csv.DictWriter(f,fieldnames=rows[0]);wr.writeheader();wr.writerows(rows)
            new.unlink()
            info=render_suite(folder)
            self.assertTrue(info['rich_metadata_used'])
            self.assertIn('06_normalized_tilt_actions_band.png',info['figures'])
            self.assertIn('advanced/trajectories_rich.csv',info['source'])

    def test_paired_utc_and_missing_method_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d);setup_dir(folder,sample_paths())
            inp=folder/'trajectories.csv'
            with inp.open(newline='') as f:r=list(csv.DictReader(f))
            with inp.open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=r[0]);w.writeheader();w.writerows([x for x in r if x['method']!='sac'])
            with self.assertRaisesRegex(ValueError,'Incomplete'):_read_eval(folder)

if __name__=='__main__':unittest.main()
