"""Study-quality scenario design, inference, and offline plotting regression."""
import json
import sys
import types
from unittest.mock import patch
import tempfile
import unittest
from datetime import date,timedelta,datetime,timezone
from pathlib import Path

from SAC_workspace.src.evaluation.scenario_sampling import (
    build_selection, validate_selection, save_selection, resolve_cli_selection)
from SAC_workspace.src.evaluation.stratified_statistics import (
    make_episode_features, analyse_features, render_statistical_report, load_episode_features)


def frame(first,last):
    current=date.fromisoformat(first);end=date.fromisoformat(last);scenarios=[]
    while current<=end:
        scenarios.append({'scenario_id':current.isoformat(), 'year':current.year,
                          'month':current.month,'day':current.day,'source_year':current.year})
        current+=timedelta(days=1)
    return {'scenarios':scenarios,'episode':{'start_time':'12:00','periods':118,'freq':'5min'}}


def path(method,sid,energy):
    y,m,d=map(int,sid.split('-'))
    base=datetime(y,m,d,12,0,tzinfo=timezone.utc)
    fields=[]
    for t in range(117):
        fields.append({'timestamp_utc_iso':(base+timedelta(minutes=5*(t+1))).isoformat(),
                       'solar_zenith_deg':45.,'solar_azimuth_deg':180.,
                       'dni_wm2':750.,'dhi_wm2':125.,'ghi_wm2':850.,
                       'weather_condition':'clear'})
    ideal=method=='dual_axis_instant_ideal'
    return dict(method=method,scenario_id=sid,seed=42,
                initial_pose=[45.,180.],infos=fields,
                energy=[energy/117]*117,power=[energy/117*12000]*117,
                reward=[energy/117]*117,cost=[0.]*117,
                tilt=[45.]*117,azimuth=[180.]*117,
                action_tilt=[float('nan') if ideal else 0.]*117,
                action_azimuth=[float('nan') if ideal else 0.]*117,
                delta_tilt_deg=[float('nan') if ideal else 0.]*117,
                delta_azimuth_deg=[float('nan') if ideal else 0.]*117,
                gross_energy_kwh=energy,
                movement_cost_kwh=None if ideal else 0.,
                net_energy_kwh=None if ideal else energy)


class ScenarioSampleTests(unittest.TestCase):
    def test_2557_day_frame_selects_252_balanced_dates(self):
        manifest=frame('2018-01-01','2024-12-31')
        self.assertEqual(len(manifest['scenarios']),2557)
        p=build_selection(manifest,days_per_month=3,selection_seed=13)
        self.assertEqual(p['selected_count'],252)
        self.assertEqual(p['year_month_strata_count'],84)
        self.assertEqual(p['population_count'],2557)
        self.assertEqual(sum(s['population_days'] for s in p['strata']),2557)
        self.assertEqual(len(validate_selection(p,manifest)),252)
        self.assertEqual(p,build_selection(manifest,3,13))
        self.assertNotEqual(p['selected_scenario_ids'],build_selection(manifest,3,14)['selected_scenario_ids'])
        self.assertEqual(build_selection(manifest,3,13,exhaustive=True)['selected_count'],2557)
        selected={(int(s[:4]),int(s[5:7])) for s in p['selected_scenario_ids']}
        self.assertEqual(len(selected),84)
        self.assertEqual(len(p['selected_scenario_ids']),len(set(p['selected_scenario_ids'])))
        other=frame('2018-01-01','2023-12-31')
        with self.assertRaises(ValueError):validate_selection(p,other)

    def test_excluded_training_scenarios_and_plan_tamper_rejected(self):
        m=frame('2020-01-01','2020-02-29')
        excluded={'2020-01-01','2020-01-11','2020-02-29'}
        p=build_selection(m,days_per_month=2,selection_seed=8,excluded_ids=excluded)
        self.assertEqual(p['frame_count_before_exclusions'],60)
        self.assertEqual(p['population_count'],57)
        self.assertEqual(p['selected_count'],4)
        self.assertFalse(set(p['selected_scenario_ids']) & excluded)
        validate_selection(p,m)
        corrupt=json.loads(json.dumps(p))
        corrupt['strata'][0]['population_days']+=1
        with self.assertRaises(ValueError):validate_selection(corrupt,m)

    def test_sac_legacy_reference_alias_and_ideal_mock_interval(self):
        from SAC_workspace.src.config.config import SACConfig
        from SAC_workspace.src.evaluation import evaluation as evaluation
        from pathlib import Path
        cfg=SACConfig({}, {'fixed_eval_scenarios': ['2020-01-01']}, {}, Path('mock'), tuple())
        self.assertEqual(evaluation.resolve_scenario_ids(cfg), ['2020-01-01'])
        physics=types.ModuleType('mbpo.env.pvlib_physics')
        physics.dual_axis_tracking_targets=lambda z,a:(z,a)
        physics.compute_panel_power_w=lambda tilt, az, zen, saz, dni, ghi, dhi, area, efficiency: 1200.
        fixed={'scenario_id':'2020-01-01','seed':42,'initial_pose':[30.,180.],
               'energy':[.1], 'infos':[{'solar_zenith_deg':40.,'solar_azimuth_deg':182.,
               'dni_wm2':800.,'ghi_wm2':900.,'dhi_wm2':100.,
               'timestamp_utc_iso':'2020-01-01T12:05:00+00:00'}]}
        env=types.SimpleNamespace(unwrapped=types.SimpleNamespace(area=1.,efficiency=.2),horizon=1)
        with patch.dict(sys.modules,{'mbpo':types.ModuleType('mbpo'),
                 'mbpo.env':types.ModuleType('mbpo.env'),'mbpo.env.pvlib_physics':physics}):
            out=evaluation.ideal_instantaneous_dual_axis(None,fixed,env=env)
        self.assertAlmostEqual(out['gross_energy_kwh'],.1)
        self.assertIsNone(out['net_energy_kwh'])

    def test_complete_paired_weights_and_offline_figures(self):
        # Four year/month groups -> eight sampled dates for plotting and paired stats.
        manifest={'scenarios': [r for r in frame('2020-01-01','2021-02-28')['scenarios'] if r['month'] in (1,2)]}
        plan=build_selection(manifest,days_per_month=2,selection_seed=7)
        self.assertEqual(plan['selected_count'],8)
        paths=[]
        for sid in plan['selected_scenario_ids']:
            for method,value in [('sac',0.8),('dual_axis_slew_limited',0.7),
                                 ('fixed_no_motion',0.5),('dual_axis_instant_ideal',1.1)]:
                paths.append(path(method,sid,value))
        features=make_episode_features(paths,plan)
        result=analyse_features(features,plan,n_boot=120)
        self.assertAlmostEqual(result['methods']['sac']['net_energy_kwh']['weighted_mean'],0.8)
        self.assertAlmostEqual(result['paired_net_differences']['sac_minus_fixed_no_motion']['weighted_mean'],.3)
        self.assertAlmostEqual(result['paired_net_differences']['sac_minus_fixed_no_motion']['ratio_of_weighted_means_percent'],60.)
        self.assertEqual(result['methods']['dual_axis_instant_ideal']['net_energy_kwh'],None)
        self.assertEqual(result['selected_scenarios'],8)
        self.assertEqual(result['methods']['sac']['net_energy_kwh']['represented_days'],len(manifest['scenarios']))
        self.assertEqual(result['methods']['sac']['movement_total_deg']['weighted_mean'],0.)
        with tempfile.TemporaryDirectory() as td:
            save_selection(plan,td)
            out=Path(td)/'stratified'
            report=render_statistical_report(out,plan,paths=paths,n_boot=120)
            self.assertTrue((out/'sampling_statistics.pdf').exists())
            self.assertGreaterEqual(len(report['figures']),6)
            self.assertEqual(len(load_episode_features(out)),len(features))
            regenerated=render_statistical_report(out,n_boot=120)
            self.assertEqual(regenerated['methods']['sac']['net_energy_kwh'],report['methods']['sac']['net_energy_kwh'])
            corrupted=features[:-1]
            with self.assertRaisesRegex(ValueError,'Not all controllers'):
                analyse_features(corrupted,plan,n_boot=120)

if __name__=='__main__':unittest.main()
