import unittest
from SAC_workspace.src.evaluation.evaluation import summary, _ci_clustered

class PairedEvaluationTests(unittest.TestCase):
    def test_ci_deterministic(self):
        a=_ci_clustered({'d1':[1,1.2],'d2':[2,2.2]},seed=12)
        b=_ci_clustered({'d1':[1,1.2],'d2':[2,2.2]},seed=12)
        self.assertEqual(a,b)
        self.assertGreaterEqual(a[0],a[1]);self.assertLessEqual(a[0],a[2])
    def test_paired_energy_metrics(self):
        paths=[]
        for method,energy in [('sac',.8),('dual_axis_slew_limited',.9),('fixed_no_motion',.4),('dual_axis_instant_ideal',1.0)]:
            for sid in ('2019-01-01','2019-01-02'):
                paths.append({'method':method,'scenario_id':sid,'seed':42,'net_energy_kwh':energy,
                              'gross_energy_kwh':energy,'movement_cost_kwh':0})
        s=summary(paths)
        self.assertAlmostEqual(s['paired_vs_sac']['dual_axis_slew_limited']['sac_minus_reference_net_kwh'],-.1)
        self.assertEqual(s['scenario_count'],2)


class ReferenceScenarioAndIdealReuseTests(unittest.TestCase):
    """Regression: no empty default eval and no fourth simulator rollout."""

    def test_default_scenarios_from_original_reference_when_compact_is_none(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from SAC_workspace.src.evaluation import evaluation
        cfg=SimpleNamespace(eval_env={'fixed_eval_scenarios':None})
        with patch.object(evaluation.reference_config, '_constant',
                          return_value=['2018-02-15','2019-05-15']) as source:
            self.assertEqual(evaluation.resolve_scenario_ids(cfg),
                             ['2018-02-15','2019-05-15'])
            source.assert_called_once()
        cfg_user = SimpleNamespace(eval_env={'fixed_eval_scenarios': ['2024-06-21']})
        self.assertEqual(evaluation.resolve_scenario_ids(cfg_user), ['2024-06-21'])
        cfg_dup = SimpleNamespace(
            eval_env={'fixed_eval_scenarios': ['2024-06-21', '2024-06-21']})
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            evaluation.resolve_scenario_ids(cfg_dup)

    def test_ideal_uses_recorded_weather_without_stepping_env(self):
        import sys
        import types
        from unittest.mock import patch
        from SAC_workspace.src.evaluation.evaluation import ideal_instantaneous_dual_axis
        phy = types.ModuleType('mbpo.env.pvlib_physics')
        calls=[]
        phy.dual_axis_tracking_targets=lambda zen, az: (min(float(zen),90.),float(az))
        def power(tilt,az,zen,solar_az,dni,ghi,dhi,area,efficiency):
            calls.append((tilt,az,zen,solar_az,dni,ghi,dhi,area,efficiency))
            return (dni+dhi)*area*efficiency
        phy.compute_panel_power_w=power
        mbpo=types.ModuleType('mbpo');mbpo.__path__=[]
        envpkg=types.ModuleType('mbpo.env');envpkg.__path__=[]
        fake_env=types.SimpleNamespace(horizon=2,unwrapped=types.SimpleNamespace(area=1.2,efficiency=.18))
        fixed={'scenario_id':'2018-02-15','seed':42,'initial_pose':[30,180],
               'energy':[0.0, 0.0],
               'infos':[{'solar_zenith_deg':35.,'solar_azimuth_deg':155.,'dni_wm2':600.,
                         'ghi_wm2':500.,'dhi_wm2':80.,'interval_hours':5./60.,
                         'timestamp_utc_iso':'2018-02-15T12:05:00+00:00'},
                        {'solar_zenith_deg':34.,'solar_azimuth_deg':156.,'dni_wm2':650.,
                         'ghi_wm2':530.,'dhi_wm2':85.,'interval_hours':5./60.,
                         'timestamp_utc_iso':'2018-02-15T12:10:00+00:00'}]}
        with patch.dict(sys.modules,{'mbpo':mbpo,'mbpo.env':envpkg,'mbpo.env.pvlib_physics':phy}):
            out=ideal_instantaneous_dual_axis(None,fixed,env=fake_env)
        self.assertEqual(len(calls),2)
        self.assertEqual(out['net_energy_kwh'],None)
        self.assertEqual(out['initial_pose'],[30,180])
        self.assertAlmostEqual(out['gross_energy_kwh'], ((680+735)*1.2*.18)*(5./60.)/1000.)
        self.assertEqual(out['infos'][0],fixed['infos'][0])

if __name__ == '__main__':
    unittest.main()
