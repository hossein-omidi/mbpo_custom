"""End-to-end contract against the *real* PVTrackingEnv, not a mock.

Run inside the cloned repository with data/pv_weather/nsrdb manifest present.
This test is SKIPPED otherwise; a skip is NOT evidence of full integration.
"""
from pathlib import Path
import unittest
import numpy as np

from SAC_workspace.src.config.config import load_config, _BASE


class RealPhysicsParityTests(unittest.TestCase):
    def test_direct_pvtracking_matches_sac_wrapper_for_117_steps(self):
        if not _BASE.is_file():
            self.skipTest('Not executed inside full repository checkout')
        try:
            cfg = load_config(require_data=True)
            from mbpo.env.pv_tracking import PVTrackingEnv
            from SAC_workspace.src.environment.environment import SACEnvironment
        except (ImportError, FileNotFoundError) as e:
            self.skipTest(f'Original pvlib/gym/NSRDB runtime not installed: {e}')
        from mbpo.env.nsrdb_weather import load_scenario_manifest
        reference_ids = {row['scenario_id'] for row in load_scenario_manifest(cfg.manifest)['scenarios']}
        sid = '2018-02-15' if '2018-02-15' in reference_ids else sorted(reference_ids)[0]
        raw = PVTrackingEnv(**cfg.train_env)
        sac = SACEnvironment(cfg.train_env)
        try:
            raw.seed(42)
            sac.seed(42)
            direct_s = raw.reset(scenario_id=sid)
            wrapped_s = sac.reset(scenario_id=sid)
            np.testing.assert_array_equal(direct_s, wrapped_s)
            for step in range(117):
                a = np.array([0.6 * np.sin(step/7.), -0.7 * np.cos(step/9.)], dtype=np.float32)
                ds, dr, dd, di = raw.step(a)
                ws, wr, wd, wi = sac.step(a)
                np.testing.assert_array_equal(ds, ws)
                self.assertEqual(dr, wr)
                self.assertEqual(dd, wd)
                self.assertEqual(dd, step == 116)
                self.assertEqual(di['timestamp_utc_iso'], wi['timestamp_utc_iso'])
                self.assertEqual(wi['weather_resampling'], 'none_native_5min')
                self.assertAlmostEqual(wr, wi['energy_kwh'] - wi['movement_cost'], places=10)
                self.assertAlmostEqual(float(ws[5]), (116-step)/117., places=5)
            self.assertEqual(sac.steps, 117)
        finally:
            raw.close()
            sac.close()

    def test_ideal_counterfactual_matches_original_pvlib_power(self):
        """Real NSRDB episode: ideal power is computed from fixed path without resimulation."""
        if not _BASE.is_file():
            self.skipTest('Not executed inside full repository checkout')
        try:
            cfg = load_config(require_data=True)
            from SAC_workspace.src.evaluation.evaluation import (
                run_episode, ideal_instantaneous_dual_axis)
            from SAC_workspace.src.environment.environment import SACEnvironment
            from mbpo.env.pvlib_physics import (
                dual_axis_tracking_targets, compute_panel_power_w)
        except (ImportError, FileNotFoundError) as e:
            self.skipTest(f'Original pvlib/gym/NSRDB runtime not installed: {e}')
        env = SACEnvironment(cfg.eval_env)
        try:
            fixed = run_episode(cfg,'fixed_no_motion','2018-02-15',42,env=env)
            ideal = ideal_instantaneous_dual_axis(cfg,fixed,env=env)
            self.assertEqual(len(ideal['power']),117)
            self.assertIsNone(ideal['net_energy_kwh'])
            for i in (0, 35, 70, 116):
                info=fixed['infos'][i]
                tilt,az=dual_axis_tracking_targets(
                    info['solar_zenith_deg'],info['solar_azimuth_deg'])
                expected=compute_panel_power_w(
                    tilt,az,info['solar_zenith_deg'],info['solar_azimuth_deg'],
                    info['dni_wm2'],info['ghi_wm2'],info['dhi_wm2'],
                    area=env.unwrapped.area,efficiency=env.unwrapped.efficiency)
                self.assertAlmostEqual(ideal['power'][i],expected,places=10)
                self.assertEqual(ideal['infos'][i]['timestamp_utc_iso'],
                                 fixed['infos'][i]['timestamp_utc_iso'])
        finally:
            env.close()


if __name__ == '__main__':
    unittest.main()
