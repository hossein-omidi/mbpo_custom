"""NSRDB multi-year empirical-weather scenario env tests."""

import os
import sys

import numpy as np
import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from mbpo.env.pv_tracking import PVTrackingEnv
from mbpo.env.nsrdb_weather import (
    default_manifest_path,
    load_scenario_manifest,
    resample_weather_to_episode_times,
    scenario_episode_date,
)
from scripts.eval_utils import make_baseline_rollout

MANIFEST = default_manifest_path()
PYTHON = sys.executable


@pytest.fixture(scope='module')
def manifest_available():
    if not os.path.isfile(MANIFEST):
        pytest.skip('NSRDB manifest missing; run prepare_nsrdb_multiyear_catalog.py')
    return load_scenario_manifest(MANIFEST)


def _env_kwargs(**overrides):
    base = dict(
        latitude=35.08,
        longitude=-106.65,
        start_date='2018-01-01',
        end_date='2024-12-31',
        randomize_day=False,
        randomize_scenario=True,
        randomize_initial_orientation=False,
        weather_scenario_mode='nsrdb_multiyear',
        weather_source='nsrdb_multiyear',
        scenario_manifest=MANIFEST,
        irradiance_perturbation_std=0.0,
        observation_mode='physical',
        movement_penalty=0.0,
    )
    base.update(overrides)
    return base


def test_manifest_has_multi_year_days(manifest_available):
    by_mday = {}
    for s in manifest_available['scenarios']:
        key = (int(s['month']), int(s['day']))
        by_mday.setdefault(key, set()).add(int(s.get('source_year', s['year'])))
    assert sum(1 for ys in by_mday.values() if len(ys) > 1) >= 50
    sample = manifest_available['scenarios'][0]
    assert 'scenario_id' in sample or 'id' in sample
    assert 'timestamps' in sample
    assert 'valid_step_count' in sample
    assert 'file_path' in sample or 'year_file' in sample


def test_same_seed_same_scenario_energy(manifest_available):
    env = PVTrackingEnv(**_env_kwargs())
    try:
        env.seed(123)
        env.reset()
        sid = env._current_scenario['scenario_id']
        w1 = env.weather_profile[['ghi', 'dni', 'dhi']].values.copy()
        e1 = _rollout_energy(env)
        env.seed(123)
        env.reset()
        assert env._current_scenario['scenario_id'] == sid
        w2 = env.weather_profile[['ghi', 'dni', 'dhi']].values.copy()
        assert np.allclose(w1, w2)
        e2 = _rollout_energy(env)
        assert abs(e1 - e2) < 1e-5
    finally:
        env.close()


def test_april_21_differs_across_years(manifest_available):
    env = PVTrackingEnv(**_env_kwargs())
    try:
        pool = env._nsrdb_by_mday.get((4, 21), [])
        if len(pool) < 2:
            pytest.skip('need >=2 years for April 21 in manifest')
        ghi_traces = []
        for s in pool[:5]:
            e = PVTrackingEnv(**_env_kwargs())
            e._current_scenario = s
            e.current_date = scenario_episode_date(s, tz=e.location.tz)
            e.times = e._build_times(e.current_date)
            e.weather_profile = e._build_weather_profile(e.times)
            ghi_traces.append(e.weather_profile['ghi'].values.copy())
            e.close()
        assert len({tuple(np.round(g, 1)) for g in ghi_traces}) >= 2
    finally:
        env.close()


def test_feb_29_in_manifest(manifest_available):
    feb29 = [
        s for s in manifest_available['scenarios']
        if int(s['month']) == 2 and int(s['day']) == 29
    ]
    if not feb29:
        pytest.skip('no Feb 29 scenarios (non-leap years only in manifest)')
    years = {int(s['source_year']) for s in feb29}
    assert 2020 in years or 2024 in years
    env = PVTrackingEnv(**_env_kwargs())
    try:
        s = feb29[0]
        env._current_scenario = s
        env.current_date = scenario_episode_date(s, tz=env.location.tz)
        env.times = env._build_times(env.current_date)
        prof = env._build_weather_profile(env.times)
        assert not prof[['ghi', 'dni', 'dhi']].isnull().any().any()
    finally:
        env.close()


def test_no_missing_episode_windows(manifest_available):
    cache = {}
    for s in manifest_available['scenarios'][:20]:
        year = int(s.get('source_year', s['year']))
        if year not in cache:
            from mbpo.env.nsrdb_weather import NsrdbYearCache
            cache[year] = NsrdbYearCache(manifest_available).get_year(year)
        times = pd_date_range_from_scenario(s, manifest_available)
        prof = resample_weather_to_episode_times(cache[year], times)
        assert not prof[['ghi', 'dni', 'dhi', 'temperature', 'wind_speed']].isnull().any().any()


def pd_date_range_from_scenario(s, manifest):
    import pandas as pd
    ep = manifest.get('episode', {})
    start_time = ep.get('start_time', '13:30')
    periods = ep.get('periods', 79)
    freq = ep.get('freq', '7min30s')
    date = scenario_episode_date(s, tz='UTC')
    return pd.date_range(
        start='{} {}'.format(date.date(), start_time),
        periods=int(periods),
        freq=freq,
        tz='UTC',
    )


def test_pvlib_power_changes_with_orientation(manifest_available):
    env = PVTrackingEnv(**_env_kwargs())
    try:
        env.seed(0)
        env.reset()
        env.tilt, env.azimuth = 30.0, 180.0
        sp = env._solar_position(env.current_time)
        w = env._current_weather()
        p_south = env._power_from_orientation(sp.zenith, sp.azimuth, 30.0, 180.0)
        p_track = env._power_from_orientation(sp.zenith, sp.azimuth, sp.zenith, sp.azimuth)
        assert p_track > p_south or w['ghi'] < 5.0
    finally:
        env.close()


def test_state_not_from_tmy(manifest_available):
    """NSRDB scenario GHI should differ from PVGIS-TMY for same calendar day."""
    env_nsrdb = PVTrackingEnv(**_env_kwargs(
        randomize_scenario=False, scenario_manifest=MANIFEST))
    env_tmy = PVTrackingEnv(**_env_kwargs(
        latitude=35.0,
        longitude=-106.0,
        weather_scenario_mode='pvgis_tmy',
        weather_source='historical',
        scenario_manifest=None,
        start_date='2020-01-01',
        end_date='2020-12-31',
        randomize_scenario=False,
        randomize_day=True,
    ))
    try:
        pool = [s for s in env_nsrdb._nsrdb_scenarios if int(s['month']) == 6 and int(s['day']) == 21]
        if not pool:
            pytest.skip('no June 21 NSRDB scenario')
        s = pool[0]
        env_nsrdb._current_scenario = s
        env_nsrdb.current_date = scenario_episode_date(s, tz='UTC')
        env_nsrdb.times = env_nsrdb._build_times(env_nsrdb.current_date)
        env_nsrdb.weather_profile = env_nsrdb._build_weather_profile(env_nsrdb.times)
        ghi_nsrdb = float(env_nsrdb.weather_profile['ghi'].sum())

        env_tmy.seed(0)
        env_tmy.reset()
        env_tmy.current_date = scenario_episode_date(s, tz='UTC')
        env_tmy.times = env_tmy._build_times(env_tmy.current_date)
        env_tmy.weather_profile = env_tmy._build_weather_profile(env_tmy.times)
        ghi_tmy = float(env_tmy.weather_profile['ghi'].sum())
        assert abs(ghi_nsrdb - ghi_tmy) > 1.0
    finally:
        env_nsrdb.close()
        env_tmy.close()


def test_baselines_share_scenario_id(manifest_available):
    env = PVTrackingEnv(**_env_kwargs(fixed_eval_scenarios=['2020-06-21']))
    try:
        paths = {}
        for method in ('sun_tracking', 'fixed_no_motion'):
            env.seed(7)
            paths[method] = make_baseline_rollout(env, method, path_length=78, seed=7)
        sid0 = paths['sun_tracking']['infos'][0].get('scenario_id')
        sid1 = paths['fixed_no_motion']['infos'][0].get('scenario_id')
        assert sid0 == sid1 == '2020-06-21'
    finally:
        env.close()


def test_control_interval_matches_7min30s(manifest_available):
    env = PVTrackingEnv(**_env_kwargs())
    try:
        env.reset()
        _, _, _, info = env.step(np.zeros(2, dtype=np.float32))
        assert abs(info['control_interval_minutes'] - 7.5) < 1e-6
        assert info.get('weather_native_interval_minutes') == 5
        assert info.get('weather_resampling') == 'time_interpolate_to_episode_grid'
        assert info['interval_hours'] == pytest.approx(7.5 / 60.0)
    finally:
        env.close()


def test_within_episode_weather_follows_fixed_trajectory(manifest_available):
    """Stochasticity is across episodes only; within episode W_e is a fixed NSRDB trace."""
    env = PVTrackingEnv(**_env_kwargs())
    try:
        env.seed(11)
        env.reset()
        sid0 = env._current_scenario['scenario_id']
        profile_at_reset = env.weather_profile.copy()
        ghi_trace = []
        for _ in range(env.num_action_steps):
            _, _, _, info = env.step(np.zeros(2, dtype=np.float32))
            ghi_trace.append(float(info['ghi_wm2']))
            assert info['scenario_id'] == sid0
            expected = float(profile_at_reset.iloc[env.step_index]['ghi'])
            assert abs(float(info['ghi_wm2']) - expected) < 1e-3
        assert len({round(g, 1) for g in ghi_trace}) >= 2
        assert np.allclose(
            env.weather_profile[['ghi', 'dni', 'dhi']].values,
            profile_at_reset[['ghi', 'dni', 'dhi']].values)
    finally:
        env.close()


def test_weather_scenario_mode_in_info(manifest_available):
    env = PVTrackingEnv(**_env_kwargs())
    try:
        env.reset()
        _, _, _, info = env.step(np.zeros(2, dtype=np.float32))
        assert info['weather_source'] == 'nsrdb_multiyear'
        assert info['weather_scenario_mode'] == 'nsrdb_multiyear'
        assert info.get('scenario_id') is not None
        assert info.get('episode_diffuse_fraction') is not None
    finally:
        env.close()


def _rollout_energy(env):
    total = 0.0
    done = False
    while not done:
        _, r, done, _ = env.step(np.zeros(2, dtype=np.float32))
        total += float(r)
    return total
