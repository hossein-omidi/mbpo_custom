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
    load_nsrdb_year_csv,
    load_scenario_manifest,
    resample_weather_to_episode_times,
    scenario_episode_date,
)
from mbpo.env.nsrdb_iotools import read_nsrdb_csv_to_env_weather
from mbpo.env.pvlib_physics import compute_panel_power_w
from scripts.eval_utils import make_baseline_rollout, verify_rollout_pvlib_power

MANIFEST = default_manifest_path()
PYTHON = sys.executable


@pytest.fixture(scope='module')
def manifest_available():
    if not os.path.isfile(MANIFEST):
        pytest.skip('NSRDB manifest missing; run prepare_nsrdb_newyork_catalog.py')
    return load_scenario_manifest(MANIFEST)


def _env_kwargs(**overrides):
    base = dict(
        latitude=40.72,
        longitude=-74.01,
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
    start_time = ep.get('start_time', '12:00')
    periods = ep.get('periods', 118)
    freq = ep.get('freq', '5min')
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
        # Compare near solar noon: sun-pointing only beats fixed-south when
        # direct beam is present (overcast/diffuse-only days are exempt).
        midday = len(env.times) // 2
        env.step_index = midday
        env.current_time = env.times[midday]
        env.tilt, env.azimuth = 30.0, 180.0
        sp = env._solar_position(env.current_time)
        w = env._current_weather()
        p_south = env._power_from_orientation(sp.zenith, sp.azimuth, 30.0, 180.0)
        p_track = env._power_from_orientation(sp.zenith, sp.azimuth, sp.zenith, sp.azimuth)
        assert p_track > p_south or w['dni'] < 10.0
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


def test_in_train_eval_cycles_validation_scenarios(manifest_available):
    """MBPO in-train eval must visit each fixed_eval_scenario once per epoch."""
    ids = [
        '2018-02-15', '2019-05-15', '2020-08-15', '2021-11-15', '2024-06-21',
    ]
    env = PVTrackingEnv(**_env_kwargs(
        randomize_scenario=False,
        fixed_eval_scenarios=ids,
    ))
    try:
        env.begin_evaluation_rollouts(len(ids))
        seen = []
        for _ in range(len(ids)):
            env.reset()
            seen.append(env._current_scenario['scenario_id'])
        assert seen == ids
    finally:
        env.close()


def test_seed_maps_fixed_scenario_when_not_cycling(manifest_available):
    ids = [
        '2018-02-15', '2019-05-15', '2020-08-15', '2021-11-15', '2024-06-21',
    ]
    env = PVTrackingEnv(**_env_kwargs(
        randomize_scenario=False,
        fixed_eval_scenarios=ids,
    ))
    try:
        env.seed(7)
        env.reset()
        sid1 = env._current_scenario['scenario_id']
        env.seed(7)
        env.reset()
        sid2 = env._current_scenario['scenario_id']
        assert sid1 == sid2 == ids[7 % len(ids)]
    finally:
        env.close()


def test_baselines_share_scenario_id(manifest_available):
    env = PVTrackingEnv(**_env_kwargs(fixed_eval_scenarios=['2020-06-21']))
    try:
        paths = {}
        for method in ('sun_tracking', 'fixed_no_motion'):
            env.seed(7)
            paths[method] = make_baseline_rollout(env, method, path_length=117, seed=7)
        sid0 = paths['sun_tracking']['infos'][0].get('scenario_id')
        sid1 = paths['fixed_no_motion']['infos'][0].get('scenario_id')
        assert sid0 == sid1 == '2020-06-21'
    finally:
        env.close()


def test_control_interval_matches_native_5min(manifest_available):
    env = PVTrackingEnv(**_env_kwargs())
    try:
        env.reset()
        assert env.num_action_steps == 117
        assert len(env.times) == 118
        _, _, _, info = env.step(np.zeros(2, dtype=np.float32))
        assert abs(info['control_interval_minutes'] - 5.0) < 1e-6
        assert info.get('weather_native_interval_minutes') == 5
        assert info.get('weather_resampling') == 'none_native_5min'
        assert info['interval_hours'] == pytest.approx(5.0 / 60.0)
        assert info['num_action_steps'] == 117
    finally:
        env.close()


def test_episode_terminates_after_117_transitions(manifest_available):
    env = PVTrackingEnv(**_env_kwargs())
    try:
        env.reset()
        steps = 0
        done = False
        while not done:
            _, _, done, _ = env.step(np.zeros(2, dtype=np.float32))
            steps += 1
        assert steps == 117
    finally:
        env.close()


def test_nsrdb_weather_exact_timestamp_alignment(manifest_available):
    """Episode weather rows must match NSRDB index exactly (no interpolation)."""
    from mbpo.env.nsrdb_weather import NsrdbYearCache

    env = PVTrackingEnv(**_env_kwargs())
    try:
        env.seed(3)
        env.reset()
        year = int(env._current_scenario['source_year'])
        raw = NsrdbYearCache(env._nsrdb_manifest).get_year(year)
        for i, t in enumerate(env.times):
            assert t in raw.index
            assert env.weather_profile.index[i] == t
            assert float(env.weather_profile.iloc[i]['ghi']) == float(raw.loc[t, 'ghi'])
    finally:
        env.close()


def test_same_scenario_reset_is_deterministic(manifest_available):
    """Re-sampling the same scenario_id yields an identical weather trajectory."""
    env = PVTrackingEnv(**_env_kwargs())
    try:
        sid = '2020-08-15'
        env.reset(scenario_id=sid)
        profile1 = env.weather_profile.copy()
        env.reset(scenario_id=sid)
        profile2 = env.weather_profile.copy()
        assert env._current_scenario['scenario_id'] == sid
        assert np.allclose(
            profile1[['ghi', 'dni', 'dhi']].values,
            profile2[['ghi', 'dni', 'dhi']].values)
    finally:
        env.close()


def test_weighted_scenario_sampling_uses_manifest_probabilities():
    from mbpo.env.nsrdb_weather import normalized_scenario_probabilities

    scenarios = [
        {'scenario_id': 'a', 'weight': 1.0},
        {'scenario_id': 'b', 'weight': 3.0},
    ]
    probs = normalized_scenario_probabilities(scenarios)
    assert probs.shape == (2,)
    assert probs[1] == pytest.approx(0.75)
    assert probs.sum() == pytest.approx(1.0)

    uniform = normalized_scenario_probabilities([
        {'scenario_id': 'x'}, {'scenario_id': 'y'}])
    assert uniform.tolist() == pytest.approx([0.5, 0.5])

    bad = normalized_scenario_probabilities([
        {'scenario_id': 'a', 'weight': -1.0},
        {'scenario_id': 'b', 'weight': float('nan')},
        {'scenario_id': 'c'},
    ])
    assert bad.tolist() == pytest.approx([1 / 3.0, 1 / 3.0, 1 / 3.0])


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


def test_all_years_2018_2024_load(manifest_available):
    """Each NSRDB year CSV loads via pvlib reader with 5-min UTC spacing."""
    from mbpo.env.nsrdb_weather import discover_year_csv_files, validate_nsrdb_year_weather

    data_dir = manifest_available.get('_data_dir', os.path.dirname(MANIFEST))
    year_map = discover_year_csv_files(data_dir, years=range(2018, 2025))
    assert set(year_map.keys()) == set(range(2018, 2025)), sorted(year_map)
    for year, path in sorted(year_map.items()):
        weather, meta = read_nsrdb_csv_to_env_weather(path)
        assert meta.get('_reader') in ('read_nsrdb_psm4', 'read_psm3')
        validate_nsrdb_year_weather(weather, path=path)
        assert 'newyork_{}_5min'.format(year) in os.path.basename(path)


def test_nsrdb_path_does_not_load_historical_catalog(manifest_available):
    """NSRDB env must not touch PVGIS-TMY historical_weather catalog."""
    env = PVTrackingEnv(**_env_kwargs())
    try:
        assert env.weather_source == 'nsrdb_multiyear'
        assert env._historical_weather_catalog is None
        assert env._nsrdb_year_cache is not None
        env.reset()
        assert env.weather_profile is not None
        assert 'ghi' in env.weather_profile.columns
    finally:
        env.close()


def test_nsrdb_csv_read_via_pvlib_iotools(manifest_available):
    """Local SAM CSVs load through pvlib read_nsrdb_psm4 / read_psm3 (tests/test1.py path)."""
    sample = manifest_available['scenarios'][0]
    year_path = sample.get('file_path') or sample.get('year_file')
    if not year_path or not os.path.isfile(year_path):
        year_path = os.path.join(
            os.path.dirname(MANIFEST),
            'newyork_{}_5min.csv'.format(int(sample.get('source_year', sample['year']))))
    assert os.path.isfile(year_path), year_path
    weather, meta = read_nsrdb_csv_to_env_weather(year_path)
    assert meta.get('_reader') in ('read_nsrdb_psm4', 'read_psm3')
    assert weather.index.tz is not None
    for col in ('ghi', 'dni', 'dhi', 'temperature', 'wind_speed'):
        assert col in weather.columns
    via_loader = load_nsrdb_year_csv(year_path)
    assert np.allclose(
        via_loader[['ghi', 'dni', 'dhi']].values,
        weather[['ghi', 'dni', 'dhi']].values)


def test_env_power_matches_pvlib_physics(manifest_available):
    """env.step power uses shared pvlib_physics (POA × area × efficiency)."""
    env = PVTrackingEnv(**_env_kwargs())
    try:
        env.seed(42)
        env.reset()
        path = make_baseline_rollout(env, 'fixed_no_motion', path_length=20, seed=42)
        ok, max_err, _ = verify_rollout_pvlib_power(
            path, area=env.area, efficiency=env.efficiency, atol=0.5, max_steps=20)
        assert ok, 'pvlib power mismatch max_err={}'.format(max_err)
        _, _, _, info = env.step(np.zeros(2, dtype=np.float32))
        expected = compute_panel_power_w(
            info['tilt'], info['azimuth'],
            info['solar_zenith_deg'], info['solar_azimuth_deg'],
            info['dni_wm2'], info['ghi_wm2'], info['dhi_wm2'],
            area=env.area, efficiency=env.efficiency)
        assert abs(float(info['power']) - expected) < 0.5
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
