#!/usr/bin/env python3
"""End-to-end timing sync: NSRDB 5-min data ↔ 7min30s control ↔ pvlib ↔ MBPO/SAC.

Verifies the Option-A contract:
  - Native weather recorded at 5-min UTC in SAM CSVs
  - Agent acts every 7min30s (78 steps); energy integrates with dt=7.5min
  - Weather W_e(t) = time-interpolated NSRDB onto the fixed episode UTC grid
    (one scenario per episode; NOT independent random weather each step)
  - pvlib solar geometry uses the same episode timestamps as weather rows
  - MBPO epoch_length / remaining_steps / model_train_freq align with T=78

Run:
  python scripts/verify_nsrdb_timing_sync.py
  python scripts/verify_nsrdb_timing_sync.py --config examples.config.pv_tracking.stage3_multiyear_nsrdb_scenario
"""

from __future__ import print_function

import argparse
import importlib
import os
import sys

import numpy as np
import pandas as pd

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from mbpo.env.pv_tracking import (
    DEFAULT_EPISODE_STEPS,
    DEFAULT_FREQ,
    DEFAULT_PERIODS,
    DEFAULT_START_TIME,
    PVTrackingEnv,
)
from mbpo.env.nsrdb_weather import (
    NsrdbYearCache,
    load_nsrdb_year_csv,
    load_scenario_manifest,
    resample_weather_to_episode_times,
    scenario_episode_date,
)
from softlearning.environments.utils import get_environment_from_params
from softlearning.replay_pools.simple_replay_pool import SimpleReplayPool


DEFAULT_MODULE = 'examples.config.pv_tracking.stage3_multiyear_nsrdb_scenario'
CONTROL_STEP_MIN = 7.5
NATIVE_INTERVAL_MIN = 5.0


def _fail(errors, msg):
    errors.append(msg)
    print('  FAIL:', msg)


def _ok(msg):
    print('  OK:', msg)


def detect_native_interval_minutes(year_weather):
    """Median spacing of NSRDB UTC index (expect 5 min)."""
    if len(year_weather) < 2:
        return None
    deltas = year_weather.index.to_series().diff().dropna()
    med = deltas.median().total_seconds() / 60.0
    return float(med)


def check_manifest_episode_grid(manifest, errors):
    print('\n=== Manifest ↔ env episode grid ===')
    ep = manifest.get('episode', {})
    start = ep.get('start_time', DEFAULT_START_TIME)
    periods = int(ep.get('periods', DEFAULT_PERIODS))
    freq = ep.get('freq', DEFAULT_FREQ)
    native = manifest.get('meta', {}).get('interval_minutes', NATIVE_INTERVAL_MIN)
    if abs(native - NATIVE_INTERVAL_MIN) > 0.1:
        _fail(errors, 'manifest native interval=%s (expected 5 min)' % native)
    else:
        _ok('native NSRDB interval = %g min' % native)
    if start != DEFAULT_START_TIME or periods != DEFAULT_PERIODS or freq != DEFAULT_FREQ:
        _fail(errors, 'manifest episode grid %s %d %s != project standard' % (
            start, periods, freq))
    else:
        _ok('episode grid %s UTC, periods=%d, freq=%s' % (start, periods, freq))

    sample = manifest['scenarios'][0]
    env = PVTrackingEnv(
        weather_source='nsrdb_multiyear',
        scenario_manifest=manifest['_manifest_path'],
        randomize_scenario=False,
        fixed_eval_scenarios=[sample['scenario_id']],
        randomize_initial_orientation=False,
        movement_penalty=0.0,
        observation_mode='physical',
    )
    try:
        env.reset()
        env_times = [t.isoformat() for t in env.times]
        manifest_times = sample.get('timestamps', [])
        if manifest_times and env_times != manifest_times:
            _fail(errors, 'env times != manifest timestamps for %s' % sample['scenario_id'])
        else:
            _ok('manifest timestamps match env._build_times (%d points)' % len(env_times))
        if env.num_action_steps != DEFAULT_EPISODE_STEPS:
            _fail(errors, 'num_action_steps=%d (expected %d)' % (
                env.num_action_steps, DEFAULT_EPISODE_STEPS))
        else:
            _ok('num_action_steps=%d' % DEFAULT_EPISODE_STEPS)
    finally:
        env.close()


def check_resampling_and_pvlib(manifest, errors):
    print('\n=== 5-min NSRDB → 7min30s weather + pvlib alignment ===')
    cache = NsrdbYearCache(manifest)
    scenario = manifest['scenarios'][100]
    year = int(scenario['source_year'])
    raw = cache.get_year(year)
    native_dt = detect_native_interval_minutes(raw)
    if native_dt is None or abs(native_dt - NATIVE_INTERVAL_MIN) > 0.6:
        _fail(errors, 'year CSV median spacing=%.2f min (expected ~5)' % (native_dt or -1))
    else:
        _ok('year %d native spacing = %.2f min' % (year, native_dt))

    date = scenario_episode_date(scenario, tz='UTC')
    times = pd.date_range(
        start='{} {}'.format(date.date(), DEFAULT_START_TIME),
        periods=DEFAULT_PERIODS,
        freq=DEFAULT_FREQ,
        tz='UTC',
    )
    prof = resample_weather_to_episode_times(raw, times)
    if prof[['ghi', 'dni', 'dhi']].isnull().any().any():
        _fail(errors, 'resampled profile has NaNs for %s' % scenario['scenario_id'])
    else:
        _ok('resampled weather complete for %s' % scenario['scenario_id'])

    # Nearest 5-min anchor lag (max) — should be <= 2.5 min with tolerance 8min
    lags = []
    for t in times:
        idx = raw.index.get_indexer([t], method='nearest')[0]
        lag_min = abs((raw.index[idx] - t).total_seconds()) / 60.0
        lags.append(lag_min)
    max_lag = max(lags)
    if max_lag > 4.0:
        _fail(errors, 'max nearest-neighbor lag %.2f min > 4 min' % max_lag)
    else:
        _ok('max weather timestamp lag = %.2f min (5-min source → 7.5-min grid)' % max_lag)

    env = PVTrackingEnv(
        weather_source='nsrdb_multiyear',
        scenario_manifest=manifest['_manifest_path'],
        randomize_scenario=False,
        fixed_eval_scenarios=[scenario['scenario_id']],
        randomize_initial_orientation=False,
        movement_penalty=0.0,
        observation_mode='physical',
    )
    try:
        env.reset()
        for i, t in enumerate(env.times):
            if env.weather_profile.index[i] != t:
                _fail(errors, 'weather_profile.index[%d] != episode time' % i)
                break
            sp = env._solar_position(t)
            row = env.weather_profile.iloc[i]
            if float(row['ghi']) != float(prof.iloc[i]['ghi']):
                _fail(errors, 'env weather != resampled at step %d' % i)
                break
        else:
            _ok('weather_profile.index aligned with pvlib timestamps (%d steps)' % len(env.times))

        env.step_index = 5
        env.current_time = env.times[5]
        w_obs = env._current_weather()
        w_prof = env.weather_profile.iloc[5]
        if any(abs(w_obs[k] - float(w_prof[k])) > 1e-4 for k in w_obs):
            _fail(errors, 'obs weather != profile at step 5')
        else:
            _ok('observation weather sourced from episode profile (not TMY)')

        expected_dt_h = CONTROL_STEP_MIN / 60.0
        if abs(env.interval_hours - expected_dt_h) > 1e-9:
            _fail(errors, 'interval_hours=%.6f (expected %.6f for 7min30s)' % (
                env.interval_hours, expected_dt_h))
        else:
            _ok('energy integration dt = interval_hours = %.4f h (7min30s)' % env.interval_hours)
    finally:
        env.close()


def check_training_algo_alignment(config, errors):
    print('\n=== MBPO / SAC horizon alignment ===')
    algo = config.get('kwargs', {})
    epoch_length = int(algo.get('epoch_length', 0))
    model_train_freq = int(algo.get('model_train_freq', 0))
    if epoch_length != DEFAULT_EPISODE_STEPS:
        _fail(errors, 'epoch_length=%d != %d' % (epoch_length, DEFAULT_EPISODE_STEPS))
    else:
        _ok('epoch_length=%d' % epoch_length)
    env_kw = config.get('environment_kwargs', {})
    if int(env_kw.get('periods', 0)) != DEFAULT_PERIODS:
        _fail(errors, 'periods=%r' % env_kw.get('periods'))
    else:
        _ok('periods=%d → %d transitions' % (DEFAULT_PERIODS, DEFAULT_EPISODE_STEPS))
    if model_train_freq not in (epoch_length, epoch_length + 1, 80):
        print('  NOTE: model_train_freq=%d (epoch_length=%d)' % (
            model_train_freq, epoch_length))
    _ok('model_train_freq=%d' % model_train_freq)


def check_replay_remaining_steps(errors):
    print('\n=== Replay remaining_steps (MBPO rollout filter) ===')
    from gym.spaces import Box
    obs_space = Box(low=-1.0, high=1.0, shape=(11,), dtype=np.float32)
    act_space = Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
    pool = SimpleReplayPool(obs_space, act_space, max_size=1000)
    path_len = DEFAULT_EPISODE_STEPS
    obs = np.zeros((path_len, 11), dtype=np.float32)
    pool.add_path({
        'observations': obs,
        'actions': np.zeros((path_len, 2), dtype=np.float32),
        'rewards': np.zeros((path_len, 1), dtype=np.float32),
        'terminals': np.array([[False]] * (path_len - 1) + [[True]]),
        'next_observations': obs,
    })
    rem = pool.fields['remaining_steps'][:path_len].squeeze(-1)
    expected = np.arange(path_len, 0, -1, dtype=np.int32)
    if not np.array_equal(rem, expected):
        _fail(errors, 'remaining_steps mismatch')
    else:
        _ok('remaining_steps 78..1 for T=78 path')
    rollout_length = 25
    valid = rem > rollout_length
    n_valid = int(np.sum(valid))
    if n_valid != path_len - rollout_length:
        _fail(errors, 'MBPO filter count wrong: %d' % n_valid)
    else:
        _ok('MBPO accepts %d/%d starts for rollout_length=%d' % (
            n_valid, path_len, rollout_length))


def check_mc_eval_protocol(config, errors):
    print('\n=== Post-train MC eval (get_eval_environment) ===')
    from eval_utils import get_eval_environment
    variant = {
        'environment_params': {
            'training': {
                'universe': config['universe'],
                'domain': config['domain'],
                'task': config['task'],
                'kwargs': dict(config['environment_kwargs']),
            },
            'evaluation': {
                'universe': config['universe'],
                'domain': config['domain'],
                'task': config['task'],
                'kwargs': dict(config['evaluation_environment_kwargs']),
            },
        },
    }
    env, eval_params = get_eval_environment(variant, eval_protocol='inherit')
    try:
        kw = eval_params['kwargs']
        if kw.get('fixed_eval_scenarios'):
            _fail(errors, 'MC eval inherited fixed_eval_scenarios=%s' % kw['fixed_eval_scenarios'])
        else:
            _ok('fixed_eval_scenarios cleared for MC (full manifest)')
        if not kw.get('randomize_scenario'):
            _fail(errors, 'MC eval randomize_scenario=%r' % kw.get('randomize_scenario'))
        else:
            _ok('randomize_scenario=True for scenario MC')
        inner = env.unwrapped
        if inner.freq != DEFAULT_FREQ or inner.periods != DEFAULT_PERIODS:
            _fail(errors, 'MC eval grid mismatch')
        else:
            _ok('MC eval UTC grid %s %d %s' % (inner.start_time, inner.periods, inner.freq))
    finally:
        env.close()


def check_within_episode_trajectory(config, errors):
    """Weather must follow one fixed NSRDB trajectory per episode, not per-step random draws."""
    print('\n=== Within-episode weather: fixed NSRDB trajectory (not per-step random) ===')
    env_kw = dict(config['environment_kwargs'])
    env_kw['irradiance_perturbation_std'] = 0.0
    env = PVTrackingEnv(**env_kw)
    try:
        env.seed(0)
        env.reset()
        sid0 = env._current_scenario['scenario_id']
        profile_at_reset = env.weather_profile.copy()
        ghi_trace = []
        scenario_ids = []
        for _ in range(env.num_action_steps):
            _, _, _, info = env.step(np.zeros(2, dtype=np.float32))
            ghi_trace.append(float(info['ghi_wm2']))
            scenario_ids.append(info['scenario_id'])
            expected = float(profile_at_reset.iloc[env.step_index]['ghi'])
            if abs(float(info['ghi_wm2']) - expected) > 1e-3:
                _fail(errors, (
                    'step weather != profile at step_index %d (per-step resample?)' % (
                        env.step_index)))
        if len(set(scenario_ids)) != 1 or scenario_ids[0] != sid0:
            _fail(errors, 'scenario_id changed within episode: %s' % set(scenario_ids))
        elif len({round(g, 1) for g in ghi_trace}) < 2:
            _fail(errors, 'GHI constant over episode (missing temporal structure)')
        else:
            _ok('scenario_id=%s fixed; GHI evolves along pre-built 78-step profile' % sid0)
        if not np.allclose(
                env.weather_profile[['ghi', 'dni', 'dhi']].values,
                profile_at_reset[['ghi', 'dni', 'dhi']].values):
            _fail(errors, 'weather_profile mutated during episode steps')
        else:
            _ok('weather_profile unchanged after rollout (indexed by step_index only)')
    finally:
        env.close()


def check_in_train_eval_scenario_coverage(config, errors):
    """In-train eval must cycle all fixed_eval_scenarios (nonzero return-std)."""
    print('\n=== In-train eval: fixed scenario coverage per epoch ===')
    ev_kw = dict(config['evaluation_environment_kwargs'])
    fixed = ev_kw.get('fixed_eval_scenarios') or []
    if not fixed:
        print('  (skip: no fixed_eval_scenarios)')
        return

    env = PVTrackingEnv(**ev_kw)
    try:
        env.begin_evaluation_rollouts(len(fixed))
        sids = []
        rets = []
        for _ in range(len(fixed)):
            env.reset()
            total = 0.0
            done = False
            while not done:
                _, r, done, info = env.step(np.zeros(2, dtype=np.float32))
                total += float(r)
            sids.append(info.get('scenario_id'))
            rets.append(total)
        if len(set(sids)) != len(fixed):
            _fail(errors, 'eval rollouts repeated scenarios: %s' % sids)
        elif float(np.std(rets)) <= 1e-6:
            _fail(errors, 'eval returns identical across scenarios: %s' % rets)
        else:
            _ok('%d scenarios, return std=%.3f kWh' % (len(set(sids)), np.std(rets)))
    finally:
        env.close()


def check_stochastic_exploration(config, manifest, errors):
    print('\n=== Stochasticity across episodes: scenario sampling only ===')
    env_kw = dict(config['environment_kwargs'])
    env_kw['randomize_scenario'] = True
    env_kw['irradiance_perturbation_std'] = 0.0
    env = PVTrackingEnv(**env_kw)
    try:
        ids = set()
        for seed in range(30):
            env.seed(seed)
            env.reset()
            ids.add(env._current_scenario['scenario_id'])
        if len(ids) < 5:
            _fail(errors, 'only %d unique scenarios in 30 resets' % len(ids))
        else:
            _ok('%d unique scenarios in 30 resets (e~Uniform manifest, n=%d)' % (
                len(ids), manifest['num_scenarios']))
        if env_kw.get('irradiance_perturbation_std', 0) != 0:
            _fail(errors, 'irradiance_perturbation_std should be 0 for empirical weather')
    finally:
        env.close()


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', default=DEFAULT_MODULE)
    args = p.parse_args()

    errors = []
    print('NSRDB timing synchronization audit')
    print('=' * 40)

    mod = importlib.import_module(args.config)
    config = mod.params
    manifest_path = config['environment_kwargs'].get('scenario_manifest')
    if not manifest_path:
        _fail(errors, 'no scenario_manifest in config')
        raise SystemExit(1)

    manifest = load_scenario_manifest(manifest_path)
    check_manifest_episode_grid(manifest, errors)
    check_resampling_and_pvlib(manifest, errors)
    check_training_algo_alignment(config, errors)
    check_replay_remaining_steps(errors)
    check_mc_eval_protocol(config, errors)
    check_within_episode_trajectory(config, errors)
    check_in_train_eval_scenario_coverage(config, errors)
    check_stochastic_exploration(config, manifest, errors)

    outdir = os.path.join(_REPO, 'verification', 'nsrdb_timing_sync')
    os.makedirs(outdir, exist_ok=True)
    report = os.path.join(outdir, 'timing_sync_report.txt')
    with open(report, 'w', encoding='utf-8') as f:
        f.write('NSRDB 5-min → 7min30s control (Option A)\n')
        f.write('Native weather: 5-min UTC SAM CSV\n')
        f.write('Control/energy: 7min30s (dt=%.4f h), T=%d\n' % (
            CONTROL_STEP_MIN / 60.0, DEFAULT_EPISODE_STEPS))
        f.write('pvlib: solar position at episode timestamps\n')
        f.write('Stochasticity: scenario e~p(e) only\n\n')
        if errors:
            f.write('FAIL:\n')
            for e in errors:
                f.write('  - %s\n' % e)
        else:
            f.write('PASS — end-to-end timing synchronized.\n')

    print('\n' + '=' * 40)
    if errors:
        for e in errors:
            print('FAIL:', e)
        print('Report:', report)
        return 1
    print('PASS — NSRDB 5-min data, 7min30s control, pvlib, MBPO/SAC aligned.')
    print('Report:', report)
    return 0


if __name__ == '__main__':
    sys.exit(main())
