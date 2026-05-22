#!/usr/bin/env python3
"""End-to-end check: UTC time standard, horizon alignment, plot/CSV indexing.

Run after code changes or before a paper eval:

  conda activate mbpo
  cd /home/ecer/PVRL/mbpo
  python scripts/verify_utc_uniformity.py
  python scripts/verify_utc_uniformity.py --config examples/config/pv_tracking/0.py
"""

from __future__ import print_function

import argparse
import importlib.util
import os
import sys

import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _REPO)
sys.path.insert(0, _SCRIPT_DIR)

from softlearning.environments.utils import get_environment_from_params

from mbpo.env.pv_tracking import (
    PV_TIMEZONE,
    DEFAULT_START_TIME,
    DEFAULT_PERIODS,
    DEFAULT_FREQ,
    DEFAULT_EPISODE_STEPS,
)
from mbpo.static.pv_tracking import (
    DEFAULT_TZ,
    DEFAULT_START_HOUR,
    DEFAULT_NUM_ACTIONS,
    DEFAULT_STEP_HOURS,
    StaticFns,
)
from eval_utils import (
    PV_TIMEZONE as EU_TZ,
    PV_EPISODE_START_TIME,
    PV_EPISODE_PERIODS,
    PV_EPISODE_FREQ,
    PV_EPISODE_MAX_STEPS,
    apply_pv_utc_schedule,
    apply_eval_protocol,
    build_episode_times,
    rollout_time_axis,
    rollout_xlabel,
    get_eval_environment,
)


def _load_config(path):
    path = os.path.expanduser(path)
    spec = importlib.util.spec_from_file_location('pv_config', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.params


def _fail(msg):
    print('FAIL:', msg)
    sys.exit(1)


def _ok(msg):
    print('OK:', msg)


def check_constant_alignment():
    print('\n=== Constants (env / eval_utils / static) ===')
    if str(PV_TIMEZONE) != 'UTC':
        _fail('PV_TIMEZONE=%r' % PV_TIMEZONE)
    if EU_TZ != PV_TIMEZONE:
        _fail('eval_utils PV_TIMEZONE mismatch')
    if DEFAULT_TZ != 'UTC':
        _fail('static DEFAULT_TZ=%r' % DEFAULT_TZ)
    if DEFAULT_START_TIME != PV_EPISODE_START_TIME:
        _fail('start_time mismatch env vs eval_utils')
    if DEFAULT_PERIODS != PV_EPISODE_PERIODS:
        _fail('periods mismatch')
    if DEFAULT_EPISODE_STEPS != PV_EPISODE_MAX_STEPS:
        _fail('episode steps mismatch')
    end_env = 6.0 + (DEFAULT_PERIODS - 1) * 0.25
    end_static = StaticFns.episode_end_hour()
    if abs(end_env - end_static) > 0.01:
        _fail('grid end hour env %.2f vs static %.2f' % (end_env, end_static))
    if DEFAULT_START_HOUR != 6.0 or DEFAULT_NUM_ACTIONS != 63 or DEFAULT_STEP_HOURS != 0.25:
        _fail('static timing constants unexpected')
    _ok('UTC tz; start 06:00; periods=64; steps=63; end hour %.2f' % end_static)


def check_training_env(config):
    print('\n=== Training environment ===')
    params = config
    env_params = {
        'universe': params['universe'],
        'domain': params['domain'],
        'task': params['task'],
        'kwargs': dict(params['environment_kwargs']),
    }
    env = get_environment_from_params(env_params)
    inner = env.unwrapped

    if str(inner.location.tz) != 'UTC':
        _fail('training Location.tz=%r' % inner.location.tz)
    if inner.start_time != '06:00' or inner.periods != 64 or inner.freq != '15min':
        _fail('training grid %s %d %s' % (inner.start_time, inner.periods, inner.freq))
    if inner.num_action_steps != 63:
        _fail('num_action_steps=%d' % inner.num_action_steps)

    algo = params.get('kwargs', {})
    if algo.get('epoch_length') != 63:
        _fail('epoch_length=%r (expected 63)' % algo.get('epoch_length'))

    _ok('training env tz=%s grid %s periods=%d steps=%d epoch_length=%s' % (
        inner.location.tz, inner.start_time, inner.periods,
        inner.num_action_steps, algo.get('epoch_length')))

    validate_rollout_indexing(env, label='training')
    env.close()


def check_eval_forces_utc(config):
    print('\n=== Evaluation / baselines (apply_pv_utc_schedule) ===')
    variant = {
        'environment_params': {
            'training': {
                'universe': config['universe'],
                'domain': config['domain'],
                'task': config['task'],
                'kwargs': dict(config['environment_kwargs']),
            },
        },
    }
    # Stale variant might say Denver — eval must still force UTC.
    variant['environment_params']['training']['kwargs']['tz'] = 'America/Denver'
    env, eval_params = get_eval_environment(variant, eval_protocol='inherit')
    kwargs = eval_params['kwargs']
    if kwargs.get('tz') != 'UTC':
        _fail('eval tz=%r after inherit' % kwargs.get('tz'))
    if kwargs.get('start_time') != '06:00' or kwargs.get('periods') != 64:
        _fail('eval grid not forced to UTC standard')
    inner = env.unwrapped
    if str(inner.location.tz) != 'UTC':
        _fail('eval Location.tz=%r' % inner.location.tz)

    _ok('eval forces UTC even if variant had America/Denver')
    validate_rollout_indexing(env, label='eval')
    env.close()

    try:
        apply_eval_protocol({'tz': 'UTC'}, 'local_day')
        _fail('local_day should raise')
    except ValueError:
        _ok('local_day protocol disabled')


def validate_rollout_indexing(env, label=''):
    print('\n--- Rollout indexing (%s) ---' % label)
    inner = env.unwrapped
    start_hour = 6.0
    step_h = inner.interval_hours
    n = inner.num_action_steps

    obs = env.reset()
    act = np.zeros(env.action_space.shape, dtype=np.float32)
    infos = []
    step = 0
    done = False
    while not done:
        _, r, done, info = env.step(act)
        infos.append(info)
        expected_t = start_hour + (step + 1) * step_h
        for key in ('time', 'clock_hour', 'clock_hour_utc', 'clock_hour_env_tz'):
            if key in info and abs(float(info[key]) - expected_t) > 0.02:
                _fail('%s step %d %s=%.3f expected %.3f' % (
                    label, step, key, info[key], expected_t))
        if info.get('timezone') != 'UTC':
            _fail('info timezone=%r' % info.get('timezone'))
        ts = info.get('timestamp_utc_iso', '')
        if ts and '+00:00' not in ts and 'Z' not in ts:
            _fail('timestamp_utc_iso not UTC: %s' % ts)
        if 'clock_hour_local' in info:
            _fail('clock_hour_local must not be in info (no local conversion)')
        step += 1

    if step != n:
        _fail('%s steps=%d expected %d' % (label, step, n))
    if abs(float(infos[-1]['time']) - (start_hour + n * step_h)) > 0.02:
        _fail('final clock hour wrong')

    # Plot/CSV axis: monotonic UTC hours, not relative 0..15 only
    path = {'rewards': [0.0] * len(infos), 'infos': infos}
    x = rollout_time_axis(path)
    if not np.allclose(x, [info['clock_hour_utc'] for info in infos]):
        _fail('rollout_time_axis != clock_hour_utc')
    if x[0] < 6.0 or x[-1] < 21.0:
        _fail('plot axis range unexpected %.2f..%.2f' % (x[0], x[-1]))
    if rollout_xlabel(path) != 'Clock hour UTC (post-step)':
        _fail('rollout_xlabel=%r' % rollout_xlabel(path))

    _ok('%s: %d steps; clock %.2f→%.2f UTC; plot label OK' % (
        label, step, x[0], x[-1]))


def check_pvlib_index_alignment():
    print('\n=== pvlib DatetimeIndex (same as env._build_times) ===')
    times = build_episode_times('2020-12-21', {})
    if str(times.tz) != 'UTC':
        _fail('episode times tz=%s' % times.tz)
    if len(times) != 64:
        _fail('len(times)=%d' % len(times))
    if times[0].hour != 6 or times[0].minute != 0:
        _fail('start timestamp %s' % times[0])
    if times[-1].hour != 21 or times[-1].minute != 45:
        _fail('end timestamp %s' % times[-1])
    _ok('64 timestamps 2020-12-21 06:00 → 21:45 UTC')


def check_sampler_max_path_length(config):
    print('\n=== Training sampler horizon ===')
    from examples.development.base import MAX_PATH_LENGTH_PER_DOMAIN
    if MAX_PATH_LENGTH_PER_DOMAIN.get('PVTracking') != 63:
        _fail('MAX_PATH_LENGTH_PER_DOMAIN[PVTracking]=%r' % (
            MAX_PATH_LENGTH_PER_DOMAIN.get('PVTracking'),))
    _ok('sampler max_path_length=63 (matches env.num_action_steps)')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--config', type=str, default='examples/config/pv_tracking/0.py')
    args = parser.parse_args()

    print('PV tracking UTC / horizon uniformity verification')
    config = _load_config(os.path.join(_REPO, args.config))

    check_constant_alignment()
    check_pvlib_index_alignment()
    check_sampler_max_path_length(config)
    check_training_env(config)
    check_eval_forces_utc(config)

    print('\n=== ALL CHECKS PASSED ===')
    print('Train, eval, baselines, plots, and static model horizon use the same UTC MDP.')
    print('Winter zero power before ~14h UTC is night (solar_alt<=0), not a timezone bug.')


if __name__ == '__main__':
    main()
