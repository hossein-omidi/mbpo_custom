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
    DEFAULT_START_HOUR,
    DEFAULT_END_HOUR,
    episode_clock_hour,
)
from mbpo.static.pv_tracking import (
    DEFAULT_TZ,
    DEFAULT_START_HOUR as STATIC_START_HOUR,
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
    _grid_end_clock_label,
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
    end_env = episode_clock_hour(DEFAULT_START_TIME) + (DEFAULT_PERIODS - 1) * 0.25
    end_static = StaticFns.episode_end_hour()
    if abs(end_env - end_static) > 0.01:
        _fail('grid end hour env %.2f vs static %.2f' % (end_env, end_static))
    if abs(STATIC_START_HOUR - DEFAULT_START_HOUR) > 0.01:
        _fail('static start hour %.2f vs env %.2f' % (STATIC_START_HOUR, DEFAULT_START_HOUR))
    if DEFAULT_NUM_ACTIONS != DEFAULT_EPISODE_STEPS or DEFAULT_STEP_HOURS != 0.25:
        _fail('static timing constants unexpected')
    _ok('UTC tz; start %s; periods=%d; steps=%d; end %s' % (
        DEFAULT_START_TIME, DEFAULT_PERIODS, DEFAULT_EPISODE_STEPS,
        _grid_end_clock_label()))


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
    if (inner.start_time != DEFAULT_START_TIME or inner.periods != DEFAULT_PERIODS
            or inner.freq != DEFAULT_FREQ):
        _fail('training grid %s %d %s' % (inner.start_time, inner.periods, inner.freq))
    if inner.num_action_steps != DEFAULT_EPISODE_STEPS:
        _fail('num_action_steps=%d' % inner.num_action_steps)

    algo = params.get('kwargs', {})
    if algo.get('epoch_length') != DEFAULT_EPISODE_STEPS:
        _fail('epoch_length=%r (expected %d)' % (
            algo.get('epoch_length'), DEFAULT_EPISODE_STEPS))

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
    # Stale variant might say Denver — eval must still force UTC daylight grid.
    variant['environment_params']['training']['kwargs']['tz'] = 'America/Denver'
    variant['environment_params']['training']['kwargs']['start_time'] = '06:00'
    variant['environment_params']['training']['kwargs']['periods'] = 64
    env, eval_params = get_eval_environment(variant, eval_protocol='inherit')
    kwargs = eval_params['kwargs']
    if kwargs.get('tz') != 'UTC':
        _fail('eval tz=%r after inherit' % kwargs.get('tz'))
    if (kwargs.get('start_time') != DEFAULT_START_TIME
            or kwargs.get('periods') != DEFAULT_PERIODS):
        _fail('eval grid not forced to UTC daylight standard')
    inner = env.unwrapped
    if str(inner.location.tz) != 'UTC':
        _fail('eval Location.tz=%r' % inner.location.tz)

    _ok('eval forces UTC daylight grid even if variant had stale timing')
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
    start_hour = episode_clock_hour(inner.start_time)
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

    path = {'rewards': [0.0] * len(infos), 'infos': infos}
    x = rollout_time_axis(path)
    if not np.allclose(x, [info['clock_hour_utc'] for info in infos]):
        _fail('rollout_time_axis != clock_hour_utc')
    if x[0] < start_hour or x[-1] > DEFAULT_END_HOUR + 0.5:
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
    if len(times) != DEFAULT_PERIODS:
        _fail('len(times)=%d' % len(times))
    sh, sm = map(int, DEFAULT_START_TIME.split(':'))
    if times[0].hour != sh or times[0].minute != sm:
        _fail('start timestamp %s' % times[0])
    end_parts = _grid_end_clock_label().split(':')
    if times[-1].hour != int(end_parts[0]) or times[-1].minute != int(end_parts[1]):
        _fail('end timestamp %s (expected %s)' % (times[-1], _grid_end_clock_label()))
    _ok('%d timestamps 2020-12-21 %s → %s UTC' % (
        DEFAULT_PERIODS, DEFAULT_START_TIME, _grid_end_clock_label()))


def check_sampler_max_path_length(config):
    print('\n=== Training sampler horizon ===')
    from examples.development.base import MAX_PATH_LENGTH_PER_DOMAIN
    expected = DEFAULT_EPISODE_STEPS
    if MAX_PATH_LENGTH_PER_DOMAIN.get('PVTracking') != expected:
        _fail('MAX_PATH_LENGTH_PER_DOMAIN[PVTracking]=%r' % (
            MAX_PATH_LENGTH_PER_DOMAIN.get('PVTracking'),))
    _ok('sampler max_path_length=%d (matches env.num_action_steps)' % expected)


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
    print('Train, eval, baselines, plots, and static model horizon use the same UTC daylight MDP.')
    print('Episode grid: %s–%s UTC (%d steps).' % (
        DEFAULT_START_TIME, _grid_end_clock_label(), DEFAULT_EPISODE_STEPS))


if __name__ == '__main__':
    main()
