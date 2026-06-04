#!/usr/bin/env python3
"""Pre-train verification for Stage 3 full-year annual-scenario RL."""

from __future__ import print_function

import argparse
import importlib
import os
import subprocess
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np

from softlearning.environments.utils import get_environment_from_params
from examples.config.pv_tracking.verified_dates import (
    STAGE3_STRESS_TEST_DATES,
    STAGE3_VALIDATION_DATES,
    dates_by_season,
    training_day_count,
)
from mbpo.env.pv_tracking import PVTrackingEnv
from eval_utils import make_baseline_rollout, season_from_day_of_year


DEFAULT_CONFIG = 'examples.config.pv_tracking.stage3_fullyear_stable_mbpo'
DEFAULT_CONFIG_PATH = 'examples/config/pv_tracking/stage3_fullyear_stable_mbpo.py'


def load_params(config_module):
    mod = importlib.import_module(config_module)
    return mod.params, mod


def build_variant(params):
    return {
        'environment_params': {
            'training': {
                'domain': params['domain'],
                'task': params['task'],
                'universe': params['universe'],
                'kwargs': dict(params['environment_kwargs']),
            },
            'evaluation': {
                'domain': params['domain'],
                'task': params['task'],
                'universe': params['universe'],
                'kwargs': dict(params.get('evaluation_environment_kwargs', {})),
            },
        },
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default=DEFAULT_CONFIG)
    p.add_argument('--config-path', default=DEFAULT_CONFIG_PATH)
    p.add_argument('--outdir', default='verification/stage3_fullyear')
    args = p.parse_args()

    os.makedirs(os.path.join(_REPO_ROOT, args.outdir), exist_ok=True)
    errors = []
    lines = ['Stage 3 RL preflight verification', '=' * 36, '']

    for cmd in (
        [sys.executable, 'scripts/verify_training_config.py', '--file-only', '--config', args.config],
        [sys.executable, 'scripts/validate_pv_rollouts.py', '--config-path', args.config_path],
        [sys.executable, 'scripts/verify_movement_cost_fairness.py', '--config-path', args.config_path],
    ):
        print('[verify_stage3]', ' '.join(cmd))
        if subprocess.run(cmd, cwd=_REPO_ROOT).returncode != 0:
            errors.append('failed: %s' % cmd[1])

    params, mod = load_params(args.config)
    train = params['environment_kwargs']
    ev = params.get('evaluation_environment_kwargs', {})

    lines.append('CONFIG: %s' % getattr(mod, 'CONFIG_VERSION', ''))
    lines.append('Training days available: %d (no excluded_dates)' % training_day_count(
        train.get('start_date'), train.get('end_date'), train.get('excluded_dates')))
    if train.get('excluded_dates'):
        errors.append('RL protocol: training must not use excluded_dates (got %s)' % (
            train.get('excluded_dates')))
    if ev.get('fixed_eval_dates'):
        fixed = list(ev.get('fixed_eval_dates'))
        if fixed != list(STAGE3_VALIDATION_DATES):
            errors.append(
                'in-train eval fixed_eval_dates must be STAGE3_VALIDATION_DATES '
                '(got %s)' % fixed)
        else:
            lines.append(
                'In-train eval: STAGE3_VALIDATION_DATES (%d seasonally spaced dates)' % (
                    len(fixed)))
        if ev.get('randomize_day'):
            errors.append(
                'in-train eval must set randomize_day=False when using fixed_eval_dates')
    if not train.get('randomize_day'):
        errors.append('training randomize_day must be True for annual RL')
    if train.get('weather_source') != 'historical':
        errors.append('Stage 3 expects weather_source=historical')

    perturb = float(train.get('irradiance_perturbation_std', 0.0))
    lines.append('irradiance_perturbation_std=%s (0=annual scenario, deterministic weather per day)' % (
        perturb))
    if perturb > 0.0:
        lines.append('  note: optional bounded augmentation enabled (not main paper default)')

    variant = build_variant(params)
    env = get_environment_from_params(variant['environment_params']['training'])
    inner = env.unwrapped
    seen = set()
    for seed in range(300):
        env.seed(seed)
        env.reset()
        seen.add(str(env.unwrapped.current_date.date()))
    lines.append('300 training resets → %d unique days (expect >> 50)' % len(seen))
    if len(seen) < 50:
        errors.append('training day sampling too narrow: %d unique days' % len(seen))

    # Annual scenario: different days → different weather trajectories (perturbation off).
    if perturb <= 0.0:
        e_summer = _episode_energy(PVTrackingEnv(
            start_date='2020-06-21', end_date='2020-06-21', randomize_day=False,
            randomize_initial_orientation=False, weather_source='historical',
            irradiance_perturbation_std=0.0, movement_penalty=0.0,
            observation_mode='physical'), 1)
        e_winter = _episode_energy(PVTrackingEnv(
            start_date='2020-12-07', end_date='2020-12-07', randomize_day=False,
            randomize_initial_orientation=False, weather_source='historical',
            irradiance_perturbation_std=0.0, movement_penalty=0.0,
            observation_mode='physical'), 1)
        lines.append('Annual scenario energies: summer=%.4f winter=%.4f kWh' % (
            e_summer, e_winter))
        if abs(e_summer - e_winter) < 1e-6:
            errors.append('summer and winter days should differ without augmentation')
        e_same = _episode_energy(PVTrackingEnv(
            start_date='2020-06-21', end_date='2020-06-21', randomize_day=False,
            randomize_initial_orientation=False, weather_source='historical',
            irradiance_perturbation_std=0.0, movement_penalty=0.0,
            observation_mode='physical'), 42)
        e_same2 = _episode_energy(PVTrackingEnv(
            start_date='2020-06-21', end_date='2020-06-21', randomize_day=False,
            randomize_initial_orientation=False, weather_source='historical',
            irradiance_perturbation_std=0.0, movement_penalty=0.0,
            observation_mode='physical'), 42)
        if abs(e_same - e_same2) > 1e-6:
            errors.append('same day+seed should be deterministic when perturbation=0')

    # Optional augmentation: same day, different seeds may differ when perturb > 0.
    if perturb > 0:
        d = '2020-06-21'
        e1 = _episode_energy(PVTrackingEnv(
            start_date=d, end_date=d, randomize_day=False,
            randomize_initial_orientation=False,
            weather_source='historical',
            irradiance_perturbation_std=perturb,
            movement_penalty=0.0, observation_mode='physical'), 111)
        e2 = _episode_energy(PVTrackingEnv(
            start_date=d, end_date=d, randomize_day=False,
            randomize_initial_orientation=False,
            weather_source='historical',
            irradiance_perturbation_std=perturb,
            movement_penalty=0.0, observation_mode='physical'), 222)
        lines.append('Same day %s energy seed111=%.4f seed222=%.4f kWh' % (d, e1, e2))
        if abs(e1 - e2) < 1e-6:
            errors.append('perturbation did not change same-day energy across seeds')

    eval_env = get_environment_from_params(variant['environment_params']['evaluation'])
    path = make_baseline_rollout(
        eval_env, 'sun_tracking', path_length=min(5, inner.num_action_steps), seed=99999)
    for info in path['infos'][:2]:
        poa = float(info['poa_global'])
        power = float(info['power'])
        expected = max(poa, 0.0) * inner.area * inner.efficiency
        if abs(expected - power) > 0.05:
            errors.append('baseline power != pvlib path')
    eval_env.close()

    eval_env = get_environment_from_params(variant['environment_params']['evaluation'])
    fixed_path = make_baseline_rollout(
        eval_env, 'fixed_no_motion', path_length=min(10, inner.num_action_steps), seed=42)
    fixed_move = sum(float(i['movement_cost']) for i in fixed_path['infos'])
    if fixed_move > 1e-9:
        errors.append('fixed_no_motion must have zero movement_cost (got %.6f)' % fixed_move)
    else:
        lines.append('fixed_no_motion: movement_cost=0 (action=0, frozen pose)')
    eval_env.close()
    env.close()

    lines.append('')
    lines.append('Stress-test dates (optional, not train exclusions):')
    for season, ds in dates_by_season(STAGE3_STRESS_TEST_DATES).items():
        lines.append('  %s: %s' % (season, ', '.join(ds)))

    report = os.path.join(_REPO_ROOT, args.outdir, 'preflight_report.txt')
    with open(report, 'w') as f:
        f.write('\n'.join(lines))
        f.write('\n\n')
        if errors:
            f.write('FAIL:\n')
            for e in errors:
                f.write('  - %s\n' % e)
        else:
            f.write('PASS — RL annual protocol ready.\n')

    print('\n'.join(lines))
    if errors:
        print('FAIL')
        for e in errors:
            print(' ', e)
        return 1
    print('PASS —', report)
    return 0


def _episode_energy(env, seed):
    env.seed(seed)
    env.reset()
    total = 0.0
    done = False
    while not done:
        _, _, done, info = env.step(np.zeros(2, dtype=np.float32))
        total += float(info['energy_kwh'])
    env.close()
    return total


if __name__ == '__main__':
    sys.exit(main())
