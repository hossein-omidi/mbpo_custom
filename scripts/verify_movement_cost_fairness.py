#!/usr/bin/env python3
"""Verify movement_penalty is consistent for train, eval, and baselines.

Checks:
  - Config train/eval kwargs share movement_penalty and discount=1
  - reward == energy_kwh - movement_cost on env steps
  - sun_tracking and fixed_no_motion baselines pay the same penalty via env.step()
"""

from __future__ import print_function

import argparse
import importlib.util
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from softlearning.environments.utils import get_environment_from_params
from eval_utils import make_baseline_rollout, get_eval_environment


def load_config(path):
    path = os.path.abspath(path)
    spec = importlib.util.spec_from_file_location('cfg', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.params


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
        'algorithm_params': {'kwargs': dict(params.get('kwargs', {}))},
    }


def check_config(params):
    errors = []
    train = params['environment_kwargs']
    ev = params.get('evaluation_environment_kwargs', {})
    algo = params.get('kwargs', {})
    pen_t = float(train.get('movement_penalty', 0.0))
    pen_e = float(ev.get('movement_penalty', 0.0))
    if pen_t != pen_e:
        errors.append('train movement_penalty %s != eval %s' % (pen_t, pen_e))
    if float(algo.get('discount', 0.99)) != 1.0:
        errors.append('expected discount=1.0, got %s' % algo.get('discount'))
    return pen_t, errors


def check_env_steps(env, label, n_steps=5):
    obs = env.reset()
    errors = []
    for _ in range(n_steps):
        action = env.action_space.sample()
        _, reward, done, info = env.step(action)
        expected = float(info['energy_kwh']) - float(info['movement_cost'])
        if abs(reward - expected) > 1e-6:
            errors.append('%s: reward %.8f != energy-movement %.8f' % (
                label, reward, expected))
        if abs(reward - (info['reward_energy'] - info['reward_movement'])) > 1e-6:
            errors.append('%s: info reward fields inconsistent' % label)
        if done:
            break
    return errors


def check_baselines(variant, penalty, path_length=10):
    errors = []
    for name in ('sun_tracking', 'fixed_no_motion'):
        env, _ = get_eval_environment(
            variant, fixed_eval_dates='2020-06-21', eval_weather_source='clearsky')
        env.seed(0)
        path = make_baseline_rollout(env, name, path_length=path_length, seed=0)
        inner = env.unwrapped
        if float(inner.movement_penalty) != penalty:
            errors.append('%s env movement_penalty=%s expected %s' % (
                name, inner.movement_penalty, penalty))
        total_reward = float(path['rewards'].sum())
        total_energy = sum(info['energy_kwh'] for info in path['infos'])
        total_move = sum(info['movement_cost'] for info in path['infos'])
        if abs(total_reward - (total_energy - total_move)) > 1e-5:
            errors.append('%s: sum(reward) != sum(energy)-sum(movement)' % name)
        env.close()
    return errors


def _eval_env_kwargs(summary):
    """Return env kwargs from evaluation_summary.json.

    evaluate_agent writes flat kwargs under eval_config; older summaries may nest
    them under eval_config.kwargs.
    """
    eval_cfg = summary.get('eval_config') or {}
    nested = eval_cfg.get('kwargs')
    if isinstance(nested, dict) and nested:
        return nested
    return eval_cfg if isinstance(eval_cfg, dict) else {}


def check_eval_dir(eval_dir):
    import json
    import glob

    summary_path = os.path.join(eval_dir, 'evaluation_summary.json')
    if not os.path.isfile(summary_path):
        return ['missing %s' % summary_path]
    summary = json.load(open(summary_path))
    kw = _eval_env_kwargs(summary)
    errors = []
    if 'movement_penalty' not in kw:
        errors.append(
            'eval_config missing movement_penalty (checked eval_config and eval_config.kwargs)')
        return errors

    penalty = float(kw['movement_penalty'])
    # Spot-check rollout CSVs: reward == energy - movement_cost
    for pattern in (
        os.path.join(eval_dir, 'rollouts', '*.csv'),
        os.path.join(eval_dir, 'baseline_rollouts', 'sun_tracking', '*.csv'),
        os.path.join(eval_dir, 'baseline_rollouts', 'fixed_no_motion', '*.csv'),
    ):
        csv_paths = glob.glob(pattern)
        if not csv_paths:
            continue
        path = csv_paths[0]
        import csv
        with open(path, newline='', encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
        if not rows:
            continue
        total_reward = sum(float(r.get('reward', 0) or 0) for r in rows)
        total_energy = sum(float(r.get('energy_kwh', 0) or 0) for r in rows)
        total_move = sum(float(r.get('movement_cost', 0) or 0) for r in rows)
        if abs(total_reward - (total_energy - total_move)) > 1e-4:
            errors.append('%s: sum(reward) != sum(energy)-sum(movement)' % (
                os.path.basename(path)))
        for r in rows[:3]:
            mc = float(r.get('movement_cost', 0) or 0)
            if penalty > 0 and mc > 0:
                break
        else:
            if penalty > 0 and 'sun_tracking' in path and total_move <= 0:
                errors.append('sun_tracking CSV has zero movement_cost with penalty=%s' % penalty)
    return errors


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config-path', default='examples/config/pv_tracking/stage0_single_day_mbpo_paper.py')
    p.add_argument('--eval-dir', default=None, help='Optional post-eval check')
    args = p.parse_args()

    params = load_config(os.path.join(_REPO_ROOT, args.config_path))
    penalty, errors = check_config(params)
    print('movement_penalty=%s  discount=%s' % (
        penalty, params['kwargs'].get('discount')))

    variant = build_variant(params)
    train_env = get_environment_from_params(variant['environment_params']['training'])
    errors += check_env_steps(train_env, 'train')
    train_env.close()

    eval_env, _ = get_eval_environment(
        variant, fixed_eval_dates='2020-06-21', eval_weather_source='clearsky')
    errors += check_env_steps(eval_env, 'eval')
    eval_env.close()

    errors += check_baselines(variant, penalty)

    if args.eval_dir:
        errors += check_eval_dir(os.path.join(_REPO_ROOT, args.eval_dir))

    if errors:
        print('FAIL')
        for e in errors:
            print(' ', e)
        return 1
    print('PASS — movement cost fair for train, eval, and baselines')
    return 0


if __name__ == '__main__':
    sys.exit(main())
