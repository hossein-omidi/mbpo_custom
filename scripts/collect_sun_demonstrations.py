#!/usr/bin/env python3
"""Collect sun_tracking demonstration trajectories for imitation / analysis.

Does not train a policy — writes a compressed .npz replay for optional BC pretrain.

Example (Stage 0 single day):

  python scripts/collect_sun_demonstrations.py \\
    --config examples.config.pv_tracking.stage3_multiyear_nsrdb_scenario \\
    --out demonstration/pv_stage0_sun.npz \\
    --num-episodes 200

See docs/TRAINING_PROTOCOL.md §6.
"""

from __future__ import print_function

import argparse
import importlib
import os
import sys

import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from softlearning.environments.utils import get_environment_from_params
from eval_utils import make_baseline_rollout


def load_config_module(config_path):
    mod = importlib.import_module(config_path)
    return getattr(mod, 'params')


def build_training_env(params):
    variant = {
        'environment_params': {
            'training': {
                'domain': params['domain'],
                'task': params['task'],
                'universe': params['universe'],
                'kwargs': dict(params['environment_kwargs']),
            }
        }
    }
    return get_environment_from_params(variant['environment_params']['training'])


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', default='examples.config.pv_tracking.stage3_multiyear_nsrdb_scenario',
                   help='Config module (environment_kwargs used)')
    p.add_argument('--out', required=True, help='Output .npz path')
    p.add_argument('--num-episodes', type=int, default=100)
    p.add_argument('--baseline', default='sun_tracking',
                   choices=['sun_tracking', 'fixed_no_motion'])
    p.add_argument('--seed', type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    params = load_config_module(args.config)
    env = build_training_env(params)
    env.seed(args.seed)

    inner = env.unwrapped
    path_length = inner.num_action_steps

    obs_list = []
    act_list = []
    rew_list = []
    term_list = []
    next_obs_list = []
    dates = []

    print('Collecting %d episodes (%s), path_length=%d' % (
        args.num_episodes, args.baseline, path_length))
    print('  weather=%s  dates=%s..%s' % (
        params['environment_kwargs'].get('weather_source'),
        params['environment_kwargs'].get('start_date'),
        params['environment_kwargs'].get('end_date'),
    ))

    for ep in range(args.num_episodes):
        env.seed(args.seed + ep)
        path = make_baseline_rollout(
            env, args.baseline, path_length=path_length, seed=args.seed + ep)
        obs_list.append(path['observations'])
        act_list.append(path['actions'])
        rew_list.append(path['rewards'])
        term_list.append(path['terminals'])
        next_obs_list.append(path['next_observations'])
        infos = path.get('infos', [])
        if infos:
            dates.append(infos[0].get('date', ''))

    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    np.savez_compressed(
        args.out,
        observations=np.concatenate(obs_list, axis=0),
        actions=np.concatenate(act_list, axis=0),
        rewards=np.concatenate(rew_list, axis=0),
        terminals=np.concatenate(term_list, axis=0),
        next_observations=np.concatenate(next_obs_list, axis=0),
        episode_dates=np.array(dates, dtype=object),
        config_version=params.get('config_version', ''),
        baseline=args.baseline,
        num_episodes=args.num_episodes,
    )
    print('Wrote %s (%d transitions)' % (
        args.out, sum(len(o) for o in obs_list)))
    print('Next: optional BC pretrain (not wired to mbpo run_local); see docs/TRAINING_PROTOCOL.md')
    return 0


if __name__ == '__main__':
    sys.exit(main())
