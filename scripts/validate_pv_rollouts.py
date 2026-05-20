#!/usr/bin/env python3
"""Validate PV rollout scheduling and PV env behavior.

This script checks that the configured MBPO rollout length increases as expected
and that the PVTracking environment produces valid observations, rewards, and
info values for random actions.
"""

import argparse
import importlib.util
import os
import numpy as np

from softlearning.environments.utils import get_environment_from_params


def get_config(config_path):
    config_path = os.path.expanduser(config_path)
    spec = importlib.util.spec_from_file_location('pv_tracking_config', config_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, 'params')


def print_rollout_schedule(rollout_schedule, epochs=120):
    min_epoch, max_epoch, min_length, max_length = rollout_schedule
    print('Rollout schedule:', rollout_schedule)
    print('Epoch | Rollout length')
    for epoch in range(0, epochs + 1):
        if epoch <= min_epoch:
            y = min_length
        else:
            dx = (epoch - min_epoch) / max(float(max_epoch - min_epoch), 1.0)
            dx = min(dx, 1.0)
            y = dx * (max_length - min_length) + min_length
        length = int(np.ceil(y))
        length = max(min_length, min(length, max_length))
        if epoch % 10 == 0 or epoch in (0, 1, 2, 5, 10, 20, 50, 100):
            print(f'{epoch:3d}   | {length}')


def validate_env(env, num_steps=5):
    obs = env.reset()
    print('\nEnvironment validation:')
    print('Observation shape:', obs.shape)
    print('Observation sample:', obs)

    for step in range(num_steps):
        action = env.action_space.sample()
        next_obs, reward, done, info = env.step(action)
        print(f'  step={step} action={action.tolist()} reward={reward:.5f} done={done}')
        print('    obs min/max:', np.min(next_obs), np.max(next_obs))
        print('    info keys:', sorted(info.keys()))
        if done:
            print('    Episode ended early at step', step)
            break


def main():
    parser = argparse.ArgumentParser(
        description='Validate PV MBPO rollout schedule and PV env behavior.')
    parser.add_argument(
        '--config-path',
        type=str,
        default='examples/config/pv_tracking/0.py',
        help='Path to the config file, e.g. examples/config/pv_tracking/0.py.')
    parser.add_argument(
        '--epochs',
        type=int,
        default=120,
        help='Number of epochs to print rollout lengths for.')
    parser.add_argument(
        '--env-steps',
        type=int,
        default=5,
        help='Number of random env steps to execute.')
    args = parser.parse_args()

    params = get_config(args.config_path)
    rollout_schedule = params['kwargs'].get('rollout_schedule', [0, 100, 1, 1])
    real_ratio = params['kwargs'].get('real_ratio', 0.1)
    rollout_batch_size = params['kwargs'].get('rollout_batch_size', 1000)

    print('Config path:', args.config_path)
    print('Real/model batch ratio:', real_ratio)
    print('Rollout batch size:', rollout_batch_size)

    print_rollout_schedule(rollout_schedule, epochs=args.epochs)

    variant = {
        'environment_params': {
            'training': {
                'domain': 'PVTracking',
                'task': 'v0',
                'universe': 'gym',
                'kwargs': params['environment_kwargs'],
            }
        }
    }
    env = get_environment_from_params(variant['environment_params']['training'])
    validate_env(env, num_steps=args.env_steps)


if __name__ == '__main__':
    main()
