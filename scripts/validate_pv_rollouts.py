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
from mbpo.static.pv_tracking import StaticFns


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


def validate_static_fns(env):
    """Check horizon termination and observation postprocessing."""
    print('\nStaticFns validation:')
    print('  episode_end_hour:', StaticFns.episode_end_hour())
    print('  episode_end_angle:', StaticFns.episode_end_angle())

    inner = env.unwrapped
    periods = getattr(inner, 'periods', 64)
    max_steps = periods + 1

    obs = env.reset()
    act = np.zeros(env.action_space.shape, dtype=np.float32)
    terminal_hits = 0
    last_obs = obs
    done = False
    for step in range(max_steps):
        next_obs, reward, done, info = env.step(act)
        batch_obs = np.asarray(last_obs, dtype=np.float32)[None]
        batch_next = np.asarray(next_obs, dtype=np.float32)[None]
        batch_act = act[None]
        term = StaticFns.termination_fn(batch_obs, batch_act, batch_next)
        decoded_time = float(StaticFns.time_of_day_from_obs(batch_next))
        env_time = float(info.get('time', np.nan))
        if abs(decoded_time - env_time) > 0.05:
            raise AssertionError(
                'time_of_day_from_obs {:.4f} != env info time {:.4f} at step {}'.format(
                    decoded_time, env_time, step))
        if term.any():
            terminal_hits += 1
            print(
                '  StaticFns termination at env step {} (done={}, '
                'env_time={:.2f}, decoded_time={:.2f})'.format(
                    step, done, env_time, decoded_time))
        last_obs = next_obs
        if done:
            break

    assert done, 'Env should end episode within {} steps'.format(max_steps)
    assert terminal_hits >= 1, (
        'Expected horizon termination on final transition; '
        'last decoded_time={:.4f}, end_hour={:.4f}'.format(
            decoded_time, StaticFns.episode_end_hour()))

    low = env.observation_space.low
    high = env.observation_space.high
    corrupted = np.random.randn(*last_obs.shape).astype(np.float32) * 5.0
    fixed = StaticFns.postprocess_next_obs(corrupted, low, high)
    for start, end in StaticFns.PV_CYCLIC_SLICES:
        block = fixed[start:end]
        norm = np.linalg.norm(block)
        assert 0.99 <= norm <= 1.01, (start, end, norm)
    print('  postprocess_next_obs: cyclic norms OK')
    print('  StaticFns validation passed.')


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
    validate_static_fns(env)
    validate_env(env, num_steps=args.env_steps)


if __name__ == '__main__':
    main()
