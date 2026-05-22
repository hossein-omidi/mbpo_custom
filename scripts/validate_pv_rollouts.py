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
from mbpo.static.pv_tracking import (
    StaticFns,
    LEGACY_OBS_DIM,
    PHYSICAL_OBS_DIM,
    cyclic_slices_for_obs,
    observation_mode_from_obs,
)


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


def _parse_start_hour(start_time):
    parts = str(start_time).split(':')
    return float(parts[0]) + float(parts[1]) / 60.0


def validate_episode_timing(env):
    """Validate wall-clock progression via env info (not observation decoding)."""
    inner = env.unwrapped
    start_hour = _parse_start_hour(inner.start_time)
    step_hours = inner.interval_hours
    num_actions = inner.num_action_steps
    end_hour = start_hour + num_actions * step_hours

    print('\nEpisode timing validation (env info):')
    print('  start_time:', inner.start_time, '({:.2f} h)'.format(start_hour))
    print('  interval_hours:', step_hours)
    print('  num_action_steps:', num_actions)
    print('  expected end hour: {:.2f}'.format(end_hour))

    obs = env.reset()
    act = np.zeros(env.action_space.shape, dtype=np.float32)
    times = []
    step = 0
    done = False
    while not done:
        next_obs, reward, done, info = env.step(act)
        env_time = float(info['time'])
        times.append(env_time)
        expected_time = start_hour + (step + 1) * step_hours
        if abs(env_time - expected_time) > 0.02:
            raise AssertionError(
                'info time {:.4f} != expected {:.4f} at step {}'.format(
                    env_time, expected_time, step))
        step += 1
        obs = next_obs

    assert step == num_actions, (
        'Episode should last {} actions, got {}'.format(num_actions, step))
    assert abs(times[0] - (start_hour + step_hours)) < 0.02
    assert abs(times[-1] - end_hour) < 0.02, (
        'Final info time {:.4f} != expected end {:.4f}'.format(times[-1], end_hour))
    assert times == sorted(times), 'info time should be non-decreasing'
    print('  steps executed:', step)
    print('  first info time: {:.2f} h'.format(times[0]))
    print('  final info time: {:.2f} h'.format(times[-1]))
    print('  Episode timing validation passed.')


def validate_reward_consistency(env, num_steps=10):
    """Reward = energy_kwh - movement_cost (independent of observation mode)."""
    print('\nReward consistency validation:')
    env.reset()
    for step in range(num_steps):
        action = env.action_space.sample()
        _, reward, done, info = env.step(action)
        expected = float(info['reward_energy']) - float(info['reward_movement'])
        if abs(reward - expected) > 1e-6:
            raise AssertionError(
                'reward {:.8f} != energy {:.8f} - movement {:.8f} at step {}'.format(
                    reward, info['reward_energy'], info['reward_movement'], step))
        if abs(reward - (info['energy_kwh'] - info['movement_cost'])) > 1e-6:
            raise AssertionError('reward does not match info energy/movement fields')
    print('  reward == energy_kwh - movement_cost for {} steps: OK'.format(num_steps))


def validate_observation_bounds(env, num_steps=5):
    """Observations stay within observation_space; irradiance norms may clip at 1."""
    print('\nObservation bounds validation:')
    low = env.observation_space.low
    high = env.observation_space.high
    obs = env.reset()
    _check_obs(obs, low, high, 'reset')
    for step in range(num_steps):
        action = env.action_space.sample()
        obs, _, done, info = env.step(action)
        _check_obs(obs, low, high, 'step {}'.format(step))
        if obs.shape[0] >= 6:
            raw_ghi = float(info.get('ghi_wm2', np.nan))
            if np.isfinite(raw_ghi) and raw_ghi > 2000.0:
                assert obs[5] <= 1.0 + 1e-5, (
                    'GHI norm should clip at 1 when raw ghi > 2000 W/m^2')
        if done:
            break
    print('  observation_space bounds satisfied.')


def _check_obs(obs, low, high, label):
    obs = np.asarray(obs, dtype=np.float32)
    if np.any(obs < low - 1e-5) or np.any(obs > high + 1e-5):
        raise AssertionError(
            '{} obs out of bounds: min={} max={} (allowed [{}, {}])'.format(
                label, obs.min(), obs.max(), low.min(), high.max()))


def validate_static_fns_legacy(env):
    """Legacy 15-D: horizon via time sin/cos in observation."""
    print('\nStaticFns validation (legacy 15-D):')
    print('  episode_end_hour:', StaticFns.episode_end_hour())

    inner = env.unwrapped
    max_steps = inner.periods + 1
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
        env_time = float(info.get('time', np.nan))
        decoded_time = float(StaticFns.time_of_day_from_obs(batch_next))
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
    assert terminal_hits >= 1, 'Expected horizon termination on final transition'
    _validate_static_postprocess(env, last_obs)
    print('  StaticFns legacy validation passed.')


def validate_static_fns_physical(env):
    """Physical 11-D: env horizon via done; model horizon via rollout_length only."""
    print('\nStaticFns validation (physical 11-D):')
    print('  episode_end_hour:', StaticFns.episode_end_hour())
    print('  model-rollout horizon: rollout_length cap (StaticFns terminates on non-finite only)')

    inner = env.unwrapped
    max_steps = inner.periods + 1
    obs = env.reset()
    act = np.zeros(env.action_space.shape, dtype=np.float32)
    spurious_terminals = 0
    last_obs = obs
    done = False
    for step in range(max_steps):
        next_obs, reward, done, info = env.step(act)
        batch_obs = np.asarray(last_obs, dtype=np.float32)[None]
        batch_next = np.asarray(next_obs, dtype=np.float32)[None]
        batch_act = act[None]
        term = StaticFns.termination_fn(batch_obs, batch_act, batch_next)
        env_time = float(info.get('time', np.nan))

        if term.any() and not done:
            spurious_terminals += 1
            raise AssertionError(
                'Physical StaticFns must not terminate before env done (step {}, '
                'env_time={:.2f})'.format(step, env_time))
        last_obs = next_obs
        if done:
            break

    assert done, 'Env should end episode within {} steps'.format(max_steps)
    assert spurious_terminals == 0, 'No spurious model-rollout terminals before env done'
    _validate_static_postprocess(env, last_obs)
    validate_model_rollout_depth_cap(env)
    print('  StaticFns physical validation passed.')


def validate_model_rollout_depth_cap(env):
    """Physical StaticFns must not early-stop; MBPO rollout_length caps depth."""
    obs = env.reset()
    act = np.zeros(env.action_space.shape, dtype=np.float32)
    rollout_length = 3
    for step in range(rollout_length):
        next_obs, _, done, _ = env.step(act)
        term = StaticFns.termination_fn(
            obs[None].astype(np.float32),
            act[None].astype(np.float32),
            next_obs[None].astype(np.float32),
        )
        if term.any() and not done:
            raise AssertionError(
                'StaticFns terminated at rollout step {} before env done'.format(step))
        obs = next_obs
    print('  model rollout depth: StaticFns allows {} imagined steps without solar stop'.format(
        rollout_length))


def _validate_static_postprocess(env, last_obs):
    low = env.observation_space.low
    high = env.observation_space.high
    corrupted = np.random.randn(*last_obs.shape).astype(np.float32) * 5.0
    fixed = StaticFns.postprocess_next_obs(corrupted, low, high)
    for start, end in cyclic_slices_for_obs(last_obs):
        block = fixed[start:end]
        norm = np.linalg.norm(block)
        assert 0.99 <= norm <= 1.01, (start, end, norm)
    print('  postprocess_next_obs: cyclic norms OK')


def validate_static_fns(env):
    obs = env.reset()
    mode = observation_mode_from_obs(obs)
    if mode == 'physical':
        validate_static_fns_physical(env)
    else:
        validate_static_fns_legacy(env)


def validate_env(env, num_steps=5):
    obs = env.reset()
    print('\nEnvironment validation:')
    print('Observation shape:', obs.shape)
    print('Feature names:', getattr(env.unwrapped, 'obs_feature_names', []))

    for step in range(num_steps):
        action = env.action_space.sample()
        next_obs, reward, done, info = env.step(action)
        print('  step={} reward={:.5f} done={} time={:.2f}'.format(
            step, reward, done, info.get('time', float('nan'))))
        print('    obs min/max:', np.min(next_obs), np.max(next_obs))
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
    inner = env.unwrapped
    obs = env.reset()
    env_kwargs = params['environment_kwargs']
    obs_mode = getattr(inner, 'observation_mode', 'legacy')
    print('\nTraining environment (from config environment_kwargs):')
    print('  observation_mode:', env_kwargs.get('observation_mode', 'legacy (default)'))
    print('  active observation_mode:', obs_mode)
    print('  observation_space.shape:', env.observation_space.shape)
    print('  reset obs shape:', np.asarray(obs).shape)

    if obs_mode == 'physical':
        assert obs.shape == (PHYSICAL_OBS_DIM,), (
            'physical mode expects {}-D obs, got {}'.format(PHYSICAL_OBS_DIM, obs.shape))
        assert observation_mode_from_obs(obs) == 'physical'
    else:
        assert obs.shape == (LEGACY_OBS_DIM,), (
            'legacy mode expects {}-D obs, got {}'.format(LEGACY_OBS_DIM, obs.shape))
        assert observation_mode_from_obs(obs) == 'legacy'

    validate_episode_timing(env)
    validate_reward_consistency(env, num_steps=args.env_steps)
    validate_observation_bounds(env, num_steps=args.env_steps)
    validate_static_fns(env)
    validate_env(env, num_steps=args.env_steps)
    validate_legacy_mode_spotcheck()
    validate_reward_power_decoupling(env)


def validate_legacy_mode_spotcheck():
    """Legacy 15-D still builds and matches StaticFns time decoding."""
    from softlearning.environments.utils import get_environment_from_params
    print('\nLegacy mode spot-check (15-D):')
    env = get_environment_from_params({
        'domain': 'PVTracking', 'task': 'v0', 'universe': 'gym',
        'kwargs': {'observation_mode': 'legacy'},
    })
    obs = env.reset()
    assert obs.shape == (LEGACY_OBS_DIM,)
    decoded = float(StaticFns.time_of_day_from_obs(obs))
    assert abs(decoded - 6.0) < 0.05, 'legacy reset time decode'
    env.close()
    print('  legacy shape and time decode: OK')


def validate_reward_power_decoupling(env):
    """Reward from raw power path; observation norms are clipped separately."""
    print('\nReward/power decoupling check:')
    obs = env.reset()
    act = np.zeros(env.action_space.shape, dtype=np.float32)
    next_obs, reward, _, info = env.step(act)
    expected = float(info['energy_kwh']) - float(info['movement_cost'])
    assert abs(reward - expected) < 1e-6
    assert float(info['ghi_wm2']) >= 0.0
    assert float(next_obs[5]) <= 1.0 + 1e-5
    print('  reward uses raw weather/power; ghi_norm clipped in obs: OK')


if __name__ == '__main__':
    main()
