#!/usr/bin/env python3
"""Compare a trained PV tracking policy against simple baselines.

This script evaluates a checkpointed policy and fixed baseline strategies in the
PVTracking environment, reporting energy and movement costs for each rollout.
"""

import argparse
import glob
import json
import os
import pickle
import re

import numpy as np

from softlearning.environments.utils import get_environment_from_params
from softlearning.policies.utils import get_policy_from_variant
from softlearning.utils.keras import _apply_keras_hdf5_compat_patches


def resolve_checkpoint_path(checkpoint_pattern):
    checkpoint_path = os.path.expanduser(checkpoint_pattern)
    if os.path.exists(checkpoint_path):
        return checkpoint_path.rstrip('/')

    wildcard_path = re.sub(r'<[^>]+>', '*', checkpoint_path)
    matches = glob.glob(wildcard_path)
    if matches:
        return max(matches, key=os.path.getmtime)

    root = os.path.expanduser('~/ray_mbpo/PVTracking')
    if os.path.isdir(root):
        search_pattern = os.path.join(root, '**', 'checkpoint_*')
        matches = glob.glob(search_pattern, recursive=True)
        if matches:
            return max(matches, key=os.path.getmtime)

    raise FileNotFoundError('Checkpoint directory not found: %s' % checkpoint_pattern)


def load_variant(experiment_root, variant_file='params.json'):
    path = os.path.join(experiment_root, variant_file)
    if not os.path.exists(path):
        raise FileNotFoundError('Variant file not found: %s' % path)
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_policy_weights(checkpoint_dir):
    weights_path = os.path.join(checkpoint_dir, 'policy_weights.pkl')
    if os.path.exists(weights_path):
        with open(weights_path, 'rb') as f:
            return pickle.load(f)

    checkpoint_file = os.path.join(checkpoint_dir, 'checkpoint.pkl')
    if not os.path.exists(checkpoint_file):
        raise FileNotFoundError('No policy_weights.pkl or checkpoint.pkl in: %s' % checkpoint_dir)

    _apply_keras_hdf5_compat_patches()
    with open(checkpoint_file, 'rb') as f:
        picklable = pickle.load(f)
    if isinstance(picklable, dict) and 'policy_weights' in picklable:
        return picklable['policy_weights']
    raise KeyError('policy_weights not found in checkpoint.pkl')


def deep_update(original, override):
    if not isinstance(override, dict):
        return override
    updated = dict(original)
    for key, value in override.items():
        if key in updated and isinstance(updated[key], dict):
            updated[key] = deep_update(updated[key], value)
        else:
            updated[key] = value
    return updated


def normalize_angle_diff(target, current):
    diff = (target - current + 180.0) % 360.0 - 180.0
    return diff


def make_environment(variant, override=None):
    env_params = variant['environment_params']['training']
    if override:
        env_params = deep_update(env_params, override)
    return get_environment_from_params(env_params)


def run_rollout(policy, env, max_length=63, deterministic=True):
    obs = env.reset()
    total_reward = 0.0
    total_energy = 0.0
    total_movement = 0.0
    infos = []
    done = False
    step = 0

    while not done and step < max_length:
        action = policy.actions_np(obs[None])[0]
        next_obs, reward, done, info = env.step(action)
        total_reward += reward
        total_energy += info.get('energy_kwh', 0.0)
        total_movement += info.get('movement_cost', 0.0)
        infos.append(info)
        obs = next_obs
        step += 1

    return {
        'return': total_reward,
        'energy_kwh': total_energy,
        'movement_cost': total_movement,
        'length': step,
        'infos': infos,
    }


def run_baseline(env, baseline_type, max_length=63):
    obs = env.reset()
    total_reward = 0.0
    total_energy = 0.0
    total_movement = 0.0
    infos = []
    done = False
    step = 0

    while not done and step < max_length:
        solar_position = env._solar_position(env.current_time)
        target_tilt = float(np.clip(solar_position.zenith, *env.tilt_limits))
        target_azimuth = float(solar_position.azimuth)

        if baseline_type == 'fixed_no_motion':
            action = np.array([0.0, 0.0], dtype=np.float32)
        elif baseline_type == 'sun_tracking':
            tilt_diff = target_tilt - env.tilt
            azimuth_diff = normalize_angle_diff(target_azimuth, env.azimuth)
            action = np.array([
                np.clip(tilt_diff / env.max_delta_tilt, -1.0, 1.0),
                np.clip(azimuth_diff / env.max_delta_azimuth, -1.0, 1.0),
            ], dtype=np.float32)
        else:
            raise ValueError('Unknown baseline: %s' % baseline_type)

        next_obs, reward, done, info = env.step(action)
        total_reward += reward
        total_energy += info.get('energy_kwh', 0.0)
        total_movement += info.get('movement_cost', 0.0)
        infos.append(info)
        obs = next_obs
        step += 1

    return {
        'return': total_reward,
        'energy_kwh': total_energy,
        'movement_cost': total_movement,
        'length': step,
        'infos': infos,
    }


def print_summary(name, stats):
    print(f'[{name}] return={stats["return"]:.4f} energy_kwh={stats["energy_kwh"]:.4f} movement_cost={stats["movement_cost"]:.4f} length={stats["length"]}')


def main():
    parser = argparse.ArgumentParser(
        description='Compare a PV policy checkpoint against simple baselines.')
    parser.add_argument('checkpoint', type=str, help='Path to checkpoint directory or glob pattern')
    parser.add_argument('--outdir', type=str, default='evaluation', help='Directory to save summary files')
    parser.add_argument('--num-rollouts', '-n', type=int, default=10, help='Number of rollouts for each policy/baseline')
    parser.add_argument('--max-path-length', '-l', type=int, default=63, help='Rollout horizon')
    parser.add_argument('--variant-file', type=str, default='params.json', help='Variant JSON filename stored in the experiment root')
    parser.add_argument('--deterministic', action='store_true', help='Run the policy deterministically')
    parser.add_argument('--baseline-types', nargs='+', default=['fixed_no_motion', 'sun_tracking'], help='Baselines to compare')
    args = parser.parse_args()

    checkpoint_dir = resolve_checkpoint_path(args.checkpoint)
    variant = load_variant(os.path.dirname(checkpoint_dir), args.variant_file)
    policy_weights = load_policy_weights(checkpoint_dir)

    env = make_environment(variant)
    policy = get_policy_from_variant(variant, env, Qs=[None])
    policy.set_weights(policy_weights)

    os.makedirs(args.outdir, exist_ok=True)
    summary_path = os.path.join(args.outdir, 'baseline_comparison_summary.txt')

    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write('Baseline comparison for checkpoint: %s\n' % checkpoint_dir)
        f.write('Config file: %s\n' % args.variant_file)
        f.write('\n')

        print('Evaluating learned policy...')
        policy_stats = []
        for i in range(args.num_rollouts):
            stats = run_rollout(policy, env, max_length=args.max_path_length, deterministic=args.deterministic)
            policy_stats.append(stats)
            print_summary(f'policy_{i+1}', stats)
            f.write(f'policy_{i+1} {stats}\n')

        def summarize_group(name, group_stats):
            returns = [s['return'] for s in group_stats]
            energies = [s['energy_kwh'] for s in group_stats]
            movements = [s['movement_cost'] for s in group_stats]
            f.write(f'\n{name} summary:\n')
            f.write(f'  mean_return={np.mean(returns):.4f} std_return={np.std(returns):.4f}\n')
            f.write(f'  mean_energy={np.mean(energies):.4f} std_energy={np.std(energies):.4f}\n')
            f.write(f'  mean_movement={np.mean(movements):.4f} std_movement={np.std(movements):.4f}\n')
            print(f'[{name}] mean_return={np.mean(returns):.4f} std_return={np.std(returns):.4f} mean_energy={np.mean(energies):.4f} mean_movement={np.mean(movements):.4f}')

        summarize_group('policy', policy_stats)

        for baseline in args.baseline_types:
            baseline_stats = []
            print(f'Evaluating baseline: {baseline}')
            for i in range(args.num_rollouts):
                stats = run_baseline(env, baseline, max_length=args.max_path_length)
                baseline_stats.append(stats)
                print_summary(f'{baseline}_{i+1}', stats)
                f.write(f'{baseline}_{i+1} {stats}\n')
            summarize_group(baseline, baseline_stats)

    print('Baseline comparison saved to:', summary_path)


if __name__ == '__main__':
    main()
