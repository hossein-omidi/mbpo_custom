#!/usr/bin/env python3
"""Evaluate a trained agent checkpoint and save rollout plots.

This script is standalone and does not change repository source files.
It loads the saved checkpoint produced by the training procedure,
constructs the evaluation environment from variant parameters,
does rollouts with the loaded policy, saves aggregate reward/length plots,
and exports each rollout as a full trajectory CSV with observations,
actions, rewards, and environment info for each PV timestep.
"""

import argparse
import glob
import json
import os
import pickle
import re
from distutils.util import strtobool

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError as e:
    raise SystemExit(
        'matplotlib is required for this script. Install it with: pip install matplotlib')

try:
    import numpy as np
except ImportError:
    raise SystemExit('numpy is required for this script. Install it with: pip install numpy')

try:
    import tensorflow as tf
except ImportError:
    raise SystemExit('tensorflow is required for this script. Install it with: pip install tensorflow')

from softlearning.environments.utils import get_environment_from_params
from softlearning.policies.utils import get_policy_from_variant
from softlearning.samplers import rollouts
from softlearning.utils.keras import _apply_keras_hdf5_compat_patches


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate a trained agent checkpoint and save plots.')
    parser.add_argument(
        'checkpoint',
        type=str,
        help='Path to a saved checkpoint directory, e.g. /path/to/checkpoint_000001')
    parser.add_argument(
        '--outdir',
        type=str,
        default='evaluation',
        help='Directory to save evaluation plots and metrics.')
    parser.add_argument(
        '--num-rollouts', '-n',
        type=int,
        default=5,
        help='Number of evaluation rollouts to run.')
    parser.add_argument(
        '--max-path-length', '-l',
        type=int,
        default=1000,
        help='Maximum length of each rollout.')
    parser.add_argument(
        '--deterministic',
        type=lambda x: bool(strtobool(x)),
        nargs='?', const=True,
        default=True,
        help='Run the policy deterministically during evaluation.')
    parser.add_argument(
        '--render-mode',
        type=str,
        default=None,
        choices=('human', 'rgb_array', None),
        help='Optional render mode for the environment.')
    parser.add_argument(
        '--variant-file',
        type=str,
        default='params.json',
        help='Variant JSON filename stored in the experiment root.')
    parser.add_argument(
        '--eval-env-override',
        type=str,
        default=None,
        help='Optional JSON file for overriding evaluation environment params.')
    return parser.parse_args()


def load_variant(experiment_root, variant_file):
    path = os.path.join(experiment_root, variant_file)
    if not os.path.exists(path):
        raise FileNotFoundError('Variant file not found: %s' % path)
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_policy_weights(checkpoint_dir):
    """Load policy weights from the lightweight checkpoint file if present."""
    weights_path = os.path.join(checkpoint_dir, 'policy_weights.pkl')
    if os.path.exists(weights_path):
        with open(weights_path, 'rb') as f:
            return pickle.load(f)

    checkpoint_file = os.path.join(checkpoint_dir, 'checkpoint.pkl')
    if not os.path.exists(checkpoint_file):
        raise FileNotFoundError(
            'No policy_weights.pkl or checkpoint.pkl in: %s' % checkpoint_dir)
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


def get_eval_environment(variant, override_path=None):
    environment_params = variant['environment_params']
    eval_env_params = (
        environment_params.get('evaluation')
        if 'evaluation' in environment_params
        else environment_params['training'])

    if override_path is not None:
        with open(override_path, 'r', encoding='utf-8') as f:
            override = json.load(f)
        eval_env_params = deep_update(eval_env_params, override)

    return get_environment_from_params(eval_env_params)


def resolve_checkpoint_path(checkpoint_pattern):
    checkpoint_path = os.path.expanduser(checkpoint_pattern)

    if os.path.exists(checkpoint_path):
        return checkpoint_path.rstrip('/')

    # Convert angle-bracket placeholders to glob stars
    wildcard_path = re.sub(r'<[^>]+>', '*', checkpoint_path)
    matches = glob.glob(wildcard_path)
    if matches:
        return max(matches, key=os.path.getmtime)

    # If user provided a directory under monitoring tree, search recursively
    root = os.path.expanduser('~/ray_mbpo/PVTracking')
    if os.path.isdir(root):
        search_pattern = os.path.join(root, '**', 'checkpoint_*')
        matches = glob.glob(search_pattern, recursive=True)
        if matches:
            return max(matches, key=os.path.getmtime)

    raise FileNotFoundError(
        'Checkpoint directory not found: %s\n'
        'Use a real checkpoint path, not literal placeholders.\n'
        'Example: /home/ecer/ray_mbpo/PVTracking/pv_tracking/seed:9314_2026-05-20_10-17-430gnvtopf/checkpoint_51'
        % checkpoint_pattern)


def get_policy(variant, environment, policy_weights):
    policy = get_policy_from_variant(variant, environment, Qs=[None])
    policy.set_weights(policy_weights)
    return policy


def default_max_path_length(variant, cli_default):
    """Use PV episode length when the config targets PVTracking."""
    try:
        domain = variant['environment_params']['training']['domain']
        if domain == 'PVTracking':
            return 63
    except (KeyError, TypeError):
        pass
    sampler_kwargs = variant.get('sampler_params', {}).get('kwargs', {})
    return sampler_kwargs.get('max_path_length', cli_default)


def rollout_metrics(paths):
    rewards = []
    lengths = []
    for path in paths:
        if 'rewards' not in path:
            raise KeyError('Rollout path missing rewards.')
        rewards.append(float(np.sum(path['rewards'])))
        lengths.append(int(len(path['rewards'])))
    return np.array(rewards), np.array(lengths)


def save_summary(outdir, checkpoint_dir, rewards, lengths, deterministic, max_path_length):
    summary_path = os.path.join(outdir, 'evaluation_summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write('Checkpoint: %s\n' % checkpoint_dir)
        f.write('Num rollouts: %d\n' % len(rewards))
        f.write('Deterministic: %s\n' % str(deterministic))
        f.write('Max path length: %d\n' % max_path_length)
        f.write('\n')
        f.write('Rollout rewards:\n')
        for idx, reward in enumerate(rewards, 1):
            f.write('  rollout_%d: %.6f\n' % (idx, reward))
        f.write('\n')
        f.write('Rollout lengths:\n')
        for idx, length in enumerate(lengths, 1):
            f.write('  rollout_%d: %d\n' % (idx, length))
        f.write('\n')
        f.write('Rollout trajectory CSV files are saved under the `rollouts/` subfolder.\n')
    return summary_path


def save_rollout_paths(outdir, paths):
    rollouts_dir = os.path.join(outdir, 'rollouts')
    os.makedirs(rollouts_dir, exist_ok=True)

    for idx, path in enumerate(paths, start=1):
        observations = np.asarray(path['observations'])
        actions = np.asarray(path['actions'])
        rewards = np.asarray(path['rewards'])
        terminals = np.asarray(path.get('terminals', [False] * len(rewards)))
        infos = path.get('infos', [])

        info_keys = []
        if infos:
            info_keys = sorted({key for info in infos for key in info.keys()})

        header = []
        if observations.ndim == 2:
            obs_dim = observations.shape[1]
            header += [f'obs_{i}' for i in range(obs_dim)]
        else:
            header += ['obs']
        if actions.ndim == 2:
            act_dim = actions.shape[1]
            header += [f'action_{i}' for i in range(act_dim)]
        else:
            header += ['action']
        header += ['reward', 'terminal']
        header += info_keys

        rows = []
        for t in range(len(rewards)):
            row = []
            if observations.ndim == 2:
                row.extend(observations[t].tolist())
            else:
                row.append(float(observations[t]))
            if actions.ndim == 2:
                row.extend(actions[t].tolist())
            else:
                row.append(float(actions[t]))
            row.append(float(rewards[t]))
            row.append(bool(terminals[t]))
            info = infos[t] if t < len(infos) else {}
            for key in info_keys:
                row.append(info.get(key, ''))
            rows.append(row)

        csv_path = os.path.join(rollouts_dir, f'rollout_{idx}.csv')
        with open(csv_path, 'w', encoding='utf-8') as f:
            f.write(','.join(header) + '\n')
            for row in rows:
                f.write(','.join(str(x) for x in row) + '\n')

    return rollouts_dir


def plot_rewards(outdir, rewards):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(np.arange(1, len(rewards) + 1), rewards, marker='o', linestyle='-', color='#2171b5')
    ax.set_title('Evaluation episode rewards')
    ax.set_xlabel('Rollout index')
    ax.set_ylabel('Total reward')
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.set_xticks(np.arange(1, len(rewards) + 1))
    filepath = os.path.join(outdir, 'evaluation_rewards.png')
    fig.tight_layout()
    fig.savefig(filepath, dpi=150)
    plt.close(fig)
    return filepath


def plot_lengths(outdir, lengths):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(np.arange(1, len(lengths) + 1), lengths, color='#41ab5d')
    ax.set_title('Evaluation episode lengths')
    ax.set_xlabel('Rollout index')
    ax.set_ylabel('Episode length')
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.set_xticks(np.arange(1, len(lengths) + 1))
    filepath = os.path.join(outdir, 'evaluation_lengths.png')
    fig.tight_layout()
    fig.savefig(filepath, dpi=150)
    plt.close(fig)
    return filepath


def plot_rollout_series(outdir, path, idx):
    infos = path.get('infos', [])
    times = None
    if infos and 'time' in infos[0]:
        times = np.array([info.get('time', np.nan) for info in infos], dtype=np.float64)
        if np.isfinite(times).all():
            times = (times - times[0]) / 3600.0
    else:
        times = np.arange(len(path['rewards']), dtype=np.float64)

    rewards = np.asarray(path['rewards'], dtype=np.float64)
    actions = np.asarray(path['actions'], dtype=np.float64)
    observations = np.asarray(path['observations'], dtype=np.float64)
    power = np.asarray([info.get('power', np.nan) for info in infos], dtype=np.float64)
    tilt = np.asarray([info.get('tilt', np.nan) for info in infos], dtype=np.float64)
    azimuth = np.asarray([info.get('azimuth', np.nan) for info in infos], dtype=np.float64)
    poa_global = np.asarray([info.get('poa_global', np.nan) for info in infos], dtype=np.float64)

    rollout_dir = os.path.join(outdir, 'rollout_plots')
    os.makedirs(rollout_dir, exist_ok=True)

    def save_plot(x, y, title, ylabel, name):
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(x, y, marker='o', linestyle='-')
        ax.set_title(f'Rollout {idx} {title}')
        ax.set_xlabel('Time (hours)' if np.isfinite(times).all() else 'Step')
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle='--', alpha=0.4)
        filepath = os.path.join(rollout_dir, name)
        fig.tight_layout()
        fig.savefig(filepath, dpi=150)
        plt.close(fig)
        return filepath

    files = []
    files.append(save_plot(times, power, 'Power', 'Power (W)', f'rollout_{idx}_power.png'))
    files.append(save_plot(times, tilt, 'Tilt', 'Tilt (deg)', f'rollout_{idx}_tilt.png'))
    files.append(save_plot(times, azimuth, 'Azimuth', 'Azimuth (deg)', f'rollout_{idx}_azimuth.png'))
    files.append(save_plot(times, rewards, 'Reward', 'Reward', f'rollout_{idx}_reward.png'))
    if not np.all(np.isnan(poa_global)):
        files.append(save_plot(times, poa_global, 'POA Global', 'POA Global', f'rollout_{idx}_poa_global.png'))

    return files


def plot_rollout_combined(outdir, path, idx):
    infos = path.get('infos', [])
    times = None
    if infos and 'time' in infos[0]:
        times = np.array([info.get('time', np.nan) for info in infos], dtype=np.float64)
        if np.isfinite(times).all():
            times = (times - times[0]) / 3600.0
    else:
        times = np.arange(len(path['rewards']), dtype=np.float64)

    rewards = np.asarray(path['rewards'], dtype=np.float64)
    power = np.asarray([info.get('power', np.nan) for info in infos], dtype=np.float64)
    tilt = np.asarray([info.get('tilt', np.nan) for info in infos], dtype=np.float64)
    azimuth = np.asarray([info.get('azimuth', np.nan) for info in infos], dtype=np.float64)

    rollout_dir = os.path.join(outdir, 'rollout_plots')
    os.makedirs(rollout_dir, exist_ok=True)

    x_label = 'Time (hours)' if np.isfinite(times).all() else 'Step'
    x = times

    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(10, 12))
    axes[0].plot(x, power, marker='o', linestyle='-', color='#1f77b4')
    axes[0].set_ylabel('Power (W)')
    axes[0].set_title(f'Rollout {idx} — Power / Tilt / Azimuth / Reward')

    axes[1].plot(x, tilt, marker='o', linestyle='-', color='#ff7f0e')
    axes[1].set_ylabel('Tilt (deg)')

    axes[2].plot(x, azimuth, marker='o', linestyle='-', color='#2ca02c')
    axes[2].set_ylabel('Azimuth (deg)')

    axes[3].plot(x, rewards, marker='o', linestyle='-', color='#d62728')
    axes[3].set_ylabel('Reward')
    axes[3].set_xlabel(x_label)

    for ax in axes:
        ax.grid(True, linestyle='--', alpha=0.4)

    filepath = os.path.join(rollout_dir, f'rollout_{idx}_combined.png')
    fig.tight_layout()
    fig.savefig(filepath, dpi=150)
    plt.close(fig)
    return filepath


def main(args):
    checkpoint_path = resolve_checkpoint_path(args.checkpoint)
    if os.path.isfile(checkpoint_path):
        checkpoint_path = os.path.dirname(checkpoint_path)

    if not os.path.isdir(checkpoint_path):
        raise FileNotFoundError('Checkpoint directory not found: %s' % checkpoint_path)

    experiment_root = os.path.dirname(checkpoint_path)
    if not os.path.exists(os.path.join(experiment_root, args.variant_file)):
        raise FileNotFoundError('Experiment root variant file not found: %s' % experiment_root)

    os.makedirs(args.outdir, exist_ok=True)

    gpu_options = tf.GPUOptions(allow_growth=True)
    session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    tf.keras.backend.set_session(session)

    variant = load_variant(experiment_root, args.variant_file)
    policy_weights = load_policy_weights(checkpoint_path)

    environment = get_eval_environment(variant, args.eval_env_override)
    policy = get_policy(variant, environment, policy_weights)

    path_length = args.max_path_length
    if path_length == 1000:
        path_length = default_max_path_length(variant, path_length)

    with policy.set_deterministic(args.deterministic):
        paths = rollouts(
            args.num_rollouts,
            environment,
            policy,
            path_length=path_length,
            render_mode=args.render_mode)

    rewards, lengths = rollout_metrics(paths)
    rollouts_dir = save_rollout_paths(args.outdir, paths)
    summary_path = save_summary(
        args.outdir,
        checkpoint_path,
        rewards,
        lengths,
        deterministic=args.deterministic,
        max_path_length=path_length)
    reward_plot = plot_rewards(args.outdir, rewards)
    length_plot = plot_lengths(args.outdir, lengths)

    rollout_plot_files = []
    for idx, path in enumerate(paths, start=1):
        rollout_plot_files.extend(plot_rollout_series(args.outdir, path, idx))
        rollout_plot_files.append(plot_rollout_combined(args.outdir, path, idx))

    print('Evaluation complete.')
    print('Saved:')
    print('  %s' % summary_path)
    print('  %s' % reward_plot)
    print('  %s' % length_plot)
    print('  %s' % rollouts_dir)
    for path in rollout_plot_files:
        print('  %s' % path)


if __name__ == '__main__':
    args = parse_args()
    main(args)
