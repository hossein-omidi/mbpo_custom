#!/usr/bin/env python3
"""Evaluate a trained agent checkpoint and save rollout plots.

This script is standalone and does not change repository source files.
It loads the saved checkpoint produced by the training procedure,
constructs the evaluation environment from variant parameters,
does rollouts with the loaded policy, and writes PNG figures.
"""

import argparse
import json
import os
import pickle
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


def load_checkpoint(checkpoint_dir):
    checkpoint_file = os.path.join(checkpoint_dir, 'checkpoint.pkl')
    if not os.path.exists(checkpoint_file):
        raise FileNotFoundError('Checkpoint file not found: %s' % checkpoint_file)
    with open(checkpoint_file, 'rb') as f:
        return pickle.load(f)


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


def get_policy(variant, environment, picklable):
    policy = get_policy_from_variant(variant, environment, Qs=[None])
    if 'policy_weights' not in picklable:
        raise KeyError('policy_weights not found in checkpoint.')
    policy.set_weights(picklable['policy_weights'])
    return policy


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
    return summary_path


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


def main(args):
    checkpoint_path = args.checkpoint.rstrip('/')
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
    picklable = load_checkpoint(checkpoint_path)

    environment = get_eval_environment(variant, args.eval_env_override)
    policy = get_policy(variant, environment, picklable)

    with policy.set_deterministic(args.deterministic):
        paths = rollouts(
            args.num_rollouts,
            environment,
            policy,
            path_length=args.max_path_length,
            render_mode=args.render_mode)

    rewards, lengths = rollout_metrics(paths)
    summary_path = save_summary(
        args.outdir,
        checkpoint_path,
        rewards,
        lengths,
        deterministic=args.deterministic,
        max_path_length=args.max_path_length)
    reward_plot = plot_rewards(args.outdir, rewards)
    length_plot = plot_lengths(args.outdir, lengths)

    print('Evaluation complete.')
    print('Saved:')
    print('  %s' % summary_path)
    print('  %s' % reward_plot)
    print('  %s' % length_plot)


if __name__ == '__main__':
    args = parse_args()
    main(args)
