#!/usr/bin/env python3
"""Rank PV tracking checkpoints by hold-out total energy (scientific confirmation).

Training saves best_eval_checkpoint when evaluation/return-average improves.
This script re-evaluates candidate checkpoints on fixed or held-out dates with
the same UTC MDP as training and reports mean total_energy_kwh vs baselines.
"""

import argparse
import glob
import json
import os
import re
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from pv_trial_paths import resolve_trial_dir, format_trial_hint, is_placeholder_trial

import numpy as np
import tensorflow as tf

from softlearning.policies.utils import get_policy_from_variant
from softlearning.utils.keras import _apply_keras_hdf5_compat_patches

from eval_utils import (
    EVAL_PROTOCOL_INHERIT,
    compare_method_table,
    get_eval_environment,
    get_rollout_metadata,
    make_baseline_rollout,
    run_learned_policy_rollout,
    validate_policy_environment_observation_dims,
    PV_EPISODE_MAX_STEPS,
)


def _checkpoint_dirs(experiment_root):
    dirs = []
    best = os.path.join(experiment_root, 'best_eval_checkpoint')
    if os.path.isdir(best):
        dirs.append(('best_eval_checkpoint', best))
    pattern = os.path.join(experiment_root, 'checkpoint_*')
    numbered = []
    for path in glob.glob(pattern):
        name = os.path.basename(path)
        m = re.search(r'checkpoint_(\d+)', name)
        if m and os.path.isdir(path):
            numbered.append((int(m.group(1)), name, path))
    for _, name, path in sorted(numbered):
        dirs.append((name, path))
    return dirs


def _load_policy_weights(checkpoint_dir):
    import pickle
    weights_path = os.path.join(checkpoint_dir, 'policy_weights.pkl')
    if os.path.exists(weights_path):
        with open(weights_path, 'rb') as f:
            return pickle.load(f)
    checkpoint_file = os.path.join(checkpoint_dir, 'checkpoint.pkl')
    _apply_keras_hdf5_compat_patches()
    with open(checkpoint_file, 'rb') as f:
        picklable = pickle.load(f)
    return picklable['policy_weights']


def _mean_energy(paths):
    return float(np.mean([get_rollout_metadata(p)['total_energy_kwh'] for p in paths]))


def evaluate_checkpoint(
        checkpoint_dir, variant, args, session):
    policy_weights = _load_policy_weights(checkpoint_dir)
    eval_env, eval_env_params = get_eval_environment(
        variant,
        test_start_date=args.test_start_date,
        test_end_date=args.test_end_date,
        fixed_eval_dates=args.fixed_eval_dates,
        eval_protocol=args.eval_protocol,
    )
    policy = get_policy_from_variant(variant, eval_env, Qs=[None])
    policy.set_weights(policy_weights)
    validate_policy_environment_observation_dims(
        policy, eval_env, policy_weights=policy_weights, eval_env_params=eval_env_params)

    paths = []
    paths_by_name = {}
    deterministic = not args.stochastic
    for idx in range(args.num_rollouts):
        paths.append(run_learned_policy_rollout(
            policy,
            eval_env,
            args.max_path_length,
            seed=idx,
            deterministic=deterministic,
        ))
    paths_by_name['learned_policy'] = paths

    if args.compare_baselines:
        for name in args.baseline_types:
            bpaths = []
            for idx in range(args.num_rollouts):
                env, _ = get_eval_environment(
                    variant,
                    test_start_date=args.test_start_date,
                    test_end_date=args.test_end_date,
                    fixed_eval_dates=args.fixed_eval_dates,
                    eval_protocol=args.eval_protocol,
                )
                bpaths.append(make_baseline_rollout(
                    env, name, args.max_path_length, seed=idx))
            paths_by_name[name] = bpaths

    energy = _mean_energy(paths)
    reward = float(np.mean([float(np.sum(p['rewards'])) for p in paths]))
    row = {
        'checkpoint': os.path.basename(checkpoint_dir),
        'path': checkpoint_dir,
        'mean_total_energy_kwh': energy,
        'mean_total_reward': reward,
    }
    if args.compare_baselines:
        table = compare_method_table(paths_by_name)
        for t in table:
            row['energy_%s' % t['method']] = t['total_energy_kwh_mean']
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'experiment_root',
        type=str,
        nargs='?',
        default='latest',
        help='Ray trial dir (seed:...), or "latest" (default).')
    parser.add_argument(
        '--ray-root',
        type=str,
        default=None,
        help='PV tracking log root (default ~/ray_mbpo/PVTracking/pv_tracking).')
    parser.add_argument('--variant-file', type=str, default='params.json')
    parser.add_argument('--num-rollouts', '-n', type=int, default=4)
    parser.add_argument('--max-path-length', type=int, default=PV_EPISODE_MAX_STEPS)
    parser.add_argument(
        '--stochastic',
        action='store_true',
        help='Use sampled SAC actions instead of canonical deterministic tanh(mu).')
    parser.add_argument('--fixed-eval-dates', type=str, default=None)
    parser.add_argument('--test-start-date', type=str, default=None)
    parser.add_argument('--test-end-date', type=str, default=None)
    parser.add_argument('--eval-protocol', type=str, default=EVAL_PROTOCOL_INHERIT)
    parser.add_argument('--compare-baselines', action='store_true')
    parser.add_argument('--baseline-types', nargs='+',
                        default=['fixed_no_motion', 'sun_tracking'])
    parser.add_argument('--limit', type=int, default=None,
                        help='Evaluate at most N checkpoints (newest first)')
    args = parser.parse_args()

    try:
        root = resolve_trial_dir(args.experiment_root, root=args.ray_root)
    except FileNotFoundError as exc:
        print(exc)
        print(format_trial_hint(args.ray_root))
        sys.exit(1)
    if args.experiment_root in ('latest', None, '') or is_placeholder_trial(
            args.experiment_root):
        print('[select_best_checkpoint] Using trial: %s' % root)

    variant_path = os.path.join(root, args.variant_file)
    with open(variant_path, 'r', encoding='utf-8') as f:
        variant = json.load(f)

    candidates = _checkpoint_dirs(root)
    if args.limit:
        best = [item for item in candidates if item[0] == 'best_eval_checkpoint']
        numbered = [item for item in candidates if item[0] != 'best_eval_checkpoint']
        candidates = best + numbered[-args.limit:]

    gpu_options = tf.GPUOptions(allow_growth=True)
    session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    tf.keras.backend.set_session(session)

    rows = []
    for name, ckpt_dir in candidates:
        print('[select_best_checkpoint] Evaluating %s ...' % name)
        try:
            rows.append(evaluate_checkpoint(ckpt_dir, variant, args, session))
        except Exception as exc:
            print('[select_best_checkpoint] SKIP %s: %s' % (name, exc))

    rows.sort(key=lambda r: r['mean_total_energy_kwh'], reverse=True)
    print('\nRanked by mean total_energy_kwh (hold-out, UTC MDP):')
    for i, row in enumerate(rows, 1):
        line = '  %d. %s  energy=%.4f kWh  reward=%.4f' % (
            i, row['checkpoint'], row['mean_total_energy_kwh'], row['mean_total_reward'])
        if args.compare_baselines:
            for key, val in sorted(row.items()):
                if key.startswith('energy_'):
                    line += '  %s=%.4f' % (key, val)
        print(line)
    if rows:
        print('\nRecommended for reporting: %s' % rows[0]['path'])


if __name__ == '__main__':
    main()
