#!/usr/bin/env python3
"""Compare a trained PV tracking policy against simple baselines.

Uses the same evaluation environment settings, per-rollout seeds, and
metrics as scripts/evaluate_agent.py (via scripts/eval_utils.py).
"""

import argparse
import glob
import json
import os
import pickle
import re
import sys

import numpy as np
import tensorflow as tf

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from softlearning.policies.utils import get_policy_from_variant
from softlearning.utils.keras import _apply_keras_hdf5_compat_patches

from eval_utils import (
    EVAL_PROTOCOL_INHERIT,
    EVAL_PROTOCOL_LEGACY_UTC,
    EVAL_PROTOCOL_UTC,
    compare_method_table,
    describe_eval_config,
    get_eval_environment,
    make_baseline_rollout,
    run_learned_policy_rollout,
    summarize_paths,
    validate_eval_coverage,
    validate_policy_environment_observation_dims,
    write_eval_scenario_confirmation,
    write_reward_time_report,
)


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


def _write_stats(f, label, stats):
    f.write('%s (n=%d): mean=%.4f std=%.4f min=%.4f max=%.4f\n' % (
        label, stats['count'], stats['mean'], stats['std'], stats['min'], stats['max']))


def main():
    parser = argparse.ArgumentParser(
        description='Compare a PV policy checkpoint against simple baselines.')
    parser.add_argument('checkpoint', type=str, help='Path to checkpoint directory or glob pattern')
    parser.add_argument('--outdir', type=str, default='evaluation', help='Directory to save summary files')
    parser.add_argument('--num-rollouts', '-n', type=int, default=10, help='Number of rollouts per method')
    parser.add_argument('--max-path-length', '-l', type=int, default=39, help='Rollout horizon')
    parser.add_argument('--variant-file', type=str, default='params.json', help='Variant JSON in experiment root')
    parser.add_argument('--deterministic', action='store_true', help='Run the policy deterministically')
    parser.add_argument('--test-start-date', type=str, default=None, help='Hold-out start date (YYYY-MM-DD)')
    parser.add_argument('--test-end-date', type=str, default=None, help='Hold-out end date (YYYY-MM-DD)')
    parser.add_argument('--fixed-eval-dates', type=str, default=None,
                        help='Comma-separated fixed evaluation dates (YYYY-MM-DD)')
    parser.add_argument('--baseline-types', nargs='+',
                        default=['fixed_no_motion', 'sun_tracking'],
                        help='Baselines to compare')
    parser.add_argument('--min-rollouts', type=int, default=10,
                        help='Warn if fewer rollouts with randomize_day=True')
    parser.add_argument('--debug-first-rollout', action='store_true',
                        help='Log observation shapes for the first learned-policy rollout only')
    parser.add_argument(
        '--eval-protocol',
        type=str,
        default=EVAL_PROTOCOL_INHERIT,
        choices=(EVAL_PROTOCOL_INHERIT, EVAL_PROTOCOL_UTC, EVAL_PROTOCOL_LEGACY_UTC),
        help='UTC daylight episode grid (inherit/utc/legacy_utc are equivalent).')
    args = parser.parse_args()

    checkpoint_dir = resolve_checkpoint_path(args.checkpoint)
    variant = load_variant(os.path.dirname(checkpoint_dir), args.variant_file)
    policy_weights = load_policy_weights(checkpoint_dir)

    gpu_options = tf.GPUOptions(allow_growth=True)
    session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    tf.keras.backend.set_session(session)

    eval_env, eval_env_params = get_eval_environment(
        variant,
        test_start_date=args.test_start_date,
        test_end_date=args.test_end_date,
        fixed_eval_dates=args.fixed_eval_dates,
        eval_protocol=args.eval_protocol,
    )
    for warning in validate_eval_coverage(args.num_rollouts, eval_env_params, args.min_rollouts):
        print('[compare_baselines] WARNING: %s' % warning)

    policy = get_policy_from_variant(variant, eval_env, Qs=[None])
    policy.set_weights(policy_weights)
    dim_info = validate_policy_environment_observation_dims(
        policy, eval_env, policy_weights=policy_weights, eval_env_params=eval_env_params)
    print('[compare_baselines] Verified policy_input_dim=%d env_observation_dim=%d '
          'observation_mode=%r' % (
              dim_info['policy_dim'], dim_info['env_dim'], dim_info['env_mode']))

    os.makedirs(args.outdir, exist_ok=True)
    summary_path = os.path.join(args.outdir, 'baseline_comparison_summary.txt')

    policy_paths = []
    baseline_paths = {name: [] for name in args.baseline_types}

    for idx in range(args.num_rollouts):
        env, _ = get_eval_environment(
            variant,
            test_start_date=args.test_start_date,
            test_end_date=args.test_end_date,
            fixed_eval_dates=args.fixed_eval_dates,
            eval_protocol=args.eval_protocol,
        )
        policy_paths.append(
            run_learned_policy_rollout(
                policy,
                env,
                args.max_path_length,
                seed=idx,
                deterministic=args.deterministic,
                policy_input_dim=dim_info['policy_dim'],
                debug_first_step=(args.debug_first_rollout and idx == 0),
            ))

    for name in args.baseline_types:
        for idx in range(args.num_rollouts):
            env, _ = get_eval_environment(
                variant,
                test_start_date=args.test_start_date,
                test_end_date=args.test_end_date,
                fixed_eval_dates=args.fixed_eval_dates,
                eval_protocol=args.eval_protocol,
            )
            baseline_paths[name].append(
                make_baseline_rollout(env, name, args.max_path_length, seed=idx))

    paths_by_name = {'learned_policy': policy_paths}
    paths_by_name.update(baseline_paths)

    policy_stats = summarize_paths(policy_paths)
    baseline_summaries = {name: summarize_paths(baseline_paths[name]) for name in args.baseline_types}
    comparison_rows = compare_method_table(paths_by_name)

    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write('PV Tracking Baseline Comparison\n')
        f.write('Checkpoint: %s\n' % checkpoint_dir)
        f.write('Deterministic: %s\n' % args.deterministic)
        f.write('Rollouts per method: %d\n' % args.num_rollouts)
        f.write('Policy input dim: %d\n' % dim_info['policy_dim'])
        f.write('Env observation dim: %d (mode=%r)\n' % (
            dim_info['env_dim'], dim_info['env_mode']))
        f.write('\nEvaluation environment:\n')
        for line in describe_eval_config(eval_env_params):
            f.write('  %s\n' % line)
        f.write('\nAggregate comparison:\n')
        _write_stats(f, '  learned_policy reward', policy_stats['reward'])
        _write_stats(f, '  learned_policy energy_kwh', policy_stats['total_energy_kwh'])
        _write_stats(f, '  learned_policy movement', policy_stats['total_movement_cost'])
        for name, stats in baseline_summaries.items():
            f.write('\n  baseline: %s\n' % name)
            _write_stats(f, '    reward', stats['reward'])
            _write_stats(f, '    energy_kwh', stats['total_energy_kwh'])
            _write_stats(f, '    movement', stats['total_movement_cost'])
        f.write('\nTiming comparison (mean over rollouts; peak_pwr_t is env clock hour UTC):\n')
        f.write(
            '  method           reward    energy    mean_pwr  peak_pwr  '
            'peak_pwr_t  peak_rew_t  movement\n')
        for row in comparison_rows:
            f.write(
                '  %-16s %8.4f %8.4f %8.1f %8.1f %8.2f %8.2f %8.4f\n' % (
                    row['method'],
                    row['total_reward_mean'],
                    row['total_energy_kwh_mean'],
                    row['mean_power_w_mean'],
                    row['peak_power_w_mean'],
                    row['peak_power_time_mean'],
                    row['peak_reward_time_mean'],
                    row['movement_cost_mean']))

    write_reward_time_report(args.outdir, policy_paths, paths_by_name=paths_by_name)
    scenario_report = write_eval_scenario_confirmation(
        args.outdir, eval_env_params, paths_by_name, args.max_path_length)

    print('Baseline comparison saved to:', summary_path)
    print('Scenario confirmation:', scenario_report)
    print('Learned policy: mean_reward=%.4f mean_energy=%.4f kWh' % (
        policy_stats['reward']['mean'], policy_stats['total_energy_kwh']['mean']))
    for name, stats in baseline_summaries.items():
        print('  %s: mean_reward=%.4f mean_energy=%.4f kWh' % (
            name, stats['reward']['mean'], stats['total_energy_kwh']['mean']))


if __name__ == '__main__':
    main()
