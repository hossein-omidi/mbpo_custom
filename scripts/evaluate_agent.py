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
import sys
from collections import defaultdict
from distutils.util import strtobool
from datetime import datetime

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

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

from softlearning.policies.utils import get_policy_from_variant
from softlearning.samplers import rollout
from softlearning.utils.keras import _apply_keras_hdf5_compat_patches

from eval_utils import (
    TIME_WINDOWS,
    SOLAR_ALTITUDE_WINDOWS,
    analyze_rollout_path,
    aggregate_time_windows,
    aggregate_solar_altitude_windows,
    compare_method_table,
    describe_eval_config,
    env_timezone_from_paths,
    get_eval_environment,
    get_rollout_metadata,
    make_baseline_rollout,
    save_rollout_csv,
    summarize_by_group,
    summarize_paths,
    validate_eval_coverage,
    validate_policy_environment_observation_dims,
    EVAL_PROTOCOL_LEGACY_UTC,
    EVAL_PROTOCOL_INHERIT,
    EVAL_PROTOCOL_UTC,
    rollout_time_axis,
    rollout_xlabel,
    night_intervals_from_path,
    write_eval_scenario_confirmation,
    write_reward_time_report,
)


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
        default=10,
        help='Number of evaluation rollouts to run.')
    parser.add_argument(
        '--max-path-length', '-l',
        type=int,
        default=1000,
        help='Maximum length of each rollout.')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        default=False,
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
    parser.add_argument(
        '--test-start-date',
        type=str,
        default=None,
        help='Optional evaluation start date (YYYY-MM-DD) for day sampling.')
    parser.add_argument(
        '--test-end-date',
        type=str,
        default=None,
        help='Optional evaluation end date (YYYY-MM-DD) for day sampling.')
    parser.add_argument(
        '--fixed-eval-dates',
        type=str,
        default=None,
        help='Comma-separated list of exact dates (YYYY-MM-DD) for fixed evaluation rollouts.')
    parser.add_argument(
        '--compare-baselines',
        action='store_true',
        default=False,
        help='Run baseline tracking methods for comparison.')
    parser.add_argument(
        '--baseline-types',
        nargs='+',
        default=['fixed_no_motion', 'sun_tracking'],
        help='Baselines to run when --compare-baselines is set.')
    parser.add_argument(
        '--min-rollouts',
        type=int,
        default=10,
        help='Minimum recommended rollouts for random-day evaluation (warn if fewer).')
    parser.add_argument(
        '--no-report-by-season',
        action='store_true',
        default=False,
        help='Skip per-season breakdown in the summary and plots.')
    parser.add_argument(
        '--eval-weather-source',
        type=str,
        default=None,
        choices=('random', 'clearsky'),
        help='Override weather_source for evaluation (default: use variant config).')
    parser.add_argument(
        '--eval-protocol',
        type=str,
        default=EVAL_PROTOCOL_INHERIT,
        choices=(EVAL_PROTOCOL_INHERIT, EVAL_PROTOCOL_UTC, EVAL_PROTOCOL_LEGACY_UTC),
        help='UTC episode grid 06:00-21:45 (inherit/utc/legacy_utc are equivalent).')
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


def load_checkpoint_picklable(checkpoint_dir):
    checkpoint_file = os.path.join(checkpoint_dir, 'checkpoint.pkl')
    if not os.path.exists(checkpoint_file):
        return None
    _apply_keras_hdf5_compat_patches()
    with open(checkpoint_file, 'rb') as f:
        picklable = pickle.load(f)
    return picklable


def _policy_shape_summary(policy):
    return [w.shape for w in policy.get_weights()]


def _policy_weights_shape_summary(policy_weights):
    return [w.shape for w in policy_weights]


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


def get_policy(variant, environment, policy_weights, checkpoint_picklable=None):
    if (checkpoint_picklable is not None and
            isinstance(checkpoint_picklable, dict) and
            'policy' in checkpoint_picklable):
        try:
            policy = checkpoint_picklable['policy']
            # Ensure the restored policy object has its weights loaded.
            policy.set_weights(policy_weights)
            return policy
        except Exception as e:
            print('[evaluate_agent] Warning: unable to use pickled policy from checkpoint: %s' % str(e))

    training_environment = None
    if (checkpoint_picklable is not None and
            isinstance(checkpoint_picklable, dict) and
            'training_environment' in checkpoint_picklable):
        training_environment = checkpoint_picklable['training_environment']

    if training_environment is not None:
        policy = get_policy_from_variant(variant, training_environment, Qs=[None])
        try:
            policy.set_weights(policy_weights)
        except ValueError as e:
            print(
                '[evaluate_agent] Warning: policy weights mismatch on checkpoint training environment: %s' %
                str(e))
        else:
            if (training_environment.active_observation_shape == environment.active_observation_shape and
                    training_environment.action_space.shape == environment.action_space.shape):
                return policy

    policy = get_policy_from_variant(variant, environment, Qs=[None])
    try:
        policy.set_weights(policy_weights)
    except ValueError as e:
        input_dim = _infer_policy_input_dim(policy_weights)
        env_input_dim = int(np.prod(environment.active_observation_shape))
        env_mode = getattr(
            getattr(environment, 'unwrapped', environment),
            'observation_mode',
            'unknown',
        )
        if input_dim != env_input_dim:
            raise ValueError(
                'Policy observation dim (%d) does not match evaluation env dim (%d) '
                '(env observation_mode=%r). Use observation_mode=legacy (15-D) for '
                'old checkpoints or observation_mode=physical (11-D) for new training. '
                'Do not slice observations: index 10 is power_norm (legacy) vs cos_aoi '
                '(physical). Original error: %s' % (
                    input_dim, env_input_dim, env_mode, e))
        raise ValueError(
            'Policy weights shape mismatch when loading policy on evaluation environment. '
            'Evaluation env active_observation_shape=%s action_space=%s, '
            'policy weight shapes=%s, error=%s' % (
                environment.active_observation_shape,
                environment.action_space.shape,
                _policy_weights_shape_summary(policy_weights),
                str(e)))
    return policy


def _infer_policy_input_dim(policy_weights):
    if not policy_weights:
        raise ValueError('No policy weights provided.')
    first_weight = policy_weights[0]
    if hasattr(first_weight, 'shape') and len(first_weight.shape) == 2:
        return int(first_weight.shape[0])
    raise ValueError(
        'Unable to infer policy input dimension from saved weights: %s' %
        _policy_weights_shape_summary(policy_weights))


class ObservationSliceWrapper(object):
    def __init__(self, env, slice_dim):
        self._env = env
        self._slice_dim = int(slice_dim)

    @property
    def observation_space(self):
        return self._env.observation_space

    @property
    def action_space(self):
        return self._env.action_space

    @property
    def active_observation_shape(self):
        return (self._slice_dim,)

    def convert_to_active_observation(self, observation):
        active_observation = getattr(
            self._env, 'convert_to_active_observation', lambda x: x)(observation)
        return np.asarray(active_observation, dtype=np.float32)[..., :self._slice_dim]

    def __getattr__(self, name):
        return getattr(self._env, name)


class PolicyInputSliceWrapper(object):
    def __init__(self, policy, slice_dim):
        self._policy = policy
        self._slice_dim = int(slice_dim)

    def actions_np(self, conditions):
        if isinstance(conditions, (list, tuple)):
            processed = [
                np.asarray(condition, dtype=np.float32)[..., :self._slice_dim]
                for condition in conditions
            ]
            return self._policy.actions_np(processed)

        conditions = np.asarray(conditions, dtype=np.float32)
        if conditions.ndim == 1:
            conditions = conditions[None]
        conditions = conditions[..., :self._slice_dim]
        return self._policy.actions_np(conditions)

    def __getattr__(self, name):
        return getattr(self._policy, name)


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


def _write_stats_block(f, label, stats):
    f.write('%s (n=%d):\n' % (label, stats['count']))
    f.write('  mean=%.6f  std=%.6f  min=%.6f  max=%.6f\n' % (
        stats['mean'], stats['std'], stats['min'], stats['max']))


def save_summary(
        outdir,
        checkpoint_dir,
        paths,
        deterministic,
        max_path_length,
        eval_env_params,
        baseline_paths_by_name=None,
        report_by_season=True,
        warnings=None):
    """Write human-readable and JSON summaries with aggregate statistics."""
    summary_path = os.path.join(outdir, 'evaluation_summary.txt')
    json_path = os.path.join(outdir, 'evaluation_summary.json')

    policy_stats = summarize_paths(paths)
    season_stats = None
    weather_stats = None
    if report_by_season:
        season_stats = summarize_by_group(paths, lambda m: m['season'])
        weather_stats = summarize_by_group(paths, lambda m: m['weather_condition'])

    baseline_stats = {}
    if baseline_paths_by_name:
        for name, bpaths in baseline_paths_by_name.items():
            baseline_stats[name] = summarize_paths(bpaths)

    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write('PV Tracking Evaluation Summary\n')
        f.write('=' * 32 + '\n')
        f.write('Checkpoint: %s\n' % checkpoint_dir)
        f.write('Deterministic policy: %s\n' % deterministic)
        f.write('Max path length: %d\n' % max_path_length)
        f.write('Num rollouts: %d\n' % len(paths))
        f.write('\nEvaluation environment:\n')
        for line in describe_eval_config(eval_env_params):
            f.write('  %s\n' % line)
        if warnings:
            f.write('\nWarnings:\n')
            for w in warnings:
                f.write('  - %s\n' % w)
        f.write('\nAggregate metrics (learned policy):\n')
        _write_stats_block(f, '  Total reward', policy_stats['reward'])
        _write_stats_block(f, '  Total energy (kWh)', policy_stats['total_energy_kwh'])
        _write_stats_block(f, '  Total movement cost', policy_stats['total_movement_cost'])
        _write_stats_block(f, '  Episode length', policy_stats['episode_length'])

        if season_stats:
            f.write('\nBy season (learned policy):\n')
            for season, stats in season_stats.items():
                f.write('  %s:\n' % season)
                _write_stats_block(f, '    Reward', stats['reward'])
                _write_stats_block(f, '    Energy kWh', stats['total_energy_kwh'])

        if weather_stats:
            f.write('\nBy weather condition (learned policy):\n')
            for weather, stats in weather_stats.items():
                f.write('  %s: mean_reward=%.4f mean_energy=%.4f (n=%d)\n' % (
                    weather,
                    stats['reward']['mean'],
                    stats['total_energy_kwh']['mean'],
                    stats['reward']['count'],
                ))

        f.write('\nPer-rollout detail (with peak times):\n')
        for idx, path in enumerate(paths, 1):
            meta = analyze_rollout_path(path)
            f.write(
                '  rollout_%d: date=%s season=%s weather=%s '
                'reward=%.4f energy_kwh=%.4f movement=%.4f\n' % (
                    idx, meta['date'], meta['season'], meta['weather_condition'],
                    meta['total_reward'], meta['total_energy_kwh'],
                    meta['total_movement_cost']))
            f.write(
                '    peak_power=%.1f W @ %.2f h | peak_reward=%.5f @ %.2f h | '
                'mean_power=%.1f W\n' % (
                    meta['peak_power_w'], meta['peak_power_time_hour'],
                    meta['at_peak_reward']['reward'], meta['peak_reward_time_hour'],
                    meta['mean_power_w']))

        if baseline_stats:
            f.write('\nBaseline comparison (same env settings, matched seeds):\n')
            for name, stats in baseline_stats.items():
                f.write('  %s:\n' % name)
                _write_stats_block(f, '    Reward', stats['reward'])
                _write_stats_block(f, '    Energy kWh', stats['total_energy_kwh'])
                _write_stats_block(f, '    Movement cost', stats['total_movement_cost'])

        f.write('\nOutputs:\n')
        f.write('  rollouts/rollout_<n>.csv — full trajectories\n')
        f.write('  rollout_plots/rollout_<n>_combined.png — time-series panels\n')
        f.write('  evaluation_rewards.png, evaluation_by_season.png\n')
        f.write('  reward_time_analysis.txt — peak times and window breakdown\n')

    per_rollout_analysis = [analyze_rollout_path(p) for p in paths]
    payload = {
        'checkpoint': checkpoint_dir,
        'deterministic': deterministic,
        'max_path_length': max_path_length,
        'eval_config': eval_env_params.get('kwargs', {}),
        'warnings': warnings or [],
        'policy': policy_stats,
        'per_rollout': per_rollout_analysis,
        'reward_by_time_window': aggregate_time_windows(paths),
    }
    if season_stats:
        payload['by_season'] = season_stats
    if baseline_stats:
        payload['baselines'] = baseline_stats

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, default=str)

    return summary_path, json_path


def plot_rewards(outdir, paths):
    rewards = [get_rollout_metadata(p)['total_reward'] for p in paths]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(np.arange(1, len(rewards) + 1), rewards, marker='o', linestyle='-', color='#2171b5')
    ax.axhline(np.mean(rewards), color='#636363', linestyle='--', label='mean')
    ax.set_title('Evaluation episode total reward')
    ax.set_xlabel('Rollout index')
    ax.set_ylabel('Total reward')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.4)
    filepath = os.path.join(outdir, 'evaluation_rewards.png')
    fig.tight_layout()
    fig.savefig(filepath, dpi=150)
    plt.close(fig)
    return filepath


def plot_by_season(outdir, paths):
    season_groups = defaultdict(list)
    for path in paths:
        meta = get_rollout_metadata(path)
        season_groups[meta['season']].append(meta['total_reward'])

    seasons = sorted(season_groups.keys())
    means = [np.mean(season_groups[s]) for s in seasons]
    stds = [np.std(season_groups[s]) for s in seasons]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(seasons, means, yerr=stds, capsize=4, color='#41ab5d', alpha=0.85)
    ax.set_title('Mean total reward by season')
    ax.set_ylabel('Total reward')
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    filepath = os.path.join(outdir, 'evaluation_by_season.png')
    fig.tight_layout()
    fig.savefig(filepath, dpi=150)
    plt.close(fig)
    return filepath


def plot_reward_time_windows(outdir, paths):
    env_tz = env_timezone_from_paths(paths)
    window_agg = aggregate_time_windows(paths)
    windows = list(TIME_WINDOWS.keys())
    x = np.arange(len(windows))
    mean_reward = [window_agg.get(w, {}).get('mean_reward', np.nan) for w in windows]
    mean_power = [window_agg.get(w, {}).get('mean_power_w', np.nan) for w in windows]

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(9, 6))
    fig.suptitle('UTC clock-hour windows (tz=%s)' % env_tz, fontsize=11)
    axes[0].bar(x, mean_reward, color='#d62728', alpha=0.85)
    axes[0].set_ylabel('Mean step reward')
    axes[0].set_title('Reward by UTC clock window (avg over rollouts)')
    axes[0].grid(axis='y', linestyle='--', alpha=0.4)

    axes[1].bar(x, mean_power, color='#1f77b4', alpha=0.85)
    axes[1].set_ylabel('Mean power (W)')
    axes[1].set_title('Power by UTC clock window (avg over rollouts)')
    axes[1].set_xlabel('Window (UTC clock hour bands)')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(windows)
    axes[1].grid(axis='y', linestyle='--', alpha=0.4)

    filepath = os.path.join(outdir, 'evaluation_reward_by_time_window.png')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(filepath, dpi=150)
    plt.close(fig)
    return filepath


def plot_solar_altitude_windows(outdir, paths):
    solar_agg = aggregate_solar_altitude_windows(paths)
    windows = list(SOLAR_ALTITUDE_WINDOWS.keys())
    x = np.arange(len(windows))
    mean_reward = [solar_agg.get(w, {}).get('mean_reward', np.nan) for w in windows]
    mean_power = [solar_agg.get(w, {}).get('mean_power_w', np.nan) for w in windows]

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
    fig.suptitle('Solar-altitude windows (physics)', fontsize=11)
    axes[0].bar(x, mean_reward, color='#d62728', alpha=0.85)
    axes[0].set_ylabel('Mean step reward')
    axes[0].set_title('Reward by solar altitude (avg over rollouts)')
    axes[0].grid(axis='y', linestyle='--', alpha=0.4)

    axes[1].bar(x, mean_power, color='#1f77b4', alpha=0.85)
    axes[1].set_ylabel('Mean power (W)')
    axes[1].set_title('Power by solar altitude (avg over rollouts)')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(windows, rotation=12, ha='right')
    axes[1].grid(axis='y', linestyle='--', alpha=0.4)

    filepath = os.path.join(outdir, 'evaluation_reward_by_solar_altitude.png')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(filepath, dpi=150)
    plt.close(fig)
    return filepath


def plot_method_comparison(outdir, paths_by_name):
    rows = compare_method_table(paths_by_name)
    methods = [r['method'] for r in rows]
    x = np.arange(len(methods))
    width = 0.35

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].bar(x, [r['total_reward_mean'] for r in rows], color='#d62728')
    axes[0].set_title('Total reward (mean)')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(methods, rotation=15, ha='right')

    axes[1].bar(x, [r['total_energy_kwh_mean'] for r in rows], color='#1f77b4')
    axes[1].set_title('Total energy kWh (mean)')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(methods, rotation=15, ha='right')

    axes[2].bar(x, [r['movement_cost_mean'] for r in rows], color='#7f7f7f')
    axes[2].set_title('Movement cost (mean)')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(methods, rotation=15, ha='right')

    for ax in axes:
        ax.grid(axis='y', linestyle='--', alpha=0.4)

    filepath = os.path.join(outdir, 'evaluation_method_comparison.png')
    fig.tight_layout()
    fig.savefig(filepath, dpi=150)
    plt.close(fig)
    return filepath


def plot_rollout_combined(outdir, path, idx):
    meta = analyze_rollout_path(path)
    infos = path.get('infos', [])
    x = rollout_time_axis(path, relative=False)
    x_label = rollout_xlabel(path)

    rewards = np.asarray(path['rewards'], dtype=np.float64)
    actions = np.asarray(path['actions'], dtype=np.float64)
    power = np.asarray([info.get('power', np.nan) for info in infos], dtype=np.float64)
    tilt = np.asarray([info.get('tilt', np.nan) for info in infos], dtype=np.float64)
    azimuth = np.asarray([info.get('azimuth', np.nan) for info in infos], dtype=np.float64)
    movement = np.asarray([info.get('movement_cost', 0.0) for info in infos], dtype=np.float64)
    cumulative_energy = np.cumsum(
        [info.get('energy_kwh', 0.0) for info in infos], dtype=np.float64)

    rollout_dir = os.path.join(outdir, 'rollout_plots')
    os.makedirs(rollout_dir, exist_ok=True)

    title = (
        'Rollout %d — %s (%s, %s)\nreward=%.3f  energy=%.3f kWh  movement=%.3f' % (
            idx, meta['date'], meta['season'], meta['weather_condition'],
            meta['total_reward'], meta['total_energy_kwh'], meta['total_movement_cost']))

    fig, axes = plt.subplots(6, 1, sharex=True, figsize=(11, 14))
    fig.suptitle(title, fontsize=11)

    night_spans = night_intervals_from_path(path)
    for lo, hi in night_spans:
        for ax in axes:
            ax.axvspan(lo, hi, color='#e0e0e0', alpha=0.45, zorder=0)

    t_peak_power = meta.get('peak_power_time_hour', np.nan)
    t_peak_reward = meta.get('peak_reward_time_hour', np.nan)

    axes[0].plot(x, power, color='#1f77b4', linewidth=1.5, label='power')
    if np.isfinite(t_peak_power):
        axes[0].axvline(t_peak_power, color='#1f77b4', linestyle=':', alpha=0.8,
                        label='peak power')
    axes[0].set_ylabel('Power (W)')
    axes[0].legend(loc='upper left', fontsize=8)
    if night_spans:
        axes[0].text(
            0.01, 0.95, 'gray = night (solar alt ≤ 0°)',
            transform=axes[0].transAxes, fontsize=7, va='top')

    axes[1].plot(x, cumulative_energy, color='#9467bd', linewidth=1.5)
    axes[1].set_ylabel('Cum. energy (kWh)')

    axes[2].plot(x, tilt, color='#ff7f0e', linewidth=1.5)
    axes[2].set_ylabel('Tilt (deg)')

    axes[3].plot(x, azimuth, color='#2ca02c', linewidth=1.5)
    axes[3].set_ylabel('Azimuth (deg)')

    if actions.ndim == 2 and actions.shape[1] >= 2:
        axes[4].plot(x, actions[:, 0], label='tilt cmd', color='#8c564b')
        axes[4].plot(x, actions[:, 1], label='azimuth cmd', color='#e377c2')
        axes[4].legend(loc='upper right', fontsize=8)
    axes[4].set_ylabel('Action [-1,1]')

    axes[5].plot(x, rewards, color='#d62728', label='reward', linewidth=1.2)
    ax_twin = axes[5].twinx()
    ax_twin.plot(x, movement, color='#7f7f7f', linestyle='--', label='movement', linewidth=1.0)
    if np.isfinite(t_peak_reward):
        axes[5].axvline(t_peak_reward, color='#d62728', linestyle=':', alpha=0.8,
                        label='peak reward')
    axes[5].set_ylabel('Reward')
    ax_twin.set_ylabel('Movement cost')
    axes[5].set_xlabel(x_label)
    axes[5].legend(loc='upper left', fontsize=7)

    for ax in axes:
        ax.grid(True, linestyle='--', alpha=0.35)

    filepath = os.path.join(rollout_dir, 'rollout_%d_combined.png' % idx)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
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
    checkpoint_picklable = load_checkpoint_picklable(checkpoint_path)
    policy_weights = load_policy_weights(checkpoint_path)

    eval_environment, eval_env_params = get_eval_environment(
        variant,
        args.eval_env_override,
        args.test_start_date,
        args.test_end_date,
        args.fixed_eval_dates,
        eval_weather_source=args.eval_weather_source,
        eval_protocol=args.eval_protocol,
    )
    eval_warnings = validate_eval_coverage(
        args.num_rollouts, eval_env_params, min_rollouts=args.min_rollouts)
    for warning in eval_warnings:
        print('[evaluate_agent] WARNING: %s' % warning)
    policy = get_policy(
        variant,
        eval_environment,
        policy_weights,
        checkpoint_picklable=checkpoint_picklable,
    )
    dim_info = validate_policy_environment_observation_dims(
        policy,
        eval_environment,
        policy_weights=policy_weights,
        eval_env_params=eval_env_params,
    )
    print('[evaluate_agent] Verified policy_input_dim=%d env_observation_dim=%d '
          'observation_mode=%r' % (
              dim_info['policy_dim'], dim_info['env_dim'], dim_info['env_mode']))

    path_length = args.max_path_length
    if path_length == 1000:
        path_length = default_max_path_length(variant, path_length)

    paths = []
    baseline_metrics = {}
    with policy.set_deterministic(args.deterministic):
        for idx in range(args.num_rollouts):
            eval_environment.seed(idx)
            path = rollout(
                eval_environment,
                policy,
                path_length=path_length,
                render_mode=args.render_mode)
            paths.append(path)

    rollouts_dir = save_rollout_csv(
        os.path.join(args.outdir, 'rollouts'), paths, prefix='rollout')

    baseline_paths_by_name = {}
    if args.compare_baselines:
        baseline_dir = os.path.join(args.outdir, 'baseline_rollouts')
        for name in args.baseline_types:
            baseline_paths_by_name[name] = []
            for idx in range(args.num_rollouts):
                baseline_env, _ = get_eval_environment(
                    variant,
                    args.eval_env_override,
                    args.test_start_date,
                    args.test_end_date,
                    args.fixed_eval_dates,
                    eval_weather_source=args.eval_weather_source,
                    eval_protocol=args.eval_protocol,
                )
                path = make_baseline_rollout(
                    baseline_env, name, path_length, seed=idx)
                baseline_paths_by_name[name].append(path)
            save_rollout_csv(
                os.path.join(baseline_dir, name),
                baseline_paths_by_name[name],
                prefix='rollout')

    paths_by_name = {'learned_policy': paths}
    if args.compare_baselines:
        for name, bpaths in baseline_paths_by_name.items():
            paths_by_name[name] = bpaths

    summary_path, json_path = save_summary(
        args.outdir,
        checkpoint_path,
        paths,
        deterministic=args.deterministic,
        max_path_length=path_length,
        eval_env_params=eval_env_params,
        baseline_paths_by_name=baseline_paths_by_name if args.compare_baselines else None,
        report_by_season=not args.no_report_by_season,
        warnings=eval_warnings)

    reward_time_report = write_reward_time_report(
        args.outdir, paths, paths_by_name=paths_by_name if args.compare_baselines else None)
    scenario_report = write_eval_scenario_confirmation(
        args.outdir, eval_env_params, paths_by_name, path_length)

    reward_plot = plot_rewards(args.outdir, paths)
    plot_files = [
        reward_plot,
        plot_solar_altitude_windows(args.outdir, paths),
        plot_reward_time_windows(args.outdir, paths),
    ]
    if not args.no_report_by_season and len(paths) > 1:
        plot_files.append(plot_by_season(args.outdir, paths))
    if args.compare_baselines and len(paths_by_name) > 1:
        plot_files.append(plot_method_comparison(args.outdir, paths_by_name))

    rollout_plot_files = []
    for idx, path in enumerate(paths, start=1):
        rollout_plot_files.append(plot_rollout_combined(args.outdir, path, idx))

    print('Evaluation complete.')
    print('Saved:')
    print('  %s' % summary_path)
    print('  %s' % json_path)
    print('  %s' % reward_time_report)
    print('  %s' % scenario_report)
    for p in plot_files:
        print('  %s' % p)
    print('  %s' % rollouts_dir)
    if args.compare_baselines:
        print('  %s' % os.path.join(args.outdir, 'baseline_rollouts'))
    for path in rollout_plot_files:
        print('  %s' % path)


if __name__ == '__main__':
    args = parse_args()
    main(args)
