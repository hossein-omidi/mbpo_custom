#!/usr/bin/env python3
"""Advanced PV evaluation: aligned baselines, ensemble bands, optional weather compare.

Uses the same real-env + policy-only stack as evaluate_agent.py (no MBPO dynamics model).
Reuses eval_utils for environment construction and baselines.

Examples:

  python scripts/evaluate_agent_advanced.py "$CKPT" \\
    --outdir evaluation/pv_daylight_utc_advanced \\
    --fixed-eval-dates 2020-12-07,2020-12-21 \\
    --replicates-per-date 8 \\
    --policy-mode deterministic \\
    --compare-weather-sources

Outputs (under --outdir):
  dashboard.pdf              — multi-page summary
  aligned/aligned_<date>.png — learned vs baselines, same seed/date
  ensemble/ensemble_<date>.png — learned mean ± std over replicates (UTC hour)
  weather_compare/weather_<date>.png — random vs clearsky (learned policy)
  rollouts/, baseline_rollouts/, reports (same as standard eval)
"""

from __future__ import print_function

import argparse
import json
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

try:
    import tensorflow as tf
except ImportError:
    raise SystemExit('tensorflow is required')

from softlearning.samplers import rollout

from eval_utils import (
    EVAL_PROTOCOL_INHERIT,
    get_eval_environment,
    get_rollout_metadata,
    make_baseline_rollout,
    night_intervals_from_path,
    rollout_time_axis,
    rollout_xlabel,
    save_rollout_csv,
    summarize_paths,
    validate_policy_environment_observation_dims,
    write_eval_scenario_confirmation,
    write_reward_time_report,
    compare_method_table,
)

# Reuse checkpoint / policy loading from evaluate_agent.py
from evaluate_agent import (
    default_max_path_length,
    get_policy,
    load_checkpoint_picklable,
    load_policy_weights,
    load_variant,
    resolve_checkpoint_path,
)

METHOD_STYLES = {
    'learned_policy': {'color': '#1f77b4', 'label': 'learned', 'lw': 2.0},
    'sun_tracking': {'color': '#2ca02c', 'label': 'sun tracking', 'lw': 1.8},
    'fixed_no_motion': {'color': '#ff7f0e', 'label': 'fixed', 'lw': 1.8},
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('checkpoint', help='Checkpoint directory (e.g. best_eval_checkpoint)')
    p.add_argument('--outdir', default='evaluation/pv_daylight_utc_advanced')
    p.add_argument('--fixed-eval-dates', required=True,
                   help='Comma-separated YYYY-MM-DD dates for aligned / ensemble plots')
    p.add_argument('--replicates-per-date', type=int, default=8,
                   help='Stochastic replicates per date for mean ± std bands')
    p.add_argument('--align-seed', type=int, default=0,
                   help='Seed for aligned comparison (same for all methods on a date)')
    p.add_argument('--max-path-length', type=int, default=1000)
    p.add_argument('--eval-protocol', default=EVAL_PROTOCOL_INHERIT,
                   choices=(EVAL_PROTOCOL_INHERIT, 'utc', 'legacy_utc'))
    p.add_argument('--policy-mode', choices=('deterministic', 'stochastic'),
                   default='deterministic',
                   help='deterministic: SAC mean action; stochastic: sample actions')
    p.add_argument('--eval-weather-source', choices=('historical', 'clearsky'),
                   default='historical', help='Weather for main rollouts and ensemble')
    p.add_argument('--compare-weather-sources', action='store_true',
                   help='Also run clearsky learned rollouts for overlay plots')
    p.add_argument('--vary-init-orientation', action='store_true',
                   help='Randomize initial panel pose across replicates (more variance)')
    p.add_argument('--baseline-types', nargs='+',
                   default=['fixed_no_motion', 'sun_tracking'])
    p.add_argument('--variant-file', default='params.json')
    p.add_argument('--eval-env-override', default=None)
    p.add_argument('--no-dashboard', action='store_true')
    p.add_argument('--no-csv', action='store_true')
    p.add_argument('--no-reports', action='store_true')
    return p.parse_args()


def parse_dates(date_str):
    return [d.strip() for d in date_str.split(',') if d.strip()]


def series_from_path(path, field):
    infos = path.get('infos', [])
    if field == 'power':
        return np.array([info.get('power', np.nan) for info in infos], dtype=np.float64)
    if field == 'tilt':
        return np.array([info.get('tilt', np.nan) for info in infos], dtype=np.float64)
    if field == 'azimuth':
        return np.array([info.get('azimuth', np.nan) for info in infos], dtype=np.float64)
    if field == 'cumulative_energy':
        e = np.array([info.get('energy_kwh', 0.0) for info in infos], dtype=np.float64)
        return np.cumsum(e)
    raise ValueError('Unknown field: %s' % field)


def stack_series(paths, field):
    """Stack a scalar series from each path; requires equal length."""
    rows = [series_from_path(p, field) for p in paths]
    if not rows:
        return None, None, None
    n = len(rows[0])
    for r in rows:
        if len(r) != n:
            raise ValueError(
                'Paths have different lengths (%d vs %d); use same date/grid/horizon.'
                % (n, len(r)))
    arr = np.vstack(rows)
    return arr.mean(axis=0), arr.std(axis=0), arr


def make_env(variant, args, weather_source, fixed_eval_dates):
    return get_eval_environment(
        variant,
        args.eval_env_override,
        test_start_date=None,
        test_end_date=None,
        fixed_eval_dates=fixed_eval_dates,
        eval_weather_source=weather_source,
        eval_randomize_initial_orientation=args.vary_init_orientation,
        eval_protocol=args.eval_protocol,
    )


def run_learned(env, policy, path_length, seed, policy_stochastic):
    if hasattr(env, 'seed'):
        env.seed(seed)
    with policy.set_deterministic(not policy_stochastic):
        return rollout(env, policy, path_length=path_length)


def run_baseline(env, baseline_type, path_length, seed):
    return make_baseline_rollout(env, baseline_type, path_length, seed=seed)


def run_method_on_date(variant, args, path_length, policy, method, date, seed,
                       weather_source, policy_stochastic=False):
    """Single episode for one method on one calendar date."""
    env, _ = make_env(variant, args, weather_source, date)
    try:
        if method == 'learned_policy':
            return run_learned(env, policy, path_length, seed, policy_stochastic)
        return run_baseline(env, method, path_length, seed)
    finally:
        env.close()


def plot_night_bands(ax, path):
    for lo, hi in night_intervals_from_path(path):
        ax.axvspan(lo, hi, color='#e0e0e0', alpha=0.4, zorder=0)


def plot_aligned_comparison(outpath, paths_by_method, date, seed):
    """Three methods on same axes: power, cumulative energy, tilt."""
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(11, 9))
    x_label = rollout_xlabel(next(iter(paths_by_method.values())))

    ref_path = paths_by_method.get('learned_policy') or next(iter(paths_by_method.values()))
    plot_night_bands(axes[0], ref_path)

    for method, path in paths_by_method.items():
        style = METHOD_STYLES.get(method, {'color': 'gray', 'label': method, 'lw': 1.5})
        t = rollout_time_axis(path)
        axes[0].plot(t, series_from_path(path, 'power'), color=style['color'],
                     lw=style['lw'], label=style['label'])
        axes[1].plot(t, series_from_path(path, 'cumulative_energy'), color=style['color'],
                     lw=style['lw'], label=style['label'])
        axes[2].plot(t, series_from_path(path, 'tilt'), color=style['color'],
                     lw=style['lw'], label=style['label'])

    meta = get_rollout_metadata(ref_path)
    fig.suptitle(
        'Aligned comparison — %s  seed=%d  (%s, %s)\n'
        'Same UTC grid; real pvlib env (no learned dynamics model)' % (
            date, seed, meta.get('season'), meta.get('weather_condition')),
        fontsize=11)

    axes[0].set_ylabel('Power (W)')
    axes[1].set_ylabel('Cum. energy (kWh)')
    axes[2].set_ylabel('Tilt (deg)')
    axes[2].set_xlabel(x_label)
    for ax in axes:
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, linestyle='--', alpha=0.35)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_ensemble_band(outpath, paths, date, field, ylabel, title_suffix):
    if len(paths) < 2:
        return None
    t = rollout_time_axis(paths[0])
    mean, std, _ = stack_series(paths, field)
    plot_night_bands_on_ax = night_intervals_from_path(paths[0])

    fig, ax = plt.subplots(figsize=(11, 4))
    for lo, hi in plot_night_bands_on_ax:
        ax.axvspan(lo, hi, color='#e0e0e0', alpha=0.4, zorder=0)
    ax.plot(t, mean, color=METHOD_STYLES['learned_policy']['color'], lw=2.0,
            label='mean (n=%d)' % len(paths))
    ax.fill_between(t, mean - std, mean + std, color=METHOD_STYLES['learned_policy']['color'],
                    alpha=0.25, label='±1 std')
    ax.set_title('%s — learned policy %s\n%s' % (date, title_suffix, ylabel))
    ax.set_xlabel(rollout_xlabel(paths[0]))
    ax.set_ylabel(ylabel)
    ax.legend(loc='best')
    ax.grid(True, linestyle='--', alpha=0.35)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    return outpath


def plot_weather_compare(outpath, path_random, path_clear, date, seed):
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(11, 7))
    for ax in axes:
        for lo, hi in night_intervals_from_path(path_random):
            ax.axvspan(lo, hi, color='#e0e0e0', alpha=0.4, zorder=0)

    t_r = rollout_time_axis(path_random)
    t_c = rollout_time_axis(path_clear)
    axes[0].plot(t_r, series_from_path(path_random, 'power'), color='#1f77b4', lw=2,
                 label='historical weather')
    axes[0].plot(t_c, series_from_path(path_clear, 'power'), color='#9467bd', lw=2,
                 linestyle='--', label='clearsky')
    axes[1].plot(t_r, series_from_path(path_random, 'cumulative_energy'), color='#1f77b4', lw=2,
                 label='historical weather')
    axes[1].plot(t_c, series_from_path(path_clear, 'cumulative_energy'), color='#9467bd', lw=2,
                 linestyle='--', label='clearsky')

    fig.suptitle(
        'Learned policy — weather comparison — %s seed=%d\n'
        'Solid=historical  Dashed=clearsky (same seed, same date)' % (date, seed),
        fontsize=11)
    axes[0].set_ylabel('Power (W)')
    axes[1].set_ylabel('Cum. energy (kWh)')
    axes[1].set_xlabel(rollout_xlabel(path_random))
    for ax in axes:
        ax.legend(loc='best')
        ax.grid(True, linestyle='--', alpha=0.35)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def add_pdf_page(pdf, image_path):
    if not os.path.isfile(image_path):
        return
    img = plt.imread(image_path)
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.imshow(img)
    ax.axis('off')
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def write_advanced_report(outdir, args, dates, summary_lines):
    path = os.path.join(outdir, 'advanced_evaluation_report.txt')
    with open(path, 'w', encoding='utf-8') as f:
        f.write('PV Tracking — advanced evaluation report\n')
        f.write('=' * 44 + '\n\n')
        f.write('policy_mode: %s\n' % args.policy_mode)
        f.write('eval_weather_source: %s\n' % args.eval_weather_source)
        f.write('replicates_per_date: %d\n' % args.replicates_per_date)
        f.write('align_seed: %d\n' % args.align_seed)
        f.write('vary_init_orientation: %s\n' % args.vary_init_orientation)
        f.write('compare_weather_sources: %s\n' % args.compare_weather_sources)
        f.write('dates: %s\n\n' % ', '.join(dates))
        f.write('Evaluation uses real PVTrackingEnv + trained policy only (no MBPO model).\n\n')
        f.write('Matched method comparisons use aligned same-date rollouts only; ')
        f.write('ensemble rollouts are reported separately.\n\n')
        for line in summary_lines:
            f.write('%s\n' % line)
    return path


def main():
    args = parse_args()
    dates = parse_dates(args.fixed_eval_dates)
    if not dates:
        raise SystemExit('--fixed-eval-dates is required (comma-separated YYYY-MM-DD).')

    os.makedirs(args.outdir, exist_ok=True)
    for sub in ('aligned', 'ensemble', 'weather_compare'):
        os.makedirs(os.path.join(args.outdir, sub), exist_ok=True)

    checkpoint_path = resolve_checkpoint_path(args.checkpoint)
    if os.path.isfile(checkpoint_path):
        checkpoint_path = os.path.dirname(checkpoint_path)
    experiment_root = os.path.dirname(checkpoint_path)

    gpu_options = tf.GPUOptions(allow_growth=True)
    session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    tf.keras.backend.set_session(session)

    variant = load_variant(experiment_root, args.variant_file)
    policy_weights = load_policy_weights(checkpoint_path)
    picklable = load_checkpoint_picklable(checkpoint_path)

    path_length = args.max_path_length
    if path_length == 1000:
        path_length = default_max_path_length(variant, path_length)

    policy_stochastic = args.policy_mode == 'stochastic'

    # Reference env for dim check
    ref_env, eval_env_params = make_env(
        variant, args, args.eval_weather_source, ','.join(dates))
    policy = get_policy(variant, ref_env, policy_weights, checkpoint_picklable=picklable)
    dim_info = validate_policy_environment_observation_dims(
        policy, ref_env, policy_weights=policy_weights, eval_env_params=eval_env_params)
    print('[advanced] policy_dim=%d env_dim=%d mode=%r' % (
        dim_info['policy_dim'], dim_info['env_dim'], dim_info['env_mode']))
    ref_env.close()

    aligned_learned = []
    all_baselines = {b: [] for b in args.baseline_types}
    aligned_paths = {}
    ensemble_by_date = {d: [] for d in dates}
    weather_pairs = {}
    summary_lines = []

    methods_aligned = ['learned_policy'] + list(args.baseline_types)

    for date in dates:
        print('[advanced] date=%s aligned seed=%d' % (date, args.align_seed))
        aligned = {}
        for method in methods_aligned:
            path = run_method_on_date(
                variant, args, path_length, policy, method, date,
                args.align_seed, args.eval_weather_source,
                policy_stochastic=policy_stochastic and method == 'learned_policy')
            aligned[method] = path
            if method == 'learned_policy':
                aligned_learned.append(path)
            else:
                all_baselines[method].append(path)

        aligned_paths[date] = aligned
        apath = os.path.join(args.outdir, 'aligned', 'aligned_%s.png' % date)
        plot_aligned_comparison(apath, aligned, date, args.align_seed)

        if args.compare_weather_sources:
            print('[advanced] date=%s weather random vs clearsky' % date)
            p_rand = run_method_on_date(
                variant, args, path_length, policy, 'learned_policy', date,
                args.align_seed, 'random', policy_stochastic=policy_stochastic)
            p_clear = run_method_on_date(
                variant, args, path_length, policy, 'learned_policy', date,
                args.align_seed, 'clearsky', policy_stochastic=policy_stochastic)
            weather_pairs[date] = (p_rand, p_clear)
            wpath = os.path.join(args.outdir, 'weather_compare', 'weather_%s.png' % date)
            plot_weather_compare(wpath, p_rand, p_clear, date, args.align_seed)

        print('[advanced] date=%s ensemble replicates=%d' % (date, args.replicates_per_date))
        for rep in range(args.replicates_per_date):
            seed = args.align_seed + 1000 * (dates.index(date) + 1) + rep
            path = run_method_on_date(
                variant, args, path_length, policy, 'learned_policy', date,
                seed, args.eval_weather_source, policy_stochastic=policy_stochastic)
            ensemble_by_date[date].append(path)

        stats = summarize_paths(ensemble_by_date[date])
        summary_lines.append(
            '%s: learned energy mean=%.4f std=%.4f kWh (n=%d replicates)' % (
                date,
                stats['total_energy_kwh']['mean'],
                stats['total_energy_kwh']['std'],
                stats['total_energy_kwh']['count'],
            ))

        epath = os.path.join(args.outdir, 'ensemble', 'ensemble_%s_power.png' % date)
        plot_ensemble_band(epath, ensemble_by_date[date], date, 'power', 'Power (W)', 'power')
        epath2 = os.path.join(args.outdir, 'ensemble', 'ensemble_%s_energy.png' % date)
        plot_ensemble_band(
            epath2, ensemble_by_date[date], date, 'cumulative_energy',
            'Cum. energy (kWh)', 'cumulative energy')

    aligned_paths_by_name = {'learned_policy': aligned_learned}
    for b in args.baseline_types:
        aligned_paths_by_name[b] = all_baselines[b]

    if not args.no_csv:
        save_rollout_csv(
            os.path.join(args.outdir, 'rollouts'), aligned_learned, prefix='rollout')
        ensemble_paths = []
        for date in dates:
            ensemble_paths.extend(ensemble_by_date[date])
        save_rollout_csv(
            os.path.join(args.outdir, 'ensemble_rollouts'), ensemble_paths, prefix='rollout')
        base_dir = os.path.join(args.outdir, 'baseline_rollouts')
        for b, paths in all_baselines.items():
            save_rollout_csv(os.path.join(base_dir, b), paths, prefix='rollout')

    if not args.no_reports:
        write_eval_scenario_confirmation(
            args.outdir, eval_env_params, aligned_paths_by_name, path_length)
        write_reward_time_report(
            args.outdir, aligned_learned, paths_by_name=aligned_paths_by_name)
        rows = compare_method_table(aligned_paths_by_name)
        with open(os.path.join(args.outdir, 'method_comparison.json'), 'w') as f:
            json.dump(rows, f, indent=2)
        write_advanced_report(args.outdir, args, dates, summary_lines)

    if not args.no_dashboard:
        pdf_path = os.path.join(args.outdir, 'dashboard.pdf')
        print('[advanced] writing %s' % pdf_path)
        with PdfPages(pdf_path) as pdf:
            for date in dates:
                add_pdf_page(pdf, os.path.join(
                    args.outdir, 'aligned', 'aligned_%s.png' % date))
                add_pdf_page(pdf, os.path.join(
                    args.outdir, 'ensemble', 'ensemble_%s_power.png' % date))
                add_pdf_page(pdf, os.path.join(
                    args.outdir, 'ensemble', 'ensemble_%s_energy.png' % date))
                if date in weather_pairs:
                    add_pdf_page(pdf, os.path.join(
                        args.outdir, 'weather_compare', 'weather_%s.png' % date))

    print('[advanced] complete. Outputs in %s' % args.outdir)
    print('  dashboard: %s' % os.path.join(args.outdir, 'dashboard.pdf'))
    print('  aligned/: %d figures' % len(dates))
    print('  ensemble/: %d dates × 2 metrics' % len(dates))


if __name__ == '__main__':
    main()
