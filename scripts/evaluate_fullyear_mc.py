#!/usr/bin/env python3
"""Monte Carlo full-year evaluation with uncertainty bands and season summaries.

Compares learned policy vs sun_tracking vs fixed_no_motion on the real PVTrackingEnv.

Default (--date-set annual): independent eval seeds over full annual day support (RL protocol).
Optional stress_test / holdout: fixed calendar dates for diagnostic ensemble bands only.

See docs/RL_EVAL_PROTOCOL.md
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

from examples.config.pv_tracking.verified_dates import (
    SEASON_ORDER,
    STAGE3_FINAL_TEST_DATES,
    STAGE3_HOLDOUT_DATES,
    STAGE3_STRESS_TEST_DATES,
    dates_by_season,
    season_of_date,
)
from eval_utils import (
    EVAL_PROTOCOL_INHERIT,
    SEASON_CALENDAR_ORDER,
    compute_total_energy_kwh,
    get_rollout_metadata,
    night_intervals_from_path,
    rollout_time_axis,
    rollout_xlabel,
    save_rollout_csv,
    season_from_calendar_date,
    write_eval_scenario_confirmation,
    write_eval_statistics_readme,
)

from evaluate_agent import (
    default_max_path_length,
    get_policy,
    load_checkpoint_picklable,
    load_policy_weights,
    load_variant,
    plot_rollout_combined,
    resolve_checkpoint_path,
    save_summary,
)
from plot_paper_eval import generate_paper_figures, episode_metrics
from evaluate_agent_advanced import (
    METHOD_STYLES,
    make_env,
    plot_aligned_comparison,
    plot_ensemble_band,
    run_baseline,
    run_learned,
    run_method_on_date,
    validate_policy_environment_observation_dims,
)

METHODS = ('learned_policy', 'sun_tracking', 'fixed_no_motion')
BASELINE_METHODS = ('sun_tracking', 'fixed_no_motion')


def parse_dates_arg(date_str):
    return [d.strip() for d in date_str.split(',') if d.strip()]


def resolve_date_set(name, custom_dates):
    if custom_dates:
        return custom_dates
    if name in ('final_test', 'stress_test'):
        return list(STAGE3_FINAL_TEST_DATES)
    if name in ('holdout',):
        return list(STAGE3_STRESS_TEST_DATES)
    if name == 'annual':
        return None
    raise ValueError('Unknown date-set %r' % name)


def run_method_rollout(variant, args, path_length, policy, method, seed,
                       weather_source=None, policy_stochastic=False):
    """One frozen-policy episode: random day from annual support (RL eval)."""
    env, _ = make_env(variant, args, weather_source, None)
    try:
        if method == 'learned_policy':
            return run_learned(env, policy, path_length, seed, policy_stochastic)
        return run_baseline(env, method, path_length, seed)
    finally:
        env.close()


def _season_key(meta):
    """Calendar season for MC seasonal yield (month buckets)."""
    date = meta.get('date')
    if date not in (None, '', 'unknown'):
        return season_from_calendar_date(date)
    return meta.get('season_calendar', meta.get('season', 'unknown'))


def sample_std(values):
    v = np.asarray(values, dtype=np.float64)
    return float(np.std(v, ddof=1)) if len(v) > 1 else 0.0


def sem(values):
    v = np.asarray(values, dtype=np.float64)
    if len(v) <= 1:
        return 0.0
    return float(np.std(v, ddof=1) / np.sqrt(len(v)))


def ci95(values):
    v = np.asarray(values, dtype=np.float64)
    if len(v) <= 1:
        return 0.0
    return float(1.96 * np.std(v, ddof=1) / np.sqrt(len(v)))


def plot_ensemble_actions(outpath, paths, date):
    if len(paths) < 2:
        return
    actions = np.stack([np.asarray(p['actions']) for p in paths])
    t = rollout_time_axis(paths[0])
    mean = actions.mean(axis=0)
    std = actions.std(axis=0)
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(11, 5))
    for ax, idx, label in zip(axes, (0, 1), ('tilt cmd', 'azimuth cmd')):
        for lo, hi in night_intervals_from_path(paths[0]):
            ax.axvspan(lo, hi, color='#e0e0e0', alpha=0.35, zorder=0)
        ax.plot(t, mean[:, idx], color='#1f77b4', lw=2, label='mean')
        ax.fill_between(t, mean[:, idx] - std[:, idx], mean[:, idx] + std[:, idx],
                        color='#1f77b4', alpha=0.25, label='±1 std')
        ax.set_ylabel(label)
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, linestyle='--', alpha=0.35)
    axes[-1].set_xlabel(rollout_xlabel(paths[0]))
    fig.suptitle('Learned actions — %s (n=%d replicates)' % (date, len(paths)), fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_season_energy_bars(outdir, records, error='sem', title_suffix='', file_tag=''):
    """Grouped bar chart: mean daily energy by calendar season and method."""
    seasons = [s for s in SEASON_CALENDAR_ORDER if any(
        r.get('season_calendar', r.get('season')) == s for r in records)]
    methods = [m for m in METHODS if any(r['method'] == m for r in records)]
    x = np.arange(len(seasons))
    width = 0.8 / max(len(methods), 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    err_fn = sem if error == 'sem' else sample_std

    for i, method in enumerate(methods):
        means = []
        errs = []
        for season in seasons:
            vals = [r['energy_kwh'] for r in records
                    if r.get('season_calendar', r.get('season')) == season
                    and r['method'] == method]
            means.append(float(np.mean(vals)) if vals else np.nan)
            errs.append(err_fn(vals) if vals else 0.0)
        style = METHOD_STYLES.get(method, {'color': 'gray', 'label': method})
        offset = (i - (len(methods) - 1) / 2.0) * width
        ax.bar(x + offset, means, width, yerr=errs, capsize=4,
               color=style['color'], alpha=0.88, label=style.get('label', method))

    ax.set_xticks(x)
    ax.set_xticklabels(seasons)
    ax.set_ylabel('Daily energy yield (kWh)')
    ax.set_title('Energy yield by calendar season and tracker%s\n(error bars = %s across MC episodes)' % (
        title_suffix, error))
    ax.legend(loc='best')
    ax.grid(axis='y', linestyle='--', alpha=0.35)
    tag = ('_%s' % file_tag) if file_tag else ('_%s' % error)
    path = os.path.join(outdir, 'season_energy_yield%s.png' % tag)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_season_ratio_bars(outdir, records, baseline='sun_tracking', error='sem'):
    """Learned / baseline energy ratio by calendar season (paired by seed/replicate)."""
    seasons = [s for s in SEASON_CALENDAR_ORDER if any(
        r.get('season_calendar', r.get('season')) == s for r in records)]
    ratios_by_season = {s: [] for s in seasons}
    err_fn = sem if error == 'sem' else sample_std
    for season in seasons:
        for r in records:
            if r['method'] != 'learned_policy':
                continue
            if r.get('season_calendar', r.get('season')) != season:
                continue
            b = [x for x in records
                 if x['method'] == baseline
                 and x.get('date') == r.get('date')
                 and x.get('replicate', 0) == r.get('replicate', 0)
                 and x.get('seed') == r.get('seed')]
            if b and b[0]['energy_kwh'] > 0:
                ratios_by_season[season].append(r['energy_kwh'] / b[0]['energy_kwh'])

    x = np.arange(len(seasons))
    means = [float(np.mean(ratios_by_season[s])) if ratios_by_season[s] else np.nan
             for s in seasons]
    errs = [err_fn(ratios_by_season[s]) if ratios_by_season[s] else 0.0 for s in seasons]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x, means, yerr=errs, capsize=4, color='#1f77b4', alpha=0.88)
    ax.axhline(1.0, color='#2ca02c', linestyle='--', lw=1.5, label='sun tracker parity')
    ax.set_xticks(x)
    ax.set_xticklabels(seasons)
    ax.set_ylabel('Energy ratio (learned / %s)' % baseline.replace('_', ' '))
    ax.set_title('Relative yield vs %s by season' % baseline.replace('_', ' '))
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.35)
    path = os.path.join(outdir, 'season_learned_vs_%s_ratio.png' % baseline)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def write_mc_report(outdir, args, dates, records):
    path = os.path.join(outdir, 'mc_evaluation_report.txt')
    is_annual = args.date_set == 'annual' or dates == ['annual']
    with open(path, 'w', encoding='utf-8') as f:
        f.write('Stage 3 Monte Carlo evaluation report\n')
        f.write('=' * 40 + '\n\n')
        if is_annual:
            f.write('MODE: MAIN PAPER — annual-scenario MC (random days, matched seeds)\n')
            f.write('date_set: annual\n')
            f.write('num_rollouts: %d\n' % args.num_rollouts)
            f.write('eval_seed_base: %d\n' % args.eval_seed_base)
        else:
            f.write('MODE: DIAGNOSTIC STRESS TEST (fixed calendar dates — NOT main paper eval)\n')
            f.write('date_set: %s\n' % args.date_set)
            f.write('dates (%d): %s\n' % (len(dates), ', '.join(dates)))
            f.write('replicates_per_date: %d\n' % args.replicates_per_date)
        f.write('policy_mode: %s\n' % args.policy_mode)
        f.write('vary_init_orientation: %s\n' % args.vary_init_orientation)
        f.write('eval_protocol: inherit\n\n')
        if not is_annual:
            by_season = dates_by_season(dates)
            f.write('Dates by season:\n')
            for s, ds in by_season.items():
                f.write('  %s: %s\n' % (s, ', '.join(ds)))
        f.write('\nPer method (net energy kWh: mean ± std, n episodes):\n')
        for method in METHODS:
            vals = [r['net_energy_kwh'] for r in records if r['method'] == method]
            if vals:
                f.write('  %s: %.4f ± %.4f (n=%d)\n' % (
                    method, np.mean(vals), np.std(vals, ddof=1) if len(vals) > 1 else 0.0,
                    len(vals)))
        f.write('\nSeason bars use season_calendar (month buckets), not env equinox labels.\n')
        f.write('Paired methods share seed; E[energy] and std use sample statistics (ddof=1).\n')
        f.write('\nAll rollouts: real PVTrackingEnv, pvlib power path, T=78.\n')
    return path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('checkpoint')
    p.add_argument('--outdir', default='evaluation/pv_stage3_fullyear_mc')
    p.add_argument('--date-set', choices=('annual', 'final_test', 'stress_test', 'holdout', 'custom'),
                   default='annual',
                   help='annual=RL seed rollouts over full year; stress_test=fixed calendar panel')
    p.add_argument('--fixed-eval-dates', default=None,
                   help='Override dates (comma-separated). Implies --date-set custom.')
    p.add_argument('--num-rollouts', type=int, default=16,
                   help='Rollouts for --date-set annual (independent eval seeds)')
    p.add_argument('--eval-seed-base', type=int, default=100000,
                   help='Base eval seed (independent of training)')
    p.add_argument('--replicates-per-date', type=int, default=8,
                   help='MC replicates per date for learned policy bands')
    p.add_argument('--align-seed', type=int, default=0,
                   help='Seed for aligned single-run comparison (all methods)')
    p.add_argument('--max-path-length', type=int, default=1000)
    p.add_argument('--eval-protocol', default=EVAL_PROTOCOL_INHERIT)
    p.add_argument('--policy-mode', choices=('deterministic', 'stochastic'),
                   default='deterministic')
    p.add_argument('--vary-init-orientation', action='store_true',
                   help='Randomize initial panel pose across replicates')
    p.add_argument('--error-bars', choices=('sem', 'std'), default='sem')
    p.add_argument('--variant-file', default='params.json')
    p.add_argument('--no-csv', action='store_true')
    p.add_argument('--no-pdf', action='store_true')
    p.add_argument('--run-standard-eval', action='store_true',
                   help='Also write standard evaluate_agent summaries/plots')
    return p.parse_args()


def main():
    args = parse_args()
    args.eval_env_override = None
    args.eval_weather_source = None
    custom = parse_dates_arg(args.fixed_eval_dates) if args.fixed_eval_dates else None
    if custom:
        args.date_set = 'custom'
    dates = resolve_date_set(args.date_set, custom)

    os.makedirs(args.outdir, exist_ok=True)
    for sub in ('aligned', 'ensemble', 'ensemble_actions', 'rollout_plots'):
        os.makedirs(os.path.join(args.outdir, sub), exist_ok=True)

    checkpoint_path = resolve_checkpoint_path(args.checkpoint)
    if os.path.isfile(checkpoint_path):
        checkpoint_path = os.path.dirname(checkpoint_path)
    experiment_root = os.path.dirname(checkpoint_path)

    gpu_options = tf.GPUOptions(allow_growth=True)
    session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    tf.keras.backend.set_session(session)

    variant = load_variant(experiment_root, args.variant_file)
    path_length = args.max_path_length
    if path_length == 1000:
        path_length = default_max_path_length(variant, path_length)

    policy_stochastic = args.policy_mode == 'stochastic'
    weather_source = None  # inherit via eval_protocol

    ref_env, eval_env_params = make_env(
        variant, args, weather_source,
        ','.join(dates) if dates else None)
    policy_weights = load_policy_weights(checkpoint_path)
    picklable = load_checkpoint_picklable(checkpoint_path)
    policy = get_policy(variant, ref_env, policy_weights, checkpoint_picklable=picklable)
    validate_policy_environment_observation_dims(
        policy, ref_env, policy_weights=policy_weights, eval_env_params=eval_env_params)
    ref_env.close()

    records = []
    aligned_by_method = {m: [] for m in METHODS}

    if args.date_set == 'annual':
        for idx in range(args.num_rollouts):
            seed = int(args.eval_seed_base) + idx
            print('[mc] annual rollout %d/%d seed=%d' % (idx + 1, args.num_rollouts, seed))
            for method in METHODS:
                path = run_method_rollout(
                    variant, args, path_length, policy, method, seed,
                    weather_source, policy_stochastic=(
                        policy_stochastic and method == 'learned_policy'))
                meta = get_rollout_metadata(path)
                em = episode_metrics(path)
                sk = _season_key(meta)
                records.append({
                    'date': meta.get('date'), 'season': meta.get('season'),
                    'season_calendar': sk,
                    'method': method, 'replicate': idx + 1, 'seed': seed,
                    'energy_kwh': meta['total_energy_kwh'],
                    'gross_energy_kwh': em['gross_energy_kwh'],
                    'movement_cost': em['movement_cost'],
                    'net_energy_kwh': em['net_energy_kwh'],
                    'reward': meta['total_reward'],
                    'weather': meta.get('weather_condition'),
                })
                aligned_by_method[method].append(path)
        rep_records = records
    else:
        if not dates:
            raise SystemExit('No evaluation dates resolved.')
        ensemble_by_date = {d: [] for d in dates}
        for date in dates:
            season = season_of_date(date)
            print('[mc] stress date=%s season=%s seed=%d' % (date, season, args.align_seed))
            aligned = {}
            for method in METHODS:
                path = run_method_on_date(
                    variant, args, path_length, policy, method, date,
                    args.align_seed, weather_source,
                    policy_stochastic=policy_stochastic and method == 'learned_policy')
                aligned[method] = path
                meta = get_rollout_metadata(path)
                records.append({
                    'date': date, 'season': season, 'method': method,
                    'replicate': 0, 'seed': args.align_seed,
                    'energy_kwh': meta['total_energy_kwh'],
                    'reward': meta['total_reward'],
                    'weather': meta.get('weather_condition'),
                })
                aligned_by_method[method].append(path)
            plot_aligned_comparison(
                os.path.join(args.outdir, 'aligned', 'aligned_%s.png' % date),
                aligned, date, args.align_seed)
            if args.run_standard_eval and date == dates[0]:
                plot_rollout_combined(args.outdir, aligned['learned_policy'], 1)
            for rep in range(args.replicates_per_date):
                seed = args.align_seed + 1000 * (dates.index(date) + 1) + rep
                path = run_method_on_date(
                    variant, args, path_length, policy, 'learned_policy', date,
                    seed, weather_source, policy_stochastic=policy_stochastic)
                ensemble_by_date[date].append(path)
                meta = get_rollout_metadata(path)
                records.append({
                    'date': date, 'season': season, 'method': 'learned_policy',
                    'replicate': rep + 1, 'seed': seed,
                    'energy_kwh': meta['total_energy_kwh'],
                    'reward': meta['total_reward'],
                    'weather': meta.get('weather_condition'),
                })
                for bmethod in BASELINE_METHODS:
                    bpath = run_method_on_date(
                        variant, args, path_length, policy, bmethod, date,
                        seed, weather_source, policy_stochastic=False)
                    bmeta = get_rollout_metadata(bpath)
                    records.append({
                        'date': date, 'season': season, 'method': bmethod,
                        'replicate': rep + 1, 'seed': seed,
                        'energy_kwh': bmeta['total_energy_kwh'],
                        'reward': bmeta['total_reward'],
                        'weather': bmeta.get('weather_condition'),
                    })
            plot_ensemble_band(
                os.path.join(args.outdir, 'ensemble', 'ensemble_%s_power.png' % date),
                ensemble_by_date[date], date, 'power', 'Power (W)', 'power')
            plot_ensemble_band(
                os.path.join(args.outdir, 'ensemble', 'ensemble_%s_energy.png' % date),
                ensemble_by_date[date], date, 'cumulative_energy',
                'Cumulative energy (kWh)', 'cumulative energy')
            plot_ensemble_band(
                os.path.join(args.outdir, 'ensemble', 'ensemble_%s_tilt.png' % date),
                ensemble_by_date[date], date, 'tilt', 'Panel tilt (deg)', 'tilt')
            plot_ensemble_actions(
                os.path.join(args.outdir, 'ensemble_actions', 'actions_%s.png' % date),
                ensemble_by_date[date], date)
        rep_records = [r for r in records if r.get('replicate', 0) > 0]

    # Season summary plots (use replicate-level records for uncertainty)
    plot_season_energy_bars(args.outdir, rep_records, error=args.error_bars,
                            title_suffix=' (MC replicates)', file_tag='mc_%s' % args.error_bars)
    plot_season_energy_bars(
        args.outdir,
        [r for r in records if r.get('replicate', 0) == 0],
        error='std', title_suffix=' (aligned single run)', file_tag='aligned')
    plot_season_ratio_bars(args.outdir, rep_records, baseline='sun_tracking',
                           error=args.error_bars)

    if not args.no_csv:
        save_rollout_csv(
            os.path.join(args.outdir, 'rollouts'),
            aligned_by_method['learned_policy'], prefix='aligned_learned')
        for bmethod in BASELINE_METHODS:
            save_rollout_csv(
                os.path.join(args.outdir, 'baseline_rollouts', bmethod),
                aligned_by_method[bmethod], prefix='aligned')

    write_mc_report(args.outdir, args, dates or ['annual'], records)
    if args.date_set == 'annual':
        write_eval_statistics_readme(args.outdir, args.num_rollouts, args.eval_seed_base)
    with open(os.path.join(args.outdir, 'mc_records.json'), 'w') as f:
        json.dump(records, f, indent=2)

    paths_by_name = {
        'learned_policy': aligned_by_method['learned_policy'],
        'sun_tracking': aligned_by_method['sun_tracking'],
        'fixed_no_motion': aligned_by_method['fixed_no_motion'],
    }
    write_eval_scenario_confirmation(
        args.outdir, eval_env_params, paths_by_name, path_length)

    is_stress = args.date_set in ('stress_test', 'holdout', 'final_test', 'custom')
    generate_paper_figures(
        args.outdir,
        paths_by_name,
        eval_env_params,
        error='std' if args.date_set == 'annual' else args.error_bars,
        trial_dir=experiment_root,
        is_stress=is_stress,
    )

    if args.run_standard_eval:
        save_summary(
            args.outdir, checkpoint_path,
            aligned_by_method['learned_policy'],
            deterministic=not policy_stochastic,
            max_path_length=path_length,
            eval_env_params=eval_env_params,
            baseline_paths_by_name={
                'sun_tracking': aligned_by_method['sun_tracking'],
                'fixed_no_motion': aligned_by_method['fixed_no_motion'],
            },
            report_by_season=True)

    if not args.no_pdf:
        pdf_path = os.path.join(args.outdir, 'mc_dashboard.pdf')
        with PdfPages(pdf_path) as pdf:
            for fig_path in sorted([
                os.path.join(args.outdir, 'season_energy_yield_mc_%s.png' % args.error_bars),
                os.path.join(args.outdir, 'season_learned_vs_sun_tracking_ratio.png'),
            ]):
                if os.path.isfile(fig_path):
                    img = plt.imread(fig_path)
                    fig, ax = plt.subplots(figsize=(11, 8.5))
                    ax.imshow(img)
                    ax.axis('off')
                    pdf.savefig(fig, bbox_inches='tight')
                    plt.close(fig)
            if dates and dates != ['annual']:
                for date in dates[:4]:
                    for sub, prefix in (('aligned', 'aligned'), ('ensemble', 'ensemble')):
                        fp = os.path.join(args.outdir, sub, '%s_%s_power.png' % (prefix, date))
                        if os.path.isfile(fp):
                            img = plt.imread(fp)
                            fig, ax = plt.subplots(figsize=(11, 8.5))
                            ax.imshow(img)
                            ax.axis('off')
                            pdf.savefig(fig, bbox_inches='tight')
                            plt.close(fig)
        print('[mc] dashboard:', pdf_path)

    print('[mc] complete → %s' % args.outdir)


if __name__ == '__main__':
    main()
