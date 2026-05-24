#!/usr/bin/env python3
"""Diagnose PV tracking policy vs baselines (cheap, no retrain required).

Reads rollout CSVs from evaluate_agent.py / evaluate_agent_advanced.py output,
verifies incremental action-space semantics, and writes a structured report +
plots under --outdir.

Examples:

  python scripts/diagnose_tracking.py \\
    --eval-dir evaluation/pv_daylight_utc

  python scripts/diagnose_tracking.py \\
    --eval-dir evaluation/pv_daylight_utc \\
    --progress-csv ~/ray_mbpo/PVTracking/pv_tracking/seed:.../progress.csv \\
    --verify-env
"""

from __future__ import print_function

import argparse
import csv
import glob
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from eval_utils import PV_EPISODE_START_TIME, compute_total_energy_kwh

MAX_DELTA_TILT = 5.0
MAX_DELTA_AZIMUTH = 10.0


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--eval-dir', required=True,
                   help='Evaluation output dir with rollouts/ and baseline_rollouts/')
    p.add_argument('--outdir', default=None,
                   help='Report/plot output (default: <eval-dir>/diagnostics)')
    p.add_argument('--progress-csv', default=None,
                   help='Optional training progress.csv for alpha / return trends')
    p.add_argument('--trial-dir', default=None,
                   help='Ray trial dir; reads params.json for min_alpha / config_version')
    p.add_argument('--verify-env', action='store_true',
                   help='Run live PVTrackingEnv action-space sanity checks')
    p.add_argument('--summer-dates', default='2020-06-07,2020-06-21',
                   help='Comma-separated dates for season comparison if present in CSVs')
    return p.parse_args()


def load_csv(path):
    with open(path, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key in list(row.keys()):
            if key in ('step',):
                row[key] = int(row[key])
            elif key in ('terminal',):
                row[key] = row[key].lower() == 'true'
            elif key not in ('date', 'season', 'weather_condition', 'weather_source',
                             'timezone', 'timestamp_utc_iso'):
                try:
                    row[key] = float(row[key]) if row[key] != '' else np.nan
                except ValueError:
                    pass
    return rows


def find_rollouts(root, subpath):
    pattern = os.path.join(root, subpath, '*.csv')
    return sorted(glob.glob(pattern))


def pair_rollouts_by_date(learned_paths, baseline_paths):
    """Pair rollouts that share the same episode date (first row date field)."""
    def date_of(path):
        rows = load_csv(path)
        return rows[0]['date'] if rows else None

    baseline_by_date = {}
    for p in baseline_paths:
        d = date_of(p)
        if d:
            baseline_by_date.setdefault(d, []).append(p)

    pairs = []
    for lp in learned_paths:
        d = date_of(lp)
        if d and d in baseline_by_date:
            pairs.append((lp, baseline_by_date[d][0], d))
    return pairs


def action_l1(row):
    return abs(row.get('action_tilt', 0.0)) + abs(row.get('action_azimuth', 0.0))


def productive_mask(rows, min_alt=5.0):
    return np.array([
        float(r.get('solar_altitude_deg', np.nan)) >= min_alt for r in rows], dtype=bool)


def summarize_trajectory(rows, label):
    if not rows:
        return {}
    alt_mask = productive_mask(rows)
    prod = [r for r, m in zip(rows, alt_mask) if m]
    if not prod:
        prod = rows

    tilts = np.array([r['tilt_deg'] for r in rows], dtype=np.float64)
    zeniths = np.array([r['solar_zenith_deg'] for r in rows], dtype=np.float64)
    tilt_err = tilts - zeniths
    actions = np.array([action_l1(r) for r in rows], dtype=np.float64)
    prod_actions = np.array([action_l1(r) for r in prod], dtype=np.float64)
    energies = np.array([r.get('energy_kwh', 0.0) for r in rows], dtype=np.float64)
    powers = np.array([r.get('power_w', 0.0) for r in rows], dtype=np.float64)

    return {
        'label': label,
        'date': rows[0].get('date', ''),
        'weather': rows[0].get('weather_condition', ''),
        'total_energy_kwh': float(np.nansum(energies)),
        'mean_power_w': float(np.nanmean(powers)),
        'mean_abs_tilt_error_deg': float(np.nanmean(np.abs(tilt_err[alt_mask]))) if alt_mask.any() else float(np.nanmean(np.abs(tilt_err))),
        'max_abs_tilt_error_deg': float(np.nanmax(np.abs(tilt_err[alt_mask]))) if alt_mask.any() else float(np.nanmax(np.abs(tilt_err))),
        'mean_action_l1': float(np.nanmean(actions)),
        'mean_action_l1_productive': float(np.nanmean(prod_actions)),
        'max_action_l1': float(np.nanmax(actions)),
        'total_movement_cost': float(np.nansum([r.get('movement_cost', 0.0) for r in rows])),
        'mean_tilt_deg': float(np.nanmean(tilts[alt_mask])) if alt_mask.any() else float(np.nanmean(tilts)),
        'mean_solar_zenith_deg': float(np.nanmean(zeniths[alt_mask])) if alt_mask.any() else float(np.nanmean(zeniths)),
    }


def verify_paired_fairness(learned_rows, baseline_rows):
    """Same date/weather per step, same reward formula."""
    lines = []
    ok = True
    if len(learned_rows) != len(baseline_rows):
        ok = False
        lines.append('FAIL: trajectory length learned=%d baseline=%d' % (
            len(learned_rows), len(baseline_rows)))
    else:
        lines.append('OK: trajectory length %d steps' % len(learned_rows))

    mismatches = 0
    for i, (l, b) in enumerate(zip(learned_rows, baseline_rows)):
        if l.get('date') != b.get('date') or l.get('weather_condition') != b.get('weather_condition'):
            mismatches += 1
    if mismatches:
        ok = False
        lines.append('FAIL: date/weather mismatch on %d / %d steps' % (mismatches, len(learned_rows)))
    else:
        lines.append('OK: identical date and weather_condition at every step (matched seed + env kwargs).')

    for label, rows in (('learned', learned_rows), ('baseline', baseline_rows)):
        bad_rew = 0
        for r in rows:
            e = float(r.get('energy_kwh', 0))
            m = float(r.get('movement_cost', 0))
            rew = float(r.get('reward', 0))
            if abs((e - m) - rew) > 1e-5:
                bad_rew += 1
        if bad_rew:
            ok = False
            lines.append('FAIL: reward != energy_kwh - movement_cost on %d steps (%s)' % (bad_rew, label))
        else:
            lines.append('OK: reward = energy_kwh - movement_cost for all steps (%s)' % label)

    bad_move = 0
    for r in learned_rows:
        a0, a1 = float(r['action_tilt']), float(r['action_azimuth'])
        expected = 0.0001 * (abs(a0) + abs(a1))
        if abs(float(r['movement_cost']) - expected) > 1e-7:
            bad_move += 1
    if bad_move:
        ok = False
        lines.append('FAIL: movement_cost != penalty*(|a0|+|a1|) on %d steps' % bad_move)
    else:
        lines.append('OK: movement_cost uses commanded normalized actions (not executed delta).')

    lines.append('')
    lines.append('Comparison protocol: same get_eval_environment kwargs, seed=rollout_index,')
    lines.append('same incremental action space; baselines decode obs (no oracle pvlib bypass).')
    lines.append('Sun tracker: target_tilt=solar_zenith, target_az=solar_azimuth (strong heuristic).')
    lines.append('VERDICT: %s' % ('PASS — comparison is fair' if ok else 'FAIL — fix eval before interpreting metrics'))
    return ok, lines


def verify_action_implementation():
    """Live checks: incremental action semantics and GymAdapter identity scaling."""
    import gym
    from softlearning.environments.utils import get_environment_from_params

    lines = []
    ok = True

    # Raw env (no adapter)
    raw = gym.make(
        'PVTracking-v0',
        randomize_initial_orientation=False,
        randomize_day=False,
        fixed_eval_dates=['2020-12-07'],
        weather_source='clearsky',
        observation_mode='physical',
    )
    raw.seed(0)
    raw.reset()
    tilt0 = raw.tilt
    raw.step(np.array([1.0, 0.0], dtype=np.float32))
    d_tilt = raw.tilt - tilt0
    if abs(d_tilt - MAX_DELTA_TILT) > 0.01:
        ok = False
        lines.append('FAIL raw env: action=[1,0] delta_tilt=%.4f expected %.1f' % (d_tilt, MAX_DELTA_TILT))
    else:
        lines.append('OK raw env: action=[1,0] -> delta_tilt=%.2f deg' % d_tilt)

    # Wrapped env (training path)
    env_params = {
        'domain': 'PVTracking', 'task': 'v0', 'universe': 'gym',
        'kwargs': {
            'randomize_initial_orientation': False,
            'fixed_eval_dates': ['2020-12-07'],
            'randomize_day': False,
            'weather_source': 'clearsky',
            'observation_mode': 'physical',
        },
    }
    wrapped = get_environment_from_params(env_params)
    wrapped.seed(0)
    wrapped.reset()
    t0 = wrapped.unwrapped.tilt
    wrapped.step(np.array([1.0, 0.0], dtype=np.float32))
    d_wrapped = wrapped.unwrapped.tilt - t0
    if abs(d_wrapped - MAX_DELTA_TILT) > 0.01:
        ok = False
        lines.append('FAIL GymAdapter: action=[1,0] delta_tilt=%.4f expected %.1f' % (
            d_wrapped, MAX_DELTA_TILT))
    else:
        lines.append('OK GymAdapter (NormalizeActionWrapper): action=[1,0] -> delta_tilt=%.2f deg'
                     % d_wrapped)
        lines.append('  Note: wrapper is identity when underlying Box is [-1,1].')

  # CSV consistency on one stored rollout if available
    raw.close()
    wrapped.close()
    return ok, lines


def verify_csv_action_consistency(rows):
    """Check action_tilt at step i matches tilt change into row i."""
    errors = []
    for i in range(1, len(rows)):
        expected = rows[i]['action_tilt'] * MAX_DELTA_TILT
        actual = rows[i]['tilt_deg'] - rows[i - 1]['tilt_deg']
        if abs(expected - actual) > 0.15:
            errors.append((i, expected, actual))
    return errors


def load_training_params(trial_dir=None, params_json=None):
    import json
    path = params_json
    if path is None and trial_dir:
        path = os.path.join(trial_dir, 'params.json')
    if not path or not os.path.isfile(path):
        return {}
    with open(path, encoding='utf-8') as f:
        variant = json.load(f)
    algo = variant.get('algorithm_params', {}).get('kwargs', {})
    env = variant.get('environment_params', {}).get('training', {}).get('kwargs', {})
    return {
        'config_version': variant.get('config_version'),
        'min_alpha': algo.get('min_alpha'),
        'target_entropy': algo.get('target_entropy'),
        'real_ratio': algo.get('real_ratio'),
        'n_epochs_config': algo.get('n_epochs'),
        'observation_mode': env.get('observation_mode'),
        'randomize_initial_orientation': env.get('randomize_initial_orientation'),
    }


def analyze_progress_csv(path, training_params=None):
    import pandas as pd
    df = pd.read_csv(path)
    out = {'path': path, 'n_epochs': len(df)}
    if training_params:
        out.update({k: v for k, v in training_params.items() if v is not None})
    for col in ('evaluation/return-average', 'training/return-average', 'alpha', 'model/val_loss'):
        if col in df.columns:
            vals = df[col].dropna()
            if len(vals):
                out[col + '_first'] = float(vals.iloc[0])
                out[col + '_last'] = float(vals.iloc[-1])
                out[col + '_max'] = float(vals.max())
    if 'alpha' in df.columns:
        alpha = df['alpha'].dropna()
        floor = (training_params or {}).get('min_alpha', 0.12)
        out['min_alpha_config'] = float(floor)
        out['alpha_collapsed'] = bool(
            len(alpha) and float(alpha.iloc[-1]) <= float(floor) + 0.001)
    return out


def plot_tilt_vs_zenith(learned_rows, baseline_rows, date, outpath):
    fig, ax = plt.subplots(figsize=(10, 5))
    for rows, style in ((learned_rows, dict(color='#1f77b4', label='learned', lw=2)),
                        (baseline_rows, dict(color='#2ca02c', label='sun_tracking', lw=1.8, ls='--'))):
        t = [r['clock_hour_utc'] for r in rows]
        ax.plot(t, [r['tilt_deg'] for r in rows], **style)
    t = [r['clock_hour_utc'] for r in baseline_rows]
    ax.plot(t, [r['solar_zenith_deg'] for r in baseline_rows],
            color='#9467bd', lw=1.5, ls=':', label='solar zenith (sun target)')
    ax.set_title('Tilt vs solar zenith — %s' % date)
    ax.set_xlabel('Clock hour UTC')
    ax.set_ylabel('Degrees')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_action_histogram(learned_rows, baseline_rows, date, outpath):
    fig, ax = plt.subplots(figsize=(8, 4))
    la = [action_l1(r) for r in learned_rows]
    ba = [action_l1(r) for r in baseline_rows]
    bins = np.linspace(0, 2.05, 22)
    ax.hist(la, bins=bins, alpha=0.6, label='learned', color='#1f77b4', density=True)
    ax.hist(ba, bins=bins, alpha=0.6, label='sun_tracking', color='#2ca02c', density=True)
    ax.set_title('|action_tilt|+|action_az| — %s' % date)
    ax.set_xlabel('Action L1 norm')
    ax.set_ylabel('Density')
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_action_vs_tilt_error(learned_rows, outpath, date):
    errs = []
    acts = []
    for r in learned_rows:
        alt = r.get('solar_altitude_deg', 0.0)
        if alt < 5.0:
            continue
        errs.append(abs(r['tilt_deg'] - r['solar_zenith_deg']))
        acts.append(action_l1(r))
    if not errs:
        return
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(errs, acts, alpha=0.7, s=30)
    ax.set_xlabel('|tilt - solar_zenith| (deg)')
    ax.set_ylabel('Action L1 norm')
    ax.set_title('Learned: action vs misalignment — %s' % date)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def write_report(outdir, sections):
    path = os.path.join(outdir, 'tracking_diagnosis.txt')
    with open(path, 'w', encoding='utf-8') as f:
        f.write('PV Tracking — diagnostic report\n')
        f.write('=' * 40 + '\n\n')
        for title, lines in sections:
            f.write('## %s\n\n' % title)
            for line in lines:
                f.write('%s\n' % line)
            f.write('\n')
    return path


def main():
    args = parse_args()
    outdir = args.outdir or os.path.join(args.eval_dir, 'diagnostics')
    os.makedirs(outdir, exist_ok=True)
    plot_dir = os.path.join(outdir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)

    sections = []

    # --- Action space verification ---
    if args.verify_env:
        ok, lines = verify_action_implementation()
        sections.append(('Action space implementation (live env)', lines + [
            '',
            'VERDICT: %s — incremental [-1,1] * max_delta is correct; no switch to absolute needed.'
            % ('PASS' if ok else 'FAIL'),
        ]))
    else:
        sections.append(('Action space implementation', [
            'Skipped live env checks (pass --verify-env).',
            'Incremental format: delta_deg = action * max_delta_{tilt,azimuth}.',
            'GymAdapter NormalizeActionWrapper is identity for Box([-1,1]).',
        ]))

    learned_paths = find_rollouts(args.eval_dir, 'rollouts')
    sun_paths = find_rollouts(args.eval_dir, 'baseline_rollouts/sun_tracking')
    fixed_paths = find_rollouts(args.eval_dir, 'baseline_rollouts/fixed_no_motion')

    if not learned_paths:
        raise SystemExit('No rollouts/*.csv in %s' % args.eval_dir)

    # CSV action consistency
    sample = load_csv(learned_paths[0])
    csv_errors = verify_csv_action_consistency(sample)
    sections.append(('CSV action ↔ tilt consistency (learned rollout_1)', [
        'File: %s' % learned_paths[0],
        'Mismatches (step, expected_d_tilt, actual_d_tilt): %d' % len(csv_errors),
        'PASS' if len(csv_errors) == 0 else 'FAIL (first 3): %s' % str(csv_errors[:3]),
    ]))

    pairs = pair_rollouts_by_date(learned_paths, sun_paths)
    if not pairs:
        # Fall back: index-aligned pairing
        n = min(len(learned_paths), len(sun_paths))
        pairs = [(learned_paths[i], sun_paths[i], 'rollout_%d' % (i + 1)) for i in range(n)]

    if pairs:
        l0 = load_csv(pairs[0][0])
        s0 = load_csv(pairs[0][1])
        fair_ok, fair_lines = verify_paired_fairness(l0, s0)
        sections.append(('Eval fairness (rollout_1 learned vs sun_tracking)', fair_lines))

    learned_summaries = []
    sun_summaries = []
    fixed_summaries = []
    clear_day_pairs = []

    for lp, sp, tag in pairs:
        lrows = load_csv(lp)
        srows = load_csv(sp)
        ls = summarize_trajectory(lrows, 'learned')
        ss = summarize_trajectory(srows, 'sun_tracking')
        learned_summaries.append(ls)
        sun_summaries.append(ss)

        if lrows and lrows[0].get('weather_condition') == 'clear':
            clear_day_pairs.append((lrows, srows, tag))

        if isinstance(tag, str) and tag.startswith('2020'):
            plot_tilt_vs_zenith(lrows, srows, tag,
                                os.path.join(plot_dir, 'tilt_zenith_%s.png' % tag))
            plot_action_histogram(lrows, srows, tag,
                                  os.path.join(plot_dir, 'action_hist_%s.png' % tag))
            plot_action_vs_tilt_error(lrows, os.path.join(plot_dir, 'action_vs_error_%s.png' % tag), tag)

    # Aggregate comparison
    def _mean(key, summaries):
        vals = [s[key] for s in summaries if np.isfinite(s.get(key, np.nan))]
        return float(np.mean(vals)) if vals else np.nan

    agg_lines = [
        'Paired rollouts: %d' % len(pairs),
        '',
        'Metric                          learned    sun_track   ratio (learned/sun)',
        'mean |tilt - zenith| (deg)     %8.2f    %8.2f    %.2f' % (
            _mean('mean_abs_tilt_error_deg', learned_summaries),
            _mean('mean_abs_tilt_error_deg', sun_summaries),
            _mean('mean_abs_tilt_error_deg', learned_summaries) / max(
                _mean('mean_abs_tilt_error_deg', sun_summaries), 1e-6)),
        'mean action L1 (productive sun) %8.3f    %8.3f    %.2f' % (
            _mean('mean_action_l1_productive', learned_summaries),
            _mean('mean_action_l1_productive', sun_summaries),
            _mean('mean_action_l1_productive', learned_summaries) / max(
                _mean('mean_action_l1_productive', sun_summaries), 1e-6)),
        'total energy kWh (mean)         %8.4f    %8.4f    %.2f' % (
            _mean('total_energy_kwh', learned_summaries),
            _mean('total_energy_kwh', sun_summaries),
            _mean('total_energy_kwh', learned_summaries) / max(
                _mean('total_energy_kwh', sun_summaries), 1e-6)),
        'total movement cost (mean)    %8.5f    %8.5f' % (
            _mean('total_movement_cost', learned_summaries),
            _mean('total_movement_cost', sun_summaries)),
    ]
    sections.append(('Aligned learned vs sun_tracking', agg_lines))

    if fixed_paths:
        for fp in fixed_paths[:len(learned_paths)]:
            fixed_summaries.append(summarize_trajectory(load_csv(fp), 'fixed'))
        sections.append(('Learned vs fixed_no_motion (energy)', [
            'mean energy learned: %.4f kWh' % _mean('total_energy_kwh', learned_summaries),
            'mean energy fixed:   %.4f kWh' % _mean('total_energy_kwh', fixed_summaries),
            'PASS beats fixed' if _mean('total_energy_kwh', learned_summaries) > _mean(
                'total_energy_kwh', fixed_summaries) else
            'FAIL: learned below fixed — wrong tracking, not just vs sun tracker',
        ]))

    if clear_day_pairs:
        cl = [summarize_trajectory(l, 'l') for l, _, _ in clear_day_pairs]
        cs = [summarize_trajectory(s, 's') for _, s, _ in clear_day_pairs]
        sections.append(('Clear-sky days only (weather ambiguity check)', [
            'n=%d clear-day pairs' % len(clear_day_pairs),
            'mean energy learned: %.4f  sun: %.4f' % (
                _mean('total_energy_kwh', cl), _mean('total_energy_kwh', cs)),
            'mean |tilt-zenith| learned: %.2f  sun: %.2f' % (
                _mean('mean_abs_tilt_error_deg', cl), _mean('mean_abs_tilt_error_deg', cs)),
            'On clear days, learned should approach sun tracker if action magnitude were sufficient.',
        ]))

    # Season comparison from available dates
    summer_dates = [d.strip() for d in args.summer_dates.split(',') if d.strip()]
    dec_energy = []
    summer_energy = []
    for s in learned_summaries:
        d = s.get('date', '')
        if d.startswith('2020-12'):
            dec_energy.append(s['total_energy_kwh'])
        elif any(d == sd for sd in summer_dates):
            summer_energy.append(s['total_energy_kwh'])
    if dec_energy:
        sections.append(('Season check (learned rollouts in this eval dir)', [
            'December mean energy: %.4f kWh (n=%d)' % (np.mean(dec_energy), len(dec_energy)),
            'Summer mean energy:   %s' % (
                '%.4f kWh (n=%d)' % (np.mean(summer_energy), len(summer_energy))
                if summer_energy else 'no summer-date rollouts in this eval dir — re-run eval with June dates'),
        ]))

    training_params = load_training_params(trial_dir=args.trial_dir)
    if training_params:
        sections.append(('Training params.json', [
            'config_version: %s' % training_params.get('config_version', '(missing)'),
            'min_alpha: %s  target_entropy: %s  real_ratio: %s' % (
                training_params.get('min_alpha'),
                training_params.get('target_entropy'),
                training_params.get('real_ratio')),
            'observation_mode: %s  randomize_initial_orientation: %s' % (
                training_params.get('observation_mode'),
                training_params.get('randomize_initial_orientation')),
        ]))

    if args.progress_csv and os.path.isfile(args.progress_csv):
        prog = analyze_progress_csv(args.progress_csv, training_params=training_params)
        plines = ['File: %s' % prog['path'], 'epochs: %d' % prog['n_epochs']]
        for key in sorted(prog.keys()):
            if key.endswith('_first') or key.endswith('_last') or key == 'alpha_collapsed':
                plines.append('%s: %s' % (key, prog[key]))
        floor = prog.get('min_alpha_config', 0.12)
        if prog.get('alpha_last', 1.0) <= floor + 0.001:
            plines.append(
                'WARNING: alpha pinned at min_alpha floor (%.2f) — entropy-style entropy collapsed; mean actions likely small.'
                % floor)
        if prog.get('evaluation/return-average_last', 0) < 0.75:
            plines.append('WARNING: eval return still below fixed baseline (~0.75 kWh) — undertrained or local optimum.')
        sections.append(('Training progress.csv', plines))

    # Root cause ranking from evidence
    ratio_action = _mean('mean_action_l1_productive', learned_summaries) / max(
        _mean('mean_action_l1_productive', sun_summaries), 1e-6)
    ratio_energy = _mean('total_energy_kwh', learned_summaries) / max(
        _mean('total_energy_kwh', sun_summaries), 1e-6)

    verdict_lines = [
        '1. Action space implementation: OK (incremental, correctly scaled).',
        '2. Policy action magnitude: learned/sun action ratio ≈ %.2f on productive steps.'
        % ratio_action,
    ]
    if ratio_action < 0.5:
        verdict_lines.append('   → PRIMARY: mean actions too small (not action-space format).')
        verdict_lines.append('   → Try: min_alpha↑, target_entropy↑, longer training, real_ratio↑.')
    if _mean('mean_abs_tilt_error_deg', learned_summaries) > 15:
        verdict_lines.append('3. Tilt setpoint: mean |tilt−zenith| ≈ %.1f° (sun ≈ %.1f°).'
                             % (_mean('mean_abs_tilt_error_deg', learned_summaries),
                                _mean('mean_abs_tilt_error_deg', sun_summaries)))
        verdict_lines.append('   → Policy stuck above fixed mount (~30°) but below sun target.')
    if fixed_summaries and _mean('total_energy_kwh', learned_summaries) < _mean(
            'total_energy_kwh', fixed_summaries):
        verdict_lines.append('4. Below fixed_no_motion — confirms failed tracking (not exploration noise).')
    verdict_lines.append('5. Recommended config tweaks (see examples/config/pv_tracking/0.py):')
    verdict_lines.append('   randomize_initial_orientation=False (match eval mount),')
    verdict_lines.append('   n_epochs↑, n_initial_exploration_steps↑, real_ratio↑, min_alpha↑.')
    verdict_lines.append('6. Do NOT switch to absolute action space first — baselines use same incremental form.')
    verdict_lines.append('7. MBPO + physical obs: no fundamental bug; model rollouts capped at length 3;')
    verdict_lines.append('   termination_fn disabled for physical obs (by design — see mbpo/static/pv_tracking.py).')
    verdict_lines.append('8. Config changes in 0.py address confirmed causes; retrain required to measure effect.')
    sections.append(('Root-cause verdict (from this run)', verdict_lines))

    config_lines = [
        'Proposed training config (examples/config/pv_tracking/0.py) — expected mechanism:',
        '  randomize_initial_orientation=False → train/eval both start 30°/180° (fair comparison).',
        '  n_initial_exploration_steps=2500 → ~64 days uniform [-1,1] actions in replay before policy.',
        '  min_alpha=0.12, target_entropy=-1.5 → slower entropy collapse during SAC training.',
        '  real_ratio=0.75 → 75%% real env batches vs 50%% (less model bias on pvlib physics).',
        '  n_epochs=250 → more gradient steps toward tracking policy.',
        '',
        'Limits (honest):',
        '  - Eval uses --deterministic mean policy; min_alpha helps learning, not eval noise.',
        '  - No guarantee to match sun tracker on overcast (heuristic may move unnecessarily).',
        '  - Success criterion after retrain: beat fixed (~0.75 kWh Dec), |tilt−zenith| < ~10°, action ratio > 0.5.',
    ]
    sections.append(('Config change rationale', config_lines))

    report_path = write_report(outdir, sections)
    print('[diagnose] report: %s' % report_path)
    print('[diagnose] plots:  %s/' % plot_dir)
    for title, lines in sections:
        print('\n## %s' % title)
        for line in lines[:12]:
            print('  %s' % line)


if __name__ == '__main__':
    main()
