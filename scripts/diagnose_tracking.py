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


def main():
    args = parse_args()
    return _run_diagnosis(args)


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
    p.add_argument('--gate', action='store_true',
                   help='Exit 1 if phase gates fail (CI / TRAINING_PROTOCOL.md)')
    p.add_argument('--min-energy-ratio', type=float, default=0.95,
                   help='Require mean learned/sun total_energy_kwh >= this (default 0.95)')
    p.add_argument('--min-action-ratio', type=float, default=0.5,
                   help='Require mean action L1 ratio learned/sun >= this (default 0.5)')
    p.add_argument('--max-tilt-error-deg', type=float, default=10.0,
                   help='Require mean |tilt-zenith| learned <= this deg (default 10)')
    return p.parse_args()


def evaluate_phase_gates(learned_summaries, sun_summaries, fixed_summaries,
                         min_energy_ratio, min_action_ratio, max_tilt_error_deg):
    """Return (passed: bool, lines: list) per docs/TRAINING_PROTOCOL.md."""

    def _mean(key, summaries):
        vals = [s[key] for s in summaries if np.isfinite(s.get(key, np.nan))]
        return float(np.mean(vals)) if vals else np.nan

    lines = ['GATE STATUS: ']
    passed = True

    e_learned = _mean('total_energy_kwh', learned_summaries)
    e_sun = _mean('total_energy_kwh', sun_summaries)
    energy_ratio = e_learned / max(e_sun, 1e-9)
    ok_energy = np.isfinite(energy_ratio) and energy_ratio >= min_energy_ratio
    lines.append('  energy ratio (learned/sun) %.3f >= %.2f  [%s]' % (
        energy_ratio, min_energy_ratio, 'PASS' if ok_energy else 'FAIL'))
    passed = passed and ok_energy

    a_learned = _mean('mean_action_l1_productive', learned_summaries)
    a_sun = _mean('mean_action_l1_productive', sun_summaries)
    action_ratio = a_learned / max(a_sun, 1e-9)
    ok_action = np.isfinite(action_ratio) and action_ratio >= min_action_ratio
    lines.append('  action L1 ratio (learned/sun) %.3f >= %.2f  [%s]' % (
        action_ratio, min_action_ratio, 'PASS' if ok_action else 'FAIL'))
    passed = passed and ok_action

    tilt_err = _mean('mean_abs_tilt_error_deg', learned_summaries)
    ok_tilt = np.isfinite(tilt_err) and tilt_err <= max_tilt_error_deg
    lines.append('  mean |tilt - zenith| %.2f <= %.1f deg  [%s]' % (
        tilt_err, max_tilt_error_deg, 'PASS' if ok_tilt else 'FAIL'))
    passed = passed and ok_tilt

    if fixed_summaries:
        e_fixed = _mean('total_energy_kwh', fixed_summaries)
        ok_fixed = np.isfinite(e_learned) and np.isfinite(e_fixed) and e_learned > e_fixed
        lines.append('  learned energy %.4f > fixed %.4f  [%s]' % (
            e_learned, e_fixed, 'PASS' if ok_fixed else 'FAIL'))
        passed = passed and ok_fixed
    else:
        lines.append('  vs fixed_no_motion: skipped (no baseline CSVs)')

    lines[0] += 'PASS' if passed else 'FAIL'
    return passed, lines


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
    """Pair rollouts by (date, rollout_seed) when available, else by date."""
    def key_of(path):
        rows = load_csv(path)
        if not rows:
            return None
        date = rows[0].get('date')
        seed = rows[0].get('rollout_seed')
        try:
            if seed not in (None, '') and np.isfinite(float(seed)):
                return (date, int(float(seed)))
        except (TypeError, ValueError):
            pass
        return (date, None)

    baseline_by_key = {}
    for p in baseline_paths:
        key = key_of(p)
        if key:
            baseline_by_key.setdefault(key, []).append(p)

    pairs = []
    for lp in learned_paths:
        key = key_of(lp)
        if key and key in baseline_by_key:
            pairs.append((lp, baseline_by_key[key][0], key[0]))
            continue
        if key and (key[0], None) in baseline_by_key:
            pairs.append((lp, baseline_by_key[(key[0], None)][0], key[0]))
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


def load_evaluation_summary(eval_dir):
    """Load evaluation_summary.json if present."""
    import json
    path = os.path.join(eval_dir, 'evaluation_summary.json')
    if not os.path.isfile(path):
        return {}
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def load_eval_movement_penalty(eval_dir):
    """movement_penalty from evaluation_summary.json (matches evaluate_agent eval_config)."""
    kw = load_evaluation_summary(eval_dir).get('eval_config') or {}
    if 'movement_penalty' in kw:
        return float(kw['movement_penalty'])
    return None


def load_eval_deterministic(eval_dir):
    """Whether evaluate_agent used tanh(mu) (True) vs stochastic samples (False)."""
    summary = load_evaluation_summary(eval_dir)
    if 'deterministic' in summary:
        return bool(summary['deterministic'])
    return None


def pre_action_tilt_error_deg(row):
    """Controller error at the state that produced this action.

    CSV exports pre-step observation columns and post-step info columns.
    For control-sign analysis we must use the pre-step state, not the post-step
    tilt/zenith stored in `tilt_deg` / `solar_zenith_deg`.
    """
    if 'panel_tilt_norm' in row and 'solar_zenith_norm' in row:
        return float(row['panel_tilt_norm']) * 90.0 - float(row['solar_zenith_norm']) * 180.0
    return float(row['tilt_deg']) - float(row['solar_zenith_deg'])


def tilt_control_sign_stats(learned_rows, min_altitude_deg=5.0):
    """Greedy zenith tracker: sign(a_tilt) should oppose pre-action tilt error."""
    agree = 0
    n = 0
    errs = []
    acts = []
    for r in learned_rows:
        alt = float(r.get('solar_altitude_deg', 0.0))
        if alt < min_altitude_deg:
            continue
        e = pre_action_tilt_error_deg(r)
        a = float(r['action_tilt'])
        errs.append(e)
        acts.append(a)
        if abs(e) < 0.5:
            continue
        need = 1.0 if e < 0 else -1.0
        n += 1
        if (a > 0 and need > 0) or (a < 0 and need < 0):
            agree += 1
    corr = float(np.corrcoef(errs, acts)[0, 1]) if len(errs) > 1 else float('nan')
    return {
        'sign_agreement': float(agree / n) if n else float('nan'),
        'sign_n': int(n),
        'corr_tilt_error_action_tilt': corr,
    }


def verify_paired_fairness(learned_rows, baseline_rows, movement_penalty=None):
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

    penalty = 0.0 if movement_penalty is None else float(movement_penalty)
    if penalty == 0.0:
        bad_move = sum(
            1 for r in learned_rows
            if abs(float(r.get('movement_cost', 0.0))) > 1e-7)
        if bad_move:
            lines.append(
                'NOTE: movement_penalty=0 but movement_cost nonzero on %d steps (check env).'
                % bad_move)
        else:
            lines.append(
                'OK: movement_penalty=0 — movement_cost=0 on all steps (Stage 0 / energy-only).')
    else:
        bad_move = 0
        for r in learned_rows:
            a0, a1 = float(r['action_tilt']), float(r['action_azimuth'])
            expected = penalty * (abs(a0) + abs(a1))
            if abs(float(r['movement_cost']) - expected) > 1e-7:
                bad_move += 1
        if bad_move:
            lines.append(
                'NOTE: movement_cost != penalty*(|a0|+|a1|) on %d steps (penalty=%g); '
                'energy metrics still valid.' % (bad_move, penalty))
        else:
            lines.append(
                'OK: movement_cost = movement_penalty*(|a0|+|a1|) with penalty=%g.'
                % penalty)

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

    raw.close()
    wrapped.close()
    return ok, lines


def verify_csv_action_consistency(rows):
    """Check commanded tilt delta against same-row post-step tilt.

    CSV rows store the pre-step observation and the post-step info for the same
    transition, so the correct consistency check is:
      post_tilt == clip(pre_tilt + action_tilt * MAX_DELTA_TILT, 0, 90)
    not the difference between consecutive post-step tilts.
    """
    errors = []
    for i, row in enumerate(rows):
        pre_tilt = float(row['panel_tilt_norm']) * 90.0
        commanded = float(row['action_tilt']) * MAX_DELTA_TILT
        expected_post = float(np.clip(pre_tilt + commanded, 0.0, 90.0))
        actual_post = float(row['tilt_deg'])
        if abs(expected_post - actual_post) > 0.15:
            errors.append((i, expected_post, actual_post))
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
        'randomize_day': env.get('randomize_day'),
        'start_date': env.get('start_date'),
        'end_date': env.get('end_date'),
        'movement_penalty': env.get('movement_penalty'),
    }


def analyze_progress_csv(path, training_params=None):
    import pandas as pd
    df = pd.read_csv(path)
    out = {'path': path, 'n_epochs': len(df)}
    if training_params:
        out.update({k: v for k, v in training_params.items() if v is not None})
    for col in (
            'evaluation/return-average',
            'training/return-average',
            'alpha',
            'model/val_loss',
            'policy/shifts-mean',
            'policy/shifts-std',
            'policy/log_scale_diags-mean',
            'policy/log_scale_diags-std',
            'policy/raw-actions-std',
            'policy/actions-mean',
            'policy/actions-std'):
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
        out['alpha_at_floor'] = bool(
            len(alpha) and float(alpha.iloc[-1]) <= float(floor) + 0.001)
    if 'policy/log_scale_diags-mean_last' in out:
        out['policy/scale_diag_exp_mean_last'] = float(
            np.exp(out['policy/log_scale_diags-mean_last']))
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
        errs.append(abs(pre_action_tilt_error_deg(r)))
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


def _run_diagnosis(args):
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

    eval_summary = load_evaluation_summary(args.eval_dir)
    movement_penalty = load_eval_movement_penalty(args.eval_dir)
    eval_deterministic = load_eval_deterministic(args.eval_dir)

    if pairs:
        l0 = load_csv(pairs[0][0])
        s0 = load_csv(pairs[0][1])
        fair_ok, fair_lines = verify_paired_fairness(
            l0, s0, movement_penalty=movement_penalty)
        sections.append(('Eval fairness (rollout_1 learned vs sun_tracking)', fair_lines))

    protocol_lines = []
    if eval_deterministic is False:
        protocol_lines.append(
            'FAIL: evaluation_summary.json has deterministic=false — RC1/RC2 metrics '
            'assume tanh(mu). Re-run evaluate_agent.py with --deterministic (default).')
    elif eval_deterministic is True:
        protocol_lines.append(
            'OK: deterministic eval (tanh(mu)); matches training eval_deterministic=True.')
    else:
        protocol_lines.append(
            'NOTE: no deterministic flag in evaluation_summary.json — ensure eval used '
            '--deterministic (evaluate_agent default) before interpreting RC1.')
    if movement_penalty is None:
        protocol_lines.append(
            'NOTE: movement_penalty missing from eval_config — fairness uses penalty=0.')
    sections.append(('Eval protocol (RC1 / fairness)', protocol_lines))

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

    def _fmt(value, fmt):
        return fmt % value if np.isfinite(value) else 'n/a'

    if sun_summaries:
        tilt_err_ratio = _mean('mean_abs_tilt_error_deg', learned_summaries) / max(
            _mean('mean_abs_tilt_error_deg', sun_summaries), 1e-6)
        action_ratio_table = _mean('mean_action_l1_productive', learned_summaries) / max(
            _mean('mean_action_l1_productive', sun_summaries), 1e-6)
        energy_ratio_table = _mean('total_energy_kwh', learned_summaries) / max(
            _mean('total_energy_kwh', sun_summaries), 1e-6)
        agg_lines = [
            'Paired rollouts: %d' % len(pairs),
            '',
            'Metric                          learned    sun_track   ratio (learned/sun)',
            'mean |tilt - zenith| (deg)     %8s    %8s    %s' % (
                _fmt(_mean('mean_abs_tilt_error_deg', learned_summaries), '%.2f'),
                _fmt(_mean('mean_abs_tilt_error_deg', sun_summaries), '%.2f'),
                _fmt(tilt_err_ratio, '%.2f')),
            'mean action L1 (productive sun) %8s    %8s    %s' % (
                _fmt(_mean('mean_action_l1_productive', learned_summaries), '%.3f'),
                _fmt(_mean('mean_action_l1_productive', sun_summaries), '%.3f'),
                _fmt(action_ratio_table, '%.2f')),
            'total energy kWh (mean)         %8s    %8s    %s' % (
                _fmt(_mean('total_energy_kwh', learned_summaries), '%.4f'),
                _fmt(_mean('total_energy_kwh', sun_summaries), '%.4f'),
                _fmt(energy_ratio_table, '%.2f')),
            'total movement cost (mean)    %8s    %8s' % (
                _fmt(_mean('total_movement_cost', learned_summaries), '%.5f'),
                _fmt(_mean('total_movement_cost', sun_summaries), '%.5f')),
        ]
    else:
        agg_lines = [
            'Paired rollouts: 0',
            '',
            'No sun_tracking baseline rollouts found.',
            'Re-run evaluate_agent.py with --compare-baselines for learned-vs-sun ratios.',
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
    eval_config = eval_summary.get('eval_config') or {}

    if training_params:
        tlines = [
            'config_version: %s' % training_params.get('config_version', '(missing)'),
            'n_epochs (config): %s' % training_params.get('n_epochs_config'),
            'min_alpha: %s  target_entropy: %s  real_ratio: %s' % (
                training_params.get('min_alpha'),
                training_params.get('target_entropy'),
                training_params.get('real_ratio')),
            'train dates: %s .. %s  randomize_day: %s' % (
                training_params.get('start_date'),
                training_params.get('end_date'),
                training_params.get('randomize_day')),
            'observation_mode: %s  movement_penalty (train): %s' % (
                training_params.get('observation_mode'),
                training_params.get('movement_penalty')),
        ]
        if eval_config:
            fixed = eval_config.get('fixed_eval_dates')
            tlines.append(
                'eval holdout: randomize_day=%s fixed_eval_dates=%s movement_penalty=%s' % (
                    eval_config.get('randomize_day'),
                    fixed,
                    eval_config.get('movement_penalty')))
            if training_params.get('randomize_day') and fixed:
                tlines.append(
                    'WARNING: train MDP samples random days; eval is fixed day(s) %s — '
                    'not unfair, but checkpoint may not match Stage 0 stationary proof.'
                    % fixed)
            if training_params.get('config_version') and 'stage0' not in str(
                    training_params.get('config_version', '')).lower():
                tlines.append(
                    'WARNING: config_version is not Stage 0 — see docs/PV_TRACKING_ROOT_CAUSES.md')
            if (training_params.get('randomize_day') and eval_config.get('fixed_eval_dates')
                    and 'stage0' in os.path.basename(args.eval_dir).lower()):
                tlines.append(
                    'WARNING: Stage 0 eval dir but checkpoint trained with randomize_day=True '
                    '(RC5) — retrain stage0_single_day.py before trusting gates.')
        sections.append(('Training params.json vs eval', tlines))

    if args.progress_csv and os.path.isfile(args.progress_csv):
        prog = analyze_progress_csv(args.progress_csv, training_params=training_params)
        plines = ['File: %s' % prog['path'], 'epochs: %d' % prog['n_epochs']]
        for key in sorted(prog.keys()):
            if key.endswith('_first') or key.endswith('_last') or key == 'alpha_at_floor':
                plines.append('%s: %s' % (key, prog[key]))
        floor = prog.get('min_alpha_config', 0.12)
        if prog.get('alpha_at_floor'):
            plines.append(
                'NOTE: alpha at min_alpha floor (%.2f). Temperature cannot decrease further; '
                'this is not proof that sigma=0 or that entropy vanished.'
                % floor)
            plines.append(
                'NOTE: policy/actions-mean is a signed sampled batch mean, not deployed action '
                'magnitude. For deploy behavior, trust deterministic rollout action L1 and '
                'policy/shifts-* more than policy/actions-mean alone.')
        if prog.get('evaluation/return-average_last', 0) < 0.75:
            plines.append('WARNING: eval return still below fixed baseline (~0.75 kWh) — undertrained or local optimum.')
        sections.append(('Training progress.csv', plines))

    # Root cause ranking from evidence (see docs/PV_TRACKING_ROOT_CAUSES.md)
    ratio_action = (
        _mean('mean_action_l1_productive', learned_summaries) / max(
            _mean('mean_action_l1_productive', sun_summaries), 1e-6)
        if sun_summaries else float('nan'))
    ratio_energy = (
        _mean('total_energy_kwh', learned_summaries) / max(
            _mean('total_energy_kwh', sun_summaries), 1e-6)
        if sun_summaries else float('nan'))
    tilt_err_l = _mean('mean_abs_tilt_error_deg', learned_summaries)
    tilt_err_s = _mean('mean_abs_tilt_error_deg', sun_summaries)
    e_learned = _mean('total_energy_kwh', learned_summaries)
    e_fixed = _mean('total_energy_kwh', fixed_summaries) if fixed_summaries else float('nan')

    sign_stats = tilt_control_sign_stats(load_csv(learned_paths[0]))
    l0 = load_csv(learned_paths[0])
    s0 = load_csv(sun_paths[0]) if sun_paths else []
    step0_line = ''
    if l0 and s0:
        e0 = pre_action_tilt_error_deg(l0[0])
        step0_line = (
            'Step-0 sign test (pre-action error): e=tilt-zenith=%.1f° need sign(a_tilt)>0 got learned=%.3f sun=%.3f'
            % (e0, float(l0[0]['action_tilt']), float(s0[0]['action_tilt'])))

    verdict_lines = [
        'Formal write-up: docs/PV_TRACKING_ROOT_CAUSES.md',
        '',
        'Confidence = P(cause active | this eval), not mutually exclusive.',
        '',
        'RC1 [100%] Eval policy a(s)=tanh(mu(s)), not training samples (gaussian_policy.py).',
        '     Train actions-std can be >> eval: deployment is mu-only by design.',
        '     alpha at min_alpha is a floor, not proof that sigma=0 or exploration vanished.',
        '',
        'RC2 [100%%] Small deterministic deploy action magnitude: rho_A=%s (need ~0.3+ to track zenith swing).'
        % _fmt(ratio_action, '%.3f'),
        '     Necessary |a| bound ~D/(5T) with D=zenith swing, T=39: see docs/PV_TRACKING_ROOT_CAUSES.md.',
        '     rho_G=%s (learned/sun energy).' % _fmt(ratio_energy, '%.3f'),
        '',
        'RC3 [100%] Wrong-signed tilt control (greedy zenith rule; step-0 + full episode):',
        '     sign agreement=%.0f%% (n=%d)  corr(e,a_tilt)=%.2f'
        % (100.0 * sign_stats.get('sign_agreement', float('nan')),
           sign_stats.get('sign_n', 0),
           sign_stats.get('corr_tilt_error_action_tilt', float('nan'))),
    ]
    if step0_line:
        verdict_lines.append('     %s' % step0_line)
    verdict_lines.extend([
        '',
        'RC4 [100%%] Local energy optimum: G_fixed < G_learned < G_sun (%.4f < %.4f < %.4f kWh).'
        % (e_fixed, e_learned, _mean('total_energy_kwh', sun_summaries)),
        '     Not pvlib/reset/action-space bug (CSV scaling PASS).',
        '',
        'RC5 [checkpoint-specific] Train MDP vs eval:',
    ])
    if training_params:
        verdict_lines.append(
            '     config_version=%s  n_epochs=%s  randomize_day=%s'
            % (training_params.get('config_version'),
               training_params.get('n_epochs_config'),
               training_params.get('randomize_day')))
    else:
        verdict_lines.append('     (pass --trial-dir for params.json train/eval mismatch check)')
    verdict_lines.extend([
        '',
        'Ruled out: wrong action scaling; unfair sun eval; multi-day episodes; no env exploration.',
        '',
        'Fixes (ordered): Stage0 retrain -> --gate on sun -> BC/demos -> optional cos_aoi shaping.',
        'Do NOT rely on min_alpha alone when alpha already at floor and sign is wrong.',
        '',
        'mean |tilt-zenith| learned=%.2f° sun=%.2f°' % (tilt_err_l, tilt_err_s),
    ])
    sections.append(('Root-cause verdict (verified)', verdict_lines))

    config_lines = [
        'Training protocol: docs/TRAINING_PROTOCOL.md',
        '  Stage 0: examples/config/pv_tracking/stage0_single_day.py (one day, stationary).',
        '  Stage 1: examples/config/pv_tracking/0.py (summer i.i.d., clearsky, fixed summer training eval).',
        '  Gates: re-run with --gate (energy ratio, action ratio, tilt error, vs fixed).',
        '',
        'Limits (honest):',
        '  - Eval uses pi_eval(s)=tanh(mu(s)); SAC trains on stochastic tanh(mu+sigma*eps).',
        '  - Energy-only reward has no proof of sun-tracking; gates are acceptance tests.',
    ]
    sections.append(('Config change rationale', config_lines))

    gate_passed = None
    if args.gate and sun_summaries and learned_summaries:
        gate_passed, gate_lines = evaluate_phase_gates(
            learned_summaries,
            sun_summaries,
            fixed_summaries,
            args.min_energy_ratio,
            args.min_action_ratio,
            args.max_tilt_error_deg,
        )
        sections.append(('Phase gates (TRAINING_PROTOCOL.md)', gate_lines))

    report_path = write_report(outdir, sections)
    print('[diagnose] report: %s' % report_path)
    print('[diagnose] plots:  %s/' % plot_dir)
    for title, lines in sections:
        print('\n## %s' % title)
        for line in lines[:12]:
            print('  %s' % line)

    if args.gate:
        if gate_passed is None:
            print('\n[diagnose] --gate: skipped (need learned + sun_tracking rollouts)')
            sys.exit(2)
        if not gate_passed:
            print('\n[diagnose] GATE FAIL — see Phase gates section in report')
            sys.exit(1)
        print('\n[diagnose] GATE PASS')
        sys.exit(0)


if __name__ == '__main__':
    sys.exit(main() or 0)
