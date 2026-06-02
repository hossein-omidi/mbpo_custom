#!/usr/bin/env python3
"""Paper figures for Stage 3 annual-scenario MBPO-SAC evaluation.

Expects frozen-policy rollouts on the real PVTrackingEnv with matched seeds across
methods (learned_policy, sun_tracking, fixed_no_motion). Main evaluation protocol:
annual day sampling, irradiance_perturbation_std=0, no calendar hold-out.
"""

from __future__ import print_function

import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from examples.config.pv_tracking.verified_dates import SEASON_ORDER
from eval_utils import (
    compute_total_energy_kwh,
    describe_eval_config,
    get_rollout_metadata,
    night_intervals_from_path,
    rollout_time_axis,
    rollout_xlabel,
    season_from_day_of_year,
)

METHODS = ('learned_policy', 'sun_tracking', 'fixed_no_motion')
METHOD_LABELS = {
    'learned_policy': 'MBPO-SAC',
    'sun_tracking': 'Sun tracker',
    'fixed_no_motion': 'Fixed tracker',
}
METHOD_COLORS = {
    'learned_policy': '#1f77b4',
    'sun_tracking': '#2ca02c',
    'fixed_no_motion': '#ff7f0e',
}


def episode_metrics(path):
    """Per-episode gross energy, movement cost, net energy, total reward."""
    infos = path.get('infos', [])
    gross = float(np.sum([info.get('energy_kwh', 0.0) for info in infos]))
    movement = float(np.sum([info.get('movement_cost', 0.0) for info in infos]))
    net = float(np.sum(path.get('rewards', [])))
    angular = float(np.sum([
        abs(info.get('delta_tilt_deg', 0.0)) + abs(info.get('delta_azimuth_deg', 0.0))
        for info in infos]))
    meta = get_rollout_metadata(path)
    return {
        'gross_energy_kwh': gross,
        'movement_cost': movement,
        'net_energy_kwh': net,
        'total_reward': net,
        'angular_movement_deg': angular,
        'date': meta.get('date'),
        'season': meta.get('season'),
        'seed': (infos[0].get('rollout_seed') if infos else None),
    }


def build_episode_table(paths_by_name):
    rows = []
    for method, paths in paths_by_name.items():
        for idx, path in enumerate(paths):
            m = episode_metrics(path)
            rows.append({
                'method': method,
                'rollout_index': idx,
                'seed': m['seed'],
                **m,
            })
    return rows


def _error_bar(values, error='std'):
    v = np.asarray(values, dtype=np.float64)
    if len(v) == 0:
        return 0.0, 0
    if error == 'sem':
        return float(np.std(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0, len(v)
    return float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, len(v)


def plot_annual_performance_bars(outdir, paths_by_name, error='std'):
    """Gross energy, movement cost, net energy, total reward — all methods."""
    metrics = [
        ('gross_energy_kwh', 'Gross PV energy (kWh)'),
        ('movement_cost', 'Movement / actuator cost'),
        ('net_energy_kwh', 'Net energy (kWh)'),
        ('total_reward', 'Total reward'),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.ravel()
    methods = [m for m in METHODS if m in paths_by_name]

    for ax, (key, ylabel) in zip(axes, metrics):
        x = np.arange(len(methods))
        means, errs, ns = [], [], []
        for method in methods:
            vals = [episode_metrics(p)[key] for p in paths_by_name[method]]
            means.append(float(np.mean(vals)))
            err, n = _error_bar(vals, error)
            errs.append(err)
            ns.append(n)
        ax.bar(
            x, means, yerr=errs, capsize=5, color=[METHOD_COLORS[m] for m in methods],
            alpha=0.9, edgecolor='white')
        ax.set_xticks(x)
        ax.set_xticklabels([METHOD_LABELS[m] for m in methods], rotation=12, ha='right')
        ax.set_ylabel(ylabel)
        ax.grid(axis='y', linestyle='--', alpha=0.35)
        ax.set_title('%s (mean ± %s, n=%d rollouts)' % (
            ylabel, error, max(ns) if ns else 0))

    fig.suptitle(
        'Annual-scenario evaluation — matched seeds, real pvlib env (T=78)',
        fontsize=12, y=1.02)
    fig.tight_layout()
    path = os.path.join(outdir, 'annual_performance_bars.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


def plot_energy_decomposition(outdir, paths_by_name, error='std'):
    """Gross − movement = net (stacked validation + grouped bars)."""
    methods = [m for m in METHODS if m in paths_by_name]
    x = np.arange(len(methods))
    gross_m, gross_e = [], []
    move_m, move_e = [], []
    net_m, net_e = [], []

    for method in methods:
        gross = [episode_metrics(p)['gross_energy_kwh'] for p in paths_by_name[method]]
        move = [episode_metrics(p)['movement_cost'] for p in paths_by_name[method]]
        net = [episode_metrics(p)['net_energy_kwh'] for p in paths_by_name[method]]
        gross_m.append(np.mean(gross))
        move_m.append(np.mean(move))
        net_m.append(np.mean(net))
        gross_e.append(_error_bar(gross, error)[0])
        move_e.append(_error_bar(move, error)[0])
        net_e.append(_error_bar(net, error)[0])

    fig, ax = plt.subplots(figsize=(9, 5))
    w = 0.35
    ax.bar(x - w / 2, gross_m, w, yerr=gross_e, label='Gross energy', color='#9ecae1',
           capsize=4, alpha=0.95)
    ax.bar(x + w / 2, net_m, w, yerr=net_e, label='Net energy', color='#3182bd',
           capsize=4, alpha=0.95)
    ax.bar(x + w / 2, move_m, w, bottom=net_m, yerr=move_e, label='Movement cost (stacked)',
           color='#bdbdbd', capsize=3, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS[m] for m in methods])
    ax.set_ylabel('kWh / cost units')
    ax.set_title('Energy decomposition: gross − movement = net (mean ± %s)' % error)
    ax.legend(loc='best')
    ax.grid(axis='y', linestyle='--', alpha=0.35)
    fig.tight_layout()
    path = os.path.join(outdir, 'energy_decomposition.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_seasonal_comparison(outdir, paths_by_name, error='std'):
    """Net energy and movement cost by season/quarter."""
    paths_out = []
    for metric, ylabel, fname in (
        ('net_energy_kwh', 'Net energy (kWh)', 'seasonal_net_energy.png'),
        ('movement_cost', 'Movement cost', 'seasonal_movement_cost.png'),
    ):
        seasons = [s for s in SEASON_ORDER if any(
            episode_metrics(p)['season'] == s for paths in paths_by_name.values()
            for p in paths)]
        methods = [m for m in METHODS if m in paths_by_name]
        x = np.arange(len(seasons))
        width = 0.8 / max(len(methods), 1)
        fig, ax = plt.subplots(figsize=(10, 5))
        for i, method in enumerate(methods):
            means, errs = [], []
            for season in seasons:
                vals = [
                    episode_metrics(p)[metric]
                    for p in paths_by_name[method]
                    if episode_metrics(p)['season'] == season]
                means.append(float(np.mean(vals)) if vals else np.nan)
                errs.append(_error_bar(vals, error)[0])
            offset = (i - (len(methods) - 1) / 2.0) * width
            ax.bar(x + offset, means, width, yerr=errs, capsize=4,
                   color=METHOD_COLORS[method], alpha=0.88,
                   label=METHOD_LABELS[method])
        ax.set_xticks(x)
        ax.set_xticklabels(seasons)
        ax.set_ylabel(ylabel)
        ax.set_title('%s by season (mean ± %s)' % (ylabel, error))
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.35)
        fig.tight_layout()
        path = os.path.join(outdir, fname)
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths_out.append(path)
    return paths_out


def plot_daily_gain_distributions(outdir, paths_by_name):
    """Distributions of paired daily gains: MBPO-SAC − baseline."""
    learned = paths_by_name.get('learned_policy', [])
    if not learned:
        return None
    n = len(learned)
    gains = {'vs_fixed': [], 'vs_sun': []}
    for i in range(n):
        ml = episode_metrics(learned[i])
        if 'fixed_no_motion' in paths_by_name and i < len(paths_by_name['fixed_no_motion']):
            mf = episode_metrics(paths_by_name['fixed_no_motion'][i])
            gains['vs_fixed'].append(ml['net_energy_kwh'] - mf['net_energy_kwh'])
        if 'sun_tracking' in paths_by_name and i < len(paths_by_name['sun_tracking']):
            ms = episode_metrics(paths_by_name['sun_tracking'][i])
            gains['vs_sun'].append(ml['net_energy_kwh'] - ms['net_energy_kwh'])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, key, title in zip(
            axes,
            ('vs_fixed', 'vs_sun'),
            ('MBPO-SAC − Fixed', 'MBPO-SAC − Sun tracker')):
        v = gains[key]
        if not v:
            ax.set_visible(False)
            continue
        ax.hist(v, bins=min(15, max(5, len(v) // 2)), color=METHOD_COLORS['learned_policy'],
                alpha=0.75, edgecolor='white')
        ax.axvline(np.mean(v), color='black', linestyle='--', label='mean=%.4f' % np.mean(v))
        ax.axvline(0.0, color='#666', linestyle='-', linewidth=0.8)
        ax.set_xlabel('Net energy gain (kWh)')
        ax.set_title('%s (n=%d matched seeds)' % (title, len(v)))
        ax.legend(fontsize=8)
        ax.grid(axis='y', linestyle='--', alpha=0.35)
    fig.suptitle('Distribution of daily net-energy gains (same-day matched rollouts)', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(outdir, 'daily_gain_distributions.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_movement_efficiency(outdir, paths_by_name):
    """Angular movement vs net energy scatter per method."""
    fig, ax = plt.subplots(figsize=(7, 5))
    for method in METHODS:
        if method not in paths_by_name:
            continue
        xs, ys = [], []
        for p in paths_by_name[method]:
            m = episode_metrics(p)
            xs.append(m['angular_movement_deg'])
            ys.append(m['net_energy_kwh'])
        ax.scatter(xs, ys, alpha=0.65, s=40, color=METHOD_COLORS[method],
                   label=METHOD_LABELS[method])
    ax.set_xlabel('Total angular movement (deg, |Δtilt|+|Δaz|)')
    ax.set_ylabel('Net energy (kWh)')
    ax.set_title('Movement efficiency')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.35)
    fig.tight_layout()
    path = os.path.join(outdir, 'movement_efficiency.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_representative_daily_trajectories(outdir, paths_by_name, top_k=2):
    """Aligned same-day trajectories for methods with highest MBPO-SAC vs sun gain."""
    learned = paths_by_name.get('learned_policy', [])
    sun = paths_by_name.get('sun_tracking', [])
    if not learned or not sun:
        return []

    scores = []
    for i in range(min(len(learned), len(sun))):
        gain = episode_metrics(learned[i])['net_energy_kwh'] - episode_metrics(sun[i])['net_energy_kwh']
        scores.append((gain, i))
    scores.sort(reverse=True)
    selected = [idx for _, idx in scores[:top_k]]

    saved = []
    for rank, idx in enumerate(selected, 1):
        date = episode_metrics(learned[idx])['date']
        fig, axes = plt.subplots(7, 1, sharex=True, figsize=(11, 15))
        fig.suptitle('Representative day %s — rollout %d (MBPO−sun gain=%.4f kWh)' % (
            date, idx + 1, scores[rank - 1][0]), fontsize=11)

        for method in METHODS:
            if method not in paths_by_name or idx >= len(paths_by_name[method]):
                continue
            path = paths_by_name[method][idx]
            infos = path.get('infos', [])
            x = rollout_time_axis(path)
            power = [info.get('power', np.nan) for info in infos]
            axes[0].plot(x, power, color=METHOD_COLORS[method], lw=1.6,
                         label=METHOD_LABELS[method], alpha=0.9)

        ref = learned[idx]
        for lo, hi in night_intervals_from_path(ref):
            for ax in axes:
                ax.axvspan(lo, hi, color='#e8e8e8', alpha=0.4, zorder=0)

        for method in METHODS:
            if method not in paths_by_name or idx >= len(paths_by_name[method]):
                continue
            path = paths_by_name[method][idx]
            infos = path.get('infos', [])
            x = rollout_time_axis(path)
            rewards = np.asarray(path['rewards'], dtype=np.float64)
            gross = np.cumsum([info.get('energy_kwh', 0.0) for info in infos])
            net = np.cumsum(rewards)
            tilt = [info.get('tilt', np.nan) for info in infos]
            az = [info.get('azimuth', np.nan) for info in infos]
            salt = [info.get('solar_altitude_deg', np.nan) for info in infos]
            saz = [info.get('solar_azimuth_deg', np.nan) for info in infos]
            actions = np.asarray(path.get('actions', []))
            move = [info.get('movement_cost', 0.0) for info in infos]
            c = METHOD_COLORS[method]
            axes[1].plot(x, gross, '--', color=c, alpha=0.5)
            axes[1].plot(x, net, color=c, lw=1.5, label=METHOD_LABELS[method])
            axes[2].plot(x, tilt, color=c, lw=1.2)
            axes[3].plot(x, az, color=c, lw=1.2)
            axes[4].plot(x, salt, color=c, lw=1.0, alpha=0.85)
            if actions.ndim == 2 and actions.shape[0] == len(x):
                axes[5].plot(x, actions[:, 0], color=c, lw=1.0)
            axes[6].plot(x, rewards, color=c, lw=1.2)
            axes[6].plot(x, move, color=c, ls=':', alpha=0.6)

        axes[0].set_ylabel('Power (W)')
        axes[0].legend(loc='upper left', fontsize=8)
        axes[1].set_ylabel('Cumulative kWh')
        axes[1].legend(loc='upper left', fontsize=7)
        axes[2].set_ylabel('Panel tilt (°)')
        axes[3].set_ylabel('Panel azimuth (°)')
        axes[4].set_ylabel('Solar altitude (°)')
        axes[5].set_ylabel('Action tilt cmd')
        axes[6].set_ylabel('Reward / movement')
        axes[6].set_xlabel(rollout_xlabel(ref))
        for ax in axes:
            ax.grid(True, linestyle='--', alpha=0.3)

        fname = os.path.join(outdir, 'representative_daily_%s.png' % date)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        fig.savefig(fname, dpi=150)
        plt.close(fig)
        saved.append(fname)
    return saved


def plot_mc_timeseries_band(outdir, paths_by_name, field='power', ylabel='Power (W)'):
    """Pool rollouts by UTC clock hour: mean line ± 1 std (annual MC)."""
    buckets = {m: defaultdict(list) for m in paths_by_name}
    for method, paths in paths_by_name.items():
        for path in paths:
            for info in path.get('infos', []):
                t = info.get('clock_hour_utc', info.get('clock_hour', np.nan))
                val = info.get('power' if field == 'power' else 'energy_kwh', np.nan)
                if field == 'cumulative_net':
                    continue
                if np.isfinite(t) and np.isfinite(val):
                    buckets[method][round(float(t), 3)].append(val)

    hours = sorted(set(h for m in buckets for h in buckets[m]))
    if not hours:
        return None

    fig, ax = plt.subplots(figsize=(11, 4.5))
    for method, b in buckets.items():
        mean_y, std_y = [], []
        for h in hours:
            vals = b.get(h, [])
            mean_y.append(np.mean(vals) if vals else np.nan)
            std_y.append(np.std(vals) if len(vals) > 1 else 0.0)
        mean_y = np.asarray(mean_y)
        std_y = np.asarray(std_y)
        ax.plot(hours, mean_y, color=METHOD_COLORS[method], lw=2,
                label=METHOD_LABELS[method])
        ax.fill_between(hours, mean_y - std_y, mean_y + std_y,
                        color=METHOD_COLORS[method], alpha=0.2)
    ax.set_xlabel('UTC clock hour (post-step)')
    ax.set_ylabel(ylabel)
    ax.set_title('%s — annual MC (mean ± 1 std over rollouts)' % ylabel)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.35)
    path = os.path.join(outdir, 'mc_timeseries_%s.png' % field)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_training_diagnostics(trial_dir, outdir):
    """Optional training curves from progress.csv."""
    progress = os.path.join(trial_dir, 'progress.csv')
    if not os.path.isfile(progress):
        return []
    import pandas as pd
    df = pd.read_csv(progress)
    metrics = [
        ('evaluation/return-average', 'Evaluation return (avg)'),
        ('model/val_loss', 'BNN validation loss'),
        ('model_rollout_length', 'Model rollout length'),
        ('real_batch_ratio', 'Real batch ratio'),
        ('alpha', 'SAC temperature α'),
        ('Q_loss', 'Q loss'),
    ]
    xcol = 'training_iteration' if 'training_iteration' in df.columns else None
    if xcol is None:
        return []
    saved = []
    for col, title in metrics:
        if col not in df.columns:
            continue
        fig, ax = plt.subplots(figsize=(8, 3.5))
        ax.plot(df[xcol], df[col], marker='o', ms=3, lw=1.5)
        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.grid(True, linestyle='--', alpha=0.35)
        fname = os.path.join(outdir, 'training_%s.png' % col.replace('/', '_'))
        fig.tight_layout()
        fig.savefig(fname, dpi=150)
        plt.close(fig)
        saved.append(fname)
    return saved


def write_paper_metrics_json(outdir, paths_by_name, eval_env_params, protocol_note):
    rows = build_episode_table(paths_by_name)
    summary = {}
    for method in METHODS:
        sub = [r for r in rows if r['method'] == method]
        if not sub:
            continue
        summary[method] = {
            'n_rollouts': len(sub),
            'gross_energy_kwh': _stats([r['gross_energy_kwh'] for r in sub]),
            'movement_cost': _stats([r['movement_cost'] for r in sub]),
            'net_energy_kwh': _stats([r['net_energy_kwh'] for r in sub]),
            'total_reward': _stats([r['total_reward'] for r in sub]),
            'angular_movement_deg': _stats([r['angular_movement_deg'] for r in sub]),
        }
    payload = {
        'protocol': protocol_note,
        'eval_config': eval_env_params.get('kwargs', {}),
        'methods': summary,
        'episodes': rows,
    }
    path = os.path.join(outdir, 'paper_metrics.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, default=str)
    return path


def _stats(values):
    v = np.asarray(values, dtype=np.float64)
    return {
        'mean': float(np.mean(v)),
        'std': float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
        'variance': float(np.var(v, ddof=1)) if len(v) > 1 else 0.0,
        'min': float(np.min(v)),
        'max': float(np.max(v)),
        'n': int(len(v)),
    }


def write_protocol_readme(outdir, eval_env_params, protocol_note, is_stress=False):
    path = os.path.join(outdir, 'EVAL_PROTOCOL_README.txt')
    with open(path, 'w', encoding='utf-8') as f:
        f.write('Stage 3 PV tracking — evaluation protocol\n')
        f.write('=' * 40 + '\n\n')
        if is_stress:
            f.write('MODE: DIAGNOSTIC STRESS TEST (fixed calendar dates)\n')
            f.write('This is NOT the main paper evaluation.\n\n')
        else:
            f.write('MODE: MAIN PAPER EVALUATION (annual-scenario MC)\n')
            f.write(protocol_note + '\n\n')
        f.write('Environment (frozen policy, real pvlib):\n')
        for line in describe_eval_config(eval_env_params):
            f.write('  %s\n' % line)
        f.write('\nMetrics:\n')
        f.write('  gross_energy_kwh = sum step energy_kwh (pvlib)\n')
        f.write('  movement_cost = sum step movement penalty\n')
        f.write('  net_energy_kwh = total_reward = gross - movement\n')
        f.write('Baselines use identical seeds, horizon T=78, and env contract.\n')
    return path


def generate_paper_figures(
        outdir,
        paths_by_name,
        eval_env_params,
        protocol_note=None,
        error='std',
        trial_dir=None,
        is_stress=False,
        top_representative_days=2):
    """Create paper_figures/ under outdir with all standard plots."""
    paper_dir = os.path.join(outdir, 'paper_figures')
    os.makedirs(paper_dir, exist_ok=True)
    if protocol_note is None:
        protocol_note = (
            'Environmental variability via empirical annual day sampling; '
            'pvlib deterministic physics; irradiance_perturbation_std=0; '
            'no calendar hold-out.')

    if not paths_by_name or 'learned_policy' not in paths_by_name:
        print('[plot_paper_eval] skip: need learned_policy rollouts')
        return []

    write_protocol_readme(paper_dir, eval_env_params, protocol_note, is_stress=is_stress)
    write_paper_metrics_json(paper_dir, paths_by_name, eval_env_params, protocol_note)

    outputs = []
    if len(paths_by_name) >= 2:
        outputs.append(plot_annual_performance_bars(paper_dir, paths_by_name, error=error))
        outputs.append(plot_energy_decomposition(paper_dir, paths_by_name, error=error))
        outputs.extend(plot_seasonal_comparison(paper_dir, paths_by_name, error=error))
        outputs.append(plot_daily_gain_distributions(paper_dir, paths_by_name))
        outputs.append(plot_movement_efficiency(paper_dir, paths_by_name))
        outputs.extend(plot_representative_daily_trajectories(
            paper_dir, paths_by_name, top_k=top_representative_days))
        p = plot_mc_timeseries_band(paper_dir, paths_by_name, field='power')
        if p:
            outputs.append(p)

    if trial_dir:
        train_dir = os.path.join(paper_dir, 'training_diagnostics')
        os.makedirs(train_dir, exist_ok=True)
        outputs.extend(plot_training_diagnostics(trial_dir, train_dir))

    manifest = os.path.join(paper_dir, 'figures_manifest.txt')
    with open(manifest, 'w', encoding='utf-8') as f:
        f.write('\n'.join(outputs))
    print('[plot_paper_eval] wrote %d figures → %s' % (len(outputs), paper_dir))
    return outputs


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description='Regenerate paper figures from saved rollout CSVs.')
    p.add_argument('--eval-dir', required=True, help='evaluation output with rollouts/')
    args = p.parse_args()
    print('Load rollouts from %s and call generate_paper_figures from evaluate_agent.' % args.eval_dir)
