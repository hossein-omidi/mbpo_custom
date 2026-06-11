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
from eval_utils import SEASON_CALENDAR_ORDER
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


def _sample_stats(values):
    """Sample mean E[X] and sample std σ for finite MC rollouts (ddof=1)."""
    v = np.asarray(values, dtype=np.float64)
    n = len(v)
    if n == 0:
        return np.nan, np.nan, 0
    mean = float(np.mean(v))
    std = float(np.std(v, ddof=1)) if n > 1 else 0.0
    return mean, std, n


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
        'Annual-scenario evaluation — matched seeds, real pvlib env (T=117)',
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
        seasons = [s for s in SEASON_CALENDAR_ORDER if any(
            get_rollout_metadata(p).get('season_calendar', episode_metrics(p)['season']) == s
            for paths in paths_by_name.values()
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
                    if get_rollout_metadata(p).get(
                        'season_calendar', episode_metrics(p)['season']) == season]
                means.append(float(np.mean(vals)) if vals else np.nan)
                errs.append(_error_bar(vals, error)[0])
            offset = (i - (len(methods) - 1) / 2.0) * width
            ax.bar(x + offset, means, width, yerr=errs, capsize=4,
                   color=METHOD_COLORS[method], alpha=0.88,
                   label=METHOD_LABELS[method])
        ax.set_xticks(x)
        ax.set_xticklabels(seasons)
        ax.set_ylabel(ylabel)
        ax.set_title('%s by calendar season (mean ± %s)' % (ylabel, error))
        if seasons and methods:
            ax.legend(loc='best', fontsize=9)
        ax.grid(axis='y', linestyle='--', alpha=0.35)
        fig.tight_layout()
        path = os.path.join(outdir, fname)
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths_out.append(path)
    return paths_out


# NSRDB cloud_type codes (PSM v4): 0/1 clear, 7 cirrus, 2-6/8/9 cloud classes.
_NSRDB_CLOUD_GROUPS = (
    ('clear', {0, 1}),
    ('cirrus', {7}),
    ('cloudy', {2, 3, 4, 5, 6, 8, 9}),
)
WEATHER_LABEL_ORDER = ['clear', 'cirrus', 'partly_cloudy', 'cloudy', 'overcast', 'other']


def episode_weather_label(path):
    """Weather label from NSRDB's own cloud_type (episode mode); falls back to
    the clearsky-ratio condition label when the dataset lacks cloud_type."""
    meta = get_rollout_metadata(path)
    mode = meta.get('episode_cloud_type_mode')
    if mode is not None and np.isfinite(mode):
        code = int(round(float(mode)))
        for label, codes in _NSRDB_CLOUD_GROUPS:
            if code in codes:
                return label
        return 'other'
    return meta.get('weather_condition', 'other') or 'other'


def plot_weather_condition_comparison(outdir, paths_by_name, error='std'):
    """Net energy per method grouped by NSRDB cloud label (data-native, not synthetic)."""
    labels_present = []
    label_by_path = {}
    for method, paths in paths_by_name.items():
        for idx, p in enumerate(paths):
            label_by_path[(method, idx)] = episode_weather_label(p)
    for label in WEATHER_LABEL_ORDER:
        if any(v == label for v in label_by_path.values()):
            labels_present.append(label)
    if not labels_present:
        return None

    methods = [m for m in METHODS if m in paths_by_name]
    x = np.arange(len(labels_present))
    width = 0.8 / max(len(methods), 1)
    fig, ax = plt.subplots(figsize=(10, 5))
    counts = {label: 0 for label in labels_present}
    for i, method in enumerate(methods):
        means, errs = [], []
        for label in labels_present:
            vals = [
                episode_metrics(p)['net_energy_kwh']
                for idx, p in enumerate(paths_by_name[method])
                if label_by_path[(method, idx)] == label]
            counts[label] = max(counts[label], len(vals))
            means.append(float(np.mean(vals)) if vals else np.nan)
            errs.append(_error_bar(vals, error)[0])
        offset = (i - (len(methods) - 1) / 2.0) * width
        ax.bar(x + offset, means, width, yerr=errs, capsize=4,
               color=METHOD_COLORS[method], alpha=0.88,
               label=METHOD_LABELS[method])
    ax.set_xticks(x)
    ax.set_xticklabels(['%s\n(n=%d)' % (l, counts[l]) for l in labels_present])
    ax.set_ylabel('Net energy (kWh)')
    ax.set_title('Net energy by NSRDB weather label (cloud_type; mean ± %s)' % error)
    ax.legend(loc='best', fontsize=9)
    ax.grid(axis='y', linestyle='--', alpha=0.35)
    fig.tight_layout()
    path = os.path.join(outdir, 'weather_condition_net_energy.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_return_process_evaluation(outdir, paths_by_name):
    """MC evaluation return R = sum_t r_t: report E[R] and σ across rollouts per method.

    Post-train protocol: independent episodes indexed by eval seed; bars show
    sample mean ± sample std (not SEM unless n is large).
    """
    methods = [m for m in METHODS if m in paths_by_name]
    if not methods:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Panel A: bar chart E[R] ± σ
    ax = axes[0]
    x = np.arange(len(methods))
    means, stds, ns = [], [], []
    for method in methods:
        rewards = [episode_metrics(p)['total_reward'] for p in paths_by_name[method]]
        mu, sig, n = _sample_stats(rewards)
        means.append(mu)
        stds.append(sig)
        ns.append(n)
    ax.bar(x, means, yerr=stds, capsize=6, color=[METHOD_COLORS[m] for m in methods],
           alpha=0.9, edgecolor='white')
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS[m] for m in methods], rotation=12, ha='right')
    ax.set_ylabel('Episode total return (kWh net)')
    ax.set_title('E[R] ± σ over MC rollouts (annual scenario)')
    ax.grid(axis='y', linestyle='--', alpha=0.35)
    for i, (mu, sig, n) in enumerate(zip(means, stds, ns)):
        if np.isfinite(mu):
            ax.text(i, mu + sig + 0.01 * max(abs(mu), 1e-6), 'n=%d' % n,
                    ha='center', va='bottom', fontsize=8)

    # Panel B: rollout-index process (scatter + E[R] line)
    ax = axes[1]
    for method in methods:
        rewards = [episode_metrics(p)['total_reward'] for p in paths_by_name[method]]
        idx = np.arange(1, len(rewards) + 1)
        mu, sig, _ = _sample_stats(rewards)
        ax.scatter(idx, rewards, alpha=0.55, s=28, color=METHOD_COLORS[method],
                   label=METHOD_LABELS[method])
        ax.axhline(mu, color=METHOD_COLORS[method], linestyle='--', linewidth=1.2, alpha=0.85)
        ax.fill_between([idx.min(), idx.max()], mu - sig, mu + sig,
                        color=METHOD_COLORS[method], alpha=0.12)
    ax.set_xlabel('Rollout index (matched eval seeds)')
    ax.set_ylabel('Total return R')
    ax.set_title('Return process — per-rollout R with E[R] ± σ band')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, linestyle='--', alpha=0.35)

    fig.suptitle(
        'Frozen-policy MC evaluation: R = Σ_t (energy − movement), T=117',
        fontsize=11, y=1.02)
    fig.tight_layout()
    path = os.path.join(outdir, 'return_process_evaluation.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


def plot_cumulative_return_intraday(outdir, paths_by_name, field='cumulative_net'):
    """Intraday cumulative net energy: mean trajectory ± 1σ across MC rollouts (learned policy)."""
    learned = paths_by_name.get('learned_policy', [])
    if not learned:
        return None
    buckets = defaultdict(list)
    for path in learned:
        cum = 0.0
        for info in path.get('infos', []):
            cum += float(info.get('energy_kwh', 0.0)) - float(info.get('movement_cost', 0.0))
            t = info.get('clock_hour_utc', info.get('clock_hour', np.nan))
            if np.isfinite(t):
                buckets[round(float(t), 3)].append(cum)
    hours = sorted(buckets.keys())
    if len(hours) < 2:
        return None
    mean_y = np.array([np.mean(buckets[h]) for h in hours])
    std_y = np.array([np.std(buckets[h], ddof=1) if len(buckets[h]) > 1 else 0.0 for h in hours])

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(hours, mean_y, color=METHOD_COLORS['learned_policy'], lw=2,
            label='E[cumulative net energy]')
    ax.fill_between(hours, mean_y - std_y, mean_y + std_y,
                    color=METHOD_COLORS['learned_policy'], alpha=0.25,
                    label='±1σ across rollouts')
    ax.set_xlabel('UTC clock hour (post-step)')
    ax.set_ylabel('Cumulative net energy (kWh)')
    ax.set_title('Intraday return accumulation — MBPO-SAC (MC mean ± σ)')
    ax.legend(loc='best')
    ax.grid(True, linestyle='--', alpha=0.35)
    path = os.path.join(outdir, 'cumulative_return_intraday.png')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_baseline_sun_vs_fixed(outdir, paths_by_name, error='std'):
    """Sun tracker vs fixed: net energy mean ± uncertainty (paired MC, pvlib path)."""
    methods = [m for m in ('sun_tracking', 'fixed_no_motion') if m in paths_by_name]
    if len(methods) < 2:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    x = np.arange(len(methods))
    means, errs, ns = [], [], []
    for method in methods:
        vals = [episode_metrics(p)['net_energy_kwh'] for p in paths_by_name[method]]
        means.append(float(np.mean(vals)))
        err, n = _error_bar(vals, error)
        errs.append(err)
        ns.append(n)
    axes[0].bar(x, means, yerr=errs, capsize=5,
                color=[METHOD_COLORS[m] for m in methods], alpha=0.9, edgecolor='white')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([METHOD_LABELS[m] for m in methods])
    axes[0].set_ylabel('Net energy (kWh)')
    axes[0].set_title('Baselines: mean ± %s (n=%d matched rollouts)' % (
        error, max(ns) if ns else 0))
    axes[0].grid(axis='y', linestyle='--', alpha=0.35)

    n = min(len(paths_by_name['sun_tracking']), len(paths_by_name['fixed_no_motion']))
    deltas = []
    for i in range(n):
        es = episode_metrics(paths_by_name['sun_tracking'][i])['net_energy_kwh']
        ef = episode_metrics(paths_by_name['fixed_no_motion'][i])['net_energy_kwh']
        deltas.append(es - ef)
    deltas = np.asarray(deltas, dtype=np.float64)
    axes[1].hist(deltas, bins=min(15, max(5, len(deltas) // 2)),
                 color=METHOD_COLORS['sun_tracking'], alpha=0.75, edgecolor='white')
    axes[1].axvline(float(np.mean(deltas)), color='black', linestyle='--',
                    label='E[sun−fixed]=%.4f' % float(np.mean(deltas)))
    if len(deltas) > 1:
        axes[1].axvline(float(np.mean(deltas) + np.std(deltas, ddof=1)),
                        color='#666666', linestyle=':', alpha=0.8)
        axes[1].axvline(float(np.mean(deltas) - np.std(deltas, ddof=1)),
                        color='#666666', linestyle=':', alpha=0.8)
    axes[1].axvline(0.0, color='#999999', linewidth=0.8)
    axes[1].set_xlabel('Paired Δ net energy (kWh)')
    axes[1].set_title('Sun tracker − fixed (same scenario per seed)')
    axes[1].legend(fontsize=8)
    axes[1].grid(axis='y', linestyle='--', alpha=0.35)

    fig.suptitle('Baseline comparison — pvlib env.step, empirical weather MC', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    path = os.path.join(outdir, 'baseline_sun_vs_fixed_mc.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


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
        ax.axvline(np.mean(v), color='black', linestyle='--', label='E[gain]=%.4f' % np.mean(v))
        ax.axvline(0.0, color='#666666', linestyle='-', linewidth=0.8)
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
    from plot_training_progress import prepare_progress_dataframe

    df, meta = prepare_progress_dataframe(pd.read_csv(progress))
    x = df['global_step'] if 'global_step' in df.columns else None
    if x is None:
        return []
    metrics = [
        ('evaluation/return-average', 'Evaluation return (avg)'),
        ('model/val_loss', 'BNN validation loss'),
        ('model_rollout_length', 'Model rollout length'),
        ('real_batch_ratio', 'Real batch ratio'),
        ('alpha', 'SAC temperature α'),
        ('Q_loss', 'Q loss'),
    ]
    xlabel = meta.get('x_label', 'Training step')
    saved = []
    for col, title in metrics:
        if col not in df.columns:
            continue
        fig, ax = plt.subplots(figsize=(8, 3.5))
        ax.plot(x, df[col], marker='o', ms=3, lw=1.5)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
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
        f.write('Baselines use identical seeds, horizon T=117, and env contract.\n')
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

    def _safe(name, fn, *a, **kw):
        try:
            p = fn(*a, **kw)
            if p:
                outputs.append(p)
        except Exception as exc:
            print('[plot_paper_eval] %s failed: %s' % (name, exc))

    if len(paths_by_name) >= 2:
        _safe('return_process_evaluation', plot_return_process_evaluation, paper_dir, paths_by_name)
        _safe('cumulative_return_intraday', plot_cumulative_return_intraday, paper_dir, paths_by_name)
        _safe('annual_performance_bars', plot_annual_performance_bars, paper_dir, paths_by_name, error=error)
        _safe('energy_decomposition', plot_energy_decomposition, paper_dir, paths_by_name, error=error)
        try:
            outputs.extend(plot_seasonal_comparison(paper_dir, paths_by_name, error=error) or [])
        except Exception as exc:
            print('[plot_paper_eval] seasonal_comparison failed: %s' % exc)
        _safe('weather_condition_comparison', plot_weather_condition_comparison,
              paper_dir, paths_by_name, error=error)
        _safe('daily_gain_distributions', plot_daily_gain_distributions, paper_dir, paths_by_name)
        _safe('baseline_sun_vs_fixed', plot_baseline_sun_vs_fixed, paper_dir, paths_by_name, error=error)
        _safe('movement_efficiency', plot_movement_efficiency, paper_dir, paths_by_name)
        outputs.extend(plot_representative_daily_trajectories(
            paper_dir, paths_by_name, top_k=top_representative_days) or [])
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
