#!/usr/bin/env python3
"""Verify PV observation state affects pvlib power/energy; compare days and weather.

Uses the same PVTrackingEnv + pvlib path as training (no MBPO changes).

Examples:

  cd /home/user01/mbpo_custom && conda activate mbpo
  python scripts/verify_pv_state_space.py
  python scripts/verify_pv_state_space.py --outdir verification/pv_state_space
"""

from __future__ import print_function

import argparse
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import gym
import softlearning.environments.adapters.gym_adapter  # noqa: F401

from pvlib.location import Location

from mbpo.env.historical_weather import (
    available_month_days,
    build_weather_profile_from_catalog,
    load_historical_weather_catalog,
)
from mbpo.env.pv_tracking import (
    DEFAULT_FREQ,
    DEFAULT_PERIODS,
    DEFAULT_START_TIME,
    PHYSICAL_OBS_LABELS,
    PV_TIMEZONE,
)
from eval_utils import (
    decode_pv_observation,
    make_baseline_rollout,
    normalize_angle_diff,
    summarize_paired_mc,
    verify_rollout_pvlib_power,
)
from mbpo.env.nsrdb_weather import default_manifest_path
from examples.config.pv_tracking.verified_dates import (
    CLEAR_HISTORICAL_DAY as CLEAR_DAY,
    CLOUDY_HISTORICAL_DAY as CLOUDY_DAY,
    WINTER_HISTORICAL_DAY as WINTER_DAY,
)

COMPARE_DATES = (CLEAR_DAY, CLOUDY_DAY, WINTER_DAY)

WEATHER_CONDITION_ORDER = ('clear', 'partly_cloudy', 'overcast')
WEATHER_CONDITION_COLORS = {
    'clear': '#2ca02c',
    'partly_cloudy': '#ffbb78',
    'overcast': '#7f7f7f',
}
# Site of the bundled *historical* (legacy TMY) catalog only — Albuquerque.
# The NSRDB multi-year scenario path uses the NYC site (see make_nsrdb_env).
DEFAULT_SITE_LAT = 35.0
DEFAULT_SITE_LON = -106.0
DEFAULT_SITE_ALT = 1600.0

# Maps observation index → pvlib power sensitivity factor (see compute_power_factor_sensitivity).
OBS_INDEX_TO_FACTOR = (
    'solar_geometry',   # solar_zenith_norm
    'solar_geometry',   # solar_azimuth_sin
    'solar_geometry',   # solar_azimuth_cos
    'irradiance',       # dni_norm
    'irradiance',       # dhi_norm
    'irradiance',       # ghi_norm
    'none',             # temperature_norm (not in pvlib power here)
    'panel_orientation',  # panel_tilt_norm
    'panel_orientation',  # panel_azimuth_sin
    'panel_orientation',  # panel_azimuth_cos
    'panel_orientation',  # cos_aoi (derived; shares panel+solar)
)

FACTOR_LABELS = (
    'panel_orientation',
    'solar_geometry',
    'irradiance',
)


def make_env(date, weather_source='historical', observation_mode='physical'):
    return gym.make(
        'PVTracking-v0',
        start_date=date,
        end_date=date,
        randomize_day=False,
        randomize_initial_orientation=False,
        fixed_eval_dates=[date],
        weather_source=weather_source,
        observation_mode=observation_mode,
    )


def make_nsrdb_env(manifest=None, observation_mode='physical'):
    """NSRDB multi-year env for scenario MC (e ~ Uniform(manifest) per episode).

    Within each episode weather is a fixed NSRDB trajectory — not random per step.
    """
    return gym.make(
        'PVTracking-v0',
        latitude=40.72,
        longitude=-74.01,
        start_date='2018-01-01',
        end_date='2024-12-31',
        randomize_day=False,
        randomize_scenario=True,
        randomize_initial_orientation=False,
        weather_scenario_mode='nsrdb_multiyear',
        weather_source='nsrdb_multiyear',
        scenario_manifest=manifest or default_manifest_path(),
        irradiance_perturbation_std=0.0,
        movement_penalty=0.0,
        observation_mode=observation_mode,
    )


def run_nsrdb_mc_baselines(n_rollouts, seed_base=42, path_length=117):
    """Paired MC: sun tracker vs fixed on same NSRDB scenarios (pvlib via env.step)."""
    paths = {'sun_tracking': [], 'fixed_no_motion': []}
    for i in range(int(n_rollouts)):
        seed = int(seed_base) + i
        for method in paths:
            env = make_nsrdb_env()
            try:
                paths[method].append(
                    make_baseline_rollout(env, method, path_length, seed=seed))
            finally:
                env.close()
    return paths


def plot_nsrdb_mc_baseline_comparison(outdir, paths_by_name):
    """Sun vs fixed net energy: mean ± 1σ over NSRDB scenario MC."""
    methods = ('sun_tracking', 'fixed_no_motion')
    if not all(m in paths_by_name for m in methods):
        return None
    n = min(len(paths_by_name['sun_tracking']), len(paths_by_name['fixed_no_motion']))
    sun_e = [float(np.sum(p['rewards'])) for p in paths_by_name['sun_tracking'][:n]]
    fix_e = [float(np.sum(p['rewards'])) for p in paths_by_name['fixed_no_motion'][:n]]
    deltas = np.asarray(sun_e, dtype=np.float64) - np.asarray(fix_e, dtype=np.float64)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    x = [0, 1]
    means = [np.mean(sun_e), np.mean(fix_e)]
    stds = [np.std(sun_e, ddof=1) if n > 1 else 0.0, np.std(fix_e, ddof=1) if n > 1 else 0.0]
    axes[0].bar(x, means, yerr=stds, capsize=5, color=['#2ca02c', '#7f7f7f'], alpha=0.9)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(['Sun tracker', 'Fixed 30°/180°'])
    axes[0].set_ylabel('Net energy (kWh)')
    axes[0].set_title('NSRDB scenario MC (n=%d, matched seeds)' % n)
    axes[0].grid(axis='y', alpha=0.3)

    axes[1].hist(deltas, bins=min(12, max(4, n // 2)), color='#2ca02c', alpha=0.75)
    axes[1].axvline(float(np.mean(deltas)), color='k', ls='--',
                    label='E[sun−fixed]=%.4f' % float(np.mean(deltas)))
    if n > 1:
        axes[1].axvline(float(np.mean(deltas) + np.std(deltas, ddof=1)), color='#888888', ls=':')
        axes[1].axvline(float(np.mean(deltas) - np.std(deltas, ddof=1)), color='#888888', ls=':')
    axes[1].axvline(0.0, color='#aaaaaa', lw=0.8)
    axes[1].set_xlabel('Paired Δ (kWh)')
    axes[1].set_title('Same scenario per seed')
    axes[1].legend(fontsize=8)
    axes[1].grid(axis='y', alpha=0.3)

    path = os.path.join(outdir, 'nsrdb_mc_baseline_sun_vs_fixed.png')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def format_nsrdb_mc_report(paths_by_name, n_rollouts):
    """Lines for report: probabilistic baseline comparison."""
    ref = 'learned_policy' if 'learned_policy' in paths_by_name else 'sun_tracking'
    summary = summarize_paired_mc(paths_by_name, reference=ref)
    lines = [
        'NSRDB multi-year baseline MC (e ~ Uniform(manifest), pvlib env.step):',
        '  N=%d matched seeds; net energy = sum_t reward_t (kWh).' % n_rollouts,
    ]
    for method, stats in summary.get('methods', {}).items():
        lines.append('  %-18s  E=%.4f  σ=%.4f  n=%d' % (
            method, stats['mean'], stats['std'], stats['count']))
    for label, stats in summary.get('paired_deltas', {}).items():
        lines.append('  paired %-16s  E[Δ]=%+.4f  σ_Δ=%.4f' % (
            label, stats['mean'], stats['std']))
    for method, paths in paths_by_name.items():
        if paths:
            ok, max_err, _ = verify_rollout_pvlib_power(paths[0])
            lines.append('  pvlib spot-check %s: %s (max err %.3g W)' % (
                method, 'PASS' if ok else 'WARN', max_err))
    lines.append(
        '  (Deterministic checks above — tilt sweep, OAT — use single fixed dates; '
        'not MC because they test local pvlib sensitivity, not weather distribution.)')
    lines.append('')
    return lines


def run_sun_tracking_episode(env, seed=0):
    """Greedy sun tracker: same rule as eval_utils make_baseline_rollout."""
    env.seed(seed)
    obs = env.reset()
    inner = env.unwrapped
    total_energy = 0.0
    powers = []
    obs_rows = []
    infos = []

    done = False
    while not done:
        decoded = decode_pv_observation(obs)
        from mbpo.env.pvlib_physics import sun_tracker_target_tilt_deg
        target_tilt = sun_tracker_target_tilt_deg(decoded['solar_zenith_deg'])
        target_az = decoded['solar_azimuth_deg']
        delta_tilt = np.clip(
            target_tilt - decoded['panel_tilt_deg'],
            -inner.max_delta_tilt, inner.max_delta_tilt)
        delta_az = np.clip(
            normalize_angle_diff(target_az, decoded['panel_azimuth_deg']),
            -inner.max_delta_azimuth, inner.max_delta_azimuth)
        action = np.array([
            delta_tilt / inner.max_delta_tilt,
            delta_az / inner.max_delta_azimuth,
        ], dtype=np.float32)
        obs, reward, done, info = env.step(action)
        total_energy += float(info['energy_kwh'])
        powers.append(float(info['power']))
        obs_rows.append(np.asarray(obs, dtype=np.float64).reshape(-1))
        infos.append(info)
    return {
        'total_energy_kwh': total_energy,
        'powers': np.array(powers),
        'observations': np.stack(obs_rows),
        'infos': infos,
    }


def run_fixed_orientation_episode(env, tilt=30.0, azimuth=180.0, seed=0):
    env.seed(seed)
    env.reset()
    inner = env.unwrapped
    inner.tilt = float(tilt)
    inner.azimuth = float(azimuth)
    total_energy = 0.0
    powers = []
    obs_rows = []
    infos = []
    obs = inner._build_observation(
        inner._solar_position(inner.current_time).zenith,
        inner._solar_position(inner.current_time).azimuth,
        inner._current_weather(),
        inner.last_power,
    )
    done = False
    while not done:
        action = np.zeros(2, dtype=np.float32)
        obs, reward, done, info = env.step(action)
        total_energy += float(info['energy_kwh'])
        powers.append(float(info['power']))
        obs_rows.append(np.asarray(obs, dtype=np.float64).reshape(-1))
        infos.append(info)
    return {
        'total_energy_kwh': total_energy,
        'powers': np.array(powers),
        'observations': np.stack(obs_rows),
        'infos': infos,
    }


def sweep_tilt_power(date, weather_source='historical', n_tilts=19):
    """Vary panel tilt at peak-GHI step with azimuth aligned to sun — pvlib power span."""
    env = make_env(date, weather_source=weather_source)
    env.reset()
    inner = env.unwrapped
    # Use the timestep with highest GHI in the episode (meaningful solar geometry).
    ghi = inner.weather_profile['ghi'].values
    peak_idx = int(np.argmax(ghi))
    inner.step_index = peak_idx
    inner.current_time = inner.times[peak_idx]
    sp = inner._solar_position(inner.current_time)
    weather = inner._current_weather()
    sun_az = float(sp.azimuth)

    tilts = np.linspace(0.0, 90.0, n_tilts)
    powers = []
    for tilt in tilts:
        p = inner._power_from_orientation(
            sp.zenith, sp.azimuth, float(tilt), sun_az)
        powers.append(float(p))
    env.close()
    return tilts, np.array(powers)


def _episode_times_for_date(date, start_time=DEFAULT_START_TIME,
                            periods=DEFAULT_PERIODS, freq=DEFAULT_FREQ,
                            tz=PV_TIMEZONE):
    return pd.date_range(
        start='{} {}'.format(pd.Timestamp(date).date(), start_time),
        periods=periods,
        freq=freq,
        tz=tz,
    )


def compute_full_year_weather_distribution(
        start_date='2020-01-01',
        end_date='2020-12-31',
        start_time=DEFAULT_START_TIME,
        periods=DEFAULT_PERIODS,
        freq=DEFAULT_FREQ,
        latitude=DEFAULT_SITE_LAT,
        longitude=DEFAULT_SITE_LON,
        altitude=DEFAULT_SITE_ALT,
        tz=PV_TIMEZONE):
    """Aggregate weather-condition shares over the training episode window for each day.

    Uses the same TMY catalog + classify_weather_conditions rules as PVTrackingEnv
    (historical weather_source). Counts are over episode timesteps (periods-1 actions
    use the same profile rows as env steps 0..T-1).
    """
    location = Location(latitude, longitude, tz=tz, altitude=altitude)
    catalog = load_historical_weather_catalog(freq='15min')

    calendar_days = pd.date_range(
        start=pd.Timestamp(start_date),
        end=pd.Timestamp(end_date),
        freq='D',
        tz=tz,
    )
    allowed = available_month_days(catalog)
    calendar_days = [
        d for d in calendar_days
        if (int(d.month), int(d.day)) in allowed
    ]

    step_counts = {c: 0 for c in WEATHER_CONDITION_ORDER}
    day_dominant_counts = {c: 0 for c in WEATHER_CONDITION_ORDER}
    month_step_counts = {
        m: {c: 0 for c in WEATHER_CONDITION_ORDER}
        for m in range(1, 13)
    }
    per_day_rows = []

    for day in calendar_days:
        times = _episode_times_for_date(
            day, start_time=start_time, periods=periods, freq=freq, tz=tz)
        profile = build_weather_profile_from_catalog(location, times, catalog)
        # Env steps align with profile rows 0 .. len(times)-2 (117 steps).
        n_steps = max(len(times) - 1, 1)
        cond = profile['condition'].values[:n_steps]
        for c in cond:
            if c not in step_counts:
                step_counts[c] = 0
            step_counts[c] += 1
            month_step_counts[int(day.month)][c] += 1
        counts = pd.Series(cond).value_counts()
        dominant = str(counts.index[0])
        if dominant not in day_dominant_counts:
            day_dominant_counts[dominant] = 0
        day_dominant_counts[dominant] += 1
        per_day_rows.append({
            'date': str(day.date()),
            'month': int(day.month),
            'dominant': dominant,
            'clear_frac': float((cond == 'clear').mean()),
            'partly_cloudy_frac': float((cond == 'partly_cloudy').mean()),
            'overcast_frac': float((cond == 'overcast').mean()),
            'ghi_mean': float(profile['ghi'].values[:n_steps].mean()),
        })

    n_steps_total = sum(step_counts.values())
    n_days = len(per_day_rows)
    step_pct = {
        c: 100.0 * step_counts.get(c, 0) / max(n_steps_total, 1)
        for c in WEATHER_CONDITION_ORDER
    }
    day_pct = {
        c: 100.0 * day_dominant_counts.get(c, 0) / max(n_days, 1)
        for c in WEATHER_CONDITION_ORDER
    }
    month_step_pct = {}
    for m in range(1, 13):
        total_m = sum(month_step_counts[m].values())
        month_step_pct[m] = {
            c: 100.0 * month_step_counts[m].get(c, 0) / max(total_m, 1)
            for c in WEATHER_CONDITION_ORDER
        }

    return {
        'start_date': start_date,
        'end_date': end_date,
        'n_days': n_days,
        'n_steps': n_steps_total,
        'step_counts': step_counts,
        'step_pct': step_pct,
        'day_dominant_counts': day_dominant_counts,
        'day_pct': day_pct,
        'month_step_counts': month_step_counts,
        'month_step_pct': month_step_pct,
        'per_day': pd.DataFrame(per_day_rows),
    }


def _pct_bar_labels(pct_dict, order=WEATHER_CONDITION_ORDER):
    return [pct_dict.get(c, 0.0) for c in order]


def plot_full_year_weather_distribution(outdir, stats):
    """Bar charts: full-year weather-type percentages (step vs day-dominant)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    x = np.arange(len(WEATHER_CONDITION_ORDER))
    colors = [WEATHER_CONDITION_COLORS[c] for c in WEATHER_CONDITION_ORDER]

    for ax, pct_key, title in (
        (axes[0], 'step_pct',
         'By timestep (% of episode steps, all days)'),
        (axes[1], 'day_pct',
         'By calendar day (% days with dominant condition)'),
    ):
        vals = _pct_bar_labels(stats[pct_key])
        bars = ax.bar(x, vals, color=colors, edgecolor='white', linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(WEATHER_CONDITION_ORDER)
        ax.set_ylabel('Share (%)')
        ax.set_ylim(0, 100)
        ax.set_title(title)
        ax.grid(axis='y', alpha=0.3)
        for bar, v in zip(bars, vals):
            if v >= 3.0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1.0,
                    '%.1f%%' % v,
                    ha='center',
                    va='bottom',
                    fontsize=9,
                )
    fig.suptitle(
        'Historical TMY weather mix — episode window %s–%s UTC (%d days, %d steps)' % (
            DEFAULT_START_TIME,
            '23:15',
            stats['n_days'],
            stats['n_steps'],
        ),
        fontsize=11,
        y=1.02,
    )
    path = os.path.join(outdir, 'full_year_weather_distribution.png')
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


def plot_monthly_weather_distribution(outdir, stats):
    """Stacked bars: monthly share of clear / partly_cloudy / overcast (by step)."""
    months = list(range(1, 13))
    month_labels = [
        'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec',
    ]
    bottom = np.zeros(len(months))
    fig, ax = plt.subplots(figsize=(12, 5))
    for cond in WEATHER_CONDITION_ORDER:
        vals = np.array([
            stats['month_step_pct'][m].get(cond, 0.0) for m in months
        ])
        ax.bar(
            months, vals, bottom=bottom, label=cond,
            color=WEATHER_CONDITION_COLORS[cond],
            edgecolor='white',
            linewidth=0.5,
        )
        bottom += vals
    ax.set_xticks(months)
    ax.set_xticklabels(month_labels)
    ax.set_ylabel('Share of episode steps (%)')
    ax.set_ylim(0, 100)
    ax.set_title(
        'Monthly weather mix (historical TMY, %s episode grid)' % DEFAULT_START_TIME)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(axis='y', alpha=0.25)
    path = os.path.join(outdir, 'full_year_weather_by_month.png')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def format_weather_distribution_report(stats):
    """Text lines for state_space_verification.txt."""
    lines = [
        'Full-year weather-type distribution (historical TMY, training episode window):',
        '  Catalog: one fixed trajectory per calendar day (month-day); not stochastic per reset.',
        '  Classification: GHI/DNI vs pvlib clearsky (clear / partly_cloudy / overcast).',
        '  Range: %s .. %s  (%d days, %d episode steps)' % (
            stats['start_date'], stats['end_date'], stats['n_days'], stats['n_steps']),
        '',
        '  Share of all episode timesteps:',
    ]
    for c in WEATHER_CONDITION_ORDER:
        lines.append('    %s: %6.1f%%  (n=%d)' % (
            c, stats['step_pct'][c], stats['step_counts'].get(c, 0)))
    lines.append('')
    lines.append('  Share of calendar days (dominant condition in episode window):')
    for c in WEATHER_CONDITION_ORDER:
        lines.append('    %s: %6.1f%%  (n=%d days)' % (
            c, stats['day_pct'][c], stats['day_dominant_counts'].get(c, 0)))
    lines.append('')
    lines.append('  Monthly timestep share (clear / partly_cloudy / overcast):')
    month_names = [
        'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec',
    ]
    for m, name in enumerate(month_names, start=1):
        p = stats['month_step_pct'][m]
        lines.append(
            '    %s: clear=%.0f%%  partly_cloudy=%.0f%%  overcast=%.0f%%' % (
                name, p['clear'], p['partly_cloudy'], p['overcast']))
    lines.append('')
    return lines


def episode_weather_summary(date, weather_source):
    env = make_env(date, weather_source=weather_source)
    inner = env.unwrapped
    profile = inner.weather_profile
    times = profile.index
    cond = profile['condition'].values if 'condition' in profile else None
    env.close()
    clearsky = inner.location.get_clearsky(times)
    ghi_ratio = profile['ghi'].values / np.maximum(clearsky['ghi'].values, 1.0)
    return {
        'date': date,
        'weather_source': weather_source,
        'ghi_mean': float(profile['ghi'].mean()),
        'dni_mean': float(profile['dni'].mean()),
        'ghi_ratio_mean': float(ghi_ratio.mean()),
        'overcast_frac': float((cond == 'overcast').mean()) if cond is not None else np.nan,
        'conditions': list(pd.Series(cond).value_counts().to_dict()) if cond is not None else [],
    }


def plot_orientation_sweep(outdir, date_cloudy, weather_source='historical'):
    tilts, powers = sweep_tilt_power(date_cloudy, weather_source=weather_source)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(tilts, powers, 'o-', color='#1f77b4', lw=2)
    ax.set_xlabel('Panel tilt (deg) at episode start')
    ax.set_ylabel('Power (W) from pvlib')
    ax.set_title('State (tilt) affects power — %s historical' % date_cloudy)
    ax.grid(True, alpha=0.3)
    pmin, pmax = powers.min(), powers.max()
    ax.text(0.02, 0.95, 'range %.1f–%.1f W (%.0f%% span)' % (
        pmin, pmax, 100.0 * (pmax - pmin) / max(pmax, 1e-6)),
        transform=ax.transAxes, va='top', fontsize=9)
    path = os.path.join(outdir, 'orientation_sweep_power.png')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path, float(pmax - pmin)


def plot_episode_comparison(outdir, results_by_key):
    """Bar chart: cumulative energy sun-tracking vs fixed per day/weather."""
    labels = []
    e_sun = []
    e_fix = []
    for key in sorted(results_by_key.keys()):
        labels.append(key)
        e_sun.append(results_by_key[key]['sun']['total_energy_kwh'])
        e_fix.append(results_by_key[key]['fixed']['total_energy_kwh'])
    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w / 2, e_sun, w, label='sun tracking', color='#2ca02c')
    ax.bar(x + w / 2, e_fix, w, label='fixed 30°/180°', color='#7f7f7f')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha='right')
    ax.set_ylabel('Episode energy (kWh)')
    ax.set_title('Cumulative energy varies by day and control (pvlib env)')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    path = os.path.join(outdir, 'episode_energy_by_scenario.png')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_observation_variation(outdir, results_by_key):
    """Line plots of physical obs components across episode for each scenario."""
    n = len(results_by_key)
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(11, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, n))
    for i, (key, data) in enumerate(sorted(results_by_key.items())):
        obs = data['sun']['observations']
        steps = np.arange(obs.shape[0])
        axes[0].plot(steps, obs[:, 0], color=colors[i], lw=1.5, label=key)
        axes[1].plot(steps, obs[:, 3], color=colors[i], lw=1.5)
        axes[1].plot(steps, obs[:, 5], color=colors[i], lw=1.5, ls='--', alpha=0.7)
        axes[2].plot(steps, obs[:, -1], color=colors[i], lw=1.5)
    axes[0].set_ylabel('solar_zenith_norm')
    axes[1].set_ylabel('dni_norm / ghi_norm')
    axes[2].set_ylabel('cos_aoi')
    axes[2].set_xlabel('Step')
    axes[0].legend(loc='upper right', fontsize=7, ncol=2)
    axes[0].set_title('Observation trajectories (sun tracking) — different days/weather')
    for ax in axes:
        ax.grid(True, alpha=0.3)
    path = os.path.join(outdir, 'observation_trajectories.png')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_irradiance_profiles(outdir, dates, weather_source='historical'):
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(11, 6))
    for date in dates:
        env = make_env(date, weather_source=weather_source)
        inner = env.unwrapped
        t = inner.times[:len(inner.weather_profile)]
        hours = t.hour + t.minute / 60.0
        prof = inner.weather_profile
        axes[0].plot(hours, prof['ghi'].values, lw=1.8, label=date)
        axes[1].plot(hours, prof['dni'].values, lw=1.8, label=date)
        env.close()
    axes[0].set_ylabel('GHI (W/m²)')
    axes[1].set_ylabel('DNI (W/m²)')
    axes[1].set_xlabel('Clock hour UTC')
    axes[0].set_title('Weather inputs (%s) — episode grid' % weather_source)
    axes[0].legend()
    for ax in axes:
        ax.grid(True, alpha=0.3)
    path = os.path.join(outdir, 'irradiance_profiles.png')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_power_timeseries(outdir, results_by_key):
    fig, ax = plt.subplots(figsize=(11, 5))
    for key, data in sorted(results_by_key.items()):
        if 'sun' not in data:
            continue
        infos = data['sun']['infos']
        hours = [float(i.get('clock_hour', i.get('time', 0))) for i in infos]
        powers = data['sun']['powers']
        ax.plot(hours, powers, lw=1.8, label='%s (sun)' % key)
    ax.set_xlabel('Clock hour UTC')
    ax.set_ylabel('Power (W)')
    ax.set_title('Intra-day power — sun tracking (state + weather → pvlib)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    path = os.path.join(outdir, 'power_timeseries_sun.png')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _power_from_physics(inner, solar_zenith, solar_azimuth, tilt, azimuth, dni, dhi, ghi):
    """Same pvlib path as PVTrackingEnv._power_from_orientation (explicit irradiance)."""
    from mbpo.env.pvlib_physics import compute_panel_power_w
    return compute_panel_power_w(
        tilt, azimuth, solar_zenith, solar_azimuth,
        dni, ghi, dhi, area=inner.area, efficiency=inner.efficiency)


def _oat_delta_power(inner, info, factor, eps_frac=0.1):
    """One-at-a-time |ΔP| for a single factor at one timestep."""
    zen = float(info.get('solar_zenith_deg', np.nan))
    az = float(info.get('solar_azimuth_deg', np.nan))
    tilt = float(info.get('tilt_deg', info.get('tilt', 0.0)))
    panel_az = float(info.get('azimuth_deg', info.get('azimuth', 0.0)))
    w = {
        'dni': float(info.get('dni_wm2', info.get('dni', 0.0))),
        'dhi': float(info.get('dhi_wm2', info.get('dhi', 0.0))),
        'ghi': float(info.get('ghi_wm2', info.get('ghi', 0.0))),
    }

    def P(tilt_v, panel_az_v, zen_v, az_v, dni, dhi, ghi):
        return _power_from_physics(inner, zen_v, az_v, tilt_v, panel_az_v, dni, dhi, ghi)

    P0 = P(tilt, panel_az, zen, az, w['dni'], w['dhi'], w['ghi'])

    if factor == 'panel_orientation':
        d = max(abs(P(tilt + 5.0, panel_az, zen, az, w['dni'], w['dhi'], w['ghi']) - P0),
                abs(P(tilt - 5.0, panel_az, zen, az, w['dni'], w['dhi'], w['ghi']) - P0),
                abs(P(tilt, panel_az + 10.0, zen, az, w['dni'], w['dhi'], w['ghi']) - P0),
                abs(P(tilt, panel_az - 10.0, zen, az, w['dni'], w['dhi'], w['ghi']) - P0))
    elif factor == 'solar_geometry':
        d = max(abs(P(tilt, panel_az, zen + 3.0, az, w['dni'], w['dhi'], w['ghi']) - P0),
                abs(P(tilt, panel_az, zen - 3.0, az, w['dni'], w['dhi'], w['ghi']) - P0),
                abs(P(tilt, panel_az, zen, az + 10.0, w['dni'], w['dhi'], w['ghi']) - P0),
                abs(P(tilt, panel_az, zen, az - 10.0, w['dni'], w['dhi'], w['ghi']) - P0))
    elif factor == 'irradiance':
        dni_p = w['dni'] * (1.0 + eps_frac)
        dhi_p = w['dhi'] * (1.0 + eps_frac)
        ghi_p = w['ghi'] * (1.0 + eps_frac)
        dni_m = w['dni'] * (1.0 - eps_frac)
        dhi_m = w['dhi'] * (1.0 - eps_frac)
        ghi_m = w['ghi'] * (1.0 - eps_frac)
        d = max(
            abs(P(tilt, panel_az, zen, az, dni_p, w['dhi'], w['ghi']) - P0),
            abs(P(tilt, panel_az, zen, az, dni_m, w['dhi'], w['ghi']) - P0),
            abs(P(tilt, panel_az, zen, az, w['dni'], dhi_p, w['ghi']) - P0),
            abs(P(tilt, panel_az, zen, az, w['dni'], dhi_m, w['ghi']) - P0),
            abs(P(tilt, panel_az, zen, az, w['dni'], w['dhi'], ghi_p) - P0),
            abs(P(tilt, panel_az, zen, az, w['dni'], w['dhi'], ghi_m) - P0),
        )
    else:
        d = 0.0
    return float(d)


def compute_power_factor_sensitivity(env, run_data, min_altitude_deg=5.0):
    """Mean one-at-a-time |ΔP| per factor over episode (pvlib-local sensitivity)."""
    inner = env.unwrapped
    accum = {f: [] for f in FACTOR_LABELS}
    for step_i, info in enumerate(run_data['infos']):
        info = dict(info)
        info['step'] = step_i
        if float(info.get('solar_altitude_deg', 0)) < min_altitude_deg:
            continue
        for factor in FACTOR_LABELS:
            accum[factor].append(_oat_delta_power(inner, info, factor))
    return {f: float(np.mean(v)) if v else 0.0 for f, v in accum.items()}


def normalize_percentages(values):
    total = float(sum(max(v, 0.0) for v in values.values()))
    if total <= 0:
        return {k: 0.0 for k in values}
    return {k: 100.0 * max(v, 0.0) / total for k, v in values.items()}


def correlation_power_shares(observations, powers):
    """Associative share: squared correlation of each obs dim with power (sums to 100%)."""
    obs = np.asarray(observations, dtype=np.float64)
    p = np.asarray(powers, dtype=np.float64).reshape(-1)
    if len(p) < 3:
        return {}
    shares = {}
    for i, name in enumerate(PHYSICAL_OBS_LABELS):
        col = obs[:, i]
        if np.std(col) < 1e-9 or np.std(p) < 1e-9:
            r2 = 0.0
        else:
            r = float(np.corrcoef(col, p)[0, 1])
            r2 = r * r
        shares[name] = r2
    total = sum(shares.values())
    if total <= 0:
        return {k: 0.0 for k in shares}
    return {k: 100.0 * v / total for k, v in shares.items()}


def analyze_intraday_state_variation(observations):
    """Per-feature temporal std and range (variation through the day)."""
    obs = np.asarray(observations, dtype=np.float64)
    return {
        'temporal_std': obs.std(axis=0),
        'temporal_range': obs.max(axis=0) - obs.min(axis=0),
        'mean_l1_step_change': np.mean(np.abs(np.diff(obs, axis=0)), axis=0)
            if len(obs) > 1 else np.zeros(obs.shape[1]),
    }


def compare_day_obs_trajectories(results, day_keys):
    """Pairwise mean |obs| difference between day types (sun-tracking rollouts).

    day_keys: {'clear': 'hist_clear_...', ...}
    """
    stats = {}
    obs_by_label = {}
    for label, result_key in day_keys.items():
        obs_by_label[label] = results[result_key]['sun']['observations']
        stats[label] = analyze_intraday_state_variation(obs_by_label[label])

    pairs = []
    labels = list(day_keys.keys())
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            a, b = labels[i], labels[j]
            diff = float(np.mean(np.abs(obs_by_label[a] - obs_by_label[b])))
            pairs.append((a, b, diff))
    return stats, pairs


def plot_state_variation_heatmap(outdir, stats_by_key):
    """Heatmap: rows=obs features, cols=day — cell = temporal std."""
    keys = sorted(stats_by_key.keys())
    mat = np.stack([stats_by_key[k]['temporal_std'] for k in keys], axis=1)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(mat, aspect='auto', cmap='YlOrRd')
    ax.set_xticks(np.arange(len(keys)))
    ax.set_xticklabels(keys, rotation=20, ha='right')
    ax.set_yticks(np.arange(len(PHYSICAL_OBS_LABELS)))
    ax.set_yticklabels(PHYSICAL_OBS_LABELS, fontsize=8)
    ax.set_title('Intra-day state variation (temporal std per feature)')
    fig.colorbar(im, ax=ax, label='std over episode steps')
    path = os.path.join(outdir, 'state_variation_heatmap.png')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_all_obs_trajectories_by_day(outdir, results, day_keys):
    """One subplot per observation feature; lines = day types."""
    n_feat = len(PHYSICAL_OBS_LABELS)
    n_cols = 3
    n_rows = int(np.ceil(n_feat / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, sharex=True, figsize=(14, 2.2 * n_rows))
    axes = np.atleast_2d(axes)
    colors = {'clear': '#2ca02c', 'cloudy': '#1f77b4', 'winter': '#ff7f0e'}
    for fi, feat in enumerate(PHYSICAL_OBS_LABELS):
        ax = axes[fi // n_cols, fi % n_cols]
        for label, result_key in sorted(day_keys.items()):
            obs = results[result_key]['sun']['observations']
            steps = np.arange(obs.shape[0])
            ax.plot(steps, obs[:, fi], lw=1.4, label=label, color=colors.get(label))
        ax.set_title(feat, fontsize=9)
        ax.grid(True, alpha=0.25)
    for fi in range(n_feat, n_rows * n_cols):
        axes[fi // n_cols, fi % n_cols].axis('off')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3, fontsize=9)
    fig.suptitle('Observation trajectories differ by day type (sun tracking)', fontsize=11)
    path = os.path.join(outdir, 'state_trajectories_by_feature.png')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_power_attribution_bars(outdir, oat_pct_by_key, title_suffix=''):
    keys = list(oat_pct_by_key.keys())
    factors = list(FACTOR_LABELS)
    x = np.arange(len(keys))
    w = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, factor in enumerate(factors):
        vals = [oat_pct_by_key[k].get(factor, 0.0) for k in keys]
        ax.bar(x + (i - 1) * w, vals, w, label=factor)
    ax.set_xticks(x)
    ax.set_xticklabels(keys, rotation=15, ha='right')
    ax.set_ylabel('% of total |ΔP| sensitivity')
    ax.set_title('Power sensitivity attribution (pvlib OAT)%s' % title_suffix)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    path = os.path.join(outdir, 'power_attribution_oat.png')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_correlation_share_bars(outdir, corr_pct, day_label):
    names = list(corr_pct.keys())
    vals = [corr_pct[n] for n in names]
    fig, ax = plt.subplots(figsize=(10, 5))
    y = np.arange(len(names))
    ax.barh(y, vals, color='#9467bd')
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel('% share (r² normalized)')
    ax.set_title('Associative power share by obs feature — %s' % day_label)
    ax.grid(axis='x', alpha=0.3)
    path = os.path.join(outdir, 'power_correlation_share_%s.png' % day_label.replace(' ', '_'))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def write_report(outdir, lines):
    path = os.path.join(outdir, 'state_space_verification.txt')
    with open(path, 'w', encoding='utf-8') as f:
        f.write('PV state-space verification (pvlib / PVTrackingEnv)\n')
        f.write('=' * 50 + '\n\n')
        for line in lines:
            f.write(line + '\n')
    return path


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--outdir', default=os.path.join(_REPO_ROOT, 'verification', 'pv_state_space'))
    p.add_argument('--cloudy-date', default=CLOUDY_DAY)
    p.add_argument('--clear-date', default=CLEAR_DAY)
    p.add_argument('--year-start', default='2020-01-01',
                   help='Start of full-year weather distribution scan')
    p.add_argument('--year-end', default='2020-12-31',
                   help='End of full-year weather distribution scan')
    p.add_argument('--skip-full-year-weather', action='store_true',
                   help='Skip full-year weather-type percentage plots')
    p.add_argument('--mode', choices=('tmy', 'nsrdb', 'both'), default='both',
                   help='tmy=fixed-date TMY checks; nsrdb=scenario MC baselines; both=default')
    p.add_argument('--nsrdb-mc-rollouts', type=int, default=12,
                   help='NSRDB paired MC rollouts for sun vs fixed (0=skip)')
    p.add_argument('--nsrdb-mc-seed-base', type=int, default=42)
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    report = []
    plot_paths = []

    report.append('Physical observation dim: %d' % len(PHYSICAL_OBS_LABELS))
    report.append('Features: %s' % ', '.join(PHYSICAL_OBS_LABELS))
    report.append('')

    run_tmy = args.mode in ('tmy', 'both')
    run_nsrdb = args.mode in ('nsrdb', 'both')

    # --- NSRDB multi-year: probabilistic baseline MC (sun vs fixed) ---
    if run_nsrdb and args.nsrdb_mc_rollouts > 0:
        if not os.path.isfile(default_manifest_path()):
            report.append('NSRDB MC: SKIP (manifest missing)')
            report.append('')
        else:
            mc_paths = run_nsrdb_mc_baselines(
                args.nsrdb_mc_rollouts, seed_base=args.nsrdb_mc_seed_base)
            report.extend(format_nsrdb_mc_report(mc_paths, args.nsrdb_mc_rollouts))
            plot_paths.append(plot_nsrdb_mc_baseline_comparison(args.outdir, mc_paths))

    # --- Full-year weather-type percentages (TMY catalog, episode window) ---
    if run_tmy and not args.skip_full_year_weather:
        fy_stats = compute_full_year_weather_distribution(
            start_date=args.year_start,
            end_date=args.year_end,
        )
        report.extend(format_weather_distribution_report(fy_stats))
        plot_paths.append(plot_full_year_weather_distribution(args.outdir, fy_stats))
        plot_paths.append(plot_monthly_weather_distribution(args.outdir, fy_stats))
        csv_path = os.path.join(args.outdir, 'full_year_weather_per_day.csv')
        fy_stats['per_day'].to_csv(csv_path, index=False)
        report.append('  Per-day CSV: %s' % csv_path)
        report.append('')

    ok_orient = True
    ok_days = True
    ok_obs = True
    ok_traj_diff = True

    if run_tmy:
        # --- Orientation affects power (deterministic single day — not MC) ---
        path, power_span = plot_orientation_sweep(
            args.outdir, args.cloudy_date, weather_source='historical')
        plot_paths.append(path)
        ok_orient = power_span > 10.0
        report.append('Orientation → power: tilt sweep span = %.1f W  [%s]' % (
            power_span, 'PASS' if ok_orient else 'FAIL (flat power vs tilt)'))
        report.append('')

    # --- Scenarios: historical cloudy/clear/winter + clearsky reference ---
    if not run_tmy:
        scenarios = {}
        results = {}
    else:
        scenarios = {
            'hist_clear_%s' % args.clear_date: (args.clear_date, 'historical'),
            'hist_cloudy_%s' % args.cloudy_date: (args.cloudy_date, 'historical'),
            'hist_winter_%s' % WINTER_DAY: (WINTER_DAY, 'historical'),
            'clearsky_%s' % args.clear_date: (args.clear_date, 'clearsky'),
        }
        results = {}
        for key, (date, wsrc) in scenarios.items():
            wsum = episode_weather_summary(date, wsrc)
            report.append('Weather %s | %s | ghi_ratio=%.3f overcast=%.0f%%' % (
                key, wsrc, wsum['ghi_ratio_mean'],
                100.0 * wsum['overcast_frac'] if np.isfinite(wsum['overcast_frac']) else 0))
            env = make_env(date, weather_source=wsrc)
            results[key] = {
                'sun': run_sun_tracking_episode(env, seed=0),
                'fixed': run_fixed_orientation_episode(env, seed=0),
            }
            env.close()
            report.append('  sun track energy=%.4f kWh  fixed=%.4f kWh  [single day, not MC]' % (
                results[key]['sun']['total_energy_kwh'],
                results[key]['fixed']['total_energy_kwh']))

        report.append('')
        plot_paths.append(plot_episode_comparison(args.outdir, results))
        plot_paths.append(plot_observation_variation(args.outdir, results))
        plot_paths.append(plot_irradiance_profiles(
            args.outdir, [args.clear_date, args.cloudy_date, WINTER_DAY]))
        plot_paths.append(plot_power_timeseries(args.outdir, results))

        day_keys = {
            'clear': 'hist_clear_%s' % args.clear_date,
            'cloudy': 'hist_cloudy_%s' % args.cloudy_date,
            'winter': 'hist_winter_%s' % WINTER_DAY,
        }
        stats_by_day, pair_diffs = compare_day_obs_trajectories(results, day_keys)

        report.append('Intra-day observation variation (sun tracking, historical):')
        report.append('  (temporal std = how much each feature changes through the episode)')
        for label, key in sorted(day_keys.items()):
            st = stats_by_day[label]
            report.append('  %s: mean temporal std=%.4f  mean range=%.4f  mean |Δobs/step|=%.4f' % (
                label,
                float(np.mean(st['temporal_std'])),
                float(np.mean(st['temporal_range'])),
                float(np.mean(st['mean_l1_step_change'])),
            ))
        report.append('')
        report.append('Pairwise |obs| difference (full trajectories):')
        ok_traj_diff = True
        for a, b, diff in pair_diffs:
            ok = diff > 0.02
            ok_traj_diff = ok_traj_diff and ok
            report.append('  %s vs %s: mean L1=%.4f  [%s]' % (
                a, b, diff, 'PASS' if ok else 'WARN'))
        report.append('')

        plot_paths.append(plot_state_variation_heatmap(args.outdir, stats_by_day))
        plot_paths.append(plot_all_obs_trajectories_by_day(args.outdir, results, day_keys))

        std_clear = stats_by_day['clear']['temporal_std']
        std_cloudy = stats_by_day['cloudy']['temporal_std']
        ghi_idx = PHYSICAL_OBS_LABELS.index('ghi_norm')
        ok_ghi_cloudy = std_cloudy[ghi_idx] < std_clear[ghi_idx] * 0.85 or std_cloudy[ghi_idx] > 0.01
        report.append('Cloudy vs clear ghi_norm temporal std: %.4f vs %.4f  [%s]' % (
            std_cloudy[ghi_idx], std_clear[ghi_idx],
            'PASS' if ok_ghi_cloudy else 'NOTE'))

        oat_pct_by_label = {}
        corr_paths = []
        for label, key in day_keys.items():
            date, wsrc = scenarios[key]
            env = make_env(date, weather_source=wsrc)
            sens = compute_power_factor_sensitivity(env, results[key]['sun'])
            env.close()
            oat_pct_by_label[label] = normalize_percentages(sens)
            report.append('Power sensitivity %% (OAT, %s day):' % label)
            for factor in FACTOR_LABELS:
                report.append('  %s: %.1f%%' % (factor, oat_pct_by_label[label][factor]))
            corr = correlation_power_shares(
                results[key]['sun']['observations'],
                results[key]['sun']['powers'],
            )
            report.append('Associative %% (corr(obs_i, power)², %s):' % label)
            for name in PHYSICAL_OBS_LABELS:
                report.append('  %s: %.1f%%' % (name, corr.get(name, 0.0)))
            corr_paths.append(plot_correlation_share_bars(
                args.outdir, corr, label))
        report.append('')
        report.append(
            'Note: OAT %% = local pvlib one-at-a-time perturbation (panel / sun geometry / irradiance).')
        report.append(
            '      Correlation %% = how much each obs dimension co-varies with power (not causal).')
        report.append(
            '      temperature_norm has 0%% OAT (not used in current power model).')
        plot_paths.append(plot_power_attribution_bars(args.outdir, oat_pct_by_label))
        plot_paths.extend(corr_paths)

        e_clear = results[day_keys['clear']]['sun']['total_energy_kwh']
        e_cloudy = results[day_keys['cloudy']]['sun']['total_energy_kwh']
        obs_clear = results[day_keys['clear']]['sun']['observations']
        obs_cloudy = results[day_keys['cloudy']]['sun']['observations']
        obs_diff = float(np.mean(np.abs(obs_clear - obs_cloudy)))
        ok_days = abs(e_clear - e_cloudy) > 0.05
        ok_obs = obs_diff > 0.02
        report.append('')
        report.append('Different days change energy: |E_clear - E_cloudy| = %.4f kWh  [%s]' % (
            abs(e_clear - e_cloudy), 'PASS' if ok_days else 'WARN'))
        report.append('Different days change |obs|: mean L1 = %.4f  [%s]' % (
            obs_diff, 'PASS' if ok_obs else 'WARN'))
        report.append('')
        report.append('Recommended cloudy training date (historical TMY): %s' % args.cloudy_date)
        report.append('  (low GHI/clearsky ratio in episode window; use weather_source=historical)')
        report.append('')
    report.append('Plots:')
    for pth in plot_paths:
        report.append('  %s' % pth)

    report_path = write_report(args.outdir, report)
    print('[verify_pv_state_space] report: %s' % report_path)
    for line in report[:35]:
        print(' ', line)
    print(' ...')
    if run_tmy:
        ok_all = ok_orient and ok_days and ok_obs and ok_traj_diff
    else:
        ok_all = True
    return 0 if ok_all else 1


if __name__ == '__main__':
    sys.exit(main())
