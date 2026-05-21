"""Shared helpers for PV tracking agent evaluation scripts."""

from __future__ import division, print_function

import json
import os
from collections import defaultdict, OrderedDict

import numpy as np

from softlearning.environments.utils import get_environment_from_params


# Matches mbpo/env/pv_tracking.py observation layout.
PV_OBS_LABELS = (
    'solar_zenith_norm',
    'solar_azimuth_sin',
    'solar_azimuth_cos',
    'dni_norm',
    'dhi_norm',
    'ghi_norm',
    'temperature_norm',
    'panel_tilt_norm',
    'panel_azimuth_sin',
    'panel_azimuth_cos',
    'last_power_norm',
    'time_sin',
    'time_cos',
    'day_sin',
    'day_cos',
)


def deep_update(original, override):
    if not isinstance(override, dict):
        return override
    updated = dict(original)
    for key, value in override.items():
        if key in updated and isinstance(updated[key], dict):
            updated[key] = deep_update(updated[key], value)
        else:
            updated[key] = value
    return updated


def season_from_day_of_year(day_of_year):
    day = int(day_of_year)
    if 80 <= day <= 171:
        return 'spring'
    if 172 <= day <= 263:
        return 'summer'
    if 264 <= day <= 354:
        return 'fall'
    return 'winter'


def normalize_angle_diff(target, current):
    return (target - current + 180.0) % 360.0 - 180.0


def decode_pv_observation(obs):
    """Decode normalized PVTracking observation to physical units."""
    obs = np.asarray(obs, dtype=np.float64).reshape(-1)
    solar_azimuth = np.rad2deg(
        np.arctan2(obs[1], obs[2])) % 360.0
    panel_azimuth = np.rad2deg(
        np.arctan2(obs[8], obs[9])) % 360.0
    angle = np.mod(np.arctan2(obs[11], obs[12]), 2.0 * np.pi)
    time_of_day = angle * 24.0 / (2.0 * np.pi)
    return {
        'solar_zenith_deg': float(obs[0] * 180.0),
        'solar_azimuth_deg': float(solar_azimuth),
        'panel_tilt_deg': float(obs[7] * 90.0),
        'panel_azimuth_deg': float(panel_azimuth),
        'time_of_day_hour': float(time_of_day),
        'dni_norm': float(obs[3]),
        'ghi_norm': float(obs[5]),
    }


TIME_WINDOWS = OrderedDict([
    ('morning', (6.0, 11.0)),
    ('midday', (11.0, 14.0)),
    ('afternoon', (14.0, 17.0)),
    ('evening', (17.0, 22.0)),
])


def get_eval_environment(
        variant,
        override_path=None,
        test_start_date=None,
        test_end_date=None,
        fixed_eval_dates=None,
        eval_randomize_day=None,
        eval_weather_source=None,
        eval_randomize_initial_orientation=False):
    """Build evaluation env; defaults favor diverse held-out-style testing."""
    environment_params = variant['environment_params']
    eval_env_params = (
        environment_params.get('evaluation')
        if 'evaluation' in environment_params
        else environment_params['training'])

    if override_path is not None:
        with open(override_path, 'r', encoding='utf-8') as f:
            override = json.load(f)
        eval_env_params = deep_update(eval_env_params, override)

    eval_env_params = deep_update(eval_env_params, {'kwargs': {}})
    kwargs = eval_env_params['kwargs']

    if fixed_eval_dates:
        kwargs['fixed_eval_dates'] = [
            d.strip() for d in fixed_eval_dates.split(',') if d.strip()]
        kwargs['randomize_day'] = False
    elif test_start_date or test_end_date:
        if test_start_date:
            kwargs['start_date'] = test_start_date
        if test_end_date:
            kwargs['end_date'] = test_end_date
        kwargs['randomize_day'] = True
    elif eval_randomize_day is not None:
        kwargs['randomize_day'] = bool(eval_randomize_day)
    else:
        # Default: random days across configured range (diverse generalization test).
        kwargs['randomize_day'] = True

    if eval_weather_source is not None:
        kwargs['weather_source'] = eval_weather_source

    kwargs['randomize_initial_orientation'] = bool(eval_randomize_initial_orientation)

    eval_env_params['kwargs'] = kwargs
    return get_environment_from_params(eval_env_params), eval_env_params


def describe_eval_config(eval_env_params):
    """Human-readable summary of evaluation environment settings."""
    kwargs = eval_env_params.get('kwargs', {})
    lines = [
        'start_date: %s' % kwargs.get('start_date', '(from env default)'),
        'end_date: %s' % kwargs.get('end_date', '(from env default)'),
        'randomize_day: %s' % kwargs.get('randomize_day', True),
        'randomize_initial_orientation: %s' % kwargs.get(
            'randomize_initial_orientation', False),
        'weather_source: %s' % kwargs.get('weather_source', '(from env default)'),
        'fixed_eval_dates: %s' % kwargs.get('fixed_eval_dates', None),
        'periods: %s' % kwargs.get('periods', '(from env default)'),
        'freq: %s' % kwargs.get('freq', '(from env default)'),
    ]
    return lines


def validate_eval_coverage(num_rollouts, eval_env_params, min_rollouts=10):
    """Return warning strings when evaluation may be too narrow."""
    warnings = []
    kwargs = eval_env_params.get('kwargs', {})
    fixed_dates = kwargs.get('fixed_eval_dates') or []
    if kwargs.get('randomize_day', True) and num_rollouts < min_rollouts:
        warnings.append(
            'Only %d rollouts with randomize_day=True; use at least %d for '
            'stable estimates across days/weather.' % (num_rollouts, min_rollouts))
    if fixed_dates and num_rollouts > len(fixed_dates):
        warnings.append(
            'num_rollouts=%d exceeds %d fixed_eval_dates; some dates will repeat.'
            % (num_rollouts, len(fixed_dates)))
    if not kwargs.get('randomize_day', True) and not fixed_dates:
        warnings.append(
            'randomize_day=False and no fixed_eval_dates: all rollouts may use '
            'the same calendar day.')
    return warnings


def _extract_underlying_env(env):
    if hasattr(env, 'unwrapped'):
        return env.unwrapped
    if hasattr(env, '_env'):
        return env._env
    return env


def make_baseline_rollout(env, baseline_type, path_length, seed=None):
    """Run a baseline policy with correct PV observation decoding."""
    underlying = _extract_underlying_env(env)
    if seed is not None and hasattr(env, 'seed'):
        env.seed(seed)

    observations = []
    actions = []
    rewards = []
    terminals = []
    next_observations = []
    infos = []

    obs = env.reset()
    done = False
    step = 0
    while step < path_length and not done:
        decoded = decode_pv_observation(obs)
        solar_zenith = decoded['solar_zenith_deg']
        solar_azimuth = decoded['solar_azimuth_deg']
        current_tilt = decoded['panel_tilt_deg']
        current_azimuth = decoded['panel_azimuth_deg']

        if baseline_type in ('fixed', 'fixed_no_motion'):
            target_tilt = 30.0
            target_azimuth = 180.0
        elif baseline_type == 'single_axis':
            target_tilt = 30.0
            target_azimuth = solar_azimuth
        elif baseline_type in ('sun_seeking', 'sun_tracking'):
            target_tilt = solar_zenith
            target_azimuth = solar_azimuth
        else:
            raise ValueError('Unknown baseline type: %s' % baseline_type)

        delta_tilt = np.clip(
            target_tilt - current_tilt,
            -underlying.max_delta_tilt,
            underlying.max_delta_tilt,
        )
        delta_azimuth = np.clip(
            normalize_angle_diff(target_azimuth, current_azimuth),
            -underlying.max_delta_azimuth,
            underlying.max_delta_azimuth,
        )

        action = np.array([
            delta_tilt / underlying.max_delta_tilt,
            delta_azimuth / underlying.max_delta_azimuth,
        ], dtype=np.float32)

        next_obs, reward, terminal, info = env.step(action)

        observations.append(obs)
        actions.append(action)
        rewards.append(reward)
        terminals.append(terminal)
        next_observations.append(next_obs)
        infos.append(info)

        obs = next_obs
        done = terminal
        step += 1

    return {
        'observations': np.asarray(observations),
        'actions': np.asarray(actions),
        'rewards': np.asarray(rewards),
        'terminals': np.asarray(terminals),
        'next_observations': np.asarray(next_observations),
        'infos': infos,
    }


def compute_total_energy_kwh(path):
    infos = path.get('infos', [])
    if not infos:
        return np.nan
    energies = [info.get('energy_kwh', np.nan) for info in infos]
    if all(np.isfinite(e) for e in energies):
        return float(np.sum(energies))
    power = np.array([info.get('power', np.nan) for info in infos], dtype=np.float64)
    if len(power) == 0:
        return np.nan
    return float(np.sum(power) * 0.25 / 1000.0)


def _index_at_max(arr):
    arr = np.asarray(arr, dtype=np.float64)
    if len(arr) == 0:
        return 0
    return int(np.nanargmax(arr))


def analyze_rollout_path(path):
    """Per-rollout timing diagnostics for reward/power alignment."""
    infos = path.get('infos', [])
    rewards = np.asarray(path.get('rewards', []), dtype=np.float64)
    if len(rewards) == 0:
        return {}

    times = np.array([info.get('time', np.nan) for info in infos], dtype=np.float64)
    power = np.array([info.get('power', np.nan) for info in infos], dtype=np.float64)
    energy = np.array([info.get('energy_kwh', np.nan) for info in infos], dtype=np.float64)
    movement = np.array([info.get('movement_cost', 0.0) for info in infos], dtype=np.float64)

    i_reward = _index_at_max(rewards)
    i_power = _index_at_max(power)
    i_energy = _index_at_max(energy)

    def _snap(i):
        info = infos[i] if i < len(infos) else {}
        return {
            'step': i,
            'time_hour': float(info.get('time', np.nan)),
            'reward': float(rewards[i]),
            'power_w': float(info.get('power', np.nan)),
            'energy_kwh': float(info.get('energy_kwh', np.nan)),
            'movement_cost': float(info.get('movement_cost', np.nan)),
            'tilt_deg': float(info.get('tilt', np.nan)),
            'azimuth_deg': float(info.get('azimuth', np.nan)),
            'solar_altitude_deg': float(info.get('solar_altitude_deg', np.nan)),
            'solar_azimuth_deg': float(info.get('solar_azimuth_deg', np.nan)),
        }

    by_window = {}
    for name, (lo, hi) in TIME_WINDOWS.items():
        mask = (times >= lo) & (times < hi)
        if not np.any(mask):
            by_window[name] = {
                'mean_reward': np.nan, 'mean_power_w': np.nan,
                'sum_energy_kwh': np.nan, 'sum_movement_cost': np.nan, 'n_steps': 0,
            }
            continue
        by_window[name] = {
            'mean_reward': float(np.mean(rewards[mask])),
            'mean_power_w': float(np.mean(power[mask])),
            'sum_energy_kwh': float(np.sum(energy[mask])),
            'sum_movement_cost': float(np.sum(movement[mask])),
            'n_steps': int(np.sum(mask)),
        }

    meta = get_rollout_metadata(path)
    peak_power_time = _snap(i_power)['time_hour']
    peak_reward_time = _snap(i_reward)['time_hour']
    meta.update({
        'mean_power_w': float(np.nanmean(power)),
        'peak_power_w': float(np.nanmax(power)),
        'peak_power_time_hour': peak_power_time,
        'peak_reward_time_hour': peak_reward_time,
        'peak_reward_step': i_reward,
        'peak_power_step': i_power,
        'at_peak_reward': _snap(i_reward),
        'at_peak_power': _snap(i_power),
        'reward_by_window': by_window,
        'peak_reward_lag_hours': (
            float(peak_reward_time - peak_power_time)
            if np.isfinite(peak_reward_time) and np.isfinite(peak_power_time)
            else np.nan),
    })
    return meta


def aggregate_time_windows(paths):
    """Average per-window stats across rollouts."""
    agg = {name: defaultdict(list) for name in TIME_WINDOWS}
    for path in paths:
        analysis = analyze_rollout_path(path)
        for name, stats in analysis.get('reward_by_window', {}).items():
            for key, val in stats.items():
                if key != 'n_steps' and np.isfinite(val):
                    agg[name][key].append(val)
    summary = {}
    for name, buckets in agg.items():
        summary[name] = {
            key: float(np.mean(vals)) if vals else np.nan
            for key, vals in buckets.items()
        }
    return summary


def compare_method_table(paths_by_name):
    """Build comparison rows for policy vs baselines."""
    rows = []
    for method, paths in paths_by_name.items():
        analyses = [analyze_rollout_path(p) for p in paths]
        rows.append({
            'method': method,
            'n_rollouts': len(paths),
            'total_reward_mean': float(np.mean([a['total_reward'] for a in analyses])),
            'total_energy_kwh_mean': float(np.mean([a['total_energy_kwh'] for a in analyses])),
            'mean_power_w_mean': float(np.mean([a['mean_power_w'] for a in analyses])),
            'peak_power_w_mean': float(np.mean([a['peak_power_w'] for a in analyses])),
            'peak_power_time_mean': float(np.mean([a['peak_power_time_hour'] for a in analyses])),
            'peak_reward_time_mean': float(np.mean([a['peak_reward_time_hour'] for a in analyses])),
            'movement_cost_mean': float(np.mean([a['total_movement_cost'] for a in analyses])),
            'tilt_mean': float(np.mean([
                np.nanmean([info.get('tilt', np.nan) for info in p.get('infos', [])])
                for p in paths])),
            'azimuth_mean': float(np.mean([
                np.nanmean([info.get('azimuth', np.nan) for info in p.get('infos', [])])
                for p in paths])),
        })
    return rows


def get_rollout_metadata(path):
    infos = path.get('infos', [])
    info0 = infos[0] if infos else {}
    day_of_year = info0.get('day_of_year')
    season = info0.get('season') or (
        season_from_day_of_year(day_of_year)
        if day_of_year is not None else 'unknown')
    return {
        'date': info0.get('date', 'unknown'),
        'day_of_year': day_of_year,
        'season': season,
        'weather_condition': info0.get('weather_condition', 'unknown'),
        'weather_source': info0.get('weather_source', 'unknown'),
        'episode_length': len(path.get('rewards', [])),
        'total_reward': float(np.sum(path.get('rewards', []))),
        'total_energy_kwh': compute_total_energy_kwh(path),
        'total_movement_cost': float(np.sum(
            [info.get('movement_cost', 0.0) for info in infos])) if infos else np.nan,
    }


def summarize_values(values):
    values = np.asarray(values, dtype=np.float64)
    return OrderedDict([
        ('count', int(len(values))),
        ('mean', float(np.mean(values))),
        ('std', float(np.std(values))),
        ('min', float(np.min(values))),
        ('max', float(np.max(values))),
    ])


def summarize_paths(paths):
    rewards = [float(np.sum(p['rewards'])) for p in paths]
    lengths = [len(p['rewards']) for p in paths]
    energies = [get_rollout_metadata(p)['total_energy_kwh'] for p in paths]
    movements = [get_rollout_metadata(p)['total_movement_cost'] for p in paths]
    return OrderedDict([
        ('reward', summarize_values(rewards)),
        ('episode_length', summarize_values(lengths)),
        ('total_energy_kwh', summarize_values(energies)),
        ('total_movement_cost', summarize_values(movements)),
    ])


def summarize_by_group(paths, key_fn):
    groups = defaultdict(list)
    for path in paths:
        meta = get_rollout_metadata(path)
        groups[key_fn(meta)].append(path)
    return {
        group: summarize_paths(group_paths)
        for group, group_paths in sorted(groups.items())
    }


def rollout_time_axis(path, relative=False):
    """Hours for plotting; default is absolute time-of-day from env info."""
    infos = path.get('infos', [])
    if not infos or 'time' not in infos[0]:
        return np.arange(len(path['rewards']), dtype=np.float64)
    times = np.array([info.get('time', np.nan) for info in infos], dtype=np.float64)
    if not np.isfinite(times).all():
        return np.arange(len(path['rewards']), dtype=np.float64)
    if relative:
        return times - times[0]
    return times


def write_reward_time_report(outdir, paths, paths_by_name=None):
    """Text report: reward vs power timing and time-window breakdown."""
    report_path = os.path.join(outdir, 'reward_time_analysis.txt')
    window_agg = aggregate_time_windows(paths)

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('PV Tracking Reward-Time Analysis\n')
        f.write('=' * 36 + '\n\n')
        f.write(
            'Reward per step = energy_kwh - movement_cost. '
            'Peak step reward often occurs when movement is low, not when power is highest.\n\n')

        f.write('Learned policy — per rollout peak times:\n')
        for idx, path in enumerate(paths, 1):
            a = analyze_rollout_path(path)
            f.write(
                '  rollout_%d (%s): peak_power=%.1f W @ %.2f h | '
                'peak_reward=%.5f @ %.2f h | lag=%.2f h\n' % (
                    idx, a.get('date', '?'),
                    a.get('peak_power_w', np.nan), a.get('peak_power_time_hour', np.nan),
                    a['at_peak_reward']['reward'], a.get('peak_reward_time_hour', np.nan),
                    a.get('peak_reward_lag_hours', np.nan)))
            f.write(
                '    at peak reward: tilt=%.1f az=%.1f solar_alt=%.1f solar_az=%.1f move=%.5f\n' % (
                    a['at_peak_reward']['tilt_deg'], a['at_peak_reward']['azimuth_deg'],
                    a['at_peak_reward']['solar_altitude_deg'],
                    a['at_peak_reward']['solar_azimuth_deg'],
                    a['at_peak_reward']['movement_cost']))

        f.write('\nLearned policy — mean by time window (avg over rollouts):\n')
        f.write('  window      mean_reward  mean_power_W  sum_energy   sum_movement\n')
        for name in TIME_WINDOWS:
            w = window_agg.get(name, {})
            f.write('  %-10s  %11.5f  %11.1f  %10.5f  %12.5f\n' % (
                name,
                w.get('mean_reward', np.nan),
                w.get('mean_power_w', np.nan),
                w.get('sum_energy_kwh', np.nan),
                w.get('sum_movement_cost', np.nan)))

        if paths_by_name:
            f.write('\nMethod comparison (same eval settings):\n')
            rows = compare_method_table(paths_by_name)
            f.write(
                '  method           reward    energy    mean_pwr  peak_pwr  '
                'peak_pwr_t  peak_rew_t  movement\n')
            for row in rows:
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

        f.write('\nInterpretation:\n')
        f.write(
            '  - If evening mean_reward is highest while midday mean_power is low, the agent '
            'may be minimizing movement (low penalty) rather than maximizing energy.\n')
        f.write(
            '  - Compare total_energy_kwh and peak_power_time across methods; '
            'reward alone is not a proxy for tracking quality.\n')

    return report_path


def save_rollout_csv(outdir, paths, prefix='rollout'):
    os.makedirs(outdir, exist_ok=True)
    for idx, path in enumerate(paths, start=1):
        observations = np.asarray(path['observations'])
        actions = np.asarray(path['actions'])
        rewards = np.asarray(path['rewards'])
        terminals = np.asarray(path.get('terminals', [False] * len(rewards)))
        infos = path.get('infos', [])

        header = [
            'step', 'time_hour', 'reward', 'terminal',
            'power_w', 'energy_kwh', 'movement_cost',
            'reward_energy', 'reward_movement',
            'tilt_deg', 'azimuth_deg',
            'solar_zenith_deg', 'solar_azimuth_deg', 'solar_altitude_deg',
            'date', 'season', 'weather_condition', 'weather_source',
            'action_tilt', 'action_azimuth',
        ]
        header += list(PV_OBS_LABELS)

        rows = []
        for t in range(len(rewards)):
            info = infos[t] if t < len(infos) else {}
            row = [
                t,
                info.get('time', ''),
                float(rewards[t]),
                bool(terminals[t]),
                info.get('power', ''),
                info.get('energy_kwh', ''),
                info.get('movement_cost', ''),
                info.get('reward_energy', info.get('energy_kwh', '')),
                info.get('reward_movement', info.get('movement_cost', '')),
                info.get('tilt', ''),
                info.get('azimuth', ''),
                info.get('solar_zenith_deg', ''),
                info.get('solar_azimuth_deg', ''),
                info.get('solar_altitude_deg', ''),
                info.get('date', ''),
                info.get('season', ''),
                info.get('weather_condition', ''),
                info.get('weather_source', ''),
                float(actions[t][0]) if actions.ndim == 2 else actions[t],
                float(actions[t][1]) if actions.ndim == 2 else '',
            ]
            if observations.ndim == 2:
                row.extend(observations[t].tolist())
            else:
                row.append(float(observations[t]))
            rows.append(row)

        csv_path = os.path.join(outdir, '%s_%d.csv' % (prefix, idx))
        with open(csv_path, 'w', encoding='utf-8') as f:
            f.write(','.join(header) + '\n')
            for row in rows:
                f.write(','.join(str(x) for x in row) + '\n')
    return outdir
