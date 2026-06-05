"""Shared helpers for PV tracking agent evaluation scripts."""

from __future__ import division, print_function

import json
import os
from collections import defaultdict, OrderedDict

import numpy as np

from softlearning.environments.utils import get_environment_from_params

try:
    from mbpo.env.pv_tracking import (
        PV_TIMEZONE as _ENV_PV_TIMEZONE,
        DEFAULT_START_TIME as _ENV_START_TIME,
        DEFAULT_PERIODS as _ENV_PERIODS,
        DEFAULT_FREQ as _ENV_FREQ,
        DEFAULT_EPISODE_STEPS as _ENV_EPISODE_STEPS,
    )
except ImportError:
    _ENV_PV_TIMEZONE = 'UTC'
    _ENV_START_TIME = '13:30'
    _ENV_PERIODS = 40
    _ENV_FREQ = '15min'
    _ENV_EPISODE_STEPS = 39


# Matches mbpo/env/pv_tracking.py (legacy 15-D layout).
PV_OBS_LABELS_LEGACY = (
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
    'power_norm',
    'time_of_day_sin',
    'time_of_day_cos',
    'day_of_year_sin',
    'day_of_year_cos',
)

PV_OBS_LABELS_PHYSICAL = (
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
    'cos_aoi',
)

# Backward-compatible alias for legacy evaluation scripts.
PV_OBS_LABELS = PV_OBS_LABELS_LEGACY


def pv_obs_labels_for_vector(obs):
    dim = int(np.asarray(obs).reshape(-1).shape[0])
    if dim == len(PV_OBS_LABELS_PHYSICAL):
        return PV_OBS_LABELS_PHYSICAL
    return PV_OBS_LABELS_LEGACY


PHYSICAL_OBS_DIM = len(PV_OBS_LABELS_PHYSICAL)
LEGACY_OBS_DIM = len(PV_OBS_LABELS_LEGACY)

# Single source: mbpo/env/pv_tracking.py (pvlib Location.tz + DatetimeIndex grid).
PV_TIMEZONE = _ENV_PV_TIMEZONE
PV_EPISODE_START_TIME = _ENV_START_TIME
PV_EPISODE_PERIODS = _ENV_PERIODS
PV_EPISODE_FREQ = _ENV_FREQ
PV_EPISODE_MAX_STEPS = _ENV_EPISODE_STEPS
# Default site (must match env __init__ defaults).
PV_DEFAULT_LATITUDE = 35.0
PV_DEFAULT_LONGITUDE = -106.0
PV_DEFAULT_ALTITUDE = 1600.0

EVAL_PROTOCOL_INHERIT = 'inherit'
EVAL_PROTOCOL_LEGACY_UTC = 'legacy_utc'
EVAL_PROTOCOL_UTC = 'utc'
# Deprecated: raises if used (different MDP; breaks checkpoint comparability).
EVAL_PROTOCOL_LOCAL_DAY = 'local_day'

UTC_EPISODE_GRID = {
    'tz': PV_TIMEZONE,
    'start_time': PV_EPISODE_START_TIME,
    'periods': PV_EPISODE_PERIODS,
    'freq': PV_EPISODE_FREQ,
}

def observation_mode_for_dim(dim):
    if dim == PHYSICAL_OBS_DIM:
        return 'physical'
    if dim == LEGACY_OBS_DIM:
        return 'legacy'
    return None


def infer_policy_input_dim(policy_weights=None, policy=None):
    """First-layer weight rows == policy input dimension."""
    if policy_weights:
        weight0 = policy_weights[0]
        if hasattr(weight0, 'shape') and len(weight0.shape) == 2:
            return int(weight0.shape[0])
    if policy is not None:
        try:
            model = policy.deterministic_actions_model
            shape = model.inputs[0].shape
            dim = shape[-1].value if hasattr(shape[-1], 'value') else shape[-1]
            if dim is not None:
                return int(dim)
        except Exception:
            pass
    raise ValueError('Unable to infer policy input dimension from weights or policy.')


def env_raw_observation_dim(env):
    """Raw Box observation size and underlying PV observation_mode."""
    inner = _extract_underlying_env(env)
    env_mode = getattr(inner, 'observation_mode', None)
    if hasattr(env, 'observation_space') and hasattr(env.observation_space, 'shape'):
        dim = int(np.prod(env.observation_space.shape))
    else:
        dim = int(np.prod(env.active_observation_shape))
    return dim, env_mode


def validate_policy_environment_observation_dims(
        policy, env, policy_weights=None, eval_env_params=None):
    """Raise if checkpoint policy input dim != env observation dim."""
    policy_dim = infer_policy_input_dim(policy_weights=policy_weights, policy=policy)
    env_dim, env_mode = env_raw_observation_dim(env)
    active_dim = int(np.prod(getattr(env, 'active_observation_shape', (env_dim,))))
    required_mode = observation_mode_for_dim(policy_dim)

    kwargs = {}
    if eval_env_params:
        kwargs = eval_env_params.get('kwargs', {})
    configured_mode = kwargs.get('observation_mode')

    if policy_dim != env_dim or policy_dim != active_dim:
        raise ValueError(
            'Policy observation dim (%d) does not match evaluation env dim '
            '(raw=%d, active=%d, env observation_mode=%r, configured=%r). '
            'This checkpoint requires observation_mode=%r. Do not slice, pad, '
            'or reorder observations to force a match.' % (
                policy_dim, env_dim, active_dim, env_mode, configured_mode,
                required_mode))
    if required_mode and configured_mode and configured_mode != required_mode:
        raise ValueError(
            'Configured observation_mode=%r does not match checkpoint dim %d '
            '(expected %r).' % (configured_mode, policy_dim, required_mode))
    if required_mode and env_mode and env_mode != required_mode:
        raise ValueError(
            'Environment observation_mode=%r does not match checkpoint dim %d '
            '(expected %r).' % (env_mode, policy_dim, required_mode))
    return {
        'policy_dim': policy_dim,
        'env_dim': env_dim,
        'active_dim': active_dim,
        'env_mode': env_mode or required_mode,
        'required_mode': required_mode,
    }


def prepare_policy_observation_batch(raw_observation, expected_dim, env=None):
    """Build a numeric batch of shape (1, expected_dim) for policy.predict.

    Never pass [vector] as a Python list — Keras may treat it as shape (1,).
    """
    if env is not None:
        convert = getattr(env, 'convert_to_active_observation', None)
        if convert is not None:
            raw_observation = convert(raw_observation)
    vec = np.asarray(raw_observation, dtype=np.float32).reshape(-1)
    if vec.size != int(expected_dim):
        raise ValueError(
            'Observation size mismatch for policy batch: expected %d-D vector, '
            'got shape %s (size=%d). Use prepare_policy_observation_batch() and '
            'do not wrap the vector in a bare Python list for policy.actions_np.' % (
                expected_dim, vec.shape, vec.size))
    return vec.reshape(1, int(expected_dim))


def _observation_debug_preview(vec, max_items=6):
    vec = np.asarray(vec, dtype=np.float64).reshape(-1)
    preview = ', '.join('%.4f' % x for x in vec[:max_items])
    if vec.size > max_items:
        preview += ', ...'
    return preview


def log_first_rollout_observation_debug(stage, env, observation, policy_dim, info=None):
    """One-time debug print for the first rollout (Issue 1 audit)."""
    inner = _extract_underlying_env(env)
    raw = np.asarray(observation, dtype=np.float64).reshape(-1)
    if raw.size == policy_dim + 1 and raw.shape[0] == 1:
        print('[obs_debug] %s: note — received batch shape %s; expected vector (%d,).' % (
            stage, raw.shape, policy_dim))
    print('[obs_debug] %s:' % stage)
    print('  policy_expected_dim=%d' % policy_dim)
    print('  env.observation_mode=%r' % getattr(inner, 'observation_mode', None))
    print('  env.observation_space.shape=%s' % (getattr(env.observation_space, 'shape', None),))
    print('  env.active_observation_shape=%s' % (getattr(env, 'active_observation_shape', None),))
    print('  obs type=%s shape=%s dtype=%s' % (type(observation), raw.shape, raw.dtype))
    print('  obs preview=[%s]' % _observation_debug_preview(raw))
    if info:
        print('  info time=%s tz=%s solar_alt=%s power=%s' % (
            info.get('time'), info.get('timezone'),
            info.get('solar_altitude_deg'), info.get('power')))


def run_learned_policy_rollout(
        policy,
        env,
        path_length,
        seed=None,
        deterministic=True,
        policy_input_dim=None,
        debug_first_step=False):
    """Roll out a trained policy with validated (1, dim) observation batches."""
    if policy_input_dim is None:
        policy_input_dim = int(np.prod(env.active_observation_shape))

    if seed is not None and hasattr(env, 'seed'):
        env.seed(seed)

    observations = []
    actions = []
    rewards = []
    infos = []

    obs = env.reset()
    if debug_first_step:
        log_first_rollout_observation_debug('reset', env, obs, policy_input_dim)

    done = False
    step = 0
    with policy.set_deterministic(deterministic):
        while not done and step < path_length:
            batch = prepare_policy_observation_batch(obs, policy_input_dim, env=env)
            if debug_first_step and step == 0:
                log_first_rollout_observation_debug(
                    'pre_policy_batch', env, batch, policy_input_dim)
            action = policy.actions_np(batch)[0]
            next_obs, reward, done, info = env.step(action)
            if debug_first_step and step == 0:
                log_first_rollout_observation_debug(
                    'post_step', env, next_obs, policy_input_dim, info=info)

            observations.append(obs)
            actions.append(action)
            rewards.append(reward)
            infos.append(info)
            obs = next_obs
            step += 1

    return {
        'observations': np.asarray(observations),
        'actions': np.asarray(actions),
        'rewards': np.asarray(rewards),
        'infos': infos,
    }


def apply_pv_utc_schedule(kwargs):
    """Force unified UTC episode grid for train, test, baselines, and plots."""
    merged = dict(kwargs)
    merged.update(UTC_EPISODE_GRID)
    return merged


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
    """Equinox-based season label (matches PVTrackingEnv.step info['season'])."""
    day = int(day_of_year)
    if 80 <= day <= 171:
        return 'spring'
    if 172 <= day <= 263:
        return 'summer'
    if 264 <= day <= 354:
        return 'fall'
    return 'winter'


def season_from_calendar_date(date_str):
    """Calendar-month season for post-train reporting (Dec–Feb winter, etc.).

    Differs from env ``season`` before ~Jun 21: e.g. 2020-06-01 is env-spring
    (day 153 < 172) but calendar-summer. Use this for seasonal bar charts.
    """
    import pandas as pd
    month = int(pd.Timestamp(str(date_str)).month)
    if month in (12, 1, 2):
        return 'winter'
    if month in (3, 4, 5):
        return 'spring'
    if month in (6, 7, 8):
        return 'summer'
    return 'fall'


SEASON_CALENDAR_ORDER = ('winter', 'spring', 'summer', 'fall')


def normalize_angle_diff(target, current):
    return (target - current + 180.0) % 360.0 - 180.0


def decode_pv_observation(obs):
    """Decode normalized PVTracking observation to physical units."""
    obs = np.asarray(obs, dtype=np.float64).reshape(-1)
    solar_azimuth = np.rad2deg(
        np.arctan2(obs[1], obs[2])) % 360.0
    panel_azimuth = np.rad2deg(
        np.arctan2(obs[8], obs[9])) % 360.0
    decoded = {
        'solar_zenith_deg': float(obs[0] * 180.0),
        'solar_azimuth_deg': float(solar_azimuth),
        'panel_tilt_deg': float(obs[7] * 90.0),
        'panel_azimuth_deg': float(panel_azimuth),
        'dni_norm': float(obs[3]),
        'ghi_norm': float(obs[5]),
    }
    if obs.shape[0] == len(PV_OBS_LABELS_PHYSICAL):
        decoded['cos_aoi'] = float(obs[10])
    else:
        angle = np.mod(np.arctan2(obs[11], obs[12]), 2.0 * np.pi)
        decoded['time_of_day_hour'] = float(angle * 24.0 / (2.0 * np.pi))
        decoded['power_norm'] = float(obs[10])
    return decoded


# Clock-hour bins in UTC on the daylight episode grid (13:30–23:15 UTC).
# Use SOLAR_ALTITUDE_WINDOWS for sun-up / peak-sun physics.
TIME_WINDOWS = OrderedDict([
    ('morning', (13.5, 16.5)),
    ('midday', (16.5, 19.0)),
    ('afternoon', (19.0, 21.5)),
    ('evening', (21.5, 23.5)),
])

# Physics-based bins (degrees solar altitude); independent of clock labels.
SOLAR_ALTITUDE_WINDOWS = OrderedDict([
    ('night_alt_lt_5', (None, 5.0)),
    ('low_sun_5_20', (5.0, 20.0)),
    ('mid_sun_20_30', (20.0, 30.0)),
    ('high_sun_30_plus', (30.0, None)),
])


def env_timezone_from_paths(paths):
    for path in paths:
        infos = path.get('infos', [])
        if infos and infos[0].get('timezone'):
            return str(infos[0]['timezone'])
    return 'UTC'


def _mask_solar_altitude(altitudes, lo, hi):
    altitudes = np.asarray(altitudes, dtype=np.float64)
    mask = np.isfinite(altitudes)
    if lo is not None:
        mask &= altitudes >= lo
    if hi is not None:
        mask &= altitudes < hi
    return mask


def _aggregate_window_stats(rewards, power, energy, movement, mask):
    if not np.any(mask):
        return {
            'mean_reward': np.nan, 'mean_power_w': np.nan,
            'sum_energy_kwh': np.nan, 'sum_movement_cost': np.nan, 'n_steps': 0,
        }
    return {
        'mean_reward': float(np.mean(rewards[mask])),
        'mean_power_w': float(np.mean(power[mask])),
        'sum_energy_kwh': float(np.sum(energy[mask])),
        'sum_movement_cost': float(np.sum(movement[mask])),
        'n_steps': int(np.sum(mask)),
    }


def apply_eval_protocol(kwargs, eval_protocol):
    """Apply project UTC episode grid (train/test/plots must match)."""
    if eval_protocol == EVAL_PROTOCOL_LOCAL_DAY:
        raise ValueError(
            'eval_protocol=local_day is disabled. This project uses tz=UTC only '
            '(%s-%s UTC, %d steps).' % (
                PV_EPISODE_START_TIME, _grid_end_clock_label(), PV_EPISODE_MAX_STEPS))
    if eval_protocol not in (
            None, '', EVAL_PROTOCOL_INHERIT, EVAL_PROTOCOL_LEGACY_UTC, EVAL_PROTOCOL_UTC):
        raise ValueError(
            'Unknown eval_protocol=%r. Use inherit, utc, or legacy_utc.' % (eval_protocol,))
    merged = apply_pv_utc_schedule(kwargs)
    label = 'UTC episode grid %s-%s (%d steps)' % (
        PV_EPISODE_START_TIME,
        _grid_end_clock_label(),
        PV_EPISODE_MAX_STEPS,
    )
    return merged, label


def _episode_start_hour(start_time=None):
    """Fractional UTC hour from 'HH:MM' (no timezone conversion)."""
    st = start_time or PV_EPISODE_START_TIME
    hour, minute = map(int, str(st).split(':'))
    return hour + minute / 60.0


def _grid_step_hours(freq=None):
    import pandas as pd
    return pd.Timedelta(freq or PV_EPISODE_FREQ).total_seconds() / 3600.0


def _grid_end_clock_label(start_time=None, periods=None, freq=None):
    """Last timestamp wall clock for the configured UTC episode grid."""
    start_h = _episode_start_hour(start_time)
    n_periods = int(periods or PV_EPISODE_PERIODS)
    end_hour = start_h + (n_periods - 1) * _grid_step_hours(freq)
    end_hour = end_hour % 24.0
    return '%02d:%02d' % (int(end_hour), int(round((end_hour % 1) * 60)))


def build_episode_times(date_str, env_kwargs=None):
    """pvlib-aligned episode DatetimeIndex (same construction as PVTrackingEnv)."""
    import pandas as pd
    kwargs = env_kwargs or {}
    tz = kwargs.get('tz', PV_TIMEZONE)
    return pd.date_range(
        start='%s %s' % (date_str, kwargs.get('start_time', PV_EPISODE_START_TIME)),
        periods=int(kwargs.get('periods', PV_EPISODE_PERIODS)),
        freq=kwargs.get('freq', PV_EPISODE_FREQ),
        tz=tz,
    )


def compute_episode_solar_bounds(
        date_str,
        env_kwargs=None,
        latitude=PV_DEFAULT_LATITUDE,
        longitude=PV_DEFAULT_LONGITUDE,
        altitude=PV_DEFAULT_ALTITUDE,
        min_productive_alt_deg=5.0):
    """Solar geometry on the configured episode grid (UTC, pvlib Location.tz)."""
    import pandas as pd
    from pvlib.location import Location

    kwargs = dict(env_kwargs or {})
    tz = kwargs.get('tz', PV_TIMEZONE)
    times = build_episode_times(date_str, kwargs)
    loc = Location(latitude, longitude, tz=tz, altitude=altitude)
    sp = loc.get_solarposition(times)
    cs = loc.get_clearsky(times)
    alt = (90.0 - sp['zenith']).values
    ghi = cs['ghi'].values
    clock = times.hour + times.minute / 60.0

    def _hour_where(mask):
        if not np.any(mask):
            return np.nan, np.nan
        hrs = clock[mask]
        return float(hrs[0]), float(hrs[-1])

    sun_up, sun_down = _hour_where(alt > 0.0)
    prod_lo, prod_hi = _hour_where(alt >= min_productive_alt_deg)
    peak_idx = int(np.nanargmax(ghi))
    return {
        'date': date_str,
        'tz': str(tz),
        'grid_start': times[0].isoformat(),
        'grid_end': times[-1].isoformat(),
        'n_timestamps': len(times),
        'sunrise_utc_hour': sun_up,
        'sunset_utc_hour': sun_down,
        'productive_sun_utc': (prod_lo, prod_hi),
        'ghi_peak_utc_hour': float(clock[peak_idx]),
        'ghi_peak_wm2': float(ghi[peak_idx]),
        'n_steps_night_alt_le_0': int(np.sum(alt <= 0.0)),
        'n_steps_productive_alt_ge_5': int(np.sum(alt >= min_productive_alt_deg)),
    }


def format_solar_bounds_line(bounds):
    prod = bounds.get('productive_sun_utc', (np.nan, np.nan))
    return (
        '  %s: grid %s–%s UTC | sun up %.2f–%.2f h | '
        'productive (alt≥5°) %.2f–%.2f h | GHI peak %.2f h | '
        'night steps (alt≤0): %d / %d timestamps' % (
            bounds['date'],
            bounds.get('grid_start', '')[:16],
            bounds.get('grid_end', '')[-14:],
            bounds.get('sunrise_utc_hour', np.nan),
            bounds.get('sunset_utc_hour', np.nan),
            prod[0], prod[1],
            bounds.get('ghi_peak_utc_hour', np.nan),
            bounds.get('n_steps_night_alt_le_0', 0),
            bounds.get('n_timestamps', PV_EPISODE_PERIODS),
        ))


def night_intervals_from_path(path):
    """UTC clock-hour spans where solar_altitude_deg <= 0 (for plot shading)."""
    infos = path.get('infos', [])
    if not infos:
        return []
    times = rollout_time_axis(path)
    alts = np.array(
        [info.get('solar_altitude_deg', np.nan) for info in infos], dtype=np.float64)
    intervals = []
    start = None
    for i, alt in enumerate(alts):
        if np.isfinite(alt) and alt <= 0.0:
            if start is None:
                start = times[i]
        elif start is not None:
            intervals.append((start, times[i]))
            start = None
    if start is not None:
        intervals.append((start, times[-1] if len(times) else start))
    return intervals


def infer_eval_protocol_from_checkpoint(policy_dim, env_kwargs):
    """Informational: all supported checkpoints use UTC grid after apply_pv_utc_schedule."""
    mode = observation_mode_for_dim(policy_dim)
    return EVAL_PROTOCOL_UTC, mode


def get_eval_environment(
        variant,
        override_path=None,
        test_start_date=None,
        test_end_date=None,
        fixed_eval_dates=None,
        eval_randomize_day=None,
        eval_weather_source=None,
        eval_randomize_initial_orientation=None,
        eval_protocol=EVAL_PROTOCOL_INHERIT):
    """Build frozen-policy evaluation env on the real PVTrackingEnv.

    RL protocol: Monte Carlo over episodes (annual days or NSRDB scenarios e~p(e))
    with independent eval seeds — not calendar-day hold-out. Within each episode
    weather follows one fixed exogenous trajectory (not per-step random draws).
    pvlib is deterministic given that trajectory. Optional irradiance_perturbation_std
    is bounded episode-level augmentation only (paper default 0).
    """
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
        kwargs['randomize_day'] = True

    # Eval always sees full annual support (never inherit training exclusions).
    kwargs.pop('excluded_dates', None)
    if fixed_eval_dates is None:
        kwargs.pop('fixed_eval_dates', None)
        # Post-train MC: do not inherit in-train fixed_eval_scenarios (5 validation ids).
        kwargs.pop('fixed_eval_scenarios', None)

    ws = kwargs.get('weather_source') or kwargs.get('weather_scenario_mode')
    if ws in ('nsrdb_multiyear',) and fixed_eval_dates is None:
        kwargs['randomize_scenario'] = True
        kwargs['randomize_day'] = False

    if eval_weather_source is not None:
        kwargs['weather_source'] = eval_weather_source

    if eval_randomize_initial_orientation is not None:
        kwargs['randomize_initial_orientation'] = bool(eval_randomize_initial_orientation)
    elif kwargs.get('randomize_day', True):
        kwargs.setdefault('randomize_initial_orientation', True)

    kwargs, protocol_label = apply_eval_protocol(kwargs, eval_protocol)
    eval_env_params['kwargs'] = kwargs
    if protocol_label:
        eval_env_params['eval_protocol_label'] = protocol_label
    eval_env_params['eval_protocol'] = eval_protocol
    return get_environment_from_params(eval_env_params), eval_env_params


def describe_eval_config(eval_env_params):
    """Human-readable summary of evaluation environment settings."""
    kwargs = eval_env_params.get('kwargs', {})
    lines = [
        'eval_protocol: %s' % eval_env_params.get(
            'eval_protocol', EVAL_PROTOCOL_INHERIT),
        'episode_preset: %s' % eval_env_params.get(
            'eval_protocol_label', '(from variant / env defaults)'),
        'tz (env clock): %s' % kwargs.get('tz', '(env default UTC)'),
        'start_time: %s' % kwargs.get('start_time', PV_EPISODE_START_TIME),
        'periods: %s (%d env steps)' % (
            kwargs.get('periods', PV_EPISODE_PERIODS),
            int(kwargs.get('periods', PV_EPISODE_PERIODS)) - 1),
        'freq: %s' % kwargs.get('freq', PV_EPISODE_FREQ),
        'time_standard: %s (train/test/plots)' % PV_TIMEZONE,
        'start_date: %s' % kwargs.get('start_date', '(from env default)'),
        'end_date: %s' % kwargs.get('end_date', '(from env default)'),
        'randomize_day: %s' % kwargs.get('randomize_day', True),
        'randomize_initial_orientation: %s' % kwargs.get(
            'randomize_initial_orientation', False),
        'weather_source: %s' % kwargs.get('weather_source', '(from env default)'),
        'weather_scenario_mode: %s' % kwargs.get(
            'weather_scenario_mode', '(from weather_source)'),
        'randomize_scenario: %s' % kwargs.get('randomize_scenario', False),
        'fixed_eval_scenarios: %s' % kwargs.get('fixed_eval_scenarios', None),
        'movement_penalty: %s' % kwargs.get('movement_penalty', '(from env default)'),
        'observation_mode: %s' % kwargs.get('observation_mode', 'legacy (default)'),
        'fixed_eval_dates: %s' % kwargs.get('fixed_eval_dates', None),
        'control_grid: %s UTC, %d periods, %s (%d actions)' % (
            kwargs.get('start_time', PV_EPISODE_START_TIME),
            int(kwargs.get('periods', PV_EPISODE_PERIODS)),
            kwargs.get('freq', PV_EPISODE_FREQ),
            PV_EPISODE_MAX_STEPS),
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


def _greedy_poa_orientation(underlying, solar_zenith, solar_azimuth, weather=None):
    """Myopic POA-maximizing tilt/azimuth (oracle upper bound, not deployable)."""
    if weather is None:
        weather = underlying._current_weather()
    best_power = -1.0
    best_tilt, best_az = float(solar_zenith), float(solar_azimuth)
    az_center = float(solar_azimuth)
    for tilt in np.linspace(0.0, 90.0, 19):
        for az_offset in np.linspace(-90.0, 90.0, 19):
            az = float(np.mod(az_center + az_offset, 360.0))
            power = underlying._power_from_orientation(
                solar_zenith, solar_azimuth, tilt, az)
            if power > best_power:
                best_power = power
                best_tilt, best_az = float(tilt), az
    return best_tilt, best_az


def make_baseline_rollout(env, baseline_type, path_length, seed=None):
    """Run a baseline policy with correct PV observation decoding.

    Baselines only set target tilt/azimuth each step; power and energy_kwh in
    info come from env.step → PVTrackingEnv._power_from_orientation (mbpo.env.pvlib_physics),
    identical to the learned policy path.
    """
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

        if baseline_type == 'fixed_no_motion':
            # True fixed mount: zero incremental action → Δtilt=Δaz=0 → movement_cost=0.
            action = np.zeros(2, dtype=np.float32)
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
            continue
        if baseline_type in ('fixed', 'fixed_tilt_south'):
            target_tilt = 30.0
            target_azimuth = 180.0
        elif baseline_type == 'single_axis':
            target_tilt = 30.0
            target_azimuth = solar_azimuth
        elif baseline_type in ('sun_seeking', 'sun_tracking'):
            target_tilt = solar_zenith
            target_azimuth = solar_azimuth
        elif baseline_type in ('poa_greedy_oracle', 'greedy_poa_oracle'):
            target_tilt, target_azimuth = _greedy_poa_orientation(
                underlying, solar_zenith, solar_azimuth)
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
    step_h = infos[0].get('interval_hours')
    if step_h is None and len(infos) >= 2:
        t0 = infos[0].get('time')
        t1 = infos[1].get('time')
        if t0 is not None and t1 is not None:
            step_h = float(t1) - float(t0)
    if step_h is None:
        step_h = _grid_step_hours()
    return float(np.sum(power) * float(step_h) / 1000.0)


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
            'clock_hour_utc': float(
                info.get('clock_hour_utc', info.get('clock_hour', info.get('time', np.nan)))),
        }

    altitudes = np.array(
        [info.get('solar_altitude_deg', np.nan) for info in infos], dtype=np.float64)

    by_window = {}
    for name, (lo, hi) in TIME_WINDOWS.items():
        mask = (times >= lo) & (times < hi)
        by_window[name] = _aggregate_window_stats(
            rewards, power, energy, movement, mask)

    by_solar_window = {}
    for name, (lo, hi) in SOLAR_ALTITUDE_WINDOWS.items():
        mask = _mask_solar_altitude(altitudes, lo, hi)
        by_solar_window[name] = _aggregate_window_stats(
            rewards, power, energy, movement, mask)

    info0 = infos[0] if infos else {}
    meta = get_rollout_metadata(path)
    meta['env_timezone'] = info0.get('timezone', PV_TIMEZONE)
    peak_power_time = _snap(i_power)['time_hour']
    peak_reward_time = _snap(i_reward)['time_hour']
    meta.update({
        'mean_power_w': float(np.nanmean(power)),
        'peak_power_w': float(np.nanmax(power)),
        'peak_power_time_hour': peak_power_time,
        'peak_power_time_utc_hour': _snap(i_power)['clock_hour_utc'],
        'peak_reward_time_hour': peak_reward_time,
        'peak_reward_time_utc_hour': _snap(i_reward)['clock_hour_utc'],
        'peak_reward_step': i_reward,
        'peak_power_step': i_power,
        'at_peak_reward': _snap(i_reward),
        'at_peak_power': _snap(i_power),
        'reward_by_window': by_window,
        'reward_by_solar_window': by_solar_window,
        'peak_reward_lag_hours': (
            float(peak_reward_time - peak_power_time)
            if np.isfinite(peak_reward_time) and np.isfinite(peak_power_time)
            else np.nan),
    })
    return meta


def _aggregate_named_windows(paths, window_names, window_key):
    agg = {name: defaultdict(list) for name in window_names}
    for path in paths:
        analysis = analyze_rollout_path(path)
        for name, stats in analysis.get(window_key, {}).items():
            for key, val in stats.items():
                if key != 'n_steps' and np.isfinite(val):
                    agg[name][key].append(val)
    return {
        name: {key: float(np.mean(vals)) if vals else np.nan for key, vals in buckets.items()}
        for name, buckets in agg.items()
    }


def aggregate_time_windows(paths):
    """Average per clock-hour window stats across rollouts (env timezone)."""
    return _aggregate_named_windows(paths, TIME_WINDOWS, 'reward_by_window')


def aggregate_solar_altitude_windows(paths):
    """Average per solar-altitude window stats across rollouts."""
    return _aggregate_named_windows(paths, SOLAR_ALTITUDE_WINDOWS, 'reward_by_solar_window')


def compare_method_table(paths_by_name):
    """Build comparison rows for policy vs baselines (mean over MC rollouts)."""
    rows = []
    for method, paths in paths_by_name.items():
        analyses = [analyze_rollout_path(p) for p in paths]
        rewards = [a['total_reward'] for a in analyses]
        energies = [a['total_energy_kwh'] for a in analyses]
        rows.append({
            'method': method,
            'n_rollouts': len(paths),
            'total_reward_mean': float(np.mean(rewards)) if rewards else np.nan,
            'total_reward_std': float(np.std(rewards, ddof=1)) if len(rewards) > 1 else 0.0,
            'total_energy_kwh_mean': float(np.mean(energies)) if energies else np.nan,
            'total_energy_kwh_std': float(np.std(energies, ddof=1)) if len(energies) > 1 else 0.0,
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


def _rollout_net_energy(path):
    return float(np.sum(path.get('rewards', [])))


def _rollout_seed(path, fallback_index=0):
    infos = path.get('infos', []) or []
    if infos:
        seed = infos[0].get('rollout_seed')
        if seed not in (None, ''):
            try:
                return int(seed)
            except (TypeError, ValueError):
                pass
    return int(fallback_index)


def align_paired_rollouts(paths_by_name, reference='learned_policy'):
    """Align methods by rollout index (matched-seed MC). Returns list of dicts per pair."""
    ref_paths = paths_by_name.get(reference, [])
    n = len(ref_paths)
    pairs = []
    for i in range(n):
        row = {'index': i, 'seed': _rollout_seed(ref_paths[i], i)}
        row[reference] = ref_paths[i]
        meta = get_rollout_metadata(ref_paths[i])
        row['scenario_id'] = meta.get('scenario_id')
        row['date'] = meta.get('date')
        for method, paths in paths_by_name.items():
            if method == reference:
                continue
            if i < len(paths):
                row[method] = paths[i]
        pairs.append(row)
    return pairs


def summarize_paired_mc(paths_by_name, reference='learned_policy'):
    """MC estimators: per-method mean/std and paired deltas (same scenario per seed)."""
    pairs = align_paired_rollouts(paths_by_name, reference=reference)
    if not pairs:
        return {}

    methods = sorted(paths_by_name.keys())
    out = {'n_pairs': len(pairs), 'methods': {}, 'paired_deltas': {}}

    for method in methods:
        vals = [_rollout_net_energy(pairs[i][method])
                for i in range(len(pairs)) if method in pairs[i]]
        out['methods'][method] = summarize_values(vals)

    def _paired_delta(method_a, method_b):
        deltas = []
        scenario_mismatch = 0
        for row in pairs:
            if method_a not in row or method_b not in row:
                continue
            ma = get_rollout_metadata(row[method_a])
            mb = get_rollout_metadata(row[method_b])
            if (ma.get('scenario_id') and mb.get('scenario_id')
                    and ma['scenario_id'] != mb['scenario_id']):
                scenario_mismatch += 1
            deltas.append(_rollout_net_energy(row[method_a]) - _rollout_net_energy(row[method_b]))
        stats = summarize_values(deltas)
        stats['scenario_id_mismatches'] = scenario_mismatch
        return stats

    if 'sun_tracking' in methods and 'fixed_no_motion' in methods:
        out['paired_deltas']['sun_minus_fixed'] = _paired_delta(
            'sun_tracking', 'fixed_no_motion')
    if 'learned_policy' in methods and 'sun_tracking' in methods:
        out['paired_deltas']['learned_minus_sun'] = _paired_delta(
            'learned_policy', 'sun_tracking')
    if 'learned_policy' in methods and 'fixed_no_motion' in methods:
        out['paired_deltas']['learned_minus_fixed'] = _paired_delta(
            'learned_policy', 'fixed_no_motion')
    return out


def verify_rollout_pvlib_power(path, area=1.0, efficiency=0.18, atol=0.5, max_steps=10):
    """Spot-check env.step power matches pvlib POA × area × efficiency."""
    from mbpo.env.pvlib_physics import compute_panel_power_w
    infos = path.get('infos', [])
    if not infos:
        return True, 0.0, []
    errors = []
    for info in infos[:max_steps]:
        dni = float(info.get('dni_wm2', info.get('dni', 0.0)))
        dhi = float(info.get('dhi_wm2', info.get('dhi', 0.0)))
        ghi = float(info.get('ghi_wm2', info.get('ghi', 0.0)))
        tilt = float(info.get('tilt', 0.0))
        az = float(info.get('azimuth', 0.0))
        zen = float(info.get('solar_zenith_deg', 0.0))
        saz = float(info.get('solar_azimuth_deg', 0.0))
        expected = compute_panel_power_w(
            tilt, az, zen, saz, dni, ghi, dhi,
            area=area, efficiency=efficiency)
        reported = float(info.get('power', 0.0))
        if expected < 1.0 and abs(reported) < 1.0:
            continue
        errors.append(abs(reported - expected))
    if not errors:
        return True, 0.0, []
    max_err = float(max(errors))
    return max_err <= atol, max_err, errors


def write_paired_mc_comparison_report(outdir, paths_by_name, eval_mode='mc', error='std'):
    """Text report: mean±std where MC applies; paired deltas on matched scenarios."""
    path = os.path.join(outdir, 'PAIRED_MC_COMPARISON.txt')
    summary = summarize_paired_mc(paths_by_name)
    pairs = align_paired_rollouts(paths_by_name)

    with open(path, 'w', encoding='utf-8') as f:
        f.write('Paired Monte Carlo comparison (pvlib env, T=%d)\n' % PV_EPISODE_MAX_STEPS)
        f.write('=' * 48 + '\n\n')
        if eval_mode == 'nsrdb':
            f.write('Distribution: e = (year, month, day) ~ Uniform(manifest)\n')
            f.write('Each rollout index i uses the same seed → same scenario_id for all methods.\n')
        else:
            f.write('Distribution: calendar day ~ training support (TMY or annual MC)\n')
            f.write('Each rollout index i uses seed_i = eval_seed_base + i (matched across methods).\n')
        f.write('\nEstimators (finite N rollouts, ddof=1):\n')
        f.write('  E[X]   = (1/N) sum_i X_i\n')
        f.write('  σ_X    = sqrt(1/(N-1) sum_i (X_i - E[X])^2)\n')
        f.write('  Δ_i    = X_i^A - X_i^B  on the SAME scenario (paired)\n')
        f.write('  E[Δ], σ_Δ describe A vs B controlling weather/orientation reset.\n')
        f.write('  Error bars in plots use %s (σ for std, SEM = σ/sqrt(N) for mean of mean).\n\n' % error)

        f.write('Per-method net energy (kWh = sum_t reward_t):\n')
        for method, stats in summary.get('methods', {}).items():
            f.write('  %-18s  E=%.4f  σ=%.4f  n=%d  [min=%.4f max=%.4f]\n' % (
                method, stats['mean'], stats['std'], stats['count'],
                stats['min'], stats['max']))

        f.write('\nPaired deltas (same scenario per row):\n')
        for label, stats in summary.get('paired_deltas', {}).items():
            f.write('  %-22s  E[Δ]=%+.4f kWh  σ_Δ=%.4f  n=%d' % (
                label, stats['mean'], stats['std'], stats['count']))
            if stats.get('scenario_id_mismatches', 0):
                f.write('  WARN mismatches=%d' % stats['scenario_id_mismatches'])
            f.write('\n')

        f.write('\nScenario alignment (first %d rollouts):\n' % min(8, len(pairs)))
        for row in pairs[:8]:
            parts = ['idx=%d seed=%s' % (row['index'], row.get('seed'))]
            if row.get('scenario_id'):
                parts.append('scenario=%s' % row['scenario_id'])
            for method in sorted(paths_by_name.keys()):
                if method in row:
                    meta = get_rollout_metadata(row[method])
                    parts.append('%s=%.4f' % (method, meta['total_energy_kwh']))
            f.write('  %s\n' % ' | '.join(parts))

        f.write('\npvlib consistency (spot-check first 10 steps per method):\n')
        for method, paths in sorted(paths_by_name.items()):
            if not paths:
                continue
            ok, max_err, _ = verify_rollout_pvlib_power(paths[0])
            f.write('  %-18s  %s  max_abs_err=%.4g W\n' % (
                method, 'PASS' if ok else 'CHECK', max_err))

        f.write('\nNote: deterministic physics checks (OAT sensitivity, single-day tilt sweep)\n')
        f.write('belong in verify_pv_state_space.py — not repeated here.\n')
    return path


def get_rollout_metadata(path):
    infos = path.get('infos', [])
    info0 = infos[0] if infos else {}
    day_of_year = info0.get('day_of_year')
    date = info0.get('date', 'unknown')
    season = info0.get('season') or (
        season_from_day_of_year(day_of_year)
        if day_of_year is not None else 'unknown')
    season_calendar = (
        season_from_calendar_date(date)
        if date not in (None, '', 'unknown') else 'unknown')
    return {
        'date': date,
        'day_of_year': day_of_year,
        'season': season,
        'season_calendar': season_calendar,
        'weather_condition': info0.get('weather_condition', 'unknown'),
        'weather_source': info0.get('weather_source', 'unknown'),
        'scenario_id': info0.get('scenario_id'),
        'scenario_year': info0.get('scenario_year'),
        'seed': info0.get('rollout_seed'),
        'episode_length': len(path.get('rewards', [])),
        'total_reward': float(np.sum(path.get('rewards', []))),
        'total_energy_kwh': compute_total_energy_kwh(path),
        'total_movement_cost': float(np.sum(
            [info.get('movement_cost', 0.0) for info in infos])) if infos else np.nan,
    }


def summarize_values(values):
    """Sample mean and sample std (ddof=1) over finite MC rollouts."""
    values = np.asarray(values, dtype=np.float64)
    n = int(len(values))
    std = float(np.std(values, ddof=1)) if n > 1 else 0.0
    return OrderedDict([
        ('count', n),
        ('mean', float(np.mean(values)) if n else np.nan),
        ('std', std),
        ('min', float(np.min(values)) if n else np.nan),
        ('max', float(np.max(values)) if n else np.nan),
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


def write_eval_statistics_readme(outdir, num_rollouts, eval_seed_base):
    """Document post-train estimators (not SAC entropy / Q bounds)."""
    path = os.path.join(outdir, 'EVAL_STATISTICS.txt')
    with open(path, 'w', encoding='utf-8') as f:
        f.write('Post-training evaluation statistics\n')
        f.write('=' * 40 + '\n\n')
        f.write('Checkpoint / policy\n')
        f.write('  Loaded from trial params.json + checkpoint policy_weights.\n')
        f.write('  Deploy: tanh(mu) when deterministic=True (matches training eval).\n')
        f.write('  Integrity: evaluate_agent validates policy_input_dim vs env obs dim.\n\n')
        f.write('Per-step reward (all methods, same env.step):\n')
        f.write('  r_t = energy_kwh_t - movement_penalty * (|a0| + |a1|)\n')
        f.write('  Episode return R = sum_t r_t  (discount=1 in PV env).\n\n')
        f.write('Monte Carlo over rollouts (this eval):\n')
        f.write('  N = %d independent episodes, seed_i = %d + i.\n' % (
            num_rollouts, eval_seed_base))
        f.write('  E[R]  = (1/N) sum_i R_i   (reported as mean)\n')
        f.write('  sigma = sqrt(1/(N-1) sum_i (R_i - E[R])^2)  (ddof=1, not SAC std)\n\n')
        f.write('In-training progress.csv (different estimator):\n')
        f.write('  evaluation/return-average = mean of n_eval episodes that epoch.\n')
        f.write('  evaluation/return-std = std of those n_eval episodes (Ray Tune log).\n\n')
        f.write('Season labels in summaries:\n')
        f.write('  season_calendar: month buckets (Jun-Aug = summer).\n')
        f.write('  season (env): equinox buckets in PVTrackingEnv (Jun 1 can be spring).\n\n')
        f.write('Baselines:\n')
        f.write('  sun_tracking: slew toward solar zenith/azimuth each step.\n')
        f.write('  fixed_no_motion: action=0 (panel frozen at reset pose; movement_cost=0).\n')
        f.write('  fixed_tilt_south: optional slew to 30/180 (not default in Stage 3).\n\n')
        f.write('Episode clock (UTC): 13:30 start, 5min steps, %d transitions.\n' % (
            PV_EPISODE_MAX_STEPS))
        f.write('  Same grid as training; see eval_config episode_preset in summary.\n')
    return path


def rollout_xlabel(path):
    """Matplotlib x-axis label: UTC post-step clock hour (project standard)."""
    if path.get('infos'):
        return 'Clock hour UTC (post-step)'
    return 'Step index'


def rollout_time_axis(path, relative=False):
    """Plot x-axis in UTC clock hours from env info (never local conversion)."""
    infos = path.get('infos', [])
    if not infos:
        return np.arange(len(path.get('rewards', [])), dtype=np.float64)

    times = np.array([
        info.get(
            'clock_hour_utc',
            info.get('clock_hour', info.get('clock_hour_env_tz', info.get('time', np.nan))),
        )
        for info in infos
    ], dtype=np.float64)

    if not np.isfinite(times).all():
        return np.arange(len(path['rewards']), dtype=np.float64)
    if relative:
        return times - times[0]
    return times


def write_eval_scenario_confirmation(outdir, eval_env_params, paths_by_name, max_path_length):
    """Document matched evaluation settings across learned policy and baselines."""
    path = os.path.join(outdir, 'eval_scenario_confirmation.txt')
    kwargs = eval_env_params.get('kwargs', {})

    def _path_seed(path, fallback_seed):
        infos = path.get('infos', []) or []
        if infos:
            seed = infos[0].get('rollout_seed')
            if seed not in (None, ''):
                try:
                    return int(seed)
                except (TypeError, ValueError):
                    pass
        return int(fallback_seed)

    with open(path, 'w', encoding='utf-8') as f:
        f.write('PV Tracking — matched evaluation scenario\n')
        f.write('=' * 40 + '\n\n')
        for line in describe_eval_config(eval_env_params):
            f.write('%s\n' % line)
        f.write('max_path_length (eval): %d\n' % max_path_length)
        f.write('\nFairness checklist (same for all methods on each rollout index):\n')
        f.write('  [x] identical env kwargs (tz, start_time, periods, freq, weather)\n')
        f.write('  [x] randomize_initial_orientation=%s\n' % kwargs.get(
            'randomize_initial_orientation', False))
        f.write('  [x] per-rollout seed recorded in rollout metadata / CSV\n')
        f.write('  [x] same reward = energy_kwh - movement_cost\n')
        f.write('  [x] power/energy from pvlib via env.step (all methods)\n')
        f.write('  [x] baselines do not call the neural policy\n')
        f.write('\nPer-method rollout dates (seed order):\n')
        for method, paths in sorted(paths_by_name.items()):
            f.write('  %s:\n' % method)
            for idx, p in enumerate(paths, 1):
                meta = get_rollout_metadata(p)
                seed = _path_seed(p, idx - 1)
                sid = meta.get('scenario_id') or ''
                sid_part = (' scenario=%s' % sid) if sid else ''
                f.write('    rollout_%d seed=%d date=%s%s weather=%s steps=%d energy=%.4f kWh\n' % (
                    idx, seed, meta.get('date'), sid_part, meta.get('weather_condition'),
                    meta.get('episode_length'), meta.get('total_energy_kwh')))
        if len(paths_by_name) >= 2:
            paired = summarize_paired_mc(paths_by_name)
            f.write('\nPaired MC summary (net energy kWh, same seed per index):\n')
            for method, stats in paired.get('methods', {}).items():
                f.write('  %-18s  mean=%.4f  std=%.4f  n=%d\n' % (
                    method, stats['mean'], stats['std'], stats['count']))
            for label, stats in paired.get('paired_deltas', {}).items():
                f.write('  %-22s  mean_delta=%+.4f  std=%.4f\n' % (
                    label, stats['mean'], stats['std']))
        f.write('\nTime standard: tz=%s, grid %s–%s UTC, %d steps per episode.\n' % (
            kwargs.get('tz', PV_TIMEZONE),
            kwargs.get('start_time', PV_EPISODE_START_TIME),
            _grid_end_clock_label(),
            max_path_length,
        ))
        f.write('Plots and CSV use clock_hour_utc only (no local conversion).\n')
        f.write(
            '\nConfigured grid vs solar day (pvlib, lat=%.1f lon=%.1f):\n' % (
                PV_DEFAULT_LATITUDE, PV_DEFAULT_LONGITUDE))
        f.write(
            '  Daylight UTC grid %s–%s: tuned for productive sun at 35N/106W. '
            'Expect 0–2 pre-sunrise steps (power≈0, solar_alt<0) at the grid start in winter.\n' % (
                kwargs.get('start_time', PV_EPISODE_START_TIME),
                _grid_end_clock_label(),
            ))
        dates = sorted({
            get_rollout_metadata(p).get('date')
            for paths in paths_by_name.values()
            for p in paths
            if get_rollout_metadata(p).get('date') not in (None, 'unknown')
        })
        for date in dates:
            f.write(format_solar_bounds_line(
                compute_episode_solar_bounds(date, env_kwargs=kwargs)) + '\n')
    return path


def write_reward_time_report(outdir, paths, paths_by_name=None):
    """Text report: reward vs power timing and time-window breakdown."""
    report_path = os.path.join(outdir, 'reward_time_analysis.txt')
    window_agg = aggregate_time_windows(paths)
    solar_agg = aggregate_solar_altitude_windows(paths)

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('PV Tracking Reward-Time Analysis\n')
        f.write('=' * 36 + '\n\n')
        f.write('Time standard: %s (train/test/plots)\n' % PV_TIMEZONE)
        f.write(
            'info[time] / clock_hour_utc = post-step wall-clock hour in UTC. '
            'Episode grid %s–%s UTC, %d steps.\n' % (
                PV_EPISODE_START_TIME, _grid_end_clock_label(), PV_EPISODE_MAX_STEPS))
        f.write(
            'Clock windows (morning/midday/...) are UTC bins on the daylight episode grid. '
            'Use solar-altitude windows for sun-up / peak-sun physics.\n\n')
        f.write(
            'Reward per step = energy_kwh - movement_cost. '
            'Peak step reward often occurs when movement is low, not when power is highest.\n\n')

        f.write('Learned policy — per rollout peak times (UTC clock hour):\n')
        for idx, path in enumerate(paths, 1):
            a = analyze_rollout_path(path)
            date = a.get('date', '?')
            t_pwr = a.get('peak_power_time_hour', np.nan)
            t_rew = a.get('peak_reward_time_hour', np.nan)
            f.write(
                '  rollout_%d (%s): peak_power=%.1f W @ %.2f h UTC | '
                'peak_reward=%.5f @ %.2f h UTC | lag=%.2f h\n' % (
                    idx, date,
                    a.get('peak_power_w', np.nan), t_pwr,
                    a['at_peak_reward']['reward'], t_rew,
                    a.get('peak_reward_lag_hours', np.nan)))
            f.write(
                '    at peak reward: tilt=%.1f az=%.1f solar_alt=%.1f solar_az=%.1f move=%.5f\n' % (
                    a['at_peak_reward']['tilt_deg'], a['at_peak_reward']['azimuth_deg'],
                    a['at_peak_reward']['solar_altitude_deg'],
                    a['at_peak_reward']['solar_azimuth_deg'],
                    a['at_peak_reward']['movement_cost']))

        f.write('\nLearned policy — mean by UTC clock window (avg over rollouts):\n')
        f.write('  window      mean_reward  mean_power_W  sum_energy   sum_movement\n')
        for name in TIME_WINDOWS:
            w = window_agg.get(name, {})
            f.write('  %-10s  %11.5f  %11.1f  %10.5f  %12.5f\n' % (
                name,
                w.get('mean_reward', np.nan),
                w.get('mean_power_w', np.nan),
                w.get('sum_energy_kwh', np.nan),
                w.get('sum_movement_cost', np.nan)))

        f.write('\nLearned policy — mean by solar altitude (physics, tz-independent):\n')
        f.write('  window            mean_reward  mean_power_W  sum_energy   sum_movement\n')
        for name in SOLAR_ALTITUDE_WINDOWS:
            w = solar_agg.get(name, {})
            f.write('  %-18s  %11.5f  %11.1f  %10.5f  %12.5f\n' % (
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

        f.write('\nPeak-power notes (UTC daylight grid at 35N/106W):\n')
        f.write(
            '  A) GHI peak is often near 17-20h UTC in December (solar noon at this longitude).\n')
        f.write(
            '  B) Gray bands on plots = solar altitude ≤ 0°; expect 0-2 pre-sunrise steps at grid start in winter.\n')
        f.write(
            '  C) All plot/CSV clocks are UTC only (no civil-time conversion).\n')
        f.write(
            '  D) pvlib uses the same DatetimeIndex tz as the env for solar position and power.\n')
        f.write(
            '  E) Compare total_energy_kwh vs baselines on solar-altitude windows.\n')
        f.write('\nInterpretation:\n')
        f.write(
            '  - Plot x-axis is UTC clock hour on the episode grid (%s–%s).\n' % (
                PV_EPISODE_START_TIME, _grid_end_clock_label()))
        f.write(
            '  - Movement with power≈0 at the first steps is pre-sunrise grid margin, not a time bug.\n')
        f.write(
            '  - Compare total_energy_kwh and peak_power_time across methods; '
            'reward alone is not a proxy for tracking quality.\n')
        f.write(
            '  - Run: python scripts/solar_time_sanity.py --date 2020-12-21\n')

    return report_path


def save_rollout_csv(outdir, paths, prefix='rollout'):
    os.makedirs(outdir, exist_ok=True)
    for idx, path in enumerate(paths, start=1):
        observations = np.asarray(path['observations'])
        actions = np.asarray(path['actions'])
        rewards = np.asarray(path['rewards'])
        terminals = np.asarray(path.get('terminals', [False] * len(rewards)))
        infos = path.get('infos', [])

        env_tz = infos[0].get('timezone', '') if infos else ''
        header = [
            'step', 'rollout_seed', 'clock_hour_utc', 'timezone', 'timestamp_utc_iso',
            'reward', 'terminal',
            'power_w', 'energy_kwh', 'movement_cost',
            'reward_energy', 'reward_movement',
            'tilt_deg', 'azimuth_deg',
            'solar_zenith_deg', 'solar_azimuth_deg', 'solar_altitude_deg',
            'date', 'season', 'weather_condition', 'weather_source',
            'action_tilt', 'action_azimuth',
        ]
        obs_labels = pv_obs_labels_for_vector(
            observations[0] if observations.ndim == 2 else observations)
        header += list(obs_labels)

        rows = []
        for t in range(len(rewards)):
            info = infos[t] if t < len(infos) else {}
            row = [
                t,
                info.get('rollout_seed', ''),
                info.get(
                    'clock_hour_utc',
                    info.get('clock_hour_env_tz', info.get('clock_hour', info.get('time', ''))),
                ),
                info.get('timezone', env_tz or PV_TIMEZONE),
                info.get('timestamp_utc_iso', ''),
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
