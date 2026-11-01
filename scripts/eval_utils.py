#!/usr/bin/env python3
"""Utilities for evaluating PV tracking policies."""

from __future__ import print_function

import argparse
import csv
import glob
import importlib.util
import json
import os
import sys
from collections import OrderedDict

import numpy as np
import pandas as pd

from mbpo.env.pvlib_physics import (
    compute_panel_power_w,
    compute_poa_global,
    panel_tilt_deg_from_norm,
    solar_zenith_norm_from_deg,
)

from mbpo.env.pv_tracking import (
    DEFAULT_EPISODE_STEPS,
    DEFAULT_START_HOUR,
    DEFAULT_STEP_HOURS,
    DEFAULT_START_TIME,
    DEFAULT_PERIODS,
    DEFAULT_FREQ,
    episode_clock_hour,
    PV_TIMEZONE,
    DEFAULT_END_HOUR,
)

from mbpo.env.nsrdb_weather import (
    get_eval_environment,
    load_scenario_manifest,
    scenario_id,
)

# Re-export constants with expected names for backward compatibility
PV_EPISODE_START_TIME = DEFAULT_START_TIME
PV_EPISODE_PERIODS = DEFAULT_PERIODS
PV_EPISODE_FREQ = DEFAULT_FREQ
PV_EPISODE_MAX_STEPS = DEFAULT_EPISODE_STEPS


def apply_pv_utc_schedule(config):
    """Force evaluation to use the shared fixed UTC daylight envelope."""
    # This ensures eval always uses UTC regardless of training config
    config = dict(config)
    config['tz'] = PV_TIMEZONE
    config['start_time'] = DEFAULT_START_TIME
    config['periods'] = DEFAULT_PERIODS
    config['freq'] = DEFAULT_FREQ
    return config


def apply_eval_protocol(config, protocol='inherit'):
    """Apply eval protocol: 'inherit' or 'local_day'."""
    if protocol == 'local_day':
        # Legacy protocol: use local civil-day grid (NOT for paper eval)
        raise ValueError('local_day protocol is deprecated for paper eval')
    return apply_pv_utc_schedule(config)


def build_episode_times(start_time=DEFAULT_START_TIME, periods=DEFAULT_PERIODS, freq=DEFAULT_FREQ, tz=PV_TIMEZONE):
    """Build episode timestamps for plotting/CSV."""
    # Handle legacy call: build_episode_times(date_str, config_dict)
    if isinstance(periods, dict):
        config = periods
        start_time = config.get('start_time', DEFAULT_START_TIME)
        periods = config.get('periods', DEFAULT_PERIODS)
        freq = config.get('freq', DEFAULT_FREQ)
        tz = config.get('tz', PV_TIMEZONE)
    start = pd.Timestamp(start_time).tz_localize(tz)
    times = pd.date_range(start=start, periods=periods, freq=freq, tz=tz)
    return times


def rollout_time_axis(obs, tz=PV_TIMEZONE):
    """Return time axis for rollout plotting (post-step UTC hour)."""
    H = DEFAULT_EPISODE_STEPS
    times = build_episode_times(tz=tz)
    # Return post-step times (skip first timestamp which is pre-step)
    start_hour = DEFAULT_START_HOUR
    return [start_hour + (i + 1) * DEFAULT_STEP_HOURS for i in range(H)]


def rollout_xlabel(path=None):
    """X-axis label for rollout plots."""
    return 'Clock hour UTC (post-step)'


def _grid_end_clock_label():
    return '%02d:%02d' % (
        int(DEFAULT_END_HOUR) % 24,
        int((DEFAULT_END_HOUR - int(DEFAULT_END_HOUR)) * 60)
    )


def _grid_step_hours(freq):
    return pd.Timedelta(freq).total_seconds() / 3600.0


from softlearning.misc.utils import initialize_tf_variables

from mbpo.static.pv_tracking import (
    COMPACT_OBS_DIM,
    LEGACY_OBS_DIM,
    PHYSICAL_OBS_DIM,
    observation_mode_from_obs,
    cyclic_slices_for_obs,
    StaticFns,
    _angular_error_deg,
)

def normalize_angle_diff(target, current):
    return (target - current + 180.0) % 360.0 - 180.0


def decode_pv_observation(obs):
    """Decode normalized PVTracking observation to physical units.
    
    Supports three observation modes:
    - compact (6D): [dni_norm, dhi_norm, power_norm, panel_tilt_norm, panel_azimuth_norm, tau]
    - physical (12D): physical obs with tau (12 dims)
    - legacy (15D): legacy obs with tau (15 dims)
    """
    from mbpo.env.pvlib_physics import panel_tilt_deg_from_norm
    obs = np.asarray(obs, dtype=np.float64).reshape(-1)
    
    if obs.shape[0] == COMPACT_OBS_DIM:
        # Compact 6D: [DNI_norm, DHI_norm, power_norm, panel_tilt_norm, panel_azimuth_norm, tau]
        dni_norm = float(np.clip(obs[0], 0.0, 1.0))
        dhi_norm = float(np.clip(obs[1], 0.0, 1.0))
        power_norm = float(np.clip(obs[2], 0.0, 1.0))
        tilt_norm = float(np.clip(obs[3], 0.0, 1.0))
        azimuth_norm = float(np.clip(obs[4], -1.0, 1.0))
        tau = float(np.clip(obs[5], 0.0, 1.0))
        
        # Recover physical values
        dni = dni_norm * 2000.0
        dhi = dhi_norm * 2000.0
        power = power_norm * 2000.0  # POWER_NORM = 2000
        tilt_deg = panel_tilt_deg_from_norm(tilt_norm)
        azimuth_deg = (azimuth_norm + 1.0) * 180.0  # reverse: (gamma%360)/180 - 1 -> azimuth_norm
        # azimuth_deg in [0, 360]
        
        decoded = {
            'dni_wm2': dni,
            'dhi_wm2': dhi,
            'power_w': power,
            'panel_tilt_deg': tilt_deg,
            'panel_tilt_norm': tilt_norm,
            'panel_azimuth_deg': azimuth_deg % 360.0,
            'panel_azimuth_norm': azimuth_norm,
            'tau': tau,
        }
        return decoded
    
    # Physical obs with tau (12 dims) - tau is at index 11 (last)
    elif obs.shape[0] == PHYSICAL_OBS_DIM:
        solar_azimuth = np.rad2deg(
            np.arctan2(obs[1], obs[2])) % 360.0
        panel_azimuth = np.rad2deg(
            np.arctan2(obs[8], obs[9])) % 360.0
        zenith_norm = float(np.clip(obs[0], 0.0, 1.0))
        decoded = {
            'solar_zenith_deg': float(
                np.clip(zenith_norm * 180.0, 0.0, 180.0)),
            'solar_zenith_norm': zenith_norm,
            'solar_azimuth_deg': float(solar_azimuth),
            'panel_tilt_deg': panel_tilt_deg_from_norm(obs[7]),
            'panel_tilt_norm': float(np.clip(obs[7], 0.0, 1.0)),
            'panel_azimuth_deg': float(panel_azimuth),
            'dni_norm': float(obs[3]),
            'ghi_norm': float(obs[5]),
            'cos_aoi': float(obs[10]),
            'tau': float(obs[11]),
        }
        return decoded
    
    # Legacy obs with tau (15 dims)
    else:
        angle = np.mod(np.arctan2(obs[11], obs[12]), 2.0 * np.pi)
        decoded = {
            'time_of_day_hour': float(angle * 24.0 / (2.0 * np.pi)),
            'power_norm': float(obs[10]),
        }
        return decoded


def _greedy_poa_orientation(underlying, solar_zenith, solar_azimuth, weather=None):
    """Myopic POA-maximizing tilt/azimuth via pvlib get_total_irradiance (oracle)."""
    from mbpo.env.pvlib_physics import compute_panel_power_w
    if weather is None:
        weather = underlying._current_weather()
    best_power = -1.0
    best_tilt, best_az = float(solar_zenith), float(solar_azimuth)
    az_center = float(solar_azimuth)
    for tilt in np.linspace(0.0, 90.0, 19):
        for az_offset in np.linspace(-90.0, 90.0, 19):
            az = float(np.mod(az_center + az_offset, 360.0))
            power = compute_panel_power_w(
                tilt, az,
                solar_zenith, solar_azimuth,
                weather['dni'], weather['ghi'], weather['dhi'],
                area=underlying.area,
                efficiency=underlying.efficiency,
            )
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
        mode = observation_mode_from_obs(obs)
        
        if mode == 'compact':
            # For compact mode, we don't have solar position in observation
            # Get it from the underlying environment
            solar_zenith = underlying._current_weather().get('solar_zenith')
            if solar_zenith is None:
                # Compute from current time
                current_time = underlying.current_time
                solar_pos = underlying._solar_position(current_time)
                solar_zenith = solar_pos.zenith
                solar_azimuth = solar_pos.azimuth
            else:
                solar_azimuth = underlying._current_weather().get('solar_azimuth', 0)
            current_tilt = decoded['panel_tilt_deg']
            current_azimuth = decoded['panel_azimuth_deg']
        else:
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
        elif baseline_type == 'sun_tracking':
            # Myopic POA-maximizing oracle: at each step, pick action that maximizes POA irradiance.
            target_tilt, target_az = _greedy_poa_orientation(underlying, solar_zenith, solar_azimuth)
            delta_tilt = np.clip(target_tilt - current_tilt, -1.0, 1.0)
            delta_az = np.clip(target_az - current_azimuth, -1.0, 1.0)
            action = np.array([delta_tilt, delta_az], dtype=np.float32)
            next_obs, reward, terminal, info = env.step(action)
            observations.append(obs)
            actions.append(action)
            rewards.append(reward)
            terminals.append(terminal)
            next_observations.append(next_obs)
            infos.append(info)
            obs = next_obs
            done = terminal
        else:
            raise ValueError('Unknown baseline_type: {}'.format(baseline_type))

        step += 1
        if step >= path_length:
            break

    return {
        'observations': np.array(observations),
        'actions': np.array(actions),
        'rewards': np.array(rewards),
        'terminals': np.array(terminals),
        'next_observations': np.array(next_observations),
        'infos': infos,
    }


def _extract_underlying_env(env):
    """Unwrap gym wrappers to get the underlying PVTrackingEnv."""
    if hasattr(env, 'unwrapped'):
        return env.unwrapped
    while hasattr(env, 'env'):
        env = env.env
    return env