#!/usr/bin/env python3
"""Check PVTracking time labels, solar position, and irradiance alignment.

Project standard: tz='UTC', daylight episode grid 13:30-23:15 UTC, 39 steps.
info['clock_hour_utc'] is the post-step wall-clock hour on that index.
Optional Denver columns in output are for human comparison only — not used in training or plots.
"""

from __future__ import print_function

import argparse
import os
import sys

import numpy as np
import pandas as pd
from pvlib.location import Location

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _REPO)
sys.path.insert(0, _SCRIPT_DIR)

import softlearning.environments.adapters.gym_adapter  # noqa: F401

from eval_utils import TIME_WINDOWS


def _clock_hours(times):
    return times.hour + times.minute / 60.0


def print_fixed_hours(lat, lon, date, tz):
    loc = Location(lat, lon, tz=tz)
    print('\nFixed clock hours on {} (tz={})'.format(date, tz))
    print('  clock_h  solar_alt  zenith   ghi    dni')
    for h in [6, 9, 12, 15, 18, 19, 21]:
        ts = pd.DatetimeIndex([pd.Timestamp('{} {:02d}:00'.format(date, h), tz=tz)])
        sp = loc.get_solarposition(ts).iloc[0]
        cs = loc.get_clearsky(ts).iloc[0]
        print('  {:6.1f}  {:8.1f}  {:6.1f}  {:5.0f}  {:5.0f}'.format(
            h, 90.0 - sp.zenith, sp.zenith, cs.ghi, cs.dni))


def analyze_episode_grid(lat, lon, date, tz, start_time='13:30', periods=40, freq='15min'):
    loc = Location(lat, lon, tz=tz)
    times = pd.date_range(
        start='{} {}'.format(date, start_time),
        periods=periods,
        freq=freq,
        tz=tz,
    )
    sp = loc.get_solarposition(times)
    cs = loc.get_clearsky(times)
    alt = 90.0 - sp['zenith'].values
    ghi = cs['ghi'].values
    clk = _clock_hours(times)

    imax = int(np.argmax(ghi))
    print('\nEpisode grid {} -> {} ({} steps, tz={})'.format(
        times[0], times[-1], periods, tz))
    print('  GHI peak at clock {:.2f} h, solar_alt {:.1f} deg'.format(clk[imax], alt[imax]))
    if np.any(alt > 10):
        print('  Sun alt > 10 deg: clock {:.2f} h - {:.2f} h'.format(
            clk[alt > 10][0], clk[alt > 10][-1]))

    print('\n  eval_utils TIME_WINDOWS (clock in env tz):')
    print('  window      clock_range   mean_ghi  mean_alt   n')
    for name, (lo, hi) in TIME_WINDOWS.items():
        m = (clk >= lo) & (clk < hi)
        print('  {:10s}  {:5.1f}-{:5.1f}h   {:7.0f}  {:7.1f}  {:3d}'.format(
            name, lo, hi,
            ghi[m].mean() if m.any() else 0.0,
            alt[m].mean() if m.any() else 0.0,
            int(m.sum())))

    print('\n  Solar-altitude windows (physics, tz-independent labels):')
    solar_bins = (
        ('night_alt<5', alt < 5),
        ('low_5-20', (alt >= 5) & (alt < 20)),
        ('mid_20-30', (alt >= 20) & (alt < 30)),
        ('high_30+', alt >= 30),
    )
    for name, mask in solar_bins:
        print('  {:12s}  mean_ghi={:7.0f}  mean_alt={:6.1f}  n={}'.format(
            name,
            ghi[mask].mean() if mask.any() else 0.0,
            alt[mask].mean() if mask.any() else 0.0,
            int(mask.sum())))


def print_fixed_hours_with_denver(lat, lon, date, tz='UTC', local_tz='America/Denver'):
    """Validation table: UTC clock, site-local time, solar position, clearsky GHI."""
    loc = Location(lat, lon, tz=tz)
    print('\nValidation table {} (env tz={}, local={})'.format(date, tz, local_tz))
    print('  UTC_h  local_h  solar_alt  solar_az   ghi   dni')
    for h in [6, 9, 12, 15, 18, 19, 21]:
        ts = pd.DatetimeIndex([pd.Timestamp('{} {:02d}:00'.format(date, h), tz=tz)])
        sp = loc.get_solarposition(ts).iloc[0]
        cs = loc.get_clearsky(ts).iloc[0]
        local = ts[0].tz_convert(local_tz)
        local_h = local.hour + local.minute / 60.0
        print('  {:5.1f}  {:7.2f}  {:8.1f}  {:8.1f}  {:5.0f}  {:5.0f}'.format(
            h, local_h, 90.0 - sp.zenith, sp.azimuth, cs.ghi, cs.dni))


def log_step_alignment(lat, lon, tz, date, n_steps=5):
    """First steps: verify timestamp, irradiance, orientation, and power align."""
    import gym
    from eval_utils import decode_pv_observation, normalize_angle_diff

    print('\nStep alignment (first %d steps, date=%s tz=%s)' % (n_steps, date, tz))
    print('  step  UTC_clock  Denver_h  tilt  az   solar_alt  ghi   dni   dhi   power  reward')
    env = gym.make(
        'PVTracking-v0',
        latitude=lat,
        longitude=lon,
        tz=tz,
        start_date=date,
        end_date=date,
        randomize_day=False,
        randomize_initial_orientation=False,
        weather_source='clearsky',
        observation_mode='physical',
        fixed_eval_dates=[date],
    )
    obs = env.reset()
    for step in range(n_steps):
        decoded = decode_pv_observation(obs)
        action = np.zeros(2, dtype=np.float32)
        obs, reward, done, info = env.step(action)
        utc_h = float(info.get('clock_hour', info.get('time')))
        hour = int(utc_h) % 24
        minute = int(round((utc_h - int(utc_h)) * 60.0)) % 60
        ts = pd.Timestamp(str(info.get('date', date)), tz=tz) + pd.Timedelta(hours=hour, minutes=minute)
        local = ts.tz_convert('America/Denver')
        denver_h = local.hour + local.minute / 60.0
        print('  {:4d}  {:8.2f}  {:8.2f}  {:4.0f}  {:4.0f}  {:9.1f}  {:5.0f}  {:5.0f}  {:5.0f}  {:6.1f}  {:7.5f}'.format(
            step, utc_h, denver_h,
            info.get('tilt'), info.get('azimuth'),
            info.get('solar_altitude_deg'),
            info.get('ghi_wm2'), info.get('dni_wm2'), info.get('dhi_wm2'),
            info.get('power'), reward))
        if done:
            break
    env.close()


def run_env_rollout(lat, lon, tz, date, weather_source='clearsky'):
    import gym
    env = gym.make(
        'PVTracking-v0',
        latitude=lat,
        longitude=lon,
        tz=tz,
        start_date=date,
        end_date=date,
        randomize_day=False,
        randomize_initial_orientation=False,
        weather_source=weather_source,
        observation_mode='physical',
    )
    env.reset()
    powers = []
    from mbpo.env.pv_tracking import DEFAULT_EPISODE_STEPS
    for _ in range(DEFAULT_EPISODE_STEPS):
        _, _, _, info = env.step(np.zeros(2, dtype=np.float32))
        powers.append((info['time'], info['solar_altitude_deg'], info['ghi_wm2'], info['power']))
    env.close()
    times = np.array([p[0] for p in powers], dtype=np.float64)
    alts = np.array([p[1] for p in powers], dtype=np.float64)
    ip = int(np.argmax([p[3] for p in powers]))
    print('\nEnv rollout {} tz={} weather={}'.format(date, tz, weather_source))
    print('  peak power {:.1f} W at info[time]={:.2f} h, solar_alt={:.1f}'.format(
        powers[ip][3], times[ip], alts[ip]))
    print('  (info time is post-step clock in env timezone)')


def main():
    parser = argparse.ArgumentParser(description='PV time/solar/irradiance sanity check.')
    parser.add_argument('--lat', type=float, default=35.0)
    parser.add_argument('--lon', type=float, default=-106.0)
    parser.add_argument('--date', type=str, default='2020-12-21')
    parser.add_argument('--tz', type=str, default='UTC',
                        help='Env timezone (default PVTrackingEnv tz).')
    parser.add_argument('--compare-tz', type=str, default=None,
                        help='Optional second tz table (e.g. America/Denver); not used in train/plots.')
    parser.add_argument('--env-rollout', action='store_true',
                        help='Run one zero-action env episode and print peak time.')
    parser.add_argument('--debug-steps', type=int, default=0, metavar='N',
                        help='Log first N steps for timestamp/irradiance/power alignment.')
    args = parser.parse_args()

    print('PVTracking solar/time sanity')
    print('lat={}, lon={}'.format(args.lat, args.lon))

    print_fixed_hours(args.lat, args.lon, args.date, args.tz)
    if args.compare_tz:
        print_fixed_hours_with_denver(args.lat, args.lon, args.date, args.tz, args.compare_tz)
        print_fixed_hours(args.lat, args.lon, args.date, args.compare_tz)

    analyze_episode_grid(args.lat, args.lon, args.date, args.tz)
    if args.compare_tz and args.compare_tz != args.tz:
        analyze_episode_grid(args.lat, args.lon, args.date, args.compare_tz)

    if args.debug_steps > 0:
        log_step_alignment(args.lat, args.lon, args.tz, args.date, n_steps=args.debug_steps)

    if args.env_rollout:
        run_env_rollout(args.lat, args.lon, args.tz, args.date, 'clearsky')
        run_env_rollout(args.lat, args.lon, args.tz, args.date, 'random')

    print('\n--- Summary ---')
    print('Peak-power classification guide:')
    print('  A) Physically correct in UTC if solar_alt/GHI peak near 17-20h UTC in December.')
    print('  B) Misleading label if "evening" is read as local time (Denver noon ~11-13h).')
    print('  C) Episode grid is fixed UTC wall clock (see env DEFAULT_START_TIME / periods).')
    print('  D) Bug only if step table shows irradiance/power shifted vs solar_alt on same step.')
    print('  E) Policy quality is independent — compare eval energy vs baselines.')
    if args.tz.upper() == 'UTC' and args.lon < -90:
        print('With tz=UTC at {:.0f}N {:.0f}E, clock hour is UTC, not local solar time.'.format(
            args.lat, args.lon))
        print('Peak GHI often appears near 17-20h UTC (~10-13h US Mountain) in December.')
        print('Clock windows in eval_utils follow the daylight UTC grid (see TIME_WINDOWS).')
        print('No civil-time conversion is used in train, eval, or plots.')


if __name__ == '__main__':
    main()
