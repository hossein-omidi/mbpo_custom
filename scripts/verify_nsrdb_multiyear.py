#!/usr/bin/env python3
"""Preflight for NSRDB multi-year scenario training (stage3_multiyear_nsrdb_scenario)."""

from __future__ import print_function

import argparse
import importlib
import os
import subprocess
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import numpy as np

from mbpo.env.pv_tracking import PVTrackingEnv
from mbpo.env.nsrdb_weather import load_scenario_manifest, scenario_id


DEFAULT_MODULE = 'examples.config.pv_tracking.stage3_multiyear_nsrdb_scenario'
DEFAULT_PATH = os.path.join(
    _REPO, 'examples/config/pv_tracking/stage3_multiyear_nsrdb_scenario.py')


def _episode_energy(env, seed):
    env.seed(seed)
    env.reset()
    total = 0.0
    done = False
    while not done:
        _, r, done, _ = env.step(np.zeros(2, dtype=np.float32))
        total += float(r)
    env.close()
    return total


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default=DEFAULT_MODULE)
    p.add_argument('--config-path', default=DEFAULT_PATH)
    p.add_argument('--outdir', default='verification/nsrdb_multiyear')
    args = p.parse_args()

    os.makedirs(os.path.join(_REPO, args.outdir), exist_ok=True)
    errors = []
    lines = ['NSRDB multi-year preflight', '=' * 28, '']

    for cmd in (
        [sys.executable, 'scripts/verify_training_config.py', '--file-only',
         '--config', args.config],
        [sys.executable, 'scripts/validate_pv_rollouts.py', '--config-path', args.config_path],
        [sys.executable, 'scripts/verify_movement_cost_fairness.py', '--config-path', args.config_path],
    ):
        print('[verify_nsrdb]', ' '.join(cmd))
        if subprocess.run(cmd, cwd=_REPO).returncode != 0:
            errors.append('failed: %s' % cmd[1])

    mod = importlib.import_module(args.config)
    params = mod.params
    env_kw = dict(params['environment_kwargs'])
    manifest_path = env_kw.get('scenario_manifest')
    lines.append('CONFIG: %s' % getattr(mod, 'CONFIG_VERSION', ''))
    lines.append('manifest: %s' % manifest_path)

    try:
        manifest = load_scenario_manifest(manifest_path)
    except FileNotFoundError as exc:
        errors.append(str(exc))
        manifest = None

    if manifest is not None:
        n = len(manifest['scenarios'])
        years = sorted({int(s['year']) for s in manifest['scenarios']})
        lines.append('scenarios: %d  years: %s' % (n, years))
        if n < 100:
            errors.append('manifest has only %d scenarios (expected hundreds+)' % n)

        by_mday = {}
        for s in manifest['scenarios']:
            key = (int(s['month']), int(s['day']))
            by_mday.setdefault(key, []).append(int(s['year']))
        multi_days = sum(1 for ys in by_mday.values() if len(set(ys)) > 1)
        lines.append('calendar days with multiple years: %d' % multi_days)
        if multi_days < 50:
            errors.append(
                'few multi-year calendar days (%d); download more years' % multi_days)

    env = PVTrackingEnv(**env_kw)
    try:
        seen_ids = set()
        for seed in range(50):
            env.seed(seed)
            env.reset()
            sid = scenario_id(env._current_scenario)
            seen_ids.add(sid)
        lines.append('50 random resets -> %d unique scenarios' % len(seen_ids))

        # Same scenario_id + seed => deterministic energy.
        env.seed(42)
        env.reset()
        target = scenario_id(env._current_scenario)
        e1 = 0.0
        done = False
        while not done:
            _, r, done, _ = env.step(np.zeros(2))
            e1 += r
        for attempt in range(30):
            env.seed(42)
            env.reset()
            if scenario_id(env._current_scenario) == target:
                break
        else:
            errors.append('could not reproduce same scenario_id with seed=42')
        e2 = 0.0
        done = False
        while not done:
            _, r, done, _ = env.step(np.zeros(2))
            e2 += r
        if abs(e1 - e2) > 1e-4:
            errors.append('same scenario+seed not deterministic: %.6f vs %.6f' % (e1, e2))
        else:
            lines.append('same scenario+seed deterministic: energy=%.4f kWh' % e1)

        pool = env._nsrdb_by_mday.get((4, 21), [])
        if len(pool) >= 2:
            vals = []
            for s in sorted(pool, key=lambda x: int(x['year']))[:5]:
                e = PVTrackingEnv(**env_kw)
                e._current_scenario = s
                from mbpo.env.nsrdb_weather import scenario_episode_date
                e.current_date = scenario_episode_date(s, tz=e.location.tz)
                e.times = e._build_times(e.current_date)
                e.weather_profile = e._build_weather_profile(e.times)
                e.tilt, e.azimuth = 30.0, 180.0
                e.step_index = 0
                tot = 0.0
                done = False
                while not done:
                    _, r, done, _ = e.step(np.zeros(2, dtype=np.float32))
                    tot += float(r)
                e.close()
                vals.append(tot)
            if len(set(round(v, 4) for v in vals)) < 2:
                errors.append(
                    'April 21 scenarios do not differ across years: %s' % vals)
            else:
                lines.append('April 21 multi-year energies: %s' % [
                    round(v, 4) for v in vals])
        else:
            lines.append('April 21: only %d scenario(s) in manifest' % len(pool))
    finally:
        env.close()

    report = os.path.join(_REPO, args.outdir, 'preflight_report.txt')
    with open(report, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n\n')
        if errors:
            f.write('FAIL:\n')
            for e in errors:
                f.write('  - %s\n' % e)
        else:
            f.write('PASS — NSRDB multi-year ready.\n')

    print('\n'.join(lines))
    if errors:
        for e in errors:
            print(' ', e)
        return 1
    print('PASS —', report)
    return 0


if __name__ == '__main__':
    sys.exit(main())
