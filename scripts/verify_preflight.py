#!/usr/bin/env python3
"""Preflight checks for any registered conf (used by train.sh --verify)."""
from __future__ import print_function

import argparse
import importlib
import os
import subprocess
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

DEFAULT_MODULE = 'examples.config.pv_tracking.conf1'
DEFAULT_PATH = os.path.join(
    _REPO_ROOT, 'examples/config/pv_tracking/conf1.py')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default=DEFAULT_MODULE)
    p.add_argument('--config-path', default=DEFAULT_PATH)
    p.add_argument('--outdir', default='verification/preflight')
    args = p.parse_args()

    os.makedirs(os.path.join(_REPO_ROOT, args.outdir), exist_ok=True)
    errors = []

    for cmd in (
        [sys.executable, 'scripts/verify_training_config.py', '--file-only', '--config', args.config],
        [sys.executable, 'scripts/validate_pv_rollouts.py', '--config-path', args.config_path],
        [sys.executable, 'scripts/verify_movement_cost_fairness.py', '--config-path', args.config_path],
        [sys.executable, 'scripts/verify_utc_uniformity.py'],
    ):
        print('[verify_preflight]', ' '.join(cmd))
        if subprocess.run(cmd, cwd=_REPO_ROOT).returncode != 0:
            errors.append('failed: %s' % cmd[1])

    mod = importlib.import_module(args.config)
    print('[verify_preflight] CONFIG_VERSION=%s' % getattr(mod, 'CONFIG_VERSION', '?'))

    env_kw = getattr(mod, 'params', {}).get('environment_kwargs', {})
    if env_kw.get('weather_source') == 'nsrdb_multiyear':
        for script in ('verify_nsrdb_multiyear.py', 'verify_nsrdb_timing_sync.py'):
            cmd = [sys.executable, 'scripts/' + script,
                   '--config', args.config]
            if script == 'verify_nsrdb_multiyear.py':
                cmd.extend(['--config-path', args.config_path])
            print('[verify_preflight]', ' '.join(cmd))
            if subprocess.run(cmd, cwd=_REPO_ROOT).returncode != 0:
                errors.append('failed: %s' % script)

    if errors:
        raise SystemExit('Preflight FAILED:\n  ' + '\n  '.join(errors))
    print('Preflight OK (%s)' % args.config)


if __name__ == '__main__':
    main()
