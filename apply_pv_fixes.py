#!/usr/bin/env python3
"""Small, verified-source fixes for MBPO custom PV simulator (commit eba24fd).

Usage from repository root:
    python apply_pv_fixes.py --check   # review exact changes, NO writes
    python apply_pv_fixes.py --apply   # backup + make minimal source edits

Changes:
  - NumPy Generator/RandomState compatible scenario index sampling (3 sites).
  - Geometry-mode actuator energy uses executed tilt rather than clipped request,
    in BOTH real PVTrackingEnv and MBPO StructuredFakeEnv.

Does not change the legacy action-penalty mode, training configurations,
weather data, experiment window, or existing artifacts.
"""
from __future__ import annotations

import argparse
import ast
import difflib
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

FILES = (
    Path('mbpo/env/pv_tracking.py'),
    Path('MBPO_workspace/src/dynamics.py'),
)


def patch_real(src):
    old = "class PVTrackingEnv(gym.Env):"
    new = '''def _rng_index(rng, high):
    """Select an index using NumPy Generator or legacy RandomState."""
    if hasattr(rng, 'integers'):
        return int(rng.integers(high))
    return int(rng.randint(high))


class PVTrackingEnv(gym.Env):'''
    src = replace_once(src, old, new, 'RNG helper')

    src = replace_count(src,
        'self.np_random.randint(len(self.start_dates))',
        '_rng_index(self.np_random, len(self.start_dates))', 2,
        'calendar-day and non-NSRDB random-date sampling')
    src = replace_once(src,
        'pool[self.np_random.randint(len(pool))]',
        'pool[_rng_index(self.np_random, len(pool))]',
        'year sampling for selected month/day')
    src = replace_once(src,
        '    def step(self, action):\n        delta_tilt =',
        '    def step(self, action):\n        previous_tilt = float(self.tilt)\n        delta_tilt =',
        'record previous tilt')
    src = replace_once(src,
        '        self.tilt = clip_panel_tilt_deg(self.tilt + delta_tilt)\n        self.azimuth =',
        '        self.tilt = clip_panel_tilt_deg(self.tilt + delta_tilt)\n'
        '        executed_delta_tilt = float(self.tilt - previous_tilt)\n        self.azimuth =',
        'compute executed tilt delta')
    src = replace_once(src,
        "        elif self.movement_cost_mode == 'geometry':\n"
        "            movement_cost = actuator_movement_cost_kwh(\n"
        "                delta_tilt,\n"
        "                delta_azimuth,",
        "        elif self.movement_cost_mode == 'geometry':\n"
        "            movement_cost = actuator_movement_cost_kwh(\n"
        "                executed_delta_tilt,\n"
        "                delta_azimuth,",
        'real geometry actuator energy')
    return src


def patch_fake(src):
    src = replace_once(src,
        '        nxt[:,3]=np.clip(states[:,3]*90.+requested_tilt_delta,0.,90.)/90.',
        '        old_tilt_deg=states[:,3]*90.\n'
        '        executed_tilt_deg=np.clip(old_tilt_deg+requested_tilt_delta,0.,90.)\n'
        '        executed_tilt_delta=executed_tilt_deg-old_tilt_deg\n'
        '        nxt[:,3]=executed_tilt_deg/90.',
        'fake environment executed tilt')
    src = replace_once(src,
        'for dt,da in zip(requested_tilt_delta,requested_az_delta)],dtype=np.float32)',
        'for dt,da in zip(executed_tilt_delta,requested_az_delta)],dtype=np.float32)',
        'fake environment geometry actuator energy')
    return src


def replace_count(src, before, after, expected, label):
    n = src.count(before)
    if n != expected:
        raise ValueError(f'{label}: expected {expected} exact source occurrences; found {n}. '
                         'Source may have changed; do not force patch.')
    return src.replace(before, after)


def replace_once(src, before, after, label):
    return replace_count(src, before, after, 1, label)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check', action='store_true', help='Show proposed diff only; no writes')
    mode.add_argument('--apply', action='store_true', help='Backup and apply source changes')
    args = parser.parse_args()

    root = Path.cwd().resolve()
    if not all((root / p).is_file() for p in FILES):
        raise SystemExit('Run from mbpo_custom repository root (both source files required).')

    changes = {}
    for rel, fn in ((FILES[0], patch_real), (FILES[1], patch_fake)):
        path = root / rel
        original = path.read_text(encoding='utf-8')
        modified = fn(original)
        ast.parse(modified, filename=str(rel))
        changes[rel] = (original, modified)

    print('PATCH PREFLIGHT: all exact source patterns matched; both files parse as Python.')
    for rel, (before, after) in changes.items():
        diff = difflib.unified_diff(before.splitlines(keepends=True),
                                    after.splitlines(keepends=True),
                                    fromfile='a/' + str(rel), tofile='b/' + str(rel))
        print(''.join(diff), end='')

    if args.check:
        print('\nCHECK ONLY: repository files were NOT changed.')
        return 0

    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    backup = root / '.pv_fix_backups' / stamp
    for rel, (before, _) in changes.items():
        dst = backup / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(before, encoding='utf-8')
    print(f'Backups created at: {backup}')
    written = []
    try:
        for rel, (_, after) in changes.items():
            path = root / rel
            tmp = path.with_name(path.name + '.pvfix.tmp')
            try:
                tmp.write_text(after, encoding='utf-8')
                tmp.chmod(path.stat().st_mode)
                tmp.replace(path)
            finally:
                if tmp.exists():
                    tmp.unlink()
            written.append(rel)
    except Exception:
        for rel in written:
            shutil.copy2(backup / rel, root / rel)
        raise
    print('\nAPPLIED: minimal changes to both source files. No training configuration changed.')
    print('NEXT: run the V4 regression audit and inspect: git diff --check && git diff')
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, ValueError, SyntaxError) as exc:
        print(f'PATCH ABORTED: {exc}', file=sys.stderr)
        sys.exit(2)
