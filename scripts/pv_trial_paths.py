"""Resolve Ray Tune PV tracking trial directories (seed:... folders).

All Stage 3 helpers should use this module so paths stay consistent.

Default trial location (Stage 3 config):
  <repo>/mbpo_runs/PVTracking/pv_tracking/seed:<id>_<timestamp>/

Legacy location (Stages 0–2):
  ~/ray_mbpo/PVTracking/pv_tracking/seed:*/
"""

from __future__ import print_function

import json
import os
import re

# Repo root = parent of scripts/
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_PV_TRACKING_ROOT = os.path.expanduser(
    '~/ray_mbpo/PVTracking/pv_tracking')

STAGE3_RUNS_ROOT = os.path.join(_REPO_ROOT, 'mbpo_runs', 'PVTracking', 'pv_tracking')

STAGE3_ARTIFACT = os.path.join(
    _REPO_ROOT, 'sequential_stage_artifacts', 'stage3_fullyear_trial_dir.txt')

STAGE3_CONFIG_MARKERS = (
    'stage3_fullyear',
    'stage3_fullyear_stable_mbpo',
    'pv_tracking_stage3',
    'rl_annual_scenario',
    'rl_annual',
)

_PLACEHOLDER_MARKERS = (
    'YOUR_SEED_DIR',
    'YOUR_SEED',
    '<seed>',
    'seed:YOUR',
)


def is_placeholder_trial(path):
    if not path:
        return True
    p = os.path.basename(os.path.expanduser(path.rstrip('/')))
    if any(m in p for m in _PLACEHOLDER_MARKERS):
        return True
    return p in ('seed:YOUR_SEED_DIR',)


def search_roots(preferred=None):
    """Ordered list of directories that may contain seed:* trials."""
    roots = []
    if preferred:
        roots.append(os.path.expanduser(preferred))
    env_root = os.environ.get('RAY_ROOT')
    if env_root:
        roots.append(os.path.expanduser(env_root))
    roots.append(STAGE3_RUNS_ROOT)
    roots.append(DEFAULT_PV_TRACKING_ROOT)
    seen = set()
    out = []
    for r in roots:
        r = os.path.expanduser(r)
        if r not in seen:
            seen.add(r)
            out.append(r)
    return out


def list_trial_dirs(root=None, limit=10):
    root = os.path.expanduser(root or STAGE3_RUNS_ROOT)
    if not os.path.isdir(root):
        return []
    pattern = re.compile(r'^seed:\d+_')
    trials = []
    for name in os.listdir(root):
        full = os.path.join(root, name)
        if os.path.isdir(full) and pattern.match(name):
            trials.append(full)
    trials.sort(key=os.path.getmtime, reverse=True)
    return trials[:limit]


def latest_trial_dir(root=None):
    trials = list_trial_dirs(root, limit=1)
    if not trials:
        return None
    return trials[0]


def _read_artifact(path):
    if not path or not os.path.isfile(path):
        return None
    raw = open(path, encoding='utf-8').read().strip()
    if not raw or is_placeholder_trial(raw):
        return None
    expanded = os.path.expanduser(raw)
    if os.path.isdir(expanded) and os.path.isfile(os.path.join(expanded, 'params.json')):
        return expanded
    return None


def _is_stage3_trial(trial_dir):
    params = os.path.join(trial_dir, 'params.json')
    if not os.path.isfile(params):
        return False
    try:
        version = json.load(open(params, encoding='utf-8')).get('config_version', '')
    except (IOError, ValueError):
        return False
    v = str(version).lower()
    return any(m in v for m in STAGE3_CONFIG_MARKERS)


def find_stage3_trial(artifact_path=None, preferred_root=None):
    """Best-effort Stage 3 trial directory.

    Priority:
      1. $TRIAL environment variable
      2. Artifact file (stage3_fullyear_trial_dir.txt)
      3. Newest Stage 3 trial under search_roots
      4. Newest any trial under search_roots (with warning)
    """
    env_trial = os.environ.get('TRIAL', '').strip()
    if env_trial and not is_placeholder_trial(env_trial):
        try:
            return resolve_trial_dir(env_trial), None
        except FileNotFoundError:
            pass

    artifact = _read_artifact(artifact_path or STAGE3_ARTIFACT)
    if artifact:
        return artifact, None

    for root in search_roots(preferred_root):
        if not os.path.isdir(root):
            continue
        for trial in list_trial_dirs(root, limit=50):
            if _is_stage3_trial(trial):
                return trial, None

    for root in search_roots(preferred_root):
        latest = latest_trial_dir(root)
        if latest:
            warn = (
                'Using newest trial (not verified as Stage 3 config): %s\n'
                '  Run ./scripts/run_stage3_fullyear.sh train for a fresh Stage 3 trial.'
                % latest)
            return latest, warn

    return None, None


def resolve_trial_dir(path_or_latest=None, root=None):
    """Return absolute trial directory containing params.json.

    path_or_latest:
      - None / 'latest' / placeholder → newest seed:* under root (or search_roots)
      - path to seed:... dir or progress.csv or checkpoint_*
    """
    if path_or_latest not in (None, '', 'latest') and not is_placeholder_trial(path_or_latest):
        path = os.path.expanduser(path_or_latest.rstrip('/'))
        if os.path.isfile(path):
            path = os.path.dirname(path)
        if os.path.basename(path).startswith('checkpoint') or path.endswith(
                ('best_eval_checkpoint', 'latest_checkpoint')):
            path = os.path.dirname(path)
        params = os.path.join(path, 'params.json')
        if os.path.isfile(params):
            return path
        hint_roots = search_roots(root)
        recent = []
        for r in hint_roots:
            recent.extend(list_trial_dirs(r, limit=3))
        raise FileNotFoundError(
            'Trial directory not found: %s\n'
            'Expected seed:.../ with params.json.\n'
            'Recent trials:\n  %s' % (
                path_or_latest,
                '\n  '.join(recent[:5]) or '(none — run train first)'))

    roots = search_roots(root)
    for r in roots:
        trial = latest_trial_dir(r)
        if trial is not None:
            return trial

    lines = ['No PV tracking trials found. Searched:']
    for r in roots:
        lines.append('  %s' % r)
    lines.append('Run: ./scripts/run_stage3_fullyear.sh train')
    raise FileNotFoundError('\n'.join(lines))


def format_trial_hint(root=None):
    trials = []
    for r in search_roots(root):
        trials.extend(list_trial_dirs(r, limit=3))
    lines = ['Set TRIAL to a real seed folder, for example:']
    if trials:
        lines.append('  export TRIAL=%s' % trials[0])
    else:
        lines.append('  (no trials yet — run ./scripts/run_stage3_fullyear.sh train)')
    lines.append('Stage 3 trials are saved under:')
    lines.append('  %s' % STAGE3_RUNS_ROOT)
    return '\n'.join(lines)


def trial_status_lines():
    """Human-readable status for run_stage3_fullyear.sh status."""
    lines = [
        'PV tracking paths',
        '=' * 40,
        'Repo:              %s' % _REPO_ROOT,
        'Stage 3 runs:      %s' % STAGE3_RUNS_ROOT,
        'Legacy runs:       %s' % DEFAULT_PV_TRACKING_ROOT,
        'Artifact file:     %s' % STAGE3_ARTIFACT,
    ]
    art = _read_artifact(STAGE3_ARTIFACT)
    lines.append('Artifact points:   %s' % (art or '(missing — train not finished)'))
    lines.append('')
    for label, root in (
        ('Stage 3 dir', STAGE3_RUNS_ROOT),
        ('Legacy dir', DEFAULT_PV_TRACKING_ROOT),
    ):
        trials = list_trial_dirs(root, limit=5)
        lines.append('%s (%d recent):' % (label, len(trials)))
        if trials:
            for t in trials:
                flag = ' [stage3]' if _is_stage3_trial(t) else ''
                prog = os.path.join(t, 'progress.csv')
                n = ''
                if os.path.isfile(prog):
                    try:
                        import pandas as pd
                        n = ' epochs=%d' % len(pd.read_csv(prog))
                    except Exception:
                        n = ' (progress.csv present)'
                lines.append('  %s%s%s' % (t, n, flag))
        else:
            lines.append('  (empty)')
    trial, warn = find_stage3_trial()
    lines.append('')
    if trial:
        lines.append('Resolved TRIAL:    %s' % trial)
        if warn:
            lines.append('Warning: %s' % warn)
    else:
        lines.append('Resolved TRIAL:    (none — run train first)')
    return lines
