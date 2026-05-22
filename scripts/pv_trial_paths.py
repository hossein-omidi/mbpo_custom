"""Resolve Ray Tune PV tracking trial directories (seed:... folders)."""

from __future__ import print_function

import os
import re

DEFAULT_PV_TRACKING_ROOT = os.path.expanduser(
    '~/ray_mbpo/PVTracking/pv_tracking')

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


def list_trial_dirs(root=None, limit=10):
    root = os.path.expanduser(root or DEFAULT_PV_TRACKING_ROOT)
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


def resolve_trial_dir(path_or_latest=None, root=None):
    """Return absolute trial directory containing params.json.

    path_or_latest:
      - None / 'latest' / placeholder → newest seed:* under root
      - path to seed:... dir or progress.csv or checkpoint_*
    """
    root = os.path.expanduser(root or DEFAULT_PV_TRACKING_ROOT)

    if path_or_latest in (None, '', 'latest') or is_placeholder_trial(path_or_latest):
        trial = latest_trial_dir(root)
        if trial is None:
            raise FileNotFoundError(
                'No PV tracking trials under %s. Train first or pass a seed:... path.'
                % root)
        return trial

    path = os.path.expanduser(path_or_latest.rstrip('/'))
    if os.path.isfile(path):
        path = os.path.dirname(path)
    if os.path.basename(path).startswith('checkpoint') or path.endswith(
            ('best_eval_checkpoint', 'latest_checkpoint')):
        path = os.path.dirname(path)

    params = os.path.join(path, 'params.json')
    if os.path.isfile(params):
        return path

    raise FileNotFoundError(
        'Trial directory not found: %s\n'
        'Expected a Ray folder like %s/seed:1234_.../ with params.json inside.\n'
        'Recent trials:\n  %s' % (
            path_or_latest,
            root,
            '\n  '.join(list_trial_dirs(root, limit=5)) or '(none)'))


def format_trial_hint(root=None):
    trials = list_trial_dirs(root, limit=5)
    lines = ['Set TRIAL to a real seed folder, for example:']
    if trials:
        lines.append('  export TRIAL=%s' % trials[0])
        lines.append('Or use latest automatically:')
        lines.append('  export TRIAL=$(ls -td %s/seed:*/ | head -1)' % (
            os.path.expanduser(root or DEFAULT_PV_TRACKING_ROOT)))
    else:
        lines.append('  (no seed:* trials found under %s)' % (
            os.path.expanduser(root or DEFAULT_PV_TRACKING_ROOT)))
    return '\n'.join(lines)
