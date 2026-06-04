"""Per-run directories: runs/<run_name>/ for parallel training + result packaging."""

from __future__ import print_function

import json
import os
import re
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS_ROOT = os.path.join(_REPO_ROOT, 'runs')


def run_dir(run_name):
    return os.path.join(RUNS_ROOT, os.path.expanduser(str(run_name).strip()))


def run_meta_path(run_name):
    return os.path.join(run_dir(run_name), 'run.json')


def trial_pointer_path(run_name):
    return os.path.join(run_dir(run_name), 'trial_dir.txt')


def checkpoints_root(run_name):
    return os.path.join(run_dir(run_name), 'checkpoints')


def ray_trial_root(run_name):
    return os.path.join(checkpoints_root(run_name), 'PVTracking', 'pv_tracking')


def ray_tmp_dir(run_name):
    return os.path.join(run_dir(run_name), 'ray_tmp')


def results_dir(run_name):
    return os.path.join(run_dir(run_name), 'results')


def write_run_meta(run_name, conf_name, module, cpus, trial_cpus, extra=None):
    rd = run_dir(run_name)
    os.makedirs(rd, exist_ok=True)
    meta = {
        'run_name': run_name,
        'conf': conf_name,
        'config_module': module,
        'cpus': cpus,
        'trial_cpus': trial_cpus,
        'started_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'checkpoints_root': checkpoints_root(run_name),
        'ray_trial_root': ray_trial_root(run_name),
    }
    if extra:
        meta.update(extra)
    with open(run_meta_path(run_name), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)
    return meta


def read_run_meta(run_name):
    path = run_meta_path(run_name)
    if not os.path.isfile(path):
        return None
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def list_trial_dirs(root, limit=20):
    root = os.path.abspath(os.path.expanduser(root))
    if not os.path.isdir(root):
        return []
    pat = re.compile(r'^seed:\d+_')
    out = []
    for name in os.listdir(root):
        full = os.path.join(root, name)
        if os.path.isdir(full) and pat.match(name):
            out.append(full)
    out.sort(key=os.path.getmtime, reverse=True)
    return out[:limit]


def latest_trial_under(root):
    trials = list_trial_dirs(root, limit=1)
    return trials[0] if trials else None


def write_trial_pointer(run_name, trial_dir):
    trial_dir = os.path.abspath(os.path.expanduser(trial_dir))
    os.makedirs(run_dir(run_name), exist_ok=True)
    with open(trial_pointer_path(run_name), 'w', encoding='utf-8') as f:
        f.write(trial_dir + '\n')
    return trial_dir


def read_trial_pointer(run_name):
    path = trial_pointer_path(run_name)
    if not os.path.isfile(path):
        return None
    raw = open(path, encoding='utf-8').read().strip()
    if raw and os.path.isdir(raw):
        return raw
    return None


def resolve_trial_for_run(run_name, conf_name=None):
    """Trial dir for a named run (pointer file, then newest under run checkpoints)."""
    ptr = read_trial_pointer(run_name)
    if ptr and os.path.isfile(os.path.join(ptr, 'params.json')):
        return ptr, None

    meta = read_run_meta(run_name)
    root = ray_trial_root(run_name)
    if meta:
        root = meta.get('ray_trial_root', root)

    latest = latest_trial_under(root)
    if latest:
        warn = None
        if conf_name and not _trial_matches_conf(latest, conf_name):
            warn = 'Newest trial may not match conf %s: %s' % (conf_name, latest)
        return latest, warn

    return None, 'No trial under %s — run: ./train.sh %s <conf>' % (root, run_name)


def _trial_matches_conf(trial_dir, conf_name):
    try:
        params = json.load(open(os.path.join(trial_dir, 'params.json'), encoding='utf-8'))
        v = str(params.get('config_version', '')).lower()
        return conf_name.replace('_', '') in v or 'conf' in v
    except (IOError, ValueError):
        return True


def resolve_checkpoint(trial_dir):
    trial_dir = os.path.abspath(trial_dir.rstrip('/'))
    best = os.path.join(trial_dir, 'best_eval_checkpoint')
    if os.path.isdir(best):
        return best
    return os.path.join(trial_dir, 'latest_checkpoint')


def sync_trial_pointer_from_disk(run_name):
    """After training, point runs/<name>/trial_dir.txt at newest seed:* folder."""
    root = ray_trial_root(run_name)
    latest = latest_trial_under(root)
    if latest:
        return write_trial_pointer(run_name, latest)
    return None
