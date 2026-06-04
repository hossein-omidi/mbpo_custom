"""Portable repo paths for PV configs (no machine-specific absolutes)."""
import os

_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def repo_root():
    return _REPO_ROOT


def default_log_dir(subdir='mbpo_runs'):
    """Checkpoints root: $MBPO_LOG_DIR or <repo>/mbpo_runs."""
    return os.path.abspath(os.environ.get(
        'MBPO_LOG_DIR',
        os.path.join(_REPO_ROOT, subdir),
    ))
