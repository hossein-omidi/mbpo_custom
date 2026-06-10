"""Tests for Ray progress.csv segment detection and global_step plotting."""

import importlib.util
import os
import sys

import pandas as pd
import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRIPT = os.path.join(_REPO_ROOT, 'scripts', 'plot_training_progress.py')


def _load_plot_module():
    spec = importlib.util.spec_from_file_location('plot_training_progress', _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, os.path.join(_REPO_ROOT, 'scripts'))
    spec.loader.exec_module(mod)
    return mod


def _synthetic_progress(epoch_values):
    return pd.DataFrame({
        'epoch': epoch_values,
        'training_iteration': [e + 1 for e in epoch_values],
        'evaluation/return-average': [float(i) for i in range(len(epoch_values))],
        'done': [False] * len(epoch_values),
    })


def test_single_segment_global_step_matches_epoch():
    mod = _load_plot_module()
    raw = _synthetic_progress(list(range(10)))
    df, meta = mod.prepare_progress_dataframe(raw)
    assert meta['n_segments'] == 1
    assert meta['n_restores'] == 0
    assert list(df['global_step']) == list(range(10))
    assert list(df['segment_id']) == [0] * 10


def test_three_segment_global_step_user_example():
    mod = _load_plot_module()
    epochs = (
        list(range(599))
        + list(range(599))
        + list(range(286))
    )
    raw = _synthetic_progress(epochs)
    df, meta = mod.prepare_progress_dataframe(raw)
    assert meta['n_segments'] == 3
    assert meta['n_restores'] == 2
    assert int(df['global_step'].iloc[0]) == 0
    assert int(df['global_step'].iloc[598]) == 598
    assert int(df['global_step'].iloc[599]) == 599
    assert int(df['global_step'].iloc[1197]) == 1197
    assert int(df['global_step'].iloc[1198]) == 1198
    assert int(df['global_step'].iloc[-1]) == 1483
    assert meta['original_row_count'] == len(epochs)


def test_segment_detected_on_training_iteration_reset_only():
    mod = _load_plot_module()
    raw = pd.DataFrame({
        'training_iteration': [1, 2, 3, 1, 2],
        'evaluation/return-average': [0.1, 0.2, 0.3, 0.4, 0.5],
    })
    df, meta = mod.prepare_progress_dataframe(raw)
    assert meta['n_segments'] == 2
    assert list(df['segment_id']) == [0, 0, 0, 1, 1]
    # Cumulative offsets: segment0 steps 0..2, segment1 continues at 3..4
    assert list(df['global_step']) == [0, 1, 2, 3, 4]
