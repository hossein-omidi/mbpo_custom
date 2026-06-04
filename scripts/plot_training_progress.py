#!/usr/bin/env python3
"""Plot training/evaluation curves from a Ray Tune progress.csv file."""

import argparse
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from pv_trial_paths import resolve_trial_dir, find_stage3_trial, format_trial_hint, is_placeholder_trial

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_METRICS = (
    'evaluation/return-average',
    'training/return-average',
    'model/val_loss',
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Plot metrics from an MBPO progress.csv file.')
    parser.add_argument(
        'progress_csv',
        type=str,
        nargs='?',
        default='latest',
        help='Trial dir (seed:...), progress.csv path, or "latest" (default).')
    parser.add_argument(
        '--ray-root',
        type=str,
        default=None,
        help='PV tracking log root (default ~/ray_mbpo/PVTracking/pv_tracking).')
    parser.add_argument(
        '--outdir',
        type=str,
        default='training_plots',
        help='Directory for saved figures.')
    parser.add_argument(
        '--metrics', '-m',
        nargs='+',
        default=list(DEFAULT_METRICS),
        help='Column names to plot (must exist in progress.csv).')
    return parser.parse_args()


def resolve_progress_path(path, ray_root=None):
    if path in (None, '', 'latest') or is_placeholder_trial(path):
        trial, warn = find_stage3_trial(preferred_root=ray_root)
        if trial is None:
            raise FileNotFoundError(format_trial_hint(ray_root))
        if warn:
            print('[plot_training_progress] WARNING: %s' % warn)
        print('[plot_training_progress] Using trial: %s' % trial)
        path = trial
    path = path.rstrip('/')
    if os.path.isfile(path):
        progress = path
    else:
        progress = os.path.join(path, 'progress.csv')
    if not os.path.isfile(progress):
        raise FileNotFoundError(
            'progress.csv not found at: %s\n%s' % (path, format_trial_hint(ray_root)))
    if os.path.getsize(progress) < 10:
        raise FileNotFoundError(
            'progress.csv is empty (training just started?): %s\n'
            'Wait for epoch 1 to finish, then re-run plot.' % progress)
    return progress


RETURN_METRIC_PAIRS = (
    ('evaluation/return-average', 'evaluation/return-std'),
    ('training/return-average', 'training/return-std'),
)


def plot_metric_with_std(df, mean_col, std_col, outdir):
    """In-train MBPO eval: Ray reports batch mean and std over n_eval episodes."""
    if mean_col not in df.columns:
        print('Skipping missing metric: %s' % mean_col)
        return None

    x = df['training_iteration'] if 'training_iteration' in df.columns else range(len(df))
    y = df[mean_col]
    has_std = std_col in df.columns

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, y, marker='o', linestyle='-', linewidth=2, color='#2171b5',
            label='E[R] (batch mean)')
    if has_std:
        sig = df[std_col].fillna(0.0)
        ax.fill_between(x, y - sig, y + sig, color='#2171b5', alpha=0.2,
                        label='±σ (batch std)')
        ax.legend(loc='best', fontsize=9)
    ax.set_title('%s — expected return process' % mean_col)
    ax.set_xlabel('Training iteration')
    ax.set_ylabel(mean_col)
    ax.grid(True, linestyle='--', alpha=0.4)
    fig.tight_layout()

    safe_name = mean_col.replace('/', '_')
    filepath = os.path.join(outdir, '%s.png' % safe_name)
    fig.savefig(filepath, dpi=150)
    plt.close(fig)
    return filepath


def plot_metric(df, metric, outdir):
    if metric not in df.columns:
        print('Skipping missing metric: %s' % metric)
        return None

    x = df['training_iteration'] if 'training_iteration' in df.columns else range(len(df))
    y = df[metric]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, y, marker='o', linestyle='-', linewidth=2)
    ax.set_title(metric)
    ax.set_xlabel('Training iteration')
    ax.set_ylabel(metric)
    ax.grid(True, linestyle='--', alpha=0.4)
    fig.tight_layout()

    safe_name = metric.replace('/', '_')
    filepath = os.path.join(outdir, '%s.png' % safe_name)
    fig.savefig(filepath, dpi=150)
    plt.close(fig)
    return filepath


def main():
    args = parse_args()
    progress_path = resolve_progress_path(args.progress_csv, ray_root=args.ray_root)
    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(progress_path)
    # Drop duplicate terminal rows Ray sometimes appends at the end.
    if 'done' in df.columns:
        df = df[df['done'] != True]  # noqa: E712

    created = []
    plotted_return = set()
    for mean_col, std_col in RETURN_METRIC_PAIRS:
        if mean_col in args.metrics or mean_col in df.columns:
            path = plot_metric_with_std(df, mean_col, std_col, args.outdir)
            if path:
                created.append(path)
                plotted_return.add(mean_col)

    for metric in args.metrics:
        if metric in plotted_return:
            continue
        path = plot_metric(df, metric, args.outdir)
        if path:
            created.append(path)

    summary_path = os.path.join(args.outdir, 'training_summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write('Source: %s\n' % progress_path)
        f.write('Rows: %d\n' % len(df))
        for metric in args.metrics:
            if metric in df.columns:
                f.write('%s: first=%.6f last=%.6f\n' % (
                    metric, df[metric].iloc[0], df[metric].iloc[-1]))

    print('Training plots saved to: %s' % args.outdir)
    for path in created:
        print('  %s' % path)
    print('  %s' % summary_path)


if __name__ == '__main__':
    main()
