#!/usr/bin/env python3
"""Plot training/evaluation curves from a Ray Tune progress.csv file."""

import argparse
import os

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
        help='Path to progress.csv (trial directory or file path).')
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


def resolve_progress_path(path):
    path = path.rstrip('/')
    if os.path.isfile(path):
        return path
    candidate = os.path.join(path, 'progress.csv')
    if os.path.isfile(candidate):
        return candidate
    raise FileNotFoundError('progress.csv not found at: %s' % path)


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
    progress_path = resolve_progress_path(args.progress_csv)
    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(progress_path)
    # Drop duplicate terminal rows Ray sometimes appends at the end.
    if 'done' in df.columns:
        df = df[df['done'] != True]  # noqa: E712

    created = []
    for metric in args.metrics:
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
