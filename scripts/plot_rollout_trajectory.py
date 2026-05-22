#!/usr/bin/env python3
"""Plot a combined trajectory figure from a saved rollout CSV file.

This script reads one rollout CSV exported by scripts/evaluate_agent.py and
produces a single combined plot containing:
- power
- tilt
- azimuth
- reward
- time axis

Example:
    python scripts/plot_rollout_trajectory.py \
      --csv evaluation/pv_tracking/rollouts/rollout_1.csv \
      --outdir evaluation/pv_tracking/rollout_plots
"""
from __future__ import print_function
import argparse
import csv
import glob
import os
import re
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError as e:
    raise SystemExit('matplotlib is required for this script. Install it with: pip install matplotlib')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Plot a combined rollout trajectory figure from saved CSV.')
    parser.add_argument('--csv', required=True,
                        help='Path to rollout CSV file.')
    parser.add_argument('--outdir', default='evaluation',
                        help='Directory where output plot is saved.')
    parser.add_argument('--name', default=None,
                        help='Output filename without extension.')
    return parser.parse_args()


def resolve_csv_path(csv_pattern):
    csv_path = os.path.expanduser(csv_pattern)
    if os.path.exists(csv_path):
        return csv_path

    wildcard_path = re.sub(r'<[^>]+>', '*', csv_path)
    matches = glob.glob(wildcard_path)
    if matches:
        return max(matches, key=os.path.getmtime)

    # allow glob wildcards directly
    matches = glob.glob(csv_path)
    if matches:
        return max(matches, key=os.path.getmtime)

    raise FileNotFoundError(
        'Rollout CSV not found: %s\n'
        'Use a real rollout CSV path or pattern.\n'
        'Example: evaluation/pv_tracking/rollouts/rollout_1.csv'
        % csv_pattern)


def load_rollout_csv(filepath):
    filepath = resolve_csv_path(filepath)
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [row for row in reader if row]

    data = {col: [] for col in header}
    for row in rows:
        for col, value in zip(header, row):
            data[col].append(value)

    result = {}
    for col, values in data.items():
        try:
            result[col] = np.array([float(v) if v != '' else np.nan for v in values], dtype=np.float64)
        except ValueError:
            result[col] = np.array(values)
    return result


def plot_rollout(data, outdir, name):
    os.makedirs(outdir, exist_ok=True)
    if name is None:
        name = 'rollout_combined'

    time = data.get('clock_hour_utc')
    x_label = 'Clock hour UTC (post-step)'
    if time is None or not np.isfinite(time).all():
        time = data.get('clock_hour_env_tz', data.get('time'))
        if time is None or not np.isfinite(np.asarray(time, dtype=np.float64)).all():
            time = None
        else:
            x_label = 'Clock hour UTC (legacy column name, post-step)'
    if time is not None and np.isfinite(time).all():
        time = np.asarray(time, dtype=np.float64)
    else:
        time = np.arange(len(data.get('reward', [])), dtype=np.float64)
        x_label = 'Step'

    power = data.get('power_w', data.get('power', np.full_like(time, np.nan)))
    tilt = data.get('tilt_deg', data.get('tilt', np.full_like(time, np.nan)))
    azimuth = data.get('azimuth_deg', data.get('azimuth', np.full_like(time, np.nan)))
    reward = data.get('reward', np.full_like(time, np.nan))

    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(10, 12))
    axes[0].plot(time, power, marker='o', linestyle='-', color='#1f77b4')
    axes[0].set_ylabel('Power (W)')
    axes[0].grid(True, linestyle='--', alpha=0.4)

    axes[1].plot(time, tilt, marker='o', linestyle='-', color='#ff7f0e')
    axes[1].set_ylabel('Tilt (deg)')
    axes[1].grid(True, linestyle='--', alpha=0.4)

    axes[2].plot(time, azimuth, marker='o', linestyle='-', color='#2ca02c')
    axes[2].set_ylabel('Azimuth (deg)')
    axes[2].grid(True, linestyle='--', alpha=0.4)

    axes[3].plot(time, reward, marker='o', linestyle='-', color='#d62728')
    axes[3].set_ylabel('Reward')
    axes[3].set_xlabel(x_label)
    axes[3].grid(True, linestyle='--', alpha=0.4)

    fig.suptitle('Rollout trajectory')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    outpath = os.path.join(outdir, f'{name}.png')
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    return outpath


def main():
    args = parse_args()
    data = load_rollout_csv(os.path.expanduser(args.csv))
    outpath = plot_rollout(data, os.path.expanduser(args.outdir), args.name)
    print('Saved:', outpath)


if __name__ == '__main__':
    main()
