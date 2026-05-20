#!/usr/bin/env python3
"""Evaluate a checkpoint, produce plots, and optionally launch viskit.

Usage examples:

  # Evaluate checkpoint and produce plots, do not start server
  python scripts/evaluate_and_viskit.py \
    --ckpt-dir ~/ray_mbpo/PVTracking/pv_tracking/seed:.../checkpoint_0 \
    --outdir evaluation/pv_tracking --no-server

  # Evaluate then launch viskit on default port 6008
  python scripts/evaluate_and_viskit.py \
    --ckpt-dir ~/ray_mbpo/PVTracking/pv_tracking/seed:.../checkpoint_0 \
    --outdir evaluation/pv_tracking

This script is intentionally small and delegates work to existing scripts
in the repository so no large structural changes are required.
"""
from __future__ import print_function
import argparse
import glob
import os
import re
import subprocess
import sys


def run_command(cmd, env=None):
    print('Running:', ' '.join(cmd))
    subprocess.check_call(cmd, env=env)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt-dir', required=True,
                        help='Checkpoint directory (checkpoint_N)')
    parser.add_argument('--outdir', default='evaluation/pv_tracking',
                        help='Output directory for evaluation/plots')
    parser.add_argument('--num-rollouts', type=int, default=10)
    parser.add_argument('--max-path-length', type=int, default=63)
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--port', type=int, default=6008,
                        help='Port to run viskit on')
    parser.add_argument('--no-server', action='store_true',
                        help="Don't launch viskit server; only evaluate and plot")
    args = parser.parse_args()

    def resolve_ckpt_dir(path):
        p = os.path.expanduser(path)

        # If user passed a placeholder like <seed> or <timestamp>, replace with glob
        if '<' in p or '>' in p:
            pat = re.sub(r'<[^>]*>', '*', p)
            matches = glob.glob(pat)
            if not matches:
                # fallback: search common experiment root
                base = os.path.expanduser('~/ray_mbpo/PVTracking')
                matches = glob.glob(os.path.join(base, '**', 'checkpoint_*'), recursive=True)
            if not matches:
                raise FileNotFoundError('No checkpoint matched pattern: {}'.format(path))
            return max(matches, key=os.path.getmtime)

        # If path contains glob chars, expand
        if any(c in p for c in '*?[]'):
            matches = glob.glob(p)
            if not matches:
                raise FileNotFoundError('No checkpoint matched pattern: {}'.format(path))
            return max(matches, key=os.path.getmtime)

        # If exact path exists, use it
        if os.path.exists(p):
            return p

        # If parent exists, look for checkpoint_* in the parent directory
        parent = os.path.dirname(p)
        if os.path.isdir(parent):
            matches = glob.glob(os.path.join(parent, 'checkpoint_*'))
            if matches:
                return max(matches, key=os.path.getmtime)

        # Last resort: search the common experiment root for checkpoints
        base = os.path.expanduser('~/ray_mbpo/PVTracking')
        matches = glob.glob(os.path.join(base, '**', 'checkpoint_*'), recursive=True)
        if matches:
            return max(matches, key=os.path.getmtime)

        raise FileNotFoundError('Could not resolve checkpoint directory: {}'.format(path))

    ckpt_dir = resolve_ckpt_dir(args.ckpt_dir)
    outdir = os.path.expanduser(args.outdir)
    os.makedirs(outdir, exist_ok=True)

    py = sys.executable

    # 1) Run evaluation (delegates to existing script)
    eval_cmd = [py, 'scripts/evaluate_agent.py', ckpt_dir,
                '--outdir', outdir,
                '--num-rollouts', str(args.num_rollouts),
                '--max-path-length', str(args.max_path_length)]
    if args.deterministic:
        eval_cmd.append('--deterministic')
    run_command(eval_cmd)

    # 2) Generate training plots (optional, uses existing plotting script if trial dir is parent)
    # If ckpt_dir is like .../seed:<seed>_<ts>/checkpoint_<N>, TRIAL_DIR is its parent
    # The trial directory (where progress.csv and params.json live) is the
    # parent of the checkpoint directory: .../seed:.../checkpoint_<N> -> parent is seed:...
    trial_dir = os.path.dirname(ckpt_dir)
    if os.path.isdir(trial_dir):
        plots_out = os.path.join(outdir, 'training_plots')
        os.makedirs(plots_out, exist_ok=True)
        plot_cmd = [py, 'scripts/plot_training_progress.py', trial_dir,
                    '--outdir', plots_out]
        try:
            run_command(plot_cmd)
        except subprocess.CalledProcessError:
            print('plot_training_progress.py failed; continuing')

    # 3) Launch viskit pointing at top-level experiment directory
    if not args.no_server:
        viskit_path = os.path.expanduser('~/ray_mbpo/PVTracking')
        print('\nLaunching viskit server. Press Ctrl-C to stop.')
        try:
            run_command(['viskit', viskit_path, '--port', str(args.port)])
        except FileNotFoundError:
            print('viskit command not found. If you installed viskit with pip, ensure it is in PATH.')
            print('You can still view plots saved in', outdir)


if __name__ == '__main__':
    main()
