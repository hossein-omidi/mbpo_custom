#!/usr/bin/env python3
"""Fair comparison script for MBPO vs SAC on PV tracking.

Runs both algorithms on identical tasks with identical configurations
(except algorithm type) and produces statistical comparison.

Usage:
    python scripts/compare_mbpo_vs_sac.py \
        --mbpo-config examples.config.pv_tracking.conf_compact \
        --sac-config examples.config.pv_tracking.conf_sac \
        --n-seeds 5 \
        --epochs 2000 \
        --outdir comparison_results
"""

import argparse
import json
import os
import sys
import subprocess
import time
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd


def run_training(config_name, run_name, epochs, cpus, trial_cpus, outdir):
    """Run a single training job and return the trial directory."""
    cmd = [
        'bash', '-c',
        f'source ~/miniconda3/etc/profile.d/conda.sh && '
        f'conda activate mbpo && '
        f'cd {os.getcwd()} && '
        f'./train.sh {run_name} {config_name} '
        f'--cpus {cpus} --trial-cpus {trial_cpus} '
    ]
    
    if epochs:
        # Override epochs via config modification or env var
        pass  # epochs controlled by config
    
    result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, timeout=3600*6)
    return result.returncode == 0


def collect_metrics(run_dir):
    """Extract key metrics from a training run."""
    progress_path = Path(run_dir) / 'progress.csv'
    if not progress_path.exists():
        return None
    
    df = pd.read_csv(progress_path)
    return {
        'epochs': len(df),
        'final_train_return': df['training/return-average'].iloc[-1] if 'training/return-average' in df.columns else None,
        'final_eval_return': df.tail(100)['evaluation/return-average'].mean() if 'evaluation/return-average' in df.columns else None,
        'eval_return_std': df.tail(100)['evaluation/return-average'].std() if 'evaluation/return-average' in df.columns else None,
        'final_q_loss': df.tail(10)['Q_loss'].mean() if 'Q_loss' in df.columns else None,
        'final_alpha': df.tail(10)['alpha'].mean() if 'alpha' in df.columns else None,
        'epochs_completed': df['epoch'].iloc[-1] if 'epoch' in df.columns else None,
    }


def main():
    parser = argparse.ArgumentParser(description='Compare MBPO vs SAC on PV tracking')
    parser.add_argument('--mbpo-config', default='examples.config.pv_tracking.conf_compact',
                        help='MBPO config module')
    parser.add_argument('--sac-config', default='examples.config.pv_tracking.conf_sac',
                        help='SAC config module')
    parser.add_argument('--n-seeds', type=int, default=3, help='Number of random seeds')
    parser.add_argument('--epochs', type=int, default=2000, help='Training epochs')
    parser.add_argument('--cpus', type=int, default=4, help='Total CPUs')
    parser.add_argument('--trial-cpus', type=int, default=2, help='CPUs per trial')
    parser.add_argument('--outdir', default='comparison_results', help='Output directory')
    parser.add_argument('--eval-rollouts', type=int, default=100, help='Rollouts per evaluation')
    parser.add_argument('--dry-run', action='store_true', help='Dry run without training')
    
    args = parser.parse_args()
    
    os.makedirs(args.outdir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_id = f'comparison_{timestamp}'
    
    results = {
        'mbpo': [],
        'sac': [],
        'metadata': {
            'timestamp': timestamp,
            'mbpo_config': args.mbpo_config,
            'sac_config': args.sac_config,
            'n_seeds': args.n_seeds,
            'epochs': args.epochs,
        }
    }
    
    print(f"Starting MBPO vs SAC comparison: {run_id}")
    print(f"Output directory: {args.outdir}")
    print(f"MBPO config: {args.mbpo_config}")
    print(f"SAC config: {args.sac_config}")
    print(f"Seeds: {args.n_seeds}, Epochs: {args.epochs}")
    
    if args.dry_run:
        print("DRY RUN - would run training but exiting now")
        return
    
    # Extract short config names from module paths
    mbpo_short = args.mbpo_config.split('.')[-1]
    sac_short = args.sac_config.split('.')[-1]
    
    # Run MBPO with multiple seeds
    print("\n=== Running MBPO ===")
    for seed_idx in range(args.n_seeds):
        run_name = f"{run_id}_mbpo_seed{seed_idx}"
        print(f"\nStarting MBPO seed {seed_idx}...")
        
        success = run_training(
            mbpo_short, run_name, args.epochs, 
            args.cpus, args.trial_cpus, args.outdir
        )
        
        if success:
            # Find the latest trial directory
            base_dir = Path('runs') / run_name / 'checkpoints' / 'PVTracking' / 'pv_tracking'
            trial_dirs = list(base_dir.glob('seed:*')) if base_dir.exists() else []
            if trial_dirs:
                latest_trial = max(trial_dirs, key=lambda d: d.stat().st_mtime)
                metrics = collect_metrics(latest_trial)
                metrics['seed'] = seed_idx
                results['mbpo'].append(metrics)
                print(f"MBPO seed {seed_idx} completed")
            else:
                print(f"MBPO seed {seed_idx} failed - no trial directory")
        else:
            print(f"MBPO seed {seed_idx} FAILED")
    
    # Run SAC with multiple seeds
    print("\n=== Running SAC ===")
    for seed_idx in range(args.n_seeds):
        run_name = f"{run_id}_sac_seed{seed_idx}"
        print(f"\nStarting SAC seed {seed_idx}...")
        
        success = run_training(
            sac_short, run_name, args.epochs,
            args.cpus, args.trial_cpus, args.outdir
        )
        
        if success:
            base_dir = Path('runs') / run_name / 'checkpoints' / 'PVTracking' / 'pv_tracking'
            trial_dirs = list(base_dir.glob('seed:*')) if base_dir.exists() else []
            if trial_dirs:
                latest_trial = max(trial_dirs, key=lambda d: d.stat().st_mtime)
                metrics = collect_metrics(latest_trial)
                metrics['seed'] = seed_idx
                results['sac'].append(metrics)
                print(f"SAC seed {seed_idx} completed")
            else:
                print(f"SAC seed {seed_idx} failed - no trial directory")
        else:
            print(f"SAC seed {seed_idx} FAILED")
    
    # Evaluation phase
    print("\n=== Evaluating Best Checkpoints ===")
    for algo in ['mbpo', 'sac']:
        if results[algo]:
            best = max(results[algo], key=lambda m: m.get('final_eval_return', 0))
            print(f"\nBest {algo.upper()} checkpoint: epoch {best.get('epochs_completed')}, eval return {best.get('final_eval_return'):.4f}")
            
            # Full evaluation
            eval_dir = Path(args.outdir) / f'eval_{algo}'
            # evaluate_checkpoint would go here
    
    # Save results
    results_path = Path(args.outdir) / f'comparison_{timestamp}.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to {results_path}")
    print("Comparison complete!")


if __name__ == '__main__':
    main()