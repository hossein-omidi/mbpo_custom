#!/usr/bin/env python3
"""Generate (never execute) verified PVWatts commands. Defaults are NOT safe in original CLIs."""
import argparse
import shlex
from pathlib import Path

def quote(x):return shlex.quote(str(x))

def commands(seed, sac_episodes, sac_run, mbpo_run, eval_run, mbpo_checkpoint=None):
    if sac_episodes<1 or seed<0:raise ValueError('Invalid episode count/seed')
    mchk=mbpo_checkpoint or f'{mbpo_run}/latest.pt'
    mflags='--power-model pvwatts_sapm --wind-reference-height-m 2.0 --gamma-pdc -0.004 --sapm-mount open_rack_glass_polymer --wind-powerlaw-exponent 0.14'
    return {
        'sac_training':f'python -m SAC_workspace.workflows.03_train --run-dir {quote(sac_run)} --seed {seed} --epochs {sac_episodes} {mflags}',
        'mbpo_training':f'python -m MBPO_workspace.workflows.02_train --run-dir {quote(mbpo_run)} --seed {seed} --epochs 500 {mflags}',
        'evaluation':f'python -m MBPO_workspace.workflows.03_evaluate --sac-checkpoint {quote(sac_run+"/latest.pt")} --mbpo-checkpoint {quote(mchk)} --sac-training-csv {quote(sac_run+"/episodes.csv")} --mbpo-training-csv {quote(mbpo_run+"/episodes.csv")} --outdir {quote(eval_run)} --all-scenarios {mflags}',
    }
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=0);p.add_argument('--sac-episodes',type=int,default=500)
    p.add_argument('--sac-run',default='SAC_workspace/runs/sac_matched_seed0')
    p.add_argument('--mbpo-run',default='MBPO_workspace/runs/mbpo_matched_seed0')
    p.add_argument('--eval-run',default='MBPO_workspace/evaluation/matched_budget_seed0')
    p.add_argument('--mbpo-checkpoint',default=None)
    a=p.parse_args()
    print('# Commands below are printed ONLY, never executed. First run repo_preflight.py.\n')
    print('# Exactly 500 episodes x 117 = 58,500 real environment steps.\n')
    for k,v in commands(a.seed,a.sac_episodes,a.sac_run,a.mbpo_run,a.eval_run,a.mbpo_checkpoint).items():
        print(f'# {k}\n{v}\n')
    print('# WARNING: --all-scenarios does not guarantee held-out evaluation. Use the original precommitted disjoint evaluation protocol before scientific claims.')
