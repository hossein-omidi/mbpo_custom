"""Check automatically discovered run, checkpoint and report status.

python -m PV_experiments.status --name mc_5pct
"""
from __future__ import annotations
import argparse,json
from .core import (load_experiment,PROJECT_ROOT,inspect_training_run,experiment_dir)


def run(name,root=PROJECT_ROOT):
    cfg=load_experiment(name,root)
    print(f"Experiment: {name}, seeds={cfg['seeds']}, epochs={cfg['epochs']}")
    print(f"Frozen assumed weather input SHA256: {cfg['uncertainty_sha256']}")
    for seed in cfg['seeds']:
        sac,_=inspect_training_run(cfg,'sac',seed,root)
        mbpo,_=inspect_training_run(cfg,'mbpo',seed,root)
        print(f'  seed {seed}: SAC={sac} | MBPO={mbpo}')
        d=experiment_dir(cfg,root)/'evaluations'
        for mode in ('smoke','model','validation','test'):
            target=(d/mode/f'seed{seed}')
            if mode=='model':
                matching=list(target.glob('realization*/model_validation_summary.json')) if target.exists() else []
                state=f'{len(matching)} finished realizations'
            else:
                state='complete' if (target/'mc_summary.json').is_file() else 'not complete'
            print(f'    {mode}: {state}')
    output=experiment_dir(cfg,root)/'reports'
    print('  learning report:', 'complete' if (output/'sample_complexity'/'report_manifest.json').is_file() else 'not complete')
    print('  validation MC report:', 'complete' if (output/'monte_carlo_validation'/'report_manifest.json').is_file() else 'not complete')
    print('  test MC report:', 'complete' if (output/'monte_carlo_test'/'report_manifest.json').is_file() else 'not complete')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--name',required=True)
    a=p.parse_args(argv)
    run(a.name)

if __name__=='__main__':main()
