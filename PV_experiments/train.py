"""Train all planned SAC/MBPO seeds with frozen uncertainty settings.

Example: python -m PV_experiments.train --name mc_5pct --seeds 0:4 --epochs 2000
Subsequently: python -m PV_experiments.train --name mc_5pct
"""
from __future__ import annotations
from mbpo.geometry10 import MODE, STATE_DIM
import argparse,sys
from pathlib import Path
from .core import (PROJECT_ROOT,base_path,initialize_experiment,load_experiment,
                   parse_seeds,selected_seeds,run_path,inspect_training_run,
                   uncertainty_path,experiment_dir,execute,read_json,
                   command_override,require_files,atomic_json)


def train_plan(args,root=PROJECT_ROOT,dry_run=False):
    specified=parse_seeds(args.seeds)
    cfg=initialize_experiment(args.name,seeds=specified,epochs=args.epochs,
        uncertainty_file=args.uncertainty_spec,device=args.device,
        model_epochs=args.model_epochs,model_max_train_samples=args.model_max_train_samples,
        root=root,dry_run=dry_run)
    selected=selected_seeds(cfg,specified)
    return cfg,selected


def run(args,root=PROJECT_ROOT):
    cfg,seeds=train_plan(args,root,dry_run=args.dry_run)
    print(f"Experiment: {cfg['name']} | seeds={seeds} | epochs={cfg['epochs']} | "
          f"uncertainty SHA256={cfg['uncertainty_sha256'][:16]}…",flush=True)
    frozen=uncertainty_path(cfg,root)
    if not args.dry_run:
        audit=experiment_dir(cfg,root)/'preflight'/'environment_audit.json'
        verified=False
        if audit.exists():
            try:
                data=read_json(audit)
                verified=(data.get('status')=='passed' and data.get('fixed_draw_replay') is True and data.get('observation_mode')==MODE and data.get('state_dim')==STATE_DIM)
            except (ValueError,TypeError):verified=False
        if args.skip_verification and not verified:
            raise RuntimeError('--skip-verification requires a successful verification file generated for this exact managed experiment')
        if not verified and not args.skip_verification:
            command=[sys.executable,'-m','PostProcessing_MC_patch.tools.verify_mc_environment',
                     '--uncertainty-spec',str(frozen),'--out',str(audit)]
            execute(cfg,'physical_preflight',command,root=root)
        else:print('Preflight already passed; reusing verified environment audit.',flush=True)
    for seed in seeds:
        for algorithm in ('sac','mbpo'):
            state,_=inspect_training_run(cfg,algorithm,seed,root)
            if state=='completed':
                print(f'SKIP completed: {algorithm} seed {seed}',flush=True)
                continue
            directory=run_path(cfg,algorithm,seed,root)
            cmd=[sys.executable,'-m','MBPO_workspace.workflows.08_monte_carlo_train',
                 '--algorithm',algorithm,'--uncertainty-spec',str(frozen),
                 '--run-dir',str(directory),'--seed',str(seed),'--epochs',str(cfg['epochs']),
                 '--device',cfg['device']]
            if algorithm=='mbpo':cmd+=command_override(cfg)
            if state=='incomplete':
                if not args.resume:
                    raise RuntimeError(f'{algorithm} seed {seed} has incomplete files at {directory}. '
                                       'Use --resume with matching latest.pt or inspect the failed run; no files were deleted.')
                ck=directory/'latest.pt';require_files([ck]);cmd+=['--resume',str(ck)]
            execute(cfg,f'train_{algorithm}',cmd,seed=seed,root=root,dry_run=args.dry_run)
            if not args.dry_run:
                after,_=inspect_training_run(cfg,algorithm,seed,root)
                if after!='completed':raise RuntimeError(f'{algorithm} seed {seed}: workflow returned success without completed run metadata')
    print('Training plan complete. Next: python -m PV_experiments.evaluate --name '
          +cfg['name']+' --mode smoke',flush=True)


def build_parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--name',required=True,help='Immutable experiment ID')
    p.add_argument('--seeds',default=None,help='0:4 inclusive, or 0,2,4; default on first run 0:4')
    p.add_argument('--epochs',type=int,default=None,help='Default on first run: 2000; frozen thereafter')
    p.add_argument('--uncertainty-spec',default=None,help='Default on first run: prior 5%% hypothetical spec')
    p.add_argument('--device',default=None,help='Default on first run: cpu (or cuda)')
    p.add_argument('--model-epochs',type=int,default=None)
    p.add_argument('--model-max-train-samples',type=int,default=None)
    p.add_argument('--resume',action='store_true',help='Resume incomplete runs from matching latest.pt')
    p.add_argument('--dry-run',action='store_true',help='Show training commands only; do not create files')
    p.add_argument('--skip-verification',action='store_true',help='Only use if the separate physical verifier was already run')
    return p


def main(argv=None):
    args=build_parser().parse_args(argv)
    run(args)

if __name__=='__main__':main()
