"""Discover SAC/MBPO checkpoints and evaluate without typing any run paths.

Examples:
 python -m PV_experiments.evaluate --name mc_5pct --mode smoke
 python -m PV_experiments.evaluate --name mc_5pct --mode model
 python -m PV_experiments.evaluate --name mc_5pct --mode validation
 python -m PV_experiments.evaluate --name mc_5pct --mode test --confirm-test
"""
from __future__ import annotations
import argparse,sys
from pathlib import Path
from .core import (PROJECT_ROOT,load_experiment,selected_seeds,parse_seeds,verify_pair,
                   run_path,experiment_dir,uncertainty_path,execute,command_override,
                   read_json,require_files)


def evaluation_output(cfg,mode,seed,root=PROJECT_ROOT):
    return experiment_dir(cfg,root)/'evaluations'/mode/f'seed{seed}'


def existing_report_is_valid(cfg,mode,seed,draws,out,root=PROJECT_ROOT):
    report=out/'mc_summary.json'
    if not report.exists():return False
    info=read_json(report)
    role='validation' if mode in ('smoke','validation') else 'test'
    if info.get('split_role')!=role or info.get('draws_per_day')!=draws:
        raise ValueError(f'Existing results have different role/draw count: {report}. Use a different experiment name.')
    if info.get('uncertainty_spec_sha256')!=__import__('mbpo.env.trajectory_uncertainty',fromlist=['spec_fingerprint']).spec_fingerprint(read_json(uncertainty_path(cfg,root))):
        raise ValueError(f'Existing results have different uncertainty fingerprint: {report}')
    if mode=='smoke':
        if info.get('n_selected_days')!=2 or info.get('partial_partition_evaluation') is not True:
            raise ValueError('Existing smoke results do not match required two-day partial evaluation')
    elif info.get('partial_partition_evaluation') is not False:
        raise ValueError('Existing full evaluation appears partial or truncated')
    for p in ('all_day_draw_method.csv','daily_power_mean_std_by_utc_step.csv'):
        require_files([out/p])
    return True


def evaluate_one(cfg,mode,seed,draws,root=PROJECT_ROOT,dry_run=False):
    split=verify_pair(cfg,seed,root)
    out=evaluation_output(cfg,mode,seed,root)
    if out.exists() and not dry_run:
        if (out/'mc_summary.json').exists():
            if existing_report_is_valid(cfg,mode,seed,draws,out,root):
                print(f'SKIP previously completed: {mode} seed {seed}')
                return
        elif any(out.iterdir()):
            raise RuntimeError(f'An incomplete evaluation exists in {out}; inspect before rerunning; no results are overwritten.')
    sac=run_path(cfg,'sac',seed,root);mbpo=run_path(cfg,'mbpo',seed,root)
    cmd=[sys.executable,'-m','MBPO_workspace.workflows.09_monte_carlo_evaluate',
         '--sac-checkpoint',str(sac/'latest.pt'),'--mbpo-checkpoint',str(mbpo/'latest.pt'),
         '--scenario-split-json',str(sac/'scenario_split.json'),
         '--split-role','validation' if mode!='test' else 'test',
         '--draws',str(draws),'--uncertainty-spec',str(uncertainty_path(cfg,root)),
         '--outdir',str(out),'--device',cfg['device']]
    if mode=='smoke':cmd+=['--max-days','2']
    cmd+=command_override(cfg)
    execute(cfg,f'evaluate_{mode}',cmd,root=root,seed=seed,dry_run=dry_run)
    if not dry_run and not existing_report_is_valid(cfg,mode,seed,draws,out,root):
        raise RuntimeError(f'No complete evaluation report generated: {out}')


def run(args,root=PROJECT_ROOT):
    cfg=load_experiment(args.name,root)
    seeds=selected_seeds(cfg,parse_seeds(args.seeds))
    mode=args.mode
    if mode=='test' and not args.confirm_test:
        raise ValueError('Holdout test evaluation requires --confirm-test to prevent accidental use while tuning')
    draws=(3 if mode=='smoke' else 100) if args.draws is None else int(args.draws)
    if mode!='model' and draws<2:raise ValueError('At least two draws are required for Monte Carlo SD')
    if mode=='smoke' and args.draws is not None and args.draws<2:raise ValueError('Smoke test needs >=2 draws')
    for seed in seeds:
        verify_pair(cfg,seed,root)
        if mode=='model':
            if args.realization_id<0:raise ValueError('realization ID must be nonnegative')
            out=evaluation_output(cfg,mode,seed,root)/f'realization{args.realization_id}'
            if out.exists() and any(out.iterdir()) and not args.dry_run:
                if (out/'model_validation_summary.json').exists():
                    print(f'SKIP completed model validation seed {seed}')
                    continue
                raise RuntimeError(f'Model-validation output exists but completion not verified: {out}')
            cmd=[sys.executable,'-m','MBPO_workspace.workflows.06_model_validation',
                 '--run-dir',str(run_path(cfg,'mbpo',seed,root)),
                 '--checkpoint',str(run_path(cfg,'mbpo',seed,root)/'latest.pt'),
                 '--uncertainty-spec',str(uncertainty_path(cfg,root)),
                 '--realization-id',str(args.realization_id),'--outdir',str(out),
                 '--device',cfg['device']]
            cmd+=command_override(cfg)
            execute(cfg,'model_validation',cmd,root=root,seed=seed,dry_run=args.dry_run)
        else:
            if mode=='test' and not args.dry_run:
                # Require a completed physical smoke run before consuming test data.
                smoke=evaluation_output(cfg,'smoke',seed,root)/'mc_summary.json'
                if not smoke.is_file():
                    raise ValueError(f'Seed {seed}: run --mode smoke before final --mode test')
            evaluate_one(cfg,mode,seed,draws,root,dry_run=args.dry_run)
    print('Evaluation plan complete.',flush=True)


def build_parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--name',required=True)
    p.add_argument('--mode',choices=('smoke','model','validation','test'),default='smoke')
    p.add_argument('--seeds',default=None,help='Default: all seeds frozen in the experiment manifest')
    p.add_argument('--draws',type=int,default=None,help='Default 3 for smoke and 100 for validation/test')
    p.add_argument('--realization-id',type=int,default=0,help='Model-validation fixed weather draw ID')
    p.add_argument('--confirm-test',action='store_true',help='Required to expose frozen test split')
    p.add_argument('--dry-run',action='store_true')
    return p


def main(argv=None):run(build_parser().parse_args(argv))

if __name__=='__main__':main()
