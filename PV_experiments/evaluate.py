"""Discover SAC/MBPO checkpoints and evaluate without typing any run paths.

Examples:
 python -m PV_experiments.evaluate --name mc_5pct --mode smoke
 python -m PV_experiments.evaluate --name mc_5pct --mode model
 python -m PV_experiments.evaluate --name mc_5pct --mode validation
 python -m PV_experiments.evaluate --name mc_5pct --mode test --confirm-test
"""
from __future__ import annotations
from mbpo.audit_support import runtime_identity, dataset_identity
import argparse,sys,hashlib
from pathlib import Path
from .core import (PROJECT_ROOT,load_experiment,selected_seeds,parse_seeds,verify_pair,
                   run_path,experiment_dir,uncertainty_path,execute,command_override,
                   read_json,require_files,atomic_json)



def _hash_file(path):
    """Hash exact saved model weights, not just an output filename."""
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def _provenance(cfg,mode,seed,root,*,draws=None,realization_id=None):
    sac=run_path(cfg,'sac',seed,root)
    mbpo=run_path(cfg,'mbpo',seed,root)
    split=read_json(sac/'scenario_split.json')
    dataset=dataset_identity(root)
    if cfg.get('dataset_identity') is not None and dataset!=cfg['dataset_identity']:
        raise ValueError('Current dataset differs from frozen experiment')
    for folder in (sac,mbpo):
        resolved=read_json(folder/'resolved_mdp.json')
        if any(resolved.get(k)!=v for k,v in dataset.items()):
            raise ValueError('Current dataset differs from checkpoint physical identity')
    result={
      'schema':2,'experiment':cfg['name'],'seed':int(seed),'mode':mode,
      'runtime':runtime_identity(root,'model' if mode=='model' else 'evaluation'), 'dataset':dataset, 'final_epoch':int(cfg['epochs']),
      'scenario_protocol_sha256':split['protocol_sha256'],
      'uncertainty_spec_file_sha256':_hash_file(uncertainty_path(cfg,root)),
      'mbpo_checkpoint_sha256':_hash_file(mbpo/'latest.pt'),
      'model_epochs':cfg.get('model_epochs'),
      'model_max_train_samples':cfg.get('model_max_train_samples'),
      'device':cfg['device'],
    }
    from .scientific_checks import source_hashes
    result['statistical_sources']=source_hashes(root,['PV_experiments/scientific_checks.py']+([] if mode=='model' else ['MBPO_workspace/src/curve_statistics.py']))
    if mode=='model':
        result['realization_id']=int(realization_id)
    else:
        result['sac_checkpoint_sha256']=_hash_file(sac/'latest.pt')
        result['draws_per_day']=int(draws)
        result['split_role']='test' if mode=='test' else 'validation'
        result['max_days']=2 if mode=='smoke' else None
    return result


def _require_matching_provenance(out,expected):
    from mbpo.audit_release import require_evaluation_seal
    require_evaluation_seal(out,expected)


def _verify_model_validation_summary(path,cfg,seed,realization_id,protocol_sha,root):
    d=read_json(path)
    checkpoint=str(run_path(cfg,'mbpo',seed,root)/'latest.pt')
    if d.get('validation_schema')!=2: raise ValueError('Model diagnostic schema differs')
    if (d.get('seed')!=seed or d.get('validation_realization_id')!=int(realization_id)
        or d.get('scenario_protocol_sha256')!=protocol_sha
        or Path(d.get('checkpoint','')).resolve()!=Path(checkpoint).resolve()):
        raise ValueError(f'{path}: model-validation summary checkpoint/seed/partition/realization mismatch')
    from mbpo.env.trajectory_uncertainty import normalized_spec
    if d.get('weather_uncertainty')!=normalized_spec(read_json(uncertainty_path(cfg,root))):
        raise ValueError(f'{path}: model-validation weather assumptions differ from frozen experiment')
    from .scientific_checks import validate_model_tables,verify_model_accounting
    verify_model_accounting(d,run_path(cfg,'mbpo',seed,root))
    days=read_json(run_path(cfg,'mbpo',seed,root)/'scenario_split.json')['validation_ids']
    predictions,_=validate_model_tables(Path(path).parent,d,days)
    if predictions is not None:
        from .scientific_checks import verify_model_physical_parity
        verify_model_physical_parity(predictions,run_path(cfg,'mbpo',seed,root),cfg['epochs'],realization_id)
    for required in ('external_on_policy_one_step','counterfactual_action_test'):
        if required not in d:raise ValueError(f'{path}: incomplete {required} model diagnostic')


def evaluation_output(cfg,mode,seed,root=PROJECT_ROOT):
    return experiment_dir(cfg,root)/'evaluations'/mode/f'seed{seed}'


def existing_report_is_valid(cfg,mode,seed,draws,out,root=PROJECT_ROOT):
    report=out/'mc_summary.json'
    if not report.exists():return False
    info=read_json(report)
    role='validation' if mode in ('smoke','validation') else 'test'
    split=read_json(run_path(cfg,'sac',seed,root)/'scenario_split.json')
    expected_days=len(split[f'{role}_ids'])
    if (info.get('scenario_protocol_sha256')!=split.get('protocol_sha256')
          or info.get('n_available_days')!=expected_days):
        raise ValueError(f'Existing evaluation partition differs from frozen scenario manifest: {report}')
    if info.get('split_role')!=role or info.get('draws_per_day')!=draws:
        raise ValueError(f'Existing results have different role/draw count: {report}. Use a different experiment name.')
    if info.get('uncertainty_spec_sha256')!=__import__('mbpo.env.trajectory_uncertainty',fromlist=['spec_fingerprint']).spec_fingerprint(read_json(uncertainty_path(cfg,root))):
        raise ValueError(f'Existing results have different uncertainty fingerprint: {report}')
    if mode=='smoke':
        if info.get('n_selected_days')!=2 or info.get('partial_partition_evaluation') is not True:
            raise ValueError('Existing smoke results do not match required two-day partial evaluation')
    elif (info.get('partial_partition_evaluation') is not False
          or info.get('n_selected_days')!=expected_days):
        raise ValueError('Existing full evaluation appears partial or truncated')
    for p in ('all_day_draw_method.csv','daily_power_mean_std_by_utc_step.csv',
              'mc_pooled_power_mean_sd_by_utc_step.csv'):
        require_files([out/p])
    return True


def evaluate_one(cfg,mode,seed,draws,root=PROJECT_ROOT,dry_run=False):
    split=verify_pair(cfg,seed,root)
    out=evaluation_output(cfg,mode,seed,root)
    identity=None if dry_run else _provenance(cfg,mode,seed,root,draws=draws)
    if out.exists() and not dry_run:
        if (out/'mc_summary.json').exists():
            if existing_report_is_valid(cfg,mode,seed,draws,out,root):
                _require_matching_provenance(out,identity)
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
    if not dry_run:
        cmd+=['--expected-seed',str(seed),'--expected-epoch',str(cfg['epochs']),'--expected-sac-sha',identity['sac_checkpoint_sha256'],'--expected-mbpo-sha',identity['mbpo_checkpoint_sha256']]
    if mode=='test':cmd+=['--confirm-test']
    if mode=='smoke':cmd+=['--max-days','2']
    cmd+=command_override(cfg)
    execute(cfg,f'evaluate_{mode}',cmd,root=root,seed=seed,dry_run=dry_run)
    if not dry_run:
        if not existing_report_is_valid(cfg,mode,seed,draws,out,root):
            raise RuntimeError(f'No complete evaluation report generated: {out}')
        from mbpo.audit_release import seal_evaluation, verify_loaded_policies
        verify_loaded_policies(read_json(out/'mc_summary.json'),identity)
        if _provenance(cfg,mode,seed,root,draws=draws)!=identity: raise ValueError('Evaluation inputs changed during execution')
        from .scientific_checks import validate_mc_tables
        days=list(split['test_ids' if mode=='test' else 'validation_ids'])
        if mode=='smoke':days=days[:2]
        validate_mc_tables(out,days,draws,run_dirs={'sac':sac,'mbpo':mbpo} if mode!='test' else None,epoch=cfg['epochs'])
        seal_evaluation(out,identity)


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
            identity=None if args.dry_run else _provenance(
                cfg,mode,seed,root,realization_id=args.realization_id)
            if out.exists() and any(out.iterdir()) and not args.dry_run:
                result=out/'model_validation_summary.json'
                if result.exists():
                    _require_matching_provenance(out,identity)
                    _verify_model_validation_summary(
                        result,cfg,seed,args.realization_id,
                        identity['scenario_protocol_sha256'],root)
                    print(f'SKIP verified model validation seed {seed}')
                    continue
                raise RuntimeError(f'Model-validation output exists but completion not verified: {out}')
            cmd=[sys.executable,'-m','MBPO_workspace.workflows.06_model_validation',
                 '--run-dir',str(run_path(cfg,'mbpo',seed,root)),
                 '--checkpoint',str(run_path(cfg,'mbpo',seed,root)/'latest.pt'),
                 '--uncertainty-spec',str(uncertainty_path(cfg,root)),
                 '--realization-id',str(args.realization_id),'--outdir',str(out),
                 '--device',cfg['device']]
            if not args.dry_run: cmd+=['--expected-seed',str(seed),'--expected-epoch',str(cfg['epochs']),'--expected-sha',identity['mbpo_checkpoint_sha256']]
            cmd+=command_override(cfg)
            execute(cfg,'model_validation',cmd,root=root,seed=seed,dry_run=args.dry_run)
            if not args.dry_run:
                _verify_model_validation_summary(
                    out/'model_validation_summary.json',cfg,seed,args.realization_id,
                    identity['scenario_protocol_sha256'],root)
                from mbpo.audit_release import seal_evaluation, verify_loaded_policies
                verify_loaded_policies(read_json(out/'model_validation_summary.json'),identity)
                if _provenance(cfg,mode,seed,root,realization_id=args.realization_id)!=identity: raise ValueError('Model evaluation inputs changed during execution')
                from .scientific_checks import seal_model
                seal_model(out,identity)
        else:
            if mode=='test' and not args.dry_run:
                # Require a completed physical smoke run before consuming test data.
                smoke_dir=evaluation_output(cfg,'smoke',seed,root)
                smoke=smoke_dir/'mc_summary.json'
                if not smoke.is_file():
                    raise ValueError(f'Seed {seed}: run --mode smoke before final --mode test')
                previous_draws=int(read_json(smoke).get('draws_per_day',-1))
                if not existing_report_is_valid(cfg,'smoke',seed,previous_draws,smoke_dir,root):
                    raise ValueError(f'Seed {seed}: prior smoke evaluation did not pass verification')
                _require_matching_provenance(
                    smoke_dir,_provenance(cfg,'smoke',seed,root,draws=previous_draws))
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
