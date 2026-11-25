"""Unified scientific representation for the current managed SAC/MBPO experiment.

All existing learning, physical simulator, and evaluation code stays unchanged.
Run from repository root: python -m PV_experiments.represent --name mc_final --kind all
"""
from __future__ import annotations
import argparse
from pathlib import Path
from .representation.common import new_output,collection_session,write_json,atomic_json,read_json,sha,verify_runs


def run(args,root=None):
    from PV_experiments.core import PROJECT_ROOT,load_experiment,experiment_dir,selected_seeds,parse_seeds,validate_name,run_path
    root=PROJECT_ROOT if root is None else Path(root)
    if args.name is None:
        if args.kind!='dataset':raise ValueError('--name is required except for standalone --kind dataset')
        if args.seeds is not None:raise ValueError('Standalone dataset reporting has no training seeds')
        validate_name(args.tag);target=root/'outputs'/'paper_experiments'/'dataset_catalog'/args.tag
        if args.dry_run:print('Standalone dataset output:',target);return
        from MBPO_workspace.src.config import load_config
        from SAC_workspace.src.training.experiment_protocol import build_scenario_protocol
        from .representation.dataset import read_dataset,render_dataset
        phys=load_config(); split=build_scenario_protocol(phys.sac.manifest,phys.sac.protocol)
        frame,profiles,fingerprint=read_dataset(phys,split)
        with new_output(target) as out:
            render_dataset(frame,profiles,out,fingerprint)
            write_json(out/'representation_manifest.json',dict(schema=1,kind='dataset',scenario_protocol_sha256=split['protocol_sha256'],sources=fingerprint,note='Standalone catalogue description, not attached to trained checkpoint'))
        print('Published:',target);return
    cfg=load_experiment(args.name,root); seeds=selected_seeds(cfg,parse_seeds(args.seeds));base=experiment_dir(cfg,root)
    validate_name(args.tag);validate_name(args.trajectory_tag)
    if args.stage=='test' and args.kind in ('all','evaluation','collect','trajectories') and not args.confirm_test:raise ValueError('Use --confirm-test for final held-out reporting; do not tune against it')
    split=verify_runs(cfg,seeds,root)
    target=base/'representation'/args.stage/f'{args.kind}_{args.tag}'
    rich=base/'representation_trajectories'/args.stage/f'{args.scope}_{args.trajectory_tag}'
    if args.kind=='collect':target=rich
    kinds=('dataset','training','learning','evaluation','model') if args.kind=='all' else (args.kind,)
    if args.kind=='all' and args.with_trajectories:kinds+=('trajectories',)
    if args.with_trajectories and args.kind!='all':raise ValueError('--with-trajectories is only used with --kind all')
    if args.dry_run:
        print('Experiment:',cfg['name'],'seeds:',seeds,'stage:',args.stage,'kinds:',kinds);print('Output:',target)
        if 'collect' in kinds or 'trajectories' in kinds:print('Rich path:',rich,'scope:',args.scope)
        print('No output written. Actual run validates all sources and provenance.');return
    if args.kind=='collect':
        from .representation.offline import load_mc
        from .representation.trajectories import collect
        data=load_mc(cfg,seeds,args.stage,split,root)
        with collection_session(target) as out:
            collect(cfg,split,data,args.stage,args.scope,root,out)
            atomic_json(out/'representation_manifest.json',dict(schema=1,experiment=cfg['name'],seeds=seeds,stage=args.stage,scope=args.scope,kind='collect',trajectory_manifest_sha256=sha(out/'trajectory_manifest.json'),sources=data['sources']))
        print('Collection ready:',target);return
    # Figures are published only after all required inputs and computations succeed.
    with new_output(target) as out:
        sources={}; data=None; fingerprint=None
        if set(kinds)&{'evaluation','collect','trajectories'}:
            from .representation.offline import load_mc
            data=load_mc(cfg,seeds,args.stage,split,root);sources['physical_evaluation']=data['sources']
        for kind in kinds:
            d=out if kind=='collect' else out/kind;d.mkdir(parents=True,exist_ok=True)
            if kind=='dataset':
                from .representation.dataset import read_dataset,render_dataset
                from .representation.trajectories import load_physics_config
                frame,profiles,fingerprint=read_dataset(load_physics_config(cfg,root),split)
                for alg in ('sac','mbpo'):
                    for seed in seeds:
                        resolved=read_json(run_path(cfg,alg,seed,root)/'resolved_mdp.json')
                        if any(resolved.get(k)!=v for k,v in fingerprint.items()):raise ValueError('Current dataset bytes differ from training checkpoint provenance')
                render_dataset(frame,profiles,d,fingerprint);sources['dataset']=fingerprint
            elif kind=='training':
                from .representation.training import render_training
                sources['training']=render_training(cfg,seeds,root,d)
            elif kind=='learning':
                from .representation.training import render_learning
                sources['learning']=render_learning(cfg,seeds,root,d)
            elif kind=='evaluation':
                from .representation.offline import render_evaluation
                render_evaluation(data,d,args.stage)
            elif kind=='model':
                from .representation.training import render_model
                sources['model']=render_model(cfg,seeds,root,d,args.realization_id)
            elif kind=='trajectories':
                from .representation.trajectories import render_rich
                render_rich(rich,data,args.stage,args.scope,d);sources['rich_manifest']=sha(rich/'trajectory_manifest.json')
            else:raise ValueError(kind)
        # Record all figure semantics (including existing learning figures via report.json)
        modules=Path(__file__).parent/'representation';code={p.name:sha(p) for p in sorted(modules.glob('*.py'))};code['represent.py']=sha(Path(__file__));code['scientific_checks.py']=sha(Path(__file__).with_name('scientific_checks.py'));code['sample_complexity.py']=sha(root/'MBPO_workspace/src/sample_complexity.py');code['curve_statistics.py']=sha(root/'MBPO_workspace/src/curve_statistics.py')
        from .scientific_checks import report_inventory
        write_json(out/'representation_manifest.json',dict(schema=1,output_sha256=report_inventory(out),experiment=cfg['name'],seeds=seeds,stage=args.stage,kinds=list(kinds),scope=args.scope if set(kinds)&{'collect','trajectories'} else None,scenario_protocol_sha256=split['protocol_sha256'],uncertainty_sha256=cfg['uncertainty_sha256'],sources=sources,representation_code_sha256=code,note='Read-only postprocessing. No annual extrapolation, no synthetic model rollouts in physical policy evaluation. Dataset and model-validation scopes are independently labeled.'))
    print('Published:',target)


def parser():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--name',help='Managed experiment name; optional for standalone dataset plots');p.add_argument('--kind',choices=('all','dataset','training','learning','evaluation','model','collect','trajectories'),default='all');p.add_argument('--stage',choices=('validation','test'),default='validation');p.add_argument('--seeds',help='Subset of frozen seed set, e.g. 0:4');p.add_argument('--tag',default='v1',help='New output tag; never overwrites an existing report');p.add_argument('--trajectory-tag',default='v1',help='Tag for collected raw trajectory directory');p.add_argument('--scope',choices=('representative','full'),default='representative');p.add_argument('--with-trajectories',action='store_true',help='Include separately collected trajectory suite in all');p.add_argument('--realization-id',type=int,default=0,help='Existing learned-model validation realization');p.add_argument('--confirm-test',action='store_true');p.add_argument('--dry-run',action='store_true');return p


def main(argv=None):
    args=parser().parse_args(argv)
    if args.realization_id<0:raise ValueError('Negative realization ID')
    run(args)

if __name__=='__main__':main()
