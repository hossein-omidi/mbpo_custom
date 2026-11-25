"""Managed publication reports: censor-aware learning + calendar heatmaps.
Additive companion to PV_experiments.represent, no changes to its core code.
"""
from __future__ import annotations
import argparse,os,tempfile,shutil
from pathlib import Path
from contextlib import contextmanager
from .advanced_learning import load_learning,render_learning,sha_file,write_json
from .calendar_heatmaps import load_calendar,render_calendar

@contextmanager
def output_directory(target):
    target=Path(target);target.parent.mkdir(parents=True,exist_ok=True)
    lock=target.parent/f'.{target.name}.lock';fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.close(fd);tmp=None
    try:
        if target.exists():raise FileExistsError(f'{target} already exists; use a new --tag')
        tmp=Path(tempfile.mkdtemp(prefix='.advanced-report-',dir=target.parent));yield tmp
        if target.exists():raise FileExistsError(target)
        tmp.rename(target)
    finally:
        if tmp is not None and tmp.exists():shutil.rmtree(tmp)
        lock.unlink(missing_ok=True)


def run(args,root=None):
    from PV_experiments.core import PROJECT_ROOT,load_experiment,selected_seeds,parse_seeds,validate_name,experiment_dir
    root=PROJECT_ROOT if root is None else Path(root);cfg=load_experiment(args.name,root);seeds=selected_seeds(cfg,parse_seeds(args.seeds));validate_name(args.tag)
    if args.kind=='learning' and args.stage=='test':raise ValueError('Learning efficiency uses validation only; omit --stage test')
    if args.stage=='test' and not args.confirm_test:raise ValueError('Use --confirm-test for final held-out reporting')
    target=experiment_dir(cfg,root)/'representation'/args.stage/f'advanced_{args.kind}_{args.tag}'
    if args.dry_run:
        print('PLAN:',args.kind,'seeds:',seeds,'stage:',args.stage,'output:',target);return
    with output_directory(target) as out:
        sources={};protocol=None;learning=None
        if args.kind in ('learning','all'):
            d=load_learning(cfg,seeds,root);protocol=d['protocol'];learning=render_learning(d,out/'learning');sources['learning']=d['sources']
        if args.kind in ('calendar','all'):
            c=load_calendar(cfg,seeds,args.stage,root)
            if protocol is not None and protocol!=c['protocol']:raise ValueError('Learning and calendar protocols differ')
            protocol=c['protocol'];render_calendar(c,out/'calendar');sources['calendar']=c['sources']
        # Detect mutated sources during rendering; avoid publishing mismatched outputs.
        if 'learning' in sources:
            from PV_experiments.core import run_path
            for k,v in sources['learning'].items():
                folder,rel=k.split('/',1);a,seed=folder.split('_seed');p=run_path(cfg,a,int(seed),root)/rel
                if sha_file(p)!=v:raise ValueError(f'Source changed while reporting: {p}')
        if 'calendar' in sources:
            from PV_experiments.core import run_path
            from PV_experiments.evaluate import evaluation_output,_provenance,_require_matching_provenance
            for k,v in sources['calendar'].items():
                folder,rel=k.split('/',1)
                if folder.startswith('seed'):p=evaluation_output(cfg,args.stage,int(folder[4:]),root)/rel
                else:
                    a,seed=folder.split('_seed');p=run_path(cfg,a,int(seed),root)/rel
                if sha_file(p)!=v:raise ValueError(f'Source changed while reporting: {p}')
            for seed in seeds:_require_matching_provenance(evaluation_output(cfg,args.stage,seed,root),_provenance(cfg,args.stage,seed,root,draws=c['draws']))
        code={name:sha_file(Path(__file__).with_name(name)) for name in ('advanced_report.py','advanced_learning.py','calendar_heatmaps.py')}
        write_json(out/'report_manifest.json',dict(schema=1,experiment=cfg['name'],stage=args.stage,kind=args.kind,seeds=seeds,protocol_sha256=protocol,uncertainty_sha256=cfg['uncertainty_sha256'],sources=sources,code_sha256=code,note='Learning uses frozen validation only. Calendar uses indicated full held-out partition. No core changes.'))
    if learning is not None:
        print('Confirmed targets:',learning['confirmed_count'],'of',learning['n_paired_seeds'],'seeds')
        print('Conditional sample reduction:',learning['conditional_sample_reduction_percent'])
        print('All-seed restricted-budget saving (%):',learning['restricted_budget_relative_saving_percent'])
    print('Published:',target)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--name',required=True);p.add_argument('--kind',choices=('learning','calendar','all'),default='all');p.add_argument('--stage',choices=('validation','test'),default='validation');p.add_argument('--seeds');p.add_argument('--tag',default='v2');p.add_argument('--confirm-test',action='store_true');p.add_argument('--dry-run',action='store_true');run(p.parse_args(argv))

if __name__=='__main__':main()
