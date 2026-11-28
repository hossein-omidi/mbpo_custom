"""Automatically discover paired SAC/MBPO runs and generate publication figures.

python -m PV_experiments.report --name mc_5pct --kind learning
python -m PV_experiments.report --name mc_5pct --kind monte-carlo --stage test
"""
from __future__ import annotations
import argparse,csv,json,sys,math,statistics,hashlib
from collections import defaultdict
from pathlib import Path
from .core import (PROJECT_ROOT,load_experiment,selected_seeds,parse_seeds,
                   verify_pair,run_path,experiment_dir,execute,read_json,
                   require_files,atomic_json)
from .evaluate import evaluation_output


def validate_mc_source(cfg,stage,seed,root=PROJECT_ROOT):
    if stage not in ('validation','test'):
        raise ValueError('Final Monte Carlo reporting accepts only full validation or full test evaluations')
    directory=evaluation_output(cfg,stage,seed,root)
    summary=directory/'mc_summary.json'
    require_files([summary,directory/'mc_pooled_power_mean_sd_by_utc_step.csv',directory/'all_day_draw_method.csv'])
    data=read_json(summary)
    from .evaluate import existing_report_is_valid,_provenance,_require_matching_provenance
    draws=int(data['draws_per_day'])
    if not existing_report_is_valid(cfg,stage,seed,draws,directory,root):
        raise ValueError('Incomplete Monte Carlo source')
    _require_matching_provenance(directory,_provenance(cfg,stage,seed,root,draws=draws))
    if data.get('split_role')!=stage or data.get('partial_partition_evaluation') is not False:
        raise ValueError(f'Seed {seed}: full {stage} Monte Carlo evaluation required')
    if data.get('n_selected_days')!=data.get('n_available_days'):
        raise ValueError(f'Seed {seed}: incomplete held-out daily coverage')
    # Verify the actual Monte Carlo input grid before using summary numbers.
    # This catches accidental cross-seed weather mismatches, missing
    # trajectories and misleading cached summary files.
    split=read_json(run_path(cfg,'sac',seed,root)/'scenario_split.json')
    if data.get('scenario_protocol_sha256')!=split.get('protocol_sha256'):
        raise ValueError(f'Seed {seed}: MC summary does not belong to frozen scenario split')
    expected_days=set(split[f'{stage}_ids'])
    if len(expected_days)!=data.get('n_selected_days'):
        raise ValueError(f'Seed {seed}: MC scenario counts differ from frozen partition')
    hashes,baseline_returns=_validate_raw_mc_file(
        directory/'all_day_draw_method.csv',data,expected_days)
    from .scientific_checks import validate_mc_tables
    validate_mc_tables(directory,sorted(expected_days),draws,run_dirs={a:run_path(cfg,a,seed,root) for a in ('sac','mbpo')} if stage=='validation' else None,epoch=cfg['epochs'])
    data['_validated_weather_hashes']=hashes
    data['_validated_baselines']=baseline_returns
    return directory,data



def _validate_raw_mc_file(path:Path,summary:dict,days:set[str]):
    """Verify every historical-day x draw x controller record and net-energy means.

    No interpolation, no averaging by episode number, and no unpaired
    Monte Carlo records are accepted.
    """
    methods=('sac','mbpo','fixed_no_motion','sun_tracking')
    draws=int(summary['draws_per_day'])
    if draws<2:raise ValueError('Monte Carlo summary has fewer than two draws')
    expected={(day,rid) for day in days for rid in range(draws)}
    by_key={};weather_hash={};baselines={}
    with path.open(newline='',encoding='utf-8') as stream:
        reader=csv.DictReader(stream)
        necessary={'scenario_id','draw','method','net_kwh','profile_sha256'}
        if not necessary.issubset(reader.fieldnames or []):
            raise ValueError(f'{path}: missing Monte Carlo pairing columns {necessary-set(reader.fieldnames or [])}')
        for row in reader:
            sid=str(row['scenario_id']);rid=int(row['draw']);method=row['method']
            key=(sid,rid,method)
            if (sid,rid) not in expected or method not in methods or key in by_key:
                raise ValueError(f'{path}: unexpected/duplicate Monte Carlo evaluation {key}')
            net=float(row['net_kwh'])
            if not math.isfinite(net):raise ValueError(f'{path}: nonfinite net energy {key}')
            weather=str(row['profile_sha256'])
            if not weather:raise ValueError(f'{path}: empty weather fingerprint {key}')
            old=weather_hash.setdefault((sid,rid),weather)
            if old!=weather:
                raise ValueError(f'{path}: SAC/MBPO/baseline weather realization mismatch {key}')
            by_key[key]=net
            if method in ('fixed_no_motion','sun_tracking'):
                baselines[key]=net
    if len(by_key)!=len(expected)*len(methods):
        raise ValueError(f'{path}: missing Monte Carlo day/draw/controller combinations')
    if len(weather_hash)!=len(expected):
        raise ValueError(f'{path}: incomplete historical-day/draw weather grid')
    for method in methods:
        mean=statistics.fmean(by_key[(sid,rid,method)] for sid,rid in sorted(expected))
        reported=float(summary['methods'][method]['mean_net_kwh'])
        if not math.isclose(mean,reported,rel_tol=1e-8,abs_tol=1e-8):
            raise ValueError(f'{path}: {method} MC summary mean disagrees with raw transitions')
    for label in ('mbpo_minus_sac','mbpo_minus_fixed_no_motion',
                  'sac_minus_fixed_no_motion','mbpo_minus_sun_tracking',
                  'sac_minus_sun_tracking'):
        left,right=label.split('_minus_')
        mean=statistics.fmean(by_key[(sid,rid,left)]-by_key[(sid,rid,right)]
                              for sid,rid in sorted(expected))
        reported=float(summary['paired_net_energy_differences'][label]['mean_net_kwh'])
        if not math.isclose(mean,reported,rel_tol=1e-8,abs_tol=1e-8):
            raise ValueError(f'{path}: paired {label} MC summary disagrees with raw data')
    return weather_hash,baselines


def file_sha256(path:Path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def frozen_learning_input_hashes(cfg,seeds,root=PROJECT_ROOT):
    result={}
    for seed in seeds:
        for alg in ('sac','mbpo'):
            directory=run_path(cfg,alg,seed,root)
            for rel in ('run_metadata.json','scenario_split.json','episodes.csv',
                        'periodic_evaluation/evaluations.csv',
                        'periodic_evaluation/evaluation_days.csv',
                        'periodic_evaluation/evaluation_references.csv'):
                path=directory/rel
                require_files([path])
                result[f'{alg}_seed{seed}/{rel}']=file_sha256(path)
    return result


def frozen_mc_input_hashes(inputs):
    result={}
    for seed,directory,data in inputs:
        for filename in ('manager_provenance.json','mc_summary.json','all_day_draw_method.csv',
                         'mc_pooled_power_mean_sd_by_utc_step.csv'):
            result[f'seed{seed}/{filename}']=file_sha256(directory/filename)
    return result


def summary_statistics(values):
    if len(values)<2:raise ValueError('At least two independent seeds required for descriptive SD')
    if not all(math.isfinite(x) for x in values):raise ValueError('Nonfinite experiment results')
    return float(statistics.fmean(values)),float(statistics.stdev(values))


def report_mc(cfg,seeds,stage,root=PROJECT_ROOT,dry_run=False):
    for seed in seeds:verify_pair(cfg,seed,root)
    inputs=[(seed,*validate_mc_source(cfg,stage,seed,root)) for seed in seeds]
    comparable=('scenario_protocol_sha256','uncertainty_spec_sha256','n_selected_days','draws_per_day')
    first=inputs[0][2]
    for seed,_,data in inputs[1:]:
        for field in comparable:
            if data.get(field)!=first.get(field):
                raise ValueError(f'Cross-seed MC evaluation differs in {field}: seed {seed}')
        if data['_validated_weather_hashes']!=first['_validated_weather_hashes']:
            raise ValueError(f'Seed {seed}: common random numbers differ across training seeds')
        for key,base in first['_validated_baselines'].items():
            if not math.isclose(data['_validated_baselines'][key],base,
                                rel_tol=1e-8,abs_tol=1e-8):
                raise ValueError(f'Seed {seed}: deterministic baseline energy differs for {key}')
    from .scientific_checks import source_hashes
    inputs_hashes=frozen_mc_input_hashes(inputs)
    inputs_hashes.update(source_hashes(root,['PV_experiments/report.py','PV_experiments/scientific_checks.py','MBPO_workspace/src/curve_statistics.py']))
    out=experiment_dir(cfg,root)/'reports'/f'monte_carlo_{stage}'
    if out.exists() and any(out.iterdir()) and not dry_run:
        meta=out/'report_manifest.json'
        if meta.is_file():
            old=read_json(meta)
            if (old.get('seeds')==seeds and old.get('stage')==stage
                and old.get('source_protocol_sha256')==inputs[0][2].get('scenario_protocol_sha256')
                and old.get('source_file_sha256')==inputs_hashes):
                from .scientific_checks import check_report_inventory
                check_report_inventory(out,old)
                print('SKIP existing across-seed Monte Carlo summary:',out)
                return
        raise FileExistsError(f'Existing report directory: {out}. Do not overwrite published figures implicitly.')
    if dry_run:
        print('PLAN: Monte Carlo across-seed report',stage,'seeds=',seeds,'->',out)
        return
    out.mkdir(parents=True,exist_ok=True)
    metric_rows=[]
    for alg in ('sac','mbpo','fixed_no_motion','sun_tracking'):
        vals=[float(data['methods'][alg]['mean_net_kwh']) for _,_,data in inputs]
        mu,sd=summary_statistics(vals)
        metric_rows.append({'method':alg,'n_training_seeds':len(seeds),
            'mean_over_training_seed_means_net_kwh':mu,
            'sd_over_training_seed_means_net_kwh':sd,
            'historical_days':inputs[0][2]['n_selected_days'],
            'weather_draws_per_day':inputs[0][2]['draws_per_day']})
    differences=[]
    for key in ('mbpo_minus_sac','mbpo_minus_fixed_no_motion','sac_minus_fixed_no_motion',
                'mbpo_minus_sun_tracking','sac_minus_sun_tracking'):
        vals=[float(data['paired_net_energy_differences'][key]['mean_net_kwh']) for _,_,data in inputs]
        mu,sd=summary_statistics(vals)
        differences.append({'paired_method_difference':key,'n_training_seeds':len(seeds),
            'mean_difference_kwh':mu,'sd_across_seed_mean_differences_kwh':sd})
    def write(name,rows):
        with (out/name).open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write('net_energy_across_seeds.csv',metric_rows)
    write('paired_energy_differences_across_seeds.csv',differences)
    groups=defaultdict(list)
    for seed,directory,_ in inputs:
        with (directory/'mc_pooled_power_mean_sd_by_utc_step.csv').open(newline='',encoding='utf-8') as f:
            rows=list(csv.DictReader(f))
        for row in rows:
            method=row['method'];step=int(row['step']);key=(method,step)
            groups[key].append((seed,float(row['mean_power_w'])))
    for key,values in groups.items():
        seen={seed for seed,_ in values}
        if seen!=set(seeds):raise ValueError(f'Incomplete per-step MC power profiles for {key}')
    power_rows=[]
    for (method,step),values in sorted(groups.items()):
        mu,sd=summary_statistics([x for _,x in values])
        power_rows.append({'method':method,'step':step,
            'post_action_utc_unwrapped_hour':8.75+step/12,
            'n_training_seeds':len(seeds),'mean_of_seed_mean_power_w':mu,
            'sd_across_seed_mean_power_w':sd,
            'note':'Weather/day mixture averaged within seed before independent training-seed SD'})
    for method in ('sac','mbpo'):
        if sum(r['method']==method for r in power_rows)!=195:
            raise ValueError(f'Expected 195 pooled power steps for {method}')
    write('pooled_power_mean_sd_across_seeds.csv',power_rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9,5.5))
    for method in ('sac','mbpo'):
        block=sorted((x for x in power_rows if x['method']==method),key=lambda x:x['step'])
        x=[v['post_action_utc_unwrapped_hour'] for v in block]
        m=[v['mean_of_seed_mean_power_w'] for v in block]
        sd=[v['sd_across_seed_mean_power_w'] for v in block]
        line,=ax.plot(x,m,label=method.upper()+' mean',linewidth=1.9)
        import numpy as np
        ax.fill_between(x,np.array(m)-np.array(sd),np.array(m)+np.array(sd),
                        color=line.get_color(),alpha=.18,label=method.upper()+' ±1 seed SD')
    ax.set(xlabel='Post-action UTC hour (unwrapped)',ylabel='PV DC power (W)',
           title=f'{stage.title()}: power mean ± 1 SD across independent training seeds')
    ax.grid(alpha=.24);ax.legend(fontsize=8);fig.tight_layout()
    fig.savefig(out/'power_mean_pm_training_seed_sd.png',dpi=180);plt.close(fig)
    manifest={'schema':1,'stage':stage,'seeds':seeds,
        'source_protocol_sha256':inputs[0][2]['scenario_protocol_sha256'],
        'source_file_sha256':inputs_hashes,
        'uncertainty_spec_sha256':inputs[0][2]['uncertainty_spec_sha256'],
        'interpretation':'Mean and sample SD across independent training-seed means; each mean pools the same held-out historical days and paired Monte Carlo weather draws. Descriptive ±1 SD, not an inferential confidence interval.',
        'outputs':['net_energy_across_seeds.csv','paired_energy_differences_across_seeds.csv',
                   'pooled_power_mean_sd_across_seeds.csv','power_mean_pm_training_seed_sd.png']}
    from .scientific_checks import report_inventory
    manifest['output_sha256']=report_inventory(out)
    atomic_json(out/'report_manifest.json',manifest)
    print('Across-seed Monte Carlo report:',out)


def report_learning(cfg,seeds,root=PROJECT_ROOT,dry_run=False):
    if len(seeds)<2:raise ValueError('Multi-seed sample complexity needs at least 2 seeds')
    for seed in seeds:verify_pair(cfg,seed,root)
    from .scientific_checks import source_hashes
    input_hashes=frozen_learning_input_hashes(cfg,seeds,root)
    input_hashes.update(source_hashes(root,['MBPO_workspace/src/sample_complexity.py','MBPO_workspace/src/curve_statistics.py','MBPO_workspace/workflows/05_sample_complexity.py','PV_experiments/scientific_checks.py','PV_experiments/report.py']))
    for seed in seeds:
        for alg in ('sac','mbpo'):
            ledger=run_path(cfg,alg,seed,root)/'recovery_events.json'
            if ledger.exists():input_hashes[f'{alg}_seed{seed}/recovery_events.json']=file_sha256(ledger)
    out=experiment_dir(cfg,root)/'reports'/'sample_complexity'
    if out.exists() and any(out.iterdir()) and not dry_run:
        manifest=out/'report_manifest.json'
        if manifest.is_file() and (out/'report.json').is_file():
            m=read_json(manifest)
            if (m.get('seeds')==seeds and m.get('experiment')==cfg['name']
                and m.get('source_file_sha256')==input_hashes and m.get('estimator_semantics')=='confirmed_v2'):
                from .scientific_checks import check_report_inventory
                check_report_inventory(out,m)
                print('SKIP existing sample-complexity report:',out)
                return
            raise ValueError('An existing learning report uses a different seed set; do not overwrite it')
        raise FileExistsError(f'Incomplete or unmanaged sample-complexity report exists: {out}')
    cmd=[sys.executable,'-m','MBPO_workspace.workflows.05_sample_complexity',
         '--sac-runs']+[str(run_path(cfg,'sac',s,root)) for s in seeds]+[
         '--mbpo-runs']+[str(run_path(cfg,'mbpo',s,root)) for s in seeds]+['--outdir',str(out)]
    execute(cfg,'sample_complexity',cmd,root=root,dry_run=dry_run)
    if not dry_run:
        require_files([out/'report.json',out/'learning_curve_descriptive_sd.csv',
                       out/'learning_curve_mean_pm_seed_sd.png'])
        from .scientific_checks import report_inventory
        atomic_json(out/'report_manifest.json',{'schema':1,'output_sha256':report_inventory(out),'experiment':cfg['name'],
              'seeds':seeds,'source_file_sha256':input_hashes,'estimator_semantics':'confirmed_v2',
              'protocol_sha256':verify_pair(cfg,seeds[0],root)['protocol_sha256'],
              'source':'MBPO_workspace.workflows.05_sample_complexity',
              'interpretation':'Bootstrap resampling bands and separate descriptive ±1 SD across independently trained seed means and historical day means'})
        print('Sample-complexity mean ± SD figures:',out)


def run(args,root=PROJECT_ROOT):
    cfg=load_experiment(args.name,root)
    seeds=selected_seeds(cfg,parse_seeds(args.seeds))
    if args.stage=='test' and args.kind in ('monte-carlo','all') and not args.confirm_test:raise ValueError('Test reporting requires --confirm-test')
    if args.kind in ('learning','all'):report_learning(cfg,seeds,root,dry_run=args.dry_run)
    if args.kind in ('monte-carlo','all'):report_mc(cfg,seeds,args.stage,root,dry_run=args.dry_run)


def build_parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--name',required=True)
    p.add_argument('--kind',choices=('learning','monte-carlo','all'),default='learning')
    p.add_argument('--stage',choices=('validation','test'),default='validation')
    p.add_argument('--confirm-test',action='store_true')
    p.add_argument('--seeds',default=None,help='Default: all frozen training seeds')
    p.add_argument('--dry-run',action='store_true')
    return p


def main(argv=None):run(build_parser().parse_args(argv))

if __name__=='__main__':main()
