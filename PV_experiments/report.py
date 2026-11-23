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
    if data.get('split_role')!=stage or data.get('partial_partition_evaluation') is not False:
        raise ValueError(f'Seed {seed}: full {stage} Monte Carlo evaluation required')
    if data.get('n_selected_days')!=data.get('n_available_days'):
        raise ValueError(f'Seed {seed}: incomplete held-out daily coverage')
    return directory,data


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
            for rel in ('run_metadata.json','scenario_split.json',
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
        for filename in ('mc_summary.json','all_day_draw_method.csv',
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
    for seed,_,data in inputs[1:]:
        for field in comparable:
            if data.get(field)!=inputs[0][2].get(field):
                raise ValueError(f'Cross-seed MC evaluation differs in {field}: seed {seed}')
    inputs_hashes=frozen_mc_input_hashes(inputs)
    out=experiment_dir(cfg,root)/'reports'/f'monte_carlo_{stage}'
    if out.exists() and any(out.iterdir()) and not dry_run:
        meta=out/'report_manifest.json'
        if meta.is_file():
            old=read_json(meta)
            if (old.get('seeds')==seeds and old.get('stage')==stage
                and old.get('source_protocol_sha256')==inputs[0][2].get('scenario_protocol_sha256')
                and old.get('source_file_sha256')==inputs_hashes):
                print('SKIP existing across-seed Monte Carlo summary:',out)
                return
        raise FileExistsError(f'Existing report directory: {out}. Do not overwrite published figures implicitly.')
    if dry_run:
        print('PLAN: Monte Carlo across-seed report',stage,'seeds=',seeds,'->',out)
        return
    out.mkdir(parents=True,exist_ok=True)
    metric_rows=[]
    for alg in ('sac','mbpo','fixed_no_motion','sun_slew_limited'):
        vals=[float(data['methods'][alg]['mean_net_kwh']) for _,_,data in inputs]
        mu,sd=summary_statistics(vals)
        metric_rows.append({'method':alg,'n_training_seeds':len(seeds),
            'mean_over_training_seed_means_net_kwh':mu,
            'sd_over_training_seed_means_net_kwh':sd,
            'historical_days':inputs[0][2]['n_selected_days'],
            'weather_draws_per_day':inputs[0][2]['draws_per_day']})
    differences=[]
    for key in ('mbpo_minus_sac','mbpo_minus_fixed_no_motion','sac_minus_fixed_no_motion',
                'mbpo_minus_sun_slew_limited','sac_minus_sun_slew_limited'):
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
    atomic_json(out/'report_manifest.json',manifest)
    print('Across-seed Monte Carlo report:',out)


def report_learning(cfg,seeds,root=PROJECT_ROOT,dry_run=False):
    if len(seeds)<2:raise ValueError('Multi-seed sample complexity needs at least 2 seeds')
    for seed in seeds:verify_pair(cfg,seed,root)
    input_hashes=frozen_learning_input_hashes(cfg,seeds,root)
    out=experiment_dir(cfg,root)/'reports'/'sample_complexity'
    if out.exists() and any(out.iterdir()) and not dry_run:
        manifest=out/'report_manifest.json'
        if manifest.is_file() and (out/'report.json').is_file():
            m=read_json(manifest)
            if (m.get('seeds')==seeds and m.get('experiment')==cfg['name']
                and m.get('source_file_sha256')==input_hashes):
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
        atomic_json(out/'report_manifest.json',{'schema':1,'experiment':cfg['name'],
              'seeds':seeds,'source_file_sha256':input_hashes,
              'protocol_sha256':verify_pair(cfg,seeds[0],root)['protocol_sha256'],
              'source':'MBPO_workspace.workflows.05_sample_complexity',
              'interpretation':'Bootstrap resampling bands and separate descriptive ±1 SD across independently trained seed means and historical day means'})
        print('Sample-complexity mean ± SD figures:',out)


def run(args,root=PROJECT_ROOT):
    cfg=load_experiment(args.name,root)
    seeds=selected_seeds(cfg,parse_seeds(args.seeds))
    if args.kind in ('learning','all'):report_learning(cfg,seeds,root,dry_run=args.dry_run)
    if args.kind in ('monte-carlo','all'):report_mc(cfg,seeds,args.stage,root,dry_run=args.dry_run)


def build_parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--name',required=True)
    p.add_argument('--kind',choices=('learning','monte-carlo','all'),default='learning')
    p.add_argument('--stage',choices=('validation','test'),default='validation')
    p.add_argument('--seeds',default=None,help='Default: all frozen training seeds')
    p.add_argument('--dry-run',action='store_true')
    return p


def main(argv=None):run(build_parser().parse_args(argv))

if __name__=='__main__':main()
