#!/usr/bin/env python3
"""In-place, pinned-source reporting update for Geometry10. No output mutation."""
import argparse
import difflib
from pathlib import Path
import subprocess
import sys

BASE='e34371f7527eaf210186f20f2c917996a436cfd5'
HERE=Path(__file__).resolve().parent

def replace(s,a,b):
    if s.count(a)!=1:raise ValueError(f'Expected exactly one source anchor: {a[:160]!r}; got {s.count(a)}')
    return s.replace(a,b,1)

def evaluation(s):
    s=replace(s,"    if mode=='model':\n        result['realization_id']", "    from .scientific_checks import source_hashes\n    result['statistical_sources']=source_hashes(root,['PV_experiments/scientific_checks.py']+([] if mode=='model' else ['MBPO_workspace/src/curve_statistics.py']))\n    if mode=='model':\n        result['realization_id']")
    s=replace(s,"    for required in ('external_on_policy_one_step','counterfactual_action_test'):","    from .scientific_checks import validate_model_tables,verify_model_accounting\n    verify_model_accounting(d,run_path(cfg,'mbpo',seed,root))\n    days=read_json(run_path(cfg,'mbpo',seed,root)/'scenario_split.json')['validation_ids']\n    predictions,_=validate_model_tables(Path(path).parent,d,days)\n    if predictions is not None:\n        from .scientific_checks import verify_model_physical_parity\n        verify_model_physical_parity(predictions,run_path(cfg,'mbpo',seed,root),cfg['epochs'],realization_id)\n    for required in ('external_on_policy_one_step','counterfactual_action_test'):")
    s=replace(s,"    if mode=='smoke':cmd+=['--max-days','2']", "    if mode=='test':cmd+=['--confirm-test']\n    if mode=='smoke':cmd+=['--max-days','2']")
    s=replace(s,"        seal_evaluation(out,identity)\n\n\ndef run", "        from .scientific_checks import validate_mc_tables\n        days=list(split['test_ids' if mode=='test' else 'validation_ids'])\n        if mode=='smoke':days=days[:2]\n        validate_mc_tables(out,days,draws,run_dirs={'sac':sac,'mbpo':mbpo} if mode!='test' else None,epoch=cfg['epochs'])\n        seal_evaluation(out,identity)\n\n\ndef run")
    s=replace(s,"                seal_evaluation(out,identity)\n        else:","                from .scientific_checks import seal_model\n                seal_model(out,identity)\n        else:")
    return s

def mc(s):
    s=replace(s,"    p.add_argument('--split-role',choices=('test','validation'),default='test')", "    p.add_argument('--split-role',choices=('test','validation'),default='validation')\n    p.add_argument('--confirm-test',action='store_true')")
    s=replace(s,"    if args.draws<2:p.error", "    if args.split_role=='test' and not args.confirm_test:p.error('Test partition requires --confirm-test')\n    if args.draws<2:p.error")
    s=replace(s,"    agents={\n        'sac':load_policy","    from PV_experiments.scientific_checks import verify_completed_run\n    for checkpoint in (args.sac_checkpoint,args.mbpo_checkpoint):\n        folder=Path(checkpoint).resolve().parent\n        verify_completed_run(folder,json.loads((folder/'run_metadata.json').read_text()))\n    agents={\n        'sac':load_policy")
    s=replace(s,"ax.boxplot(data,labels=METHODS,showfliers=False)","ax.boxplot(data,labels=METHODS,showfliers=True)")
    return s

def model(s):
    s=replace(s,"def _prediction_rows(actual, predicted, uncertainty, metadata, quantiles=None):", "def _prediction_rows(actual, predicted, uncertainty, metadata, quantiles=None, predictive_sd=None):")
    s=replace(s,"            if quantiles is not None:\n                for k,label", "            if predictive_sd is not None:\n                row[f'predictive_sd_{name}']=float(predictive_sd[i,j])\n            if quantiles is not None:\n                for k,label")
    s=replace(s,'    from mbpo.audit_release import load_verified_checkpoint, equivalent_pose_metrics',"    from PV_experiments.scientific_checks import verify_completed_run\n    verify_completed_run(run_dir,json.loads((run_dir/'run_metadata.json').read_text()))\n    from mbpo.audit_release import load_verified_checkpoint, equivalent_pose_metrics")
    s=replace(s,'    if not ensemble.fitted:\n        raise ValueError("Checkpoint contains an unfitted dynamics ensemble")\n','')
    s=replace(s,"    try:\n        # Internal held-out transitions",'''    try:
        if not ensemble.fitted:
            accounting=_load_training_stats(run_dir / "episodes.csv")
            if accounting.get('synthetic_samples_used')!=0:
                raise ValueError('Unfitted ensemble has nonzero or missing synthetic exposure')
            summary=dict(validation_schema=2,model_status='unfitted',states_per_scenario=int(args.states_per_scenario),
                checkpoint=str(checkpoint_path),loaded_policies={'mbpo':{'sha256':loaded_sha,'seed':seed,'epoch':int(ckpt['epoch'])}},
                seed=seed,scenario_protocol_sha256=split['protocol_sha256'],validation_scenario_ids=scenario_ids,
                validation_realization_id=int(args.realization_id),weather_uncertainty=uncertainty_spec,
                internal_whole_day_holdout=None,external_on_policy_one_step=None,counterfactual_action_test=None,
                training_synthetic_data_accounting=accounting,interpretation=['No fitted dynamics; no predictive diagnostics. Physical policy results remain valid.'])
            outdir.mkdir(parents=True,exist_ok=True)
            (outdir/'model_validation_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False),encoding='utf-8')
            return summary
        # Internal held-out transitions''')
    s=replace(s,'np.asarray(rewards, dtype=np.float32))','np.asarray(rewards, dtype=np.float64))')
    s=replace(s,'holdout_score, internal_meta, holdout_quantiles)','holdout_score, internal_meta, holdout_quantiles, holdout_total)')
    s=replace(s,'for m in external_meta], quantiles)','for m in external_meta], quantiles, total_std)')
    s=replace(s,'        _csv(outdir / "external_on_policy_predictions.csv", external_rows)',"        for i,row in enumerate(external_rows):\n            row.update(actual_next_power_norm=float(next_states[i,2]),actual_reward_kwh=float(rewards[i]),predicted_reward_kwh=float(pred_rewards[i]))\n        _csv(outdir / \"external_on_policy_predictions.csv\", external_rows)")
    s=replace(s,'            "validation_schema": 2,','            "validation_schema": 2,\n            "model_status":"fitted",\n            "states_per_scenario":int(args.states_per_scenario),')
    s=replace(s,'    print("External on-policy power RMSE:",','    if summary["model_status"]=="unfitted":\n        print("Model diagnostics unavailable: unfitted ensemble; physical policy retained.")\n        return\n    print("External on-policy power RMSE:",')
    return s

def report(s):
    s=replace(s,"    data['_validated_weather_hashes']=hashes", "    from .scientific_checks import validate_mc_tables\n    validate_mc_tables(directory,sorted(expected_days),draws,run_dirs={a:run_path(cfg,a,seed,root) for a in ('sac','mbpo')} if stage=='validation' else None,epoch=cfg['epochs'])\n    data['_validated_weather_hashes']=hashes")
    s=replace(s,"                print('SKIP existing across-seed Monte Carlo summary:',out)","                from .scientific_checks import check_report_inventory\n                check_report_inventory(out,old)\n                print('SKIP existing across-seed Monte Carlo summary:',out)")
    s=replace(s,"    atomic_json(out/'report_manifest.json',manifest)","    from .scientific_checks import report_inventory\n    manifest['output_sha256']=report_inventory(out)\n    atomic_json(out/'report_manifest.json',manifest)")
    s=replace(s,"                print('SKIP existing sample-complexity report:',out)","                from .scientific_checks import check_report_inventory\n                check_report_inventory(out,m)\n                print('SKIP existing sample-complexity report:',out)")
    s=replace(s,"        atomic_json(out/'report_manifest.json',{'schema':1,'experiment':cfg['name'],","        from .scientific_checks import report_inventory\n        atomic_json(out/'report_manifest.json',{'schema':1,'output_sha256':report_inventory(out),'experiment':cfg['name'],")
    s=replace(s,"    input_hashes=frozen_learning_input_hashes(cfg,seeds,root)","    from .scientific_checks import source_hashes\n    input_hashes=frozen_learning_input_hashes(cfg,seeds,root)\n    input_hashes.update(source_hashes(root,['MBPO_workspace/src/sample_complexity.py','MBPO_workspace/src/curve_statistics.py','MBPO_workspace/workflows/05_sample_complexity.py','PV_experiments/scientific_checks.py','PV_experiments/report.py']))\n    for seed in seeds:\n        for alg in ('sac','mbpo'):\n            ledger=run_path(cfg,alg,seed,root)/'recovery_events.json'\n            if ledger.exists():input_hashes[f'{alg}_seed{seed}/recovery_events.json']=file_sha256(ledger)")
    s=replace(s,"    inputs_hashes=frozen_mc_input_hashes(inputs)","    from .scientific_checks import source_hashes\n    inputs_hashes=frozen_mc_input_hashes(inputs)\n    inputs_hashes.update(source_hashes(root,['PV_experiments/report.py','PV_experiments/scientific_checks.py','MBPO_workspace/src/curve_statistics.py']))")
    s=replace(s,"    if args.kind in ('learning','all'):report_learning", "    if args.stage=='test' and args.kind in ('monte-carlo','all') and not args.confirm_test:raise ValueError('Test reporting requires --confirm-test')\n    if args.kind in ('learning','all'):report_learning")
    s=replace(s,"    p.add_argument('--seeds',default=None,help='Default: all frozen training seeds')", "    p.add_argument('--confirm-test',action='store_true')\n    p.add_argument('--seeds',default=None,help='Default: all frozen training seeds')")
    return s

def sample(s):
    s=replace(s,'    eval_root = root / "periodic_evaluation"','    from PV_experiments.scientific_checks import verify_completed_run\n    verify_completed_run(root,metadata)\n    eval_root = root / "periodic_evaluation"')
    s=replace(s,'    train_seconds = total("episode_seconds")','    times=[_number(r["episode_seconds"],field="episode_seconds") for r in rows]\n    if any(t<0 for t in times):raise ValueError("Negative training time")\n    train_seconds=float(sum(times))')
    s=replace(s,'    return result\n\n\ndef analyze_sample_complexity','    from PV_experiments.scientific_checks import recovery_accounting\n    result.update(recovery_accounting(run["root"],final_steps,int(run["metadata"]["horizon"])))\n    return result\n\n\ndef analyze_sample_complexity')
    for a in ('SAC','MBPO','Paired'):
        s=replace(s,f'{a} 95% two-way bootstrap band',f'{a} central 95% resampling band (sensitivity)')
    s=replace(s,'    ax.set(xlabel="Training seed", ylabel="Real transitions to target",','''        missed=[r for r in per_seed_rows if r['algorithm']==algorithm and not r['reached_target']]
        if missed:
            ax.scatter([r['seed'] for r in missed],[r['last_evaluated_env_steps'] for r in missed],marker=marker,facecolors='none',edgecolors='k',s=65,label=algorithm.upper()+' not confirmed (budget)')
    ax.set(xlabel="Training seed", ylabel="Confirmation steps or censored budget",''')
    s=replace(s,'        mean_time.append(float(np.mean(vals)) if vals else 0.0)','        if len(vals)!=sum(r["algorithm"]==algorithm for r in compute_rows):raise ValueError("Missing compute timing")\n        mean_time.append(float(np.mean(vals)))')
    s=replace(s,'    ax.bar(algos, mean_time)',"    paired={a:{r['seed']:r['measured_training_plus_evaluation_seconds'] for r in compute_rows if r['algorithm']==a} for a in algos}\n    if set(paired['sac'])!=set(paired['mbpo']):raise ValueError('Unpaired timing seeds')\n    for seed in sorted(paired['sac']):ax.plot([0,1],[paired[a][seed] for a in algos],'-o',alpha=.5,label=f'Seed {seed}')\n    sd=[np.std(list(paired[a].values()),ddof=1) for a in algos] if len(paired['sac'])>1 else None\n    ax.errorbar([0,1],mean_time,yerr=sd,fmt='D',color='k',capsize=3,label='Mean ± seed SD' if sd is not None else 'Mean')\n    ax.set_xticks([0,1],['SAC','MBPO']);ax.legend(fontsize=8)")
    return s

def advanced(s):
    s=replace(s,"    reference=None\n", "    reference=None\n    recovery={}\n")
    s=replace(s,"            # Include episode data, omitted in the earlier manager learning hash list.","            from .scientific_checks import recovery_accounting\n            recovery[a,seed]=recovery_accounting(p,int(m['final_env_steps']),int(m['horizon']))\n            if (p/'recovery_events.json').exists():sources[f'{a}_seed{seed}/recovery_events.json']=sha_file(p/'recovery_events.json')\n            # Include episode data, omitted in the earlier manager learning hash list.")
    s=replace(s,"horizon=int(meta['horizon']),replay=replay)","horizon=int(meta['horizon']),replay=replay,recovery=recovery)")
    s=replace(s,"            rows.append(row);lookup[a,s]=row", "            row.update(d.get('recovery',{}).get((a,s),{}))\n            rows.append(row);lookup[a,s]=row")
    s=replace(s,"    write_json(out/'sample_efficiency_summary.json',rep)","    rep['recovery_accounting']=[dict(algorithm=a,seed=s,**v) for (a,s),v in d.get('recovery',{}).items()]\n    rep['definitions']['recovery']='Learning axis counts committed transitions; known executed lower bound includes discarded complete episode rows. Unlogged partial work and discarded timing are not reconstructible.'\n    write_json(out/'sample_efficiency_summary.json',rep)")
    return s

def training(s):
    s=replace(s,"            frames[alg,seed]=f; sources","            u=finite(f.updates); changed=np.diff(np.r_[0,u])>0\n            if np.any(np.diff(np.r_[0,u])<0):raise ValueError('Decreasing optimizer update counter')\n            finite(f.alpha)\n            if changed.any():finite(f.loc[changed,['critic_loss','actor_loss']])\n            frames[alg,seed]=f; sources")
    s=replace(s,"figs.save(fig,'T01_optimizer_diagnostics'","fig.suptitle('25-episode trailing means' + (' ± sample seed SD' if len(seeds)>1 else ' — one seed'));figs.save(fig,'T01_optimizer_diagnostics'")
    s=replace(s,"figs=Figures(out); rows=[];cf=[];sources={}; control_rows=[]", "figs=Figures(out); rows=[];cf=[];sources={}; control_rows=[]; status=[]")
    s=replace(s,"        d=read_json(path/'model_validation_summary.json'); p=path/'external_on_policy_predictions.csv';f=pd.read_csv(p)","""        d=read_json(path/'model_validation_summary.json')
        status.append(dict(seed=seed,model_status=d['model_status'],synthetic_samples_used=d['training_synthetic_data_accounting']['synthetic_samples_used']))
        if d['model_status']=='unfitted':
            for fn in ('model_validation_summary.json','manager_provenance.json'):sources[f'seed{seed}/{fn}']=sha(path/fn)
            continue
        p=path/'external_on_policy_predictions.csv';f=pd.read_csv(p,float_precision='round_trip')""")
    s=replace(s,"    table=pd.DataFrame(rows);save_csv(out,'model_quality_by_seed.csv',table);cf=pd.concat(cf);save_csv(out,'counterfactual_results.csv',cf)","    save_csv(out,'model_status_by_seed.csv',status)\n    if not rows:\n        figs.finish();return sources\n    table=pd.DataFrame(rows);save_csv(out,'model_quality_by_seed.csv',table);cf=pd.concat(cf);save_csv(out,'counterfactual_results.csv',cf)")
    s=replace(s,"        if len(seeds)>1:axes[0].errorbar", "        if len(table.seed.unique())>1:axes[0].errorbar")
    s=replace(s,"ylabel='Normalized one-step delta RMSE'", "ylabel='Mean per-fitted-seed delta RMSE (normalized)' ")
    s=replace(s,"axes[0].legend();\n    for seed,g in cf.groupby('seed')", "axes[0].legend();axes[0].set_title(f'Fitted models: {table.seed.nunique()}/{len(seeds)} selected seeds' + ('; ± seed SD' if table.seed.nunique()>1 else '; no seed SD'));\n    for seed,g in cf.groupby('seed')")
    return s

def rich(s):
    s=replace(s,"expected_seed=seed,expected_epoch=cfg['epochs']) for m in","expected_seed=seed,expected_epoch=cfg['epochs'],expected_sha=data['sources'][f'{m}_seed{seed}/latest.pt']) for m in")
    s=replace(s,"        if (o[:,None,:,0]+np.maximum", "        check_close(raw[:,:,:,0].std(axis=0,ddof=1),data['moments'][si,di,:,:,1],'Stored rich power SD',atol=1e-5)\n        check_close(raw[:,:,:,0]/12000,raw[:,:,:,1],'Rich per-step power to gross energy')\n        check_close(o[:,:,0]/12000,o[:,:,1],'Oracle per-step power to gross energy')\n        if (o[:,None,:,0]+np.maximum")
    s=replace(s,"figs.save(fig,f'R01_daily_trajectory_{sid}'","fig.suptitle(f'{sid} — seed {data[\"seeds\"][0]}, draw 0 (actual trajectory)');figs.save(fig,f'R01_daily_trajectory_{sid}'")
    s=replace(s,"figs.save(fig,f'R02_daily_actions_{sid}'","fig.suptitle(f'{sid} — signed commands, mean ± empirical seed/draw SD');figs.save(fig,f'R02_daily_actions_{sid}'")
    s=replace(s,"'How much control effort is commanded during the day?'","'What are the signed control commands during the day?'")
    s=replace(s,"ax.plot(g.regime.astype(str),g['mean'],marker='o',label=f'Seed {seed}')", "ax.scatter(g.regime.cat.codes,g['mean'],marker='o',label=f'Seed {seed}')\n    ax.set_xticks(range(3),['<0.35','0.35–<0.70','≥0.70'])")
    s=replace(s,"figset_title=f'{scope.capitalize()} scope: {len(days)} historical days'","figset_title=f'{scope.capitalize()} scope: {len(days)} days — mean ± empirical mixture SD'")
    s=replace(s,"t=g.groupby('month')[field].agg(['mean','std']);line(ax,t.index.to_numpy(),t['mean'].to_numpy(),method,t['std'].to_numpy() if S>1 else None)","t=g.groupby('month')[field].agg(['mean','std']);ax.errorbar(t.index.to_numpy()+(METHODS.index(method)-1.5)*.10,t['mean'].to_numpy(),yerr=t['std'].to_numpy() if S>1 else None,fmt='o',capsize=2,color=COLORS[method],label=LABELS[method])")
    s=replace(s,"ax.imshow(matrix,aspect='auto',origin='lower',cmap='RdBu_r'", "ax.imshow(matrix,aspect='auto',origin='lower',interpolation='nearest',cmap='RdBu_r'")
    s=replace(s,"ax.imshow(h,origin='lower',aspect='auto',extent=", "ax.imshow(h,origin='lower',aspect='auto',interpolation='nearest',extent=")
    return s

def offline(s):
    s=replace(s,"line(ax,np.arange(len(labels)),a['mean'].to_numpy(),m,a.sd_seed_means.to_numpy() if S>1 else None)","ax.errorbar(np.arange(len(labels))+(METHODS.index(m)-1.5)*.10,a['mean'].to_numpy(),yerr=a.sd_seed_means.to_numpy() if S>1 else None,fmt='o',capsize=2,color=COLORS[m],label=LABELS[m])")
    s=replace(s,"figset.save(fig,'E01_energy_and_effort'","fig.suptitle('Means ± sample SD of seed means' if S>1 else 'Means — one seed; no seed SD');figset.save(fig,'E01_energy_and_effort'")
    s=replace(s,"figset.save(fig,f'E02_{kind}_net_and_effort'","fig.suptitle('Mean per evaluated day ± sample SD of seed means' if S>1 else 'Mean per evaluated day — one seed');figset.save(fig,f'E02_{kind}_net_and_effort'")
    s=replace(s,"ax.set(title=sid,ylabel='DC power (W)')", "ax.set(title=f'{sid} — seed {data[\"seeds\"][0]}, mean ± weather-draw SD',ylabel='DC power (W)')")
    s=replace(s,"ax.set(ylabel='DC power (W)',title=title)","ax.set(ylabel='DC power (W)',title=title+' (mean ± 1 SD)' if S>1 or ax is axes[0] else title)")
    return s

def dataset(s):
    s=replace(s,"figs.save(fig,'D02_seasonal_weather'","fig.suptitle('Historical-day mean ± sample SD');figs.save(fig,'D02_seasonal_weather'")
    return replace(s,"ax.imshow(mat.T,aspect='auto',origin='lower',extent=", "ax.imshow(mat.T,aspect='auto',origin='lower',interpolation='nearest',extent=")

def advanced_report(s):
    return replace(s,"write_json(out/'report_manifest.json',dict(schema=1","from .scientific_checks import source_hashes,report_inventory\n        code.update(source_hashes(root,['PV_experiments/scientific_checks.py','MBPO_workspace/src/sample_complexity.py','MBPO_workspace/src/curve_statistics.py']))\n        write_json(out/'report_manifest.json',dict(schema=1,output_sha256=report_inventory(out)")

def represent(s):
    s=replace(s,"code['represent.py']=sha(Path(__file__))", "code['represent.py']=sha(Path(__file__));code['scientific_checks.py']=sha(Path(__file__).with_name('scientific_checks.py'));code['sample_complexity.py']=sha(root/'MBPO_workspace/src/sample_complexity.py');code['curve_statistics.py']=sha(root/'MBPO_workspace/src/curve_statistics.py')")
    s=replace(s,"write_json(out/'representation_manifest.json',dict(schema=1,experiment=cfg['name'],seeds=seeds,stage=args.stage,kinds=list(kinds)","from .scientific_checks import report_inventory\n        write_json(out/'representation_manifest.json',dict(schema=1,output_sha256=report_inventory(out),experiment=cfg['name'],seeds=seeds,stage=args.stage,kinds=list(kinds)")
    return s

TRANSFORMS={
'PV_experiments/evaluate.py':evaluation,'MBPO_workspace/workflows/09_monte_carlo_evaluate.py':mc,
'MBPO_workspace/workflows/06_model_validation.py':model,'PV_experiments/report.py':report,
'MBPO_workspace/src/sample_complexity.py':sample,'PV_experiments/advanced_learning.py':advanced,
'PV_experiments/representation/training.py':training,'PV_experiments/representation/trajectories.py':rich,
'PV_experiments/representation/offline.py':offline,'PV_experiments/representation/dataset.py':dataset,
'PV_experiments/represent.py':represent,'PV_experiments/advanced_report.py':advanced_report}

def git(root,*args,input=None):
    return subprocess.run(['git','-C',str(root),*args],input=input,stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True).stdout

DEPENDENCIES=('MBPO_workspace/src/curve_statistics.py','mbpo/audit_support.py','mbpo/audit_release.py','PV_experiments/core.py','PV_experiments/representation/statistics.py','PV_experiments/representation/common.py','PV_experiments/calendar_heatmaps.py')

def build(root):
    git(root,'merge-base','--is-ancestor',BASE,'HEAD')
    for dep in DEPENDENCIES:
        if (root/dep).read_bytes()!=git(root,'show',f'{BASE}:{dep}'):raise ValueError(f'Unexpected dependency edits: {dep}')
    files={}
    for p,transform in TRANSFORMS.items():
        before=git(root,'show',f'{BASE}:{p}').decode('utf-8');after=transform(before)
        compile(after,p,'exec');current=(root/p).read_bytes()
        if current not in (before.encode(),after.encode()):raise ValueError(f'Unexpected local edits: {p}')
        if current!=after.encode():files[p]=(current,after.encode())
    for p in sorted((HERE/'payload').rglob('*.py')):
        rel=p.relative_to(HERE/'payload').as_posix();after=p.read_bytes();compile(after,rel,'exec')
        target=root/rel
        if target.exists() and target.read_bytes()!=after:raise ValueError(f'Unexpected existing file: {rel}')
        if not target.exists():files[rel]=(b'',after)
    chunks=[]
    for p,(before,after) in files.items():
        # git's unified format includes explicit no-final-newline markers.
        a=before.decode().splitlines(keepends=True);b=after.decode().splitlines(keepends=True)
        chunks.append(f'diff --git a/{p} b/{p}\n')
        if not before:chunks.append('new file mode 100644\n')
        for line in difflib.unified_diff(a,b,fromfile='a/'+p if before else '/dev/null',tofile='b/'+p):
            chunks.append(line if line.endswith('\n') else line+'\n\\ No newline at end of file\n')
    return ''.join(chunks).encode(),list(files)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',default='.');g=p.add_mutually_exclusive_group(required=True);g.add_argument('--apply',action='store_true');g.add_argument('--dry-run',action='store_true');a=p.parse_args();root=Path(a.root).resolve()
    try:
        if Path(git(root,'rev-parse','--show-toplevel').decode().strip()).resolve()!=root:raise ValueError('--root must be the Git repository root')
        patch,files=build(root)
        if not files:print('Already applied.');return
        git(root,'apply','--check','-',input=patch)
        print('\n'.join(files));print('All source anchors, compilation and git apply --check passed.')
        if a.apply:git(root,'apply','-',input=patch);print('Applied in place. No data, checkpoints or outputs changed.')
    except subprocess.CalledProcessError as e:raise SystemExit(e.stderr.decode(errors='replace') or str(e))
    except (ValueError,OSError,SyntaxError) as e:raise SystemExit(str(e))

if __name__=='__main__':main()
