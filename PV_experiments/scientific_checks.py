"""Shared numerical release checks. No training or simulator changes."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

METHODS=('fixed_no_motion','sun_slew_limited','sac','mbpo')
FIELDS=('gross_kwh','movement_kwh','net_kwh')
ACTIONS=('policy','zero','tilt_positive','tilt_negative','azimuth_positive','azimuth_negative','diagonal_positive','diagonal_negative')
CHANNELS=('dni_norm','dhi_norm','power_norm')


def same(a,b,what,atol=1e-9,rtol=1e-8):
    a=np.asarray(a,dtype=float); b=np.asarray(b,dtype=float)
    if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all() or not np.allclose(a,b,atol=atol,rtol=rtol):
        raise ValueError(f'{what}: numerical/shape mismatch')


def grid(frame,columns,levels):
    ix=pd.MultiIndex.from_product(levels,names=columns)
    f=frame.set_index(columns,verify_integrity=True)
    if len(f)!=len(ix) or not f.index.isin(ix).all():raise ValueError(f'Incomplete or extra grid: {columns}')
    return f.reindex(ix)


def read_csv(path):
    return pd.read_csv(path,float_precision='round_trip',dtype={'scenario_id':str,'profile_sha256':str})


def validate_mc_tables(directory,days,draws,*,run_dirs=None,epoch=None):
    """Check all daily/clock/pooled tables, including final-policy draw-zero parity.

    Clock SDs are sample SD across draws; pooled SDs are empirical population SD.
    Raw per-draw time paths are intentionally not retained by standard evaluation.
    Their SD can be independently checked by the rich replay workflow.
    """
    from MBPO_workspace.src.curve_statistics import aggregate_mc_power_profiles
    directory=Path(directory); days=sorted(days); draws=int(draws)
    if not days or len(set(days))!=len(days) or draws<2:raise ValueError('Invalid MC design')
    d=grid(read_csv(directory/'all_day_draw_method.csv'),['scenario_id','draw','method'],[days,range(draws),METHODS])
    x=d[list(FIELDS)].to_numpy(dtype=float).reshape(len(days),draws,4,3)
    same(x[...,0]-x[...,1],x[...,2],'Daily net energy')
    if (x[...,:2]<-1e-10).any():raise ValueError('Negative physical energy')
    same(x[:,:,0,1],np.zeros((len(days),draws)),'Fixed actuator energy')
    hashes=d.profile_sha256.to_numpy().reshape(len(days),draws,4)
    if any(not isinstance(h,str) or len(h)!=64 or any(c not in '0123456789abcdef' for c in h) for h in hashes.flat) or not (hashes==hashes[:,:,:1]).all():raise ValueError('Invalid paired weather hash')
    p=grid(read_csv(directory/'daily_power_mean_std_by_utc_step.csv'),['scenario_id','method','step'],[days,METHODS,range(1,196)])
    cols=['power_mean_w','power_sd_w','power_q05_w','power_q95_w']
    v=p[cols].to_numpy(dtype=float).reshape(len(days),4,195,4)
    if not np.isfinite(v).all() or (v<0).any() or (v[...,2]>v[...,3]).any():raise ValueError('Invalid conditional power moments/quantiles')
    # For M nonnegative values, sum(x^2) <= sum(x)^2, so sample SD <= sqrt(M)*mean.
    # This is a necessary feasibility condition, not reconstruction of the raw draws.
    bound=np.sqrt(draws)*v[...,0]
    if (v[...,1]>bound+1e-8+1e-8*bound).any():raise ValueError('Impossible nonnegative power moments: sample SD exceeds sqrt(M)*mean')
    times=p.time_utc.to_numpy().reshape(len(days),4,195)
    for i,sid in enumerate(days):
        expected=pd.date_range(f'{sid} 08:45',periods=196,freq='5min',tz='UTC')[1:]
        for j in range(4):
            actual=pd.to_datetime(times[i,j],utc=False)
            if any(t.utcoffset().total_seconds()!=0 for t in actual) or not pd.DatetimeIndex(actual).equals(expected):raise ValueError('Invalid power UTC grid')
    same(v[...,0].sum(axis=2)/12000,x[...,0].mean(axis=1),'Mean power integral versus gross energy')
    expected=pd.DataFrame(aggregate_mc_power_profiles(p.reset_index().to_dict('records'),draws))
    keys=['method','step']; levels=[METHODS,range(1,196)]
    a=grid(expected,keys,levels); b=grid(read_csv(directory/'mc_pooled_power_mean_sd_by_utc_step.csv'),keys,levels)
    same(a.to_numpy(dtype=float),b[a.columns].to_numpy(dtype=float),'Pooled MC power statistics',atol=1e-8)
    first=read_csv(directory/'first_day_power_mean_sd.csv'); ids=first.scenario_id.unique()
    if len(ids)!=1 or ids[0] not in days:raise ValueError('Invalid first-day power scope')
    f=grid(first,keys,levels); q=p.xs(ids[0],level='scenario_id')
    same(f[cols].to_numpy(dtype=float),q[cols].to_numpy(dtype=float),'First-day power moments')
    if not np.array_equal(f.time_utc.to_numpy(),q.time_utc.to_numpy()):raise ValueError('First-day timestamps differ')
    summary=json.loads((directory/'mc_summary.json').read_text())
    validate_daily_summary(x,days,summary)
    if run_dirs is not None:
        for algorithm,folder in run_dirs.items():
            e=Path(folder)/'periodic_evaluation'; refs=read_csv(e/'evaluation_references.csv').set_index('scenario_id',verify_integrity=True)
            ev=read_csv(e/'evaluation_days.csv'); ev=ev[ev.epoch==int(epoch)].set_index('scenario_id',verify_integrity=True)
            if not set(days)<=set(refs.index) or not set(days)<=set(ev.index):raise ValueError('Final periodic reference/day coverage mismatch')
            if not (ev.loc[days,'episode_steps']==195).all() or not (refs.loc[days,'episode_steps']==195).all():raise ValueError('Periodic horizon mismatch')
            for j,prefix in enumerate(('fixed','dual_axis')):
                same(refs.loc[days,[f'{prefix}_gross_energy_kwh',f'{prefix}_movement_cost_kwh',f'{prefix}_net_energy_kwh']],x[:,0,j],'Periodic baseline versus draw zero',atol=1e-8)
            same(ev.loc[days,['gross_energy_kwh','movement_cost_kwh','net_energy_kwh']],x[:,0,METHODS.index(algorithm)],'Final learned policy versus draw zero',atol=1e-8)
    return x


def validate_daily_summary(x,days,summary):
    """Independent raw-return checks for every reported nested mean/SD/SE/quantile."""
    D,M=x.shape[:2]
    def check(y,r):
        mu=y.mean(axis=1); mean=mu.mean(); within=y.var(axis=1,ddof=1)
        between=mu.var(ddof=0); within_pop=(M-1)/M*within.mean(); total=y.var(ddof=0)
        correction=between-(1-1/D)*within.mean()/M
        metrics=dict(n_historical_days=D,draws_per_day=M,mean_net_kwh=mean,
            between_day_variance_kwh2=between,within_day_uncertainty_variance_kwh2=within_pop,
            total_descriptive_sd_kwh=np.sqrt(total),total_empirical_variance_kwh2=total,
            between_day_variance_corrected_kwh2=max(0,correction),between_day_untruncated_correction_kwh2=correction,
            within_day_unbiased_variance_kwh2=within.mean(),mc_se_mean_fixed_days_kwh=np.sqrt(within.sum()/(D*D*M)))
        for k,v in metrics.items():same(v,r[k],f'Nested MC {k}')
        same(np.quantile(y,[.05,.5,.95]),[r['pooled_quantiles_kwh'][k] for k in ('q05','median','q95')],'Nested MC quantiles')
        if set(r['each_day'])!=set(days):raise ValueError('Missing nested MC daily summaries')
        for i,day in enumerate(days):
            v=r['each_day'][day]; same([M,mu[i],np.sqrt(within[i]),np.sqrt(within[i]/M)], [v[k] for k in ('n','mean','sd_descriptive','mc_se_mean')],'MC day summary')
            same(np.quantile(y[i],[.05,.5,.95]),[v[k] for k in ('q05','median','q95')],'MC day quantiles')
    if set(summary['methods'])!=set(METHODS):raise ValueError('Incorrect MC methods in summary')
    for j,method in enumerate(METHODS):check(x[:,:,j,2],summary['methods'][method])
    pairs=[('mbpo','sac'),('mbpo','fixed_no_motion'),('sac','fixed_no_motion'),('mbpo','sun_slew_limited'),('sac','sun_slew_limited')]
    if set(summary['paired_net_energy_differences'])!={a+'_minus_'+b for a,b in pairs}:raise ValueError('Incorrect MC paired methods')
    for a,b in pairs:check(x[:,:,METHODS.index(a),2]-x[:,:,METHODS.index(b),2],summary['paired_net_energy_differences'][a+'_minus_'+b])


def verify_completed_run(root,meta):
    from mbpo.audit_release import completed_training_files
    from mbpo.audit_support import sha_file
    root=Path(root)
    if meta.get('training_status')!='completed' or not meta.get('training_output_sha256') or not meta.get('final_checkpoint_sha256'):raise ValueError('Unsealed training run; publication requires verified completed runs')
    if meta.get('training_identity_sha256')!=sha_file(root/'training_identity.json'):raise ValueError('Training identity seal mismatch')
    if meta['training_output_sha256']!=completed_training_files(root,meta) or meta['final_checkpoint_sha256']!=sha_file(root/'latest.pt'):raise ValueError('Training source seal mismatch')


def verify_run_comparability(runs):
    """Compare already verified completed runs, without equating SAC and MBPO configs.

    Frozen runtime identities describe training-time source/dependencies, not the
    current reporting code. Per-algorithm config hashes must be identical across
    seeds; shared physical/protocol/SAC configuration must agree across methods.
    """
    runs=list(runs)
    if not runs:raise ValueError('No comparable runs')
    ref=None; configs={}; seen=set(); shared=None
    shared_keys=('shared_sac_config_sha256','resolved_mdp_sha256','scenario_protocol_sha256',
                 'horizon','final_epoch','final_env_steps','validation_scenario_ids','test_scenario_ids',
                 'state_dim','action_dim','observation_mode','energy_objective','movement_cost_mode',
                 'movement_penalty','power_model','evaluation_interval_episodes','target_fraction','sustain_checkpoints')
    for run in runs:
        root=Path(run['root']); meta=run['metadata']; alg=meta['algorithm']; seed=meta['seed']
        if alg not in ('sac','mbpo') or (alg,seed) in seen:raise ValueError('Invalid or duplicate training run identity')
        seen.add((alg,seed))
        identity=json.loads((root/'training_identity.json').read_text(encoding='utf-8'))
        runtime=identity.get('runtime')
        if not isinstance(runtime,dict) or runtime.get('kind')!='training' or not runtime.get('sources') or not runtime.get('dependencies'):
            raise ValueError(f'Missing frozen training runtime: {root}')
        if ref is None:ref=runtime
        elif runtime!=ref:raise ValueError(f'Incompatible frozen training runtime: {root}')
        cfg=json.loads((root/'config.json').read_text(encoding='utf-8'))
        # Same digest convention as config_hash/config_digest in the pinned trainers.
        config_hash=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()
        if config_hash!=meta.get('config_sha256'):raise ValueError(f'Algorithm configuration hash mismatch: {root}')
        if alg in configs and configs[alg]!=config_hash:raise ValueError(f'Incompatible {alg} configurations across seeds: {root}')
        configs[alg]=config_hash
        signature=tuple(meta[k] for k in shared_keys)
        if shared is None:shared=signature
        elif signature!=shared:raise ValueError(f'Incompatible shared experiment configuration: {root}')
    if set(configs)!={'sac','mbpo'}:raise ValueError('Both SAC and MBPO are required')
    if {s for a,s in seen if a=='sac'}!={s for a,s in seen if a=='mbpo'}:raise ValueError('Unpaired training seed sets')


def recovery_accounting(root,committed,horizon):
    p=Path(root)/'recovery_events.json'; events=json.loads(p.read_text()) if p.exists() else []
    if not isinstance(events,list):raise ValueError('Invalid recovery ledger')
    known=0
    for event in events:
        n=event['discarded_complete_csv_rows']; k=event['known_repeated_transitions']
        if event['status']!='applied' or not isinstance(n,int) or n<0 or k!=n*horizon:raise ValueError('Unresolved/invalid recovery ledger')
        known+=k
    return dict(committed_training_transitions=int(committed),known_discarded_training_transitions=known,known_executed_training_transitions_lower_bound=int(committed)+known,recovery_events=len(events))


def validate_model_tables(directory,summary,days):
    """Recompute predictive metrics from persisted transition-level data."""
    directory=Path(directory); days=sorted(days)
    if sorted(summary['validation_scenario_ids'])!=days:raise ValueError('Model validation scenario coverage mismatch')
    if summary.get('model_status')=='unfitted':
        if summary['training_synthetic_data_accounting'].get('synthetic_samples_used')!=0:raise ValueError('Unfitted model has synthetic exposure')
        if summary.get('external_on_policy_one_step') is not None or summary.get('counterfactual_action_test') is not None:raise ValueError('Unavailable model has fabricated diagnostics')
        return None,None
    if summary.get('model_status')!='fitted':raise ValueError('Unknown model status')
    f=grid(read_csv(directory/'external_on_policy_predictions.csv'),['scenario_id','step'],[days,range(195)]).reset_index()
    metrics=summary['external_on_policy_one_step']
    for table,metric,scope in [(f,metrics,'external'),(read_csv(directory/'internal_holdout_predictions.csv'),summary['internal_whole_day_holdout'],'internal')]:
        if len(table)!=metric['n_transitions'] or len(table)==0:raise ValueError('Model diagnostic count mismatch')
        for ch in CHANNELS:
            y=table[f'actual_delta_{ch}'].to_numpy(); p=table[f'predicted_delta_{ch}'].to_numpy(); e=p-y
            q=table[[f'{k}_delta_{ch}' for k in ('q025','q05','q95','q975')]].to_numpy(); sd=table[f'predictive_sd_{ch}'].to_numpy()
            if not np.isfinite(q).all() or not np.isfinite(sd).all() or (np.diff(q,axis=1)<0).any() or (sd<0).any():raise ValueError('Invalid model intervals/SD')
            same(e,table[f'error_delta_{ch}'],'Prediction residual')
            m=metric[ch]
            calc=[np.sqrt(np.mean(e**2)),np.sqrt(np.mean(y**2)),np.mean(e),np.mean(np.abs(e)),np.mean(sd),np.mean((y>=q[:,1])&(y<=q[:,2])),np.mean((y>=q[:,0])&(y<=q[:,3]))]
            same(calc,[m[k] for k in ('rmse','persistence_rmse','bias','mae','mean_predictive_sd','coverage_90','coverage_95')],f'Model metrics {scope} {ch}')
            # Reproduce the sign-preserving skill convention and SD-ratio floor.
            same([(calc[1]-calc[0])/max(calc[1],1e-8),calc[0]/max(calc[4],1e-12)],
                 [m['skill_vs_persistence'],m['rmse_to_mean_predictive_sd']],f'Model skill/SD ratio {scope} {ch}')
    err=f.predicted_reward_kwh-f.actual_reward_kwh
    same([np.sqrt(np.mean(err**2)),np.mean(np.abs(err)),np.mean(err)],[metrics['reward'][k] for k in ('rmse_kwh','mae_kwh','bias_kwh')],'On-policy reward metrics')
    cf=read_csv(directory/'counterfactual_predictions.csv'); steps=np.unique(np.linspace(0,194,num=min(int(summary['states_per_scenario']),195),dtype=int))
    cf=grid(cf,['scenario_id','step','action_type'],[days,steps,ACTIONS]).reset_index()
    r=cf.predicted_reward_kwh-cf.actual_reward_kwh; p=cf.predicted_next_power_norm-cf.actual_next_power_norm
    same(r,cf.reward_error_kwh,'Counterfactual reward residual');same(p,cf.power_error_norm,'Counterfactual power residual')
    same(r,p/6,'Counterfactual reward/power error identity',atol=2e-8)
    from mbpo.audit_support import counterfactual_metrics
    from mbpo.audit_release import equivalent_pose_metrics
    c=summary['counterfactual_action_test']; same([np.sqrt(np.mean(r*r)),np.sqrt(np.mean(p*p)),len(cf)],[c['reward_rmse_kwh'],c['power_rmse_norm'],c['n_counterfactual_transitions']],'Counterfactual summary')
    same([np.mean(np.abs(r)),np.mean(r),np.mean(np.abs(p)),np.mean(cf.normalized_epistemic_disagreement)],
         [c[k] for k in ('reward_mae_kwh','reward_bias_kwh','power_mae_norm','mean_normalized_epistemic_disagreement')],'Counterfactual additional error metrics')
    for field in ('next_tilt_norm','next_azimuth_sin','next_azimuth_cos','normalized_epistemic_disagreement'):
        z=cf[field].to_numpy(dtype=float)
        if not np.isfinite(z).all():raise ValueError(f'Nonfinite counterfactual {field}')
    if (cf.normalized_epistemic_disagreement<0).any():raise ValueError('Negative counterfactual disagreement')
    records=cf.to_dict('records')
    # Use the exact fixed tolerances of the producer, never a supplied summary tolerance.
    for k,v in {**counterfactual_metrics(records),**equivalent_pose_metrics(records)}.items():
        if v is None:
            if c[k] is not None:raise ValueError('Counterfactual undefined metric mismatch')
        else:same(v,c[k],f'Counterfactual {k}')
    return f,cf


def verify_model_accounting(summary,run_dir):
    ep=pd.read_csv(Path(run_dir)/'episodes.csv'); a=summary['training_synthetic_data_accounting']
    counts=['synthetic_attempted','synthetic_accepted','synthetic_rejected','synthetic_samples_used','real_samples_used']
    x=ep[counts].to_numpy(dtype=float)
    if not np.isfinite(x).all() or (x<0).any() or not np.array_equal(x,np.floor(x)):raise ValueError('Invalid committed MBPO replay accounting')
    same(x[:,0],x[:,1]+x[:,2],'Committed synthetic generation accounting',atol=0,rtol=0)
    same(x.sum(axis=0),[a[k] for k in counts],'Model summary replay exposure',atol=0,rtol=0)
    den=x[:,3:].sum()
    if den>0:same(x[:,3].sum()/den,a['actual_synthetic_minibatch_fraction'],'Model summary synthetic fraction')
    elif a['actual_synthetic_minibatch_fraction'] is not None:raise ValueError('Synthetic fraction undefined without batch draws')


def verify_model_physical_parity(predictions,run_dir,epoch,realization_id):
    """Model on-policy *physical truth*, not model predictions, must match periodic evaluation at draw zero."""
    f=predictions.copy(); f['gross_kwh']=f.actual_next_power_norm/6
    if not np.isfinite(f[['gross_kwh','actual_reward_kwh']]).all().all() or (f.gross_kwh<0).any():raise ValueError('Invalid physical model-validation trajectory')
    if int(realization_id)!=0:return # different conditional weather realization; not the periodic target
    totals=f.groupby('scenario_id')[['gross_kwh','actual_reward_kwh']].sum()
    ev=read_csv(Path(run_dir)/'periodic_evaluation'/'evaluation_days.csv'); ev=ev[ev.epoch==int(epoch)].set_index('scenario_id',verify_integrity=True)
    if set(ev.index)!=set(totals.index) or not (ev.episode_steps==195).all():raise ValueError('Model validation versus periodic day coverage')
    ev=ev.loc[totals.index]; same(totals.actual_reward_kwh,ev.net_energy_kwh,'Model physical trajectory versus final periodic net return',atol=1e-8)
    # Gross power is decoded from float32 observations, so allow their quantization error.
    same(totals.gross_kwh,ev.gross_energy_kwh,'Model physical trajectory versus final periodic gross return',atol=2e-7)


def seal_model(directory,identity):
    from mbpo.audit_release import seal_evaluation,verify_loaded_policies
    from mbpo.audit_support import atomic_json,sha_file
    directory=Path(directory); d=json.loads((directory/'model_validation_summary.json').read_text())
    verify_loaded_policies(d,identity)
    if d.get('model_status')!='unfitted':return seal_evaluation(directory,identity)
    if set(p.name for p in directory.iterdir() if p.is_file() and p.suffix in ('.csv','.json'))!={'model_validation_summary.json'}:raise ValueError('Unexpected unfitted model artifacts')
    atomic_json(directory/'manager_provenance.json',{'identity':identity,'files':{'model_validation_summary.json':sha_file(directory/'model_validation_summary.json')}})


def source_hashes(root,paths):
    return {p:hashlib.sha256((Path(root)/p).read_bytes()).hexdigest() for p in paths}


def report_inventory(root):
    """Bind the numerical exports and rendered figure bytes to their report."""
    from mbpo.audit_support import sha_file
    root=Path(root)
    return {p.relative_to(root).as_posix():sha_file(p) for p in sorted(root.rglob('*')) if p.is_file() and p.relative_to(root).as_posix() not in ('report_manifest.json','representation_manifest.json')}


def check_report_inventory(root,manifest):
    if not manifest.get('output_sha256') or manifest['output_sha256']!=report_inventory(root):raise ValueError('Cached report is incomplete or altered; regenerate from sealed inputs')
