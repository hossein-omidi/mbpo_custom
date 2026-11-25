"""Diagnostics separated from fixed-validation sample efficiency."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .common import Figures,save_csv,style,line,sha,read_json,plt
from .statistics import check_close,finite


def render_training(cfg,seeds,root,out):
    from PV_experiments.core import run_path
    figs=Figures(out); frames={}; sources={}
    for alg in ('sac','mbpo'):
        for seed in seeds:
            p=run_path(cfg,alg,seed,root)/'episodes.csv'; f=pd.read_csv(p); steps=finite(f.env_steps); meta=read_json(p.parent/'run_metadata.json')
            if len(f)!=cfg['epochs'] or not np.array_equal(steps,np.arange(1,len(f)+1)*meta['horizon']) or steps[-1]!=meta['final_env_steps']:raise ValueError(f'Incomplete training history: {p}')
            check_close(f.gross_energy_kwh-f.movement_cost_kwh,f.return_kwh,'Training episode energy accounting')
            u=finite(f.updates); changed=np.diff(np.r_[0,u])>0
            if np.any(np.diff(np.r_[0,u])<0):raise ValueError('Decreasing optimizer update counter')
            finite(f.alpha)
            if changed.any():finite(f.loc[changed,['critic_loss','actor_loss']])
            frames[alg,seed]=f; sources[f'{alg}_seed{seed}/episodes.csv']=sha(p)
    # Training trajectories can use different days: diagnostic, never controlled performance curve.
    fig,axes=plt.subplots(2,2,figsize=(10,7),sharex=True); stats=[]
    for ax,field in zip(axes.flat,('return_kwh','critic_loss','actor_loss','alpha')):
        for alg in ('sac','mbpo'):
            a=np.stack([frames[alg,s][field].rolling(25,min_periods=1).mean().to_numpy() for s in seeds]); valid=np.isfinite(a).all(axis=0);steps=frames[alg,seeds[0]].env_steps.to_numpy(); mu=a[:,valid].mean(axis=0);sd=a[:,valid].std(axis=0,ddof=1) if len(seeds)>1 else None
            line(ax,steps[valid],mu,alg,sd); stats.extend(dict(algorithm=alg,field=field,env_steps=int(t),mean=v,sd_seed_means=(sd[i] if sd is not None else np.nan)) for i,(t,v) in enumerate(zip(steps[valid],mu)))
        ax.set_ylabel(field.replace('_',' '));style(ax)
    axes[0,0].legend();axes[-1,0].set_xlabel('Real simulator transitions');axes[-1,1].set_xlabel('Real simulator transitions');save_csv(out,'training_diagnostics.csv',stats);fig.suptitle('25-episode trailing means' + (' ± sample seed SD' if len(seeds)>1 else ' — one seed'));figs.save(fig,'T01_optimizer_diagnostics','Are optimization and exploratory returns behaving plausibly?','25-episode rolling means within seed, then sample SD across seed curves; only common finite points. Different sampled days; not policy evaluation.','episodes.csv')
    fig,axes=plt.subplots(2,1,figsize=(9,7),sharex=True); rows=[]
    for seed in seeds:
        f=frames['mbpo',seed]; attempt=finite(f.synthetic_attempted);accepted=finite(f.synthetic_accepted);rejected=finite(f.synthetic_rejected);used=finite(f.synthetic_samples_used);real=finite(f.real_samples_used)
        check_close(attempt,accepted+rejected,'Synthetic attempted = accepted + rejected')
        if min(attempt.min(),accepted.min(),rejected.min(),used.min(),real.min())<0:raise ValueError('Negative replay accounting')
        den=np.cumsum(used+real); frac=np.divide(np.cumsum(used),den,out=np.full_like(den,np.nan,dtype=float),where=den>0)
        axes[0].plot(f.env_steps,np.cumsum(accepted),label=f'Seed {seed}');axes[0].plot(f.env_steps,np.cumsum(used),ls='--',color=axes[0].lines[-1].get_color());axes[1].plot(f.env_steps,frac,label=f'Seed {seed}')
        rows.extend(dict(seed=seed,env_steps=int(t),cumulative_attempted=a,cumulative_accepted=b,cumulative_rejected=c,cumulative_synthetic_batch_draws=d,cumulative_real_batch_draws=e,synthetic_batch_fraction=g) for t,a,b,c,d,e,g in zip(f.env_steps,np.cumsum(attempt),np.cumsum(accepted),np.cumsum(rejected),np.cumsum(used),np.cumsum(real),frac))
    axes[0].set_ylabel('Cumulative count\nsolid: accepted; dashed: sampled');axes[1].set(xlabel='Real simulator transitions',ylabel='Synthetic fraction of batch draws');axes[0].legend();style(axes[0]);style(axes[1]);save_csv(out,'synthetic_replay_usage.csv',rows);figs.save(fig,'T02_model_replay_use','Are imagined transitions actually used in actor/critic batches?','Per-seed cumulative counts; repeated replay draws are not unique transitions.','MBPO episodes.csv');figs.finish();return sources


def render_learning(cfg,seeds,root,out):
    from PV_experiments.core import run_path
    from MBPO_workspace.src.sample_complexity import analyze_sample_complexity
    import tempfile,shutil
    from pathlib import Path
    if len(seeds)<2:raise ValueError('Learning comparison requires at least two paired training seeds')
    # Reuse the audited estimator; retain its numerical outputs, not its duplicate figures.
    with tempfile.TemporaryDirectory(prefix='pv_estimator_') as tmp:
        result=analyze_sample_complexity([run_path(cfg,'sac',s,root) for s in seeds],[run_path(cfg,'mbpo',s,root) for s in seeds],tmp)
        for p in Path(tmp).iterdir():
            if p.suffix in ('.csv','.json'):shutil.copyfile(p,out/p.name)
    figs=Figures(out)
    d=pd.read_csv(out/'learning_curve_descriptive_sd.csv'); diff=pd.read_csv(out/'paired_difference_descriptive_sd.csv');target=result['target_kwh_per_episode']
    fig,axes=plt.subplots(2,1,figsize=(9,7),sharex=True)
    for alg in ('sac','mbpo'):
        g=d[d.algorithm==alg].sort_values('env_steps');line(axes[0],g.env_steps.to_numpy(),g.mean_net_energy_kwh.to_numpy(),alg,g.sd_across_training_seed_means_kwh.to_numpy())
    axes[0].axhline(target,color='k',ls='--',lw=.9,label='Predefined target');axes[0].legend();axes[0].set_ylabel('Mean validation net energy (kWh/day)')
    diff=diff.sort_values('env_steps');x=diff.env_steps.to_numpy();y=diff.mean_difference_net_energy_kwh.to_numpy();sd=diff.sd_across_paired_training_seed_means_kwh.to_numpy();axes[1].plot(x,y,color='#009E73');axes[1].fill_between(x,y-sd,y+sd,color='#009E73',alpha=.15);axes[1].axhline(0,color='k',ls='--',lw=.8);axes[1].set(xlabel='Real PV simulator transitions',ylabel='MBPO − SAC (kWh/day)')
    for ax in axes:style(ax)
    figs.save(fig,'L01_learning_efficiency','Which method learns a useful policy after fewer simulator interactions?','Mean ± sample SD of training-seed means on the same frozen validation days and realization 0. Paired difference formed before averaging.','learning_curve_descriptive_sd.csv and paired_difference_descriptive_sd.csv')
    df=pd.read_csv(out/'sample_complexity_by_seed.csv');fig,ax=plt.subplots(figsize=(8,4.5))
    for alg,marker,offset in [('sac','o',-.08),('mbpo','s',.08)]:
        g=df[df.algorithm==alg];hit=g.reached_target.astype(str).str.lower()=='true';ax.scatter(g.loc[hit,'seed']+offset,g.loc[hit,'real_env_steps_to_target'],marker=marker,label=alg.upper()+' reached');ax.scatter(g.loc[~hit,'seed']+offset,g.loc[~hit,'last_evaluated_env_steps'],marker='^',facecolors='none',edgecolors='k',label=alg.upper()+' not reached (budget)')
    ax.set(xlabel='Training seed',ylabel='Real transitions',xticks=seeds);ax.legend(fontsize=8);style(ax);figs.save(fig,'L02_target_with_censoring','How many real simulator steps are required, including runs that never reach target?','Confirmed attainment (end of sustained sequence); open triangles mark last evaluated budget, not reaching time.','sample_complexity_by_seed.csv')
    fig,axes=plt.subplots(1,2,figsize=(10,4.4)); compute=pd.read_csv(out/'computational_cost_by_seed.csv')
    for ax,table,field,label in [(axes[0],df,'auc_mean_net_kwh_per_episode','Mean learning-curve AUC (kWh/day)'),(axes[1],compute,'measured_training_plus_evaluation_seconds','Logged training + evaluation time (s)')]:
        q=table.pivot(index='seed',columns='algorithm',values=field).loc[seeds,['sac','mbpo']];finite(q.to_numpy())
        for seed,row in q.iterrows():ax.plot([0,1],row.to_numpy(),marker='o',alpha=.55,label=f'Seed {seed}')
        ax.set(xticks=[0,1],xticklabels=['SAC','MBPO'],ylabel=label);style(ax)
    axes[1].legend(fontsize=7);figs.save(fig,'L03_auc_and_computational_cost','Is improved sample efficiency accompanied by greater computational cost?','Paired seed values; normalized AUC over common measured interaction range. Time excludes unlogged startup and reference construction.','sample_complexity_by_seed.csv and computational_cost_by_seed.csv')
    figs.finish()
    from .common import write_json
    write_json(out/'learning_report_notes.json',dict(note='Original estimator CSVs and report.json retained unchanged; its legacy figure paths in report.json are not exported. Canonical plots are listed in figure_manifest.json. Bootstrap intervals are retained as numerical sensitivity intervals, not presented as calibrated confidence bands.'))
    sources={}
    for alg in ('sac','mbpo'):
        for seed in seeds:
            p=run_path(cfg,alg,seed,root)
            for file in ('run_metadata.json','scenario_split.json','episodes.csv','periodic_evaluation/evaluations.csv','periodic_evaluation/evaluation_days.csv','periodic_evaluation/evaluation_references.csv'):sources[f'{alg}_seed{seed}/{file}']=sha(p/file)
    return sources


def render_model(cfg,seeds,root,out,realization=0):
    from PV_experiments.evaluate import evaluation_output,_provenance,_require_matching_provenance,_verify_model_validation_summary
    from PV_experiments.core import run_path
    figs=Figures(out); rows=[];cf=[];sources={}; control_rows=[]; status=[]
    for seed in seeds:
        path=evaluation_output(cfg,'model',seed,root)/f'realization{realization}';identity=_provenance(cfg,'model',seed,root,realization_id=realization);_require_matching_provenance(path,identity);_verify_model_validation_summary(path/'model_validation_summary.json',cfg,seed,realization,identity['scenario_protocol_sha256'],root)
        d=read_json(path/'model_validation_summary.json')
        status.append(dict(seed=seed,model_status=d['model_status'],synthetic_samples_used=d['training_synthetic_data_accounting']['synthetic_samples_used']))
        if d['model_status']=='unfitted':
            for fn in ('model_validation_summary.json','manager_provenance.json'):sources[f'seed{seed}/{fn}']=sha(path/fn)
            continue
        p=path/'external_on_policy_predictions.csv';f=pd.read_csv(p,float_precision='round_trip')
        control_rows.append(dict(seed=seed,**d['counterfactual_action_test'],**{'train_'+k:v for k,v in d['training_synthetic_data_accounting'].items() if isinstance(v,(int,float)) or v is None}))
        if set(d['validation_scenario_ids'])!=set(read_json(run_path(cfg,'mbpo',seed,root)/'scenario_split.json')['validation_ids']):raise ValueError('Model diagnostics do not cover the frozen validation partition')
        if set(f.scenario_id.astype(str))!=set(d['validation_scenario_ids']) or f.duplicated(['scenario_id','step']).any() or not (f.groupby('scenario_id').step.apply(lambda x:set(x)==set(range(195)))).all():raise ValueError('Incomplete model validation trajectories')
        for ch in ('dni_norm','dhi_norm','power_norm'):
            a=finite(f[f'actual_delta_{ch}']); b=finite(f[f'predicted_delta_{ch}']);rmse=float(np.sqrt(np.mean((b-a)**2)));base=float(np.sqrt(np.mean(a*a))); m=d['external_on_policy_one_step'][ch];check_close([rmse,base],[m['rmse'],m['persistence_rmse']],'Model prediction RMSE')
            rows.append(dict(seed=seed,channel=ch,rmse=rmse,persistence_rmse=base,bias=m['bias'],coverage90=m['coverage_90'],coverage95=m['coverage_95']))
        p2=path/'counterfactual_predictions.csv';ff=pd.read_csv(p2);
        if set(ff.scenario_id.astype(str))!=set(d['validation_scenario_ids']) or ff.duplicated(['scenario_id','step','action_type']).any() or len(ff)!=d['counterfactual_action_test']['n_counterfactual_transitions']:raise ValueError('Incomplete counterfactual validation grid')
        err=finite(ff.reward_error_kwh);check_close([np.sqrt(np.mean(err**2))],[d['counterfactual_action_test']['reward_rmse_kwh']],'Counterfactual reward RMSE');ff['seed']=seed;cf.append(ff)
        for fn in ('model_validation_summary.json','external_on_policy_predictions.csv','counterfactual_predictions.csv','manager_provenance.json'):sources[f'seed{seed}/{fn}']=sha(path/fn)
    save_csv(out,'model_status_by_seed.csv',status)
    if not rows:
        figs.finish();return sources
    table=pd.DataFrame(rows);save_csv(out,'model_quality_by_seed.csv',table);cf=pd.concat(cf);save_csv(out,'counterfactual_results.csv',cf)
    save_csv(out,'model_control_diagnostics_by_seed.csv',control_rows)
    fig,axes=plt.subplots(1,2,figsize=(10,4.5)); channels=['dni_norm','dhi_norm','power_norm'];x=np.arange(3)
    for field,off,label in [('rmse',-.15,'Ensemble'),('persistence_rmse',.15,'Persistence')]:
        t=table.groupby('channel')[field].agg(['mean','std']).loc[channels];axes[0].bar(x+off,t['mean'],.28,label=label); 
        if len(table.seed.unique())>1:axes[0].errorbar(x+off,t['mean'],yerr=t['std'],fmt='none',color='k',capsize=3)
    axes[0].set(xticks=x,xticklabels=['DNI','DHI','Power'],ylabel='Mean per-fitted-seed delta RMSE (normalized)' );axes[0].legend();axes[0].set_title(f'Fitted models: {table.seed.nunique()}/{len(seeds)} selected seeds' + ('; ± seed SD' if table.seed.nunique()>1 else '; no seed SD'));
    for seed,g in cf.groupby('seed'):axes[1].scatter(g.normalized_epistemic_disagreement,np.abs(g.reward_error_kwh),s=5,alpha=.22,label=f'Seed {seed}')
    axes[1].set(xlabel='Normalized ensemble disagreement',ylabel='Absolute counterfactual reward error (kWh)');axes[1].legend(fontsize=7);style(axes[0]);style(axes[1]);figs.save(fig,'M01_model_skill_and_counterfactual_error','Does the learned model outperform persistence and predict alternative action effects?','One fixed validation weather realization per historical day; seed SD for RMSE, counterfactual scatter descriptive.','model validation CSVs and JSON');figs.finish();return sources
