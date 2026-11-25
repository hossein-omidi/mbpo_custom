"""Publication-oriented, censor-aware empirical sample efficiency reporting.
Uses the audited estimator's run loader and common validation contract, not a
replacement RL algorithm. All steps refer to external PV simulator transitions.
"""
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ALGORITHMS=('sac','mbpo')
COLORS={'sac':'#0072B2','mbpo':'#009E73'}


def finite(x,what='data'):
    a=np.asarray(x,dtype=float)
    if not a.size or not np.isfinite(a).all():raise ValueError(f'Empty or nonfinite {what}')
    return a


def same(a,b,what,rtol=1e-8,atol=1e-9):
    if not np.allclose(finite(a,what),finite(b,what),rtol=rtol,atol=atol):raise ValueError(f'{what}: numerical mismatch')


def sha_file(path):
    import hashlib
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def write_json(path,data):Path(path).write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf-8')


def csv_out(path,rows):
    pd.DataFrame(rows).to_csv(path,index=False,float_format='%.12g')


def style(ax):
    ax.grid(alpha=.22,linewidth=.6);ax.spines[['top','right']].set_visible(False)


def save(fig,out,name):
    try:
        for ext in ('pdf','png'):fig.savefig(Path(out)/f'{name}.{ext}',dpi=300,bbox_inches='tight')
    finally:plt.close(fig)


def summary(v):
    v=finite(v);return {'mean':float(v.mean()),'sd':float(v.std(ddof=1)) if len(v)>1 else None,'n':len(v)}


def expected_steps(meta):
    h=int(meta['horizon']);interval=int(meta['evaluation_interval_episodes']);epochs=int(meta['final_epoch'])
    if min(h,interval,epochs)<1 or h*epochs!=int(meta['final_env_steps']):raise ValueError('Invalid evaluation schedule metadata')
    return np.array(sorted(set(range(interval,epochs+1,interval))|{epochs}),dtype=np.int64)*h


def event(curve,steps,target,sustain):
    """Retrospective onset and online-observable confirmation; no interpolation."""
    y=finite(curve);x=np.asarray(steps)
    if y.ndim!=1 or x.shape!=y.shape or np.any(np.diff(x)<=0) or not np.isfinite(target) or sustain<1:raise ValueError('Invalid target event inputs')
    streak=0
    for j,value in enumerate(y):
        streak=streak+1 if value>=target else 0
        if streak>=sustain:return int(x[j-sustain+1]),int(x[j])
    return None,None


def load_learning(cfg,seeds,root):
    from PV_experiments.core import verify_pair,run_path
    from MBPO_workspace.src.sample_complexity import _load_run
    if len(seeds)<2 or len(set(seeds))!=len(seeds):raise ValueError('At least two distinct paired training seeds required')
    seeds=sorted(seeds);splits=[verify_pair(cfg,s,root) for s in seeds]
    if any(s!=splits[0] for s in splits[1:]):raise ValueError('Frozen splits differ across seeds')
    runs={a:[_load_run(run_path(cfg,a,s,root),a) for s in seeds] for a in ALGORITHMS}
    allruns=runs['sac']+runs['mbpo'];meta=allruns[0]['metadata'];steps=expected_steps(meta)
    if len(steps)<2:raise ValueError('At least two scheduled evaluations needed')
    keys=('scenario_protocol_sha256','resolved_mdp_sha256','shared_sac_config_sha256','validation_scenario_ids','test_scenario_ids','horizon','state_dim','action_dim','energy_objective','movement_cost_mode','movement_penalty','power_model','target_fraction','sustain_checkpoints','bootstrap_replicates','evaluation_interval_episodes','final_epoch','final_env_steps')
    days=list(meta['validation_scenario_ids']);sources={};times={a:[] for a in ALGORITHMS};mat={};updates=None;replay=[]
    if len(days)<2 or len(set(days))!=len(days):raise ValueError('Need at least two distinct frozen validation days')
    if meta['movement_cost_mode']!='geometry' or float(meta['movement_penalty'])!=1.:raise ValueError('Expected movement-aware geometry objective (gain 1) for this study')
    reference=None
    for a in ALGORITHMS:
        matrices=[]
        for seed,run in zip(seeds,runs[a]):
            m=run['metadata'];p=Path(run['root'])
            if run['seed']!=seed or any(m.get(k)!=meta.get(k) for k in keys):raise ValueError(f'Run comparability mismatch: {p}')
            if set(run['aggregate'])!=set(steps):raise ValueError(f'{p}: scheduled evaluation checkpoints missing or unexpected; no silent intersection')
            if set(run['daily'])!=set(steps):raise ValueError('Daily evaluation checkpoints differ')
            u=np.array([run['updates'][int(k)] for k in steps]);
            if updates is None:updates=u
            elif not np.array_equal(u,updates):raise ValueError('Gradient-update budgets differ at common checkpoints')
            rows=run['references']
            if [r['scenario_id'] for r in rows]!=days:raise ValueError('Baseline scenario order mismatch')
            if any(int(r['episode_steps'])!=int(m['horizon']) for r in rows):raise ValueError('Baseline horizon mismatch')
            fullref=finite([[float(r[f'{prefix}_{field}']) for prefix in ('fixed','dual_axis') for field in ('gross_energy_kwh','movement_cost_kwh','net_energy_kwh')] for r in rows],'baseline energies').reshape(len(days),2,3)
            same(fullref[:,:,0]-fullref[:,:,1],fullref[:,:,2],'Baseline energy accounting');same(fullref[:,0,1],np.zeros(len(days)),'Fixed actuator energy')
            if (fullref[:,:,:2]<-1e-10).any():raise ValueError('Negative baseline gross or movement energy')
            ref=fullref
            if reference is None:reference=ref
            else:same(reference,ref,'Validation baseline references')
            daily=pd.read_csv(p/'periodic_evaluation/evaluation_days.csv',dtype={'scenario_id':str});
            if not (daily.episode_steps==int(m['horizon'])).all():raise ValueError('Periodic evaluation horizon mismatch')
            same(daily.gross_energy_kwh-daily.movement_cost_kwh,daily.net_energy_kwh,'Periodic evaluation net accounting')
            if (daily[['gross_energy_kwh','movement_cost_kwh']]< -1e-10).any().any():raise ValueError('Negative periodic gross or movement')
            x=finite([[run['daily'][int(t)][d] for t in steps] for d in days]);same(x.mean(axis=0),[run['aggregate'][int(t)] for t in steps],'Aggregate validation mean',rtol=1e-7)
            matrices.append(x)
            # Independently check exact schedule and cumulative logged wall time.
            ep=pd.DataFrame(run['episodes']);ev=pd.read_csv(p/'periodic_evaluation/evaluations.csv')
            es=finite(ep['env_steps']);expected=np.arange(1,int(m['final_epoch'])+1)*int(m['horizon'])
            if not np.array_equal(es,expected):raise ValueError('Episode log not complete/chronological')
            episode_seconds=finite(ep['episode_seconds'],'episode timing');ev=ev.set_index('env_steps',verify_integrity=True).loc[steps];evaluation_seconds=finite(ev['evaluation_seconds'],'evaluation timing')
            if (episode_seconds<0).any() or (evaluation_seconds<0).any():raise ValueError('Negative logged duration')
            train=np.cumsum(episode_seconds)[steps//int(m['horizon'])-1];evaluate=np.cumsum(evaluation_seconds)
            times[a].append(np.stack([train,evaluate,train+evaluate],axis=-1))
            if a=='mbpo':
                fields=('synthetic_attempted','synthetic_accepted','synthetic_rejected','synthetic_samples_used','real_samples_used')
                counts=finite(ep[list(fields)],'MBPO replay accounting')
                if (counts<0).any() or not np.array_equal(counts,np.floor(counts)):raise ValueError('Noninteger or negative replay count')
                same(counts[:,0],counts[:,1]+counts[:,2],'Attempted = accepted + rejected',rtol=0,atol=0)
                cum=np.cumsum(counts,axis=0)[steps//int(m['horizon'])-1]
                for k,vals in zip(steps,cum):
                    den=vals[3]+vals[4];replay.append(dict(seed=seed,env_steps=int(k),**{f:int(v) for f,v in zip(fields,vals)},actual_synthetic_batch_draw_fraction=float(vals[3]/den) if den else None))
            # Include episode data, omitted in the earlier manager learning hash list.
            for name in ('run_metadata.json','scenario_split.json','episodes.csv','periodic_evaluation/evaluations.csv','periodic_evaluation/evaluation_days.csv','periodic_evaluation/evaluation_references.csv'):
                sources[f'{a}_seed{seed}/{name}']=sha_file(p/name)
        mat[a]=np.stack(matrices);times[a]=np.stack(times[a])
    fraction=float(meta['target_fraction']);sustain=int(meta['sustain_checkpoints']);fixed,dual=reference[:,:,2].mean(axis=0)
    if not 0<fraction<1 or sustain<1 or sustain>len(steps):raise ValueError('Frozen target or sustain rule invalid for available checkpoints')
    if dual<=fixed:raise ValueError('Frozen relative target undefined: astronomical reference does not exceed fixed mounting. Use a separately preregistered absolute target protocol, not a post-hoc choice.')
    target=float(fixed+fraction*(dual-fixed))
    return dict(seeds=seeds,days=days,steps=steps,updates=updates,matrix=mat,times=times,target=target,fraction=fraction,sustain=sustain,fixed=float(fixed),dual=float(dual),sources=sources,protocol=meta['scenario_protocol_sha256'],mdp=meta['resolved_mdp_sha256'],horizon=int(meta['horizon']),replay=replay)


def compute(d):
    steps=np.asarray(d['steps']);seeds=list(d['seeds']);target=float(d['target']);sustain=int(d['sustain']);B=int(steps[-1]);S=len(seeds)
    if S<2 or len(set(seeds))!=S or len(steps)<2 or np.any(np.diff(steps)<=0) or not 1<=sustain<=len(steps):raise ValueError('Invalid learning grid')
    matrices={a:finite(d['matrix'][a]) for a in ALGORITHMS}
    expected=(S,len(d['days']),len(steps))
    if any(x.shape!=expected for x in matrices.values()):raise ValueError('Seed/day/checkpoint shape mismatch')
    curves={a:x.mean(axis=1) for a,x in matrices.items()};
    integrate=np.trapezoid if hasattr(np,'trapezoid') else np.trapz
    rows=[];pairs=[];attainment=[];curve_rows=[];auc={};lookup={};time_data={}
    for a in ALGORITHMS:
        y=curves[a];auc[a]=integrate(y,steps,axis=-1)/(steps[-1]-steps[0]);t=finite(d['times'][a]);
        if t.shape!=(S,len(steps),3) or (t<0).any() or (np.diff(t,axis=1)<-1e-10).any():raise ValueError('Invalid cumulative time array')
        same(t[:,:,0]+t[:,:,1],t[:,:,2],'Logged training plus evaluation time');time_data[a]=t
        for i,s in enumerate(seeds):
            onset,confirmed=event(y[i],steps,target,sustain);j=int(np.searchsorted(steps,confirmed)) if confirmed is not None else None
            row=dict(algorithm=a,seed=s,onset_steps=onset,confirmed_steps=confirmed,confirmed=confirmed is not None,last_observed_steps=B,restricted_steps=confirmed if confirmed is not None else B,auc_mean_net_kwh=float(auc[a][i]),final_net_kwh=float(y[i,-1]),logged_training_seconds=float(t[i,-1,0]),logged_evaluation_seconds=float(t[i,-1,1]),logged_total_seconds=float(t[i,-1,2]),logged_seconds_to_confirmation=float(t[i,j,2]) if j is not None else None,periodic_validation_transitions_to_confirmation=(j+1)*len(d['days'])*int(d['horizon']) if j is not None else None,periodic_validation_transitions_at_budget=len(steps)*len(d['days'])*int(d['horizon']))
            rows.append(row);lookup[a,s]=row
        events=[lookup[a,s]['confirmed_steps'] for s in seeds]
        for j,k in enumerate(steps):
            n=sum(e is not None and e<=k for e in events);attainment.append(dict(algorithm=a,env_steps=int(k),confirmed_seeds=n,n_seeds=S,fraction_confirmed=n/S))
    for i,s in enumerate(seeds):
        l,r=lookup['sac',s],lookup['mbpo',s];ns,nm=l['confirmed_steps'],r['confirmed_steps'];both=ns is not None and nm is not None
        pairs.append(dict(seed=s,status='both' if both else 'sac_only' if ns is not None else 'mbpo_only' if nm is not None else 'neither',sac_confirmed_steps=ns,mbpo_confirmed_steps=nm,conditional_reduction_percent=100*(1-nm/ns) if both else None,conditional_speedup=ns/nm if both else None,restricted_steps_saved=l['restricted_steps']-r['restricted_steps'],auc_gain_kwh=float(auc['mbpo'][i]-auc['sac'][i]),final_gain_kwh=r['final_net_kwh']-l['final_net_kwh'],logged_budget_seconds_difference=r['logged_total_seconds']-l['logged_total_seconds'],conditional_logged_time_reduction_percent=100*(1-r['logged_seconds_to_confirmation']/l['logged_seconds_to_confirmation']) if both and l['logged_seconds_to_confirmation']>0 else None))
    for a,x in {**matrices,'mbpo_minus_sac':matrices['mbpo']-matrices['sac']}.items():
        sm=x.mean(axis=1);dm=x.mean(axis=0)
        for j,k in enumerate(steps):curve_rows.append(dict(quantity=a,env_steps=int(k),mean=float(sm[:,j].mean()),sd_seed_means=float(sm[:,j].std(ddof=1)),sd_day_means=float(dm[:,j].std(ddof=1)),n_seeds=S,n_days=len(d['days'])))
    both=[r for r in pairs if r['status']=='both'];sr=[r['restricted_steps'] for r in rows if r['algorithm']=='sac'];mr=[r['restricted_steps'] for r in rows if r['algorithm']=='mbpo'];counts={k:sum(p['status']==k for p in pairs) for k in ('both','sac_only','mbpo_only','neither')}
    rel=[r['conditional_reduction_percent'] for r in both];summary_dict=dict(target_kwh_per_episode=target,target_fraction=d['fraction'],sustain_checkpoints=sustain,first_evaluated_steps=int(steps[0]),common_budget_steps=B,n_paired_seeds=S,n_validation_days=len(d['days']),fixed_mean_kwh=d['fixed'],astronomical_mean_kwh=d['dual'],confirmation_status_counts=counts,confirmed_count={a:sum(lookup[a,s]['confirmed'] for s in seeds) for a in ALGORITHMS},conditional_sample_reduction_percent=summary(rel) if rel else None,conditional_speedup=summary([p['conditional_speedup'] for p in both]) if both else None,restricted_steps={a:summary(v) for a,v in [('sac',sr),('mbpo',mr)]},restricted_budget_relative_saving_percent=float(100*(1-np.mean(mr)/np.mean(sr))),restricted_steps_saved=summary([p['restricted_steps_saved'] for p in pairs]),paired_auc_gain_kwh=summary([p['auc_gain_kwh'] for p in pairs]),paired_final_gain_kwh=summary([p['final_gain_kwh'] for p in pairs]),paired_logged_budget_seconds_difference=summary([p['logged_budget_seconds_difference'] for p in pairs]),conditional_logged_time_reduction_percent=summary([p['conditional_logged_time_reduction_percent'] for p in both if p['conditional_logged_time_reduction_percent'] is not None]) if any(p['conditional_logged_time_reduction_percent'] is not None for p in both) else None,definitions={'confirmation':'End of first run of sustain consecutive scheduled validation evaluations >= frozen target. Onset is retrospective only.','conditional_reduction':'Mean of paired 100*(1-N_MBPO/N_SAC) among seeds where BOTH algorithms confirmed. Not an unconditional estimate.','restricted_budget':'Mean min(confirmation step, final common budget), including every seed. Not an estimate of the eventual reaching time; failures at B and confirmations at B have same cost but different attainment indicator.','auc':'Trapezoid mean validation net energy over first-to-last common scheduled checkpoints; no extrapolation to zero.','uncertainty':'Sample SD across independent paired seed means, conditional on same frozen validation days and realization 0. Descriptive, not confidence intervals.','sample_accounting':'Real training transitions only in sample-reduction metrics. Periodic validation interactions are exported separately; one-time reference construction excluded.', 'time':'Cumulative logged episode and periodic evaluation seconds, not physical data acquisition time; excludes unlogged setup/checkpoint overhead and requires comparable hardware for speed claims.'})
    return dict(rows=rows,pairs=pairs,attainment=attainment,curves=curve_rows,report=summary_dict,seed_curves=curves,times=time_data)


def render_learning(d,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);r=compute(d);rep=r['report'];seeds=d['seeds'];x=np.asarray(d['steps']);scale=1000.;xx=x/scale;df=pd.DataFrame(r['rows']);pr=pd.DataFrame(r['pairs']);c=pd.DataFrame(r['curves']);at=pd.DataFrame(r['attainment'])
    for file,rows in [('learning_curve.csv',r['curves']),('sample_complexity_by_seed.csv',r['rows']),('paired_sample_reduction.csv',r['pairs']),('target_attainment.csv',r['attainment'])]:csv_out(out/file,rows)
    if d.get('replay'):
        replay=pd.DataFrame(d['replay']);replay.to_csv(out/'mbpo_replay_accounting.csv',index=False,float_format='%.12g');final=replay[replay.env_steps==rep['common_budget_steps']]
        rep['mbpo_replay_at_final_budget']=final.astype(object).where(pd.notna(final),None).to_dict(orient='records')
        rep['definitions']['replay']='Accepted synthetic transitions are generated records; samples used are minibatch draws with reuse. Neither is an external training transition.'
    write_json(out/'sample_efficiency_summary.json',rep)
    # Primary figure: separate performance and paired effect, exact sampled checkpoints.
    fig,axes=plt.subplots(2,1,figsize=(7.4,6.0),sharex=True,layout='constrained',height_ratios=[1.45,1])
    for a in ALGORITHMS:
        g=c[c.quantity==a];mu=g['mean'].to_numpy();sd=g.sd_seed_means.to_numpy();axes[0].plot(xx,mu,'o-',ms=2.5,lw=1.5,color=COLORS[a],label=a.upper());axes[0].fill_between(xx,mu-sd,mu+sd,color=COLORS[a],alpha=.15,linewidth=0)
    axes[0].axhline(d['target'],color='0.25',ls='--',lw=1,label='Frozen target');axes[0].axhline(d['fixed'],color='0.55',ls=':',lw=.9,label='Fixed reference');axes[0].axhline(d['dual'],color='0.35',ls='-.',lw=.9,label='Astronomical reference');axes[0].set_ylabel('Validation net energy\n(kWh / episode)');axes[0].legend(fontsize=8,ncol=2,loc='best');axes[0].set_title('(a) Learning performance — mean ± seed SD',loc='left',fontsize=11)
    g=c[c.quantity=='mbpo_minus_sac'];mu=g['mean'].to_numpy();sd=g.sd_seed_means.to_numpy();axes[1].plot(xx,mu,'o-',ms=2.5,color=COLORS['mbpo']);axes[1].fill_between(xx,mu-sd,mu+sd,color=COLORS['mbpo'],alpha=.15);axes[1].axhline(0,color='0.25',ls='--',lw=.9);axes[1].set(xlabel='Real PV simulator transitions (×10³)',ylabel='MBPO − SAC\n(kWh / episode)');axes[1].set_title('(b) Paired performance difference — mean ± seed SD',loc='left',fontsize=11)
    for ax in axes:style(ax)
    save(fig,out,'L01_learning_and_paired_gain')
    # Target attainment: all seeds shown; no invented reaching time.
    fig,axes=plt.subplots(1,2,figsize=(9.4,3.9),layout='constrained')
    for a in ALGORITHMS:
        g=at[at.algorithm==a];axes[0].step(np.r_[0,xx],np.r_[0,g.fraction_confirmed.to_numpy()],where='post',lw=1.8,color=COLORS[a],label=f'{a.upper()} ({rep["confirmed_count"][a]}/{len(seeds)})')
    axes[0].set(xlabel='Real transitions (×10³)',ylabel='Fraction of seeds confirmed',ylim=(-.03,1.05),xlim=(0,xx[-1]));axes[0].legend(fontsize=8);axes[0].set_title('(a) Target attainment',loc='left',fontsize=11)
    for i,s in enumerate(seeds):
        vals=[next(t for t in r['rows'] if t['seed']==s and t['algorithm']==a) for a in ALGORITHMS]
        axes[1].plot([t['restricted_steps']/scale for t in vals],[i-.12,i+.12],color='0.75',lw=.8,zorder=1)
        for a,t,off in zip(ALGORITHMS,vals,[-.12,.12]):
            axes[1].scatter(t['restricted_steps']/scale,i+off,s=35,marker='o' if t['confirmed'] else '>',facecolors=COLORS[a] if t['confirmed'] else 'none',edgecolors=COLORS[a],zorder=3)
    from matplotlib.lines import Line2D
    axes[1].legend(handles=[Line2D([],[],marker='o',ls='none',color=COLORS[a],label=a.upper()) for a in ALGORITHMS]+[Line2D([],[],marker='>',ls='none',markerfacecolor='none',color='.3',label='Not confirmed at budget')],fontsize=7,loc='best');axes[1].set(yticks=range(len(seeds)),yticklabels=[str(s) for s in seeds],xlabel='Real transitions (×10³)',ylabel='Training seed',xlim=(0,xx[-1]*1.06));axes[1].axvline(xx[-1],color='.5',ls=':',lw=.8);axes[1].set_title('(b) Per-seed confirmed sample requirement',loc='left',fontsize=11)
    for ax in axes:style(ax)
    save(fig,out,'L02_attainment_and_censoring')
    # Explicit reduction conditional on both reaching, plus unconditional restricted-cost difference.
    fig,axes=plt.subplots(1,2,figsize=(9.4,4.0),layout='constrained');n=len(seeds);status_y=np.arange(n);valid=pr.status=='both';
    if valid.any():axes[0].scatter(pr.loc[valid,'conditional_reduction_percent'],status_y[valid.to_numpy()],color=COLORS['mbpo'],s=36)
    for i,p in enumerate(r['pairs']):
        if p['status']!='both':axes[0].text(.03,i,p['status'].replace('_',' '),transform=axes[0].get_yaxis_transform(),va='center',fontsize=8,color='.45')
    axes[0].axvline(0,color='.35',ls='--',lw=.8);axes[0].set(yticks=status_y,yticklabels=seeds,ylim=(-.6,n-.4),xlabel='Sample reduction (%)',ylabel='Training seed');axes[0].set_title(f'(a) Conditional reduction ({int(valid.sum())}/{n} pairs)',loc='left',fontsize=11)
    vals=pr.restricted_steps_saved.to_numpy()/scale;axes[1].barh(status_y,vals,color=[COLORS['mbpo'] if v>=0 else COLORS['sac'] for v in vals],height=.55);axes[1].axvline(0,color='.35',lw=.8);axes[1].set(yticks=status_y,yticklabels=seeds,ylim=(-.6,n-.4),xlabel='Restricted steps saved (×10³)');axes[1].set_title('(b) All paired seeds, fixed budget',loc='left',fontsize=11)
    for ax in axes:style(ax)
    save(fig,out,'L03_sample_reduction')
    # AUC and final performance are paired, distinct from sample reduction.
    fig,axes=plt.subplots(1,2,figsize=(8.8,3.6),layout='constrained')
    for ax,field,label,title in [(axes[0],'auc_mean_net_kwh','Normalized AUC (kWh / episode)','(a) Common-budget learning quality'),(axes[1],'final_net_kwh','Final validation net energy (kWh / episode)','(b) Final performance')]:
        q=df.pivot(index='seed',columns='algorithm',values=field).loc[seeds,list(ALGORITHMS)];
        for _,row in q.iterrows():ax.plot([0,1],row.to_numpy(),'-o',color='.55',alpha=.65,ms=3,lw=.9)
        mu=q.mean();sd=q.std(ddof=1);ax.errorbar([0,1],mu,yerr=sd,fmt='D',color='k',capsize=4,label='Mean ± seed SD');ax.set(xticks=[0,1],xticklabels=['SAC','MBPO'],ylabel=label);ax.set_title(title,loc='left',fontsize=10);style(ax)
    axes[1].legend(fontsize=8);save(fig,out,'L04_auc_and_final_performance')
    # Compute time at common budget, and time at confirmation only for pairs reaching.
    fig,axes=plt.subplots(1,2,figsize=(8.8,3.8),layout='constrained')
    for a in ALGORITHMS:
        t=r['times'][a][:,:,2]/60;axes[0].plot(xx,t.mean(0),color=COLORS[a],label=a.upper());sd=t.std(0,ddof=1);axes[0].fill_between(xx,t.mean(0)-sd,t.mean(0)+sd,color=COLORS[a],alpha=.15)
    axes[0].set(xlabel='Real transitions (×10³)',ylabel='Logged time (min)');axes[0].legend(fontsize=8);axes[0].set_title('(a) Compute cost at equal sample budgets',loc='left',fontsize=10)
    q=df.pivot(index='seed',columns='algorithm',values='logged_seconds_to_confirmation').reindex(seeds);ok=q.notna().all(axis=1)
    if ok.any():
        for _,row in q[ok].iterrows():axes[1].plot([0,1],row[list(ALGORITHMS)].to_numpy(dtype=float)/60,'-o',color='.5',ms=4)
    else:axes[1].text(.5,.5,'No pairs with confirmed target',ha='center',va='center',transform=axes[1].transAxes,fontsize=9)
    axes[1].set(xticks=[0,1],xticklabels=['SAC','MBPO'],ylabel='Logged minutes to confirmation');axes[1].set_title(f'(b) Conditional time ({int(ok.sum())}/{n} pairs)',loc='left',fontsize=10)
    for ax in axes:style(ax)
    save(fig,out,'L05_compute_cost')
    # Time trajectories and manifest for reproducibility.
    csv_out(out/'logged_time_by_checkpoint.csv',[dict(algorithm=a,seed=s,env_steps=int(k),gradient_updates=int(d['updates'][j]),training_seconds=float(r['times'][a][i,j,0]),evaluation_seconds=float(r['times'][a][i,j,1]),total_seconds=float(r['times'][a][i,j,2])) for a in ALGORITHMS for i,s in enumerate(seeds) for j,k in enumerate(x)])
    write_json(out/'figure_manifest.json',[{'figure':name,'meaning':meaning} for name,meaning in [('L01_learning_and_paired_gain','Fixed validation realization 0; paired mean ± sample SD across seed means'),('L02_attainment_and_censoring','Confirmation occurs at the final checkpoint of the sustained sequence; open arrows are not confirmed'),('L03_sample_reduction','Conditional percent reduction versus all-seed restricted cost'),('L04_auc_and_final_performance','Paired seed results at common measured budget'),('L05_compute_cost','Logged times, not physical sample acquisition time')]])
    return rep
