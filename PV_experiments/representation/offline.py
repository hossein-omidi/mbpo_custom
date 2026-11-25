"""Use verified managed evaluations; never reconstruct missing paired covariance."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .statistics import METHODS,PAIRS,LABELS,COLORS,equal_mixture,from_day_moments,clock_grid,check_close,season,finite,daily_spread
from .common import Figures,style,line,save_csv,sha,read_json,write_json,plt


def load_mc(cfg,seeds,stage,split,root):
    from PV_experiments.evaluate import evaluation_output,existing_report_is_valid,_provenance,_require_matching_provenance
    from PV_experiments.report import validate_mc_source
    days=sorted(split[f'{stage}_ids']); M=None; daily=[]; moments=[]; fingerprints={}; sources={}
    for seed in seeds:
        folder=evaluation_output(cfg,stage,seed,root)
        summary=read_json(folder/'mc_summary.json'); m=int(summary['draws_per_day'])
        if M is not None and M!=m:raise ValueError('Different draw counts across seeds')
        M=m
        if not existing_report_is_valid(cfg,stage,seed,M,folder,root):raise ValueError('Incomplete managed evaluation')
        _require_matching_provenance(folder,_provenance(cfg,stage,seed,root,draws=M))
        _, verified=validate_mc_source(cfg,stage,seed,root)
        if fingerprints and fingerprints!=verified['_validated_weather_hashes']:raise ValueError('Weather draw fingerprints differ across seeds')
        fingerprints=verified['_validated_weather_hashes']
        d=pd.read_csv(folder/'all_day_draw_method.csv',dtype={'scenario_id':str,'profile_sha256':str})
        p=pd.read_csv(folder/'daily_power_mean_std_by_utc_step.csv',dtype={'scenario_id':str})
        idx=pd.MultiIndex.from_product([days,range(M),METHODS],names=['scenario_id','draw','method'])
        d=d.set_index(['scenario_id','draw','method'],verify_integrity=True)
        if len(d)!=len(idx) or not d.index.isin(idx).all():raise ValueError('Daily records do not cover frozen grid')
        d=d.reindex(idx)
        for field in ('gross_kwh','movement_kwh','net_kwh'):finite(d[field])
        check_close(d.gross_kwh-d.movement_kwh,d.net_kwh,'Daily gross - actuator = net')
        if (d[['gross_kwh','movement_kwh']]< -1e-10).any().any():raise ValueError('Negative gross energy or movement cost')
        check_close(d.xs('fixed_no_motion',level='method').movement_kwh,np.zeros(len(days)*M),'Fixed movement cost')
        pi=pd.MultiIndex.from_product([days,METHODS,range(1,196)],names=['scenario_id','method','step'])
        p=p.set_index(['scenario_id','method','step'],verify_integrity=True)
        if len(p)!=len(pi) or not p.index.isin(pi).all():raise ValueError('Incomplete or extra power trajectory moments')
        p=p.reindex(pi)
        # The index was explicitly reindexed above; vectorized reshape preserves that order.
        raw_times=p['time_utc'].to_numpy().reshape(len(days),len(METHODS),195)
        for di,sid in enumerate(days):
            expected=clock_grid(sid)
            for mi,method in enumerate(METHODS):
                timestamps=pd.to_datetime(raw_times[di,mi],utc=False)
                if any(t.utcoffset().total_seconds()!=0 for t in timestamps) or not pd.DatetimeIndex(timestamps).equals(expected):raise ValueError(f'UTC grid mismatch {sid}/{method}')
        for field in ('power_mean_w','power_sd_w'):finite(p[field])
        if (p[['power_mean_w','power_sd_w']]<0).any().any():raise ValueError('Negative power mean or SD')
        values=p[['power_mean_w','power_sd_w']].to_numpy().reshape(len(days),len(METHODS),195,2)
        dmat=d[['gross_kwh','movement_kwh','net_kwh']].to_numpy().reshape(len(days),M,len(METHODS),3)
        check_close(values[:,:,:,0].sum(axis=2)/12000,dmat[:,:,:,0].mean(axis=1),'Power integral versus gross energy')
        if daily:
            check_close(dmat[:,:,:2,:],daily[0][:,:,:2,:],'Deterministic baseline daily metrics across seeds')
            check_close(values[:,:2],moments[0][:,:2],'Deterministic baseline power moments across seeds')
        # Periodic evaluation and MC draw zero use exactly the same weather and initial pose.
        # Cross-check the two independent evaluation entry points when both cover validation.
        if stage=='validation':
            from PV_experiments.core import run_path
            for alg in ('sac','mbpo'):
                rf=run_path(cfg,alg,seed,root)/'periodic_evaluation'/'evaluation_references.csv'
                refs=pd.read_csv(rf,dtype={'scenario_id':str}).set_index('scenario_id',verify_integrity=True)
                if set(refs.index)!=set(days) or not (refs.episode_steps==195).all():raise ValueError('Periodic baseline scenario/horizon mismatch')
                refs=refs.loc[days]
                for mi,prefix in [(0,'fixed'),(1,'dual_axis')]:
                    a=refs[[f'{prefix}_gross_energy_kwh',f'{prefix}_movement_cost_kwh',f'{prefix}_net_energy_kwh']].to_numpy()
                    check_close(a,dmat[:,0,mi,:],'Periodic reference versus MC draw zero')
                sources[f'{alg}_seed{seed}/evaluation_references.csv']=sha(rf)
        daily.append(dmat);moments.append(values)
        for f in ('all_day_draw_method.csv','daily_power_mean_std_by_utc_step.csv','mc_summary.json','manager_provenance.json'):
            sources[f'seed{seed}/{f}']=sha(folder/f)
        for alg in ('sac','mbpo'):
            from PV_experiments.core import run_path
            sources[f'{alg}_seed{seed}/latest.pt']=sha(run_path(cfg,alg,seed,root)/'latest.pt')
    return dict(days=days,draws=M,seeds=seeds,daily=np.asarray(daily),moments=np.asarray(moments),fingerprints=fingerprints,sources=sources)


def daily_statistics(data,group,indices):
    x=data['daily'][:,indices]; rows=[]
    for mi,m in enumerate(METHODS):
        for fi,field in enumerate(('gross_kwh','movement_kwh','net_kwh')):
            values=x[:,:,:,mi,fi]; z=equal_mixture(values); rows.append(dict(group=group,method=m,quantity=field,n_days=len(indices),n_seeds=len(x),n_draws=x.shape[2],**{k:float(v) for k,v in z.items()},**daily_spread(values)))
    for a,b in PAIRS:
        values=x[:,:,:,METHODS.index(a),2]-x[:,:,:,METHODS.index(b),2]; z=equal_mixture(values);rows.append(dict(group=group,method=f'{a}_minus_{b}',quantity='paired_net_kwh',n_days=len(indices),n_seeds=len(x),n_draws=x.shape[2],**{k:float(v) for k,v in z.items()},**daily_spread(values)))
    return rows


def render_evaluation(data,out,stage):
    figset=Figures(out); x=data['daily']; S,D,M=x.shape[:3]; dates=pd.DatetimeIndex(data['days']); allidx=np.arange(D)
    groups={'overall':[('all',allidx)],'month':[(str(m),np.where(dates.month==m)[0]) for m in sorted(set(dates.month))],
        'season':[(s,np.where(np.array([season(m) for m in dates.month])==s)[0]) for s in ('DJF','MAM','JJA','SON')],
        'year':[(str(y),np.where(dates.year==y)[0]) for y in sorted(set(dates.year))]}
    tables={}
    for kind,gs in groups.items():
        tables[kind]=pd.DataFrame([r for label,idx in gs if len(idx) for r in daily_statistics(data,label,idx)])
        save_csv(out,f'{kind}_energy_statistics.csv',tables[kind])
    overall=tables['overall']; pos=np.arange(len(METHODS));fig,axes=plt.subplots(1,2,figsize=(11,4.4))
    for ax,fields,title in [(axes[0],('gross_kwh','net_kwh'),'Gross and net DC energy'),(axes[1],('movement_kwh',),'Actuator energy')]:
        for i,field in enumerate(fields):
            t=overall[overall.quantity==field].set_index('method').loc[list(METHODS)]; offsets=(i-(len(fields)-1)/2)*.35
            ax.bar(pos+offsets,t['mean'],width=.34 if len(fields)>1 else .55,label=field.replace('_kwh','').title(),alpha=.8)
            if S>1:ax.errorbar(pos+offsets,t['mean'],yerr=t.sd_seed_means,fmt='none',color='black',capsize=3,lw=1)
        ax.set(xticks=pos,xticklabels=['Fixed','Astronomical','SAC','MBPO'],ylabel='kWh / evaluated day',title=title);ax.legend();style(ax)
    fig.suptitle('Means ± sample SD of seed means' if S>1 else 'Means — one seed; no seed SD');figset.save(fig,'E01_energy_and_effort','Does gross improvement survive actuator consumption?','Equal held-out days and draws; error bars ± sample SD of seed means, omitted for one seed.','all_day_draw_method.csv')
    for kind in ('month','season','year'):
        t=tables[kind]; labels=[k for k,_ in groups[kind] if len(_)]; fig,axes=plt.subplots(2,1,figsize=(9,7),sharex=True)
        for ax,field in zip(axes,('net_kwh','movement_kwh')):
            for m in METHODS:
                a=t[(t.quantity==field)&(t.method==m)].set_index('group').loc[labels]
                ax.errorbar(np.arange(len(labels))+(METHODS.index(m)-1.5)*.10,a['mean'].to_numpy(),yerr=a.sd_seed_means.to_numpy() if S>1 else None,fmt='o',capsize=2,color=COLORS[m],label=LABELS[m])
            ax.set_ylabel(field.replace('_kwh','').title()+' (kWh / day)');style(ax)
        axes[0].legend(ncol=2,fontsize=8);axes[1].set_xticks(np.arange(len(labels)),[f'{k}\n(n={len(idx)})' for k,idx in groups[kind] if len(idx)]);axes[1].set_xlabel({'month':'Calendar month','season':'Meteorological season','year':'Source year'}[kind]);fig.suptitle('Mean per evaluated day ± sample SD of seed means' if S>1 else 'Mean per evaluated day — one seed');figset.save(fig,f'E02_{kind}_net_and_effort','Does the controller advantage depend on calendar conditions?','Average energy per selected historical day; seed-mean SD; not annual totals or full-population estimates.',f'{kind}_energy_statistics.csv')
    # Separate empirical operating/policy mixture spread from sample seed SD.
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    t=overall[overall.quantity=='net_kwh'].set_index('method').loc[list(METHODS)]
    axes[0].errorbar(pos,t['mean'],yerr=t.sd_total,fmt='o',color='black',capsize=4)
    axes[0].set(xticks=pos,xticklabels=['Fixed','Astronomical','SAC','MBPO'],ylabel='Net energy (kWh / evaluated day)',title='Mean ± empirical mixture SD')
    pairs=[('mbpo','sac'),('mbpo','sun_slew_limited'),('mbpo','fixed_no_motion')]
    for i,(a,b) in enumerate(pairs):
        z=x[:,:,:,METHODS.index(a),2]-x[:,:,:,METHODS.index(b),2]
        seed_values=z.mean(axis=(1,2)); sd=seed_values.std(ddof=1) if S>1 else None
        axes[1].errorbar(i,seed_values.mean(),yerr=sd,fmt='o',color='black',capsize=4)
        axes[1].scatter(np.full(S,i),seed_values,s=12,alpha=.45,color=COLORS['mbpo'])
    axes[1].axhline(0,color='k',lw=.8);axes[1].set(xticks=range(3),xticklabels=['vs SAC','vs astronomical','vs fixed'],ylabel='MBPO − comparator (kWh / day)',title='Paired seed means' + (' ± sample SD' if S>1 else ' (one seed)'))
    for ax in axes:style(ax)
    figset.save(fig,'E06_energy_spread_and_paired_gain','What is the descriptive spread and paired net benefit?','Left: empirical population SD of the seed/day/weather mixture, not a CI. Right: paired seed means and sample seed SD (omitted for one seed). Baselines vary by day/weather, not training seed.','overall_energy_statistics.csv and relative_gain_by_seed.csv')
    # Calendar months stay at their actual positions, including gaps.
    fig,ax=plt.subplots(figsize=(9,4.5)); t=tables['month']; labels=[k for k,_ in groups['month']]; months=np.array([int(k) for k in labels])
    for i,(a,b) in enumerate(pairs):
        q=t[(t.quantity=='paired_net_kwh')&(t.method==f'{a}_minus_{b}')].set_index('group').loc[labels]
        ax.errorbar(months+(i-1)*.14,q['mean'],yerr=q.sd_seed_means if S>1 else None,fmt='o',capsize=3,label=f'MBPO − {LABELS[b]}',color=COLORS[b])
    counts={int(k):len(idx) for k,idx in groups['month']}
    ax.axhline(0,color='k',lw=.8);ax.set(xticks=range(1,13),xticklabels=[f'{k}\n(n={counts[k]})' if k in counts else str(k) for k in range(1,13)],xlabel='Calendar month (n = evaluated historical days)',ylabel='Paired net difference (kWh / day)',title='Mean ± paired seed SD' if S>1 else 'Paired mean — one seed');ax.legend(fontsize=8);style(ax)
    figset.save(fig,'E07_monthly_paired_gain','How does paired net benefit vary by calendar month?','Pair before averaging; equal selected days/draws; sample SD of paired seed means, not a CI. Unevaluated months have no points.','month_energy_statistics.csv')
    # Pair before averaging. Each point is a historical day, not M independent days.
    fig,axes=plt.subplots(1,2,figsize=(11,4.4)); paired=[]
    for a,b in PAIRS:
        z=x[:,:,:,METHODS.index(a),2]-x[:,:,:,METHODS.index(b),2]
        means=z.mean(axis=(0,2)); name=f'{a}_minus_{b}'
        paired.extend(dict(scenario_id=sid,pair=name,mean_net_difference_kwh=v) for sid,v in zip(data['days'],means))
        if a=='mbpo':
            label=f'{LABELS[a]} − {LABELS[b]}'
            axes[0].step(np.sort(means),np.arange(1,D+1)/D,where='post',label=label)
    axes[0].axvline(0,color='k',lw=.8);axes[0].set(xlabel='Paired mean daily net difference (kWh)',ylabel='Empirical CDF');axes[0].legend(fontsize=8);style(axes[0])
    for mi,m in enumerate(METHODS):
        axes[1].scatter(x[:,:,:,mi,1].mean(axis=(0,2)),x[:,:,:,mi,0].mean(axis=(0,2)),s=13,alpha=.45,label=LABELS[m],color=COLORS[m])
    axes[1].set(xlabel='Mean actuator energy (kWh / day)',ylabel='Mean gross energy (kWh / day)');axes[1].legend(fontsize=8);style(axes[1]);save_csv(out,'paired_daily_means.csv',paired)
    figset.save(fig,'E03_paired_distribution_and_effort','How consistent is improvement, and what does it cost?','One point per historical day after averaging matched draws and training seeds.','all_day_draw_method.csv')
    # Ratios of total equal-weight means, not means of unstable daily ratios.
    gains=[]
    for a,b in PAIRS:
        for fi,field in [(0,'gross_kwh'),(2,'net_kwh')]:
            av=x[:,:,:,METHODS.index(a),fi].mean(axis=(1,2));bv=x[:,:,:,METHODS.index(b),fi].mean(axis=(1,2))
            for seed,v,w in zip(data['seeds'],av,bv):gains.append(dict(seed=seed,comparison=f'{a}_minus_{b}',quantity=field,absolute_kwh=v-w,relative_percent=100*(v-w)/w if w>1e-12 else np.nan))
    save_csv(out,'relative_gain_by_seed.csv',gains)
    # Exact marginal power mixture from conditional moments, not fabricated paired SD.
    z=data['moments']; clock=8.75+np.arange(1,196)/12; rows=[];fig,axes=plt.subplots(2,1,figsize=(9,7),sharex=True)
    for mi,m in enumerate(METHODS):
        st=from_day_moments(z[:,:,mi,:,0],z[:,:,mi,:,1],M)
        rows.extend(dict(method=m,step=i+1,utc_hour_unwrapped=clock[i],**{k:v[i] for k,v in st.items()}) for i in range(195))
        line(axes[0],clock,st['mean'],m,st['sd_total']);line(axes[1],clock,st['mean'],m,st['sd_seed_means'] if S>1 else None)
    for ax,title in zip(axes,('Descriptive spread over seed × day × weather','Between-training-seed SD of mean profiles' if S>1 else 'Single training seed — no seed SD')):
        ax.set(ylabel='DC power (W)',title=title+' (mean ± 1 SD)' if S>1 or ax is axes[0] else title);style(ax)
    axes[0].legend(ncol=2,fontsize=8);axes[1].set_xlabel('UTC hour (unwrapped; 25 = 01:00 next day)');save_csv(out,'power_moments.csv',rows)
    figset.save(fig,'E04_intraday_power','How do generation and its variability change within the day?','Two distinct bands: empirical mixture SD and sample SD of seed-level mean profiles; neither is a CI.','daily_power_mean_std_by_utc_step.csv')
    from .trajectories import representative_days
    chosen=representative_days(data['days']);fig,axes=plt.subplots(len(chosen),1,figsize=(9,3*len(chosen)),sharex=True,squeeze=False)
    for ax,sid in zip(axes[:,0],chosen):
        di=data['days'].index(sid)
        for mi,method in enumerate(METHODS):line(ax,clock,z[0,di,mi,:,0],method,z[0,di,mi,:,1])
        ax.set(title=f'{sid} — seed {data["seeds"][0]}, mean ± weather-draw SD',ylabel='DC power (W)');style(ax)
    axes[0,0].legend(ncol=2,fontsize=7);axes[-1,0].set_xlabel('UTC hour (unwrapped)');figset.save(fig,'E05_seasonal_example_power','How variable is the response to uncertain weather on individual seasonal days?','First selected training seed; mean ± sample SD across the matched weather draws on each outcome-blind representative day.','daily_power_mean_std_by_utc_step.csv')
    figset.finish();write_json(out/'evaluation_scope.json',dict(stage=stage,days=data['days'],draws=M,seeds=data['seeds'],note='Equal-weight frozen held-out days. No annual yield extrapolation. Seed/day/draw spread is descriptive, not inferential.'))
