#!/usr/bin/env python3
"""In-place representation-only update for 355f054; no training/evaluation edits.
Use --check first, then --apply. All source bytes are pinned before any writes.
"""
import argparse
import difflib
import hashlib
from pathlib import Path
import subprocess

BASE = '355f054fea41d3f137f658c14086ba998fc41803'
PREFIX = 'PV_experiments/representation/'
PINS = {
    'statistics.py': '434e056863c2adc5c10bb4b60c7be706afc8889d',
    'offline.py': '8f9fc753e901226051fbd7c5d166451f0a128db4',
    'training.py': '55ed6d995dfc54bd144f7339cabfb8a25a6ea44d',
    'trajectories.py': '4cc8f85a079c6a5a3a910566f824735a33aeb1fe',
}

def replace(s, old, new):
    if s.count(old) != 1:
        raise ValueError(f'Expected exactly one source anchor: {old[:100]!r}; got {s.count(old)}')
    return s.replace(old, new, 1)


def statistics(s):
    s = replace(s, '    return mean,r\n', '    # Floating-point modulo can round an infinitesimal negative angle to 360.\n    mean=np.where(mean==360.,0.,mean)\n    return mean,r\n')
    s = replace(s, "    if not np.allclose(finite(a),finite(b),atol=atol,rtol=rtol):raise ValueError(f'{what}: numerical mismatch')", "    a,b=finite(a),finite(b)\n    if a.shape!=b.shape:raise ValueError(f'{what}: shape mismatch {a.shape} != {b.shape}')\n    if not np.allclose(a,b,atol=atol,rtol=rtol):raise ValueError(f'{what}: numerical mismatch')")
    s += '''\n\ndef daily_spread(x):
    """Scalar energy array [seed, day, draw]. Distinct descriptive estimands.

    Historical-day sample SD uses marginal day means (averaged over seed/draw).
    RMS weather SD uses conditional sample variances (within seed/day).
    Neither is an independent random-effects estimate or a CI.
    """
    x=finite(x)
    if x.ndim!=3 or x.shape[2]<2:raise ValueError('Expected seed/day/draw energy array with >=2 draws')
    day=x.mean(axis=(0,2))
    return dict(sd_day_means=float(day.std(ddof=1)) if len(day)>1 else np.nan,
                rms_weather_draw_sd=float(np.sqrt(np.mean(x.var(axis=2,ddof=1)))))


def trajectory_clock(quantity,horizon=195):
    """Unwrapped UTC hours. Commands at decision time; resulting quantities at response time."""
    decision=quantity in ('action_tilt','action_azimuth','abs_action_tilt','abs_action_azimuth')
    return 8.75+(np.arange(horizon)+(0 if decision else 1))/12., ('decision' if decision else 'response')


def cumulative_plot_series(clock,mean,sd):
    """Prepend deterministic zero ONLY for rendering integrated quantities."""
    clock,mean,sd=finite(clock),finite(mean),finite(sd)
    if clock.ndim!=1 or clock.shape!=mean.shape or clock.shape!=sd.shape:raise ValueError('Cumulative plot shape mismatch')
    return np.r_[clock[0]-1/12.,clock],np.r_[0.,mean],np.r_[0.,sd]
'''
    return s


def offline(s):
    s=replace(s,'clock_grid,check_close,season,finite\n','clock_grid,check_close,season,finite,daily_spread\n')
    s=replace(s,"            z=equal_mixture(x[:,:,:,mi,fi]); rows.append(dict(group=group,method=m,quantity=field,n_days=len(indices),n_seeds=len(x),n_draws=x.shape[2],**{k:float(v) for k,v in z.items()}))", "            values=x[:,:,:,mi,fi]; z=equal_mixture(values); rows.append(dict(group=group,method=m,quantity=field,n_days=len(indices),n_seeds=len(x),n_draws=x.shape[2],**{k:float(v) for k,v in z.items()},**daily_spread(values)))")
    s=replace(s,"        z=equal_mixture(x[:,:,:,METHODS.index(a),2]-x[:,:,:,METHODS.index(b),2]);rows.append(dict(group=group,method=f'{a}_minus_{b}',quantity='paired_net_kwh',n_days=len(indices),n_seeds=len(x),n_draws=x.shape[2],**{k:float(v) for k,v in z.items()}))", "        values=x[:,:,:,METHODS.index(a),2]-x[:,:,:,METHODS.index(b),2]; z=equal_mixture(values);rows.append(dict(group=group,method=f'{a}_minus_{b}',quantity='paired_net_kwh',n_days=len(indices),n_seeds=len(x),n_draws=x.shape[2],**{k:float(v) for k,v in z.items()},**daily_spread(values)))")
    s=replace(s,"axes[1].set_xticks(np.arange(len(labels)),labels);axes[1].set_xlabel", "axes[1].set_xticks(np.arange(len(labels)),[f'{k}\\n(n={len(idx)})' for k,idx in groups[kind] if len(idx)]);axes[1].set_xlabel")
    s=replace(s,"    # Pair before averaging. Each point is a historical day, not M independent days.\n", '''    # Separate empirical operating/policy mixture spread from sample seed SD.
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
    ax.axhline(0,color='k',lw=.8);ax.set(xticks=range(1,13),xticklabels=[f'{k}\\n(n={counts[k]})' if k in counts else str(k) for k in range(1,13)],xlabel='Calendar month (n = evaluated historical days)',ylabel='Paired net difference (kWh / day)',title='Mean ± paired seed SD' if S>1 else 'Paired mean — one seed');ax.legend(fontsize=8);style(ax)
    figset.save(fig,'E07_monthly_paired_gain','How does paired net benefit vary by calendar month?','Pair before averaging; equal selected days/draws; sample SD of paired seed means, not a CI. Unevaluated months have no points.','month_energy_statistics.csv')
    # Pair before averaging. Each point is a historical day, not M independent days.
''')
    return s


def training(s):
    s=replace(s,'figs=Figures(out); frames={}; sources={}\n','figs=Figures(out); frames={}; sources={}; warmups=set()\n')
    s=replace(s,"            frames[alg,seed]=f; sources[f'{alg}_seed{seed}/episodes.csv']=sha(p)","            config_path=p.parent/'config.json'; saved=read_json(config_path); shared=saved if alg=='sac' else saved['sac']; warmups.add(int(shared['algo']['start_steps']))\n            frames[alg,seed]=f; sources[f'{alg}_seed{seed}/episodes.csv']=sha(p);sources[f'{alg}_seed{seed}/config.json']=sha(config_path)\n    if len(warmups)!=1 or min(warmups)<0:raise ValueError('Inconsistent warm-up configuration')\n    warmup=next(iter(warmups))")
    s=replace(s,"        ax.set_ylabel(field.replace('_',' '));style(ax)\n", "        if 0<warmup<=steps[-1]:ax.axvline(warmup,color='k',ls=':',lw=.9,label=f'Warm-up end ({warmup:,})' if field=='return_kwh' else '_nolegend_')\n        ax.set_ylabel(field.replace('_',' '));style(ax)\n")
    s=replace(s,'Different sampled days; not policy evaluation.','Warm-up boundary from saved configuration; loss smoothing is an episode-level average, not update-weighted. Different sampled days; not policy evaluation.')
    return s


def trajectories(s):
    s=replace(s,'from_day_moments,circular_stats,season\n','from_day_moments,circular_stats,season,trajectory_clock,cumulative_plot_series\n')
    s=replace(s,"'action_azimuth':p[:,:,:,7]}\n", "'action_azimuth':p[:,:,:,7],'abs_action_tilt':np.abs(p[:,:,:,6]),'abs_action_azimuth':np.abs(p[:,:,:,7]),'abs_delta_tilt_deg':np.abs(p[:,:,:,8]),'abs_delta_azimuth_deg':np.abs(p[:,:,:,9])}\n")
    s=replace(s,"                    a=p[:,:,:,field].reshape(S*M,195);line(ax,clock,a.mean(axis=0),method,a.std(axis=0,ddof=0))", "                    a=p[:,:,:,field].reshape(S*M,195);line(ax,trajectory_clock('action_tilt')[0],a.mean(axis=0),method,a.std(axis=0,ddof=0))")
    s=replace(s,"axes[1].set(ylabel='Normalized azimuth command',xlabel='UTC hour (unwrapped)')", "axes[1].set(ylabel='Normalized azimuth command',xlabel='Decision UTC hour (unwrapped)')")
    s=replace(s,'Commands are increments, not angles.','Commands timestamped at decision time; normalized requested increments, not executed angles.')
    s=replace(s,"summary[(method,key)]=s;rows.extend(dict(method=method,quantity=key,step=i+1,utc_hour_unwrapped=clock[i],n_days=len(days),**{k:v[i] for k,v in s.items()}) for i in range(195))", "summary[(method,key)]=s; times,basis=trajectory_clock(key);rows.extend(dict(method=method,quantity=key,step=i+1,utc_hour_unwrapped=times[i],clock_basis=basis,n_days=len(days),**{k:v[i] for k,v in s.items()}) for i in range(195))")
    s=replace(s,"                z=summary[(name,key)];line(ax,clock,z['mean'],name,z['sd_total'])", "                z=summary[(name,key)]; tx,mu,sd=clock,z['mean'],z['sd_total']\n                if key.startswith('cumulative_'):tx,mu,sd=cumulative_plot_series(tx,mu,sd)\n                line(ax,tx,mu,name,sd)")
    s=replace(s,'pairing and path integration precede aggregation. No oracle net/cost.','pairing and path integration precede aggregation. Cumulative curves begin at deterministic zero (plot only). No oracle net/cost.')
    s=replace(s,"    fig,axes=plt.subplots(2,2,figsize=(10,7),sharex=True,sharey=True)\n", '''    # Magnitudes expose cancellation in signed command averages. Executed motion
    # is shown at the resulting response timestamp, not assumed equal to request.
    fig,axes=plt.subplots(2,2,figsize=(10,7),sharex=False)
    for ax,key,label in zip(axes.flat,('abs_action_tilt','abs_action_azimuth','abs_delta_tilt_deg','abs_delta_azimuth_deg'),('Absolute tilt command','Absolute azimuth command','Absolute executed tilt (deg)','Absolute executed azimuth (deg)')):
        tx,basis=trajectory_clock(key)
        for name in METHODS:
            z=summary[(name,key)];line(ax,tx,z['mean'],name,z['sd_total'])
        ax.set(ylabel=label,xlabel=f'{basis.capitalize()} UTC hour (unwrapped)');style(ax)
    axes[0,0].legend(ncol=2,fontsize=7);fig.suptitle(f'{scope.capitalize()} scope: {len(days)} days — mean ± empirical mixture SD')
    figs.save(fig,'R10_motion_magnitudes','How much motion is requested and actually executed at each step?','Absolute value before averaging; empirical mixture SD, not a CI. Commands at decision time, executed movements at response time.','rich_curve_statistics.csv')
    fig,axes=plt.subplots(2,2,figsize=(10,7),sharex=True,sharey=True)
''')
    return s

TRANSFORMS={'statistics.py':statistics,'offline.py':offline,'training.py':training,'trajectories.py':trajectories}

# TEST_PAYLOAD inserted below.
TEST_PATH='geometry10_tests/test_representation_final.py'
TEST_SOURCE=r'''"""Synthetic, analytical regression tests; no training or weather data required."""
import json
import numpy as np
import pandas as pd
import pytest
from PV_experiments.representation.statistics import (
    METHODS, circular_stats, check_close, daily_spread, trajectory_clock,
    cumulative_plot_series, equal_mixture, from_day_moments)


def test_strict_shape_and_circular_boundary():
    with pytest.raises(ValueError,match='shape mismatch'):
        check_close(np.ones((2,1)),np.ones(2),'broadcast trap')
    check_close(1.,1.,'scalar')
    mean,r=circular_stats([359.,1.])
    assert 0<=mean<360 and min(mean,360-mean)<1e-10 and r>.99
    mean,r=circular_stats([0.,180.]);assert np.isnan(mean) and r<1e-8


def test_spread_and_clocks():
    x=np.array([[[1.,3.],[5.,7.]],[[1.,3.],[5.,7.]]])
    s=equal_mixture(x);d=daily_spread(x)
    assert s['mean']==4 and s['sd_seed_means']==0
    np.testing.assert_allclose(s['sd_total'],np.sqrt(5))
    np.testing.assert_allclose(d['sd_day_means'],np.sqrt(8))
    np.testing.assert_allclose(d['rms_weather_draw_sd'],np.sqrt(2))
    assert np.isnan(daily_spread(x[:,:1])['sd_day_means'])
    assert np.isnan(equal_mixture(x[:1])['sd_seed_means'])
    dec,basis=trajectory_clock('abs_action_azimuth');resp,basis2=trajectory_clock('power_w')
    assert (basis,basis2)==('decision','response')
    np.testing.assert_allclose([dec[0],dec[-1],resp[0],resp[-1]],[8.75,24+55/60,8+50/60,25])
    np.testing.assert_allclose(resp-dec,1/12)
    t,mu,sd=cumulative_plot_series(resp,np.arange(195)+1,np.ones(195))
    assert len(t)==196 and t[0]==8.75 and mu[0]==sd[0]==0 and mu[-1]==195


def fixture(seeds):
    # Two distinct days in January and one in March; no February observation.
    days=['2020-01-01','2021-01-02','2021-03-03'];S=len(seeds);M=2
    raw=np.zeros((S,len(days),M,4,195,12));oracle=np.zeros((len(days),M,195,5));weather=np.zeros((len(days),M,195,8))
    for d in range(len(days)):
        for m in range(M):
            weather[d,m,:,1]=200+20*d;weather[d,m,:,2]=80
            oracle[d,m,:,0]=400;oracle[d,m,:,1]=400/12000
            for si in range(S):
                for k in range(4):
                    p=raw[si,d,m,k];p[:,0]=60+20*d+10*m+5*k+(3*si if k==2 else 5*si if k==3 else 0)
                    p[:,1]=p[:,0]/12000;p[:,2]=0 if k==0 else .00001*k
                    p[:,3]=p[:,1]-p[:,2];p[:,4]=30;p[:,5]=359 if m==0 else 1
                    if k>0:p[:,6]=(-1)**m;p[:,7]=(-1)**m*.5;p[:,8]=(-1)**m;p[:,9]=(-1)**m*2
    daily=raw[...,1:4].sum(axis=4)
    p=raw[...,0];moments=np.stack([p.mean(axis=2),p.std(axis=2,ddof=1)],axis=-1)
    return dict(seeds=seeds,days=days,draws=M,daily=daily,moments=moments,sources={}),raw,weather,oracle


def capture_figures(monkeypatch):
    # Exercise actual renderers and artists; save actual PNG/PDF only for new and
    # corrected panels, limiting test runtime. Preserve manifest for every panel.
    from PV_experiments.representation.common import Figures,plt
    original=Figures.save
    class Captures(dict):pass
    seen=Captures();seen.bars={};seen.errors={}
    def save(self,fig,name,question,estimand,source):
        from matplotlib.container import BarContainer, ErrorbarContainer
        seen[name]=[[(np.asarray(l.get_xdata()),np.asarray(l.get_ydata()),l.get_label()) for l in ax.lines] for ax in fig.axes]
        seen.bars[name]=[[[p.get_height() for p in c.patches] for c in ax.containers if isinstance(c,BarContainer)] for ax in fig.axes]
        seen.errors[name]=[[[seg for lc in c.lines[2] for seg in lc.get_segments()] for c in ax.containers if isinstance(c,ErrorbarContainer)] for ax in fig.axes]
        if name in ('E01_energy_and_effort','E06_energy_spread_and_paired_gain','E07_monthly_paired_gain','R03_cumulative_energy','R10_motion_magnitudes','T01_optimizer_diagnostics'):
            original(self,fig,name,question,estimand,source)
        else:
            plt.close(fig);self.manifest.append(dict(figure=name,question=question,estimand=estimand,source=source))
    monkeypatch.setattr(Figures,'save',save)
    return seen


@pytest.mark.parametrize('seeds',[[0],[0,1]])
def test_actual_offline_renderer(tmp_path,monkeypatch,seeds):
    from PV_experiments.representation.offline import render_evaluation
    data,raw,_,_=fixture(seeds);seen=capture_figures(monkeypatch)
    render_evaluation(data,tmp_path,'validation')
    allstats=pd.read_csv(tmp_path/'overall_energy_statistics.csv');month=pd.read_csv(tmp_path/'month_energy_statistics.csv')
    for i,m in enumerate(METHODS):
        v=data['daily'][:,:,:,i,2];row=allstats[(allstats.method==m)&(allstats.quantity=='net_kwh')].iloc[0]
        np.testing.assert_allclose(row['mean'],v.mean());np.testing.assert_allclose(row.sd_total,v.std())
        if len(seeds)>1:np.testing.assert_allclose(row.sd_seed_means,v.mean(axis=(1,2)).std(ddof=1),atol=1e-12)
        else:assert np.isnan(row.sd_seed_means)
    # Actual bar heights and actual errorbar segments, not just exported tables.
    for j,fi in enumerate((0,2)):
        v=data['daily'][...,fi].mean(axis=(1,2));np.testing.assert_allclose(seen.bars['E01_energy_and_effort'][0][j],v.mean(axis=0))
        if len(seeds)>1:
            seg=np.asarray(seen.errors['E01_energy_and_effort'][0][j]);np.testing.assert_allclose(seg[:,:,1].mean(axis=1),v.mean(axis=0));np.testing.assert_allclose((seg[:,1,1]-seg[:,0,1])/2,v.std(axis=0,ddof=1),atol=1e-12)
        else:assert not seen.errors['E01_energy_and_effort'][0]
    p=month[(month.method=='mbpo_minus_sac')&(month.quantity=='paired_net_kwh')]
    assert set(p.group)=={1,3};assert p.loc[p.group==1,'n_days'].item()==2;assert np.isnan(p.loc[p.group==3,'sd_day_means'].item())
    diff=data['daily'][:,:,:,3,2]-data['daily'][:,:,:,2,2]
    np.testing.assert_allclose(p.sort_values('group')['mean'],[diff[:,:2].mean(),diff[:,2:].mean()])
    assert 'E06_energy_spread_and_paired_gain' in seen and 'E07_monthly_paired_gain' in seen
    if len(seeds)>1:
        expected=np.array([diff[:,:2].mean(axis=(1,2)).std(ddof=1),diff[:,2:].mean(axis=(1,2)).std(ddof=1)])
        np.testing.assert_allclose(p.sort_values('group').sd_seed_means,expected,atol=1e-12)
        seg=np.asarray(seen.errors['E07_monthly_paired_gain'][0][0]);np.testing.assert_allclose((seg[:,1,1]-seg[:,0,1])/2,expected,atol=1e-12)
    for name in ('E01_energy_and_effort','E06_energy_spread_and_paired_gain','E07_monthly_paired_gain'):
        for ext in ('png','pdf'):assert (tmp_path/f'{name}.{ext}').stat().st_size>0
    manifest=json.loads((tmp_path/'figure_manifest.json').read_text());assert {r['figure'] for r in manifest}==set(seen)


def test_actual_rich_renderer(tmp_path,monkeypatch):
    from PV_experiments.representation import trajectories as tr
    data,raw,w,o=fixture([0,1]);directory=tmp_path/'collection';directory.mkdir();out=tmp_path/'report';out.mkdir()
    # Bypass physical replay only; render the actual full rich statistics and plots.
    manifest=dict(schema=1,stage='validation',scope='full',days=data['days'],draws=2,seeds=[0,1],fields=list(tr.FIELDS),weather_fields=list(tr.WEATHER),oracle_fields=list(tr.ORACLE_FIELDS),methods=list(METHODS),sources={},files={f'seed{s}/{d}.npz':'fixture' for s in data['seeds'] for d in data['days']})
    (directory/'trajectory_manifest.json').write_text(json.dumps(manifest))
    def load(directory,manifest,data,sid):
        d=data['days'].index(sid);return raw[:,d],w[d],o[d]
    monkeypatch.setattr(tr,'load_rich_day',load);seen=capture_figures(monkeypatch)
    tr.render_rich(directory,data,'validation','full',out)
    stats=pd.read_csv(out/'rich_curve_statistics.csv');g=stats[(stats.method=='mbpo')]
    signed=g[g.quantity=='action_tilt'];absolute=g[g.quantity=='abs_action_tilt'];executed=g[g.quantity=='abs_delta_azimuth_deg']
    np.testing.assert_allclose(signed['mean'],0);np.testing.assert_allclose(absolute['mean'],1);np.testing.assert_allclose(executed['mean'],2)
    assert set(absolute.clock_basis)=={'decision'} and set(executed.clock_basis)=={'response'}
    np.testing.assert_allclose(absolute.utc_hour_unwrapped.iloc[0],8.75);np.testing.assert_allclose(executed.utc_hour_unwrapped.iloc[-1],25)
    c=g[g.quantity=='cumulative_net_kwh'];np.testing.assert_allclose(c['mean'].iloc[-1],data['daily'][:,:,:,3,2].mean());np.testing.assert_allclose(c.sd_total.iloc[-1],data['daily'][:,:,:,3,2].std())
    # Initial zero belongs to rendered cumulative lines only.
    assert len(c)==195
    for x,y,label in seen['R03_cumulative_energy'][1]:assert len(x)==196 and x[0]==8.75 and y[0]==0
    assert 'R10_motion_magnitudes' in seen
    for name in ('R03_cumulative_energy','R10_motion_magnitudes'):
        for ext in ('png','pdf'):assert (out/f'{name}.{ext}').stat().st_size>0


def test_actual_training_renderer(tmp_path,monkeypatch):
    import sys
    from types import SimpleNamespace
    from PV_experiments.representation.training import render_training
    def run_path(cfg,alg,seed,root):return root/f'{alg}{seed}'
    monkeypatch.setitem(sys.modules,'PV_experiments.core',SimpleNamespace(run_path=run_path))
    for alg in ('sac','mbpo'):
        for seed in (0,1):
            p=run_path({},alg,seed,tmp_path);p.mkdir();cfg={'algo':{'start_steps':200}}
            (p/'config.json').write_text(json.dumps(cfg if alg=='sac' else {'sac':cfg}))
            (p/'run_metadata.json').write_text(json.dumps({'horizon':195,'final_env_steps':585}))
            pd.DataFrame(dict(env_steps=[195,390,585],updates=[0,380,770],gross_energy_kwh=[1.,2.,3.],movement_cost_kwh=[.1,.1,.1],return_kwh=[.9,1.9,2.9],alpha=[1.,.9,.8],critic_loss=[np.nan,2.,1.],actor_loss=[np.nan,-1.,-2.],synthetic_attempted=[0,0,10],synthetic_accepted=[0,0,10],synthetic_rejected=[0,0,0],synthetic_samples_used=[0,0,100],real_samples_used=[0,97280,99740])).to_csv(p/'episodes.csv',index=False)
    out=tmp_path/'report';out.mkdir();seen=capture_figures(monkeypatch);sources=render_training({'epochs':3},[0,1],tmp_path,out)
    for ax in seen['T01_optimizer_diagnostics']:
        assert any(np.array_equal(x,[200,200]) for x,y,label in ax)
    assert len([k for k in sources if k.endswith('config.json')])==4
    for ext in ('png','pdf'):assert (out/f'T01_optimizer_diagnostics.{ext}').stat().st_size>0
'''


def blob(data):
    return hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest()


def build(root):
    patches=[]
    for name,expected in PINS.items():
        rel=PREFIX+name;p=root/rel;raw=p.read_bytes()
        if blob(raw)!=expected:raise RuntimeError(f'{rel}: working source differs from audited blob {expected}; no changes made')
        old=raw.decode('utf-8');new=TRANSFORMS[name](old)
        if not new.endswith('\n'):new+='\n'
        compile(new,rel,'exec')
        # Git handles original files with or without terminal newline; the pinned files
        # all have it, except reject if this assumption fails rather than mangle diff.
        if not old.endswith('\n'):raise RuntimeError(f'{rel}: missing terminal newline')
        patches.extend(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='a/'+rel,tofile='b/'+rel))
    if (root/TEST_PATH).exists():raise FileExistsError(f'{TEST_PATH} already exists; no changes made')
    compile(TEST_SOURCE,TEST_PATH,'exec')
    patches.append(f'diff --git a/{TEST_PATH} b/{TEST_PATH}\nnew file mode 100644\n')
    patches.extend(difflib.unified_diff([],TEST_SOURCE.splitlines(True),fromfile='/dev/null',tofile='b/'+TEST_PATH))
    return ''.join(patches).encode('utf-8')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',default='.');g=p.add_mutually_exclusive_group(required=True);g.add_argument('--check',action='store_true');g.add_argument('--apply',action='store_true');args=p.parse_args();root=Path(args.root).resolve()
    if subprocess.check_output(['git','rev-parse','--show-toplevel'],cwd=root,text=True).strip()!=str(root):raise RuntimeError('Run from repository root')
    patch=build(root)
    subprocess.run(['git','apply','--check','-'],cwd=root,input=patch,check=True)
    print('All four source hashes, anchors, syntax and git apply --check passed.')
    if args.apply:
        subprocess.run(['git','apply','-'],cwd=root,input=patch,check=True)
        subprocess.run(['git','diff','--check'],cwd=root,check=True)
        print('Applied representation-only changes. Run: python -m pytest -q geometry10_tests/test_representation_final.py')
    else:print('Check only; no files modified.')

if __name__=='__main__':main()
