"""Synthetic, analytical regression tests; no training or weather data required."""
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
    manifest=dict(action_semantics=tr.ACTION_SEMANTICS,schema=1,stage='validation',scope='full',days=data['days'],draws=2,seeds=[0,1],fields=list(tr.FIELDS),weather_fields=list(tr.WEATHER),oracle_fields=list(tr.ORACLE_FIELDS),methods=list(METHODS),sources={},files={f'seed{s}/{d}.npz':'fixture' for s in data['seeds'] for d in data['days']})
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
