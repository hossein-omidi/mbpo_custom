from __future__ import annotations
import importlib.util,sys,types,json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

PACKAGE=Path(__file__).resolve().parents[1]
# In a real checkout, use the installed modules (otherwise test this package).
ROOT=PACKAGE.parent
sys.path.insert(0,str(ROOT if (ROOT/'PV_experiments/advanced_report.py').is_file() else PACKAGE))
from PV_experiments import advanced_learning as L, calendar_heatmaps as H, advanced_report as A


def learning_fixture():
    x=np.array([9750,19500,29250,39000]);sac=np.array([[1.2,1.5,2.,2.1],[1.,1.4,1.7,1.8],[1.5,2.,2.1,2.2],[1.1,1.2,1.3,1.4]]);mbpo=np.array([[1.7,1.95,2.1,2.2],[1.4,2.,2.1,2.3],[1.5,1.7,1.8,1.85],[1.3,1.4,1.5,1.6]])
    vals={a:np.stack([v-.05,v+.05],axis=1) for a,v in [('sac',sac),('mbpo',mbpo)]}
    times={a:np.stack([np.broadcast_to(np.stack([x*rate,x*.01,x*(rate+.01)],axis=-1),(4,4,3))]).reshape(4,4,3) for a,rate in [('sac',.04),('mbpo',.06)]}
    return dict(seeds=[0,1,2,3],days=['2020-01-01','2020-01-02'],steps=x,updates=x*2,matrix=vals,times=times,target=1.9,fraction=.9,sustain=2,fixed=1.,dual=2.,horizon=195,sources={},protocol='mock')


def calendar_fixture():
    # Different historical-year coverage, February leap date, and a true zero cost.
    days=['2018-01-01','2019-01-01','2020-02-29','2020-07-15','2024-12-31'];catalogue=days+['2018-01-02','2021-01-01','2024-03-15'];v=np.zeros((2,len(days),3,4,3))
    for s in range(2):
        for di in range(len(days)):
            for r in range(3):
                for mi in range(4):
                    gross=1.+.25*di+.04*r+.12*mi+(.02*s if mi>1 else 0.);cost=0. if mi==0 else .01*mi;v[s,di,r,mi]=[gross,cost,gross-cost]
    return dict(seeds=[0,1],days=days,catalogue_days=catalogue,draws=3,values=v,stage='test',sources={},protocol='mock')


def test_event_confirmation_and_schedule():
    steps=np.array([10,20,30,33]);assert L.event([2,1,2,2],steps,2,2)==(30,33);assert L.event([1,2,1,2],steps,2,2)==(None,None);assert L.event([1,2,1,2],steps,2,1)==(20,20)
    m=dict(horizon=195,evaluation_interval_episodes=50,final_epoch=127,final_env_steps=127*195)
    np.testing.assert_array_equal(L.expected_steps(m),np.array([50,100,127])*195)


def test_reduction_with_all_four_outcomes():
    d=learning_fixture();r=L.compute(d);rep=r['report'];assert rep['confirmation_status_counts']==dict(both=1,sac_only=1,mbpo_only=1,neither=1)
    assert rep['conditional_sample_reduction_percent']['mean']==pytest.approx(25.)
    assert rep['conditional_speedup']['mean']==pytest.approx(4/3)
    assert rep['conditional_sample_reduction_percent']['n']==1 and rep['conditional_sample_reduction_percent']['sd'] is None
    assert rep['confirmed_count']=={'sac':2,'mbpo':2}
    # SAC restricted [39000,39000,29250,39000]; MBPO [29250,29250,39000,39000].
    assert rep['restricted_steps_saved']['mean']==pytest.approx(2437.5)
    assert rep['restricted_budget_relative_saving_percent']==pytest.approx(100*(1-136500/146250))
    assert r['pairs'][1]['conditional_reduction_percent'] is None
    assert all(row['last_observed_steps']==39000 for row in r['rows'])
    assert r['rows'][0]['periodic_validation_transitions_to_confirmation']==4*2*195
    g=pd.DataFrame(r['curves']);actual=g[g.quantity=='mbpo_minus_sac'].sd_seed_means.to_numpy();expected=(d['matrix']['mbpo']-d['matrix']['sac']).mean(axis=1).std(axis=0,ddof=1);np.testing.assert_allclose(actual,expected)


def test_all_fail_and_bad_timing():
    d=learning_fixture();d['target']=99;r=L.compute(d);assert r['report']['conditional_sample_reduction_percent'] is None;assert r['report']['restricted_budget_relative_saving_percent']==0;assert r['report']['confirmed_count']=={'sac':0,'mbpo':0}
    d['times']['sac']=d['times']['sac'].copy();d['times']['sac'][0,0,2]=1000
    with pytest.raises(ValueError):L.compute(d)


def test_calendar_exact_and_missing():
    d=calendar_fixture();t=H.aggregate(d,'net_kwh','month_day');row=t[(t.method=='sac')&(t.row==1)&(t.column==1)].iloc[0];assert row.n_historical_days==2 and row.n_catalogue_days==3;assert row.mean_daily_kwh==pytest.approx(d['values'][:,:2,:,2,2].mean())
    assert row.sum_evaluated_daily_means_kwh==pytest.approx(row.mean_daily_kwh*2)
    row=t[(t.method=='sac')&(t.row==1)&(t.column==2)].iloc[0];assert row.n_historical_days==0 and row.n_catalogue_days==1 and pd.isna(row.mean_daily_kwh)
    row=t[(t.method=='sac')&(t.row==2)&(t.column==30)].iloc[0];assert not row.valid_calendar_cell
    row=t[(t.method=='sac')&(t.row==2)&(t.column==29)].iloc[0];assert row.valid_calendar_cell and row.n_historical_days==1 and pd.isna(row.sd_historical_day_means_kwh)
    z=H.aggregate(d,'movement_kwh','year_month');assert (z[z.method=='fixed_no_motion'].mean_daily_kwh.dropna()==0).all()
    q=H.aggregate(d,'paired_net_gain_kwh','month_day');r=q[(q.method=='mbpo_minus_sac')&(q.row==1)&(q.column==1)].iloc[0];assert r.mean_daily_kwh==pytest.approx((d['values'][:,:2,:,3,2]-d['values'][:,:2,:,2,2]).mean())


def test_render_reports(tmp_path):
    L.render_learning(learning_fixture(),tmp_path/'learning');H.render_calendar(calendar_fixture(),tmp_path/'calendar');assert len(list((tmp_path/'learning').glob('*.pdf')))==5;assert len(list((tmp_path/'calendar').glob('*.pdf')))==9
    for p in tmp_path.rglob('*.pdf'):assert p.stat().st_size>1000
    assert json.loads((tmp_path/'learning/sample_efficiency_summary.json').read_text())['conditional_sample_reduction_percent']['mean']==25.0


def test_zero_success_plot(tmp_path):
    d=learning_fixture();d['target']=99;L.render_learning(d,tmp_path);assert (tmp_path/'L05_compute_cost.pdf').is_file()


def test_output_atomic_and_no_overwrite(tmp_path):
    dest=tmp_path/'report'
    with pytest.raises(RuntimeError):
        with A.output_directory(dest) as p:(p/'partial').write_text('x');raise RuntimeError()
    assert not dest.exists()
    with A.output_directory(dest) as p:(p/'complete').write_text('yes')
    with pytest.raises(FileExistsError):
        with A.output_directory(dest):pass
    assert (dest/'complete').read_text()=='yes'


def test_additive_installer(tmp_path):
    spec=importlib.util.spec_from_file_location('installer',PACKAGE/'install.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);src=tmp_path/'original.py';src.write_text('pass\n');guard={'original.py':m.blob(src.read_bytes())};changes=m.plan(tmp_path,guard);assert len(changes)==3;m.apply(tmp_path,changes);assert m.plan(tmp_path,guard)=={};assert src.read_text()=='pass\n';src.write_text('bad')
    with pytest.raises(ValueError):m.plan(tmp_path,guard)


def _mock_runs(tmp_path,monkeypatch):
    d=learning_fixture();seeds=d['seeds'];steps=d['steps'];days=d['days'];cfg={'name':'fixture','epochs':200};split={'protocol_sha256':'mock','train_ids':['2019-01-01'],'validation_ids':days,'test_ids':['2021-01-01']}
    core=types.ModuleType('PV_experiments.core');core.verify_pair=lambda *args:split;core.run_path=lambda cfg,a,s,root:root/f'{a}{s}';core.read_json=lambda p:json.loads(p.read_text());monkeypatch.setitem(sys.modules,core.__name__,core)
    estimator=types.ModuleType('MBPO_workspace.src.sample_complexity');runs={}
    # When the real estimator exists in the installed checkout, use it directly.
    real_loader=None
    if (ROOT/'MBPO_workspace/src/sample_complexity.py').is_file():
        spec=importlib.util.spec_from_file_location('actual_estimator',ROOT/'MBPO_workspace/src/sample_complexity.py');real=importlib.util.module_from_spec(spec);spec.loader.exec_module(real);real_loader=real._load_run
    for a in L.ALGORITHMS:
        for i,seed in enumerate(seeds):
            p=core.run_path(cfg,a,seed,tmp_path);e=p/'periodic_evaluation';e.mkdir(parents=True)
            meta=dict(run_schema=2,algorithm=a,seed=seed,training_status='completed',scenario_protocol_sha256='mock',resolved_mdp_sha256='m',shared_sac_config_sha256='s',validation_scenario_ids=days,test_scenario_ids=split['test_ids'],horizon=195,state_dim=6,action_dim=2,energy_objective='net',movement_cost_mode='geometry',movement_penalty=1.,power_model='pvwatts_sapm',target_fraction=.9,sustain_checkpoints=2,bootstrap_replicates=100,evaluation_interval_episodes=50,final_epoch=200,final_env_steps=39000)
            L.write_json(p/'run_metadata.json',meta);L.write_json(p/'scenario_split.json',split)
            ep=[dict(epoch=j,env_steps=j*195,updates=j*390,scenario_id='2019-01-01',episode_seconds=1.,**(dict(synthetic_attempted=2,synthetic_accepted=1,synthetic_rejected=1,synthetic_samples_used=5,real_samples_used=10) if a=='mbpo' else {})) for j in range(1,201)];L.csv_out(p/'episodes.csv',ep)
            refs=[dict(scenario_id=sid,fixed_net_energy_kwh=1.,fixed_gross_energy_kwh=1.,fixed_movement_cost_kwh=0.,dual_axis_net_energy_kwh=2.,dual_axis_gross_energy_kwh=2.1,dual_axis_movement_cost_kwh=.1,episode_steps=195) for sid in days];L.csv_out(e/'evaluation_references.csv',refs)
            agg=[];daily=[];v=d['matrix'][a][i]
            for j,t in enumerate(steps):
                agg.append(dict(algorithm=a,seed=seed,epoch=int(t/195),env_steps=int(t),updates=int(t*2),mean_net_energy_kwh=v[:,j].mean(),evaluation_seconds=3.))
                for di,sid in enumerate(days):daily.append(dict(algorithm=a,seed=seed,epoch=int(t/195),env_steps=int(t),scenario_id=sid,net_energy_kwh=v[di,j],gross_energy_kwh=v[di,j]+.1,movement_cost_kwh=.1,episode_steps=195))
            L.csv_out(e/'evaluations.csv',agg);L.csv_out(e/'evaluation_days.csv',daily)
            runs[str(p)]=dict(root=p,metadata=meta,seed=seed,aggregate={int(t):v[:,j].mean() for j,t in enumerate(steps)},updates={int(t):int(t*2) for t in steps},daily={int(t):{sid:v[di,j] for di,sid in enumerate(days)} for j,t in enumerate(steps)},references=refs,episodes=ep)
    estimator._load_run=real_loader or (lambda p,a:runs[str(p)]);monkeypatch.setitem(sys.modules,estimator.__name__,estimator)
    return cfg,seeds,d,runs,real_loader


def test_learning_loader_and_missing_schedule(tmp_path,monkeypatch):
    cfg,seeds,d,runs,real=_mock_runs(tmp_path,monkeypatch);loaded=L.load_learning(cfg,seeds,tmp_path);np.testing.assert_allclose(loaded['matrix']['mbpo'],d['matrix']['mbpo']);assert loaded['replay'][-1]['actual_synthetic_batch_draw_fraction']==pytest.approx(1/3)
    assert loaded['times']['sac'][0,-1,2]==212
    if real:
        # Delete a common checkpoint consistently in aggregate and daily logs; old estimator would intersect silently.
        p=tmp_path/'sac0'/'periodic_evaluation'
        for name in ('evaluations.csv','evaluation_days.csv'):
            f=pd.read_csv(p/name);f[f.env_steps!=19500].to_csv(p/name,index=False)
    else:
        r=runs[str(tmp_path/'sac0')];r['aggregate'].pop(19500);r['daily'].pop(19500);r['updates'].pop(19500)
    with pytest.raises(ValueError,match='scheduled evaluation checkpoints'):L.load_learning(cfg,seeds,tmp_path)


def test_calendar_loader_checks_baseline_and_provenance(tmp_path,monkeypatch):
    d=calendar_fixture();split={'protocol_sha256':'mock','train_ids':[s for s in d['catalogue_days'] if s not in d['days']][:2],'validation_ids':[s for s in d['catalogue_days'] if s not in d['days']][2:],'test_ids':d['days']};cfg={'name':'fixture'};core=types.ModuleType('PV_experiments.core');core.verify_pair=lambda *args:split;core.run_path=lambda cfg,a,s,root:root/f'{a}{s}';core.read_json=lambda p:json.loads(p.read_text());monkeypatch.setitem(sys.modules,core.__name__,core)
    report=types.ModuleType('PV_experiments.report');ev=types.ModuleType('PV_experiments.evaluate');ev.existing_report_is_valid=lambda *a:True;ev._provenance=lambda *a,**k:{'checkpoint':'mock'};ev._require_matching_provenance=lambda path,expected:(_ for _ in ()).throw(ValueError('stale checkpoint')) if json.loads((path/'manager_provenance.json').read_text())!=expected else None;monkeypatch.setitem(sys.modules,ev.__name__,ev)
    fingerprints={(sid,i):str(i)*64 for sid in d['days'] for i in range(3)};report.validate_mc_source=lambda cfg,stage,seed,root:(root/f'eval{seed}',{'draws_per_day':3,'_validated_weather_hashes':fingerprints});monkeypatch.setitem(sys.modules,report.__name__,report)
    for i,s in enumerate(d['seeds']):
        for a in ('sac','mbpo'):
            p=core.run_path(cfg,a,s,tmp_path);p.mkdir();L.write_json(p/'scenario_split.json',split);L.write_json(p/'run_metadata.json',dict(scenario_protocol_sha256='mock',resolved_mdp_sha256='m',shared_sac_config_sha256='s',horizon=195,energy_objective='net',movement_cost_mode='geometry',movement_penalty=1.,power_model='pvwatts_sapm'))
        p=tmp_path/f'eval{s}';p.mkdir();L.write_json(p/'mc_summary.json',{});L.write_json(p/'manager_provenance.json',{'checkpoint':'mock'});rows=[]
        for di,sid in enumerate(d['days']):
            for rid in range(3):
                for mi,a in enumerate(H.METHODS):
                    v=d['values'][i,di,rid,mi];rows.append(dict(scenario_id=sid,draw=rid,method=a,profile_sha256=fingerprints[sid,rid],**dict(zip(H.FIELDS,v))))
        L.csv_out(p/'all_day_draw_method.csv',rows)
    loaded=H.load_calendar(cfg,d['seeds'],'test',tmp_path);np.testing.assert_allclose(loaded['values'],d['values'])
    p=tmp_path/'eval1'/'all_day_draw_method.csv';f=pd.read_csv(p,dtype={'profile_sha256':str});f.loc[0,'gross_kwh']+=1;f.to_csv(p,index=False)
    with pytest.raises(ValueError,match='Gross minus movement'):H.load_calendar(cfg,d['seeds'],'test',tmp_path)
    L.write_json(tmp_path/'eval1'/'manager_provenance.json',{'checkpoint':'old'})
    with pytest.raises(ValueError,match='stale checkpoint'):H.load_calendar(cfg,d['seeds'],'test',tmp_path)
