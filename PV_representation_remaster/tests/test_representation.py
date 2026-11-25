from pathlib import Path
import sys,json,types
import numpy as np
import pandas as pd
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from PV_experiments.representation.statistics import equal_mixture,from_day_moments,circular_stats,clock_grid,METHODS,ORACLE
from PV_experiments.representation.common import new_output,sha,write_json
from PV_experiments.representation.offline import render_evaluation,load_mc
from PV_experiments.representation.trajectories import Recorder,render_rich,representative_days,FIELDS,WEATHER,ORACLE_FIELDS
from PV_experiments.representation.dataset import render_dataset


def test_variance_identity_and_conditional_moments():
    rng=np.random.default_rng(0);x=rng.normal(size=(3,4,7,5));r=equal_mixture(x);r2=from_day_moments(x.mean(2),x.std(2,ddof=1),7)
    np.testing.assert_allclose(r['mean'],x.mean((0,1,2)));np.testing.assert_allclose(r['sd_total'],x.std((0,1,2),ddof=0));
    for k in r2:np.testing.assert_allclose(r[k],r2[k])
    assert np.isnan(equal_mixture(x[:1])['sd_seed_means']).all()


def test_pairing_and_time_covariance():
    x=np.array([1,2,3,4.]);p=np.stack([x,x],axis=-1).reshape(1,1,4,2)
    assert equal_mixture(np.cumsum(p,axis=-1))['sd_total'][-1]==pytest.approx(2*x.std())
    assert equal_mixture((p+5)-p)['sd_total'].max()==0


def test_utc_and_circular_azimuth():
    t=clock_grid('2020-12-31');assert str(t[0])=='2020-12-31 08:50:00+00:00';assert str(t[-1])=='2021-01-01 01:00:00+00:00';assert len(t)==195
    mu,r=circular_stats([359,1]);assert min(mu,360-mu)<1e-8 and r>.99
    mu,r=circular_stats([0,180]);assert np.isnan(mu)


def test_atomic_output_no_overwrite(tmp_path):
    p=tmp_path/'report'
    with pytest.raises(RuntimeError):
        with new_output(p) as d:
            (d/'partial').write_text('x');raise RuntimeError()
    assert not p.exists() and not list(tmp_path.iterdir())
    with new_output(p) as d:(d/'complete').write_text('yes')
    with pytest.raises(FileExistsError):
        with new_output(p):pass
    assert (p/'complete').read_text()=='yes'


def fixture():
    days=['2020-01-05','2020-04-05','2020-07-05','2020-10-05'];S=2;M=3;D=len(days);H=195
    raw=np.zeros((S,D,M,4,H,len(FIELDS)));weather=np.zeros((D,M,H,len(WEATHER)));oracle=np.zeros((D,M,H,len(ORACLE_FIELDS)))
    x=np.sin(np.linspace(0,np.pi,H))*100
    for si in range(S):
        for di in range(D):
            for m in range(M):
                weather[di,m,:,0]=x*4;weather[di,m,:,1]=x*3;weather[di,m,:,2]=x*(.4+m*.5);weather[di,m,:,3]=20;weather[di,m,:,4]=2;weather[di,m,:,5]=.2;weather[di,m,:,6]=60;weather[di,m,:,7]=180
                oracle[di,m,:,0]=x*2+di+m+2;oracle[di,m,:,1]=oracle[di,m,:,0]/12000;oracle[di,m,:,2]=40;oracle[di,m,:,3]=180
                for mi in range(4):
                    p=raw[si,di,m,mi];p[:,0]=x*(.7+mi*.1)+(si if mi>1 else 0)+di+m;p[:,1]=p[:,0]/12000;p[:,2]=0 if mi==0 else .00001*mi;p[:,3]=p[:,1]-p[:,2];p[:,4]=30+mi;p[:,5]=180;p[:,11]=25
    daily=raw[:,:,:,:,:,1:4].sum(4);power=raw[:,:,:,:,:,0];moments=np.stack([power.mean(2),power.std(2,ddof=1)],axis=-1)
    data=dict(days=days,seeds=[0,1],draws=M,daily=daily,moments=moments,fingerprints={(d,m):f'{i:02x}{m:02x}'+'0'*60 for i,d in enumerate(days) for m in range(M)},sources={'synthetic':'fixture'})
    return data,raw,weather,oracle


def test_offline_publication_figures(tmp_path):
    data,*_=fixture();render_evaluation(data,tmp_path,'test')
    d=pd.read_csv(tmp_path/'overall_energy_statistics.csv');a=d[(d.method=='mbpo')&(d.quantity=='net_kwh')].iloc[0];assert a['mean']==pytest.approx(data['daily'][:,:,:,3,2].mean());assert a.sd_seed_means>0
    t=pd.read_csv(tmp_path/'power_moments.csv');assert len(t)==4*195;assert (tmp_path/'E04_intraday_power.pdf').stat().st_size>1000
    assert len(json.loads((tmp_path/'figure_manifest.json').read_text()))==7


def test_dataset_render(tmp_path):
    data,*_=fixture();rows=[]
    for i,s in enumerate(data['days']):
        date=pd.Timestamp(s);rows.append(dict(scenario_id=s,source_year=date.year,month=date.month,day=date.day,season='x',role=['train','validation','test','train'][i],ghi_kwh_m2=2+i,dni_kwh_m2=1+i,dhi_kwh_m2=.5,temperature_c=12+i,wind_m_s=2,diffuse_fraction=.2,clearsky_ghi_ratio=.6))
    render_dataset(pd.DataFrame(rows),np.ones((4,195)),tmp_path,{'test':'synthetic'});assert (tmp_path/'D03_annual_irradiance_heatmap.pdf').exists()


def test_rich_roundtrip_and_paired_cumulative_sd(tmp_path):
    data,raw,w,o=fixture();base=tmp_path/'raw';base.mkdir();files={}
    for si,s in enumerate(data['seeds']):
        for di,d in enumerate(data['days']):
            name=f'seed{s}/{d}.npz';p=base/name;p.parent.mkdir(exist_ok=True);np.savez_compressed(p,raw=raw[si,di],weather=w[di],oracle=o[di],profile_sha256=np.array([data['fingerprints'][d,r] for r in range(3)]),timestamps=np.array([t.isoformat() for t in clock_grid(d)]));files[name]=sha(p)
    m=dict(schema=1,stage='validation',scope='full',days=data['days'],draws=3,seeds=data['seeds'],fields=list(FIELDS),weather_fields=list(WEATHER),oracle_fields=list(ORACLE_FIELDS),methods=list(METHODS),sources=data['sources'],files=files);write_json(base/'trajectory_manifest.json',m)
    dest=tmp_path/'figures';render_rich(base,data,'validation','full',dest);f=pd.read_csv(dest/'rich_curve_statistics.csv');a=f[(f.method=='mbpo_minus_sac')&(f.quantity=='cumulative_net_difference_kwh')].sort_values('step');expected=np.cumsum(raw[:,:,:,3,:,3]-raw[:,:,:,2,:,3],axis=-1);np.testing.assert_allclose(a['mean'],expected.mean((0,1,2)),atol=1e-10);np.testing.assert_allclose(a.sd_total,expected.std((0,1,2)),atol=1e-10)
    assert not ((f.method==ORACLE)&f.quantity.str.contains('net|actuator')).any();assert (dest/'R09_paired_power_heatmap.png').exists()
    with pytest.raises(ValueError,match='stale'):
        render_rich(base,{**data,'sources':{'changed':'true'}},'validation','full',tmp_path/'reject')


def test_recorder_validates_physical_cost_and_fixed_pose():
    class Env:
        horizon=1;max_delta_tilt=5.;max_delta_azimuth=10.;slew_rate_tilt_deg_s=1.5;slew_rate_azimuth_deg_s=2.;interval_hours=1/12;movement_penalty=1.;actuator_power_w=30.;movement_cost_mode='geometry'
        @property
        def unwrapped(self):return self
        def reset(self,**kw):self.tilt=30.;self.azimuth=180.;return np.zeros(6)
        def step(self,a):
            dt=float(a[0])*5;da=float(a[1])*10;self.tilt+=dt;self.azimuth+=da;cost=30*(abs(dt)/1.5+abs(da)/2)/3.6e6;energy=100/12000;info=dict(delta_tilt_deg=dt,delta_azimuth_deg=da,tilt=self.tilt,azimuth=self.azimuth,movement_cost=cost,energy_kwh=energy,power_dc_w=100.,poa_global=500.,cell_temperature_c=25.)
            return np.zeros(6),energy-cost,True,info
    r=Recorder(Env(),'fixed_no_motion');r.reset();r.step(np.zeros(2));assert r.records[0][2]==0
    r.reset()
    with pytest.raises(ValueError,match='Fixed baseline action'):r.step([1.,0.])
    r=Recorder(Env(),'sac');r.reset();r.step([1.,-.5]);assert r.records[0][2]>0


def test_load_mc_current_csv_contract_and_reject_bad_gross(tmp_path,monkeypatch):
    data,raw,*_=fixture();cfg={'name':'test'};split={'test_ids':data['days']};base=tmp_path;source=types.ModuleType('PV_experiments.evaluate');source.evaluation_output=lambda cfg,stage,seed,root:root/f'seed{seed}';source.existing_report_is_valid=lambda *a:True;source._provenance=lambda *a,**k:{};source._require_matching_provenance=lambda *a:None
    report=types.ModuleType('PV_experiments.report');report.validate_mc_source=lambda cfg,stage,seed,root:(root/f'seed{seed}',{'_validated_weather_hashes':data['fingerprints']});core=types.ModuleType('PV_experiments.core');core.run_path=lambda cfg,alg,seed,root:root/f'{alg}{seed}'
    for name,mod in [('PV_experiments.evaluate',source),('PV_experiments.report',report),('PV_experiments.core',core)]:monkeypatch.setitem(sys.modules,name,mod)
    for si,seed in enumerate(data['seeds']):
        d=base/f'seed{seed}';d.mkdir();write_json(d/'mc_summary.json',{'draws_per_day':3});write_json(d/'manager_provenance.json',{})
        a=[];p=[]
        for di,sid in enumerate(data['days']):
            for mi,method in enumerate(METHODS):
                for draw in range(3):
                    g,c,n=data['daily'][si,di,draw,mi];a.append(dict(scenario_id=sid,draw=draw,method=method,gross_kwh=g,movement_kwh=c,net_kwh=n,profile_sha256=data['fingerprints'][sid,draw]))
                for i,t in enumerate(clock_grid(sid)):
                    mean,sd=data['moments'][si,di,mi,i];p.append(dict(scenario_id=sid,method=method,step=i+1,time_utc=t.isoformat(),power_mean_w=mean,power_sd_w=sd))
        pd.DataFrame(a).to_csv(d/'all_day_draw_method.csv',index=False);pd.DataFrame(p).to_csv(d/'daily_power_mean_std_by_utc_step.csv',index=False)
        for alg in ('sac','mbpo'):
            r=core.run_path(cfg,alg,seed,base);r.mkdir();(r/'latest.pt').write_bytes(b'mock checkpoint')
    loaded=load_mc(cfg,data['seeds'],'test',split,base);np.testing.assert_allclose(loaded['daily'],data['daily']);np.testing.assert_allclose(loaded['moments'],data['moments'])
    f=base/'seed0'/'all_day_draw_method.csv';a=pd.read_csv(f);a.loc[0,'gross_kwh']+=1.;a.to_csv(f,index=False)
    with pytest.raises(ValueError,match='Daily gross'):load_mc(cfg,data['seeds'],'test',split,base)


def test_real_repo_oracle_against_grid_when_dependencies_available():
    pytest.importorskip('pvlib');pytest.importorskip('mbpo.env.baseline_contract')
    from mbpo.env.baseline_contract import isotropic_max_poa_tilt
    from mbpo.env.pv_power_backend import compute_pv_power
    rng=np.random.default_rng(13)
    for _ in range(8):
        z=rng.uniform(0,180);az=rng.uniform(0,360);dn=rng.uniform(0,800) if z<90 else 0.;dh=rng.uniform(10,200);gh=dn*max(np.cos(np.deg2rad(z)),0)+dh;rho=rng.uniform(0,.5)
        tilt=isotropic_max_poa_tilt(z,dn,gh,dh,rho)
        def evaluate(b,a):return compute_pv_power(b,a,z,az,dn,gh,dh,temp_air_c=25,wind_speed_ms=2,wind_reference_height_m=2,surface_albedo=rho).power_w
        upper=evaluate(tilt,az);grid=max(evaluate(b,a) for b in np.linspace(0,90,19) for a in np.linspace(0,330,12));assert upper+1e-7>=grid


def test_guarded_additive_installer(tmp_path):
    import importlib.util
    spec=importlib.util.spec_from_file_location('pv_remaster_installer',ROOT/'install.py');installer=importlib.util.module_from_spec(spec);spec.loader.exec_module(installer)
    repo=tmp_path/'repo';repo.mkdir();p=repo/'core.py';p.write_text('unchanged\n');expected={'core.py':installer.blob_sha(p.read_bytes())}
    changes=installer.plan(repo,expected);assert len(changes)==8
    installer.apply(repo,changes);assert installer.plan(repo,expected)=={};assert p.read_text()=='unchanged\n';assert not list(repo.rglob('*.bak'))
    p.write_text('different')
    with pytest.raises(ValueError,match='different source'):installer.plan(repo,expected)


def test_training_and_learning_plot_interfaces(tmp_path,monkeypatch):
    from PV_experiments.representation.training import render_training,render_learning
    core=types.ModuleType('PV_experiments.core');core.run_path=lambda cfg,alg,seed,root:root/f'{alg}{seed}';monkeypatch.setitem(sys.modules,'PV_experiments.core',core)
    cfg={'epochs':4};seeds=[0,1]
    for alg in ('sac','mbpo'):
        for seed in seeds:
            p=core.run_path(cfg,alg,seed,tmp_path);p.mkdir();write_json(p/'run_metadata.json',{'horizon':195,'final_env_steps':780});write_json(p/'scenario_split.json',{})
            rows=[]
            for t in range(1,5):
                r=dict(env_steps=195*t,gross_energy_kwh=1+t*.01,movement_cost_kwh=.02,return_kwh=.98+t*.01,critic_loss=.2,actor_loss=.1,alpha=.05)
                if alg=='mbpo':r.update(synthetic_attempted=10,synthetic_accepted=8,synthetic_rejected=2,synthetic_samples_used=16,real_samples_used=84)
                rows.append(r)
            pd.DataFrame(rows).to_csv(p/'episodes.csv',index=False);e=p/'periodic_evaluation';e.mkdir()
            for f in ('evaluations.csv','evaluation_days.csv','evaluation_references.csv'):(e/f).write_text('mock')
    training=tmp_path/'training';render_training(cfg,seeds,tmp_path,training);r=pd.read_csv(training/'synthetic_replay_usage.csv');assert r.synthetic_batch_fraction.iloc[-1]==pytest.approx(.16)
    estimator=types.ModuleType('MBPO_workspace.src.sample_complexity')
    def analyze(a,b,out):
        out=Path(out);curves=[];diff=[];targets=[];compute=[]
        for t in (195,390):
            for alg in ('sac','mbpo'):curves.append(dict(algorithm=alg,env_steps=t,mean_net_energy_kwh=1.,sd_across_training_seed_means_kwh=.1))
            diff.append(dict(env_steps=t,mean_difference_net_energy_kwh=.02,sd_across_paired_training_seed_means_kwh=.03))
        for seed in seeds:
            for alg in ('sac','mbpo'):
                targets.append(dict(seed=seed,algorithm=alg,reached_target=alg=='mbpo',real_env_steps_to_target=390 if alg=='mbpo' else None,last_evaluated_env_steps=390,auc_mean_net_kwh_per_episode=.9));compute.append(dict(seed=seed,algorithm=alg,measured_training_plus_evaluation_seconds=5))
        for filename,rows in [('learning_curve_descriptive_sd.csv',curves),('paired_difference_descriptive_sd.csv',diff),('sample_complexity_by_seed.csv',targets),('computational_cost_by_seed.csv',compute)]:pd.DataFrame(rows).to_csv(out/filename,index=False)
        write_json(out/'report.json',{'target_kwh_per_episode':1.2});(out/'old_duplicate.png').write_bytes(b'old');return {'target_kwh_per_episode':1.2}
    estimator.analyze_sample_complexity=analyze;monkeypatch.setitem(sys.modules,estimator.__name__,estimator)
    dest=tmp_path/'learning';dest.mkdir();render_learning(cfg,seeds,tmp_path,dest);assert (dest/'L02_target_with_censoring.pdf').exists();assert not (dest/'old_duplicate.png').exists()


def test_collection_resume_and_full_interfaces(tmp_path,monkeypatch):
    from PV_experiments.representation import trajectories as tr
    # A minimal physical fixture with the exact 195-step run_one interface.
    class Env:
        horizon=195;max_delta_tilt=5.;max_delta_azimuth=10.;slew_rate_tilt_deg_s=1.5;slew_rate_azimuth_deg_s=2.;interval_hours=1/12;movement_penalty=1.;actuator_power_w=30.;movement_cost_mode='geometry';randomize_initial_orientation=False
        def __init__(self,kwargs):pass
        @property
        def unwrapped(self):return self
        def _solar_position(self,t):return types.SimpleNamespace(zenith=30.,azimuth=180.)
        def reset(self,scenario_id,realization_id):self.tilt=30.;self.azimuth=180.;self.i=0;self.sid=scenario_id;self.draw=realization_id;self.times=clock_grid(scenario_id);self.current_time=self.times[0]-pd.Timedelta(minutes=5);return np.zeros(6)
        def step(self,a):
            self.i+=1;dt=float(a[0])*5;da=float(a[1])*10;old=self.tilt;self.tilt=float(np.clip(self.tilt+dt,0,90));dt=self.tilt-old;self.azimuth=(self.azimuth+da)%360;cost=30*(abs(dt)/1.5+abs(da)/2)/3.6e6;power=100+self.draw;energy=power/12000;info=dict(delta_tilt_deg=dt,delta_azimuth_deg=da,tilt=self.tilt,azimuth=self.azimuth,movement_cost=cost,energy_kwh=energy,power_dc_w=power,poa_global=500.,cell_temperature_c=25.);info.update(dict(zip(WEATHER,[400.,500.,100.,20.,2.,.2,30.,180.])));self.current_time=self.times[self.i-1];return np.zeros(6),energy-cost,self.i==195,info
        def close(self):pass
    fake=types.ModuleType('SAC_workspace.src.environment.environment');fake.SACEnvironment=Env;monkeypatch.setitem(sys.modules,fake.__name__,fake)
    mc=types.ModuleType('MBPO_workspace.workflows.09_monte_carlo_evaluate');mc.METHODS=METHODS;mc.load_policy=lambda *a,**kw:None
    calls=[];sid='2020-01-05';draws=2
    def run_one(env,method,sid,draw,policy):
        calls.append((method,sid,draw));env.reset(scenario_id=sid,realization_id=draw);times=[]
        for t in range(195):env.step(np.zeros(2));times.append(env.unwrapped.current_time.isoformat())
        return {'profile_sha256':str(draw)*64,'pose':(30.,180.),'timestamps_utc':times}
    mc.run_one=run_one;monkeypatch.setitem(sys.modules,mc.__name__,mc)
    core=types.ModuleType('PV_experiments.core');core.run_path=lambda *a:tmp_path;monkeypatch.setitem(sys.modules,core.__name__,core)
    monkeypatch.setattr(tr,'load_physics_config',lambda *a:types.SimpleNamespace(sac=types.SimpleNamespace(eval_env={})))
    monkeypatch.setattr(tr,'oracle_from_weather',lambda u,infos:np.array([[120.,120/12000,30.,180.,500.]]*195))
    daily=np.zeros((2,1,2,4,3));m=np.zeros((2,1,4,195,2))
    for draw in range(2):daily[:,:,draw,:,0]=195*(100+draw)/12000;daily[:,:,draw,:,2]=daily[:,:,draw,:,0]
    m[...,0]=100.5;m[...,1]=np.std([100,101],ddof=1)
    d=dict(seeds=[0,1],days=[sid],draws=2,daily=daily,moments=m,fingerprints={(sid,draw):str(draw)*64 for draw in range(2)},sources={'fixture':'yes'})
    out=tmp_path/'collect';out.mkdir();cfg={'device':'cpu'};split={'protocol_sha256':'split'}
    tr.collect(cfg,split,d,'validation','full',tmp_path,out);assert len(calls)==16;assert (out/'baseline_audit.json').exists();tr.collect(cfg,split,d,'validation','full',tmp_path,out);assert len(calls)==16
    assert json.loads((out/'baseline_audit.json').read_text())['checked_feasible_steps']==16*195
    with pytest.raises(ValueError,match='inputs or code changed'):tr.collect(cfg,split,{**d,'sources':{'fixture':'changed'}},'validation','full',tmp_path,out)


def test_model_report_uses_normalized_deltas_and_counterfactuals(tmp_path,monkeypatch):
    from PV_experiments.representation.training import render_model
    seed=0;sid='2020-01-05';run=tmp_path/'run';run.mkdir();write_json(run/'scenario_split.json',{'validation_ids':[sid]});folder=tmp_path/'model'/'realization0';folder.mkdir(parents=True)
    e=types.ModuleType('PV_experiments.evaluate');e.evaluation_output=lambda *a:folder.parent;e._provenance=lambda *a,**kw:{'scenario_protocol_sha256':'split'};e._require_matching_provenance=lambda *a:None;e._verify_model_validation_summary=lambda *a:None
    core=types.ModuleType('PV_experiments.core');core.run_path=lambda *a:run
    monkeypatch.setitem(sys.modules,e.__name__,e);monkeypatch.setitem(sys.modules,core.__name__,core)
    rows=[];metrics={}
    for i in range(195):
        r=dict(scenario_id=sid,step=i)
        for c in ('dni_norm','dhi_norm','power_norm'):r[f'actual_delta_{c}']=.2;r[f'predicted_delta_{c}']=.1;metrics[c]=dict(rmse=.1,persistence_rmse=.2,bias=-.1,coverage_90=.9,coverage_95=.95)
        rows.append(r)
    pd.DataFrame(rows).to_csv(folder/'external_on_policy_predictions.csv',index=False)
    pd.DataFrame([dict(scenario_id=sid,step=0,action_type='policy',reward_error_kwh=.002,normalized_epistemic_disagreement=.01)]).to_csv(folder/'counterfactual_predictions.csv',index=False)
    write_json(folder/'manager_provenance.json',{});write_json(folder/'model_validation_summary.json',dict(validation_scenario_ids=[sid],external_on_policy_one_step=metrics,counterfactual_action_test=dict(n_counterfactual_transitions=1,reward_rmse_kwh=.002)))
    dest=tmp_path/'out';render_model({},[seed],tmp_path,dest);assert (dest/'M01_model_skill_and_counterfactual_error.pdf').exists();assert len(pd.read_csv(dest/'model_quality_by_seed.csv'))==3
