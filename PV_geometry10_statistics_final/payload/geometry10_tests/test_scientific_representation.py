"""Numerical tests independent of NSRDB, GPU, and trained policies."""
import json
import hashlib
import numpy as np
import pandas as pd
import pytest
from PV_experiments.scientific_checks import (METHODS,ACTIONS,CHANNELS,validate_mc_tables,validate_model_tables,verify_model_accounting,verify_model_physical_parity,recovery_accounting,same)


def dump(p,v):p.write_text(json.dumps(v,allow_nan=False))


def nested(y,days):
    D,M=y.shape; means=y.mean(1); s=y.std(1,ddof=1); b=means.var(); w=np.mean(s*s)*(M-1)/M; corrected=b-(1-1/D)*np.mean(s*s)/M
    def q(v):return dict(zip(('q05','median','q95'),np.quantile(v,[.05,.5,.95]).tolist()))
    return dict(n_historical_days=D,draws_per_day=M,mean_net_kwh=float(y.mean()),between_day_variance_kwh2=b,within_day_uncertainty_variance_kwh2=w,total_descriptive_sd_kwh=y.std(),total_empirical_variance_kwh2=y.var(),between_day_variance_corrected_kwh2=max(0,corrected),between_day_untruncated_correction_kwh2=corrected,within_day_unbiased_variance_kwh2=np.mean(s*s),mc_se_mean_fixed_days_kwh=np.linalg.norm(s)/(D*np.sqrt(M)),pooled_quantiles_kwh=q(y),each_day={d:dict(n=M,mean=means[i],sd_descriptive=s[i],mc_se_mean=s[i]/np.sqrt(M),**q(y[i])) for i,d in enumerate(days)})


@pytest.fixture
def mc(tmp_path):
    days=['2020-01-01','2020-07-01']; M=3
    # Vary independently by day, draw, method, and time; preserve positive power.
    p=np.arange(195)[None,None,None,:]*.2+np.arange(4)[None,None,:,None]*3+np.arange(2)[:,None,None,None]*13+np.arange(M)[None,:,None,None]*np.array([1,2,3,5])[None,None,:,None]+10
    g=p.sum(-1)/12000; movement=np.broadcast_to(np.array([0,.001,.002,.003])[None,None,:],g.shape); net=g-movement
    rows=[]; clock=[]; pooled=[]; dirs={}
    for i,d in enumerate(days):
        for m in range(M):
            for j,a in enumerate(METHODS):rows.append(dict(scenario_id=d,draw=m,method=a,gross_kwh=g[i,m,j],movement_kwh=movement[i,m,j],net_kwh=net[i,m,j],profile_sha256=hashlib.sha256(f'{d}:{m}'.encode()).hexdigest()))
        times=pd.date_range(f'{d} 08:45',periods=196,freq='5min',tz='UTC')[1:]
        for j,a in enumerate(METHODS):
            for t,utc in enumerate(times):
                v=p[i,:,j,t];clock.append(dict(scenario_id=d,method=a,step=t+1,time_utc=utc.isoformat(),power_mean_w=v.mean(),power_sd_w=v.std(ddof=1),power_q05_w=np.quantile(v,.05),power_q95_w=np.quantile(v,.95)))
    for j,a in enumerate(METHODS):
        for t in range(195):
            z=p[:,:,j,t]; between=z.mean(1).var();within=np.mean(z.var(1));pooled.append(dict(method=a,step=t+1,utc_hour_unwrapped=round(8.75+(t+1)/12,8),n_historical_days=2,draws_per_day=M,mean_power_w=z.mean(),sd_total_descriptive_w=z.std(),sd_between_historical_days_w=np.sqrt(between),sd_within_weather_draws_w=np.sqrt(within),variance_check_total_w2=z.var()))
    pd.DataFrame(rows).to_csv(tmp_path/'all_day_draw_method.csv',index=False)
    pd.DataFrame(clock).to_csv(tmp_path/'daily_power_mean_std_by_utc_step.csv',index=False)
    pd.DataFrame(pooled).to_csv(tmp_path/'mc_pooled_power_mean_sd_by_utc_step.csv',index=False)
    pd.DataFrame(clock).query('scenario_id == "2020-01-01"').to_csv(tmp_path/'first_day_power_mean_sd.csv',index=False)
    pairs=[('mbpo','sac'),('mbpo','fixed_no_motion'),('sac','fixed_no_motion'),('mbpo','sun_slew_limited'),('sac','sun_slew_limited')]
    dump(tmp_path/'mc_summary.json',dict(methods={a:nested(net[:,:,j],days) for j,a in enumerate(METHODS)},paired_net_energy_differences={a+'_minus_'+b:nested(net[:,:,METHODS.index(a)]-net[:,:,METHODS.index(b)],days) for a,b in pairs}))
    for a in ('sac','mbpo'):
        folder=tmp_path/a; e=folder/'periodic_evaluation';e.mkdir(parents=True);dirs[a]=folder; j=METHODS.index(a)
        pd.DataFrame(dict(scenario_id=days,epoch=100,episode_steps=195,gross_energy_kwh=g[:,0,j],movement_cost_kwh=movement[:,0,j],net_energy_kwh=net[:,0,j])).to_csv(e/'evaluation_days.csv',index=False)
        refs=dict(scenario_id=days,episode_steps=195)
        for k,prefix in enumerate(('fixed','dual_axis')):
            for field,v in [('gross_energy_kwh',g),('movement_cost_kwh',movement),('net_energy_kwh',net)]:refs[prefix+'_'+field]=v[:,0,k]
        pd.DataFrame(refs).to_csv(e/'evaluation_references.csv',index=False)
    return tmp_path,days,M,dirs,net


def check(mc):
    path,days,M,dirs,_=mc;return validate_mc_tables(path,days,M,run_dirs=dirs,epoch=100)


def test_raw_mc_matches_all_moments_and_policy_parity(mc):
    same(check(mc)[...,2],mc[-1],'Fixture raw net')


@pytest.mark.parametrize('filename,field,match',[('mc_pooled_power_mean_sd_by_utc_step.csv','sd_total_descriptive_w','Pooled MC'),('daily_power_mean_std_by_utc_step.csv','power_mean_w','Mean power integral'),('first_day_power_mean_sd.csv','power_sd_w','First-day power')])
def test_mc_wrong_statistics_rejected(mc,filename,field,match):
    path=mc[0]/filename;f=pd.read_csv(path);f.loc[0,field]+=1;f.to_csv(path,index=False)
    with pytest.raises(ValueError,match=match):check(mc)


def test_mc_summary_wrong_variance_rejected(mc):
    path=mc[0]/'mc_summary.json';d=json.loads(path.read_text());d['methods']['mbpo']['total_descriptive_sd_kwh']+=.01;dump(path,d)
    with pytest.raises(ValueError,match='total_descriptive_sd'):check(mc)


def test_mc_missing_or_duplicate_draw_rejected(mc):
    p=mc[0]/'all_day_draw_method.csv';d=pd.read_csv(p);d.iloc[:-1].to_csv(p,index=False)
    with pytest.raises(ValueError,match='grid'):check(mc)


def test_final_policy_mismatch_rejected(mc):
    p=mc[3]['mbpo']/'periodic_evaluation/evaluation_days.csv';d=pd.read_csv(p);d.loc[0,'net_energy_kwh']+=.01;d.to_csv(p,index=False)
    with pytest.raises(ValueError,match='Final learned policy'):check(mc)


def test_recovery_lower_bound(tmp_path):
    assert recovery_accounting(tmp_path,19500,195)['known_executed_training_transitions_lower_bound']==19500
    p=tmp_path/'recovery_events.json';dump(p,[dict(status='applied',discarded_complete_csv_rows=3,known_repeated_transitions=585)])
    assert recovery_accounting(tmp_path,19500,195)['known_executed_training_transitions_lower_bound']==20085
    dump(p,[dict(status='pending',discarded_complete_csv_rows=3,known_repeated_transitions=585)])
    with pytest.raises(ValueError):recovery_accounting(tmp_path,19500,195)


def test_unfitted_status_is_not_zero_error(tmp_path):
    d=dict(model_status='unfitted',validation_scenario_ids=['2020-01-01'],training_synthetic_data_accounting=dict(synthetic_samples_used=0),external_on_policy_one_step=None,counterfactual_action_test=None)
    assert validate_model_tables(tmp_path,d,['2020-01-01'])==(None,None)
    d['external_on_policy_one_step']={}
    with pytest.raises(ValueError,match='fabricated'):validate_model_tables(tmp_path,d,['2020-01-01'])


def test_model_metrics_and_physical_truth(tmp_path):
    from mbpo.audit_support import counterfactual_metrics
    from mbpo.audit_release import equivalent_pose_metrics
    days=['2020-01-01']; sid=days[0]; steps=np.arange(195); y=steps*.00001; pred=y+.0001; sd=np.full(195,.001); rows=pd.DataFrame(dict(scenario_id=sid,step=steps,actual_next_power_norm=.05,actual_reward_kwh=.05/6,predicted_reward_kwh=.05/6+.0001/6)); metrics={'n_transitions':195}
    for ch in CHANNELS:
        for key,v in [('actual',y),('predicted',pred),('error',pred-y),('q025',y-.002),('q05',y-.001),('q95',y+.001),('q975',y+.002)]:rows[f'{key}_delta_{ch}']=v
        rows[f'predictive_sd_{ch}']=sd;metrics[ch]=dict(rmse=np.sqrt(np.mean((pred-y)**2)),persistence_rmse=np.sqrt(np.mean(y*y)),bias=np.mean(pred-y),mae=np.mean(np.abs(pred-y)),mean_predictive_sd=sd.mean(),coverage_90=1.,coverage_95=1.,skill_vs_persistence=1-np.sqrt(np.mean((pred-y)**2))/max(np.sqrt(np.mean(y*y)),1e-12),rmse_to_mean_predictive_sd=np.sqrt(np.mean((pred-y)**2))/sd.mean())
    err=rows.predicted_reward_kwh-rows.actual_reward_kwh; metrics['reward']=dict(rmse_kwh=np.sqrt(np.mean(err*err)),mae_kwh=np.mean(np.abs(err)),bias_kwh=np.mean(err))
    rows.to_csv(tmp_path/'external_on_policy_predictions.csv',index=False);rows.to_csv(tmp_path/'internal_holdout_predictions.csv',index=False)
    cf=[]
    for j,a in enumerate(ACTIONS):
        p=.05+j*.001; q=p+.0001;cf.append(dict(scenario_id=sid,step=0,action_type=a,actual_reward_kwh=p/6,predicted_reward_kwh=q/6,reward_error_kwh=(q-p)/6,actual_next_power_norm=p,predicted_next_power_norm=q,power_error_norm=q-p,next_tilt_norm=.5,next_azimuth_sin=0.,next_azimuth_cos=1.,normalized_epistemic_disagreement=.1))
    pd.DataFrame(cf).to_csv(tmp_path/'counterfactual_predictions.csv',index=False)
    cm=counterfactual_metrics(cf);cm.update(n_counterfactual_transitions=len(cf),reward_rmse_kwh=np.sqrt(np.mean([r['reward_error_kwh']**2 for r in cf])),power_rmse_norm=np.sqrt(np.mean([r['power_error_norm']**2 for r in cf])))
    cm.update(equivalent_pose_metrics(cf));cm.update(reward_mae_kwh=np.mean([abs(r['reward_error_kwh']) for r in cf]),reward_bias_kwh=np.mean([r['reward_error_kwh'] for r in cf]),power_mae_norm=np.mean([abs(r['power_error_norm']) for r in cf]),mean_normalized_epistemic_disagreement=.1)
    d=dict(model_status='fitted',validation_scenario_ids=days,states_per_scenario=1,internal_whole_day_holdout=metrics,external_on_policy_one_step=metrics,counterfactual_action_test=cm)
    f,_=validate_model_tables(tmp_path,d,days)
    e=tmp_path/'periodic_evaluation';e.mkdir();pd.DataFrame([dict(scenario_id=sid,epoch=100,episode_steps=195,net_energy_kwh=195*.05/6,gross_energy_kwh=195*.05/6)]).to_csv(e/'evaluation_days.csv',index=False)
    verify_model_physical_parity(f,tmp_path,100,0)
    f.loc[0,'actual_reward_kwh']+=.001
    with pytest.raises(ValueError,match='physical trajectory'):verify_model_physical_parity(f,tmp_path,100,0)
    for key in ('reward_mae_kwh','reward_bias_kwh','power_mae_norm','alias_predicted_power_rmse_w','n_equivalent_pose_pairs','mean_normalized_epistemic_disagreement'):
        old=cm[key];cm[key]=old+1
        with pytest.raises(ValueError,match='Counterfactual'):validate_model_tables(tmp_path,d,days)
        cm[key]=old
    for key in ('skill_vs_persistence','rmse_to_mean_predictive_sd'):
        old=metrics['power_norm'][key];metrics['power_norm'][key]=old+1
        with pytest.raises(ValueError,match='skill/SD ratio'):validate_model_tables(tmp_path,d,days)
        metrics['power_norm'][key]=old
    metrics['power_norm']['coverage_90']=.9
    with pytest.raises(ValueError,match='Model metrics'):validate_model_tables(tmp_path,d,days)


def test_model_accounting(tmp_path):
    pd.DataFrame([dict(synthetic_attempted=10,synthetic_accepted=8,synthetic_rejected=2,synthetic_samples_used=39,real_samples_used=217)]).to_csv(tmp_path/'episodes.csv',index=False)
    d=dict(training_synthetic_data_accounting=dict(synthetic_attempted=10,synthetic_accepted=8,synthetic_rejected=2,synthetic_samples_used=39,real_samples_used=217,actual_synthetic_minibatch_fraction=39/256))
    verify_model_accounting(d,tmp_path)
    d['training_synthetic_data_accounting']['synthetic_samples_used']=40
    with pytest.raises(ValueError,match='replay exposure'):verify_model_accounting(d,tmp_path)


def test_learning_seed_sd_not_pooled_weather_sd():
    from MBPO_workspace.src.curve_statistics import summarize
    x=np.array([[.60,1.20],[.62,1.22],[.64,1.24]])[:,:,None]
    s=summarize(x)
    assert s['mean'][0]==pytest.approx(.92)
    assert s['sd_across_training_seed_means'][0]==pytest.approx(.02)
    assert s['sd_across_historical_day_means'][0]==pytest.approx(.6/np.sqrt(2))
    assert summarize(x+.05-x)['sd_across_training_seed_means'][0]==pytest.approx(0.)


def test_pooled_statistics_equal_direct_raw_population_moments():
    from PV_experiments.representation.statistics import equal_mixture,from_day_moments
    rng=np.random.default_rng(10); x=rng.normal(size=(5,24,100,195))*5+np.arange(24)[None,:,None,None]*2+np.arange(5)[:,None,None,None]
    a=equal_mixture(x); b=from_day_moments(x.mean(2),x.std(2,ddof=1),100)
    same(a['mean'],x.mean((0,1,2)),'Raw mean');same(b['sd_total'],x.std((0,1,2),ddof=0),'Raw population SD')
    same(a['sd_seed_means'],x.mean((1,2)).std(0,ddof=1),'Training seed SD')
    same(a['sd_total']**2,a['var_seed']+a['var_day_given_seed']+a['var_draw_given_seed_day'],'Total variance')
    assert np.isnan(equal_mixture(x[:1])['sd_seed_means']).all()


def test_cumulative_covariance_and_paired_difference():
    from PV_experiments.representation.statistics import equal_mixture
    x=np.broadcast_to(np.array([0.,10.])[None,None,:,None],(2,2,2,2))
    assert equal_mixture(x)['sd_total'][0]==pytest.approx(5.)
    assert equal_mixture(x.cumsum(-1))['sd_total'][-1]==pytest.approx(10.)
    assert equal_mixture((x+3)-x)['sd_total'][-1]==pytest.approx(0.)


def test_circular_azimuth_not_linear_average():
    from PV_experiments.representation.statistics import circular_stats
    mean,r=circular_stats([350.,10.]);assert min(abs(mean),abs(mean-360))<1e-8;assert r==pytest.approx(np.cos(np.deg2rad(10)))
    mean,r=circular_stats([0.,180.]);assert np.isnan(mean)


def test_report_inventory_detects_modified_figure(tmp_path):
    from PV_experiments.scientific_checks import report_inventory,check_report_inventory
    f=tmp_path/'figure.pdf';f.write_bytes(b'fixture'); manifest={'output_sha256':report_inventory(tmp_path)}
    check_report_inventory(tmp_path,manifest);f.write_bytes(b'changed')
    with pytest.raises(ValueError,match='altered'):check_report_inventory(tmp_path,manifest)


@pytest.mark.parametrize('bad_sd',[10000.,20.])
def test_impossible_power_sd_rejected_even_with_consistent_pooled_tables(mc,bad_sd):
    from MBPO_workspace.src.curve_statistics import aggregate_mc_power_profiles
    path,days,M,_,_=mc
    f=pd.read_csv(path/'daily_power_mean_std_by_utc_step.csv')
    assert f.loc[0,'power_mean_w']==11.
    f.loc[0,'power_sd_w']=bad_sd
    f.to_csv(path/'daily_power_mean_std_by_utc_step.csv',index=False)
    f[f.scenario_id==days[0]].to_csv(path/'first_day_power_mean_sd.csv',index=False)
    pd.DataFrame(aggregate_mc_power_profiles(f.to_dict('records'),M)).to_csv(path/'mc_pooled_power_mean_sd_by_utc_step.csv',index=False)
    with pytest.raises(ValueError,match='Impossible nonnegative power moments'):check(mc)


def test_power_sd_feasibility_boundary(mc):
    from MBPO_workspace.src.curve_statistics import aggregate_mc_power_profiles
    path,days,M,_,_=mc
    f=pd.read_csv(path/'daily_power_mean_std_by_utc_step.csv')
    f.loc[0,'power_sd_w']=np.sqrt(M)*f.loc[0,'power_mean_w']
    f.to_csv(path/'daily_power_mean_std_by_utc_step.csv',index=False)
    f[f.scenario_id==days[0]].to_csv(path/'first_day_power_mean_sd.csv',index=False)
    pd.DataFrame(aggregate_mc_power_profiles(f.to_dict('records'),M)).to_csv(path/'mc_pooled_power_mean_sd_by_utc_step.csv',index=False)
    check(mc)  # necessary bound, not proof that these moments came from the raw draws


@pytest.fixture
def comparable_runs(tmp_path):
    keys=('shared_sac_config_sha256','resolved_mdp_sha256','scenario_protocol_sha256','horizon','final_epoch','final_env_steps','validation_scenario_ids','test_scenario_ids','state_dim','action_dim','observation_mode','energy_objective','movement_cost_mode','movement_penalty','power_model','evaluation_interval_episodes','target_fraction','sustain_checkpoints')
    runs=[]
    for alg in ('sac','mbpo'):
        for seed in (0,1):
            root=tmp_path/f'{alg}{seed}';root.mkdir();cfg={'algorithm':alg,'lr':.001};dump(root/'config.json',cfg)
            dump(root/'training_identity.json',{'runtime':{'kind':'training','sources':{'trainer.py':'source-A'},'dependencies':{'numpy':'version-A'}}})
            meta=dict.fromkeys(keys,'shared');meta.update(algorithm=alg,seed=seed,config_sha256=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest())
            runs.append(dict(root=root,metadata=meta))
    return runs


def test_comparability_accepts_intentionally_distinct_algorithm_configs(comparable_runs):
    from PV_experiments.scientific_checks import verify_run_comparability
    verify_run_comparability(comparable_runs)


@pytest.mark.parametrize('field',['sources','dependencies'])
def test_comparability_rejects_runtime_mismatch(comparable_runs,field):
    from PV_experiments.scientific_checks import verify_run_comparability
    p=comparable_runs[-1]['root']/'training_identity.json';v=json.loads(p.read_text());v['runtime'][field]={'different':'version-B'};dump(p,v)
    with pytest.raises(ValueError,match='Incompatible frozen training runtime'):verify_run_comparability(comparable_runs)


def test_comparability_rejects_individually_consistent_different_mbpo_config(comparable_runs):
    from PV_experiments.scientific_checks import verify_run_comparability
    run=comparable_runs[-1];cfg={'algorithm':'mbpo','lr':.002};dump(run['root']/'config.json',cfg)
    run['metadata']['config_sha256']=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()
    with pytest.raises(ValueError,match='Incompatible mbpo configurations'):verify_run_comparability(comparable_runs)


def test_comparability_rejects_metadata_config_hash_mismatch(comparable_runs):
    from PV_experiments.scientific_checks import verify_run_comparability
    comparable_runs[-1]['metadata']['config_sha256']='wrong'
    with pytest.raises(ValueError,match='configuration hash mismatch'):verify_run_comparability(comparable_runs)


def test_comparability_rejects_shared_contract_mismatch(comparable_runs):
    from PV_experiments.scientific_checks import verify_run_comparability
    comparable_runs[-1]['metadata']['resolved_mdp_sha256']='different'
    with pytest.raises(ValueError,match='Incompatible shared experiment configuration'):verify_run_comparability(comparable_runs)
