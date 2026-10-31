"""Unit tests run offline with Python standard library + pytest (pytest optional)."""
import importlib.util
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
import pytest

FILE = Path(__file__).resolve().parents[1]/'scripts'/'stage5_evidence_audit.py'
spec = importlib.util.spec_from_file_location('evidence_audit', FILE)
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)


def episodes():
    rows=[]
    for day in ('2022-04-18','2022-07-03'):
        for method,n in [('sac',.62),('mbpo',.70),('fixed_no_motion',.65)]:
            rows.append(dict(scenario_id=day,seed='42',method=method,gross_energy_kwh=str(n),
                net_energy_kwh=str(n),movement_cost_kwh='0',initial_tilt_deg='30',initial_azimuth_deg='180'))
    return rows

def trajectories():
    rows=[]
    for r in episodes():
        n=float(r['net_energy_kwh'])
        for step in range(1,118):
            rows.append(dict(scenario_id=r['scenario_id'],seed=r['seed'],method=r['method'],
                step=str(step),power_model='pvwatts_sapm',power_w=str(12000*n/117),
                energy_kwh=str(n/117),net_step_kwh=str(n/117),movement_cost_kwh='0',
                tilt_deg='30',azimuth_deg='180',action_tilt='0',action_azimuth='0',
                utc_iso=(datetime.fromisoformat(r['scenario_id']).replace(hour=12,tzinfo=timezone.utc)
                       +timedelta(minutes=5*step)).isoformat(),
                solar_zenith_deg='50',solar_azimuth_deg='175',dni_wm2='900',dhi_wm2='100',
                ghi_wm2='700',ambient_temperature_c='25',wind_speed_m_s='2'))
    return rows

def test_paired_daily_comparison():
    d,rows=a.audit_eval_episodes(episodes())
    assert d['paired_episode_count']==2
    assert d['distinct_days']==2
    assert math.isclose(d['comparisons']['sac_minus_fixed']['daily_mean_kwh'],-.03)
    assert math.isclose(d['comparisons']['mbpo_minus_fixed']['daily_mean_kwh'],+.05)
    assert rows[0]['initial_tilt_deg']==30

def test_duplicate_and_missing_methods_are_rejected():
    e=episodes()
    with pytest.raises(ValueError,match='Duplicate'):
        a.audit_eval_episodes(e+e[:1])
    with pytest.raises(ValueError,match='Missing paired'):
        a.audit_eval_episodes(e[:-1])

def test_mismatched_initial_pose_is_rejected():
    e=episodes();e[1]['initial_tilt_deg']='45'
    with pytest.raises(ValueError,match='Initial pose mismatch'):
        a.audit_eval_episodes(e)

def test_reward_identity_is_rejected_if_corrupted():
    e=episodes();e[0]['net_energy_kwh']='.5'
    with pytest.raises(ValueError,match='Net energy identity'):
        a.audit_eval_episodes(e)

def test_detailed_power_action_and_pose_contracts():
    e,paired=a.audit_eval_episodes(episodes());rr=trajectories()
    out=a.audit_detailed(rr,paired)
    assert out['validated_trajectories']==6
    assert out['validated_transitions']==6*117
    rr[20]['tilt_deg']='40'
    with pytest.raises(ValueError,match='Action/tilt inconsistency'):
        a.audit_detailed(rr,paired)

def test_detailed_model_mismatch_is_rejected():
    e,paired=a.audit_eval_episodes(episodes());rr=trajectories();rr[0]['power_model']='legacy'
    with pytest.raises(ValueError,match='PVWatts'):
        a.audit_detailed(rr,paired)

def test_detailed_complete_horizon_is_required():
    e,paired=a.audit_eval_episodes(episodes());rr=trajectories()[:-1]
    with pytest.raises(ValueError,match='Incomplete'):
        a.audit_detailed(rr,paired)

# Stage 5.1 integrity regressions: these are read-only artifact checks, not
# demonstrations of a full SAC shadow-update or checkpoint policy rerun.

def test_invalid_utc_minute_is_rejected():
    _,pairs=a.audit_eval_episodes(episodes()); rr=trajectories()
    rr[60]['utc_iso']='2022-04-18T00:060:00+00:00'
    with pytest.raises(ValueError,match='invalid ISO timestamp|expected first|non-uniform'):
        a.audit_detailed(rr,pairs)


def test_nonuniform_five_minute_grid_is_rejected():
    _,pairs=a.audit_eval_episodes(episodes());rr=trajectories()
    rr[7]['utc_iso']='2022-04-18T12:40:01+00:00'
    with pytest.raises(ValueError,match='non-uniform timestep'):
        a.audit_detailed(rr,pairs)


def test_non_utc_timezone_is_rejected():
    _,pairs=a.audit_eval_episodes(episodes());rr=trajectories()
    rr[0]['utc_iso']='2022-04-18T12:05:00+01:00'
    with pytest.raises(ValueError,match='explicit UTC'):
        a.audit_detailed(rr,pairs)


def test_valid_paired_weather_and_corruption_detected():
    _,pairs=a.audit_eval_episodes(episodes());rr=trajectories()
    out=a.audit_detailed(rr,pairs)
    assert out['recorded_weather_parity']['paired_weather_step_groups']==234
    assert out['validated_methods']==['sac','mbpo','fixed_no_motion']
    rr[117]['dni_wm2']='901'
    with pytest.raises(ValueError,match='weather mismatch'):
        a.audit_detailed(rr,pairs)


def test_nonzero_movement_cost_reward_identity():
    ep=episodes();ep[0]['movement_cost_kwh']='0.02'
    ep[0]['net_energy_kwh']='0.60'
    audit,paired=a.audit_eval_episodes(ep)
    assert paired[0]['sac_net_kwh']==pytest.approx(0.60)
    ep[0]['net_energy_kwh']='0.62'
    with pytest.raises(ValueError,match='Net energy identity'):
        a.audit_eval_episodes(ep)


def test_oracle_separate_gross_validation_and_bound():
    eps=episodes(); rr=trajectories()
    for day in ('2022-04-18','2022-07-03'):
        eps.append(dict(scenario_id=day,seed='42',method='poa_instant_ideal',
                        gross_energy_kwh='0.8',net_energy_kwh='',movement_cost_kwh='',
                        initial_tilt_deg='30',initial_azimuth_deg='180'))
        for step in range(1,118):
            sample=next(r for r in rr if r['scenario_id']==day and r['step']==str(step))
            o=dict(sample)
            o.update(method='poa_instant_ideal',power_w=str(12000*0.8/117),
                     energy_kwh=str(0.8/117),net_step_kwh='',action_tilt='',action_azimuth='')
            rr.append(o)
    _,pairs=a.audit_eval_episodes(eps)
    result=a.validate_oracle(eps,rr,pairs)
    assert result['oracle_steps']==234
    assert result['status']=='PASS_RECORDED_CONSISTENCY_ONLY'
    oracle=next(r for r in rr if r['method']=='poa_instant_ideal')
    oracle['power_w']='0'
    with pytest.raises(ValueError,match='Oracle power-to-energy'):
        a.validate_oracle(eps,rr,pairs)


def test_weighted_summary_reproduction_and_mismatch_rejected():
    _,paired=a.audit_eval_episodes(episodes())
    p={'strata':[dict(selected_scenario_id='2022-04-18',population_days=1,sampled_days=1),
                 dict(selected_scenario_id='2022-07-03',population_days=3,sampled_days=1)],
       'population_count':4}
    summary={'methods':{m:{'mean_net_kwh':n} for m,n in
               [('sac',.62),('mbpo',.70),('fixed_no_motion',.65)]},
             'paired_net_kwh':{'mbpo_minus_fixed_no_motion':{'mean_difference_kwh':.05},
                               'mbpo_minus_sac':{'mean_difference_kwh':.08},
                               'sac_minus_fixed_no_motion':{'mean_difference_kwh':-.03}}}
    ret=a.weighted_crosscheck(paired,p,summary)
    assert ret['weight_sum']==4
    assert ret['weighted_differences_kwh']['mbpo_minus_sac']==pytest.approx(.08)
    summary['methods']['mbpo']['mean_net_kwh']=.8
    with pytest.raises(ValueError,match='Weighted mbpo mean'):
        a.weighted_crosscheck(paired,p,summary)


def test_stage4_checkpoint_alpha_vs_stage5_csv_provenance():
    report=a.training_provenance_check({'episodes':2000,'final_alpha':.02241805},
         {'epoch':2000},{'last_alpha':.0155801819637})
    assert report['status']=='FAIL_CHECKPOINT_HISTORY_MISMATCH'
    assert 'FINAL_ALPHA_MISMATCH' in report['issues'][0]
    good=a.training_provenance_check({'episodes':500,'final_alpha':.012782987207},
         {'epoch':500},{'last_alpha':.012782987207})
    assert good['status']=='MATCHES_REPORTED_STAGE4_SNAPSHOT'


def test_real_replay_end_window_distribution_is_descriptive_not_bug():
    recent=[{'scenario_id':f'2022-01-{i:02d}'} for i in range(1,16)]
    r=a.replay_recency_diagnostic(recent,capacity=1170,horizon=117)
    assert r['at_most_recent_episodes_covered']==10
    assert r['approximate_distinct_recent_scenarios']==10
    assert 'not proof' in r['warning']


def _deep_minimal():
    checked={'count':4,'terminal_mismatch_count':0}
    probe={'alpha':.012,'critic_TD_abs_error_Q1':{'mean':1.2}}
    shadow={'status':'SHADOW_UPDATE_PASSED','training_step_metrics':{'critic_loss':1.,'actor_loss':2.}}
    match={'checked_actions':4,'checkpoint_actions_match_saved_evaluation':True,
           'max_abs_normalized_action_error':0.000001}
    return {'schema':'sac-mbpo-deep-v1',
            'checkpoint':{'sac':{'epoch':1},'mbpo':{'epoch':1}},
            'config':{'physics':{},'shared_sac_algo':{'min_alpha':.001,'reward_scale':100}},
            'replay_contract':{'sac':dict(checked),'mbpo_real':dict(checked),'mbpo_model':dict(checked)},
            'learning_probes':{'sac_real':dict(probe),'mbpo_real':dict(probe),'mbpo_synthetic':dict(probe)},
            'shadow_optimizer_steps':{'sac':dict(shadow),'mbpo':dict(shadow)},
            'evaluation_action_replay':{'sac':dict(match),'mbpo':dict(match)},
            'mbpo_model_fit_to_real_replay':{'replay_skill_vs_persistence':[0,0,.02]},
            'mbpo_mixture':{'critical_caveat':'not independent'},'important_limitations':[]}


def test_stage4_reported_action_replay_mismatch_is_not_silently_accepted():
    d=_deep_minimal()
    assert not any('Action-replay' in x for x in a.audit_deep(d,[])['issues'])
    d['evaluation_action_replay']['sac']['checkpoint_actions_match_saved_evaluation']=False
    result=a.audit_deep(d,[])
    assert any('Action-replay parity not established' in x for x in result['issues'])
    assert result['stage4_claims_status']=='REPORTED_IN_STAGE4_NOT_INDEPENDENTLY_REPLAYED'


def test_stage4_shadow_nonfinite_and_failed_status_reported():
    d=_deep_minimal()
    d['shadow_optimizer_steps']['sac']['status']='FAILED'
    d['shadow_optimizer_steps']['mbpo']['training_step_metrics']['critic_loss']=float('nan')
    findings=a.audit_deep(d,[])['issues']
    assert any('Shadow optimizer failed: sac' in x for x in findings)
    assert any('Shadow step metrics missing or nonfinite: mbpo' in x for x in findings)


def test_train_eval_overlap_detection():
    training={'2018-05-01','2018-05-02'};evals={'2018-05-02','2018-05-03'}
    overlapping=a.train_eval_overlap(training,evals)
    assert overlapping['count']==1
    assert overlapping['sample_ids']==['2018-05-02']


def test_weighted_oracle_gross_mean_catches_corruption():
    _,pairs=a.audit_eval_episodes(episodes())
    selection={'strata':[{'selected_scenario_id':day,'population_days':2,'sampled_days':1}
             for day in ('2022-04-18','2022-07-03')],'population_count':4}
    summary={'methods':{m:{'mean_net_kwh':v} for m,v in
                [('sac',.62),('mbpo',.7),('fixed_no_motion',.65)]},
             'paired_net_kwh':{'mbpo_minus_fixed_no_motion':{'mean_difference_kwh':.05},
                               'mbpo_minus_sac':{'mean_difference_kwh':.08},
                               'sac_minus_fixed_no_motion':{'mean_difference_kwh':-.03}}}
    summary['methods']['poa_instant_ideal']={'mean_gross_kwh':.80}
    ep=episodes()+[dict(scenario_id=day,seed='42',method='poa_instant_ideal',gross_energy_kwh='.80')
                   for day in ('2022-04-18','2022-07-03')]
    got=a.weighted_crosscheck(pairs,selection,summary,episodes_rows=ep)
    assert got['weighted_oracle_gross_kwh']==pytest.approx(.8)
    summary['methods']['poa_instant_ideal']['mean_gross_kwh']=.81
    with pytest.raises(ValueError,match='Weighted POA oracle gross'):
        a.weighted_crosscheck(pairs,selection,summary,episodes_rows=ep)
