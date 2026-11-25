"""Data-free release-contract regressions."""
from pathlib import Path
import numpy as np
import pytest
from mbpo.audit_support import atomic_json, episode_commit, recover_history, sha_file
from mbpo.audit_release import (checkpoint_identity,load_verified_checkpoint,seal_evaluation,
    require_evaluation_seal,verify_loaded_policies,replay_at_confirmation,equivalent_pose_metrics,
    completed_training_files)


def test_committed_optimizer_update_count(tmp_path):
    p=tmp_path/'episodes.csv'; p.write_text('epoch,env_steps,updates\n1,195,0\n2,390,20\n')
    with pytest.raises(ValueError,match='optimizer update'):episode_commit(p,2,390,195,updates=19)
    ck=episode_commit(p,2,390,195,updates=20)
    with pytest.raises(ValueError,match='optimizer update'):recover_history(p,ck,2,390,195,updates=19)


def test_wrong_seed_and_epoch_even_if_digest_matches(tmp_path):
    import torch
    p=tmp_path/'latest.pt'; payload=dict(schema=2,algorithm='sac',seed=1,epoch=50,env_steps=9750,replay={},scenario_protocol_sha256='split')
    torch.save(payload,p)
    meta=dict(training_status='completed',algorithm='sac',seed=0,final_epoch=50,final_env_steps=9750,horizon=195,scenario_protocol_sha256='split',final_checkpoint_sha256=sha_file(p))
    atomic_json(tmp_path/'run_metadata.json',meta)
    with pytest.raises(ValueError,match='algorithm/seed/epoch'):load_verified_checkpoint(p,'sac',expected_seed=0,expected_epoch=50,expected_sha=sha_file(p))
    payload['seed']=0;payload['epoch']=49;torch.save(payload,p);meta['final_checkpoint_sha256']=sha_file(p);atomic_json(tmp_path/'run_metadata.json',meta)
    with pytest.raises(ValueError,match='algorithm/seed/epoch'):load_verified_checkpoint(p,'sac',expected_seed=0,expected_epoch=50,expected_sha=sha_file(p))
    payload['epoch']=50;torch.save(payload,p);meta['final_checkpoint_sha256']=sha_file(p);atomic_json(tmp_path/'run_metadata.json',meta)
    loaded,h=load_verified_checkpoint(p,'sac',expected_seed=0,expected_epoch=50,expected_sha=sha_file(p));assert loaded['seed']==0 and h==sha_file(p)
    with pytest.raises(ValueError,match='seed differs'):load_verified_checkpoint(p,'sac',expected_seed=1)
    with pytest.raises(ValueError,match='algorithm/seed/epoch'):checkpoint_identity(payload,'mbpo',0,50,9750)


def test_evaluation_seal_binds_outputs_and_policy(tmp_path):
    for name in ('all_day_draw_method.csv','daily_power_mean_std_by_utc_step.csv','mc_pooled_power_mean_sd_by_utc_step.csv','first_day_power_mean_sd.csv'):(tmp_path/name).write_text('x,y\n1,2\n')
    identity={'mode':'validation','seed':2,'final_epoch':50,'sac_checkpoint_sha256':'a','mbpo_checkpoint_sha256':'b'}
    summary={'loaded_policies':{'sac':{'sha256':'a','seed':2,'epoch':50},'mbpo':{'sha256':'b','seed':2,'epoch':50}}}
    atomic_json(tmp_path/'mc_summary.json',summary);verify_loaded_policies(summary,identity)
    seal_evaluation(tmp_path,identity);require_evaluation_seal(tmp_path,identity)
    (tmp_path/'all_day_draw_method.csv').write_text('x,y\n1,3\n')
    with pytest.raises(ValueError,match='artifacts changed'):require_evaluation_seal(tmp_path,identity)
    summary['loaded_policies']['sac']['seed']=9
    with pytest.raises(ValueError,match='intended checkpoint'):verify_loaded_policies(summary,identity)


def test_replay_confirmation_and_censoring():
    replay=[dict(seed=s,env_steps=t,synthetic_samples_used=t//2) for s in (0,1) for t in (10,20)]
    rows=[dict(algorithm='mbpo',seed=0,restricted_steps=10,confirmed=True),dict(algorithm='mbpo',seed=1,restricted_steps=20,confirmed=False),dict(algorithm='sac',seed=0,restricted_steps=20,confirmed=False)]
    a,b=replay_at_confirmation(replay,rows);assert a['synthetic_samples_used']==5 and b['synthetic_samples_used']==10
    assert a['accounting_endpoint']=='confirmation' and b['accounting_endpoint']=='censoring_budget'


def test_equivalent_pose_diagnostic():
    common=dict(scenario_id='d',step=1,next_tilt_norm=1.,next_azimuth_sin=0.,next_azimuth_cos=1.,actual_next_power_norm=.05)
    r=equivalent_pose_metrics([dict(common,predicted_next_power_norm=.05),dict(common,predicted_next_power_norm=.055),dict(common,next_tilt_norm=.9,predicted_next_power_norm=.05)])
    assert r['n_equivalent_pose_pairs']==1 and np.isclose(r['alias_predicted_power_rmse_w'],10.) and r['alias_actual_power_max_abs_difference_w']==0


def test_completed_training_seal_rejects_missing_snapshot(tmp_path):
    from mbpo.audit_support import digest
    import csv
    ev=tmp_path/'periodic_evaluation';ev.mkdir();snap=tmp_path/'policy_snapshots';snap.mkdir()
    meta=dict(algorithm='sac',seed=1,horizon=195,evaluation_interval_episodes=50,final_epoch=50,final_env_steps=9750,validation_scenario_ids=['a'])
    for name in ('config.json','resolved_mdp.json','scenario_split.json','training_identity.json','periodic_evaluation/protocol.json','periodic_evaluation/evaluation_references.csv'):(tmp_path/name).write_text('{}')
    def write(path,rows):
        with path.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        return [{k:str(v) for k,v in r.items()} for r in rows]
    write(tmp_path/'episodes.csv',[dict(epoch=i,env_steps=i*195,updates=i) for i in range(1,51)])
    common=dict(algorithm='sac',seed=1,epoch=50,env_steps=9750)
    a=write(ev/'evaluations.csv',[dict(common,updates=50)]);d=write(ev/'evaluation_days.csv',[dict(common,scenario_id='a')]);p=snap/'sac_seed1_epoch00050.pt';p.write_bytes(b'fixture')
    atomic_json(ev/'complete_00050.json',dict(epoch=50,env_steps=9750,days=digest(d),aggregate=digest(a),snapshot_sha256=sha_file(p)))
    assert len(completed_training_files(tmp_path,meta))==11
    p.unlink()
    with pytest.raises(FileNotFoundError):completed_training_files(tmp_path,meta)


def test_stratified_empirical_coverage():
    from mbpo.audit_release import stratified_coverage
    y=np.array([[0.,0.,0.],[1.,1.,1.]])
    q=np.broadcast_to(np.array([-.5,-.25,.25,.5])[:,None,None],(4,2,3)); x=np.zeros((2,10));x[1,0]=.4
    d=stratified_coverage(y,q,x,[{'scenario_id':'a'},{'scenario_id':'b'}])['groups']
    assert d['day:a']['coverage95']==[1.,1.,1.] and d['day:b']['coverage95']==[0.,0.,0.]


def test_final_checkpoint_matches_evaluated_actor(tmp_path):
    import torch
    from mbpo.audit_release import check_final_checkpoint
    (tmp_path/'policy_snapshots').mkdir(); (tmp_path/'training_identity.json').write_text('{}')
    ep=tmp_path/'episodes.csv';ep.write_text('epoch,env_steps,updates\n1,195,2\n')
    meta=dict(algorithm='sac',seed=0,final_epoch=1,final_env_steps=195,horizon=195,scenario_protocol_sha256='split')
    ck=dict(schema=2,algorithm='sac',seed=0,epoch=1,env_steps=195,replay={},scenario_protocol_sha256='split',training_identity_sha256=sha_file(tmp_path/'training_identity.json'),episode_commit=episode_commit(ep,1,195,195,updates=2),agent={'updates':2,'actor':{'w':torch.ones(2)}})
    torch.save(ck,tmp_path/'latest.pt');snap=tmp_path/'policy_snapshots/sac_seed0_epoch00001.pt';torch.save({'actor_state_dict':{'w':torch.ones(2)}},snap)
    check_final_checkpoint(tmp_path,meta)
    torch.save({'actor_state_dict':{'w':torch.zeros(2)}},snap)
    with pytest.raises(ValueError,match='evaluated actor differs'):check_final_checkpoint(tmp_path,meta)
