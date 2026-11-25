"""Integration regressions: execute only inside the patched real repository."""
import csv
import numpy as np
import pytest
from mbpo.audit_support import atomic_json


def test_terminal_producing_train_state_is_rollout_eligible():
    from mbpo.geometry10 import build,TAU
    from MBPO_workspace.src.replay import ScenarioReplayBuffer
    rb=ScenarioReplayBuffer(16,seed=0)
    s=build(50,10,5,20,180,95,210,1/195)
    n=build(0,0,0,20,180,96,211,0.)
    rb.add(s,np.zeros(2),0.,n,True,'train_day')
    rb.add(s,np.zeros(2),0.,n,True,'holdout_day')
    states,ids=rb.rollout_states(5,seed=7,with_scenarios=True)
    assert len(states)==5 and all(i=='train_day' for i in ids) and np.all(states[:,TAU]>0)


def test_raw_inf_fails_before_clip(monkeypatch):
    from mbpo.geometry10 import build
    from MBPO_workspace.src.dynamics import DynamicsEnsemble
    model=DynamicsEnsemble(num_models=2,num_elites=2,hidden_dim=8);model.fitted=True
    x=build(500,100,90,20,180,60,140,1.)[None,:]
    def bad(*args):return np.full((2,1,2),np.inf,np.float32),np.ones((2,1,2),np.float32)
    monkeypatch.setattr(model,'_members',bad)
    with pytest.raises(FloatingPointError,match='raw weather'):
        model.predict_delta(x,np.zeros((1,2),np.float32),stochastic=False)


def test_periodic_completion_requires_daily_aggregate_and_snapshot(tmp_path):
    from SAC_workspace.src.training.experiment_protocol import PeriodicEvaluator
    e=PeriodicEvaluator.__new__(PeriodicEvaluator); e.eval_dir=tmp_path/'eval';e.eval_dir.mkdir();e.snapshot_dir=tmp_path/'snap';e.snapshot_dir.mkdir()
    e.algorithm='sac';e.seed=0;e.validation_ids=['a','b'];e.protocol={'evaluation_interval_episodes':50}
    def write(name,rows):
        with (e.eval_dir/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    common=dict(algorithm='sac',seed=0,epoch=50,env_steps=9750)
    write('evaluation_days.csv',[dict(common,scenario_id='a',net_energy_kwh=1.),dict(common,scenario_id='b',net_energy_kwh=2.)])
    write('evaluations.csv',[dict(common,mean_net_energy_kwh=1.5)])
    snap=e.snapshot_dir/'sac_seed0_epoch00050.pt';snap.write_bytes(b'test fixture snapshot')
    atomic_json(e.eval_dir/'complete_00050.json',e._evaluation_manifest(50,9750))
    assert e.has_evaluation(epoch=50,env_steps=9750)
    snap.unlink();assert not e.has_evaluation(epoch=50,env_steps=9750)
