"""Data-free regression tests for the Geometry10 reliability and diagnostics patch."""
from types import SimpleNamespace
import json
import subprocess
import sys
import numpy as np
import pytest
from mbpo.audit_support import (episode_commit,recover_history,expected_steps,target_event,
    power_envelope,envelope_counts,hierarchical,counterfactual_metrics,exclusive_lock,
    preserve_global_rng)


def test_history_recovery_and_repeated_identical_tail(tmp_path):
    p=tmp_path/'episodes.csv'; prefix=b'epoch,env_steps,return_kwh\n1,195,1.1\n'; tail=b'2,390,1.2\n3,585,'
    p.write_bytes(prefix); ck=episode_commit(p,1,195,195)
    for _ in range(2):
        p.write_bytes(prefix+tail);recover_history(p,ck,1,195,195);assert p.read_bytes()==prefix
    ledger=json.loads((tmp_path/'recovery_events.json').read_text())
    assert len(ledger)==2 and all(e['status']=='applied' and e['known_repeated_transitions']==195 for e in ledger)


def test_history_corruption_refuses_without_mutation(tmp_path):
    p=tmp_path/'episodes.csv'; p.write_text('epoch,env_steps\n1,195\n')
    ck=episode_commit(p,1,195,195);p.write_text('epoch,env_steps\n1,196\n');before=p.read_bytes()
    with pytest.raises(ValueError):recover_history(p,ck,1,195,195)
    assert p.read_bytes()==before


def test_epoch_zero_recovery(tmp_path):
    p=tmp_path/'episodes.csv'; ck=episode_commit(p,0,0,195)
    p.write_text('epoch,env_steps\n1,195\n');recover_history(p,ck,0,0,195);assert not p.exists()
    assert json.loads((tmp_path/'recovery_events.json').read_text())[0]['known_repeated_transitions']==195


def test_committed_history_requires_full_schedule(tmp_path):
    p=tmp_path/'episodes.csv'; p.write_text('epoch,env_steps\n1,195\n3,390\n')
    with pytest.raises(ValueError):episode_commit(p,2,390,195)


def test_event_and_nonmultiple_final_schedule():
    steps=expected_steps(dict(horizon=195,evaluation_interval_episodes=50,final_epoch=125,final_env_steps=24375))
    assert steps.tolist()==[9750,19500,24375]
    assert target_event([.8,.95,.97],steps,.9,2)==(19500,24375)
    assert target_event([.95,.8,.95],steps,.9,2)==(None,None)
    assert target_event([.95,.8,.95],steps,.9,1)==(9750,9750)


def test_operational_envelope_not_stc_cap():
    # horizontal, sun at zenith, direct + diffuse = 1000 W/m2; no reflected term
    s=np.array([[.45,.05,.09,0.,0.,1.,0.,0.,1.,.5]],dtype=float)
    assert np.isclose(power_envelope(s,1.,.18,0.)[0],180.)
    assert power_envelope(s,1.,.18,-.004)[0]>180.
    env=SimpleNamespace(area=1.,efficiency=.18,power_model='pvwatts_sapm',gamma_pdc=-.004)
    bad=s.copy();bad[0,2]=.5; sat=s.copy();sat[0,0]=1.
    assert envelope_counts(np.vstack([s,bad,sat]),env)=={'envelope_checked':2,'envelope_violations':1,'envelope_irradiance_saturated':1}


class FakeEnsemble:
    def __init__(self): self.rng=np.random.default_rng(4)
    def predict_delta(self,states,actions,stochastic=True,*,rng=None,record=True):
        assert rng is not self.rng and not record
        w=rng.normal(0,.02,(len(states),2)); noise=rng.normal(0,.01,len(states))
        p=2*w[:,0]+noise+actions[:,0]*.02
        nxt=np.clip(states[:,:3]+np.column_stack([w,p]),0,1)
        return nxt-states[:,:3],np.zeros(len(states))


def test_hierarchical_common_random_numbers_and_rng_isolation():
    e=FakeEnsemble();before=json.dumps(e.rng.bit_generator.state,sort_keys=True)
    x=np.zeros((4,10),np.float32);x[:,:3]=.4; a=np.zeros((4,2),np.float32);b=a.copy();b[:,0]=1
    m,sd,q=hierarchical(e,x,a,seed=9,samples=128,chunk=2)
    m2,sd2,q2=hierarchical(e,x,b,seed=9,samples=128,chunk=2)
    assert np.array_equal(m[:,:2],m2[:,:2]);assert np.allclose(m2[:,2]-m[:,2],.02)
    assert q.shape==(4,4,3) and np.all(np.diff(q,axis=0)>=0)
    assert np.all(sd[:,2]>0);assert json.dumps(e.rng.bit_generator.state,sort_keys=True)==before


def test_hierarchical_envelope_counts_all_draws():
    e=FakeEnsemble();x=np.zeros((2,10),np.float32);x[:,5]=x[:,8]=1; a=np.zeros((2,2),np.float32)
    env=SimpleNamespace(area=1.,efficiency=.18,power_model='legacy');counts={}
    hierarchical(e,x,a,seed=3,samples=64,envelope_template=x,env=env,envelope_totals=counts)
    assert counts['envelope_checked']+counts['envelope_irradiance_saturated']==128


def test_action_regret_and_difference_error():
    rows=[dict(scenario_id='d',step=0,action_type='zero',actual_reward_kwh=.001,predicted_reward_kwh=.002),
          dict(scenario_id='d',step=0,action_type='move',actual_reward_kwh=.002,predicted_reward_kwh=.001)]
    r=counterfactual_metrics(rows);assert r['pairwise_agreement']==0 and np.isclose(r['mean_one_step_regret_kwh'],.001)
    assert np.isclose(r['reward_difference_rmse_kwh'],.002)


@pytest.mark.skipif(sys.platform=='win32',reason='Managed workflow uses POSIX locks under WSL/Linux')
def test_lock_survives_manager_exit_until_child_exits(tmp_path):
    path=tmp_path/'job.lock'
    with exclusive_lock(path) as fd:
        proc=subprocess.Popen([sys.executable,'-c','import time; time.sleep(1)'],pass_fds=(fd,))
    try:
        with pytest.raises(RuntimeError):
            with exclusive_lock(path): pass
    finally:proc.wait(timeout=5)
    with exclusive_lock(path):pass


def test_evaluation_rng_isolation():
    import random,torch
    random.seed(8);np.random.seed(8);torch.manual_seed(8)
    p=random.getstate();n=np.random.get_state();t=torch.get_rng_state().clone()
    with preserve_global_rng():random.random();np.random.random();torch.rand(3)
    assert random.getstate()==p;assert np.array_equal(np.random.get_state()[1],n[1]);assert torch.equal(torch.get_rng_state(),t)
