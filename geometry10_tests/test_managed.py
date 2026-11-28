"""Tests to run inside the installed workspace with its Torch/Gym/pvlib runtime."""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from mbpo.geometry10 import (build, STATE_DIM, TAU, PANEL_AZ, SOLAR, solar_features,
    check_batch, angular_rmse_deg)


def valid_state(i=0,day=0):
    # Structured synthetic examples, not claims about trained policies.
    return build(400+i,100+i*.1,90+i*.2,30,180+i*.1,65-day,120+i*.1,(195-i)/195.)


def test_config_routes_and_dimensions():
    from SAC_workspace.src.config.config import load_config as sac_config
    from MBPO_workspace.src.config import load_config as mbpo_config
    from SAC_workspace.src.training.experiment_protocol import STATE_DIM as protocol_dim
    s=sac_config(require_data=False,power_model='legacy')
    m=mbpo_config(require_data=False,power_model='legacy')
    assert s.to_dict()==m.sac.to_dict()
    assert s.train_env['observation_mode']==s.eval_env['observation_mode']=='geometry10'
    assert s.train_env['movement_penalty']==s.eval_env['movement_penalty']==1.
    assert s.train_env['movement_cost_mode']==s.eval_env['movement_cost_mode']=='geometry'
    assert protocol_dim==STATE_DIM and m.model.rollout_length==1
    assert s.algo['gamma']==1. and s.algo['reward_scale']==100.
    assert 'conf_geometry10.py' in str(s.config_paths)


def test_actor_replay_optimizer_and_checkpoint():
    from SAC_workspace.src.agents.sac_agent import SACAgent
    from SAC_workspace.src.training.replay_buffer import ReplayBuffer
    torch.manual_seed(0)
    agent=SACAgent(hidden_sizes=(16,16))
    buf=ReplayBuffer(64,seed=1)
    for i in range(32):
        s,n=valid_state(i),valid_state(i+1)
        buf.add(s,np.array([.1,-.1]),.005,n,False)
    a=agent.act(valid_state(),deterministic=True)
    assert a.shape==(2,) and np.all(np.abs(a)<=1.)
    metrics=agent.update(buf.sample(16))
    assert all(np.isfinite(v) for v in metrics.values())
    new=SACAgent(hidden_sizes=(16,16));new.load_state_dict(agent.state_dict())
    assert np.array_equal(new.act(valid_state(),True),agent.act(valid_state(),True))
    rb=ReplayBuffer(64);rb.load_state_dict(buf.state_dict());assert rb.states.shape==(64,10)
    bad=dict(agent.state_dict());bad['state_dim']=6
    with pytest.raises(ValueError):new.load_state_dict(bad)
    with pytest.raises(ValueError):agent.act(np.zeros(6),True)


def test_scenario_replay_and_ensemble_fit():
    from MBPO_workspace.src.replay import ScenarioReplayBuffer, mix_batches
    from MBPO_workspace.src.dynamics import DynamicsEnsemble
    from SAC_workspace.src.training.replay_buffer import ReplayBuffer
    rb=ScenarioReplayBuffer(256,seed=0)
    for day in range(3):
        for i in range(40):
            rb.add(valid_state(i,day),np.array([.1,.2]),.005,valid_state(i+1,day),False,f'day{day}')
    states,sids=rb.rollout_states(64,seed=10,with_scenarios=True)
    assert states.shape==(64,STATE_DIM)
    assert all(rb._validation_roles[s]=='train' for s in sids)
    for s,sid in zip(states,sids):
        assert np.isclose(s[6],(65-int(sid[-1]))/180.)
    model=DynamicsEnsemble(num_models=2,num_elites=2,hidden_dim=8,seed=0)
    model.fit(rb,epochs=2,patience=2,batch_size=16,max_samples=64)
    a=np.zeros((len(states),2),dtype=np.float32)
    delta,score=model.predict_delta(states,a,stochastic=False)
    assert delta.shape==(64,3) and np.isfinite(delta).all() and np.isfinite(score).all()
    new=DynamicsEnsemble(num_models=2,num_elites=2,hidden_dim=8,seed=0)
    new.load_state_dict(model.state_dict())
    assert np.allclose(new.predict_delta(states,a,stochastic=False)[0],delta)
    synthetic=ReplayBuffer(128)
    for s in states:synthetic.add(s,[0,0],.005,s,False)
    batch,real_n,model_n=mix_batches(rb,synthetic,32,.85,np.random.default_rng(0))
    assert batch['states'].shape==(32,10) and real_n+model_n==32 and model_n>0


def test_manager_manifest_and_cli(tmp_path):
    from PV_experiments.core import initialize_experiment
    from PV_experiments.train import build_parser
    from PV_experiments.evaluate import build_parser as eval_parser
    from PV_experiments.represent import parser
    cfg=initialize_experiment('geometry10_contract',epochs=100,seeds=[0,1],dry_run=True)
    assert cfg['observation_mode']=='geometry10' and cfg['state_dim']==10
    assert build_parser().parse_args(['--name','pilot','--epochs','100']).epochs==100
    assert eval_parser().parse_args(['--name','pilot','--mode','smoke']).mode=='smoke'
    assert parser().parse_args(['--name','pilot','--kind','collect']).kind=='collect'


def test_real_physics_and_exact_model_transition():
    """TRAIN-only day; paired fixed realization; compare old and new observations.

    This test deliberately uses actual current and next first-three channels as
    a stub prediction to isolate exact kinematics/ephemeris/reward. It is NOT a
    learned-model accuracy test. No test partition or training is accessed.
    """
    from MBPO_workspace.src.config import load_config
    from MBPO_workspace.src.dynamics import StructuredFakeEnv
    from SAC_workspace.src.environment.environment import SACEnvironment
    from SAC_workspace.src.training.experiment_protocol import build_scenario_protocol
    from mbpo.env.stage_a_pv_tracking import StageAPVTrackingEnv
    spec=json.loads(Path('PostProcessing_MC_patch/examples/hypothetical_uncertainty_5pct.json').read_text())
    cfg=load_config(weather_uncertainty=spec)
    split=build_scenario_protocol(cfg.sac.manifest,cfg.sac.protocol)
    sid=split['train_ids'][len(split['train_ids'])//2]
    kwargs=dict(cfg.sac.eval_env);kwargs['randomize_initial_orientation']=False
    env=SACEnvironment(kwargs)
    old=StageAPVTrackingEnv(**dict(kwargs,observation_mode='compact'))
    class Stub:
        delta=None
        def predict_delta(self,states,actions,stochastic=False):
            return self.delta,np.zeros(len(states),dtype=np.float32)
    stub=Stub();fake=StructuredFakeEnv(stub,env)
    try:
        env.seed(11);old.seed(11);old.set_weather_realization(0)
        s=env.reset(scenario_id=sid,realization_id=0);o=old.reset(scenario_id=sid)
        assert env.observation_space.shape==(10,) and o.shape==(6,)
        assert np.allclose(s[:4],o[:4],atol=1e-7)
        assert env.unwrapped._weather_profile_sha256==old._weather_profile_sha256
        rng=np.random.default_rng(3)
        for k in range(env.horizon):
            a=rng.uniform(-1,1,size=2).astype(np.float32)
            # Match poses reachable under both action contracts.
            target=np.clip(old.tilt+a[0]*old.max_delta_tilt,0,90)
            target_a=np.array([target/45-1,a[1]*old.max_delta_azimuth/180],np.float32)
            n,r,done,info=env.step(target_a);on,ro,odone,oinfo=old.step(a)
            assert np.isclose(r,ro,atol=1e-10) and done==odone
            assert np.isclose(info['power_dc_w'],oinfo['power_dc_w'],atol=1e-5)
            assert np.allclose(n[:4],on[:4],atol=1e-7)
            assert np.allclose(n[SOLAR],solar_features(info['solar_zenith_deg'],info['solar_azimuth_deg']),atol=1e-6)
            assert np.isclose(n[TAU],on[5],atol=1e-7)
            # Exact pose, time, ephemeris and movement accounting with stub delta.
            stub.delta=(n[:3]-s[:3])[None,:]
            pred,pr,pd,_,mask=fake.step(s[None,:],target_a[None,:],stochastic=False,scenario_ids=[sid])
            assert mask[0] and bool(pd[0])==done
            assert np.allclose(pred[0],n,atol=4e-5,rtol=0)
            # power normalization/float32 roundoff only
            assert np.isclose(pr[0],r,atol=1e-7,rtol=0)
            s=n
        assert done and s[TAU]==0
    finally:
        env.close();old.close()


def test_fake_rollout_prediction_validation():
    """Known geometry and reward reconstruction, invalid predictions rejected.

    No physical weather lookup: an injected ephemeris grid and model deltas
    isolate the structured transition, not the ensemble's predictive skill.
    """
    from MBPO_workspace.src.dynamics import StructuredFakeEnv
    from types import SimpleNamespace
    class Stub:
        def predict_delta(self,states,actions,stochastic=False):
            return self.delta, self.unc
    stub=Stub()
    ref=SimpleNamespace(horizon=195,unwrapped=SimpleNamespace(
        max_delta_tilt=5.,max_delta_azimuth=10.,movement_penalty=1.,movement_cost_mode='geometry',
        actuator_power_w=30.,slew_rate_tilt_deg_s=1.5,slew_rate_azimuth_deg_s=2.,
        interval_hours=1/12,num_action_steps=195,periods=196))
    fake=StructuredFakeEnv(stub,ref)
    grid=solar_features(np.linspace(45,125,196),np.linspace(60,300,196))
    fake.solar_context.cache={'a':grid}
    s=build(600,100,150,89,359,45,60,1.)[None,:]
    action=np.ones((1,2),np.float32)
    stub.delta=np.array([[0.,0.,.01]],np.float32);stub.unc=np.array([.1],np.float32)
    nxt,r,d,u,mask=fake.step(s,action,stochastic=False,scenario_ids=['a'])
    assert mask[0] and not d[0]
    assert np.allclose(nxt[0,SOLAR],grid[1],atol=1e-7)
    assert nxt[0,3]==1.
    expected=170./12000.-30.*(1./1.5+180./2.)/3.6e6
    assert np.isclose(r[0],expected,atol=1e-7)
    stub.delta=np.array([[0.,0.,np.inf]],np.float32)
    assert not fake.step(s,action,stochastic=False,scenario_ids=['a'])[-1][0]
    stub.delta=np.zeros((1,3),np.float32);stub.unc=np.array([-.1],np.float32)
    assert not fake.step(s,action,stochastic=False,scenario_ids=['a'])[-1][0]
    stub.delta=np.zeros((1,1),np.float32)
    with pytest.raises(ValueError):fake.step(s,action,stochastic=False,scenario_ids=['a'])
