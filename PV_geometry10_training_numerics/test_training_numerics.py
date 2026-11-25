"""Regression checks for the pinned Geometry10 numerical corrections.
No historical weather data are needed. Run from the repository root.
"""
import importlib
import math
import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F
from mbpo.geometry10 import build, circular, azimuth, pose_transition, PANEL_AZ, TAU
from SAC_workspace.src.agents.sac_agent import SquashedGaussianActor, SACAgent


@pytest.mark.parametrize('logstd', [-20., -12., -2., 2.])
def test_actor_logprob_uses_sampled_noise(monkeypatch, logstd):
    model = SquashedGaussianActor(hidden_sizes=(8,))
    linear = nn.Linear(10, 4)
    with torch.no_grad():
        linear.weight.zero_()
        linear.bias.copy_(torch.tensor([1., -1., logstd, logstd]))
    model.net = linear
    noise = torch.tensor([[2., -3.]])
    monkeypatch.setattr(torch, 'randn_like', lambda x: noise.expand_as(x))
    state = torch.zeros((1, 10))
    action, logprob = model(state)
    z = torch.tensor([[1., -1.]]) + math.exp(logstd)*noise
    expected = (-.5*noise.square()-logstd-.5*math.log(2*math.pi)
                -2*(math.log(2.)-z-F.softplus(-2*z))).sum(-1,keepdim=True)
    torch.testing.assert_close(action, torch.tanh(z))
    torch.testing.assert_close(logprob, expected, atol=1e-5, rtol=1e-6)
    logprob.sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in linear.parameters())


def test_deterministic_actor_does_not_consume_rng():
    model = SquashedGaussianActor(hidden_sizes=(8,))
    state = torch.ones(1,10)
    before = torch.get_rng_state().clone()
    a, lp = model(state, deterministic=True)
    a2, no_lp = model(state, deterministic=True, with_logprob=False)
    assert torch.equal(before,torch.get_rng_state()) and no_lp is None
    torch.testing.assert_close(a,a2)
    assert torch.isfinite(lp).all()


def test_actual_sac_terminal_targets_and_one_update():
    agent = SACAgent(hidden_sizes=(8,8))
    s = np.stack([build(100,20,15,30,359,70,1,1.)]*8)
    n = np.stack([build(110,25,17,31,1,69,2,194/195)]*8)
    # An independent terminal fixture: only final-step states terminate.
    s[0,TAU] = 1/195; n[0,TAU] = 0.
    r = np.full((8,1),.002,np.float32)
    d = np.zeros((8,1),np.float32); d[0] = 1
    batch = dict(states=s, actions=np.zeros((8,2),np.float32), rewards=r, next_states=n, dones=d)
    tensor_batch = {k:torch.as_tensor(v) for k,v in batch.items()}
    target = agent.compute_target(tensor_batch)
    torch.testing.assert_close(target[0], torch.tensor([.2]))
    metrics = agent.update(batch)
    assert agent.updates == 1 and all(np.isfinite(v) for v in metrics.values())
    assert agent.alpha >= agent.min_alpha - 1e-8


@pytest.mark.parametrize('persist,error,sign', [(0.,0.,0),(0.,2e-10,-1),(1e-10,2e-10,-1),(1e-10,5e-11,1),(.02,.01,1),(.02,.02,0)])
def test_actual_dynamics_skill_gate(monkeypatch,persist,error,sign):
    from MBPO_workspace.src.dynamics import DynamicsEnsemble
    model = DynamicsEnsemble(num_models=2,num_elites=2,hidden_dim=8)
    s=build(0.,0.,0.,30.,180.,100.,200.,1.)
    states=np.tile(s,(32,1)); nxt=states.copy(); nxt[:,TAU]=194/195
    nxt[:,2]=persist
    class Replay:
        def __len__(self): return 32
        def scenario_split(self, fraction, seed): return np.arange(16),np.arange(16,32)
    replay=Replay();replay.states=states;replay.next_states=nxt;replay.actions=np.zeros((32,2),np.float32);replay.scenario_ids=np.repeat([0,1],16)
    monkeypatch.setattr(model,'_fit_group',lambda *args:([0,1],[0.,0.]))
    def members(nets, elite, inputs, xmean, xstd, ymean, ystd):
        mu=np.zeros((2,len(inputs),len(ymean)),np.float32)
        if len(ymean)==1:mu.fill(persist+error)
        return mu,np.ones_like(mu)
    monkeypatch.setattr(model,'_members',members)
    result=model.fit(replay,epochs=1,patience=1,batch_size=8,max_samples=32)
    for k in ('model_skill_power_vs_persistence','model_skill_all_channels_vs_persistence'):
        assert np.sign(result[k])==sign
        assert (result[k]>0)==(sign>0)


@pytest.mark.parametrize('persist,error,expected_sign', [(0.,0.,0),(0.,1e-10,-1),(1e-10,2e-10,-1),(1e-10,5e-11,1),(.02,.02,0)])
def test_actual_external_skill_metric(persist,error,expected_sign):
    mod=importlib.import_module('MBPO_workspace.workflows.06_model_validation')
    actual=np.full((8,3),persist); predicted=np.full((8,3),persist+error)
    metrics=mod._channel_metrics(actual,predicted)
    for k in mod.CHANNELS:
        assert np.sign(metrics[k]['skill_vs_persistence'])==expected_sign


def test_angle_wrap_and_pose_parity_dense():
    rng=np.random.default_rng(2026); n=40000
    az0=rng.uniform(0,360,n); tilt0=rng.uniform(0,90,n)
    az0[:8]=[359.99999,0.00001,359,1,0,180,270,90]
    tilt0[:8]=[0,90,89,1,90,0,45,30]
    rem=rng.integers(1,196,n); rem[:2]=1
    a=rng.uniform(-1,1,(n,2)).astype(np.float32); a[:4]=[[1,1],[-1,-1],[1,1],[-1,-1]]
    states=np.stack([build(100,20,12,t,g,70,1,k/195) for t,g,k in zip(tilt0,az0,rem)])
    tilt,pair,tau,dt,da,done=pose_transition(states,a,5.,10.,195)
    expected_tilt=np.clip(tilt0+a[:,0].astype(float)*5,0,90)
    expected_az=(az0+a[:,1].astype(float)*10)%360
    np.testing.assert_allclose(tilt*90,expected_tilt,atol=5e-6,rtol=0)
    np.testing.assert_allclose(pair,circular(expected_az),atol=1e-7,rtol=0)
    np.testing.assert_allclose(dt,expected_tilt-tilt0,atol=5e-6,rtol=0)
    np.testing.assert_allclose(da,a[:,1].astype(float)*10,atol=1e-10,rtol=0)
    np.testing.assert_array_equal(done,rem==1)
    np.testing.assert_allclose(tau,(rem-1)/195,atol=1e-12,rtol=0)
    assert np.linalg.norm(circular(359.999)-circular(.001))<4e-5
    np.testing.assert_allclose(np.linalg.norm(pair,axis=1),1,atol=1e-12)
    cost=30*(abs(dt)/1.5+abs(da)/2)/3.6e6
    assert (cost>=0).all() and cost.max()<=30*(5/1.5+10/2)/3.6e6+1e-12


def test_budget_arithmetic_and_batch_fraction():
    from SAC_workspace.src.config.config import load_config as load_sac
    from MBPO_workspace.src.config import load_config as load_mbpo
    sac=load_sac(require_data=False,power_model='legacy');mbpo=load_mbpo(require_data=False,power_model='legacy')
    assert sac.to_dict()==mbpo.sac.to_dict()
    a=sac.algo;m=mbpo.model;h=sac.train_env['periods']-1
    assert (a['epochs'],h,a['start_steps'],a['updates_per_step'],a['batch_size'],a['replay_size'])==(2000,195,8000,2,256,50000)
    assert (a['epochs']*h-a['start_steps'])*a['updates_per_step']==764000
    assert m.model_train_freq==h and m.rollout_length==1
    assert (m.num_models,m.num_elites,m.model_epochs,m.model_batch_size,m.model_max_train_samples)==(5,3,5,512,6000)
    assert (int(a['batch_size']*m.real_ratio),a['batch_size']-int(a['batch_size']*m.real_ratio))==(217,39)
    assert min(m.model_pool_capacity,m.model_retain_epochs*m.rollout_batch_size*m.rollout_length)==1600


def test_real_environment_angular_boundaries_when_data_available():
    pytest.importorskip('gym'); pytest.importorskip('pvlib')
    import json
    from pathlib import Path
    from SAC_workspace.src.config.config import load_config
    from SAC_workspace.src.environment.environment import SACEnvironment
    from SAC_workspace.src.training.experiment_protocol import build_scenario_protocol
    from mbpo.env.pvlib_physics import actuator_movement_cost_kwh
    from mbpo.env.pv_power_backend import compute_pv_power
    root=Path(__file__).resolve().parents[1]
    spec_path=root/'PostProcessing_MC_patch/examples/hypothetical_uncertainty_5pct.json'
    if not spec_path.is_file(): pytest.skip('Real-data fixture requires managed checkout')
    cfg=load_config(require_data=False,weather_uncertainty=json.loads(spec_path.read_text()))
    if not cfg.manifest.is_file(): pytest.skip('NSRDB manifest not installed')
    cfg=load_config(require_data=True,weather_uncertainty=json.loads(spec_path.read_text()))
    split=build_scenario_protocol(cfg.manifest,cfg.protocol)
    sid=split['train_ids'][len(split['train_ids'])//2]
    env=SACEnvironment(cfg.eval_env)
    try:
        s=env.reset(scenario_id=sid,realization_id=0);u=env.unwrapped
        crossed_forward=crossed_backward=upper=lower=False
        for i in range(env.horizon):
            # Start at (30,180); move through 360, reverse through 0, repeat.
            direction=1. if (i//20)%2==0 else -1.
            a=np.array([direction,direction],np.float32)
            old_az=float(u.azimuth)
            tilt,pair,tau,dt,da,done=pose_transition(s[None,:],a[None,:],u.max_delta_tilt,u.max_delta_azimuth,env.horizon)
            n,r,terminal,info=env.step(a)
            np.testing.assert_allclose(n[3:6],np.r_[tilt,pair[0]],atol=4e-5,rtol=0)
            np.testing.assert_allclose(n[TAU],tau[0],atol=1e-7,rtol=0)
            assert bool(terminal)==bool(done[0])==(i==env.horizon-1)
            new_az=float(info['azimuth'])
            crossed_forward |= new_az < old_az-180. and direction>0
            crossed_backward |= new_az > old_az+180. and direction<0
            upper |= float(info['tilt'])==90.
            lower |= float(info['tilt'])==0.
            expected=actuator_movement_cost_kwh(dt[0],da[0],u.actuator_power_w,u.slew_rate_tilt_deg_s,u.slew_rate_azimuth_deg_s,u.movement_penalty)
            assert np.isclose(info['movement_cost'],expected,atol=1e-8,rtol=0)
            assert np.isclose(info['energy_kwh'],info['power_dc_w']*u.interval_hours/1000.,atol=1e-10,rtol=0)
            assert np.isclose(r,info['energy_kwh']-info['movement_cost'],atol=1e-10,rtol=0)
            s=n
        assert crossed_forward and crossed_backward and upper and lower
        kw=dict(area=u.area,efficiency=u.efficiency,power_model=u.power_model,temp_air_c=25.,wind_speed_ms=2.,wind_reference_height_m=u.wind_reference_height_m,gamma_pdc=u.gamma_pdc,sapm_mount=u.sapm_mount,wind_powerlaw_exponent=u.wind_powerlaw_exponent,surface_albedo=.2)
        f=lambda angle: compute_pv_power(45.,angle,60.,70.,600.,400.,100.,**kw).power_w
        assert np.isclose(f(0.),f(360.),atol=1e-10,rtol=0)
        assert abs(f(359.999999)-f(.000001))<1e-4
    finally:
        env.close()
