"""Offline tests of diagnostic contracts. Full PVWatts tests run only on the user's repo."""
import copy
import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

MODULE = Path(__file__).resolve().parents[1] / 'scripts' / 'sac_mbpo_deep_diagnostic.py'
spec=importlib.util.spec_from_file_location('deep_forensic', MODULE)
d=importlib.util.module_from_spec(spec)
spec.loader.exec_module(d)


def make_replay(episodes=3, capacity=None):
    n=episodes*d.H
    capacity=capacity or n+100
    states=np.zeros((n,6), dtype=np.float32)
    nxt=np.zeros((n,6), dtype=np.float32)
    actions=np.zeros((n,2), dtype=np.float32)
    rewards=np.zeros((n,1), dtype=np.float32)
    dones=np.zeros((n,1), dtype=np.float32)
    for ep in range(episodes):
        power=.20+ep*.05
        for j in range(d.H):
            idx=ep*d.H+j
            st=np.array([.1,.05,power,.25,.15,(d.H-j)/d.H],np.float32)
            ns=np.array([.1,.05,power,.25,.15,(d.H-j-1)/d.H],np.float32)
            states[idx], nxt[idx]=st,ns
            rewards[idx,0]=power*2000/12000
            dones[idx,0]=(j==d.H-1)
    return dict(capacity=capacity,count=n,head=n,states=states,actions=actions,
                rewards=rewards,next_states=nxt,dones=dones)


def test_synthetic_replay_reward_and_terminal_identity():
    a=make_replay(3)
    result=d.replay_contract(a,2000,5/60,0.0)
    assert result['terminal_count']==3
    assert result['reward_minus_next_power_kwh']['max_abs'] < 1e-8
    assert result['terminal_mismatch_count']==0
    assert result['tau_step_abs_error']['p95'] < 1e-6


def test_replay_corrupt_reward_fails():
    a=make_replay(3)
    a['rewards'][0,0]=1.0
    with pytest.raises(AssertionError,match='Reward/power'):
        d.replay_contract(a,2000,5/60,0.0)


def test_terminal_mask_inconsistency_fails():
    a=make_replay(3)
    a['dones'][-1,0]=0
    with pytest.raises(AssertionError,match='Terminal'):
        d.replay_contract(a,2000,5/60,0.0)


def test_fifo_full_ring_order():
    a=make_replay(3)
    n=a['count'];a['capacity']=n;a['head']=23
    for key in d.FIELDS:
        a[key]=np.roll(a[key],23,axis=0)
    result=d.real_replay_contiguity(a)
    assert result['n_checked']==n-3
    assert result['max_error']<=1e-6


def test_replay_discontinuity_detection():
    a=make_replay(3)
    a['states'][10,3]=.97
    result=d.real_replay_contiguity(a)
    assert result['fraction_different_gt_1e_3']>0


def test_sampling_formula_matches_simulation():
    small=d.minibatch_sampling_probe(1600,39,seed=10,repeats=300)
    large=d.minibatch_sampling_probe(50000,256,seed=10,repeats=300)
    assert abs(small['empirical_fraction_unique_per_batch']['mean']-
               small['expected_fraction_unique_per_batch'])<.02
    assert abs(large['empirical_fraction_unique_per_batch']['mean']-
               large['expected_fraction_unique_per_batch'])<.02


class FakeActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Linear(6,4)
    def forward(self,states,deterministic=False,with_logprob=True):
        mean, log_std=self.net(states).chunk(2,-1)
        log_std=log_std.clamp(-3,1)
        sig=log_std.exp()
        z=mean if deterministic else mean+sig*torch.randn_like(mean)
        action=z.tanh()
        if not with_logprob:return action,None
        lp=(-.5*((z-mean)/sig).square()-log_std-.5*math.log(2*math.pi)).sum(-1,keepdim=True)
        jac=(2*(math.log(2)-z-torch.nn.functional.softplus(-2*z))).sum(-1,keepdim=True)
        return action,lp-jac

class FakeCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.q1=nn.Linear(8,1)
        self.q2=nn.Linear(8,1)
    def forward(self,s,a):
        z=torch.cat([s,a],-1)
        return self.q1(z),self.q2(z)

class FakeAgent:
    def __init__(self):
        self.actor=FakeActor()
        self.critic=FakeCritic()
        self.critic_target=copy.deepcopy(self.critic)
        self.log_alpha=torch.tensor([-2.0])
        self.min_alpha=.001
        self.gamma=1.
        self.reward_scale=100.
        self.target_entropy=-2.
        self.updates=123
    @property
    def alpha(self):return float(self.log_alpha.detach().exp().clamp_min(self.min_alpha)[0])


def test_actor_critic_probe_computes_gradients_without_parameter_updates():
    a=make_replay(4)
    agent=FakeAgent()
    before={k:v.clone() for k,v in agent.actor.state_dict().items()}
    result,rows=d.sac_actor_critic_probe(agent,a,256,128,seed=16)
    assert result['sample_count']==256
    assert result['critic_gradient_norm_probe']>0
    assert result['actor_gradient_norm_probe']>0
    assert result['terminal_target_scale_error_max'] < 1e-5
    assert len(rows)==200
    for k,p in agent.actor.state_dict().items():
        assert torch.equal(p,before[k]), 'Diagnostic modified actor weights'


def test_branch_sample_is_calendar_balanced():
    dates=[f'2022-{m:02d}-05' for m in range(1,13)]
    selected=d.select_branch_days(dates,4,12)
    assert len(selected)==4
    assert {s[5:7] for s in selected} == {'01','04','07','10'}


def test_branch_summary_detects_q_misranking():
    rows=[dict(method='sac',real_return_chosen_minus_downward_kwh=-.1,
               Q_chosen_minus_downward_reward_scaled_soft=2.),
          dict(method='sac',real_return_chosen_minus_downward_kwh=.2,
               Q_chosen_minus_downward_reward_scaled_soft=1.)]
    res=d.branch_summary(rows)
    assert res['sac']['n_branches']==2
    assert res['sac']['fraction_critic_prefers_chosen_but_real_return_prefers_down']==.5
    assert res['sac']['fraction_critic_and_real_return_sign_agree']==.5


def test_serialization_no_nonstandard_nan():
    a=d.clean({'nan':float('nan'), 'inf':float('inf'),'array':np.array([1,np.nan]),'flag':np.bool_(True)})
    assert a=={'nan':None,'inf':None,'array':[1.0,None],'flag':True}


def test_full_sac_and_mbpo_checkpoint_action_replay():
    agents={'sac':FakeAgent(),'mbpo':FakeAgent()}
    grouped={'2022-07-15':{}}
    for method,agent in agents.items():
        series=[]
        for t in (1,2,3,4):
            series.append({'step':str(t),'dni_wm2':'900','dhi_wm2':'100','power_w':'180',
                           'tilt_deg':'30','azimuth_deg':'180',
                           'action_tilt':'0','action_azimuth':'0'})
        for prev,cur in zip(series[:-1],series[1:]):
            tau=(d.H-int(prev['step']))/d.H
            state=torch.tensor([[.45,.05,.09,1/3.,0.,tau]],dtype=torch.float32)
            with torch.no_grad():
                out,_=agent.actor(state,deterministic=True,with_logprob=False)
            cur['action_tilt']=str(float(out[0,0]))
            cur['action_azimuth']=str(float(out[0,1]))
        grouped['2022-07-15'][method]=series
    result=d.full_evaluation_action_replay(agents,grouped,2000,2000)
    assert result['sac']['checkpoint_actions_match_saved_evaluation']
    assert result['mbpo']['checked_actions']==3
    grouped['2022-07-15']['mbpo'][-1]['action_tilt']='1.0'
    with pytest.raises(AssertionError,match='mbpo: checkpoint'):
        d.full_evaluation_action_replay(agents,grouped,2000,2000)


class SyntheticPVEnv:
    """Tiny deterministic 117-step env for branch control-flow testing."""
    def __init__(self):
        self.unwrapped=self
        self.tilt=30.
        self.azimuth=180.
        self.i=0
    def step(self,a):
        a=np.asarray(a)
        self.tilt=float(np.clip(self.tilt+5.*a[0],0,90))
        self.azimuth=float((self.azimuth+10.*a[1])%360)
        self.i+=1
        # More tilt is less power in this toy summer environment.
        power=300.-self.tilt
        reward=power/12000
        obs=np.array([.5,.05,power/2000,self.tilt/90,self.azimuth/180-1,
                      (d.H-self.i)/d.H],np.float32)
        return obs,reward,(self.i==d.H),{'power_dc_w':power,'tilt':self.tilt}
    def close(self):pass


class ToyCritic(nn.Module):
    def forward(self,s,a):
        q=a[:,0:1]
        return q,q


class ToyAgent:
    def __init__(self):self.critic=ToyCritic()
    def act(self,s,deterministic=True):return np.array([.5,0.],np.float32)


def test_branch_rollout_measures_real_finite_horizon_return(monkeypatch):
    def new_env(cfg,sid,seed):
        env=SyntheticPVEnv()
        obs=np.array([.5,.05,.135,30/90.,0.,1.],np.float32)
        return env,obs
    monkeypatch.setattr(d,'reset_eval_env',new_env)
    agent=ToyAgent()
    # Original recorded actions: upward tilt every step; build real trajectory.
    env,obs=new_env(None,'2022-07-15',42)
    rows=[]
    for j in range(d.H):
        obs,r,done,info=env.step(agent.act(obs))
        rows.append({'step':str(j+1),'action_tilt':'.5','action_azimuth':'0',
                     'power_w':str(info['power_dc_w']),'tilt_deg':str(info['tilt'])})
    chosen=d.roll_branch(None,agent,'2022-07-15',42,rows,20,'chosen')
    downward=d.roll_branch(None,agent,'2022-07-15',42,rows,20,'down')
    assert downward['physical_return_remaining_kwh'] > chosen['physical_return_remaining_kwh']
    assert chosen['Q_estimate_reward_scaled_soft'] > downward['Q_estimate_reward_scaled_soft']


def test_stochastic_soft_branch_is_non_mutating_and_repeatable(monkeypatch):
    def new_env(cfg,sid,seed):
        env=SyntheticPVEnv()
        return env,np.array([.5,.05,.135,30/90.,0.,1.],np.float32)
    monkeypatch.setattr(d,'reset_eval_env',new_env)
    class Policy(FakeAgent):
        def act(self,s,deterministic=True):
            with torch.no_grad():
                a,_=self.actor(torch.as_tensor(s,dtype=torch.float32).reshape(1,6),
                               deterministic=deterministic,with_logprob=False)
            return a.squeeze(0).numpy().astype(np.float32)
    agent=Policy()
    env,obs=new_env(None,'2022-07-15',42)
    rows=[]
    for j in range(d.H):
        a=agent.act(obs)
        obs,r,done,info=env.step(a)
        rows.append({'step':str(j+1),'action_tilt':str(float(a[0])),
                     'action_azimuth':str(float(a[1])),'power_w':str(info['power_dc_w']),
                     'tilt_deg':str(info['tilt'])})
    torch.manual_seed(42)
    original_rng=torch.get_rng_state().clone()
    first=d.soft_branch_return(None,agent,'2022-07-15',42,rows,20,'chosen',1234)
    second=d.soft_branch_return(None,agent,'2022-07-15',42,rows,20,'chosen',1234)
    assert first==second
    assert math.isfinite(first['soft_objective_reward_scaled'])
    assert torch.equal(original_rng,torch.get_rng_state()),'Stochastic diagnostic contaminated global RNG'


def test_shadow_optimizer_step_isolated_from_loaded_checkpoint(monkeypatch):
    class TinySAC(FakeAgent):
        def __init__(self):
            super().__init__()
            self.tau=.005
            self.actor_opt=torch.optim.Adam(self.actor.parameters(),lr=3e-4)
            self.critic_opt=torch.optim.Adam(self.critic.parameters(),lr=3e-4)
            self.alpha_opt=torch.optim.Adam([self.log_alpha],lr=3e-4)
            self.log_alpha.requires_grad_(True)
        def update(self,batch):
            batch={k:torch.as_tensor(v,dtype=torch.float32) for k,v in batch.items()}
            s=batch['states'];a=batch['actions']
            # Toy differentiable loss for independent diagnostic function test.
            q1,q2=self.critic(s,a)
            target=self.reward_scale*batch['rewards']
            loss=((q1-target)**2+(q2-target)**2).mean()
            self.critic_opt.zero_grad();loss.backward();self.critic_opt.step()
            for p in self.critic.parameters():p.requires_grad_(False)
            aa,lp=self.actor(s)
            p1,p2=self.critic(s,aa)
            aloss=(self.log_alpha.detach().exp()*lp-torch.minimum(p1,p2)).mean()
            self.actor_opt.zero_grad();aloss.backward();self.actor_opt.step()
            for p in self.critic.parameters():p.requires_grad_(True)
            tloss=-(self.log_alpha*(lp.detach()+self.target_entropy)).mean()
            self.alpha_opt.zero_grad();tloss.backward();self.alpha_opt.step()
            with torch.no_grad():
                for t,p in zip(self.critic_target.parameters(),self.critic.parameters()):
                    t.lerp_(p,self.tau)
            self.updates+=1
            return {'critic_loss':float(loss.detach()),'actor_loss':float(aloss.detach()),
                    'alpha_loss':float(tloss.detach()),'alpha':self.alpha}
    original=TinySAC()
    saved=[p.detach().clone() for p in original.actor.parameters()]
    monkeypatch.setattr(d,'load_agent',lambda *args:original if False else copy.deepcopy(original))
    class Config:
        algo={'batch_size':128}
    result=d.shadow_optimizer_step('sac',Config(),{},make_replay(3),seed=19)
    assert result['status']=='SHADOW_UPDATE_PASSED'
    assert result['target_polyak_max_abs_error']<1e-6
    for p,before in zip(original.actor.parameters(),saved):
        assert torch.equal(p,before), 'Shadow update mutated source actor'
