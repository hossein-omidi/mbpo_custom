import copy, math
import numpy as np
import torch
from mbpo.geometry10 import build, check_batch, pose_transition, solar_features, TAU, SOLAR
from SAC_workspace.src.agents.sac_agent import SACAgent
from SAC_workspace.src.training.replay_buffer import ReplayBuffer
from MBPO_workspace.src.replay import ScenarioReplayBuffer,mix_batches
from MBPO_workspace.src.dynamics import DynamicsEnsemble,gaussian_nll

torch.set_num_threads(1);torch.manual_seed(71)
rng=np.random.default_rng(71)
# Generated transitions are controlled numerical fixtures, not NSRDB observations.
rb=ScenarioReplayBuffer(4096,seed=17)
for day in range(30):
    tilt=30.; az=359.; w=100.+day
    for t in range(195):
        tau=(195-t)/195; a=rng.uniform(-1,1,2).astype(np.float32)
        z=80.-t/4; sun=30.+t
        s=build(w,40+w*.1,20+w*.15,tilt,az,z,sun,tau)
        tilt=float(np.clip(tilt+float(a[0])*5,0,90));az=(az+float(a[1])*10)%360
        w+=1.5+.2*np.sin(t/20)
        nxt=build(w,40+w*.1,20+w*.15,tilt,az,z-.25,sun+1,(194-t)/195)
        rb.add(s,a,float(nxt[2])/6-0.00001,nxt,t==194,f'day{day}')
assert len(rb)==4096 and rb.head==5850%4096
tr,ho=rb.scenario_split(.2,0); assert not set(rb.scenario_ids[tr])&set(rb.scenario_ids[ho]); assert len(tr)+len(ho)==len(rb)
assert np.array_equal(rb.scenario_split(.2,99)[0],tr)
rs,sids=rb.rollout_states(400,with_scenarios=True);assert all(rb._validation_roles[s]=='train' for s in sids)
assert (rs[:,TAU]>0).all();print('PASS FIFO overwrite, day holdout, rollout origins')
rb2=ScenarioReplayBuffer(4096,seed=17);rb2.load_state_dict(copy.deepcopy(rb.state_dict()));
for k,v in rb.sample(256).items(): pass
# Reset to the same saved RNG, verify exact replay stream restoration.
rb2.load_state_dict(copy.deepcopy(rb.state_dict()));a=rb.sample(256);b=rb2.sample(256);assert all(np.array_equal(a[k],b[k]) for k in a)
print('PASS replay checkpoint and RNG continuation')
model=DynamicsEnsemble(num_models=5,num_elites=3,hidden_dim=64,seed=3)
initial=[[p.detach().clone() for p in n.parameters()] for n in model.weather_nets+model.power_nets]
# Same optimizer, architecture, minibatch and epoch count as configured; reduced fixture data.
st=model.fit(rb,max_samples=6000,epochs=5,patience=5,batch_size=512)
assert model.fit_count==1 and model.fitted and len(set(model.weather_elites))==3 and len(set(model.power_elites))==3
for before,net,opt in zip(initial,model.weather_nets+model.power_nets,model.weather_optimizers+model.power_optimizers):
    assert any(not torch.equal(p,q) for p,q in zip(before,net.parameters())) and len(opt.state)>0
    assert all(torch.isfinite(p).all() for p in net.parameters())
print('PASS all 10 dynamics networks updated; elite selection and finite parameters')
print('fixture skills',st['model_skill_power_vs_persistence'],st['model_skill_all_channels_vs_persistence'])
# Verify NLL against PyTorch distribution and gradient.
m=torch.tensor([[.4,-.2]],requires_grad=True);ls=torch.tensor([[.1,-.5]],requires_grad=True);y=torch.tensor([[.2,.3]])
loss=gaussian_nll(m,ls,y);torch.testing.assert_close(loss,-torch.distributions.Normal(m,ls.exp()).log_prob(y).sum(-1).mean());loss.backward();assert m.grad.isfinite().all() and ls.grad.isfinite().all()
print('PASS Gaussian NLL formula and gradient')
agent=SACAgent();acts=np.stack([agent.act(s) for s in rs]);delta,score=model.predict_delta(rs,acts,rng=np.random.default_rng(5));alt,_=model.predict_delta(rs,-acts,rng=np.random.default_rng(5));np.testing.assert_array_equal(delta[:,:2],alt[:,:2]);assert np.isfinite(delta).all() and np.isfinite(score).all()
print('PASS actual probabilistic rollout and common-noise action-independent weather')
# Model checkpoint including all optimizers, scalers and generator state.
model2=DynamicsEnsemble(num_models=5,num_elites=3,hidden_dim=64,seed=91);model2.load_state_dict(copy.deepcopy(model.state_dict()));
a,b=model.predict_delta(rs,acts);c,d=model2.predict_delta(rs,acts);np.testing.assert_array_equal(a,c);np.testing.assert_array_equal(b,d)
print('PASS ensemble checkpoint stochastic continuation')
# Use actual predicted deltas, exact pose and clock; use fixture solar geometry, not PVLib.
mp=ReplayBuffer(1600,seed=8)
new_tilt,new_az,tau,dt,da,done=pose_transition(rs,acts,5,10,195)
nxt=rs.copy();nxt[:,:3]=np.clip(rs[:,:3]+delta,0,1);nxt[:,3]=new_tilt;nxt[:,4:6]=new_az;nxt[:,TAU]=tau
check_batch(nxt);reward=nxt[:,2]/6-30*(abs(dt)/1.5+abs(da)/2)/3.6e6
for i in range(len(rs)):mp.add(rs[i],acts[i],reward[i],nxt[i],done[i])
for poolsize in (0,7,400):
    p=ReplayBuffer(1600,seed=10)
    for i in range(poolsize):p.add(rs[i],acts[i],reward[i],nxt[i],done[i])
    batch,rn,mn=mix_batches(rb,p,256,.85,rng); assert mn==min(poolsize,39) and rn+mn==256
    assert all(len(v)==256 for v in batch.values())
print('PASS actual batch mixer 256/0, 249/7, 217/39')
# Independently recompute target and verify parameter/optimizer updates.
for mode in ['sac','mbpo']:
    batch=rb.sample(256) if mode=='sac' else mix_batches(rb,mp,256,.85,rng)[0]
    ac=[p.detach().clone() for p in agent.actor.parameters()];qc=[p.detach().clone() for p in agent.critic.parameters()];tc=[p.detach().clone() for p in agent.critic_target.parameters()];alpha=agent.log_alpha.detach().clone()
    metrics=agent.update(batch)
    assert all(np.isfinite(v) for v in metrics.values())
    assert any(not torch.equal(p,q) for p,q in zip(ac,agent.actor.parameters()))
    assert all(not torch.equal(p,q) for p,q in zip(qc,agent.critic.parameters()))
    assert not torch.equal(alpha,agent.log_alpha)
    for p,old,q in zip(agent.critic_target.parameters(),tc,agent.critic.parameters()):torch.testing.assert_close(p,old+agent.tau*(q.detach()-old),rtol=1e-6,atol=1e-7);assert not p.requires_grad and p.grad is None
    print('PASS',mode,'actual actor/twin critics/alpha updates and exact target Polyak interpolation')
print('ALL COMPONENT CHAIN CHECKS PASSED')
