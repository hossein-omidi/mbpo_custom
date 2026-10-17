"""Modern, Ray-free MBPO runner. Reference MDP + SAME PyTorch SAC optimizer.
Real interactions are chronological; model samples are generated separately and
never inserted in the real replay or used as ground truth for ensemble fitting.
"""
from __future__ import annotations
import csv
import json
import os
import random
import time
from pathlib import Path
import numpy as np
import torch
from SAC_workspace.src.config.config import write_snapshot
from SAC_workspace.src.environment.environment import SACEnvironment
from SAC_workspace.src.agents.sac_agent import SACAgent
from SAC_workspace.src.training.replay_buffer import ReplayBuffer
from SAC_workspace.src.training.trainer import seed_all
from MBPO_workspace.src.config import MBPOConfig, config_digest
from MBPO_workspace.src.dynamics import DynamicsEnsemble,StructuredFakeEnv
from MBPO_workspace.src.replay import ScenarioReplayBuffer,mix_batches


class MBPOTrainer:
    def __init__(self, cfg: MBPOConfig, outdir, seed=0, device='cpu', resume=None,
                 env_factory=SACEnvironment):
        seed_all(int(seed))
        self.cfg=cfg;self.seed=int(seed);self.device=device
        self.outdir=Path(outdir).resolve()
        self.outdir.mkdir(parents=True,exist_ok=True)
        self.history_path=self.outdir/'episodes.csv'
        self.cfg_path=self.outdir/'config.json'
        if resume is None:
            if self.history_path.exists() or self.cfg_path.exists():
                raise FileExistsError('A run already exists. Use --resume or a new run directory.')
            self.cfg_path.write_text(json.dumps(cfg.to_dict(),indent=2,sort_keys=True),encoding='utf-8')
        else:
            if not self.cfg_path.is_file(): raise FileNotFoundError('Run config.json is missing')
            if json.loads(self.cfg_path.read_text())!=json.loads(json.dumps(cfg.to_dict())):
                raise ValueError('The requested configuration differs from the original run')
        self.env=env_factory(cfg.sac.train_env)
        self.env.seed(seed)
        self.resolved_path=self.outdir/'resolved_mdp.json'
        if resume is None:
            self.resolved_path.write_text(json.dumps(self.env.resolved_mdp,indent=2,sort_keys=True),encoding='utf-8')
        elif not self.resolved_path.is_file() or json.loads(self.resolved_path.read_text())!=self.env.resolved_mdp:
            raise ValueError('The physical simulator differs from the checkpoint run')
        a=cfg.sac.algo
        self.agent=SACAgent(hidden_sizes=a['hidden_sizes'],lr=a['lr'],gamma=a['gamma'],tau=a['tau'],
            reward_scale=a['reward_scale'],alpha_init=a['alpha_init'],
            target_entropy=a['target_entropy'],min_alpha=a['min_alpha'],device=device)
        m=cfg.model
        self.dynamics=DynamicsEnsemble(num_models=m.num_models,num_elites=m.num_elites,
            hidden_dim=m.hidden_dim,lr=m.model_lr,device=device,seed=seed+100)
        self.fake_env=StructuredFakeEnv(self.dynamics,self.env)
        self.real_pool=ScenarioReplayBuffer(capacity=a['replay_size'],seed=seed+1)
        self.model_pool=ReplayBuffer(capacity=min(m.model_pool_capacity,
              m.model_retain_epochs*m.rollout_batch_size*m.rollout_length),seed=seed+2)
        self.rng=np.random.default_rng(seed+3)
        self.epoch=0;self.env_steps=0
        self.latest_model_metrics={}
        self.last_model_step=0
        if resume is not None:self.load(resume)

    def _fit_and_rollout(self):
        m=self.cfg.model
        unique=len(np.unique(self.real_pool.scenario_ids[:len(self.real_pool)]))
        if unique<m.min_holdout_scenarios:
            self.latest_model_metrics={'model_skipped':f'Only {unique} distinct scenarios'}
            return 0,0,0.
        tic=time.monotonic()
        stats=self.dynamics.fit(self.real_pool,holdout_fraction=m.model_holdout_fraction,
            max_samples=m.model_max_train_samples,epochs=m.model_epochs,
            patience=m.model_patience,batch_size=m.model_batch_size)
        self.latest_model_metrics={**stats,'model_fit_seconds':time.monotonic()-tic}
        if m.require_model_skill and (stats['model_skill_power_vs_persistence'] <= 0 or
                stats['model_skill_all_channels_vs_persistence'] <= 0):
            # A learned one-step model that cannot beat the persistence baseline
            # on an unseen WHOLE DAY is not trustworthy for synthetic Q targets.
            # Also discard stale rollouts from previous ensemble versions.
            self.model_pool=ReplayBuffer(capacity=self.model_pool.capacity,
                seed=self.seed+2+self.dynamics.fit_count)
            self.latest_model_metrics['model_skipped']='Held-out power RMSE worse than persistence'
            self.last_model_step=self.env_steps
            return 0,0,float('nan')
        states=self.real_pool.rollout_states(m.rollout_batch_size)
        if not len(states):return 0,0,0.
        with torch.no_grad():
            inputs=torch.as_tensor(states,dtype=torch.float32,device=self.agent.device)
            actions,_=self.agent.actor(inputs,with_logprob=False)
            actions=actions.cpu().numpy().astype(np.float32)
        ns,rew,dones,uncertainty,mask=self.fake_env.step(states,actions,
            max_std=m.uncertainty_max_std)
        accepted=0
        for j in np.flatnonzero(mask):
            self.model_pool.add(states[j],actions[j],float(rew[j]),ns[j],bool(dones[j]));accepted+=1
        if accepted and (not np.isfinite(uncertainty[mask]).all()):
            raise FloatingPointError('Nonfinite model ensemble uncertainty')
        self.latest_model_metrics.update({'model_generated':int(accepted),
           'model_rejected':int(len(states)-accepted),
           'model_mean_epistemic_std_norm':float(np.mean(uncertainty[mask])) if accepted else float('nan')})
        self.last_model_step=self.env_steps
        return int(accepted),int(len(states)-accepted),float(np.mean(uncertainty[mask])) if accepted else float('nan')

    def train_episode(self):
        state=self.env.reset()
        if not np.isclose(state[5],1.,atol=1e-6):raise RuntimeError('Bad initial tau')
        episode_rewards=[];gross=[];cost=[];updates=[];nr=[];nm=[]
        generated=0;rejected=0;model_time=0.
        scenario=None
        for step in range(self.env.horizon):
            if self.env_steps<self.cfg.sac.algo['start_steps']:
                action=self.real_pool.rng.uniform(-1,1,size=2).astype(np.float32)
            else:
                action=self.agent.act(state)
            nxt,reward,done,info=self.env.step(action)
            sid=info.get('scenario_id')
            if not sid:raise RuntimeError('NSRDB scenario ID was not recorded')
            if scenario is None: scenario=sid
            elif sid!=scenario:raise RuntimeError('Weather scenario changed within an episode')
            self.real_pool.add(state,action,reward,nxt,done,sid)
            self.env_steps+=1
            if (self.env_steps>self.cfg.sac.algo['start_steps']
                    and self.env_steps%self.cfg.model.model_train_freq==0):
                nskip=self._fit_and_rollout()
                generated+=nskip[0];rejected+=nskip[1]
                model_time+=float(self.latest_model_metrics.get('model_fit_seconds',0))
            if (self.env_steps>self.cfg.sac.algo['start_steps']
                    and len(self.real_pool)>=self.cfg.sac.algo['batch_size']):
                for _ in range(self.cfg.sac.algo['updates_per_step']):
                    batch,rn,mn=mix_batches(self.real_pool,self.model_pool,
                        self.cfg.sac.algo['batch_size'],self.cfg.model.real_ratio,self.rng)
                    updates.append(self.agent.update(batch));nr.append(rn);nm.append(mn)
            episode_rewards.append(float(reward))
            gross.append(float(info['energy_kwh']))
            cost.append(float(info['movement_cost']))
            state=nxt
            if done:break
        if len(episode_rewards)!=self.env.horizon or not done:
            raise RuntimeError('Incomplete 117-transition solar episode')
        self.epoch+=1
        row={'epoch':self.epoch,'env_steps':self.env_steps,'updates':self.agent.updates,
             'scenario_id':scenario,'return_kwh':float(sum(episode_rewards)),
             'gross_energy_kwh':float(sum(gross)),'movement_cost_kwh':float(sum(cost)),
             'alpha':self.agent.alpha,'real_replay_size':len(self.real_pool),
             'model_replay_size':len(self.model_pool),'model_fits':self.dynamics.fit_count,
             'synthetic_generated':generated,'synthetic_rejected':rejected,
             'model_train_seconds':model_time,
             'mean_real_batch':float(np.mean(nr)) if nr else float('nan'),
             'mean_model_batch':float(np.mean(nm)) if nm else float('nan')}
        for name in ('critic_loss','actor_loss','alpha_loss','entropy','q_mean','target_mean'):
            row[name]=float(np.mean([m[name] for m in updates])) if updates else float('nan')
        for name in ('model_val_nll','model_val_dni_rmse_norm','model_val_dhi_rmse_norm',
                     'model_val_power_rmse_norm','model_mean_epistemic_std_norm',
                     'model_persistence_power_rmse_norm','model_skill_power_vs_persistence',
                     'model_skill_all_channels_vs_persistence'):
            row[name]=float(self.latest_model_metrics.get(name,float('nan')))
        with self.history_path.open('a',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(row))
            if f.tell()==0:writer.writeheader()
            writer.writerow(row)
        return row

    def save(self,name='latest.pt'):
        rng=self.env.unwrapped.np_random
        ck={'schema':1,'config':self.cfg.to_dict(),'config_hash':config_digest(self.cfg),
            'resolved_mdp':self.env.resolved_mdp,'seed':self.seed,
            'epoch':self.epoch,'env_steps':self.env_steps,
            'agent':self.agent.state_dict(),'dynamics':self.dynamics.state_dict(),
            'real_pool':self.real_pool.state_dict(),'model_pool':self.model_pool.state_dict(),
            'latest_model_metrics':self.latest_model_metrics,'last_model_step':self.last_model_step,
            'torch_rng':torch.get_rng_state(),
            'torch_cuda_rng':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            'numpy_rng':np.random.get_state(),'python_rng':random.getstate(),
            'env_rng':rng.get_state() if hasattr(rng,'get_state') else rng.bit_generator.state,
            'mbpo_rng':self.rng.bit_generator.state}
        path=self.outdir/name;tmp=path.with_suffix('.tmp')
        torch.save(ck,tmp);os.replace(tmp,path)
        return path

    def load(self,path):
        ck=torch.load(path,map_location='cpu',weights_only=False)
        if ck.get('schema')!=1 or ck.get('config_hash')!=config_digest(self.cfg) or ck['seed']!=self.seed:
            raise ValueError('Unrecognized MBPO checkpoint/config/seed')
        if ck['resolved_mdp']!=self.env.resolved_mdp:
            raise ValueError('Physical simulator parameters changed since training')
        self.agent.load_state_dict(ck['agent'])
        self.dynamics.load_state_dict(ck['dynamics'])
        self.real_pool.load_state_dict(ck['real_pool'])
        self.model_pool.load_state_dict(ck['model_pool'])
        self.epoch=int(ck['epoch']);self.env_steps=int(ck['env_steps'])
        self.latest_model_metrics=dict(ck['latest_model_metrics'])
        self.last_model_step=int(ck['last_model_step'])
        self.rng.bit_generator.state=ck['mbpo_rng']
        torch.set_rng_state(ck['torch_rng'])
        if torch.cuda.is_available() and ck['torch_cuda_rng'] is not None:
            torch.cuda.set_rng_state_all(ck['torch_cuda_rng'])
        np.random.set_state(ck['numpy_rng']);random.setstate(ck['python_rng'])
        rg=self.env.unwrapped.np_random
        if hasattr(rg,'set_state'): rg.set_state(ck['env_rng'])
        else:rg.bit_generator.state=ck['env_rng']
        if not self.history_path.is_file():raise FileNotFoundError('Cannot resume without episodes.csv')
        with self.history_path.open(newline='',encoding='utf-8') as f:rows=list(csv.DictReader(f))
        if len(rows)!=self.epoch or (rows and int(rows[-1]['env_steps'])!=self.env_steps):
            raise ValueError('Checkpoint does not match the episode log')

    def train(self,epochs=None):
        target=self.cfg.sac.algo['epochs'] if epochs is None else int(epochs)
        if target<self.epoch:raise ValueError('Cannot train backwards')
        try:
            while self.epoch<target:
                r=self.train_episode()
                print('epoch={:5d} real_steps={:7d} net={:.5f}kWh alpha={:.4g} model={} synthetic={} ratio={:.3g}'.format(
                    r['epoch'],r['env_steps'],r['return_kwh'],r['alpha'],r['model_fits'],
                    r['synthetic_generated'],r['mean_model_batch']/self.cfg.sac.algo['batch_size']
                    if np.isfinite(r['mean_model_batch']) else 0),flush=True)
                if self.epoch%self.cfg.sac.algo['checkpoint_interval']==0:self.save()
            self.save()
        finally:self.env.close()
