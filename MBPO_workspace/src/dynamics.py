"""Probabilistic one-step dynamics ensemble for the *observed* compact PV task.

Only the unpredictable channels [DNI, DHI, PV power] are learned. Known
kinematics, horizon, and net-energy reward are reconstructed exactly using the
existing simulator's contract and pvlib_physics helpers. This is still MBPO:
the SAC critic is trained on predicted model transitions mixed with real ones.
The compact observation is partially observable; this model is NOT guaranteed
Markov-sufficient and its validation uncertainty must be measured.
"""
from __future__ import annotations
import copy
import math
from dataclasses import dataclass
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from SAC_workspace.src.agents.sac_agent import mlp


class GaussianDeltaNet(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.net = mlp(8, 6, (hidden_dim,)*4)

    def forward(self, x):
        mean, log_std = self.net(x).chunk(2, -1)
        log_std = torch.clamp(log_std, -10., 2.)
        return mean, log_std


def gaussian_nll(mean, log_std, target):
    return ((0.5*((mean-target)/torch.exp(log_std)).square()
             + log_std + .5*math.log(2*math.pi)).sum(-1)).mean()


class DynamicsEnsemble:
    def __init__(self, num_models=5, num_elites=3, hidden_dim=64,
                 lr=.001, device='cpu', seed=0):
        if not 2 <= num_elites <= num_models:
            raise ValueError('Invalid ensemble elite count')
        self.device=torch.device(device)
        self.num_models, self.num_elites, self.hidden_dim = num_models,num_elites,hidden_dim
        self.lr=float(lr)
        self.nets=[GaussianDeltaNet(hidden_dim).to(self.device) for _ in range(num_models)]
        self.optimizers=[torch.optim.Adam(m.parameters(),lr=lr) for m in self.nets]
        self.xmean=np.zeros(8,dtype=np.float32);self.xstd=np.ones(8,dtype=np.float32)
        self.ymean=np.zeros(3,dtype=np.float32);self.ystd=np.ones(3,dtype=np.float32)
        self.elites=list(range(num_elites))
        self.rng=np.random.default_rng(seed)
        self.fitted=False
        self.fit_count=0
        self.last_metrics={}

    @staticmethod
    def make_xy(replay, indices):
        s = replay.states[indices]; a = replay.actions[indices]
        nxt = replay.next_states[indices]
        if np.any(s[:,5] <= 0): raise ValueError('Training on terminal start state')
        return np.concatenate([s,a],-1).astype(np.float32), (nxt[:,:3]-s[:,:3]).astype(np.float32)

    def _standardize(self, x):
        return (x-self.xmean)/self.xstd

    def fit(self, replay, *, holdout_fraction=.2, max_samples=6000,
            epochs=12, patience=4, batch_size=256):
        if len(replay) < 2*batch_size: raise ValueError('Not enough real transitions for model fit')
        self.fit_count += 1
        is_first_fit = (self.fit_count == 1)
        train, holdout = replay.scenario_split(holdout_fraction, seed=int(self.rng.integers(0,2**31)))
        if len(train)>max_samples: train=self.rng.choice(train,size=max_samples,replace=False)
        if len(holdout)>max_samples: holdout=self.rng.choice(holdout,size=max_samples,replace=False)
        xtr,ytr=self.make_xy(replay,train)
        xva,yva=self.make_xy(replay,holdout)
        # Preserve optimizer continuity and avoid changing network coordinates
        # every time a new scenario enters the replay. Re-fit scalers only
        # during the first model training event; current features are bounded.
        if is_first_fit:
            self.xmean=xtr.mean(0);self.xstd=xtr.std(0).clip(min=1e-4)
            self.ymean=ytr.mean(0);self.ystd=ytr.std(0).clip(min=1e-4)
        xtr=torch.from_numpy(self._standardize(xtr)).to(self.device)
        ytr=torch.from_numpy((ytr-self.ymean)/self.ystd).to(self.device)
        xva=torch.from_numpy(self._standardize(xva)).to(self.device)
        yva=torch.from_numpy((yva-self.ymean)/self.ystd).to(self.device)
        holdouts=[]
        for k,model in enumerate(self.nets):
            # Persistent network/optimizer: amortize ensemble training across
            # episodes. The normalization coordinate system is frozen above.
            opt=self.optimizers[k]
            best_score=float('inf');best_state=None;best_opt=None;stale=0
            for epoch in range(int(epochs)):
                model.train()
                order=self.rng.permutation(len(xtr))
                for start in range(0,len(order),batch_size):
                    idx=torch.as_tensor(order[start:start+batch_size],device=self.device,dtype=torch.long)
                    mu, ls=model(xtr[idx]);loss=gaussian_nll(mu,ls,ytr[idx])
                    opt.zero_grad(set_to_none=True);loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(),10.);opt.step()
                model.eval()
                with torch.no_grad():
                    mu,ls=model(xva)
                    v=float(gaussian_nll(mu,ls,yva).item())
                if not np.isfinite(v): raise FloatingPointError('Nonfinite dynamics validation NLL')
                if v < best_score - 1e-5:
                    best_score=v;best_state=copy.deepcopy(model.state_dict())
                    best_opt=copy.deepcopy(opt.state_dict());stale=0
                else:
                    stale+=1
                    if stale >= patience: break
            model.load_state_dict(best_state)
            opt.load_state_dict(best_opt)
            model.eval()
            holdouts.append(best_score)
        self.elites=list(np.argsort(holdouts)[:self.num_elites].astype(int))
        self.fitted=True
        with torch.no_grad():
            preds=[]
            for k in self.elites:
                mu,_=self.nets[k](xva)
                preds.append(mu.cpu().numpy()*self.ystd+self.ymean)
        pred=np.stack(preds).mean(0)
        rmse=np.sqrt(np.mean((pred-(yva.cpu().numpy()*self.ystd+self.ymean))**2,axis=0))
        persistence_rmse=np.sqrt(np.mean((yva.cpu().numpy()*self.ystd+self.ymean)**2,axis=0))
        self.last_metrics={'model_persistence_power_rmse_norm':float(persistence_rmse[2]),
                           'model_skill_power_vs_persistence':float(1.-rmse[2]/max(persistence_rmse[2],1e-8)),
                           'model_skill_all_channels_vs_persistence':float(1.-np.linalg.norm(rmse)/max(np.linalg.norm(persistence_rmse),1e-8)),
                           'model_val_nll':float(np.mean([holdouts[k] for k in self.elites])),
                           'model_val_dni_rmse_norm':float(rmse[0]),
                           'model_val_dhi_rmse_norm':float(rmse[1]),
                           'model_val_power_rmse_norm':float(rmse[2]),
                           'model_train_rows':int(len(train)),
                           'model_holdout_rows':int(len(holdout)),
                           'model_elites':list(map(int,self.elites))}
        return dict(self.last_metrics)

    @torch.no_grad()
    def predict_delta(self, states, actions, stochastic=True):
        if not self.fitted: raise RuntimeError('Dynamics ensemble not yet fitted')
        states=np.asarray(states,dtype=np.float32)
        actions=np.asarray(actions,dtype=np.float32)
        if states.ndim!=2 or states.shape[1]!=6 or actions.shape!=(len(states),2):
            raise ValueError('Dynamics expected batch of compact state/action pairs')
        x=np.concatenate([states,actions],-1)
        tx=torch.from_numpy(self._standardize(x)).to(self.device)
        means=[];stds=[]
        for k in self.elites:
            mu,ls=self.nets[k](tx)
            means.append(mu.cpu().numpy()*self.ystd+self.ymean)
            stds.append(np.exp(ls.cpu().numpy())*self.ystd)
        means=np.stack(means);stds=np.stack(stds)
        idx=self.rng.integers(len(self.elites),size=len(states));rows=np.arange(len(states))
        chosen=means[idx,rows]
        if stochastic: chosen=chosen+stds[idx,rows]*self.rng.normal(size=chosen.shape)
        uncertainty=means.std(axis=0).mean(axis=1)
        return chosen.astype(np.float32),uncertainty.astype(np.float32)

    def state_dict(self):
        return {'num_models':self.num_models,'num_elites':self.num_elites,
                'hidden_dim':self.hidden_dim,'lr':self.lr,
                'nets':[m.state_dict() for m in self.nets],
                'optimizers':[o.state_dict() for o in self.optimizers],
                'xmean':self.xmean,'xstd':self.xstd,'ymean':self.ymean,'ystd':self.ystd,
                'elites':self.elites,'rng':self.rng.bit_generator.state,
                'fitted':self.fitted,'fit_count':self.fit_count,'last_metrics':self.last_metrics}

    def load_state_dict(self,data):
        if (data['num_models'],data['num_elites'],data['hidden_dim'],data['lr']) != (self.num_models,self.num_elites,self.hidden_dim,self.lr):
            raise ValueError('Dynamics ensemble architecture mismatch')
        for net,s in zip(self.nets,data['nets']): net.load_state_dict(s)
        for opt,s in zip(self.optimizers,data['optimizers']): opt.load_state_dict(s)
        for key in ('xmean','xstd','ymean','ystd'): setattr(self,key,np.asarray(data[key],dtype=np.float32))
        self.elites=list(data['elites']);self.rng.bit_generator.state=data['rng'];
        self.fitted=bool(data['fitted']);self.fit_count=int(data['fit_count']);
        self.last_metrics=dict(data['last_metrics'])


class StructuredFakeEnv:
    """Kinematics and net energy are NOT learned. Only 3 solar channels are.
    Reward uses the project's own pvlib_physics movement-cost/energy functions.
    """
    def __init__(self, ensemble, reference_env):
        self.ensemble=ensemble
        e=reference_env.unwrapped
        self.max_delta_tilt=float(e.max_delta_tilt)
        self.max_delta_azimuth=float(e.max_delta_azimuth)
        self.movement_penalty=float(e.movement_penalty)
        self.movement_cost_mode=str(e.movement_cost_mode)
        self.actuator_power_w=float(e.actuator_power_w)
        self.slew_rate_tilt_deg_s=float(e.slew_rate_tilt_deg_s)
        self.slew_rate_azimuth_deg_s=float(e.slew_rate_azimuth_deg_s)
        self.interval_hours=float(e.interval_hours)
        self.power_norm=float(__import__('mbpo.env.pv_tracking',fromlist=['POWER_NORM']).POWER_NORM)
        self.horizon=int(reference_env.horizon)
        if self.horizon!=117:raise ValueError('Incorrect simulator horizon')

    def step(self, states, actions, max_std=None):
        from mbpo.env.pvlib_physics import (energy_kwh_from_power_w,actuator_movement_cost_kwh)
        states=np.asarray(states,dtype=np.float32);actions=np.asarray(actions,dtype=np.float32)
        if states.ndim!=2 or states.shape[1]!=6 or actions.shape!=(len(states),2):
            raise ValueError('Fake environment input shape mismatch')
        if not np.isfinite(states).all() or not np.isfinite(actions).all() or np.any(np.abs(actions)>1.00001):
            raise ValueError('Non-finite or out-of-bound imaginary action/state')
        delta,uncertainty=self.ensemble.predict_delta(states,actions)
        nxt=np.empty_like(states)
        nxt[:,:3]=np.clip(states[:,:3]+delta,0.,1.)
        requested_tilt_delta=actions[:,0]*self.max_delta_tilt
        requested_az_delta=actions[:,1]*self.max_delta_azimuth
        old_tilt_deg=states[:,3]*90.
        executed_tilt_deg=np.clip(old_tilt_deg+requested_tilt_delta,0.,90.)
        executed_tilt_delta=executed_tilt_deg-old_tilt_deg
        nxt[:,3]=executed_tilt_deg/90.
        nxt[:,4]=((states[:,4]+1.)*180.+requested_az_delta)%360./180.-1.
        nxt[:,5]=np.maximum(states[:,5]-1./self.horizon,0.)
        terminal=(nxt[:,5]<=1e-7)
        # Project helpers accept scalar values; identical physical units.
        power_w=nxt[:,2]*self.power_norm
        gross=np.array([energy_kwh_from_power_w(float(p),self.interval_hours) for p in power_w],dtype=np.float32)
        if self.movement_penalty<=0:
            movement=np.zeros(len(states),dtype=np.float32)
        elif self.movement_cost_mode=='geometry':
            movement=np.array([actuator_movement_cost_kwh(float(dt),float(da),
                  actuator_power_w=self.actuator_power_w,
                  slew_rate_tilt_deg_s=self.slew_rate_tilt_deg_s,
                  slew_rate_azimuth_deg_s=self.slew_rate_azimuth_deg_s,
                  scale=self.movement_penalty)
                  for dt,da in zip(executed_tilt_delta,requested_az_delta)],dtype=np.float32)
        else:
            movement=self.movement_penalty*(np.abs(requested_tilt_delta)/self.max_delta_tilt
                              + np.abs(requested_az_delta)/self.max_delta_azimuth)
        rewards=(gross-movement).astype(np.float32)
        mask=np.isfinite(nxt).all(-1)&np.isfinite(rewards)
        if max_std is not None: mask &= uncertainty<=max_std
        return nxt,rewards,terminal,uncertainty,mask
