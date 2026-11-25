# PVWATT_DEV_REPAIR_V3
"""Geometry10 one-step MBPO: learned DNI/DHI/power deltas, exact pose,
clock, solar ephemeris and actuator-energy calculation. The exogenous model
uses current solar geometry and tau, never actions or future weather."""
from __future__ import annotations
from mbpo.geometry10 import (STATE_DIM, WEATHER_DIM, POWER_DIM, TAU, PANEL_AZ, SOLAR,
    weather_features, check_batch, pose_transition, SolarContext)

import copy
import math
import numpy as np
import torch
from torch import nn
from SAC_workspace.src.agents.sac_agent import mlp

DYNAMICS_SCHEMA = 4


class GaussianDeltaNet(nn.Module):
    """Gaussian delta network retaining the prior four-layer MLP structure."""
    def __init__(self, hidden_dim=64, input_dim=8, output_dim=3):
        super().__init__()
        self.net = mlp(int(input_dim), int(2 * output_dim), (int(hidden_dim),) * 4)

    def forward(self, x):
        mean, log_std = self.net(x).chunk(2, dim=-1)
        return mean, torch.clamp(log_std, -10.0, 2.0)


def gaussian_nll(mean, log_std, target):
    return (0.5 * ((mean - target) / torch.exp(log_std)).square()
            + log_std + 0.5 * math.log(2.0 * math.pi)).sum(-1).mean()


class DynamicsEnsemble:
    """Five-member (configurable) weather + power probabilistic ensemble."""

    def __init__(self, num_models=5, num_elites=3, hidden_dim=64,
                 lr=0.001, device='cpu', seed=0):
        if not 2 <= int(num_elites) <= int(num_models):
            raise ValueError('Invalid ensemble elite count')
        if int(hidden_dim) <= 0 or not math.isfinite(float(lr)) or lr <= 0:
            raise ValueError('Invalid network hyperparameters')
        self.device = torch.device(device)
        self.num_models = int(num_models)
        self.num_elites = int(num_elites)
        self.hidden_dim = int(hidden_dim)
        self.lr = float(lr)
        self.weather_nets = [GaussianDeltaNet(hidden_dim, WEATHER_DIM, 2).to(self.device)
                             for _ in range(self.num_models)]
        self.power_nets = [GaussianDeltaNet(hidden_dim, POWER_DIM, 1).to(self.device)
                           for _ in range(self.num_models)]
        self.weather_optimizers = [torch.optim.Adam(m.parameters(), lr=lr)
                                   for m in self.weather_nets]
        self.power_optimizers = [torch.optim.Adam(m.parameters(), lr=lr)
                                 for m in self.power_nets]
        self.weather_xmean = np.zeros(WEATHER_DIM, np.float32)
        self.weather_xstd = np.ones(WEATHER_DIM, np.float32)
        self.weather_ymean = np.zeros(2, np.float32)
        self.weather_ystd = np.ones(2, np.float32)
        self.power_xmean = np.zeros(POWER_DIM, np.float32)
        self.power_xstd = np.ones(POWER_DIM, np.float32)
        self.power_ymean = np.zeros(1, np.float32)
        self.power_ystd = np.ones(1, np.float32)
        self.weather_elites = list(range(self.num_elites))
        self.power_elites = list(range(self.num_elites))
        self.elites = list(self.weather_elites)  # diagnostic compatibility
        self.rng = np.random.default_rng(seed)
        self.fitted = False
        self.fit_count = 0
        self.last_metrics = {}

    @staticmethod
    def make_xy(replay, indices):
        states = np.asarray(replay.states[indices], np.float32)
        actions = np.asarray(replay.actions[indices], np.float32)
        nxt = np.asarray(replay.next_states[indices], np.float32)
        if (states.ndim != 2 or states.shape[1] != STATE_DIM
                or actions.shape != (len(states), 2) or nxt.shape != states.shape):
            raise ValueError('Expected geometry10 observation and 2D action transitions')
        if np.any(states[:, TAU] <= 0) or not all(
                np.isfinite(arr).all() for arr in (states, actions, nxt)):
            raise ValueError('Non-finite or terminal-start training data')
        xw = weather_features(states)
        check_batch(states); check_batch(nxt)
        if np.any(np.abs(actions)>1.00001):
            raise ValueError('Invalid model-fit actions')
        yw = (nxt[:, :2] - states[:, :2]).astype(np.float32)
        # Teacher-forced next weather is used only to fit the conditional power model.
        xp = np.concatenate((states, actions, nxt[:, :2]), axis=1).astype(np.float32)
        yp = (nxt[:, 2:3] - states[:, 2:3]).astype(np.float32)
        return xw, yw, xp, yp

    @staticmethod
    def _scalers(x, y):
        return (x.mean(0).astype(np.float32), x.std(0).clip(min=1e-4).astype(np.float32),
                y.mean(0).astype(np.float32), y.std(0).clip(min=1e-4).astype(np.float32))

    def _fit_group(self, nets, optimizers, xtrain, ytrain, xvalid, yvalid,
                   xmean, xstd, ymean, ystd, epochs, patience, batch_size):
        xtr = torch.as_tensor((xtrain-xmean)/xstd, dtype=torch.float32, device=self.device)
        ytr = torch.as_tensor((ytrain-ymean)/ystd, dtype=torch.float32, device=self.device)
        xva = torch.as_tensor((xvalid-xmean)/xstd, dtype=torch.float32, device=self.device)
        yva = torch.as_tensor((yvalid-ymean)/ystd, dtype=torch.float32, device=self.device)
        if len(xtr) < 16 or len(xva) < 16 or epochs < 1 or patience < 1 or batch_size < 1:
            raise ValueError('Insufficient model fitting rows or epochs')
        scores = []
        for net, opt in zip(nets, optimizers):
            # Independent bootstrap gives members different empirical data distributions.
            boot = self.rng.integers(len(xtr), size=len(xtr))
            best_score, best_state, best_opt, stale = float('inf'), None, None, 0
            for _epoch in range(int(epochs)):
                net.train()
                order = boot[self.rng.permutation(len(boot))]
                for begin in range(0, len(order), int(batch_size)):
                    ix = torch.as_tensor(order[begin:begin+int(batch_size)],
                                         dtype=torch.long, device=self.device)
                    mu, ls = net(xtr[ix])
                    loss = gaussian_nll(mu, ls, ytr[ix])
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(net.parameters(), 10.0)
                    opt.step()
                net.eval()
                with torch.no_grad():
                    mu, ls = net(xva)
                    score = float(gaussian_nll(mu, ls, yva).item())
                if not math.isfinite(score):
                    raise FloatingPointError('Non-finite dynamics selection NLL')
                if score < best_score - 1e-5:
                    best_score, stale = score, 0
                    best_state = copy.deepcopy(net.state_dict())
                    best_opt = copy.deepcopy(opt.state_dict())
                else:
                    stale += 1
                    if stale >= int(patience):
                        break
            if best_state is None:
                raise RuntimeError('Model fit did not produce a valid epoch')
            net.load_state_dict(best_state)
            opt.load_state_dict(best_opt)
            net.eval()
            scores.append(best_score)
        return list(np.argsort(scores)[:self.num_elites].astype(int)), scores

    @torch.no_grad()
    def _members(self, nets, elite, inputs, xmean, xstd, ymean, ystd):
        inputs = np.asarray(inputs, dtype=np.float32)
        tx = torch.as_tensor((inputs-xmean)/xstd, dtype=torch.float32, device=self.device)
        mus, sigmas = [], []
        for k in elite:
            m, log_s = nets[k](tx)
            mus.append(m.cpu().numpy() * ystd + ymean)
            sigmas.append(np.exp(log_s.cpu().numpy()) * ystd)
        mus, sigmas = np.stack(mus).astype(np.float32), np.stack(sigmas).astype(np.float32)
        if not np.isfinite(mus).all() or not np.isfinite(sigmas).all() or np.any(sigmas < 0):
            raise FloatingPointError('Invalid ensemble distribution parameters')
        return mus, sigmas

    def _weather_features(self, states):
        return weather_features(states)

    def _power_features(self, states, actions, next_weather):
        return np.concatenate((states, actions, next_weather), axis=1).astype(np.float32)

    @staticmethod
    def _check(states, actions):
        states = np.asarray(states, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.float32)
        if (states.ndim != 2 or states.shape[1] != STATE_DIM or actions.shape != (len(states), 2)
                or not np.isfinite(states).all() or not np.isfinite(actions).all()
                or np.any(np.abs(actions) > 1.00001)):
            raise ValueError('Dynamics expects finite geometry10 states and bounded (N,2) actions')
        return states, actions

    def fit(self, replay, *, holdout_fraction=0.2, max_samples=6000,
            epochs=12, patience=4, batch_size=256):
        if len(replay) < 2*int(batch_size):
            raise ValueError('Insufficient real transitions')
        train, holdout = replay.scenario_split(holdout_fraction,
                                              seed=int(self.rng.integers(0, 2**31)))
        if len(train) < 16 or len(holdout) < 16:
            raise ValueError('Insufficient held-out model rows')
        if len(train) > max_samples:
            train = self.rng.choice(train, int(max_samples), replace=False)
        if len(holdout) > max_samples:
            holdout = self.rng.choice(holdout, int(max_samples), replace=False)
        xwt, ywt, xpt, ypt = self.make_xy(replay, train)
        xwv, ywv, xpv, ypv = self.make_xy(replay, holdout)
        if self.fit_count == 0:
            (self.weather_xmean, self.weather_xstd,
             self.weather_ymean, self.weather_ystd) = self._scalers(xwt, ywt)
            (self.power_xmean, self.power_xstd,
             self.power_ymean, self.power_ystd) = self._scalers(xpt, ypt)
        we, ws = self._fit_group(
            self.weather_nets, self.weather_optimizers, xwt, ywt, xwv, ywv,
            self.weather_xmean, self.weather_xstd, self.weather_ymean, self.weather_ystd,
            epochs, patience, batch_size)
        pe, ps = self._fit_group(
            self.power_nets, self.power_optimizers, xpt, ypt, xpv, ypv,
            self.power_xmean, self.power_xstd, self.power_ymean, self.power_ystd,
            epochs, patience, batch_size)
        self.weather_elites, self.power_elites = we, pe
        self.elites = list(we)
        # The power assessment uses WEATHER PREDICTED BY THE MODEL, not true next weather.
        states = np.asarray(replay.states[holdout], np.float32)
        actions = np.asarray(replay.actions[holdout], np.float32)
        next_states = np.asarray(replay.next_states[holdout], np.float32)
        mw, _sw = self._members(self.weather_nets, we, xwv,
                                self.weather_xmean, self.weather_xstd,
                                self.weather_ymean, self.weather_ystd)
        weather = np.clip(states[:, :2] + mw.mean(0), 0., 1.)
        mp, _sp = self._members(self.power_nets, pe,
                                self._power_features(states, actions, weather),
                                self.power_xmean, self.power_xstd,
                                self.power_ymean, self.power_ystd)
        predicted = np.column_stack((weather, np.clip(states[:, 2] + mp.mean(0)[:, 0], 0., 1.)))
        truth = next_states[:, :3]
        persist = states[:, :3]
        rmse = np.sqrt(np.mean(np.square(predicted-truth), axis=0))
        persist_rmse = np.sqrt(np.mean(np.square(persist-truth), axis=0))
        skill_power = 1.0-rmse[2]/max(float(persist_rmse[2]),1e-8)
        skill_all = 1.0-np.linalg.norm(rmse)/max(float(np.linalg.norm(persist_rmse)),1e-8)
        self.last_metrics = {
            'model_val_nll': float((np.mean([ws[i] for i in we]) / 2.0
                                    + np.mean([ps[i] for i in pe])) / 2.0),
            'model_val_dni_rmse_norm': float(rmse[0]),
            'model_val_dhi_rmse_norm': float(rmse[1]),
            'model_val_power_rmse_norm': float(rmse[2]),
            'model_persistence_power_rmse_norm': float(persist_rmse[2]),
            'model_skill_power_vs_persistence': float(skill_power),
            'model_skill_all_channels_vs_persistence': float(skill_all),
            'model_train_rows': int(len(train)), 'model_holdout_rows': int(len(holdout)),
            'model_train_scenarios': int(len(np.unique(replay.scenario_ids[train]))),
            'model_holdout_scenarios': int(len(np.unique(replay.scenario_ids[holdout]))),
            'model_elites_weather': list(map(int, we)),
            'model_elites_power': list(map(int, pe)),
            # Selection days are reused for early stopping: do NOT treat these
            # numbers as an independent final model generalization estimate.
            'model_metrics_role': 'model_selection_only',
        }
        self.fit_count += 1  # Do not advance on unsuccessful split or model fitting.
        self.fitted = True
        return dict(self.last_metrics)

    @torch.no_grad()
    def predict_distribution(self, states, actions):
        """Deterministic ensemble-mixture moments (delta channels).\n
        Power variance approximates aleatoric uncertainty conditional on mean
        next weather; external coverage tests must assess its calibration.
        """
        if not self.fitted:
            raise RuntimeError('Dynamics ensemble not fitted')
        states, actions = self._check(states, actions)
        wmu, wsd = self._members(self.weather_nets, self.weather_elites,
                                 self._weather_features(states),
                                 self.weather_xmean, self.weather_xstd,
                                 self.weather_ymean, self.weather_ystd)
        weather = np.clip(states[:, :2] + wmu.mean(0), 0., 1.)
        pmu, psd = self._members(self.power_nets, self.power_elites,
                                 self._power_features(states, actions, weather),
                                 self.power_xmean, self.power_xstd,
                                 self.power_ymean, self.power_ystd)
        # The deterministic predictive mean is clipped by the same physical
        # observation bounds as a model-generated transition.
        mean = np.concatenate((weather-states[:, :2],
             np.clip(states[:, 2:3]+pmu.mean(0), 0., 1.)-states[:, 2:3]), axis=1)
        ale = np.concatenate((np.sqrt(np.mean(wsd**2, axis=0)),
                              np.sqrt(np.mean(psd**2, axis=0))), axis=1)
        epi = np.concatenate((wmu.std(0), pmu.std(0)), axis=1)
        total = np.sqrt(ale**2 + epi**2)
        norm_scale = np.concatenate((self.weather_ystd, self.power_ystd))
        score = np.sqrt(np.mean((epi/np.maximum(norm_scale[None, :], 1e-8))**2, axis=1))
        if not all(np.isfinite(v).all() for v in (mean, ale, epi, total, score)):
            raise FloatingPointError('Nonfinite model distribution')
        return (mean.astype(np.float32), ale.astype(np.float32), epi.astype(np.float32),
                total.astype(np.float32), score.astype(np.float32))

    @torch.no_grad()
    def predict_delta(self, states, actions, stochastic=True):
        """One-step weather first, then response; action never controls weather."""
        if not self.fitted:
            raise RuntimeError('Dynamics ensemble not fitted')
        states, actions = self._check(states, actions)
        wmu, wsd = self._members(self.weather_nets, self.weather_elites,
                                 self._weather_features(states),
                                 self.weather_xmean, self.weather_xstd,
                                 self.weather_ymean, self.weather_ystd)
        if stochastic:
            ix = self.rng.integers(len(self.weather_elites), size=len(states))
            ii = np.arange(len(states))
            dw = wmu[ix, ii] + wsd[ix, ii]*self.rng.normal(size=(len(states), 2))
        else:
            dw = wmu.mean(0)
        next_weather = np.clip(states[:, :2]+dw, 0., 1.)
        dw = next_weather - states[:, :2]
        pmu, psd = self._members(self.power_nets, self.power_elites,
                                 self._power_features(states, actions, next_weather),
                                 self.power_xmean, self.power_xstd,
                                 self.power_ymean, self.power_ystd)
        if stochastic:
            ix = self.rng.integers(len(self.power_elites), size=len(states))
            dp = pmu[ix, np.arange(len(states))] + psd[ix, np.arange(len(states))]*self.rng.normal(size=(len(states), 1))
        else:
            dp = pmu.mean(0)
        dp = np.clip(states[:, 2:3]+dp, 0., 1.)-states[:, 2:3]
        score = np.sqrt((2.0*np.mean((wmu.std(0)/self.weather_ystd[None, :])**2,axis=1)
                          + (pmu.std(0)[:,0]/self.power_ystd[0])**2)/3.0)
        return np.concatenate((dw, dp), axis=1).astype(np.float32), score.astype(np.float32)

    def state_dict(self):
        return {
            'schema': DYNAMICS_SCHEMA, 'state_dim': STATE_DIM, 'num_models': self.num_models,
            'num_elites': self.num_elites, 'hidden_dim': self.hidden_dim, 'lr': self.lr,
            'weather_nets': [m.state_dict() for m in self.weather_nets],
            'power_nets': [m.state_dict() for m in self.power_nets],
            'weather_optimizers': [o.state_dict() for o in self.weather_optimizers],
            'power_optimizers': [o.state_dict() for o in self.power_optimizers],
            'weather_xmean': self.weather_xmean, 'weather_xstd': self.weather_xstd,
            'weather_ymean': self.weather_ymean, 'weather_ystd': self.weather_ystd,
            'power_xmean': self.power_xmean, 'power_xstd': self.power_xstd,
            'power_ymean': self.power_ymean, 'power_ystd': self.power_ystd,
            'weather_elites': self.weather_elites, 'power_elites': self.power_elites,
            'rng': self.rng.bit_generator.state, 'fitted': self.fitted,
            'fit_count': self.fit_count, 'last_metrics': self.last_metrics,
        }

    def load_state_dict(self, data):
        if data.get('schema') != DYNAMICS_SCHEMA or data.get('state_dim') != STATE_DIM:
            raise ValueError('Dynamics architecture changed: old MBPO model checkpoints cannot be resumed')
        if (data['num_models'],data['num_elites'],data['hidden_dim'],data['lr']) != (
                self.num_models,self.num_elites,self.hidden_dim,self.lr):
            raise ValueError('Dynamics ensemble hyperparameters mismatch')
        for field, nets in (('weather_nets',self.weather_nets),('power_nets',self.power_nets)):
            if len(data[field]) != len(nets):
                raise ValueError('Incorrect saved dynamics network count')
            for net, saved in zip(nets,data[field]):
                net.load_state_dict(saved)
        for field, opts in (('weather_optimizers',self.weather_optimizers),('power_optimizers',self.power_optimizers)):
            if len(data[field]) != len(opts):
                raise ValueError('Incorrect saved dynamics optimizer count')
            for opt, saved in zip(opts,data[field]):
                opt.load_state_dict(saved)
        for field in ('weather_xmean','weather_xstd','weather_ymean','weather_ystd',
                      'power_xmean','power_xstd','power_ymean','power_ystd'):
            saved = np.asarray(data[field],dtype=np.float32)
            if saved.shape != getattr(self,field).shape or not np.isfinite(saved).all() or (field.endswith('std') and np.any(saved <= 0)):
                raise ValueError(f'Invalid saved dynamics scaler: {field}')
            setattr(self, field, saved)
        self.weather_elites = list(map(int,data['weather_elites']))
        self.power_elites = list(map(int,data['power_elites']))
        if (not self.weather_elites or not self.power_elites
                or any(k<0 or k>=self.num_models for k in self.weather_elites+self.power_elites)):
            raise ValueError('Invalid saved elite network indices')
        self.elites = list(self.weather_elites)
        self.rng.bit_generator.state = data['rng']
        self.fitted, self.fit_count = bool(data['fitted']), int(data['fit_count'])
        self.last_metrics = dict(data['last_metrics'])


class StructuredFakeEnv:
    """Reconstruct exact finite-horizon pose transitions and net-energy reward."""
    def __init__(self, ensemble, reference_env):
        self.ensemble = ensemble
        self.solar_context = SolarContext(reference_env)
        e = reference_env.unwrapped
        self.max_delta_tilt = float(e.max_delta_tilt)
        self.max_delta_azimuth = float(e.max_delta_azimuth)
        self.movement_penalty = float(e.movement_penalty)
        self.movement_cost_mode = str(e.movement_cost_mode)
        self.actuator_power_w = float(e.actuator_power_w)
        self.slew_rate_tilt_deg_s = float(e.slew_rate_tilt_deg_s)
        self.slew_rate_azimuth_deg_s = float(e.slew_rate_azimuth_deg_s)
        self.interval_hours = float(e.interval_hours)
        self.power_norm = float(__import__('mbpo.env.pv_tracking',fromlist=['POWER_NORM']).POWER_NORM)
        self.horizon = int(reference_env.horizon)
        if (self.horizon <= 0 or self.horizon != int(e.num_action_steps)
                or self.horizon != int(e.periods)-1):
            raise ValueError('Fake and real environment finite horizons differ')
        if self.movement_cost_mode not in ('legacy','geometry'):
            raise ValueError('Unsupported movement-cost mode')
        if (not math.isfinite(self.movement_penalty) or self.movement_penalty < 0
                or not math.isfinite(self.interval_hours) or self.interval_hours <= 0):
            raise ValueError('Invalid fake environment parameters')

    def step(self, states, actions, max_std=None, stochastic=True, *, scenario_ids=None):
        from mbpo.env.pvlib_physics import actuator_movement_cost_kwh
        states = np.asarray(states,dtype=np.float32)
        actions = np.asarray(actions,dtype=np.float32)
        states = check_batch(states)
        if states.ndim != 2 or not len(states):
            raise ValueError('Expected nonempty geometry10 rollout batch')
        if actions.shape != (len(states),2) or not np.isfinite(actions).all() or np.any(np.abs(actions)>1.00001):
            raise ValueError('Invalid synthetic action batch')
        actions = np.clip(actions, -1., 1.)
        # Exact ephemeris from scenario + remaining time; no weather leakage.
        next_solar = self.solar_context.next(states, scenario_ids)
        new_tilt, new_az, tau, executed_tilt, executed_az, done = pose_transition(
            states, actions, self.max_delta_tilt, self.max_delta_azimuth, self.horizon)
        delta,uncertainty = self.ensemble.predict_delta(states,actions,stochastic=stochastic)
        delta = np.asarray(delta,dtype=np.float32)
        uncertainty = np.asarray(uncertainty,dtype=np.float32)
        if delta.shape != (len(states),3) or uncertainty.shape != (len(states),):
            raise ValueError('Invalid ensemble prediction dimensions')
        # Reject raw NaN/Inf before clipping can hide them.
        finite_prediction = np.isfinite(delta).all(axis=1) & np.isfinite(uncertainty) & (uncertainty >= 0)
        nxt=np.empty_like(states)
        nxt[:,:3] = np.clip(states[:,:3]+delta,0.,1.)
        nxt[:,3] = new_tilt
        nxt[:,PANEL_AZ] = new_az
        nxt[:,SOLAR] = next_solar
        nxt[:,TAU] = tau
        gross = nxt[:,2]*self.power_norm*self.interval_hours/1000.
        if self.movement_penalty<=0:
            movement=np.zeros(len(states),dtype=np.float32)
        elif self.movement_cost_mode=='geometry':
            movement=np.asarray([
                actuator_movement_cost_kwh(
                    float(dt),float(da),actuator_power_w=self.actuator_power_w,
                    slew_rate_tilt_deg_s=self.slew_rate_tilt_deg_s,
                    slew_rate_azimuth_deg_s=self.slew_rate_azimuth_deg_s,
                    scale=self.movement_penalty)
                for dt,da in zip(executed_tilt,executed_az)],dtype=np.float32)
        else:
            movement=self.movement_penalty*(np.abs(executed_tilt)/self.max_delta_tilt
                         +np.abs(executed_az)/self.max_delta_azimuth)
        reward=(gross-movement).astype(np.float32)
        mask=finite_prediction & np.isfinite(nxt).all(axis=1)&np.isfinite(reward)
        if np.any(mask): check_batch(nxt[mask])
        if max_std is not None:
            if not math.isfinite(float(max_std)) or float(max_std)<=0:
                raise ValueError('Invalid epistemic cutoff')
            mask &= uncertainty<=float(max_std)
        return nxt,reward,done,uncertainty,mask
