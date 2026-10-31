#!/usr/bin/env python3
"""Read-only SAC/MBPO neural-learning diagnosis for the 6D PVWatts experiment.

No optimizer steps, model fitting, checkpoint writes or environment data edits.
Only load trusted, locally-generated torch checkpoints (torch.load uses pickle).
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import os
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

H = 117
FIELDS = ('states', 'actions', 'rewards', 'next_states', 'dones')


def ensure(ok, message):
    if not bool(ok):
        raise AssertionError(message)


def clean(x):
    """Convert arrays/numerical types to standard JSON, nonfinite -> null."""
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return clean(x.tolist())
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return float(x) if np.isfinite(x) else None
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if x is None or isinstance(x, str):
        return x
    return str(x)


def stats(a):
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    a = a[np.isfinite(a)]
    if not len(a):
        return {'n': 0}
    return {'n': int(len(a)), 'mean': float(a.mean()), 'std': float(a.std()),
            'median': float(np.median(a)), 'p05': float(np.quantile(a, .05)),
            'p95': float(np.quantile(a, .95)), 'max_abs': float(np.max(np.abs(a)))}


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def csv_rows(path):
    if path is None:
        return []
    with open(path, newline='', encoding='utf-8') as fp:
        return list(csv.DictReader(fp))


def ordered_indices(replay):
    n = int(replay['count'])
    head = int(replay['head'])
    cap = int(replay['capacity'])
    ensure(0 <= n <= cap and 0 <= head < cap, 'Replay pointer invalid')
    if n == cap:
        return np.r_[np.arange(head, cap), np.arange(head)]
    ensure(head == n % cap, 'Partially filled replay head inconsistent with append-only FIFO')
    return np.arange(n)


def replay_arrays(replay):
    n = int(replay['count'])
    ensure(n > 0, 'Replay empty')
    arrays = {k: np.asarray(replay[k], dtype=np.float32) for k in FIELDS}
    expect = {'states': (n, 6), 'actions': (n, 2), 'rewards': (n, 1),
              'next_states': (n, 6), 'dones': (n, 1)}
    for k in FIELDS:
        ensure(arrays[k].shape == expect[k], f'Replay {k} shape {arrays[k].shape} != {expect[k]}')
        ensure(np.isfinite(arrays[k]).all(), f'Replay {k} nonfinite')
    ensure((np.abs(arrays['actions']) <= 1.0001).all(), 'Out-of-bounds replay action')
    ensure(np.isin(arrays['dones'], (0, 1)).all(), 'Nonbinary replay done')
    return arrays


def replay_contract(replay, power_norm, interval_hours, movement_penalty):
    a = replay_arrays(replay)
    order = ordered_indices(replay)
    n = len(order)
    s, ns, rew, done = (a[k][order] for k in ('states','next_states','rewards','dones'))
    assert done.shape == (n, 1)
    dtau = s[:, 5] - ns[:, 5]
    tau_error = np.abs(dtau - 1. / H)
    terminal_expected = ns[:, 5] <= 1e-7
    terminal_mismatch = done[:, 0].astype(bool) != terminal_expected
    # Real replay buffers record sequential transitions, synthetic ones do not.
    gross = ns[:, 2] * power_norm * interval_hours / 1000.
    valid = ns[:, 2] < (1. - 1e-7)
    reward_error = rew[:, 0] - gross
    if movement_penalty == 0:
        ensure(np.max(np.abs(reward_error[valid])) <= 4e-6,
               'Reward/power unit mismatch in replay')
    ensure(np.sum(terminal_mismatch) == 0, 'Terminal flags and tau mismatch')
    ensure(np.quantile(tau_error, .99) < 5e-6, 'Horizon decrement inconsistent')
    unique_actions = len(np.unique(np.round(a['actions'][order], decimals=3), axis=0))
    result = {'count': n, 'capacity': int(replay['capacity']), 'head': int(replay['head']),
              'terminal_count': int(done.sum()), 'terminal_mismatch_count': int(terminal_mismatch.sum()),
              'tau_step_abs_error': stats(tau_error), 'reward_minus_next_power_kwh': stats(reward_error[valid]),
              'power_feature_mean': float(s[:, 2].mean()),
              'power_feature_p95': float(np.quantile(s[:, 2], .95)),
              'fraction_power_clipped': float(np.mean(ns[:, 2] >= 1.-1e-7)),
              'unique_actions_rounded_0p001': unique_actions,
              'action_tilt_stats': stats(a['actions'][:,0]),
              'action_azimuth_stats': stats(a['actions'][:,1]),
              'fraction_abs_tilt_action_gt_0p98': float(np.mean(np.abs(a['actions'][:,0]) >= .98)),
              'fraction_positive_tilt_action': float(np.mean(a['actions'][:,0] > 0)),
              'real_action_tilt_boundary_fraction': float(np.mean((s[:,3] <= .01) | (s[:,3] >= .99)))}
    return result


def real_replay_contiguity(replay):
    a = replay_arrays(replay)
    idx = ordered_indices(replay)
    cur, nxt, done = a['states'][idx], a['next_states'][idx], a['dones'][idx,0]
    mask = done[:-1] < .5
    if len(idx) < 2 or not mask.any():
        return {'n_checked': 0}
    errors = np.max(np.abs(nxt[:-1][mask] - cur[1:][mask]), axis=1)
    return {'n_checked': len(errors), 'max_error': float(errors.max()),
            'fraction_different_gt_1e_5': float(np.mean(errors > 1e-5)),
            'fraction_different_gt_1e_3': float(np.mean(errors > 1e-3))}


def replay_coverage(replay, train_history=None):
    result = {}
    if 'scenario_ids' in replay:
        a = np.asarray(replay['scenario_ids'], dtype=np.int32)
        names = list(replay['scenario_names'])
        assert len(a) == int(replay['count'])
        unique, freq = np.unique(a, return_counts=True)
        months = Counter()
        for ix, n in zip(unique, freq):
            sid = names[int(ix)]
            months[sid[5:7]] += int(n)
        result['real_replay_distinct_scenarios'] = len(unique)
        result['real_replay_transitions_by_month'] = dict(sorted(months.items()))
        result['min_transitions_per_scenario'] = int(freq.min())
        result['max_transitions_per_scenario'] = int(freq.max())
    if train_history:
        by_month = Counter(r['scenario_id'][5:7] for r in train_history)
        day_count = Counter(r['scenario_id'] for r in train_history)
        result.update(training_episodes=len(train_history),
                      distinct_training_days=len(day_count),
                      repeated_training_days=sum(v>1 for v in day_count.values()),
                      max_repeats_of_one_day=max(day_count.values()),
                      training_episodes_by_month=dict(sorted(by_month.items())))
    return result


def minibatch_sampling_probe(n, batch_size, seed=122, repeats=100):
    rng = np.random.default_rng(seed)
    uniq = []
    for _ in range(repeats):
        ix = rng.integers(0, n, size=batch_size)
        uniq.append(len(np.unique(ix)) / batch_size)
    expected = n * (1. - (1. - 1. / n) ** batch_size) / batch_size
    return {'replay_count': n, 'batch_size': batch_size, 'sampled_with_replacement': True,
            'expected_fraction_unique_per_batch': float(expected),
            'empirical_fraction_unique_per_batch': stats(uniq)}


def infer_physics(ckpt):
    mdp = ckpt['resolved_mdp']
    return {k: mdp.get(k) for k in ('power_model','gamma_pdc','wind_reference_height_m',
                 'sapm_mount','wind_powerlaw_exponent','observation_mode','horizon',
                 'max_delta_tilt','max_delta_azimuth','movement_penalty','area','efficiency',
                 'start_time','periods','freq','power_source_sha256')}


def load_agent(kind, checkpoint, config):
    from SAC_workspace.src.agents.sac_agent import SACAgent
    a = config.algo if kind == 'sac' else config.sac.algo
    agent = SACAgent(hidden_sizes=a['hidden_sizes'], lr=a['lr'], gamma=a['gamma'],
        tau=a['tau'], reward_scale=a['reward_scale'], alpha_init=a['alpha_init'],
        target_entropy=a['target_entropy'], min_alpha=a['min_alpha'], device='cpu')
    agent.load_state_dict(checkpoint['agent'])
    agent.actor.eval();agent.critic.eval();agent.critic_target.eval()
    return agent


def check_checkpoint(path, kind, config, cfg_kwargs):
    """Read saved optimizer+replay state, check configuration and physics. Only trust local .pt."""
    from SAC_workspace.src.training.trainer import config_hash
    from MBPO_workspace.src.config import config_digest
    ck = torch.load(path, map_location='cpu', weights_only=False)
    ensure(ck.get('schema') == 1, f'{kind}: wrong checkpoint schema')
    digest = config_hash(config) if kind == 'sac' else config_digest(config)
    ensure(ck.get('hash' if kind == 'sac' else 'config_hash') == digest,
           f'{kind}: checkpoint config hash differs from current code')
    ensure(ck['epoch'] * H == ck['env_steps'], f'{kind}: checkpoint epoch/steps mismatch')
    ensure(ck['resolved_mdp']['power_model'] == cfg_kwargs['power_model'], f'{kind}: power model mismatch')
    if cfg_kwargs['power_model'] == 'pvwatts_sapm':
        from mbpo.env import pv_power_backend, pvlib_physics, pv_tracking
        h=hashlib.sha256()
        for module in (pv_power_backend,pvlib_physics,pv_tracking):
            h.update(Path(module.__file__).read_bytes())
        ensure(h.hexdigest()==ck['resolved_mdp'].get('power_source_sha256'),
               f'{kind}: checkpoint and current PV power source fingerprints differ')
    for group in ('actor','critic','critic_target'):
        ensure(all(torch.isfinite(t).all().item() for t in ck['agent'][group].values()),
               f'{kind}: nonfinite {group} checkpoint parameters')
    if kind == 'mbpo':
        ensure(ck.get('config') == config.to_dict(), 'MBPO embedded config inconsistent')
    return ck


def selected_sample(replay, sample_size, seed):
    n = int(replay['count'])
    rng = np.random.default_rng(seed)
    ix = rng.choice(n, size=min(sample_size, n), replace=False)
    return {k: np.asarray(replay[k], dtype=np.float32)[ix] for k in FIELDS}, ix


def tensor_batch(batch):
    return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in batch.items()}


def grad_norm(grads):
    return float(math.sqrt(sum(float(g.detach().square().sum().item()) for g in grads if g is not None)))


def sac_actor_critic_probe(agent, replay, sample_size, batch_size, seed):
    """No parameter updates. Computes actual SAC soft targets and gradients at checkpoint."""
    batch, indices = selected_sample(replay, sample_size, seed)
    b = tensor_batch(batch)
    s, a, s2, r, d = [b[k] for k in ('states','actions','next_states','rewards','dones')]
    # Freeze RNG-dependent reference by explicit local torch seed (restore global state).
    torch_state = torch.get_rng_state()
    try:
        torch.manual_seed(78991+seed)
        with torch.no_grad():
            a2, lp2 = agent.actor(s2, deterministic=False, with_logprob=True)
            tq1,tq2 = agent.critic_target(s2,a2)
            min_tq = torch.minimum(tq1,tq2)
            ent = agent.log_alpha.detach().exp().clamp_min(agent.min_alpha)*lp2
            target = agent.reward_scale*r + agent.gamma*(1-d)*(min_tq-ent)
            q1,q2 = agent.critic(s,a)
            pred = torch.minimum(q1,q2)
            td1 = q1-target;td2=q2-target
            bootstrap = agent.gamma*(1-d)*(min_tq-ent)
            ensure(torch.isfinite(target).all() and torch.isfinite(pred).all(), 'Nonfinite SAC Q/targets')
            terminal_error = torch.abs(target[d.view(-1)>.5]-(agent.reward_scale*r)[d.view(-1)>.5])
            if len(terminal_error):
                ensure(float(terminal_error.max())<1e-5,'Terminal SAC bootstrap not masked')
        # Critic full objective gradient at sampled batch; no optimizer.step().
        from torch.nn import functional as F
        critic_loss = F.mse_loss(agent.critic(s[:batch_size],a[:batch_size])[0], target[:batch_size]) + \
                      F.mse_loss(agent.critic(s[:batch_size],a[:batch_size])[1], target[:batch_size])
        critic_grad = torch.autograd.grad(critic_loss, list(agent.critic.parameters()), allow_unused=True)
        # Actor gradient with critic fixed wrt parameters, but differentiable wrt action.
        torch.manual_seed(98713+seed)
        act, logp = agent.actor(s[:batch_size],deterministic=False,with_logprob=True)
        cq1,cq2=agent.critic(s[:batch_size],act)
        alpha = agent.log_alpha.detach().exp().clamp_min(agent.min_alpha)
        actor_loss = (alpha*logp-torch.minimum(cq1,cq2)).mean()
        actor_grad=torch.autograd.grad(actor_loss,list(agent.actor.parameters()),allow_unused=True)
        # Evaluation action and local critic preferences; avoid treating Q as oracle.
        s.requires_grad_(True)
        chosen, _ = agent.actor(s,deterministic=True,with_logprob=False)
        chosen_det=chosen.detach().requires_grad_(True)
        da = chosen_det.clone(); da[:,0]=-1.
        zero = chosen_det.clone();zero[:]=0.
        ua = chosen_det.clone();ua[:,0]=1.
        qdet = torch.minimum(*agent.critic(s.detach(),chosen_det))
        qdown=torch.minimum(*agent.critic(s.detach(),da))
        qup=torch.minimum(*agent.critic(s.detach(),ua))
        qzero=torch.minimum(*agent.critic(s.detach(),zero))
        dq = torch.autograd.grad(qdet.sum(),chosen_det)[0]
        gaps=(qdet-qdown).detach().numpy().reshape(-1)
        # Time-bin TD stats, final step computed from pre-action tau=1/117.
        remaining=np.clip(np.rint(batch['states'][:,5]*H).astype(int),1,H)
        td_by_bin={}
        for name,mask in [('early',remaining>=79),('middle',(remaining<79)&(remaining>=40)),('late',remaining<40)]:
            td_by_bin[name]=stats(td1.detach().numpy().reshape(-1)[mask])
        q1q2=(q1-q2).detach().numpy()
        result={'sample_count':len(indices),'gamma':agent.gamma,'reward_scale':agent.reward_scale,
            'twin_Q_disagreement_Q1_minus_Q2':stats(q1q2),
            'target_entropy':agent.target_entropy,'alpha':agent.alpha,
            'checkpoint_optimizer_updates':agent.updates,
            'scaled_reward':stats((agent.reward_scale*r).detach().numpy()),
            'soft_bootstrap_term':stats(bootstrap.detach().numpy()),
            'entropy_term_alpha_logpi_next':stats(ent.detach().numpy()),
            'target_Q':stats(target.detach().numpy()),
            'predicted_clipped_double_Q':stats(pred.detach().numpy()),
            'critic_TD_error_Q1':stats(td1.detach().numpy()),
            'critic_TD_error_Q2':stats(td2.detach().numpy()),
            'critic_TD_abs_error_Q1':stats(torch.abs(td1).detach().numpy()),
            'terminal_target_scale_error_max':float(terminal_error.max()) if len(terminal_error) else None,
            'terminal_transition_fraction':float(d.mean()),
            'TD_Q1_by_remaining_horizon':td_by_bin,
            'critic_loss_probe':float(critic_loss.detach()),
            'actor_loss_probe':float(actor_loss.detach()),
            'critic_gradient_norm_probe':grad_norm(critic_grad),
            'actor_gradient_norm_probe':grad_norm(actor_grad),
            'deterministic_action_tilt':stats(chosen.detach().numpy()[:,0]),
            'deterministic_action_azimuth':stats(chosen.detach().numpy()[:,1]),
            'fraction_policy_tilt_positive':float(np.mean(chosen.detach().numpy()[:,0]>0)),
            'fraction_policy_tilt_near_upper_bound':float(np.mean(chosen.detach().numpy()[:,0]>.98)),
            'fraction_critic_Q_prefers_chosen_over_downward':float(np.mean(gaps>0)),
            'Q_chosen_minus_downward':stats(gaps),
            'Q_chosen_minus_no_action':stats((qdet-qzero).detach().numpy()),
            'Q_chosen_minus_upward':stats((qdet-qup).detach().numpy()),
            'critic_Q_derivative_wrt_tilt_action':stats(dq[:,0].detach().numpy()),
            'critic_Q_derivative_wrt_azimuth_action':stats(dq[:,1].detach().numpy()),
            'fraction_Q_gradient_prefers_increasing_tilt':float((dq[:,0]>0).float().mean()),
            'qualifications':['TD residuals sampled from the training replay are NOT independent validation errors.',
                    'Critic Q is soft, reward-scaled, long-horizon. A Q preference is not proof of physical optimality.',
                    'Nonzero gradient norm is not proof of nonconvergence; single-batch gradient is noisy.']}
        sample_rows=[]
        vals={k:v for k,v in [('q1',q1.detach().numpy().reshape(-1)),('q2',q2.detach().numpy().reshape(-1)),
                         ('target',target.detach().numpy().reshape(-1)),('td1',td1.detach().numpy().reshape(-1)),
                         ('td2',td2.detach().numpy().reshape(-1)),('q_gap',gaps)]}
        for j in range(min(200,len(indices))):
            sample_rows.append(dict(pool_index=int(indices[j]),tau=float(batch['states'][j,5]),
                                   reward_kwh=float(batch['rewards'][j,0]), done=int(batch['dones'][j,0]),
                                   **{k:float(v[j]) for k,v in vals.items()}))
        return result,sample_rows
    finally:
        torch.set_rng_state(torch_state)



def shadow_optimizer_step(kind, cfg, checkpoint, replay, seed):
    """In-memory independent SAC optimization from the exact checkpoint.

    Confirms real actor/critic/alpha Adam optimizer wiring and Polyak target
    arithmetic. Does NOT mutate the original agent, checkpoint, or replay.
    """
    agent=load_agent(kind,checkpoint,cfg)
    algo=(cfg.algo if kind=='sac' else cfg.sac.algo)
    batch,_=selected_sample(replay,int(algo['batch_size']),seed)
    actor_before=[p.detach().clone() for p in agent.actor.parameters()]
    critic_before=[p.detach().clone() for p in agent.critic.parameters()]
    target_before=[p.detach().clone() for p in agent.critic_target.parameters()]
    alpha_before=agent.alpha
    updates_before=agent.updates
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed+2811)
        metrics=agent.update(batch)
    def differences(before, after):
        d=[(q-p.detach()).cpu().numpy().ravel() for q,p in zip(before,after)]
        z=np.concatenate(d)
        return {'l2':float(np.linalg.norm(z)),'max_abs':float(np.max(np.abs(z)))}
    actor_delta=differences(actor_before,agent.actor.parameters())
    critic_delta=differences(critic_before,agent.critic.parameters())
    target_delta=differences(target_before,agent.critic_target.parameters())
    tau=float(agent.tau)
    polyak_err=max(float(torch.max(torch.abs(
        target.detach()-((1-tau)*old+tau*critic.detach()))).item())
        for old,critic,target in zip(target_before,agent.critic.parameters(),agent.critic_target.parameters()))
    ensure(actor_delta['l2']>0 and critic_delta['l2']>0, f'{kind}: shadow update did not change actor/critic')
    ensure(agent.updates==updates_before+1,f'{kind}: shadow optimizer update counter incorrect')
    ensure(polyak_err<5e-6,f'{kind}: target critic Polyak interpolation is incorrect')
    return {'status':'SHADOW_UPDATE_PASSED','actor_parameter_change':actor_delta,
            'critic_parameter_change':critic_delta,'target_parameter_change':target_delta,
            'target_polyak_max_abs_error':polyak_err,
            'alpha_before':alpha_before,'alpha_after':agent.alpha,
            'training_step_metrics':metrics,
            'warning':'One successful optimizer step demonstrates wiring, NOT convergence or accuracy of Q estimates.'}


def model_prediction_probe(ck, replay, sample_size, seed):
    """Compute ensemble mean deltas on real replay (IN-SAMPLE, not trusted holdout)."""
    from MBPO_workspace.src.dynamics import DynamicsEnsemble
    state=ck.get('dynamics')
    ensure(state is not None,'MBPO checkpoint missing learned ensemble')
    ensemble=DynamicsEnsemble(num_models=state['num_models'],num_elites=state['num_elites'],
                 hidden_dim=state['hidden_dim'],lr=state['lr'],device='cpu',seed=seed+510)
    ensemble.load_state_dict(state)
    batch,_=selected_sample(replay,sample_size,seed)
    xy=np.concatenate([batch['states'],batch['actions']],axis=-1)
    y=batch['next_states'][:,:3]-batch['states'][:,:3]
    x=torch.from_numpy(((xy-state['xmean'])/state['xstd']).astype(np.float32))
    mus=[];sigmas=[]
    with torch.no_grad():
        for ix in state['elites']:
            mu,ls=ensemble.nets[int(ix)](x)
            mus.append(mu.numpy()*np.asarray(state['ystd'])+np.asarray(state['ymean']))
            sigmas.append(np.exp(ls.numpy())*np.asarray(state['ystd']))
    means=np.stack(mus)
    sd=np.stack(sigmas)
    ensemble_mean=means.mean(0)
    rmse=np.sqrt(np.mean((ensemble_mean-y)**2,axis=0))
    persistence=np.sqrt(np.mean(y**2,axis=0))
    # Mixture predictive variance = mean elite aleatoric variance + variance
    # of elite predictive means (law of total variance; ensemble proxy only).
    mixture_variance=np.mean(sd**2,axis=0)+np.var(means,axis=0)
    z=(y-ensemble_mean)/np.sqrt(np.maximum(mixture_variance,1e-12))
    return {'sample_count':len(y),'ensemble_count':int(state['num_models']),'elites':list(map(int,state['elites'])),
            'dynamics_fit_count':int(state['fit_count']),
            'historical_last_model_metrics':clean(state.get('last_metrics',{})),
            'replay_RMSE_delta_normalized_DNI_DHI_power':rmse,
            'replay_RMSE_persistence_normalized_DNI_DHI_power':persistence,
            'replay_skill_vs_persistence':1.-rmse/np.clip(persistence,1e-9,None),
            'elite_mean_std_in_power_channel':stats(means[:,:,2].std(axis=0)),
            'predictive_mixture_std_power_channel':stats(np.sqrt(mixture_variance[:,2])),
            'residual_zscore_proxy_power_NOT_CALIBRATED':stats(z[:,2]),
            'warning':'This uses REAL TRAINING REPLAY, not a holdout; cannot certify out-of-sample model skill or uncertainty calibration.'}


def mbpo_mixture_report(mbpo_cfg, ck, rows):
    model=mbpo_cfg.model
    return {'configured_real_ratio':float(model.real_ratio),
            'configured_real_batch_count':int(mbpo_cfg.sac.algo['batch_size']*model.real_ratio),
            'configured_synthetic_batch_count':int(mbpo_cfg.sac.algo['batch_size'])-int(mbpo_cfg.sac.algo['batch_size']*model.real_ratio),
            'require_model_skill':bool(model.require_model_skill),
            'uncertainty_max_std':model.uncertainty_max_std,
            'model_rollout_length':int(model.rollout_length),
            'model_train_frequency_steps':int(model.model_train_freq),
            'model_pool_count':int(ck['model_pool']['count']),
            'accepted_synthetic_total_from_history':sum(int(float(row['synthetic_generated'])) for row in rows) if rows else None,
            'episodes_with_synthetic_batch':sum(float(row['mean_model_batch'])>0 for row in rows if row.get('mean_model_batch','').lower() not in ('','nan')) if rows else None,
            'critical_caveat':'Rotating day-level holdout on successive persistent fits risks validation leakage; positive skill is not independent proof of model accuracy.'}


def history_summary(rows, kind, expected_epoch, expected_steps):
    if not rows: return {'status':'not_supplied'}
    assert len(rows)>=expected_epoch, f'{kind} history shorter than checkpoint'
    last=rows[expected_epoch-1]
    ensure(int(last['epoch'])==expected_epoch and int(last['env_steps'])==expected_steps,
           f'{kind}: history/checkpoint episode or interaction count mismatch')
    ids=[str(r['scenario_id']) for r in rows[:expected_epoch]]
    def window(a,b):
        subset=rows[max(a,0):min(b,expected_epoch)]
        metrics={}
        for key in ('return_kwh','gross_energy_kwh','actor_loss','critic_loss',
                    'entropy','alpha','q_mean','target_mean','model_val_power_rmse_norm',
                    'model_skill_power_vs_persistence','mean_model_batch'):
            values=[]
            for r in subset:
                try: values.append(float(r[key]))
                except (ValueError,TypeError,KeyError): pass
            metrics[key]=stats(values)
        return metrics
    return {'status':'matched_to_checkpoint','episodes_at_checkpoint':expected_epoch,
           'real_steps_at_checkpoint':expected_steps, 'distinct_training_dates':len(set(ids)),
           'repeated_day_episodes':len(ids)-len(set(ids)),
           'max_episodes_on_one_day':max(Counter(ids).values()),
           'last_alpha':float(last['alpha']),
           'training_window_first_100':window(0,100),
           'training_window_previous_100':window(max(0,expected_epoch-200),max(0,expected_epoch-100)),
           'training_window_last_100':window(max(0,expected_epoch-100),expected_epoch),
           'warning':'Training returns vary by randomized weather days; early/late means do not establish convergence.'}


def select_branch_days(sids, n, seed):
    """Precommit across months, avoid choosing outcomes based on SAC performance."""
    if n <= 0: return []
    rng=np.random.default_rng(seed)
    months=defaultdict(list)
    for sid in sorted(sids): months[int(sid[5:7])].append(sid)
    picks=[]
    for m in (1,4,7,10):
        if months[m]:picks.append(months[m][int(rng.integers(len(months[m])))])
    if len(picks)>n: picks=picks[:n]
    rest=[s for s in sorted(sids) if s not in picks]
    if len(picks)<n and rest:
        ids=rng.choice(len(rest),size=min(n-len(picks),len(rest)),replace=False)
        picks.extend(rest[int(i)] for i in np.atleast_1d(ids))
    return picks


def read_evaluation_actions(eval_dir):
    path=Path(eval_dir)/'trajectories_detailed.csv'
    ensure(path.is_file(), f'Missing detailed trajectories: {path}')
    byday=defaultdict(lambda:defaultdict(list))
    for row in csv_rows(path):
        m=row['method'];sid=row['scenario_id']
        if m in ('sac','mbpo'):
            byday[sid][m].append(row)
    complete={sid:group for sid,group in byday.items() if all(m in group and len(group[m])==H for m in ('sac','mbpo'))}
    ensure(complete,'No complete SAC/MBPO evaluation trajectories found')
    for sid,group in complete.items():
        for method,rr in group.items():
            rr.sort(key=lambda r:int(r['step']))
            ensure([int(r['step']) for r in rr]==list(range(1,H+1)),f'Incomplete step order {sid}/{method}')
    return complete



def full_evaluation_action_replay(agents, grouped, power_norm, irradiance_norm):
    """Independent deterministic action reproduction for BOTH checkpoints.

    Every action after step 1 is tested against the previous step's observable
    6-vector. First actions require reset observation, verified in branches.
    """
    report={}
    for method in ('sac','mbpo'):
        state=[];actions=[]
        for sid,methods in grouped.items():
            rr=methods[method]
            for previous,current in zip(rr[:-1],rr[1:]):
                t=int(previous['step'])
                tilt=float(previous['tilt_deg']);az=float(previous['azimuth_deg'])
                state.append([np.clip(float(previous['dni_wm2'])/irradiance_norm,0.,1.),
                              np.clip(float(previous['dhi_wm2'])/irradiance_norm,0.,1.),
                              np.clip(float(previous['power_w'])/power_norm,0.,1.),
                              tilt/90.,(az%360.)/180.-1.,(H-t)/H])
                actions.append([float(current['action_tilt']),float(current['action_azimuth'])])
        ss=np.asarray(state,dtype=np.float32);targ=np.asarray(actions,dtype=np.float32)
        preds=[]
        with torch.no_grad():
            for j in range(0,len(ss),8192):
                a,_=agents[method].actor(torch.from_numpy(ss[j:j+8192]),
                              deterministic=True,with_logprob=False)
                preds.append(a.detach().cpu().numpy())
        pred=np.concatenate(preds)
        delta=np.max(np.abs(pred-targ),axis=1)
        report[method]={'checked_actions':int(len(delta)),
                         'max_abs_normalized_action_error':float(delta.max()),
                         'p99_abs_normalized_action_error':float(np.quantile(delta,.99)),
                         'fraction_above_1e_4':float(np.mean(delta>1e-4)),
                         'checkpoint_actions_match_saved_evaluation':bool(np.max(delta)<=1e-4)}
        ensure(report[method]['checkpoint_actions_match_saved_evaluation'],
               f'{method}: checkpoint does not reproduce evaluation actions; abort interpretation')
    return report


def reset_eval_env(cfg, sid, seed):
    from SAC_workspace.src.environment.environment import SACEnvironment
    env=SACEnvironment(cfg.eval_env)
    env.seed(int(seed))
    obs=env.reset(scenario_id=sid)
    ensure(abs(float(env.unwrapped.tilt)-30.)<1e-7 and abs(float(env.unwrapped.azimuth)-180.)<1e-7,
           'Branch replay requires original fixed 30/180 start; do not modify eval config silently')
    return env,obs


def replay_to_branch(env, obs, rr, branch_step):
    # Step k in recorded CSV is action from state before that step.
    for j in range(branch_step-1):
        r=rr[j]
        a=np.array([float(r['action_tilt']),float(r['action_azimuth'])],dtype=np.float32)
        obs, reward, done, info=env.step(a)
        if abs(float(info['power_dc_w'])-float(r['power_w']))>1e-4:
            raise AssertionError('Recorded trajectory failed exact physical replay at step '+str(j+1))
        if abs(float(info['tilt'])-float(r['tilt_deg']))>1e-4:
            raise AssertionError('Recorded orientation does not replay at step '+str(j+1))
    return obs


def roll_branch(cfg, agent, sid, seed, rr, step, override):
    env, obs=reset_eval_env(cfg,sid,seed)
    try:
        obs=replay_to_branch(env,obs,rr,step)
        with torch.no_grad():
            chosen=agent.act(obs,deterministic=True)
        recorded=np.array([float(rr[step-1]['action_tilt']),float(rr[step-1]['action_azimuth'])],dtype=np.float32)
        ensure(np.max(np.abs(chosen-recorded))<1e-4,
               'Chosen checkpoint action does not reproduce evaluation at '+str((sid,step,override)))
        current=chosen.copy()
        if override=='down':current[0]=-1.
        if override=='up':current[0]=1.
        if override=='none':current[:]=0.
        s=torch.as_tensor(obs,dtype=torch.float32).reshape(1,6)
        with torch.no_grad():
            q=lambda act: float(torch.minimum(*agent.critic(s,torch.as_tensor(act,dtype=torch.float32).reshape(1,2))).item())
            predicted_q=q(current)
        ret=0.; n=0; first_reward=None
        for j in range(step-1,H):
            action = current if j==step-1 else agent.act(obs,deterministic=True)
            obs,reward,done,info=env.step(action)
            if first_reward is None: first_reward=float(reward)
            ret+=float(reward);n+=1
            if done:break
        ensure(done and n==H-step+1,'Branched rollout did not terminate at expected horizon')
        return {'Q_estimate_reward_scaled_soft':predicted_q,'physical_return_remaining_kwh':ret,
                'first_reward_kwh':first_reward,'remaining_steps':n,
                'first_action_tilt':float(current[0]),'first_action_azimuth':float(current[1])}
    finally:
        env.close()



def soft_branch_return(cfg, agent, sid, eval_seed, rr, step, override, torch_seed):
    """One Monte Carlo draw of SAC's true conditional soft Q target objective.

    The FIRST action is conditioned on (deterministic chosen or alternative),
    hence its log-probability is not part of Q(s,a). From the NEXT state,
    sample future actions and add -alpha*log pi(a|s) at each remaining step.
    The entropy coefficient is frozen at the saved checkpoint value.
    """
    env,obs=reset_eval_env(cfg,sid,eval_seed)
    try:
        obs=replay_to_branch(env,obs,rr,step)
        chosen=agent.act(obs,deterministic=True)
        recorded=np.array([float(rr[step-1]['action_tilt']),float(rr[step-1]['action_azimuth'])],np.float32)
        ensure(np.max(np.abs(chosen-recorded))<1e-4,'Checkpoint action mismatch in stochastic branch')
        action=chosen.copy()
        if override=='down': action[0]=-1.
        scaled_soft=0.
        raw_kwh=0.
        alpha=agent.alpha
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(torch_seed))
            for j in range(step-1,H):
                future_lp=0.
                if j>step-1:
                    with torch.no_grad():
                        aa,logprob=agent.actor(torch.as_tensor(obs,dtype=torch.float32).reshape(1,6),
                                          deterministic=False,with_logprob=True)
                        action=aa.cpu().numpy().reshape(2).astype(np.float32)
                        future_lp=float(logprob.item())
                obs,reward,done,info=env.step(action)
                raw_kwh+=float(reward)
                scaled_soft+=agent.reward_scale*float(reward)-alpha*future_lp
                if done: break
        ensure(done,'Stochastic branch did not terminate')
        return {'soft_objective_reward_scaled':scaled_soft,'physical_return_kwh':raw_kwh}
    finally:
        env.close()


def branch_experiment(config, agents, eval_dir, days, steps, seed, soft_mc_repeats=0, recorded=None):
    if recorded is None:
        recorded=read_evaluation_actions(eval_dir)
    sids=select_branch_days(recorded,days,seed)
    output=[]
    for sid in sids:
        for method in ('sac','mbpo'):
            rr=recorded[sid][method]
            eval_seed=int(rr[0]['seed'])
            # Reproduce each method's original action up to branch; use same policy
            # thereafter. This tests *real closed-loop continuation* conditional
            # on a selected alternate first action, not a hindsight all-step oracle.
            for step in steps:
                chosen=roll_branch(config,agents[method],sid,eval_seed,rr,step,'chosen')
                down=roll_branch(config,agents[method],sid,eval_seed,rr,step,'down')
                record={'scenario_id':sid,'method':method,'start_step':step,'eval_seed':eval_seed,
                               'remaining_steps':chosen['remaining_steps'],
                               'real_return_chosen_kwh':chosen['physical_return_remaining_kwh'],
                               'real_return_downward_kwh':down['physical_return_remaining_kwh'],
                               'real_return_chosen_minus_downward_kwh':chosen['physical_return_remaining_kwh']-down['physical_return_remaining_kwh'],
                               'Q_chosen_minus_downward_reward_scaled_soft':chosen['Q_estimate_reward_scaled_soft']-down['Q_estimate_reward_scaled_soft'],
                               'first_step_gain_chosen_minus_downward_kwh':chosen['first_reward_kwh']-down['first_reward_kwh'],
                               'action_chosen_tilt':chosen['first_action_tilt'],
                               'action_chosen_azimuth':chosen['first_action_azimuth'],
                               'soft_mc_repeats':int(soft_mc_repeats),
                               'soft_MC_chosen_minus_downward_mean_reward_scaled':None,
                               'soft_MC_chosen_minus_downward_std_reward_scaled':None}
                if soft_mc_repeats>0:
                    gaps=[]
                    for rep in range(soft_mc_repeats):
                        # Same exogenous weather and paired torch RNG sequence.
                        key=f'{seed}-{method}-{sid}-{step}-{rep}'.encode()
                        ts=int(hashlib.sha256(key).hexdigest()[:8],16)%(2**31)
                        left=soft_branch_return(config,agents[method],sid,eval_seed,rr,step,'chosen',ts)
                        right=soft_branch_return(config,agents[method],sid,eval_seed,rr,step,'down',ts)
                        gaps.append(left['soft_objective_reward_scaled']-right['soft_objective_reward_scaled'])
                    record['soft_MC_chosen_minus_downward_mean_reward_scaled']=float(np.mean(gaps))
                    record['soft_MC_chosen_minus_downward_std_reward_scaled']=float(np.std(gaps,ddof=1)) if len(gaps)>1 else None
                output.append(record)
    return output


def branch_summary(rows):
    if not rows:
        return {'status':'not_run'}
    result={}
    for m in ('sac','mbpo'):
        sub=[r for r in rows if r['method']==m]
        if not sub:continue
        truth=np.asarray([r['real_return_chosen_minus_downward_kwh'] for r in sub])
        pred=np.asarray([r['Q_chosen_minus_downward_reward_scaled_soft'] for r in sub])
        valid=(np.abs(truth)>1e-7)&(np.abs(pred)>1e-6)
        result[m]={'n_branches':len(sub),'real_energy_gain_chosen_minus_downward_kwh':stats(truth),
            'critic_Q_gap_reward_scaled_soft':stats(pred),
            'fraction_critic_and_real_return_sign_agree':float(np.mean(np.sign(truth[valid])==np.sign(pred[valid]))) if valid.any() else None,
            'fraction_critic_prefers_chosen_but_real_return_prefers_down':float(np.mean((pred>1e-6)&(truth< -1e-7))),
            'fraction_critic_prefers_down_but_real_return_prefers_chosen':float(np.mean((pred< -1e-6)&(truth>1e-7))),
            'warning':'Critic estimates stochastic entropy-regularized scaled return; branch measures deterministic physical return. Disagreement is a diagnostic, NOT automatically a critic bug.'}
        if any(r.get('soft_MC_chosen_minus_downward_mean_reward_scaled') is not None for r in sub):
            observed=np.array([r['soft_MC_chosen_minus_downward_mean_reward_scaled'] for r in sub])
            result[m]['conditional_soft_MC_Q_gap_reward_scaled']=stats(observed)
            mask=(np.abs(observed)>1e-7)&(np.abs(pred)>1e-6)
            result[m]['fraction_critic_soft_MC_sign_agree']=float(np.mean(np.sign(observed[mask])==np.sign(pred[mask]))) if mask.any() else None
            result[m]['soft_mc_caution']='Finite Monte Carlo rollouts with fixed alpha; estimates can be noisy and do not independently validate the learned Q function.'
    return result


def write_csv(path, rows):
    if not rows:return
    fields=list(rows[0])
    with open(path,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def run(args):
    torch.set_num_threads(1)
    from SAC_workspace.src.config.config import load_config as sac_load
    from MBPO_workspace.src.config import load_config as mbpo_load
    from mbpo.env import pv_tracking
    sac_path=Path(args.sac_checkpoint).resolve()
    mbpo_path=Path(args.mbpo_checkpoint).resolve()
    ensure(sac_path.is_file() and mbpo_path.is_file(),'Missing checkpoint file(s)')
    # Preliminary metadata read only (trusted checkpoints). Resolve current physics via checkpoint.
    sc=torch.load(sac_path,map_location='cpu',weights_only=False)
    mc=torch.load(mbpo_path,map_location='cpu',weights_only=False)
    physics=infer_physics(sc)
    ensure(physics==infer_physics(mc),'SAC and MBPO checkpoint physical model differs')
    keys=('power_model','wind_reference_height_m','gamma_pdc','sapm_mount','wind_powerlaw_exponent')
    kw={k:physics[k] for k in keys}
    sac_cfg=sac_load(require_data=False,**kw)
    mbpo_cfg=mbpo_load(require_data=False,**kw)
    ensure(sac_cfg.to_dict()==mbpo_cfg.sac.to_dict(),'SAC and MBPO reference configs not identical')
    sc=check_checkpoint(sac_path,'sac',sac_cfg,kw)
    mc=check_checkpoint(mbpo_path,'mbpo',mbpo_cfg,kw)
    sagent=load_agent('sac',sc,sac_cfg)
    magent=load_agent('mbpo',mc,mbpo_cfg)
    srep=sc['replay'];mreal=mc['real_pool'];mfake=mc['model_pool']
    norm=float(pv_tracking.POWER_NORM)
    hours=5./60.
    penalty=float(physics['movement_penalty'])
    ensure(penalty==0,'Diagnostic currently requires movement_penalty=0 for reward/power identity')
    hist_s=csv_rows(args.sac_training_csv) if args.sac_training_csv else []
    hist_m=csv_rows(args.mbpo_training_csv) if args.mbpo_training_csv else []
    checks={'sac':replay_contract(srep,norm,hours,penalty),
            'mbpo_real':replay_contract(mreal,norm,hours,penalty),
            'mbpo_model':replay_contract(mfake,norm,hours,penalty)}
    for key,rep in [('sac',srep),('mbpo_real',mreal)]:
        checks[key]['real_replay_contiguity']=real_replay_contiguity(rep)
        ensure(checks[key]['real_replay_contiguity']['fraction_different_gt_1e_3']<.01,
               f'{key} real replay states are not sequential inside a day')
    updates_expected=max(0,int(sc['env_steps'])-int(sac_cfg.algo['start_steps']))*int(sac_cfg.algo['updates_per_step'])
    ensure(int(sc['agent']['updates'])==updates_expected, 'SAC optimizer update count differs from training schedule')
    updates_expected_mb=max(0,int(mc['env_steps'])-int(sac_cfg.algo['start_steps']))*int(sac_cfg.algo['updates_per_step'])
    ensure(int(mc['agent']['updates'])==updates_expected_mb, 'MBPO SAC optimizer update count differs from training schedule')
    report={'schema':'sac-mbpo-deep-v1','read_only':True,
        'checkpoint':{'sac':{'path':str(sac_path),'sha256':sha(sac_path),'epoch':sc['epoch'],'real_steps':sc['env_steps'],
                               'agent_updates':sc['agent']['updates'],'expected_updates':updates_expected},
                      'mbpo':{'path':str(mbpo_path),'sha256':sha(mbpo_path),'epoch':mc['epoch'],'real_steps':mc['env_steps'],
                               'agent_updates':mc['agent']['updates'],'expected_updates':updates_expected_mb}},
        'config':{'shared_sac_algo':sac_cfg.algo,'mbpo_model':mbpo_cfg.model.__dict__,
                  'physics':physics,'power_norm_w':norm,'irradiance_norm':float(pv_tracking.IRRADIANCE_NORM),
                  'train_initial_orientation_randomized':bool(sac_cfg.train_env['randomize_initial_orientation']),
                  'eval_initial_orientation_randomized':bool(sac_cfg.eval_env['randomize_initial_orientation'])},
        'replay_contract':checks,
        'coverage':{'sac':replay_coverage(srep,hist_s),'mbpo_real':replay_coverage(mreal,hist_m)},
        'sampling':{k:minibatch_sampling_probe(int(rep['count']),
                    sac_cfg.algo['batch_size'] if k!='mbpo_model' else
                    sac_cfg.algo['batch_size']-int(sac_cfg.algo['batch_size']*mbpo_cfg.model.real_ratio), args.seed)
            for k,rep in [('sac_real',srep),('mbpo_real',mreal),('mbpo_model',mfake)]},
        'training_history':{'sac':history_summary(hist_s,'sac',sc['epoch'],sc['env_steps']),
                            'mbpo':history_summary(hist_m,'mbpo',mc['epoch'],mc['env_steps'])},
        'mbpo_mixture':mbpo_mixture_report(mbpo_cfg,mc,hist_m),
        'important_limitations':[
            'The same six-sensor observation and SACAgent implementation are used by SAC and MBPO.',
            'Checkpoint diagnostics are snapshots of learned values; they do not prove algorithm convergence.',
            'Replay TD errors and dynamics errors are evaluated on training replay, not an independent held-out set.',
            'Branched continuations evaluate frozen deterministic policies on selected dates, not the SAC stochastic entropy-regularized objective.',
            'Changing batch sizes or training for more epochs needs a controlled new run and fixed development evaluation.',
            'MBPO rotating dynamics holdouts can leak previously trained days; independent validation remains required.',
            'A policy improvement in these diagnostic evaluation days must not be used to tune and then claim untouched test-set results.']}
    probes={}; rows=[]
    for key,agent,rep,sd in [('sac_real',sagent,srep,args.seed),
                             ('mbpo_real',magent,mreal,args.seed+1),
                             ('mbpo_synthetic',magent,mfake,args.seed+2)]:
        result, rr=sac_actor_critic_probe(agent,rep,args.samples,sac_cfg.algo['batch_size'],sd)
        probes[key]=result
        for row in rr: rows.append({'origin':key,**row})
    report['learning_probes']=probes
    report['shadow_optimizer_steps']={
        'sac':shadow_optimizer_step('sac',sac_cfg,sc,srep,args.seed+8),
        'mbpo':shadow_optimizer_step('mbpo',mbpo_cfg,mc,mreal,args.seed+9),
    }
    report['mbpo_model_fit_to_real_replay']=model_prediction_probe(mc,mreal,args.samples,args.seed+3)
    branches=[]
    if args.eval_dir:
        # Check both checkpoint identities before interpreting real branch Q gaps.
        ensure(Path(args.eval_dir).is_dir(),f'Evaluation folder not found: {args.eval_dir}')
        agents={'sac':sagent,'mbpo':magent}
        grouped=read_evaluation_actions(args.eval_dir)
        report['evaluation_action_replay']=full_evaluation_action_replay(
            agents,grouped,norm,float(pv_tracking.IRRADIANCE_NORM))
        if args.branch_days>0:
            # Reset to the actual paired 30/180 start; replay original control actions.
            steps=[int(t) for t in args.branch_steps.split(',')]
            ensure(steps and all(1<=x<=H for x in steps),'Invalid branch steps')
            branches=branch_experiment(sac_cfg,agents,args.eval_dir,args.branch_days,steps,args.seed,args.soft_mc_repeats,grouped)
            report['real_branch_rollouts']=branch_summary(branches)
        else:
            report['real_branch_rollouts']={'status':'NOT_RUN','reason':'--branch-days=0'}
    else:
        report['evaluation_action_replay']={'status':'NOT_RUN: --eval-dir not supplied'}
        report['real_branch_rollouts']={'status':'NOT_RUN: --eval-dir not supplied'}
    out=Path(args.outdir).resolve()
    out.mkdir(parents=True,exist_ok=True)
    with (out/'deep_diagnostic.json').open('w',encoding='utf-8') as fp:
        json.dump(clean(report),fp,indent=2,sort_keys=True,allow_nan=False)
    write_csv(out/'sampled_td_residuals.csv',rows)
    write_csv(out/'branched_real_continuations.csv',branches)
    readme=(
        '# SAC/MBPO deep diagnostic output\n\n'
        'This archive contains read-only analysis of local trusted checkpoints.\n'
        'Bellman errors are replay-based; branch returns use the real PVWatts environment.\n'
        'No training code, checkpoints, policy weights, dataset files, or original evaluation files are altered.\n'
        'Send this ZIP for review. Do not treat sampled replay or diagnostic dates as final independent test sets.\n'
    )
    (out/'README_RESULTS.md').write_text(readme,encoding='utf-8')
    bundle=out/'SEND_ME_SAC_MBPO_DEEP_DIAGNOSTIC.zip'
    with zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED,compresslevel=7) as z:
        for p in ['deep_diagnostic.json','sampled_td_residuals.csv','README_RESULTS.md']:
            z.write(out/p,arcname=p)
        if branches:z.write(out/'branched_real_continuations.csv',arcname='branched_real_continuations.csv')
    print('\nREAD-ONLY DEEP DIAGNOSTIC COMPLETED')
    for k in ('sac_real','mbpo_real','mbpo_synthetic'):
        m=report['learning_probes'][k]
        print(f"{k}: abs TD Q1 mean={m['critic_TD_abs_error_Q1']['mean']:.3f}, Q prefers chosen/down={m['fraction_critic_Q_prefers_chosen_over_downward']:.1%}, actor grad={m['actor_gradient_norm_probe']:.3g}")
    print('Branch:',json.dumps(clean(report['real_branch_rollouts']),indent=2)[:1300])
    print('SEND THIS FILE:',bundle)
    return report,bundle


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--sac-checkpoint',required=True,help='TRUSTED locally saved SAC .pt; NEVER untrusted files')
    p.add_argument('--mbpo-checkpoint',required=True,help='TRUSTED locally saved MBPO .pt')
    p.add_argument('--sac-training-csv',default=None)
    p.add_argument('--mbpo-training-csv',default=None)
    p.add_argument('--eval-dir',default=None,help='Paired evaluation folder containing trajectories_detailed.csv')
    p.add_argument('--outdir',default='MBPO_workspace/evaluation/deep_actor_critic_diagnostic')
    p.add_argument('--samples',type=int,default=2048,help='Maximum sampled replay rows per pool for Q/TD probes')
    p.add_argument('--branch-days',type=int,default=4,help='Number of stratified day scenarios; 0 skips real branched rollouts')
    p.add_argument('--branch-steps',default='12,42,78',help='1-based decision steps to compare alternative actions')
    p.add_argument('--soft-mc-repeats',type=int,default=0,help='Optional stochastic/entropy-regularized continuation replicates per branch (more physical rollouts)')
    p.add_argument('--seed',type=int,default=712)
    args=p.parse_args()
    ensure(args.samples>=256,'--samples should be at least a SAC minibatch of 256')
    ensure(args.branch_days>=0 and args.soft_mc_repeats>=0,'Negative branch/MC counts')
    ensure(args.soft_mc_repeats==0 or args.branch_days>0,'Soft MC requires positive --branch-days')
    run(args)


if __name__=='__main__':
    main()
