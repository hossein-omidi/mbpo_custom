#!/usr/bin/env python3
"""Read-only paired critic/policy quality audit for 6D finite-horizon PV SAC/MBPO.

No training, policy updates, environment edits, or checkpoint writes.
All comparisons are on a fixed NSRDB scenario, seed and initial orientation.
Stochastic soft-Q Monte Carlo is conditional on a particular recorded weather day,
NOT a population calibration guarantee over unseen weather.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


def candidates(action, step=0.5):
    """Finite, clipped feasible perturbations, with duplicate actions removed."""
    a = np.asarray(action, dtype=np.float32)
    if a.shape != (2,) or not np.isfinite(a).all() or np.any(np.abs(a)>1.000001):
        raise ValueError('Invalid actor action')
    proposals = [('actor', a), ('zero', np.zeros(2, np.float32)),
                 ('tilt_plus', a + [step, 0.]), ('tilt_minus', a - [step, 0.]),
                 ('azimuth_plus', a + [0., step]), ('azimuth_minus', a - [0., step])]
    out=[]
    for name, value in proposals:
        val=np.clip(np.asarray(value,dtype=np.float32),-1.,1.)
        if not any(np.allclose(val, v, atol=1e-7) for _,v in out):
            out.append((name,val))
    return out


def soft_return(rewards, future_logps, gamma, reward_scale, alpha):
    """Q^pi(s_t,a_t): sum gamma^k*(scaled r - alpha*logpi at k>=1).

    A fixed first action is conditioned on, so entropy at step t is excluded.
    """
    rewards=np.asarray(rewards,dtype=float)
    lp=np.asarray(future_logps,dtype=float)
    if rewards.ndim!=1 or len(rewards)<1 or lp.shape!=(len(rewards)-1,):
        raise ValueError('Soft-return shape mismatch')
    if not np.isfinite(rewards).all() or not np.isfinite(lp).all():
        raise ValueError('Nonfinite reward/log-probability')
    if not 0<=gamma<=1 or reward_scale<=0 or alpha<0:raise ValueError('Invalid soft-return configuration')
    power=np.power(gamma,np.arange(len(rewards)))
    return float(np.dot(power,rewards*reward_scale) -
                 alpha*np.dot(power[1:],lp))


def _mean_std(values):
    x=np.asarray(values,float)
    if x.ndim!=1 or len(x)==0 or not np.isfinite(x).all():raise ValueError('Nonfinite/empty observations')
    return float(np.mean(x)),float(np.std(x,ddof=1)) if len(x)>1 else None


def pair_agreement(q, mc):
    """Uncertainty-free descriptive fraction of candidate-pair orderings agreeing."""
    q=np.asarray(q,float);mc=np.asarray(mc,float)
    if q.shape!=mc.shape or q.ndim!=1 or not np.isfinite(q).all() or not np.isfinite(mc).all():
        raise ValueError('Invalid candidate scores')
    agree=total=0
    for i in range(len(q)):
        for j in range(i+1,len(q)):
            d1=q[i]-q[j];d2=mc[i]-mc[j]
            if abs(d1)<1e-9 or abs(d2)<1e-9:continue
            total+=1;agree+=int(d1*d2>0)
    return (agree/total if total else None),total


def paired_candidate_statistics(records, score_samples, *, min_repeats=20, confidence=.95):
    """Conditional, paired stochastic-return comparisons on ONE fixed weather day.

    The same stochastic draw seed is reused for each candidate; subtract matching
    replicate returns BEFORE estimating Monte Carlo uncertainty. Do not call this
    a weather- or training-population confidence interval. The nominal t interval
    is a diagnostic only, not multiple-comparison corrected.

    With fewer than ``min_repeats`` samples, do not declare action rankings
    statistically resolved. Even with enough repetitions, near-normal paired
    differences and independent stochastic draws across repeats are assumptions.
    """
    from scipy.stats import t as student_t
    if not 0 < confidence < 1 or min_repeats < 2:
        raise ValueError('Invalid paired Monte Carlo inference settings')
    pairs=[];resolved_agree=[]
    for i, a in enumerate(records):
        for b in records[i+1:]:
            x=np.asarray(score_samples[a['candidate']],dtype=float)
            y=np.asarray(score_samples[b['candidate']],dtype=float)
            if x.ndim != 1 or x.shape != y.shape or len(x)<2 or not (np.isfinite(x).all() and np.isfinite(y).all()):
                raise ValueError('Paired score vectors must be equally sized finite 1D arrays (n>=2)')
            d=x-y
            mean=float(d.mean());sd=float(d.std(ddof=1));se=sd/(len(d)**.5)
            qdiff=float(a['critic_min_soft_scaled']-b['critic_min_soft_scaled'])
            detdiff=float(a['deterministic_to_go_kwh']-b['deterministic_to_go_kwh'])
            # Exact simulated equivalence, e.g. two outward actions clipped
            # to the same tilt hard stop. These MUST NOT enter rank scoring.
            if np.max(np.abs(d)) <= 1e-8 and abs(detdiff) <= 1e-10:
                status='physically_equivalent_on_tested_day';low=high=mean
            elif len(d)<min_repeats:
                status='insufficient_stochastic_repeats';low=high=None
            else:
                half=float(student_t.ppf((1+confidence)/2,len(d)-1))*se
                low=mean-half;high=mean+half
                status='unresolved_conditional_difference' if low<=0<=high else 'resolved_conditional_difference'
            agree=None
            if status=='resolved_conditional_difference' and abs(qdiff)>1e-9:
                agree=bool(qdiff*mean>0)
                resolved_agree.append(agree)
            pairs.append({'candidate_a':a['candidate'],'candidate_b':b['candidate'],
                'n_paired_stochastic_repeats':int(len(d)),
                'critic_difference_scaled':qdiff,'mc_paired_difference_scaled':mean,
                'mc_paired_difference_sd_scaled':sd,'mc_paired_difference_se_scaled':se,
                'mc_paired_ci95_low_scaled':low,'mc_paired_ci95_high_scaled':high,
                'deterministic_difference_kwh':detdiff,
                'status':status,'critic_order_correct_if_resolved':agree})
    return pairs,({'resolved_count':len(resolved_agree),
                   'resolved_agreement_fraction':float(np.mean(resolved_agree)) if resolved_agree else None,
                   'note':'Conditional action-noise Monte Carlo only; nominal t intervals unadjusted for multiple comparisons; no population generalization.'})


def _critic_value(agent, state, action):
    import torch
    with torch.no_grad():
        s=torch.as_tensor(np.asarray(state,np.float32),device=agent.device).reshape(1,6)
        a=torch.as_tensor(np.asarray(action,np.float32),device=agent.device).reshape(1,2)
        qa,qb=agent.critic(s,a)
        return float(torch.minimum(qa,qb).item())


def _sample_action_lp(agent, state):
    import torch
    with torch.no_grad():
        s=torch.as_tensor(np.asarray(state,np.float32),device=agent.device).reshape(1,6)
        a,lp=agent.actor(s,deterministic=False,with_logprob=True)
    return a.squeeze(0).cpu().numpy().astype(np.float32),float(lp.item())


def baseline_rollout(env,agent,sid,seed):
    """Frozen deterministic policy; store actual pre-action states/prefix actions."""
    env.seed(int(seed)); s=env.reset(scenario_id=sid)
    initial=(float(env.unwrapped.tilt),float(env.unwrapped.azimuth))
    states=[];actions=[];rewards=[];next_states=[]
    for t in range(env.horizon):
        a=agent.act(s,deterministic=True)
        ns,r,done,_=env.step(a)
        states.append(np.asarray(s,np.float32).copy()); actions.append(a.copy())
        rewards.append(float(r));next_states.append(np.asarray(ns,np.float32).copy())
        if bool(done)!=(t==env.horizon-1):raise RuntimeError('Terminal mismatch in baseline')
        s=ns
    return dict(states=states,actions=actions,rewards=rewards,next_states=next_states,
                initial_pose=initial,return_kwh=float(sum(rewards)))


def intervene_rollout(env,agent,sid,seed,baseline,step,action,*,stochastic=False,torch_seed=0):
    """Reset to the same day and exact deterministic prefix; alter only action at step.

    For stochastic tails, fresh independent Gaussian draws across repeat seeds.
    The same repeat seed is reused for alternative first actions (paired draws).
    """
    import torch
    if not 0<=step<env.horizon:raise ValueError('Probe step outside horizon')
    env.seed(int(seed));s=env.reset(scenario_id=sid)
    if not np.allclose((env.unwrapped.tilt,env.unwrapped.azimuth),baseline['initial_pose'],atol=1e-7):
        raise RuntimeError('Paired initial pose mismatch')
    for k in range(step):
        s,_,done,_=env.step(baseline['actions'][k])
        if done:raise RuntimeError('Premature terminal in prefix')
    if not np.allclose(s,baseline['states'][step],atol=2e-6,rtol=2e-6):
        raise RuntimeError('Intervention prefix fails to reproduce the frozen policy state')
    rewards=[];logps=[]
    s,r,done,_=env.step(action);rewards.append(float(r))
    with torch.random.fork_rng(devices=[]):
        if stochastic:torch.manual_seed(int(torch_seed))
        for j in range(step+1,env.horizon):
            if done:raise RuntimeError('Premature terminal during intervention')
            if stochastic:
                a,lp=_sample_action_lp(agent,s);logps.append(lp)
            else:a=agent.act(s,deterministic=True)
            s,r,done,_=env.step(a);rewards.append(float(r))
    if len(rewards)!=env.horizon-step or not done:raise RuntimeError('Incomplete intervened episode')
    return rewards,logps


def soft_td_target(reward,done,next_min_target_q,next_logp,*,gamma,reward_scale,alpha):
    """One-step clipped-double-Q SAC-v2 target in critic (scaled-reward) units."""
    for x in (reward,next_min_target_q,next_logp,gamma,reward_scale,alpha):
        if not np.isfinite(x):raise ValueError('Nonfinite soft Bellman quantity')
    if done not in (True,False,0,1):raise ValueError('Invalid terminal flag')
    if not 0<=gamma<=1 or reward_scale<=0 or alpha<0:raise ValueError('Invalid SAC scales')
    return float(reward_scale*reward + gamma*(1-int(done))*(next_min_target_q-alpha*next_logp))


def bellman_probe(agent,trace,step,n_samples=8,base_seed=1201):
    """Reference Bellman residual on the *real* logged transition at probe step.

    This is a target-network self-consistency diagnostic, not an independent
    measurement of optimal Q or a held-out weather performance guarantee.
    """
    import torch
    state=trace['states'][step]
    action=trace['actions'][step]
    reward=float(trace['rewards'][step])
    nxt=trace['next_states'][step]
    terminal=step==len(trace['rewards'])-1
    critic_q=_critic_value(agent,state,action)
    targets=[]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(base_seed))
        with torch.no_grad():
            ns=torch.as_tensor(nxt,dtype=torch.float32,device=agent.device).reshape(1,6)
            for _ in range(n_samples):
                if terminal:qnext=lp=0.
                else:
                    a2,l2=agent.actor(ns,deterministic=False,with_logprob=True)
                    q1,q2=agent.critic_target(ns,a2)
                    qnext=float(torch.minimum(q1,q2).item());lp=float(l2.item())
                targets.append(soft_td_target(reward,terminal,qnext,lp,
                    gamma=agent.gamma,reward_scale=agent.reward_scale,alpha=agent.alpha))
    target=float(np.mean(targets))
    return {'observed_transition_min_q_scaled':critic_q,
            'mean_stochastic_one_step_td_target_scaled':target,
            'bellman_residual_min_q_minus_td_scaled':critic_q-target,
            'bellman_residual_equiv_kwh':(critic_q-target)/agent.reward_scale,
            'target_action_samples':n_samples}


def _load_expected(path):
    if not path:return {}
    fp=Path(path)/'episodes.csv'
    if not fp.exists():raise FileNotFoundError('Cannot independently verify baseline: '+str(fp))
    result={}
    with fp.open(newline='',encoding='utf-8') as f:
        for r in csv.DictReader(f):
            m=r['method']
            if m in ('mbpo','sac'):
                key=(str(r['scenario_id']),int(r['seed']),m)
                if key in result:raise ValueError('Duplicate measured episodes in '+str(fp))
                result[key]=float(r['net_energy_kwh'])
    return result


def _training_days(path):
    if path is None:return None
    fp=Path(path)
    if not fp.is_file():raise FileNotFoundError('Missing training CSV '+str(fp))
    with fp.open(newline='',encoding='utf-8') as f:
        rdr=csv.DictReader(f)
        if 'scenario_id' not in (rdr.fieldnames or ()):
            return None
        return {str(r['scenario_id']) for r in rdr if r.get('scenario_id')}


def _model_one_step_diagnostics(ckpt,cfg,baselines):
    """Descriptive ensemble elite-mean prediction error, on ACTUALLY visited actions.

    This is NOT an untouched held-out dataset unless input days were excluded
    from all ensemble fitting. Does not test counterfactual action coverage.
    """
    import torch
    from MBPO_workspace.src.dynamics import DynamicsEnsemble
    m=cfg.model
    ensemble=DynamicsEnsemble(num_models=m.num_models,num_elites=m.num_elites,
        hidden_dim=m.hidden_dim,lr=m.model_lr,device='cpu',seed=0)
    data=torch.load(ckpt,map_location='cpu',weights_only=False)
    ensemble.load_state_dict(data['dynamics'])
    if not ensemble.fitted:return {'status':'NOT_FITTED','details':[]}
    results=[]
    for sid,trace in baselines.items():
        states=np.array(trace['states'],dtype=np.float32)
        actions=np.array(trace['actions'],dtype=np.float32)
        target=np.array(trace['next_states'],dtype=np.float32)[:,:3]
        x=np.concatenate([states,actions],axis=1)
        tx=torch.as_tensor((x-ensemble.xmean)/ensemble.xstd,dtype=torch.float32)
        elite=[]
        with torch.no_grad():
            for k in ensemble.elites:
                mu,_=ensemble.nets[k](tx)
                elite.append(mu.numpy()*ensemble.ystd+ensemble.ymean)
        nxt_est=np.clip(states[:,:3]+np.mean(elite,axis=0),0.,1.)
        rmse=np.sqrt(np.mean((nxt_est-target)**2,axis=0))
        persistent=np.sqrt(np.mean((states[:,:3]-target)**2,axis=0))
        results.append(dict(scenario_id=sid,channels=['dni_norm','dhi_norm','power_norm'],
                            next_state_rmse_norm=list(map(float,rmse)),
                            persistence_rmse_norm=list(map(float,persistent)),
                            power_skill_vs_persistence=(float(1-rmse[2]/persistent[2])
                                if persistent[2]>1e-12 else None)))
    return {'status':'DESCRIPTIVE_VISITED_ACTIONS_ONLY','details':results}


def run_audit(*,sac_checkpoint=None,mbpo_checkpoint=None,days=('2024-12-23','2018-12-27'),
              steps=(35,65),repeats=3,seed=42,eval_dir=None,mbpo_training_csv=None,
              sac_training_csv=None,candidate_step=.5,model_diagnostic=False):
    if repeats<1 or not 0<candidate_step<=2:raise ValueError('Invalid probe controls')
    if not days or len(set(days))!=len(days):raise ValueError('No or duplicate date IDs')
    if not steps or len(set(steps))!=len(steps):raise ValueError('No or duplicate probe steps')
    if not (sac_checkpoint or mbpo_checkpoint):raise ValueError('At least one checkpoint required')
    from MBPO_workspace.src.config import load_config
    from MBPO_workspace.src.evaluation import load_mbpo_agent,_check_physics
    from SAC_workspace.src.evaluation.evaluation import load_agent
    from SAC_workspace.src.environment.environment import SACEnvironment
    cfg=load_config(require_data=True)
    available={}
    if sac_checkpoint:available['sac']=load_agent(cfg.sac,sac_checkpoint,device='cpu')
    if mbpo_checkpoint:available['mbpo']=load_mbpo_agent(cfg,mbpo_checkpoint,device='cpu')
    expected=_load_expected(eval_dir)
    training={'sac':_training_days(sac_training_csv),'mbpo':_training_days(mbpo_training_csv)}
    env=SACEnvironment(cfg.sac.eval_env)
    rows=[];probes=[];paired_rows=[];model_baselines={}
    try:
        for name,agent in available.items():
            _check_physics(agent.checkpoint_resolved_mdp if name=='sac' else agent.saved_mdp,env.resolved_mdp)
            for sid in days:
                baseline=baseline_rollout(env,agent,sid,seed)
                model_baselines[sid]=baseline if name=='mbpo' else model_baselines.get(sid)
                key=(sid,int(seed),name)
                if eval_dir:
                    if key not in expected:raise ValueError('Saved evaluation lacks paired baseline '+str(key))
                    discrepancy=baseline['return_kwh']-expected[key]
                    if not np.isclose(discrepancy,0.,rtol=0.,atol=1e-5):
                        raise ValueError('Baseline replay differs from saved evaluation: '+str((key,discrepancy)))
                print(f'{name.upper()} {sid}: baseline net={baseline["return_kwh"]:.6f} kWh',flush=True)
                for step in steps:
                    if not 0<=step<env.horizon:raise ValueError('Probe step outside horizon')
                    acts=candidates(baseline['actions'][step],candidate_step)
                    here=[];score_samples={}
                    for title,action in acts:
                        det_rewards,_=intervene_rollout(env,agent,sid,seed,baseline,step,action)
                        det_sum=float(sum(det_rewards))
                        if title=='actor' and not np.isclose(det_sum,sum(baseline['rewards'][step:]),atol=1e-7):
                            raise RuntimeError('Actor control intervention failed exact deterministic-replay test')
                        soft_samples=[]
                        for k in range(repeats):
                            samples,lp=intervene_rollout(env,agent,sid,seed,baseline,step,action,
                                stochastic=True,torch_seed=1000003+seed*1009+step*97+k)
                            soft_samples.append(soft_return(samples,lp,agent.gamma,agent.reward_scale,agent.alpha))
                        mean,sd=_mean_std(soft_samples)
                        score_samples[title]=list(soft_samples)
                        q=_critic_value(agent,baseline['states'][step],action)
                        record=dict(method=name,scenario_id=sid,seed=seed,step_zero_based=step,
                            candidate=title,action_tilt=float(action[0]),action_azimuth=float(action[1]),
                            critic_min_soft_scaled=q,critic_min_soft_equiv_kwh=q/agent.reward_scale,
                            mc_soft_mean_scaled=mean,mc_soft_sd_scaled=sd,
                            mc_soft_mean_equiv_kwh=mean/agent.reward_scale,
                            mc_soft_sd_equiv_kwh=(sd/agent.reward_scale if sd is not None else None),
                            critic_minus_mc_soft_scaled=q-mean,
                            deterministic_to_go_kwh=det_sum,
                            deterministic_gain_vs_actor_kwh=None,stochastic_repeat_count=repeats)
                        here.append(record);rows.append(record)
                    base=next(r for r in here if r['candidate']=='actor')
                    for item in here:item['deterministic_gain_vs_actor_kwh']=item['deterministic_to_go_kwh']-base['deterministic_to_go_kwh']
                    q=[r['critic_min_soft_scaled'] for r in here]
                    z=[r['mc_soft_mean_scaled'] for r in here]
                    agreement,npairs=pair_agreement(q,z)
                    pairs,conditional=paired_candidate_statistics(here,score_samples)
                    for pair in pairs:
                        paired_rows.append(dict(method=name,scenario_id=sid,seed=seed,
                                                step_zero_based=step,**pair))
                    best=max(here,key=lambda r:r['deterministic_to_go_kwh'])
                    td=bellman_probe(agent,baseline,step,base_seed=3001+step+seed)
                    report=dict(method=name,scenario_id=sid,step_zero_based=step,
                        one_step_real_transition_bellman=td,
                        n_candidates=len(here),n_comparable_ordered_pairs=npairs,
                        critic_vs_conditional_mc_pair_order_agreement=agreement,
                        conditional_paired_mc=conditional,
                        n_physically_equivalent_candidate_pairs=sum(
                            p['status']=='physically_equivalent_on_tested_day' for p in pairs),
                        critic_soft_bias_abs_mean_scaled=float(np.mean(np.abs(np.asarray(q)-z))),
                        best_deterministic_candidate=best['candidate'],
                        best_deterministic_candidate_gain_kwh=float(max(0.,best['deterministic_gain_vs_actor_kwh'])),
                        actor_deterministic_future_kwh=base['deterministic_to_go_kwh'],
                        interpretation='Local finite-candidate intervention; NOT global optimality or a generalization certificate')
                    probes.append(report)
                    print(f'  step {step}: best={best["candidate"]}, +{report["best_deterministic_candidate_gain_kwh"]:.6f} kWh; '
                          f'raw Q/MC agreement={agreement}; resolved paired comparisons={conditional["resolved_count"]}',flush=True)
    finally:
        env.close()
    model_info=None
    if model_diagnostic and mbpo_checkpoint:
        model_info=_model_one_step_diagnostics(mbpo_checkpoint,cfg,model_baselines)
    overlap={m:{'training_log_overlap_count':len(set(days)&training[m]) if training[m] is not None else None,
                'caution':'Zero observed overlap does NOT prove held-out evaluation if training sampled from same NSRDB catalog'}
             for m in available}
    report=dict(schema='pv-policy-quality-diagnostics-1',
        experimental_design=dict(days=list(days),steps_zero_based=list(steps),evaluation_seed=seed,
            stochastic_continuations_per_candidate=repeats,candidate_step_normalized=candidate_step,
            gamma=cfg.sac.algo['gamma'],reward_scale=cfg.sac.algo['reward_scale']),
        notes=[
          'Deterministic local gain measures a feasible first-action intervention followed by the same frozen deterministic actor; hindsight choice is not a deployable improvement policy.',
          'Soft-Q Monte Carlo returns use the SAME recorded NSRDB day and state and stochastic policy continuation; sampled Gaussian actions provide Monte Carlo variability, not independent weather replications.',
          'Soft return is 100*r_t + future sum of gamma-weighted (100*r_j - alpha*logpi_j); no entropy term for the conditioned first action.',
          'The critic approximates expectations over environment/experience, so disagreement on a single frozen weather trajectory alone is NOT proof of implementation error.',
          'Action alternatives are a finite local set; positive deterministic gains demonstrate suboptimality relative to that set for the chosen frozen day only.',
          'One-step real-transition Bellman residual is target-network self-consistency, not proof of optimality or absolute Q truth.',
          'Replay consistency and model-based dynamics validation are distinct from this critic/actor probe.',
          'No convergence or unseen-data generalization claim is possible from a single final checkpoint.',
        ],training_catalog_overlap=overlap,model_one_step=model_info,probes=probes)
    report['candidate_pair_differences']=paired_rows
    return report,rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sac-checkpoint',default=None)
    p.add_argument('--mbpo-checkpoint',default=None)
    p.add_argument('--days',nargs='+',default=['2024-12-23','2018-12-27'])
    p.add_argument('--steps',type=int,nargs='+',default=[35,65],help='ZERO-based action steps: [0,116]')
    p.add_argument('--repeats',type=int,default=3,help='Independent stochastic tail draws per action; minimum 1; 5+ recommended')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--candidate-step',type=float,default=0.5)
    p.add_argument('--eval-dir',default=None,help='Optional existing evaluation folder: assert baseline rollout matches saved energy')
    p.add_argument('--mbpo-training-csv',default=None)
    p.add_argument('--sac-training-csv',default=None)
    p.add_argument('--model-diagnostic',action='store_true',help='Add descriptive one-step elite-mean MBPO model RMSE on visited real actions')
    p.add_argument('--outdir',default='MBPO_workspace/evaluation/policy_quality_diagnostics')
    args=p.parse_args()
    report,rows=run_audit(sac_checkpoint=args.sac_checkpoint,mbpo_checkpoint=args.mbpo_checkpoint,
        days=args.days,steps=args.steps,repeats=args.repeats,seed=args.seed,
        eval_dir=args.eval_dir,mbpo_training_csv=args.mbpo_training_csv,
        sac_training_csv=args.sac_training_csv,candidate_step=args.candidate_step,model_diagnostic=args.model_diagnostic)
    out=Path(args.outdir);out.mkdir(parents=True,exist_ok=True)
    with (out/'candidate_scores.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    pair_rows=report.pop('candidate_pair_differences',[])
    if pair_rows:
        with (out/'paired_candidate_differences.csv').open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(pair_rows[0]));w.writeheader();w.writerows(pair_rows)
    (out/'quality_diagnostic.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print('\nRESULTS: '+str(out.resolve()))
    print('INSPECT: quality_diagnostic.json -> probes; candidate_scores.csv -> action scores; paired_candidate_differences.csv -> conditional paired Monte Carlo uncertainty')
    print('WARNING: Raw rank agreement ignores Monte Carlo uncertainty; only resolved conditional pairs support ranking statements. This is not proof of global optimality, convergence or held-out generalization.')

if __name__=='__main__':main()
