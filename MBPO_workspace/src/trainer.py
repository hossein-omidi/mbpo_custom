# POSTPROCESSING_TRAJECTORY_UNCERTAINTY_V1
# PVWATT_DEV_REPAIR_V3
"""Modern, Ray-free MBPO runner. Reference MDP + SAME PyTorch SAC optimizer.
Real interactions are chronological; model samples are generated separately and
never inserted in the real replay or used as ground truth for ensemble fitting.
"""
from __future__ import annotations
from mbpo.audit_support import bind_run, episode_commit, recover_history, envelope_counts, sha_file
from mbpo.geometry10 import TAU
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
from SAC_workspace.src.training.experiment_protocol import (
    PeriodicEvaluator, build_scenario_protocol, persist_scenario_protocol,
    prepare_run_metadata, finish_run_metadata, canonical_json_sha256,
)


def validate_model_fit_frequency(model_train_freq, horizon):
    """Require model refits at complete episode boundaries."""
    frequency = int(model_train_freq)
    steps = int(horizon)
    if frequency <= 0 or steps <= 0 or frequency % steps != 0:
        raise ValueError(
            'model_train_freq must be a positive integer multiple of the '
            'real episode horizon'
        )
    return frequency


class MBPOTrainer:
    def __init__(self, cfg: MBPOConfig, outdir, seed=0, device='cpu', resume=None,
                 env_factory=SACEnvironment):
        seed_all(int(seed))
        self.cfg = cfg
        self.seed = int(seed)
        self.device = device
        self.outdir = Path(outdir).resolve()
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.history_path = self.outdir / 'episodes.csv'
        self.cfg_path = self.outdir / 'config.json'
        self.resolved_path = self.outdir / 'resolved_mdp.json'

        if resume is None:
            existing = [
                self.history_path, self.cfg_path, self.resolved_path,
                self.outdir / 'scenario_split.json', self.outdir / 'run_metadata.json',
                self.outdir / 'periodic_evaluation', *self.outdir.glob('*.pt'),
            ]
            if any(path.exists() for path in existing):
                raise FileExistsError(
                    'Run data already exists in {}; choose a fresh run or --resume'.format(self.outdir))
        else:
            if not self.cfg_path.is_file():
                raise FileNotFoundError('Run config.json is missing')
            saved_cfg = json.loads(self.cfg_path.read_text(encoding='utf-8'))
            requested_cfg = json.loads(json.dumps(cfg.to_dict()))
            if saved_cfg != requested_cfg:
                raise ValueError('The requested configuration differs from the original run')

        self.env = env_factory(cfg.sac.train_env)
        self.eval_env = None
        try:
            self.env.seed(seed)
            self.eval_env = env_factory(cfg.sac.eval_env)
            self.eval_env.seed(seed)
            if self.eval_env.resolved_mdp != self.env.resolved_mdp:
                raise ValueError('Training and periodic-evaluation environments have different physical MDP fingerprints')
            horizon = int(self.env.horizon)
            if horizon <= 0:
                raise ValueError('The real PV environment has an invalid episode horizon')
            validate_model_fit_frequency(cfg.model.model_train_freq, horizon)

            a = cfg.sac.algo
            self.agent = SACAgent(
                hidden_sizes=a['hidden_sizes'], lr=a['lr'], gamma=a['gamma'],
                tau=a['tau'], reward_scale=a['reward_scale'], alpha_init=a['alpha_init'],
                target_entropy=a['target_entropy'], min_alpha=a['min_alpha'], device=device,
            )
            m = cfg.model
            self.dynamics = DynamicsEnsemble(
                num_models=m.num_models, num_elites=m.num_elites,
                hidden_dim=m.hidden_dim, lr=m.model_lr, device=device, seed=seed + 100,
            )
            self.fake_env = StructuredFakeEnv(self.dynamics, self.env)
            self.real_pool = ScenarioReplayBuffer(
                capacity=a['replay_size'], seed=seed + 1,
                validation_fraction=m.model_holdout_fraction)
            self.model_pool = ReplayBuffer(
                capacity=min(m.model_pool_capacity,
                             m.model_retain_epochs * m.rollout_batch_size * m.rollout_length),
                seed=seed + 2,
            )
            self.rng = np.random.default_rng(seed + 3)
            self.scenario_protocol = build_scenario_protocol(cfg.sac.manifest, cfg.sac.protocol)
            persist_scenario_protocol(self.outdir, self.scenario_protocol, resume=resume is not None)
            self.scenario_rng = np.random.default_rng(
                int(seed) + int(cfg.sac.protocol['split_seed']) + 17)
            # Match SAC's initial exploration stream for paired-seed comparisons.
            self.exploration_rng = np.random.default_rng(
                int(seed) + int(cfg.sac.protocol['split_seed']) + 29)
            self.evaluator = PeriodicEvaluator(
                self.eval_env, self.scenario_protocol, self.outdir, algorithm='mbpo',
                seed=seed, config_sha256=config_digest(cfg), resolved_mdp=self.env.resolved_mdp)
            prepare_run_metadata(
                self.outdir, algorithm='mbpo', seed=seed,
                config_sha256=config_digest(cfg),
                shared_sac_config_sha256=canonical_json_sha256(cfg.sac.to_dict()),
                resolved_mdp=self.env.resolved_mdp, protocol=self.scenario_protocol,
                horizon=horizon, resume=resume is not None)
            self.epoch = 0
            self.env_steps = 0
            self.latest_model_metrics = {}
            self.last_model_step = 0
            bind_run(self.outdir, resume is not None, self.env.resolved_mdp)

            if resume is None:
                cfg_tmp = self.cfg_path.with_name(self.cfg_path.name + '.tmp')
                mdp_tmp = self.resolved_path.with_name(self.resolved_path.name + '.tmp')
                cfg_tmp.write_text(json.dumps(cfg.to_dict(), indent=2, sort_keys=True), encoding='utf-8')
                mdp_tmp.write_text(json.dumps(self.env.resolved_mdp, indent=2, sort_keys=True), encoding='utf-8')
                os.replace(cfg_tmp, self.cfg_path)
                os.replace(mdp_tmp, self.resolved_path)
            else:
                if (not self.resolved_path.is_file()
                        or json.loads(self.resolved_path.read_text(encoding='utf-8')) != self.env.resolved_mdp):
                    raise ValueError('The physical simulator differs from the checkpoint run')
                self.load(resume)
            if resume is None:
                self.save()  # recoverable epoch-zero boundary
        except Exception:
            try:
                self.env.close()
            except Exception:
                pass
            if self.eval_env is not None:
                try:
                    self.eval_env.close()
                except Exception:
                    pass
            raise

    def _fit_and_rollout(self):
        m = self.cfg.model
        holdout_scenarios = int(self.real_pool.active_holdout_scenario_count)
        train_scenarios = int(self.real_pool.active_train_scenario_count)
        if holdout_scenarios < m.min_holdout_scenarios or train_scenarios < m.min_holdout_scenarios:
            self.model_pool = ReplayBuffer(capacity=self.model_pool.capacity,
                                           seed=self.seed + 2 + self.dynamics.fit_count)
            self.latest_model_metrics = {
                'model_skipped': f'Insufficient active whole-day model split: train={train_scenarios}, holdout={holdout_scenarios}',
                'model_train_scenarios': train_scenarios,
                'model_holdout_scenarios': holdout_scenarios,
                'model_fit_seconds': 0.0, 'model_rollout_seconds': 0.0,
                'model_attempted': 0, 'model_accepted': 0, 'model_rejected': 0,
            }
            return 0, 0, float('nan')

        fit_started = time.perf_counter()
        stats = self.dynamics.fit(
            self.real_pool, holdout_fraction=m.model_holdout_fraction,
            max_samples=m.model_max_train_samples, epochs=m.model_epochs,
            patience=m.model_patience, batch_size=m.model_batch_size)
        fit_seconds = time.perf_counter() - fit_started
        self.latest_model_metrics = {**stats, 'model_fit_seconds': fit_seconds,
                                     'model_fit_completed': 1, 'model_skill_pass': 0,
                                     'model_holdout_scenarios': holdout_scenarios,
                                     'model_train_scenarios': train_scenarios}
        if m.require_model_skill and (
                stats['model_skill_power_vs_persistence'] <= 0
                or stats['model_skill_all_channels_vs_persistence'] <= 0):
            self.model_pool = ReplayBuffer(capacity=self.model_pool.capacity,
                                           seed=self.seed + 2 + self.dynamics.fit_count)
            self.latest_model_metrics.update({
                'model_skipped': 'Held-out prediction did not beat persistence',
                'model_rollout_seconds': 0.0, 'model_attempted': 0,
                'model_accepted': 0, 'model_rejected': 0,
            })
            self.last_model_step = self.env_steps
            return 0, 0, float('nan')

        self.latest_model_metrics['model_skill_pass'] = 1
        states, scenario_ids = self.real_pool.rollout_states(m.rollout_batch_size, with_scenarios=True)
        if not len(states):
            self.latest_model_metrics.update({
                'model_skipped': 'No nonterminal real states available for rollout',
                'model_rollout_seconds': 0.0, 'model_attempted': 0,
                'model_accepted': 0, 'model_rejected': 0,
            })
            return 0, 0, float('nan')
        rollout_started = time.perf_counter()
        with torch.no_grad():
            inputs = torch.as_tensor(states, dtype=torch.float32, device=self.agent.device)
            actions, _ = self.agent.actor(inputs, with_logprob=False)
            actions = actions.cpu().numpy().astype(np.float32)
        ns, rew, dones, uncertainty, mask = self.fake_env.step(
            states, actions, max_std=m.uncertainty_max_std, stochastic=True, scenario_ids=scenario_ids)
        accepted = 0
        for j in np.flatnonzero(mask):
            self.model_pool.add(states[j], actions[j], float(rew[j]), ns[j], bool(dones[j]))
            accepted += 1
        if accepted and not np.isfinite(uncertainty[mask]).all():
            raise FloatingPointError('Nonfinite model ensemble uncertainty')
        rollout_seconds = time.perf_counter() - rollout_started
        self.latest_model_metrics.update({
            **self.fake_env.last_feasibility,
            'model_attempted': int(len(states)), 'model_accepted': int(accepted),
            'model_rejected': int(len(states) - accepted),
            'model_rollout_seconds': float(rollout_seconds),
            'model_mean_epistemic_std_norm': float(np.mean(uncertainty[mask])) if accepted else float('nan'),
        })
        self.last_model_step = self.env_steps
        return int(accepted), int(len(states) - accepted), (
            float(np.mean(uncertainty[mask])) if accepted else float('nan'))

    def train_episode(self):
        episode_started = time.perf_counter()
        ids = self.scenario_protocol['train_ids']
        probabilities = np.asarray(self.scenario_protocol['train_probabilities'], dtype=float)
        scenario_index = int(self.scenario_rng.choice(len(ids), p=probabilities))
        scenario_id = ids[scenario_index]
        state = self.env.reset(scenario_id=scenario_id)
        if not np.isclose(state[TAU], 1., atol=1e-6):
            raise RuntimeError('Bad initial tau')
        episode_rewards, gross, cost, updates, nr, nm = [], [], [], [], [], []
        attempted = accepted = rejected = 0
        fit_seconds = rollout_seconds = 0.0
        diagnostic_counts = dict.fromkeys(('model_fit_attempted','model_fit_completed','model_skill_pass','envelope_checked','envelope_violations','envelope_irradiance_saturated','raw_weather_outside','raw_power_outside'), 0)
        real_saturated = 0; real_envelope_states = []
        scenario = None
        for step in range(self.env.horizon):
            if self.env_steps < self.cfg.sac.algo['start_steps']:
                action = self.exploration_rng.uniform(-1, 1, size=2).astype(np.float32)
            else:
                action = self.agent.act(state)
            nxt, reward, done, info = self.env.step(action)
            sid = info.get('scenario_id')
            if not sid:
                raise RuntimeError('NSRDB scenario ID was not recorded')
            if sid != scenario_id:
                raise RuntimeError('Explicit training scenario changed within an episode')
            if scenario is None:
                scenario = sid
            elif sid != scenario:
                raise RuntimeError('Weather scenario changed within an episode')
            self.real_pool.add(state, action, reward, nxt, done, sid)
            # Actual irradiance saturation is checked before decoding.
            if max(float(info['dni_wm2']), float(info['dhi_wm2'])) >= 2000.:
                real_saturated += 1
            else:
                real_envelope_states.append(nxt.copy())
            self.env_steps += 1
            if (self.env_steps > self.cfg.sac.algo['start_steps']
                    and self.env_steps % self.cfg.model.model_train_freq == 0):
                accepted_now, rejected_now, _unc = self._fit_and_rollout()
                accepted += accepted_now
                rejected += rejected_now
                attempted += int(self.latest_model_metrics.get('model_attempted', 0))
                diagnostic_counts['model_fit_attempted'] += 1
                for key in diagnostic_counts:
                    diagnostic_counts[key] += int(self.latest_model_metrics.get(key, 0))
                fit_seconds += float(self.latest_model_metrics.get('model_fit_seconds', 0.0))
                rollout_seconds += float(self.latest_model_metrics.get('model_rollout_seconds', 0.0))
            if (self.env_steps > self.cfg.sac.algo['start_steps']
                    and len(self.real_pool) >= self.cfg.sac.algo['batch_size']):
                for _ in range(self.cfg.sac.algo['updates_per_step']):
                    batch, rn, mn = mix_batches(
                        self.real_pool, self.model_pool, self.cfg.sac.algo['batch_size'],
                        self.cfg.model.real_ratio, self.rng)
                    updates.append(self.agent.update(batch))
                    nr.append(rn)
                    nm.append(mn)
            episode_rewards.append(float(reward))
            gross.append(float(info['energy_kwh']))
            cost.append(float(info['movement_cost']))
            state = nxt
            if done:
                break
        if len(episode_rewards) != self.env.horizon or not done:
            raise RuntimeError(f'Incomplete {self.env.horizon}-transition solar episode')
        e = envelope_counts(np.asarray(real_envelope_states), self.env.unwrapped) if real_envelope_states else {'envelope_checked':0,'envelope_violations':0,'envelope_irradiance_saturated':0}
        real_checked=e['envelope_checked']; real_violations=e['envelope_violations']; real_saturated+=e['envelope_irradiance_saturated']
        self.epoch += 1
        row = {
            'epoch': self.epoch, 'env_steps': self.env_steps, 'updates': self.agent.updates,
            'scenario_id': scenario,
            'weather_realization_seed': self.env.unwrapped._weather_realization_seed,
            'weather_uncertainty_enabled': bool(self.env.unwrapped.weather_uncertainty['enabled']),
            'return_kwh': float(sum(episode_rewards)),
            'gross_energy_kwh': float(sum(gross)), 'movement_cost_kwh': float(sum(cost)),
            'alpha': self.agent.alpha, 'real_replay_size': len(self.real_pool),
            'model_replay_size': len(self.model_pool), 'model_fits': self.dynamics.fit_count,
            'synthetic_generated': int(accepted), 'synthetic_attempted': int(attempted),
            'synthetic_accepted': int(accepted), 'synthetic_rejected': int(rejected),
            'synthetic_samples_used': int(sum(nm)), 'real_samples_used': int(sum(nr)),
            **diagnostic_counts, 'real_envelope_checked':real_checked,
            'real_envelope_violations':real_violations, 'real_irradiance_saturated':real_saturated,
            'model_train_seconds': float(fit_seconds), 'model_fit_seconds': float(fit_seconds),
            'model_rollout_seconds': float(rollout_seconds),
            'model_total_seconds': float(fit_seconds + rollout_seconds),
            'mean_real_batch': float(np.mean(nr)) if nr else float('nan'),
            'mean_model_batch': float(np.mean(nm)) if nm else float('nan'),
            'episode_seconds': float(time.perf_counter() - episode_started),
        }
        for name in ('critic_loss', 'actor_loss', 'alpha_loss', 'entropy', 'q_mean', 'target_mean'):
            row[name] = float(np.mean([x[name] for x in updates])) if updates else float('nan')
        for name in ('model_val_nll', 'model_val_dni_rmse_norm', 'model_val_dhi_rmse_norm',
                     'model_val_power_rmse_norm', 'model_mean_epistemic_std_norm',
                     'model_persistence_power_rmse_norm', 'model_skill_power_vs_persistence',
                     'model_skill_all_channels_vs_persistence', 'model_holdout_scenarios', 'model_train_scenarios'):
            row[name] = float(self.latest_model_metrics.get(name, float('nan')))
        with self.history_path.open('a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            if f.tell() == 0:
                writer.writeheader()
            writer.writerow(row)
        return row

    def save(self, name='latest.pt'):
        rng = self.env.unwrapped.np_random
        ck = {
            'schema': 2, 'config': self.cfg.to_dict(), 'config_hash': config_digest(self.cfg),
            'resolved_mdp': self.env.resolved_mdp, 'seed': self.seed,
            'epoch': self.epoch, 'env_steps': self.env_steps,
            'algorithm': 'mbpo',
            'episode_commit': episode_commit(self.history_path, self.epoch, self.env_steps, self.env.horizon, updates=self.agent.updates),
            'training_identity_sha256': sha_file(self.outdir / 'training_identity.json'),
            'scenario_protocol_sha256': self.scenario_protocol['protocol_sha256'],
            'scenario_rng': self.scenario_rng.bit_generator.state,
            'exploration_rng': self.exploration_rng.bit_generator.state,
            'agent': self.agent.state_dict(), 'dynamics': self.dynamics.state_dict(),
            'real_pool': self.real_pool.state_dict(), 'model_pool': self.model_pool.state_dict(),
            'latest_model_metrics': self.latest_model_metrics, 'last_model_step': self.last_model_step,
            'torch_rng': torch.get_rng_state(),
            'torch_cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            'numpy_rng': np.random.get_state(), 'python_rng': random.getstate(),
            'env_rng': rng.get_state() if hasattr(rng, 'get_state') else rng.bit_generator.state,
            'mbpo_rng': self.rng.bit_generator.state,
        }
        path = self.outdir / name
        tmp = path.with_suffix('.tmp')
        with tmp.open('wb') as f:
            torch.save(ck,f); f.flush(); os.fsync(f.fileno())
        os.replace(tmp,path)
        return path

    def load(self, path):
        ck = torch.load(path, map_location='cpu', weights_only=False)
        if (ck.get('schema') != 2 or ck.get('config_hash') != config_digest(self.cfg)
                or ck.get('seed') != self.seed):
            raise ValueError('Unrecognized MBPO checkpoint/config/seed; old checkpoints require a fresh run')
        if ck.get('resolved_mdp') != self.env.resolved_mdp:
            raise ValueError('Physical simulator parameters changed since training')
        if ck.get('scenario_protocol_sha256') != self.scenario_protocol['protocol_sha256']:
            raise ValueError('Checkpoint scenario split differs from current protocol')
        if ck.get('training_identity_sha256') != sha_file(self.outdir / 'training_identity.json'):
            raise ValueError('Checkpoint training identity mismatch')
        from mbpo.audit_release import checkpoint_identity
        checkpoint_identity(ck,'mbpo',self.seed,ck['epoch'],ck['env_steps'])
        if ck['env_steps']!=ck['epoch']*self.env.horizon: raise ValueError('Checkpoint episode/interaction mismatch')
        self.agent.load_state_dict(ck['agent'])
        self.dynamics.load_state_dict(ck['dynamics'])
        self.real_pool.load_state_dict(ck['real_pool'])
        self.model_pool.load_state_dict(ck['model_pool'])
        self.epoch = int(ck['epoch'])
        self.env_steps = int(ck['env_steps'])
        self.latest_model_metrics = dict(ck['latest_model_metrics'])
        self.last_model_step = int(ck['last_model_step'])
        self.rng.bit_generator.state = ck['mbpo_rng']
        self.scenario_rng.bit_generator.state = ck['scenario_rng']
        self.exploration_rng.bit_generator.state = ck['exploration_rng']
        torch.set_rng_state(ck['torch_rng'])
        if torch.cuda.is_available() and ck['torch_cuda_rng'] is not None:
            torch.cuda.set_rng_state_all(ck['torch_cuda_rng'])
        np.random.set_state(ck['numpy_rng'])
        random.setstate(ck['python_rng'])
        rg = self.env.unwrapped.np_random
        if hasattr(rg, 'set_state'):
            rg.set_state(ck['env_rng'])
        else:
            rg.bit_generator.state = ck['env_rng']
        recover_history(self.history_path, ck.get('episode_commit'), self.epoch,
                        self.env_steps, self.env.horizon, updates=self.agent.updates)
        self.evaluator.reconcile(self.epoch, self.env.horizon)

    def train(self, epochs=None):
        target = self.cfg.sac.algo['epochs'] if epochs is None else int(epochs)
        if target < self.epoch:
            raise ValueError('Cannot train backwards')
        started = time.perf_counter()
        completed = False
        eval_interval = int(self.cfg.sac.protocol['evaluation_interval_episodes'])
        checkpoint_interval = int(self.cfg.sac.algo['checkpoint_interval'])
        try:
            if (self.epoch > 0 and (self.epoch % eval_interval == 0 or self.epoch == target)
                    and not self.evaluator.has_evaluation(epoch=self.epoch, env_steps=self.env_steps)):
                self.evaluator.evaluate(self.agent, epoch=self.epoch,
                                        env_steps=self.env_steps, updates=self.agent.updates)
            while self.epoch < target:
                r = self.train_episode()
                print('epoch={:5d} real_steps={:7d} net={:.5f}kWh alpha={:.4g} model={} synthetic={} ratio={:.3g}'.format(
                    r['epoch'], r['env_steps'], r['return_kwh'], r['alpha'], r['model_fits'],
                    r['synthetic_accepted'],
                    r['mean_model_batch'] / self.cfg.sac.algo['batch_size']
                    if np.isfinite(r['mean_model_batch']) else 0), flush=True)
                final_epoch = self.epoch == target
                if self.epoch % checkpoint_interval == 0 or final_epoch:
                    self.save()
                if self.epoch % eval_interval == 0 or final_epoch:
                    ev = self.evaluator.evaluate(self.agent, epoch=self.epoch,
                                                 env_steps=self.env_steps, updates=self.agent.updates)
                    print('  validation: mean_net={:.6f} kWh ({} fixed days; {:.1f}s)'.format(
                        ev['mean_net_energy_kwh'], ev['n_scenarios'], ev['evaluation_seconds']), flush=True)
            self.save()
            completed = True
        finally:
            finish_run_metadata(self.outdir, final_epoch=self.epoch, env_steps=self.env_steps,
                                training_seconds=time.perf_counter() - started,
                                status='completed' if completed else 'failed')
            self.env.close()
            if self.eval_env is not None:
                self.eval_env.close()

