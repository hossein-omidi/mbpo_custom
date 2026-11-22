#!/usr/bin/env python3
"""Apply the verified periodic-evaluation/sample-complexity patch to PVwatt_dev.

The installer is intentionally conservative: it verifies Git blob SHA-1 values
for every source file before modifying it, writes backups, applies exact method
replacements, and syntax-checks all changed Python files. It never overwrites a
source file whose content differs from the inspected PVwatt_dev base.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

BASE_BLOBS = {
    "SAC_workspace/src/training/trainer.py": "44daec7e60af102a354e2ded0c3902a8eb0f5814",
    "MBPO_workspace/src/trainer.py": "22d50292c3b6977d27ab9987fd7ae5873674af2b",
    "SAC_workspace/src/config/config.py": "5af612b69aae00bcf04a655ae51099def342d03a",
    "MBPO_workspace/src/config.py": "5aaf53c4a39063c9f762200bf0450c7090283de2",
    "MBPO_workspace/src/dynamics.py": "70982c104c58f8d7819ade49a00ac4e2c69fd7e2",
    "MBPO_workspace/src/replay.py": "5c26d71c03774f3350ec824d8f8f0e6ff967e99b",
    "SAC_workspace/workflows/01_setup_config.py": "7b320ea82b6fef21f89f0d883d1916164cbe924b",
    "MBPO_workspace/workflows/01_setup_config.py": "5d79cb6d16201dac20e0eb5a41a2e0d23d26dad6",
    "SAC_workspace/src/agents/sac_agent.py": "b5f3da0d2c525546fbc264c5151ddebedf7728b3",
    "MBPO_workspace/workflows/03_evaluate.py": "922c9f3bb32851c605e6ff416bdd501534a45593",
}


def git_blob_sha(path: Path) -> str:
    data = path.read_bytes()
    header = f"blob {len(data)}\0".encode("ascii")
    return hashlib.sha1(header + data).hexdigest()


def method_pattern(name: str) -> re.Pattern:
    return re.compile(rf"(?ms)^    def {re.escape(name)}\(.*?(?=^    def |\Z)")


def replace_method(source: str, name: str, replacement: str) -> str:
    pattern = method_pattern(name)
    matches = list(pattern.finditer(source))
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one method {name}, found {len(matches)}")
    return source[:matches[0].start()] + replacement.rstrip() + "\n\n" + source[matches[0].end():]


def replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"Expected exactly one anchor for {label}; found {count}")
    return source.replace(old, new, 1)


def patch_sac_trainer(src: str) -> str:
    src = replace_once(src, "import random\nfrom pathlib import Path\n", "import random\nimport time\nfrom pathlib import Path\n", "SAC time import")
    src = replace_once(
        src,
        "from SAC_workspace.src.training.replay_buffer import ReplayBuffer\n",
        "from SAC_workspace.src.training.replay_buffer import ReplayBuffer\n"
        "from SAC_workspace.src.training.experiment_protocol import (\n"
        "    PeriodicEvaluator, build_scenario_protocol, persist_scenario_protocol,\n"
        "    prepare_run_metadata, finish_run_metadata, canonical_json_sha256,\n"
        ")\n",
        "SAC protocol imports",
    )
    init = '''    def __init__(self, cfg: SACConfig, outdir, seed=0, device='cpu', resume=None):
        seed_all(seed)
        self.cfg, self.seed = cfg, int(seed)
        self.outdir = Path(outdir).resolve()
        self.outdir.mkdir(parents=True, exist_ok=True)
        self._episode_path = self.outdir / 'episodes.csv'
        snapshot_path = self.outdir / 'config.json'
        self._resolved_path = self.outdir / 'resolved_mdp.json'
        if resume is None and any(p.exists() for p in (
                self._episode_path, snapshot_path, self._resolved_path,
                self.outdir / 'scenario_split.json', self.outdir / 'run_metadata.json',
                self.outdir / 'periodic_evaluation')):
            raise FileExistsError(f'Run data already exists in {self.outdir}; choose a fresh run or --resume')
        if resume is not None:
            if not snapshot_path.is_file():
                raise FileNotFoundError('Resume requires the matching original config.json')
            old_snapshot = json.loads(snapshot_path.read_text(encoding='utf-8'))
            if old_snapshot != json.loads(json.dumps(cfg.to_dict())):
                raise ValueError('Run snapshot differs from current reference configuration; start a fresh run')
        else:
            write_snapshot(cfg, snapshot_path)

        self.env = SACEnvironment(cfg.train_env)
        self.eval_env = None
        try:
            self.env.seed(seed)
            if resume is None:
                self._resolved_path.write_text(
                    json.dumps(self.env.resolved_mdp, indent=2, sort_keys=True), encoding='utf-8')
            elif not self._resolved_path.is_file():
                raise FileNotFoundError('Cannot resume: resolved_mdp.json is missing')
            elif json.loads(self._resolved_path.read_text(encoding='utf-8')) != self.env.resolved_mdp:
                raise ValueError('Effective PVTrackingEnv physics changed since checkpoint')

            self.agent = SACAgent(hidden_sizes=cfg.algo['hidden_sizes'], lr=cfg.algo['lr'],
                                  gamma=cfg.algo['gamma'], tau=cfg.algo['tau'],
                                  reward_scale=cfg.algo['reward_scale'], alpha_init=cfg.algo['alpha_init'],
                                  target_entropy=cfg.algo['target_entropy'], min_alpha=cfg.algo['min_alpha'],
                                  device=device)
            self.buffer = ReplayBuffer(cfg.algo['replay_size'], seed=seed)
            self.scenario_protocol = build_scenario_protocol(cfg.manifest, cfg.protocol)
            persist_scenario_protocol(self.outdir, self.scenario_protocol, resume=resume is not None)
            self.scenario_rng = np.random.default_rng(
                int(seed) + int(cfg.protocol['split_seed']) + 17)
            # Separate exploration RNG gives paired SAC/MBPO seeds identical initial random actions.
            self.exploration_rng = np.random.default_rng(
                int(seed) + int(cfg.protocol['split_seed']) + 29)
            self.eval_env = SACEnvironment(cfg.eval_env)
            self.eval_env.seed(seed)
            if self.eval_env.resolved_mdp != self.env.resolved_mdp:
                raise ValueError('Training and periodic-evaluation environments have different physical MDP fingerprints')
            self.evaluator = PeriodicEvaluator(
                self.eval_env, self.scenario_protocol, self.outdir, algorithm='sac',
                seed=seed, config_sha256=config_hash(cfg), resolved_mdp=self.env.resolved_mdp)
            prepare_run_metadata(
                self.outdir, algorithm='sac', seed=seed,
                config_sha256=config_hash(cfg),
                shared_sac_config_sha256=canonical_json_sha256(cfg.to_dict()),
                resolved_mdp=self.env.resolved_mdp, protocol=self.scenario_protocol,
                horizon=self.env.horizon, resume=resume is not None)
            self.epoch, self.env_steps = 0, 0
            if resume is not None:
                self.load(resume)
        except Exception:
            self.env.close()
            if self.eval_env is not None:
                self.eval_env.close()
            raise
'''
    src = replace_method(src, "__init__", init)
    train_episode = '''    def train_episode(self):
        started = time.perf_counter()
        ids = self.scenario_protocol['train_ids']
        probabilities = np.asarray(self.scenario_protocol['train_probabilities'], dtype=float)
        scenario_index = int(self.scenario_rng.choice(len(ids), p=probabilities))
        scenario_id = ids[scenario_index]
        s = self.env.reset(scenario_id=scenario_id)
        if not np.isclose(s[-1], 1.0, atol=1e-6):
            raise RuntimeError('Reset did not start at full remaining horizon')
        rewards, energies, costs, metrics = [], [], [], []
        done = False
        for step in range(self.env.horizon):
            if self.env_steps < self.cfg.algo['start_steps']:
                a = self.exploration_rng.uniform(-1., 1., size=2).astype(np.float32)
            else:
                a = self.agent.act(s)
            s2, r, done, info = self.env.step(a)
            if str(info.get('scenario_id')) != scenario_id:
                raise RuntimeError('Explicit training scenario changed within an episode')
            if not np.isclose(s2[-1], (self.env.horizon-step-1)/self.env.horizon, atol=1e-6):
                raise RuntimeError('Remaining-horizon observation is inconsistent with the transition')
            self.buffer.add(s, a, r, s2, done)
            self.env_steps += 1
            if self.env_steps > self.cfg.algo['start_steps'] and len(self.buffer) >= self.cfg.algo['batch_size']:
                for _ in range(self.cfg.algo['updates_per_step']):
                    metrics.append(self.agent.update(self.buffer.sample(self.cfg.algo['batch_size'])))
            rewards.append(r)
            energies.append(info['energy_kwh'])
            costs.append(info['movement_cost'])
            s = s2
            if done:
                break
        if len(rewards) != self.env.horizon or not done:
            raise RuntimeError('Incomplete finite-horizon episode')
        self.epoch += 1
        result = {
            'epoch': self.epoch, 'env_steps': self.env_steps, 'updates': self.agent.updates,
            'scenario_id': scenario_id, 'return_kwh': float(sum(rewards)),
            'gross_energy_kwh': float(sum(energies)), 'movement_cost_kwh': float(sum(costs)),
            'alpha': self.agent.alpha, 'episode_seconds': float(time.perf_counter() - started),
        }
        for k in ('critic_loss', 'actor_loss', 'alpha_loss', 'entropy', 'q_mean', 'target_mean'):
            result[k] = float(np.mean([x[k] for x in metrics])) if metrics else float('nan')
        with self._episode_path.open('a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=result.keys())
            if f.tell() == 0:
                writer.writeheader()
            writer.writerow(result)
        return result
'''
    src = replace_method(src, "train_episode", train_episode)
    save = '''    def save(self, name='latest.pt'):
        # Checkpoints are saved only at full episode boundaries.
        data = {
            'schema': 2, 'hash': config_hash(self.cfg), 'seed': self.seed,
            'epoch': self.epoch, 'env_steps': self.env_steps,
            'scenario_protocol_sha256': self.scenario_protocol['protocol_sha256'],
            'scenario_rng': self.scenario_rng.bit_generator.state,
            'exploration_rng': self.exploration_rng.bit_generator.state,
            'agent': self.agent.state_dict(), 'replay': self.buffer.state_dict(),
            'resolved_mdp': self.env.resolved_mdp,
            'torch_rng': torch.get_rng_state(),
            'torch_cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            'numpy_rng': np.random.get_state(), 'python_rng': random.getstate(),
            'environment_rng': self.env.unwrapped.np_random.get_state()
                if hasattr(self.env.unwrapped.np_random, 'get_state')
                else self.env.unwrapped.np_random.bit_generator.state,
        }
        dest = self.outdir / name
        tmp = dest.with_suffix('.tmp')
        torch.save(data, tmp)
        os.replace(tmp, dest)
        return dest
'''
    src = replace_method(src, "save", save)
    load = '''    def load(self, path):
        # Trusted, locally generated checkpoint only; never load untrusted .pt files.
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        if (ckpt.get('schema') != 2 or ckpt.get('hash') != config_hash(self.cfg)
                or ckpt.get('seed') != self.seed):
            raise ValueError('Checkpoint schema/configuration/seed mismatch; old checkpoints require a fresh run')
        if ckpt.get('resolved_mdp') != self.env.resolved_mdp:
            raise ValueError('Saved MDP simulator defaults differ from current environment')
        if ckpt.get('scenario_protocol_sha256') != self.scenario_protocol['protocol_sha256']:
            raise ValueError('Checkpoint scenario split differs from current protocol')
        self.agent.load_state_dict(ckpt['agent'])
        self.buffer.load_state_dict(ckpt['replay'])
        self.epoch, self.env_steps = int(ckpt['epoch']), int(ckpt['env_steps'])
        self.scenario_rng.bit_generator.state = ckpt['scenario_rng']
        self.exploration_rng.bit_generator.state = ckpt['exploration_rng']
        torch.set_rng_state(ckpt['torch_rng'])
        if torch.cuda.is_available() and ckpt.get('torch_cuda_rng') is not None:
            torch.cuda.set_rng_state_all(ckpt['torch_cuda_rng'])
        np.random.set_state(ckpt['numpy_rng'])
        random.setstate(ckpt['python_rng'])
        rng = self.env.unwrapped.np_random
        if hasattr(rng, 'set_state'):
            rng.set_state(ckpt['environment_rng'])
        else:
            rng.bit_generator.state = ckpt['environment_rng']
        if not self._episode_path.exists():
            raise FileNotFoundError('Cannot resume checkpoint without episode history CSV')
        with self._episode_path.open(newline='', encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
        if len(rows) != self.epoch or (rows and int(rows[-1]['env_steps']) != self.env_steps):
            raise ValueError('Checkpoint epoch/env_steps do not match episode CSV')
'''
    src = replace_method(src, "load", load)
    train = '''    def train(self, epochs=None):
        until = self.cfg.algo['epochs'] if epochs is None else int(epochs)
        if until < self.epoch:
            raise ValueError('Requested final epoch precedes current epoch')
        started = time.perf_counter()
        completed = False
        eval_interval = int(self.cfg.protocol['evaluation_interval_episodes'])
        checkpoint_interval = int(self.cfg.algo['checkpoint_interval'])
        try:
            if (self.epoch > 0 and self.epoch % eval_interval == 0
                    and not self.evaluator.has_evaluation(epoch=self.epoch, env_steps=self.env_steps)):
                self.evaluator.evaluate(self.agent, epoch=self.epoch,
                                        env_steps=self.env_steps, updates=self.agent.updates)
            while self.epoch < until:
                result = self.train_episode()
                print(f"episode={result['epoch']:5d} steps={result['env_steps']:7d} "
                      f"net={result['return_kwh']:.5f} kWh alpha={result['alpha']:.4g}", flush=True)
                final_epoch = self.epoch == until
                if self.epoch % checkpoint_interval == 0 or final_epoch:
                    self.save()
                if self.epoch % eval_interval == 0 or final_epoch:
                    ev = self.evaluator.evaluate(self.agent, epoch=self.epoch,
                                                 env_steps=self.env_steps, updates=self.agent.updates)
                    print(f"  validation: mean_net={ev['mean_net_energy_kwh']:.6f} kWh "
                          f"({ev['n_scenarios']} fixed days; {ev['evaluation_seconds']:.1f}s)", flush=True)
            self.save()
            completed = True
        finally:
            finish_run_metadata(self.outdir, final_epoch=self.epoch, env_steps=self.env_steps,
                                training_seconds=time.perf_counter() - started,
                                status='completed' if completed else 'failed')
            self.env.close()
            if self.eval_env is not None:
                self.eval_env.close()
'''
    src = replace_method(src, "train", train)
    return src


def patch_mbpo_trainer(src: str) -> str:
    src = replace_once(
        src,
        "from MBPO_workspace.src.replay import ScenarioReplayBuffer,mix_batches\n",
        "from MBPO_workspace.src.replay import ScenarioReplayBuffer,mix_batches\n"
        "from SAC_workspace.src.training.experiment_protocol import (\n"
        "    PeriodicEvaluator, build_scenario_protocol, persist_scenario_protocol,\n"
        "    prepare_run_metadata, finish_run_metadata, canonical_json_sha256,\n"
        ")\n",
        "MBPO protocol imports",
    )
    init = '''    def __init__(self, cfg: MBPOConfig, outdir, seed=0, device='cpu', resume=None,
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
'''
    src = replace_method(src, "__init__", init)
    fit = '''    def _fit_and_rollout(self):
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

        states = self.real_pool.rollout_states(m.rollout_batch_size)
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
            states, actions, max_std=m.uncertainty_max_std, stochastic=True)
        accepted = 0
        for j in np.flatnonzero(mask):
            self.model_pool.add(states[j], actions[j], float(rew[j]), ns[j], bool(dones[j]))
            accepted += 1
        if accepted and not np.isfinite(uncertainty[mask]).all():
            raise FloatingPointError('Nonfinite model ensemble uncertainty')
        rollout_seconds = time.perf_counter() - rollout_started
        self.latest_model_metrics.update({
            'model_attempted': int(len(states)), 'model_accepted': int(accepted),
            'model_rejected': int(len(states) - accepted),
            'model_rollout_seconds': float(rollout_seconds),
            'model_mean_epistemic_std_norm': float(np.mean(uncertainty[mask])) if accepted else float('nan'),
        })
        self.last_model_step = self.env_steps
        return int(accepted), int(len(states) - accepted), (
            float(np.mean(uncertainty[mask])) if accepted else float('nan'))
'''
    src = replace_method(src, "_fit_and_rollout", fit)
    episode = '''    def train_episode(self):
        episode_started = time.perf_counter()
        ids = self.scenario_protocol['train_ids']
        probabilities = np.asarray(self.scenario_protocol['train_probabilities'], dtype=float)
        scenario_index = int(self.scenario_rng.choice(len(ids), p=probabilities))
        scenario_id = ids[scenario_index]
        state = self.env.reset(scenario_id=scenario_id)
        if not np.isclose(state[5], 1., atol=1e-6):
            raise RuntimeError('Bad initial tau')
        episode_rewards, gross, cost, updates, nr, nm = [], [], [], [], [], []
        attempted = accepted = rejected = 0
        fit_seconds = rollout_seconds = 0.0
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
            self.env_steps += 1
            if (self.env_steps > self.cfg.sac.algo['start_steps']
                    and self.env_steps % self.cfg.model.model_train_freq == 0):
                accepted_now, rejected_now, _unc = self._fit_and_rollout()
                accepted += accepted_now
                rejected += rejected_now
                attempted += int(self.latest_model_metrics.get('model_attempted', 0))
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
        self.epoch += 1
        row = {
            'epoch': self.epoch, 'env_steps': self.env_steps, 'updates': self.agent.updates,
            'scenario_id': scenario, 'return_kwh': float(sum(episode_rewards)),
            'gross_energy_kwh': float(sum(gross)), 'movement_cost_kwh': float(sum(cost)),
            'alpha': self.agent.alpha, 'real_replay_size': len(self.real_pool),
            'model_replay_size': len(self.model_pool), 'model_fits': self.dynamics.fit_count,
            'synthetic_generated': int(accepted), 'synthetic_attempted': int(attempted),
            'synthetic_accepted': int(accepted), 'synthetic_rejected': int(rejected),
            'synthetic_samples_used': int(sum(nm)), 'real_samples_used': int(sum(nr)),
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
'''
    src = replace_method(src, "train_episode", episode)
    save = '''    def save(self, name='latest.pt'):
        rng = self.env.unwrapped.np_random
        ck = {
            'schema': 2, 'config': self.cfg.to_dict(), 'config_hash': config_digest(self.cfg),
            'resolved_mdp': self.env.resolved_mdp, 'seed': self.seed,
            'epoch': self.epoch, 'env_steps': self.env_steps,
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
        torch.save(ck, tmp)
        os.replace(tmp, path)
        return path
'''
    src = replace_method(src, "save", save)
    load = '''    def load(self, path):
        ck = torch.load(path, map_location='cpu', weights_only=False)
        if (ck.get('schema') != 2 or ck.get('config_hash') != config_digest(self.cfg)
                or ck.get('seed') != self.seed):
            raise ValueError('Unrecognized MBPO checkpoint/config/seed; old checkpoints require a fresh run')
        if ck.get('resolved_mdp') != self.env.resolved_mdp:
            raise ValueError('Physical simulator parameters changed since training')
        if ck.get('scenario_protocol_sha256') != self.scenario_protocol['protocol_sha256']:
            raise ValueError('Checkpoint scenario split differs from current protocol')
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
        if not self.history_path.is_file():
            raise FileNotFoundError('Cannot resume without episodes.csv')
        with self.history_path.open(newline='', encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
        if len(rows) != self.epoch or (rows and int(rows[-1]['env_steps']) != self.env_steps):
            raise ValueError('Checkpoint does not match the episode log')
'''
    src = replace_method(src, "load", load)
    train = '''    def train(self, epochs=None):
        target = self.cfg.sac.algo['epochs'] if epochs is None else int(epochs)
        if target < self.epoch:
            raise ValueError('Cannot train backwards')
        started = time.perf_counter()
        completed = False
        eval_interval = int(self.cfg.sac.protocol['evaluation_interval_episodes'])
        checkpoint_interval = int(self.cfg.sac.algo['checkpoint_interval'])
        try:
            if (self.epoch > 0 and self.epoch % eval_interval == 0
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
'''
    src = replace_method(src, "train", train)
    return src


def patch_sac_config(src: str) -> str:
    src = replace_once(
        src,
        "    config_paths: tuple[Path, ...]\n\n    def to_dict(self) -> dict:\n        return {'train_env': self.train_env, 'eval_env': self.eval_env,\n                'algo': self.algo, 'manifest': str(self.manifest),\n                'reference_configs': [str(p) for p in self.config_paths]}\n",
        "    config_paths: tuple[Path, ...]\n    protocol: dict\n\n    def to_dict(self) -> dict:\n        return {'train_env': self.train_env, 'eval_env': self.eval_env,\n                'algo': self.algo, 'protocol': self.protocol,\n                'manifest': str(self.manifest),\n                'reference_configs': [str(p) for p in self.config_paths]}\n",
        "SACConfig protocol field",
    )
    src = replace_once(src, "'eval_interval': 10, 'checkpoint_interval': 50}",
                       "'eval_interval': 50, 'checkpoint_interval': 50}", "SAC evaluation interval")
    old = """    expected_start = str(_constant(_BASE, 'EPISODE_START_TIME'))
"""
    new = """    protocol = {
        'version': 'pv_nsrdb_month_stratified_v1',
        'split_seed': 2026,
        'validation_days_per_month': 1,
        'test_days_per_month': 2,
        'preferred_validation_ids': list(_constant(_BASE, 'NSRDB_VALIDATION_SCENARIO_IDS')),
        'evaluation_interval_episodes': 50,
        'target_fraction': 0.90,
        'sustain_checkpoints': 2,
        'bootstrap_replicates': 2000,
    }
    if protocol['evaluation_interval_episodes'] != algo['eval_interval']:
        raise ValueError('Periodic evaluation and checkpoint protocol intervals disagree')
    expected_start = str(_constant(_BASE, 'EPISODE_START_TIME'))
"""
    src = replace_once(src, old, new, "SAC protocol configuration")
    src = replace_once(src,
                       "return SACConfig(nsrdb, eval_env, algo, manifest, (_BASE, _COMPACT, _SKELETON))",
                       "return SACConfig(nsrdb, eval_env, algo, manifest, (_BASE, _COMPACT, _SKELETON), protocol)",
                       "SACConfig return")
    return src


def patch_mbpo_config(src: str) -> str:
    src = replace_once(src, "import json\n", "import json\nimport math\n", "MBPO config math import")
    src = replace_once(src, "    min_holdout_scenarios:int=2\n", "    min_holdout_scenarios:int=8\n",
                       "minimum whole-day model holdout")
    src = replace_once(src,
                       "        if self.min_holdout_scenarios<2:raise ValueError('Need >=2 scenarios for day holdout')\n"
                       "        if self.uncertainty_max_std is not None and self.uncertainty_max_std<=0:\n"
                       "            raise ValueError('uncertainty_max_std must be positive')\n",
                       "        if self.min_holdout_scenarios<2:raise ValueError('Need >=2 scenarios for day holdout')\n"
                       "        if not math.isfinite(float(self.model_lr)) or self.model_lr <= 0:\n"
                       "            raise ValueError('model_lr must be finite and positive')\n"
                       "        if not math.isfinite(float(self.model_holdout_fraction)) or not 0 < self.model_holdout_fraction < .5:\n"
                       "            raise ValueError('model_holdout_fraction must be finite and in (0, 0.5)')\n"
                       "        if self.model_max_train_samples < 2 * self.model_batch_size:\n"
                       "            raise ValueError('model_max_train_samples must be at least twice model_batch_size')\n"
                       "        effective_pool_capacity = min(self.model_pool_capacity,\n"
                       "            self.model_retain_epochs * self.rollout_batch_size * self.rollout_length)\n"
                       "        required_fake_batch = 256 - int(256 * self.real_ratio)\n"
                       "        if effective_pool_capacity < required_fake_batch:\n"
                       "            raise ValueError('Synthetic replay capacity is too small to fill the configured MBPO batch fraction')\n"
                       "        if self.uncertainty_max_std is not None and (not math.isfinite(float(self.uncertainty_max_std)) or self.uncertainty_max_std<=0):\n"
                       "            raise ValueError('uncertainty_max_std must be finite and positive')\n",
                       "MBPO model configuration validation")
    return src


def patch_replay(src: str) -> str:
    src = replace_once(src,
                       "    def __init__(self, capacity, seed=0):\n        super().__init__(capacity=capacity,seed=seed)\n",
                       "    def __init__(self, capacity, seed=0, validation_fraction=0.2):\n"
                       "        super().__init__(capacity=capacity,seed=seed)\n"
                       "        self.validation_fraction=float(validation_fraction)\n"
                       "        if not 0.0 < self.validation_fraction < 0.5:\n"
                       "            raise ValueError('validation_fraction must lie in (0, 0.5)')\n",
                       "replay validation fraction constructor")
    src = replace_once(src,
                       "    def _assign_role(self, sid, fraction=0.2):\n        if sid in self._validation_roles:return\n",
                       "    def _assign_role(self, sid, fraction=None):\n        if sid in self._validation_roles:return\n"
                       "        fraction=self.validation_fraction if fraction is None else float(fraction)\n",
                       "replay role fraction")
    src = replace_once(src,
                       "        if not 0<fraction<0.5:raise ValueError('Holdout fraction must lie in (0,0.5)')\n",
                       "        if not 0<fraction<0.5:raise ValueError('Holdout fraction must lie in (0,0.5)')\n"
                       "        if not np.isclose(float(fraction),self.validation_fraction,rtol=0,atol=1e-12):\n"
                       "            raise ValueError('Dynamics holdout fraction differs from persistent replay assignment')\n",
                       "replay split fraction consistency")
    src = replace_once(src,
                       "        state['validation_contract']='sticky_day_assignment_v1'\n",
                       "        state['validation_fraction']=self.validation_fraction\n"
                       "        state['validation_contract']='sticky_day_assignment_v2'\n",
                       "replay validation contract")
    src = replace_once(src,
                       "        if data.get('validation_contract')!='sticky_day_assignment_v1':\n",
                       "        if data.get('validation_contract')!='sticky_day_assignment_v2':\n",
                       "replay validation contract version")
    src = replace_once(src,
                       "        self.validation_seed=int(data['validation_seed'])\n",
                       "        self.validation_seed=int(data['validation_seed'])\n"
                       "        self.validation_fraction=float(data['validation_fraction'])\n"
                       "        if not 0.0 < self.validation_fraction < 0.5:\n"
                       "            raise ValueError('Corrupt saved validation fraction')\n",
                       "replay validation fraction restore")
    marker = "    def scenario_split(self,fraction,seed):\n"
    prop = """    @property
    def active_holdout_scenario_count(self):
        if self.count == 0:
            return 0
        active = np.unique(self.scenario_ids[:self.count])
        return sum(self._validation_roles.get(self.scenario_names[int(i)]) == 'holdout'
                   for i in active)

    @property
    def active_train_scenario_count(self):
        if self.count == 0:
            return 0
        active = np.unique(self.scenario_ids[:self.count])
        return sum(self._validation_roles.get(self.scenario_names[int(i)]) == 'train'
                   for i in active)

"""
    src = replace_once(src, marker, prop + marker, "active holdout scenario count")
    return src


def patch_dynamics(src: str) -> str:
    start = src.index("    @torch.no_grad()\n    def predict_delta(")
    end = src.index("\n    def state_dict(self):", start)
    replacement = '''    @torch.no_grad()
    def predict_distribution(self, states, actions):
        """Return ensemble-mixture moments in the original normalized-delta units.

        The returned order is mean, aleatoric SD, epistemic SD, total SD, and
        normalized epistemic disagreement score. The Gaussian-mixture moments
        are diagnostics; they do not assert calibrated uncertainty.
        """
        if not self.fitted:
            raise RuntimeError('Dynamics ensemble not yet fitted')
        states = np.asarray(states, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.float32)
        if states.ndim != 2 or states.shape[1] != 6 or actions.shape != (len(states), 2):
            raise ValueError('Dynamics expected batch of compact state/action pairs')
        if not np.isfinite(states).all() or not np.isfinite(actions).all():
            raise ValueError('Nonfinite state/action supplied to dynamics')
        x = np.concatenate([states, actions], -1)
        tx = torch.from_numpy(self._standardize(x)).to(self.device)
        means, stds = [], []
        for k in self.elites:
            mu, log_std = self.nets[k](tx)
            means.append(mu.cpu().numpy() * self.ystd + self.ymean)
            stds.append(np.exp(log_std.cpu().numpy()) * self.ystd)
        means = np.stack(means, axis=0)
        stds = np.stack(stds, axis=0)
        mean = means.mean(axis=0)
        epistemic = means.std(axis=0)
        aleatoric = np.sqrt(np.mean(np.square(stds), axis=0))
        total = np.sqrt(np.square(epistemic) + np.square(aleatoric))
        score = (epistemic / np.maximum(self.ystd[None, :], 1e-8)).mean(axis=1)
        if not all(np.isfinite(x).all() for x in (mean, aleatoric, epistemic, total, score)):
            raise FloatingPointError('Non-finite ensemble predictive distribution')
        return tuple(x.astype(np.float32) for x in (mean, aleatoric, epistemic, total, score))

    @torch.no_grad()
    def predict_delta(self, states, actions, stochastic=True):
        """Sample a probabilistic elite prediction or return the ensemble mean.

        stochastic=True preserves MBPO's sampled-model rollout. stochastic=False
        is deterministic and is used for out-of-sample model diagnostics.
        """
        if not self.fitted:
            raise RuntimeError('Dynamics ensemble not yet fitted')
        states = np.asarray(states, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.float32)
        if states.ndim != 2 or states.shape[1] != 6 or actions.shape != (len(states), 2):
            raise ValueError('Dynamics expected batch of compact state/action pairs')
        x = np.concatenate([states, actions], -1)
        tx = torch.from_numpy(self._standardize(x)).to(self.device)
        means, stds = [], []
        for k in self.elites:
            mu, log_std = self.nets[k](tx)
            means.append(mu.cpu().numpy() * self.ystd + self.ymean)
            stds.append(np.exp(log_std.cpu().numpy()) * self.ystd)
        means = np.stack(means, axis=0)
        stds = np.stack(stds, axis=0)
        if stochastic:
            idx = self.rng.integers(len(self.elites), size=len(states))
            rows = np.arange(len(states))
            chosen = means[idx, rows] + stds[idx, rows] * self.rng.normal(size=(len(states), 3))
        else:
            chosen = means.mean(axis=0)
        epistemic_std = means.std(axis=0)
        uncertainty = (epistemic_std / np.maximum(self.ystd[None, :], 1e-8)).mean(axis=1)
        if not np.isfinite(chosen).all() or not np.isfinite(uncertainty).all():
            raise FloatingPointError('Non-finite probabilistic dynamics prediction')
        return chosen.astype(np.float32), uncertainty.astype(np.float32)
'''
    src = src[:start] + replacement + src[end:]
    src = replace_once(src,
                       "    def step(self, states, actions, max_std=None):\n",
                       "    def step(self, states, actions, max_std=None, stochastic=True):\n",
                       "structured fake env stochastic argument")
    src = replace_once(src,
                       "        delta,uncertainty=self.ensemble.predict_delta(states,actions)\n",
                       "        delta,uncertainty=self.ensemble.predict_delta(states,actions,stochastic=stochastic)\n",
                       "structured fake env deterministic prediction")
    return src


def patch_setup_workflows(src: str, algorithm: str) -> str:
    if algorithm == "sac":
        src = replace_once(src,
                           "from SAC_workspace.src.config.config import ROOT, load_config, write_snapshot\n",
                           "from SAC_workspace.src.config.config import ROOT, load_config, write_snapshot\n"
                           "from SAC_workspace.src.training.experiment_protocol import build_scenario_protocol\n",
                           "SAC setup protocol import")
        src = replace_once(src,
                           "    print(f'117 steps | gamma={c.algo[\"gamma\"]} | reward_scale={c.algo[\"reward_scale\"]}')\n",
                           "    horizon = int(c.train_env['periods']) - 1\n"
                           "    print(f'{horizon} transitions | eval every {c.protocol[\"evaluation_interval_episodes\"]} episodes | gamma={c.algo[\"gamma\"]} | reward_scale={c.algo[\"reward_scale\"]}')\n"
                           "    protocol = build_scenario_protocol(c.manifest, c.protocol)\n"
                           "    print(f'Frozen scenario split: train={len(protocol[\"train_ids\"])}, validation={len(protocol[\"validation_ids\"])}, test={len(protocol[\"test_ids\"])}, sha256={protocol[\"protocol_sha256\"][:12]}')\n",
                           "SAC setup stale horizon and split audit")
    else:
        src = replace_once(src,
                           "from SAC_workspace.src.config.config import ROOT\n",
                           "from SAC_workspace.src.config.config import ROOT\n"
                           "from SAC_workspace.src.training.experiment_protocol import build_scenario_protocol\n",
                           "MBPO setup protocol import")
        src = replace_once(src,
                           "    print('Verified shared compact reference MDP: 6D, 2D, 117 steps, UTC native NSRDB.')\n",
                           "    horizon = int(c.sac.train_env['periods']) - 1\n"
                           "    print(f'Verified shared compact reference MDP: 6D, 2D, {horizon} transitions, UTC native NSRDB.')\n"
                           "    protocol = build_scenario_protocol(c.sac.manifest, c.sac.protocol)\n"
                           "    print(f'Frozen scenario split: train={len(protocol[\"train_ids\"])}, validation={len(protocol[\"validation_ids\"])}, test={len(protocol[\"test_ids\"])}, sha256={protocol[\"protocol_sha256\"][:12]}')\n"
                           "    print(f'MBPO settings: real_ratio={c.model.real_ratio}, rollout={c.model.rollout_length}, fit_freq={c.model.model_train_freq}, holdout_fraction={c.model.model_holdout_fraction}, min_train_days={c.model.min_holdout_scenarios}, min_holdout_days={c.model.min_holdout_scenarios}')\n",
                           "MBPO setup stale horizon and split audit")
    return src


def patch_final_evaluator(src: str) -> str:
    src = replace_once(
        src,
        "    p.add_argument('--outdir',default='MBPO_workspace/evaluation/stage_a_paired')\n",
        "    p.add_argument('--outdir',default='MBPO_workspace/evaluation/stage_a_paired')\n"
        "    p.add_argument('--scenario-split-json',default=None,\n"
        "                   help='Use the frozen train/validation/test partition from a training run')\n"
        "    p.add_argument('--split-role',choices=('validation','test'),default='test')\n",
        "final evaluator split arguments",
    )
    src = replace_once(
        src,
        "    p.add_argument('--wind-powerlaw-exponent',type=float,default=0.14)\n",
        "    p.add_argument('--wind-powerlaw-exponent',type=float,default=0.14)\n"
        "    p.add_argument('--model-epochs',type=int,default=None)\n"
        "    p.add_argument('--model-max-train-samples',type=int,default=None)\n",
        "final evaluator model override arguments",
    )
    src = replace_once(
        src,
        "    if args.exclude_training_days and not (args.mbpo_training_csv and args.sac_training_csv):\n",
        "    if args.exclude_training_days and not (args.mbpo_training_csv and args.sac_training_csv) and not args.scenario_split_json:\n",
        "final evaluator split exclusion guard",
    )
    src = replace_once(
        src,
        "    cfg=load_config(power_model=args.power_model,\n"
        "        wind_reference_height_m=args.wind_reference_height_m,\n"
        "        gamma_pdc=args.gamma_pdc,sapm_mount=args.sapm_mount,\n"
        "        wind_powerlaw_exponent=args.wind_powerlaw_exponent)\n",
        "    model_overrides={}\n"
        "    if args.model_epochs is not None:model_overrides['model_epochs']=args.model_epochs\n"
        "    if args.model_max_train_samples is not None:model_overrides['model_max_train_samples']=args.model_max_train_samples\n"
        "    cfg=load_config(model_overrides=model_overrides,power_model=args.power_model,\n"
        "        wind_reference_height_m=args.wind_reference_height_m,\n"
        "        gamma_pdc=args.gamma_pdc,sapm_mount=args.sapm_mount,\n"
        "        wind_powerlaw_exponent=args.wind_powerlaw_exponent)\n",
        "final evaluator model override configuration",
    )
    old = """    from mbpo.env.nsrdb_weather import load_scenario_manifest
    from SAC_workspace.src.evaluation.scenario_sampling import resolve_cli_selection,save_selection
    scenarios,plan=resolve_cli_selection(
        args,load_scenario_manifest(cfg.sac.manifest),
        training_csvs=[x for x in (args.mbpo_training_csv,args.sac_training_csv) if x])
    if plan is not None:
        save_selection(plan,args.outdir)
        print('Selected',plan['selected_count'],'of',plan['population_count'],
              'days, protocol=',plan['protocol'])
"""
    new = """    from mbpo.env.nsrdb_weather import load_scenario_manifest
    from SAC_workspace.src.evaluation.scenario_sampling import resolve_cli_selection,save_selection
    manifest = load_scenario_manifest(cfg.sac.manifest)
    split_protocol_sha256 = None
    split_selection = None
    if args.scenario_split_json:
        from SAC_workspace.src.training.experiment_protocol import build_scenario_protocol
        saved_split = json.loads(Path(args.scenario_split_json).read_text(encoding='utf-8'))
        expected_split = build_scenario_protocol(cfg.sac.manifest, cfg.sac.protocol)
        if saved_split != expected_split:
            raise ValueError('Scenario split does not match the current NSRDB manifest/configuration')
        role_ids = list(saved_split[f\"{args.split_role}_ids\"])
        by_id = {str(row['scenario_id']): row for row in manifest['scenarios']}
        missing = sorted(set(role_ids) - set(by_id))
        if missing:
            raise ValueError(f'Scenario split references IDs missing from manifest: {missing}')
        scenarios = [by_id[sid] for sid in role_ids]
        if len(scenarios) != len(role_ids) or not scenarios:
            raise ValueError('Frozen scenario split selection is empty or contains duplicates')
        plan = None
        split_protocol_sha256 = saved_split['protocol_sha256']
        split_selection = {
            'selection_protocol': 'frozen_scenario_split_v1',
            'role': args.split_role,
            'scenario_protocol_sha256': split_protocol_sha256,
            'selected_count': len(scenarios),
            'selected_scenario_ids': role_ids,
        }
        out_path = Path(args.outdir)
        out_path.mkdir(parents=True, exist_ok=True)
        (out_path / 'scenario_split_selection.json').write_text(
            json.dumps(split_selection, indent=2, sort_keys=True), encoding='utf-8')
        print('Selected frozen', args.split_role, 'partition:', len(scenarios), 'days')
    else:
        scenarios,plan=resolve_cli_selection(
            args,manifest,
            training_csvs=[x for x in (args.mbpo_training_csv,args.sac_training_csv) if x])
        if plan is not None:
            save_selection(plan,args.outdir)
            print('Selected',plan['selected_count'],'of',plan['population_count'],
                  'days, protocol=',plan['protocol'])
"""
    src = replace_once(src, old, new, "final evaluator frozen scenario selection")
    src = replace_once(
        src,
        "    report['independent_training_day_exclusion']=bool(args.exclude_training_days)\n",
        "    report['independent_training_day_exclusion']=bool(args.exclude_training_days or (args.scenario_split_json and args.split_role=='test'))\n"
        "    if split_selection is not None:\n"
        "        report['scenario_split_role']=args.split_role\n"
        "        report['scenario_protocol_sha256']=split_protocol_sha256\n"
        "        report['selected_scenario_ids']=split_selection['selected_scenario_ids']\n",
        "final evaluator report split metadata",
    )
    return src


def patch_agent_doc(src: str) -> str:
    return replace_once(src,
                        "Reference task: gamma=1, terminal episodes at 117 steps, learned reward scale=100.",
                        "Reference task: gamma=1, finite horizon validated from the shared environment config, learned reward scale=100.",
                        "SAC agent stale horizon docstring")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="Repository root containing PVwatt_dev workspaces")
    parser.add_argument("--dry-run", action="store_true", help="Verify anchors and syntax without writing")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    bundle = Path(__file__).resolve().parent / "new_files"
    if not (root / "SAC_workspace/src/training/trainer.py").is_file():
        parser.error(f"{root} does not look like the mbpo_custom repository root")

    source_map = {
        "SAC_workspace/src/training/trainer.py": patch_sac_trainer,
        "MBPO_workspace/src/trainer.py": patch_mbpo_trainer,
        "SAC_workspace/src/config/config.py": patch_sac_config,
        "MBPO_workspace/src/config.py": patch_mbpo_config,
        "MBPO_workspace/src/replay.py": patch_replay,
        "MBPO_workspace/src/dynamics.py": patch_dynamics,
        "SAC_workspace/workflows/01_setup_config.py": lambda s: patch_setup_workflows(s, "sac"),
        "MBPO_workspace/workflows/01_setup_config.py": lambda s: patch_setup_workflows(s, "mbpo"),
        "SAC_workspace/src/agents/sac_agent.py": patch_agent_doc,
        "MBPO_workspace/workflows/03_evaluate.py": patch_final_evaluator,
    }
    staged = {}
    for relative, transform in source_map.items():
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        original = path.read_bytes()
        current_sha = git_blob_sha(path)
        if current_sha != BASE_BLOBS[relative]:
            text = path.read_text(encoding="utf-8")
            if "VERIFIED_EXPERIMENT_PATCH_V1" in text:
                print("Already patched:", relative)
                continue
            raise RuntimeError(
                f"Base file mismatch for {relative}: expected Git blob {BASE_BLOBS[relative]}, "
                f"found {current_sha}. No files have been changed. Use the exact inspected PVwatt_dev source "
                "or reconcile local edits manually.")
        transformed = transform(original.decode("utf-8"))
        transformed = "# VERIFIED_EXPERIMENT_PATCH_V1\n" + transformed
        staged[relative] = transformed

    new_files = sorted(p for p in bundle.rglob("*")
                        if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc")
    for source in new_files:
        relative = source.relative_to(bundle).as_posix()
        dest = root / relative
        if dest.exists():
            if dest.read_bytes() != source.read_bytes():
                raise FileExistsError(f"Refusing to overwrite existing non-identical new file: {dest}")
            print("Already present:", relative)
        else:
            staged[relative] = source.read_text(encoding="utf-8")

    # Syntax-check staged Python before any write.
    import ast
    for relative, content in staged.items():
        if relative.endswith(".py"):
            ast.parse(content, filename=relative)

    if args.dry_run:
        print(f"DRY RUN OK: {len(staged)} files staged; no files written.")
        return

    backups = root / "_verified_experiment_patch_backups"
    for relative in source_map:
        if relative not in staged:
            continue
        path = root / relative
        backups_path = backups / relative
        backups_path.parent.mkdir(parents=True, exist_ok=True)
        if not backups_path.exists():
            shutil.copy2(path, backups_path)
    for relative, content in staged.items():
        dest = root / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".verified_patch.tmp")
        tmp.write_text(content, encoding="utf-8")
        tmp.replace(dest)

    manifest = {
        "patch_version": 1,
        "base_branch": "PVwatt_dev",
        "base_blob_shas": BASE_BLOBS,
        "modified_and_added_files": sorted(staged),
        "backup_directory": "_verified_experiment_patch_backups",
        "note": "Existing SAC/MBPO learning architectures are retained; checkpoints from schema 1 are intentionally incompatible.",
    }
    (root / "VERIFIED_EXPERIMENT_PATCH_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"PATCH APPLIED: {len(staged)} files written.")
    print("Backups:", backups)
    print("Next: run the validation commands in README_INSTALL.md before starting new multi-seed runs.")


if __name__ == "__main__":
    main()
