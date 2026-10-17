"""Standalone finite-horizon SAC trainer; never calls a learned dynamics model."""
from __future__ import annotations
import csv
import hashlib
import json
import os
import random
from pathlib import Path
import numpy as np
import torch
from SAC_workspace.src.config.config import SACConfig, write_snapshot
from SAC_workspace.src.environment.environment import SACEnvironment
from SAC_workspace.src.agents.sac_agent import SACAgent
from SAC_workspace.src.training.replay_buffer import ReplayBuffer


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def config_hash(cfg):
    return hashlib.sha256(json.dumps(cfg.to_dict(), sort_keys=True).encode()).hexdigest()


class SACTrainer:
    def __init__(self, cfg: SACConfig, outdir, seed=0, device='cpu', resume=None):
        seed_all(seed)
        self.cfg, self.seed = cfg, seed
        self.outdir = Path(outdir).resolve()
        self.outdir.mkdir(parents=True, exist_ok=True)
        self._episode_path = self.outdir / 'episodes.csv'
        snapshot_path = self.outdir / 'config.json'
        if resume is None and (self._episode_path.exists() or snapshot_path.exists()):
            raise FileExistsError(f'Run data already exists in {self.outdir}; choose a fresh run or --resume')
        if resume is not None:
            if not snapshot_path.is_file():
                raise FileNotFoundError('Resume requires the matching original config.json')
            old_snapshot = json.loads(snapshot_path.read_text(encoding='utf-8'))
            if old_snapshot != json.loads(json.dumps(cfg.to_dict())):
                raise ValueError('Run snapshot differs from current reference configuration')
        else:
            write_snapshot(cfg, snapshot_path)
        self.env = SACEnvironment(cfg.train_env)
        self.env.seed(seed)
        self._resolved_path = self.outdir / 'resolved_mdp.json'
        if resume is None:
            self._resolved_path.write_text(json.dumps(self.env.resolved_mdp, indent=2, sort_keys=True), encoding='utf-8')
        elif not self._resolved_path.is_file():
            raise FileNotFoundError('Cannot resume: resolved_mdp.json is missing')
        elif json.loads(self._resolved_path.read_text(encoding='utf-8')) != self.env.resolved_mdp:
            raise ValueError('Effective PVTrackingEnv physics changed since checkpoint run')
        self.agent = SACAgent(hidden_sizes=cfg.algo['hidden_sizes'], lr=cfg.algo['lr'],
                              gamma=cfg.algo['gamma'], tau=cfg.algo['tau'],
                              reward_scale=cfg.algo['reward_scale'], alpha_init=cfg.algo['alpha_init'],
                              target_entropy=cfg.algo['target_entropy'], min_alpha=cfg.algo['min_alpha'], device=device)
        self.buffer = ReplayBuffer(cfg.algo['replay_size'], seed=seed)
        self.epoch, self.env_steps = 0, 0
        if resume is not None:
            self.load(resume)

    def train_episode(self):
        s = self.env.reset()
        if not np.isclose(s[-1], 1.0, atol=1e-6):
            raise RuntimeError('Reset did not start at full remaining horizon')
        rewards = []
        energies = []
        costs = []
        metrics = []
        for step in range(self.env.horizon):
            if self.env_steps < self.cfg.algo['start_steps']:
                a = self.buffer.rng.uniform(-1., 1., size=2).astype(np.float32)
            else:
                a = self.agent.act(s)
            s2, r, done, info = self.env.step(a)
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
            if done: break
        if len(rewards) != self.env.horizon or not done:
            raise RuntimeError('Incomplete finite-horizon episode')
        self.epoch += 1
        result = {'epoch': self.epoch, 'env_steps': self.env_steps,
                  'updates': self.agent.updates, 'scenario_id': info.get('scenario_id'),
                  'return_kwh': float(sum(rewards)), 'gross_energy_kwh': float(sum(energies)),
                  'movement_cost_kwh': float(sum(costs)), 'alpha': self.agent.alpha}
        for k in ('critic_loss','actor_loss','alpha_loss','entropy','q_mean','target_mean'):
            result[k] = float(np.mean([x[k] for x in metrics])) if metrics else float('nan')
        with self._episode_path.open('a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=result.keys())
            if f.tell() == 0: writer.writeheader()
            writer.writerow(result)
        return result

    def save(self, name='latest.pt'):
        # Checkpoints are saved only at full episode boundaries.
        data = {'schema': 1, 'hash': config_hash(self.cfg), 'seed': self.seed,
                'epoch': self.epoch, 'env_steps': self.env_steps,
                'agent': self.agent.state_dict(), 'replay': self.buffer.state_dict(),
                'resolved_mdp': self.env.resolved_mdp,
                'torch_rng': torch.get_rng_state(),
                'torch_cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                'numpy_rng': np.random.get_state(), 'python_rng': random.getstate(),
                'environment_rng': self.env.unwrapped.np_random.get_state() if hasattr(self.env.unwrapped.np_random,'get_state') else self.env.unwrapped.np_random.bit_generator.state}
        dest = self.outdir / name
        tmp = dest.with_suffix('.tmp')
        torch.save(data, tmp)
        os.replace(tmp, dest)
        return dest

    def load(self, path):
        # This is a trusted, locally generated training checkpoint. Never load untrusted .pt files.
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        if ckpt['schema'] != 1 or ckpt['hash'] != config_hash(self.cfg) or ckpt['seed'] != self.seed:
            raise ValueError('Checkpoint schema/configuration/seed mismatch')
        if ckpt.get('resolved_mdp') != self.env.resolved_mdp:
            raise ValueError('Saved MDP simulator defaults differ from current environment')
        self.agent.load_state_dict(ckpt['agent'])
        self.buffer.load_state_dict(ckpt['replay'])
        self.epoch, self.env_steps = int(ckpt['epoch']), int(ckpt['env_steps'])
        torch.set_rng_state(ckpt['torch_rng'])
        if torch.cuda.is_available() and ckpt.get('torch_cuda_rng') is not None:
            torch.cuda.set_rng_state_all(ckpt['torch_cuda_rng'])
        np.random.set_state(ckpt['numpy_rng'])
        random.setstate(ckpt['python_rng'])
        rng = self.env.unwrapped.np_random
        if hasattr(rng, 'set_state'): rng.set_state(ckpt['environment_rng'])
        else: rng.bit_generator.state = ckpt['environment_rng']
        if not self._episode_path.exists():
            raise FileNotFoundError('Cannot resume checkpoint without episode history CSV')
        with self._episode_path.open(newline='', encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
        if len(rows) != self.epoch:
            raise ValueError('Checkpoint epoch does not match episode CSV; do not resume an inconsistent run')

    def train(self, epochs=None):
        until = self.cfg.algo['epochs'] if epochs is None else int(epochs)
        if until < self.epoch: raise ValueError('Requested final epoch precedes current epoch')
        try:
            while self.epoch < until:
                result = self.train_episode()
                print(f"episode={result['epoch']:5d} steps={result['env_steps']:7d} net={result['return_kwh']:.5f} kWh alpha={result['alpha']:.4g}", flush=True)
                if self.epoch % self.cfg.algo['checkpoint_interval'] == 0:
                    self.save()
            self.save()
        finally:
            self.env.close()
