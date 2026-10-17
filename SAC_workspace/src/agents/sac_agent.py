"""Model-free Soft Actor-Critic (SAC-v2): reparameterized tanh actor, clipped
Double-Q target, automatic entropy adjustment, Polyak-averaged target critics.

Reference task: gamma=1, terminal episodes at 117 steps, learned reward scale=100.
No BNN, model pool, fake environment, or imaginary rollouts.
"""
from __future__ import annotations
import copy
import math
from typing import Sequence
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

LOG_2 = math.log(2.0)


def mlp(in_dim: int, out_dim: int, sizes: Sequence[int]) -> nn.Sequential:
    dims = (in_dim, *tuple(sizes))
    layers = []
    for a, b in zip(dims[:-1], dims[1:]):
        layers.extend([nn.Linear(a, b), nn.ReLU()])
    layers.append(nn.Linear(dims[-1], out_dim))
    return nn.Sequential(*layers)


class SquashedGaussianActor(nn.Module):
    def __init__(self, state_dim=6, action_dim=2, hidden_sizes=(64, 64)):
        super().__init__()
        self.net = mlp(state_dim, 2 * action_dim, hidden_sizes)
        self.action_dim = action_dim

    def forward(self, s: torch.Tensor, deterministic: bool = False, with_logprob: bool = True):
        mean, raw_log_std = self.net(s).chunk(2, dim=-1)
        log_std = torch.clamp(raw_log_std, -20., 2.)
        std = torch.exp(log_std)
        z = mean if deterministic else mean + std * torch.randn_like(mean)
        action = torch.tanh(z)
        if not with_logprob:
            return action, None
        # Stable log|d tanh(z)/dz| = 2(log(2) - z - softplus(-2z)).
        normal_lp = (-0.5 * ((z - mean) / std).square() - log_std - 0.5 * math.log(2. * math.pi)).sum(-1, keepdim=True)
        log_det = (2. * (LOG_2 - z - F.softplus(-2. * z))).sum(-1, keepdim=True)
        return action, normal_lp - log_det


class TwinCritic(nn.Module):
    def __init__(self, state_dim=6, action_dim=2, hidden_sizes=(64, 64)):
        super().__init__()
        self.q1 = mlp(state_dim + action_dim, 1, hidden_sizes)
        self.q2 = mlp(state_dim + action_dim, 1, hidden_sizes)

    def forward(self, s: torch.Tensor, a: torch.Tensor):
        x = torch.cat((s, a), dim=-1)
        return self.q1(x), self.q2(x)


class SACAgent:
    def __init__(self, state_dim=6, action_dim=2, hidden_sizes=(64,64),
                 lr=3e-4, gamma=1., tau=.005, reward_scale=100.,
                 alpha_init=1., target_entropy=-2., min_alpha=0.001, device='cpu'):
        if (state_dim, action_dim) != (6, 2):
            raise ValueError('Reference contract expects state_dim=6, action_dim=2')
        if not 0. < tau <= 1. or not 0. <= gamma <= 1. or reward_scale <= 0. or alpha_init <= 0. or min_alpha <= 0.:
            raise ValueError('Invalid SAC hyperparameters')
        self.device = torch.device(device)
        self.gamma, self.tau, self.reward_scale = float(gamma), float(tau), float(reward_scale)
        self.target_entropy = float(target_entropy)
        self.min_alpha = float(min_alpha)
        self.actor = SquashedGaussianActor(state_dim, action_dim, hidden_sizes).to(self.device)
        self.critic = TwinCritic(state_dim, action_dim, hidden_sizes).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).to(self.device).eval()
        for p in self.critic_target.parameters(): p.requires_grad_(False)
        self.log_alpha = torch.tensor([math.log(alpha_init)], device=self.device, requires_grad=True)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=lr)
        self.updates = 0

    @property
    def alpha(self):
        return float(self.log_alpha.detach().exp().clamp_min(self.min_alpha).item())

    @torch.no_grad()
    def act(self, state, deterministic=False):
        s = torch.as_tensor(np.asarray(state, dtype=np.float32), device=self.device).reshape(1,6)
        a, _ = self.actor(s, deterministic=deterministic, with_logprob=False)
        return np.clip(a.squeeze(0).cpu().numpy(), -1., 1.).astype(np.float32)

    def compute_target(self, batch):
        s2, r, done = batch['next_states'], batch['rewards'], batch['dones']
        with torch.no_grad():
            a2, lp2 = self.actor(s2)
            q1, q2 = self.critic_target(s2, a2)
            return self.reward_scale * r + self.gamma * (1. - done) * (torch.minimum(q1, q2) - self.log_alpha.exp().clamp_min(self.min_alpha).detach() * lp2)

    def update(self, batch: dict) -> dict:
        batch = {k: torch.as_tensor(v, dtype=torch.float32, device=self.device) for k, v in batch.items()}
        s, a = batch['states'], batch['actions']
        target = self.compute_target(batch)
        q1, q2 = self.critic(s, a)
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=10.)
        self.critic_opt.step()

        # Freeze Q parameters during the policy step (but retain dQ/da).
        for p in self.critic.parameters(): p.requires_grad_(False)
        action, logp = self.actor(s)
        qa, qb = self.critic(s, action)
        actor_loss = (self.log_alpha.detach().exp().clamp_min(self.min_alpha) * logp - torch.minimum(qa, qb)).mean()
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=10.)
        self.actor_opt.step()
        for p in self.critic.parameters(): p.requires_grad_(True)

        alpha_loss = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()
        self.alpha_opt.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_opt.step()
        with torch.no_grad():
            for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_param.lerp_(param, self.tau)
        self.updates += 1
        metrics = {'critic_loss': critic_loss.item(), 'actor_loss': actor_loss.item(),
                   'alpha_loss': alpha_loss.item(), 'alpha': self.alpha,
                   'entropy': -logp.detach().mean().item(),
                   'q_mean': torch.minimum(q1, q2).detach().mean().item(),
                   'target_mean': target.detach().mean().item()}
        if not np.isfinite(np.fromiter(metrics.values(), dtype=np.float64)).all():
            raise FloatingPointError(f'Non-finite SAC metrics: {metrics}')
        return metrics

    def state_dict(self):
        return {'actor': self.actor.state_dict(), 'critic': self.critic.state_dict(),
                'critic_target': self.critic_target.state_dict(),
                'actor_opt': self.actor_opt.state_dict(), 'critic_opt': self.critic_opt.state_dict(),
                'alpha_opt': self.alpha_opt.state_dict(), 'log_alpha': self.log_alpha.detach().clone(),
                'updates': self.updates, 'gamma': self.gamma,
                'tau': self.tau, 'reward_scale': self.reward_scale, 'target_entropy': self.target_entropy,
                'min_alpha': self.min_alpha}

    def load_state_dict(self, state):
        for k in ('gamma', 'tau', 'reward_scale', 'target_entropy', 'min_alpha'):
            if abs(float(state[k]) - getattr(self, k)) > 1e-10:
                raise ValueError(f'Checkpoint {k} differs from current SAC configuration')
        self.actor.load_state_dict(state['actor'])
        self.critic.load_state_dict(state['critic'])
        self.critic_target.load_state_dict(state['critic_target'])
        self.log_alpha.data.copy_(state['log_alpha'].to(self.device))
        self.actor_opt.load_state_dict(state['actor_opt'])
        self.critic_opt.load_state_dict(state['critic_opt'])
        self.alpha_opt.load_state_dict(state['alpha_opt'])
        self.updates = int(state['updates'])
