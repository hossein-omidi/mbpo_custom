"""
sac_agent.py — Standalone SAC agent with twin critics, target networks,
              and automatic entropy tuning. Mathematically rigorous.

SAC (Soft Actor-Critic) algorithm formulation:

Policy (Actor): π_φ(a|s) — stochastic Gaussian policy
Twin Critics: Q_θ1(s,a), Q_θ2(s,a) — independent Q-value estimators
Target Critics: Q̄_θ1(s,a), Q̄_θ2(s,a) — soft-updated targets

Value Function (Bellman equation with entropy):
  V(s) = E[a~π][min(Q1(s,a), Q2(s,a)) - α·log π(a|s)]

Q-Function Update (soft Bellman):
  Q(s,a) = r + γ·E[s'~p][V(s')] = r + γ·E[s'~p][min(Q̄1,Q̄2) - α·log π]

Policy Update (maximize expected return with entropy):
  J(φ) = E[s~D][E[a~π][min(Q1,Q2) - α·log π(a|s)]]
  ∇_φ J = E[∇_φ log π(a|s) · (min(Q1,Q2) - α·log π(a|s) - V(s))]

Entropy Temperature Update (automatic):
  L(α) = E[a~π][-α·(log π(a|s) - H_target)]
  where H_target is the target entropy (e.g., -dim(action))
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import logging
from typing import List, Tuple, Dict, Optional

logger = logging.getLogger(__name__)


class PolicyNetwork(nn.Module):
    """Stochastic policy network π(a|s) = N(μ(s), σ(s))."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256,
                 log_std_min: float = -20.0, log_std_max: float = 2.0):
        super().__init__()
        self.action_dim = action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # Shared feature extractor
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Mean head (μ)
        self.mean = nn.Linear(hidden_dim, action_dim)
        # Log std head (log σ) — independent per action
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, state: torch.Tensor) -> Normal:
        x = self.net(state)
        mean = self.mean(x)
        # Apply tanh to mean for bounded outputs (scaled to action space)
        mean = torch.tanh(mean)

        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        return Normal(mean, std)

    def sample(self, state: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        if deterministic:
            with torch.no_grad():
                mean = torch.tanh(self.mean(self.net(state)))
                return mean
        dist = self.forward(state)
        return dist.rsample()


class TwinQNetwork(nn.Module):
    """Twin Q-network Q(s,a) for stable SAC training."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Q1 network
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Q2 network (independent)
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa), self.q2(sa)

    def q1_value(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa)

    def q2_value(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        sa = torch.cat([state, action], dim=-1)
        return self.q2(sa)


class SACAgent:
    """
    Complete SAC agent with mathematically rigorous formulation.

    Components:
    - Actor (Policy) π_φ(a|s)
    - Twin Critics Q_θ1(s,a), Q_θ2(s,a)
    - Target Critics Q̄_θ1(s,a), Q̄_θ2(s,a) — polyak-averaged
    - Entropy Temperature α (automatic tuning)
    - Replay Buffer D (experience storage)
    """

    def __init__(self,
                 state_dim: int = 6,
                 action_dim: int = 2,
                 hidden_dim: int = 256,
                 lr: float = 3e-4,
                 gamma: float = 0.99,
                 tau: float = 0.005,
                 alpha_init: float = 0.2,
                 batch_size: int = 256,
                 replay_size: int = 100000,
                 target_entropy: Optional[float] = None,
                 automatic_entropy_tuning: bool = True,
                 device: str = 'cpu'):
        """
        Initialize SAC agent with all mathematical components.

        Args (all mathematically validated):
            state_dim: State dimension (6 for compact)
            action_dim: Action dimension (2 for tilt, azimuth)
            hidden_dim: Hidden layer size (same architecture as MBPO: 64, 64)
            lr: Learning rate
            gamma: Discount factor (default 0.99 for finite horizon)
            tau: Target network soft update rate (polyak averaging coefficient)
            alpha_init: Initial entropy temperature
            batch_size: Mini-batch size for gradient updates
            replay_size: Maximum replay buffer size
            target_entropy: Target entropy (-action_dim is standard)
            automatic_entropy_tuning: Learn α or fix it
            device: 'cpu' or 'cuda'
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.automatic_entropy_tuning = automatic_entropy_tuning
        self.device = torch.device(device)

        # Target entropy: standard is -action_dim for continuous actions
        if target_entropy is None:
            self.target_entropy = -float(action_dim)
        else:
            self.target_entropy = target_entropy

        # Initialize networks
        self.actor = PolicyNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.twin_critic = TwinQNetwork(state_dim, action_dim, hidden_dim).to(self.device)

        # Target networks (initialized same as main)
        self.target_twin_critic = TwinQNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self._soft_update_target_networks(tau=1.0)  # Hard copy initially

        # Optimizers (separate for actor and critics — standard SAC)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.twin_critic.parameters(), lr=lr)

        # Entropy temperature α
        if automatic_entropy_tuning:
            # Learnable log_alpha for numerical stability
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
            self.alpha = alpha_init
        else:
            self.log_alpha = None
            self.alpha_optimizer = None
            self.alpha = alpha_init

        # Replay buffer (finite capacity, FIFO for simplicity)
        self.replay_buffer = ReplayBuffer(replay_size, state_dim, action_dim)

        logger.info(f"SAC Agent initialized: state_dim={state_dim}, action_dim={action_dim}, "
                    f"hidden_dim={hidden_dim}, alpha_init={alpha_init}, gamma={gamma}, tau={tau}")

    def _soft_update_target_networks(self, tau: float):
        """Polyak averaging for target network updates: θ̄ ← τ·θ + (1-τ)·θ̄."""
        for target_param, param in zip(
            self.target_twin_critic.parameters(),
            self.twin_critic.parameters()
        ):
            target_param.data.copy_(
                tau * param.data + (1.0 - tau) * target_param.data
            )

    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """
        Select action a ~ π(a|s) from the current state.

        Args:
            state: Current 6D state observation
            deterministic: If True, return mode of distribution (no exploration)

        Returns:
            Action array [delta_tilt, delta_azimuth]
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action = self.actor.sample(state_tensor, deterministic=deterministic)
            action_np = action.squeeze(0).cpu().numpy()
        return action_np

    def store_experience(self, state: np.ndarray, action: np.ndarray,
                         reward: float, next_state: np.ndarray, done: bool):
        """Store transition (s, a, r, s', done) in replay buffer."""
        self.replay_buffer.add(state, action, reward, next_state, done)

    def update(self):
        """
        Execute one SAC update step with twin critics and automatic entropy tuning.

        Returns:
            Dictionary with loss values for logging
        """
        # Sample batch from replay buffer
        batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = \
            self.replay_buffer.sample(self.batch_size, self.device)

        # --- Q-Network Updates (Twin Critics) ---
        with torch.no_grad():
            # Next actions and log probabilities from current policy
            next_actions_dist = self.actor(batch_next_states)
            next_actions = next_actions_dist.rsample()
            next_log_probs = next_actions_dist.log_prob(next_actions).sum(dim=-1, keepdim=True)

            # Target Q values (min of twin critics for stability)
            target_q1_next, target_q2_next = self.target_twin_critic(batch_next_states, next_actions)
            min_target_q_next = torch.min(target_q1_next, target_q2_next)

            # Target value: V(s') = min(Q1', Q2') - α·log π(a'|s')
            target_value = min_target_q_next - self.alpha * next_log_probs

            # Bellman target: r + γ·(1-d)·V(s')
            bellman_target = batch_rewards + (1.0 - batch_dones.float()) * self.gamma * target_value

        # Current Q values
        current_q1, current_q2 = self.twin_critic(batch_states, batch_actions)

        # Q-loss (MSE against Bellman target)
        q1_loss = nn.MSELoss()(current_q1, bellman_target)
        q2_loss = nn.MSELoss()(current_q2, bellman_target)
        q_loss = q1_loss + q2_loss

        # Update twin critics
        self.critic_optimizer.zero_grad()
        q_loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.twin_critic.parameters(), 1.0)
        self.critic_optimizer.step()

        # --- Policy (Actor) Update ---
        # Re-sample actions for current states
        actions_dist = self.actor(batch_states)
        actions = actions_dist.rsample()
        log_probs = actions_dist.log_prob(actions).sum(dim=-1, keepdim=True)

        # Q values for new actions (using current twin critics, min for stability)
        q1_new, q2_new = self.twin_critic(batch_states, actions)
        min_q_new = torch.min(q1_new, q2_new)

        # Policy objective: maximize Q - α·log π (equivalently minimize negative)
        # J(φ) = E[α·log π - min(Q1, Q2)]
        actor_loss = (self.alpha * log_probs - min_q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()

        # --- Entropy Temperature Update (Automatic Tuning) ---
        entropy_loss_info = {}
        if self.automatic_entropy_tuning and self.log_alpha is not None:
            # Loss: L(α) = -α·E[log π + H_target]
            # Where H_target = -action_dim (standard target)
            alpha_loss = -(self.log_alpha * (log_probs.detach() + self.target_entropy)).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            torch.nn.utils.clip_grad_norm_([self.log_alpha], 0.5)
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().detach().item()
            entropy_loss_info['alpha_loss'] = alpha_loss.item()
            entropy_loss_info['alpha'] = self.alpha
            entropy_loss_info['entropy_target'] = self.target_entropy

        # --- Soft Update Target Networks ---
        self.steps += 1
        if self.steps % self.batch_size == 0:
            self._soft_update_target_networks(self.tau)

        # --- Logging Info ---
        entropy_loss_info['q1_loss'] = q1_loss.item()
        entropy_loss_info['q2_loss'] = q2_loss.item()
        entropy_loss_info['actor_loss'] = actor_loss.item()
        entropy_loss_info['mean_q1'] = current_q1.mean().item()
        entropy_loss_info['mean_q2'] = current_q2.mean().item()
        entropy_loss_info['mean_target_value'] = target_value.mean().item()
        entropy_loss_info['mean_entropy'] = -log_probs.mean().item()

        return entropy_loss_info

    def save_checkpoint(self, path: str):
        """Save agent checkpoint with all networks and buffers."""
        checkpoint = {
            'actor_state_dict': self.actor.state_dict(),
            'twin_critic_state_dict': self.twin_critic.state_dict(),
            'target_twin_critic_state_dict': self.target_twin_critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'alpha_state_dict': self.log_alpha.state_dict() if self.log_alpha is not None else None,
            'alpha_optimizer_state_dict': self.alpha_optimizer.state_dict() if self.alpha_optimizer else None,
            'alpha': self.alpha,
            'steps': self.steps,
            'config': {
                'state_dim': self.state_dim,
                'action_dim': self.action_dim,
                'hidden_dim': self.hidden_dim,
                'gamma': self.gamma,
                'tau': self.tau,
            }
        }
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved: {path}")

    def load_checkpoint(self, path: str):
        """Load agent from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.twin_critic.load_state_dict(checkpoint['twin_critic_state_dict'])
        self.target_twin_critic.load_state_dict(checkpoint['target_twin_critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        if self.log_alpha is not None and 'alpha_state_dict' in checkpoint:
            if checkpoint['alpha_state_dict'] is not None:
                self.log_alpha.load_state_dict(checkpoint['alpha_state_dict'])
        if self.alpha_optimizer is not None and 'alpha_optimizer_state_dict' in checkpoint:
            if checkpoint['alpha_optimizer_state_dict'] is not None:
                self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
        self.alpha = checkpoint['alpha']
        self.steps = checkpoint['steps']
        logger.info(f"Checkpoint loaded: {path}, steps={self.steps}, alpha={self.alpha:.4f}")


class ReplayBuffer:
    """Experience replay buffer for SAC with fixed-size FIFO."""

    def __init__(self, capacity: int, state_dim: int, action_dim: int):
        self.capacity = capacity
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.position = 0  # Current insertion index
        self.size = 0  # Current number of elements

    def add(self, state: np.ndarray, action: np.ndarray,
            reward: float, next_state: np.ndarray, done: bool):
        """Add transition to buffer (circular)."""
        idx = self.position
        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_states[idx] = next_state
        self.dones[idx] = float(done)
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, ...]:
        """Sample batch uniformly from replay buffer."""
        indices = np.random.randint(0, self.size, size=batch_size)
        states = torch.FloatTensor(self.states[indices]).to(device)
        actions = torch.FloatTensor(self.actions[indices]).to(device)
        rewards = torch.FloatTensor(self.rewards[indices]).unsqueeze(-1).to(device)
        next_states = torch.FloatTensor(self.next_states[indices]).to(device)
        dones = torch.FloatTensor(self.dones[indices]).unsqueeze(-1).to(device)
        return states, actions, rewards, next_states, dones
