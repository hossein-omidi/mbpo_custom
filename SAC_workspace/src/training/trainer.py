"""
trainer.py — SAC training loop with replay buffer, batch sampling,
             and finite-horizon episode tracking. Model-free (no rollouts).
"""

import numpy as np
import torch
import logging
import time
from typing import Dict, List, Tuple, Optional
from pathlib import Path

# Import from our workspace (not from MBPO algorithms)
from SAC_workspace.src.agents.sac_agent import SACAgent, ReplayBuffer
from SAC_workspace.src.environment.environment import SACEnvironment

logger = logging.getLogger(__name__)


class SACTrainer:
    """
    SAC trainer with exact same MDP settings as MBPO but pure model-free SAC.
    No model rollouts, no surrogate dynamics.
    """

    def __init__(self,
                 env_kwargs: Dict = None,
                 agent_kwargs: Dict = None,
                 training_config: Dict = None,
                 output_dir: str = 'SAC_workspace/runs'):
        # Default MDP settings matching conf_sac/conf_compact
        self.env_kwargs = env_kwargs or {
            'observation_mode': 'compact',
            'latitude': 40.72,
            'longitude': -74.01,
            'altitude': 12.0,
            'tz': 'UTC',
            'start_time': '12:00',
            'periods': 118,
            'freq': '5min',
            'efficiency': 0.18,
            'area': 1.0,
            'max_delta_tilt': 5.0,
            'max_delta_azimuth': 10.0,
            'movement_penalty': 0.0,
            'randomize_day': True,
            'randomize_initial_orientation': True,
            'weather_source': 'clearsky',
        }

        # SAC agent defaults (same architecture as MBPO for fair comparison)
        self.agent_kwargs = agent_kwargs or {
            'state_dim': 6,
            'action_dim': 2,
            'hidden_dim': 256,  # Same hidden size as MBPO: (64, 64) expanded
            'lr': 3e-4,
            'gamma': 0.99,
            'tau': 0.005,
            'alpha_init': 0.2,
            'batch_size': 256,
            'replay_size': 100000,
            'automatic_entropy_tuning': True,
            'device': 'cpu',
        }

        # Training config (same epochs/horizon as MBPO for fair comparison)
        self.training_config = training_config or {
            'n_epochs': 2000,
            'n_initial_exploration_steps': 8000,
            'episode_length': 117,  # 117 steps (5-min intervals, 12:00-21:45)
            'eval_interval': 10,
            'save_interval': 50,
            'log_interval': 1,
        }

        # Initialize environment (same simulator as MBPO)
        self.env = SACEnvironment(**self.env_kwargs)

        # Initialize SAC agent (pure model-free)
        self.agent = SACAgent(**self.agent_kwargs)

        # Create output directory
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self._setup_logging()

        # Training state
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_steps = []
        self.training_loss_history = []
        self.current_episode = 0

    def _setup_logging(self):
        self.logger = logging.getLogger('SACTrainer')
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler(self.output_dir / 'training.log')
        handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        self.logger.addHandler(handler)

    def train_episode(self) -> Dict:
        """Run one SAC training episode (117 steps, same as MBPO horizon)."""
        state = self.env.reset()
        episode_reward = 0.0
        episode_steps = 0
        episode_losses = []

        for step in range(self.training_config['episode_length']):
            # SAC: select action from current policy (exploration)
            action = self.agent.select_action(state, deterministic=False)

            # Environment step (same pvlib physics as MBPO)
            next_state, reward, done, info = self.env.step(action)

            # Store in replay buffer (pure SAC, no model predictions)
            self.agent.store_experience(state, action, reward, next_state, float(done))

            # SAC update: only update after initial exploration steps
            if len(self.agent.replay_buffer.states) > self.agent_kwargs['batch_size']:
                if (self.current_episode * self.training_config['episode_length'] + step) > \
                   self.training_config['n_initial_exploration_steps']:
                    loss_info = self.agent.update()
                    episode_losses.append(loss_info)

            state = next_state
            episode_reward += reward
            episode_steps += 1

            if done:
                break

        # Log metrics
        mean_loss = np.mean([l['actor_loss'] for l in episode_losses]) if episode_losses else 0.0
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_steps)
        self.episode_steps.append(step + 1)
        self.current_episode += 1

        log_msg = (f"Episode {self.current_episode}: "
                   f"reward={episode_reward:.2f}, steps={episode_steps}, "
                   f"mean_loss={mean_loss:.4f}, alpha={self.agent.alpha:.4f}")
        self.logger.info(log_msg)

        return {
            'episode': self.current_episode,
            'reward': float(episode_reward),
            'steps': episode_steps,
            'mean_loss': float(mean_loss),
            'alpha': float(self.agent.alpha),
            'entropy_target': float(self.agent.target_entropy),
        }

    def train(self, n_epochs: int = None):
        """Main SAC training loop (pure model-free, no rollouts)."""
        epochs = n_epochs or self.training_config['n_epochs']
        logger.info(f"Starting SAC training for {epochs} episodes (model-free, no rollouts)")

        for ep in range(epochs):
            result = self.train_episode()

            # Save checkpoint
            if (ep + 1) % self.training_config['save_interval'] == 0:
                checkpoint_path = self.output_dir / f"sac_checkpoint_ep{ep+1}.pt"
                self.agent.save_checkpoint(str(checkpoint_path))

            # Log progress
            if (ep + 1) % self.training_config['log_interval'] == 0:
                avg_reward = np.mean(self.episode_rewards[-self.training_config['log_interval']:])
                logger.info(f"Epoch {ep+1}/{epochs}: avg_reward_10ep={avg_reward:.2f}, "
                           f"current_alpha={self.agent.alpha:.4f}")

        # Final checkpoint
        final_path = self.output_dir / "sac_final_checkpoint.pt"
        self.agent.save_checkpoint(str(final_path))
        logger.info(f"SAC training complete. Final checkpoint: {final_path}")

        return {
            'final_episode_rewards': self.episode_rewards,
            'mean_final_rewards': np.mean(self.episode_rewards[-10:]),
            'final_checkpoint_path': str(final_path),
        }
