"""Proximal Policy Optimization (PPO) agent implementation."""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Any, Tuple
from .base_agent import BaseAgent
from ..models.networks import GaussianActor, StateValueNetwork


class PPOAgent(BaseAgent):
    """Proximal Policy Optimization agent for continuous control.

    PPO is a policy gradient method that uses a clipped surrogate objective
    to prevent large policy updates, ensuring stable training.

    Reference: Schulman et al., 2017 - "Proximal Policy Optimization Algorithms"
    """

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dims: Tuple[int, ...] = (256, 256),
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        ppo_epochs: int = 10,
        mini_batch_size: int = 64,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        seed: int = 42
    ):
        """Initialize PPO agent.

        Args:
            observation_dim: Dimension of observation space
            action_dim: Dimension of action space
            hidden_dims: Hidden layer dimensions
            learning_rate: Learning rate
            gamma: Discount factor
            gae_lambda: GAE lambda parameter for advantage estimation
            clip_epsilon: PPO clipping parameter
            value_loss_coef: Coefficient for value loss
            entropy_coef: Coefficient for entropy bonus
            max_grad_norm: Maximum gradient norm for clipping
            ppo_epochs: Number of epochs to update policy per batch
            mini_batch_size: Mini-batch size for updates
            device: Device to run on
            seed: Random seed
        """
        super().__init__(observation_dim, action_dim, hidden_dims, learning_rate, gamma, device, seed)

        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size

        # Initialize networks
        self.actor = GaussianActor(observation_dim, action_dim, hidden_dims).to(self.device)
        self.critic = StateValueNetwork(observation_dim, hidden_dims).to(self.device)

        # Initialize optimizer
        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=learning_rate
        )

    def select_action(self, observation: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Select action using current policy.

        Args:
            observation: Current observation
            deterministic: If True, return mean action

        Returns:
            Selected action
        """
        with torch.no_grad():
            obs_tensor = self.to_tensor(observation).unsqueeze(0)
            action, _ = self.actor.get_action_and_log_prob(obs_tensor, deterministic)
            return action.cpu().numpy()[0]

    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        next_values: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Generalized Advantage Estimation (GAE).

        Args:
            rewards: Rewards tensor [batch_size]
            values: Value estimates [batch_size]
            dones: Done flags [batch_size]
            next_values: Next state values [batch_size]

        Returns:
            advantages: Computed advantages
            returns: Computed returns
        """
        advantages = []
        gae = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = next_values[t]
            else:
                next_value = values[t + 1]

            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)

        advantages = torch.stack(advantages)
        returns = advantages + values

        return advantages, returns

    def update(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """Update agent using PPO algorithm.

        Args:
            batch: Dictionary containing:
                - observations: [batch_size, obs_dim]
                - actions: [batch_size, action_dim]
                - rewards: [batch_size]
                - dones: [batch_size]
                - next_observations: [batch_size, obs_dim]

        Returns:
            Dictionary of training metrics
        """
        # Convert batch to tensors
        observations = self.to_tensor(batch["observations"])
        actions = self.to_tensor(batch["actions"])
        rewards = self.to_tensor(batch["rewards"])
        dones = self.to_tensor(batch["dones"])
        next_observations = self.to_tensor(batch["next_observations"])

        # Compute old log probabilities and values
        with torch.no_grad():
            _, old_log_probs = self.actor.sample(observations)
            values = self.critic(observations).squeeze()
            next_values = self.critic(next_observations).squeeze()

            # Compute advantages and returns
            advantages, returns = self.compute_gae(rewards, values, dones, next_values)

            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update for multiple epochs
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        num_updates = 0

        batch_size = observations.shape[0]
        indices = np.arange(batch_size)

        for _ in range(self.ppo_epochs):
            np.random.shuffle(indices)

            for start in range(0, batch_size, self.mini_batch_size):
                end = start + self.mini_batch_size
                mb_indices = indices[start:end]

                mb_obs = observations[mb_indices]
                mb_actions = actions[mb_indices]
                mb_old_log_probs = old_log_probs[mb_indices]
                mb_advantages = advantages[mb_indices]
                mb_returns = returns[mb_indices]

                # Compute current log probabilities and values
                mean, log_std = self.actor(mb_obs)
                std = log_std.exp()
                dist = torch.distributions.Normal(mean, std)

                # Compute log prob for the taken actions
                log_probs = dist.log_prob(mb_actions).sum(dim=-1, keepdim=True)
                entropy = dist.entropy().sum(dim=-1).mean()

                current_values = self.critic(mb_obs).squeeze()

                # Compute policy loss with clipping
                ratio = torch.exp(log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages.unsqueeze(1)
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * mb_advantages.unsqueeze(1)
                policy_loss = -torch.min(surr1, surr2).mean()

                # Compute value loss
                value_loss = nn.MSELoss()(current_values, mb_returns)

                # Total loss
                loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    self.max_grad_norm
                )
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                num_updates += 1

        # Return metrics
        return {
            "policy_loss": total_policy_loss / num_updates,
            "value_loss": total_value_loss / num_updates,
            "entropy": total_entropy / num_updates,
        }

    def save(self, path: str) -> None:
        """Save agent parameters."""
        torch.save({
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }, path)

    def load(self, path: str) -> None:
        """Load agent parameters."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
