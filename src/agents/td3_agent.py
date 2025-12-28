"""Twin Delayed Deep Deterministic Policy Gradient (TD3) agent implementation."""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Any, Tuple
from .base_agent import BaseAgent
from ..models.networks import Actor, Critic


class TD3Agent(BaseAgent):
    """Twin Delayed Deep Deterministic Policy Gradient agent for continuous control.

    TD3 is an off-policy actor-critic algorithm that builds upon DDPG by addressing
    overestimation of Q-values through:
    - Twin critics (Q1, Q2) and using the minimum estimate
    - Delayed updates of the actor and target networks
    - Target smoothing regularization (TSMR)

    Reference: Fujimoto et al., 2018 - "Addressing Function Approximation Error in
    Actor-Critic Methods"
    """

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dims: Tuple[int, ...] = (256, 256),
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        target_noise: float = 0.2,
        noise_clip: float = 0.5,
        policy_update_freq: int = 2,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        seed: int = 42
    ):
        """Initialize TD3 agent.

        Args:
            observation_dim: Dimension of observation space
            action_dim: Dimension of action space
            hidden_dims: Hidden layer dimensions
            learning_rate: Learning rate
            gamma: Discount factor
            tau: Soft update coefficient for target networks
            target_noise: Standard deviation of smoothing noise added to target actions
            noise_clip: Range for clipping smoothing noise
            policy_update_freq: Update actor and target networks every N critic updates
            device: Device to run on
            seed: Random seed
        """
        super().__init__(observation_dim, action_dim, hidden_dims, learning_rate, gamma, device, seed)

        self.tau = tau
        self.target_noise = target_noise
        self.noise_clip = noise_clip
        self.policy_update_freq = policy_update_freq
        self.critic_update_count = 0

        # Initialize actor
        self.actor = Actor(observation_dim, action_dim, hidden_dims).to(self.device)
        self.actor_target = Actor(observation_dim, action_dim, hidden_dims).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        # Initialize twin critics (Q1, Q2) and their targets
        self.critic1 = Critic(observation_dim, action_dim, hidden_dims).to(self.device)
        self.critic2 = Critic(observation_dim, action_dim, hidden_dims).to(self.device)

        self.critic1_target = Critic(observation_dim, action_dim, hidden_dims).to(self.device)
        self.critic2_target = Critic(observation_dim, action_dim, hidden_dims).to(self.device)

        # Initialize target networks with same weights
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        # Initialize optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=learning_rate)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=learning_rate)

    def select_action(self, observation: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Select action using current policy.

        Args:
            observation: Current observation
            deterministic: If True, return deterministic action (no exploration noise).
                         If False, add Gaussian exploration noise for training.

        Returns:
            Selected action in range [-1, 1]
        """
        with torch.no_grad():
            obs_tensor = self.to_tensor(observation).unsqueeze(0)
            action = self.actor(obs_tensor)
            action = action.cpu().numpy()[0]
        
        # Add exploration noise during training (deterministic=False)
        if not deterministic:
            # Gaussian noise for exploration
            noise = np.random.normal(0, self.target_noise, size=action.shape)
            # Clip noise to reasonable range
            noise = np.clip(noise, -self.noise_clip, self.noise_clip)
            action = action + noise
            # Ensure action remains in valid range [-1, 1]
            action = np.clip(action, -1.0, 1.0)
        
        return action

    def update(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """Update agent using TD3 algorithm.

        Args:
            batch: Dictionary containing:
                - observations: [batch_size, obs_dim]
                - actions: [batch_size, action_dim]
                - rewards: [batch_size]
                - next_observations: [batch_size, obs_dim]
                - dones: [batch_size]

        Returns:
            Dictionary of training metrics
        """
        # Convert batch to tensors
        observations = self.to_tensor(batch["observations"])
        actions = self.to_tensor(batch["actions"])
        rewards = self.to_tensor(batch["rewards"]).unsqueeze(1)
        next_observations = self.to_tensor(batch["next_observations"])
        dones = self.to_tensor(batch["dones"]).unsqueeze(1)

        # Update critics
        with torch.no_grad():
            # Target smoothing regularization (TSMR)
            next_actions = self.actor_target(next_observations)
            
            # Add smoothing noise to target actions
            noise = torch.randn_like(next_actions) * self.target_noise
            noise = torch.clamp(noise, -self.noise_clip, self.noise_clip)
            next_actions = next_actions + noise
            next_actions = torch.clamp(next_actions, -1.0, 1.0)

            # Compute target Q-values using minimum of twin critics
            q1_next_target = self.critic1_target(next_observations, next_actions)
            q2_next_target = self.critic2_target(next_observations, next_actions)
            min_q_next_target = torch.min(q1_next_target, q2_next_target)

            target_q_value = rewards + (1 - dones) * self.gamma * min_q_next_target

        # Critic 1 loss
        q1 = self.critic1(observations, actions)
        critic1_loss = nn.MSELoss()(q1, target_q_value)

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        # Critic 2 loss
        q2 = self.critic2(observations, actions)
        critic2_loss = nn.MSELoss()(q2, target_q_value)

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        # Delayed policy update
        metrics = {
            "critic1_loss": critic1_loss.item(),
            "critic2_loss": critic2_loss.item(),
            "actor_loss": 0.0,
        }

        self.critic_update_count += 1

        if self.critic_update_count % self.policy_update_freq == 0:
            # Actor loss (policy gradient)
            actor_actions = self.actor(observations)
            q1_actor = self.critic1(observations, actor_actions)
            actor_loss = -q1_actor.mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Soft update target networks
            self._soft_update(self.actor, self.actor_target)
            self._soft_update(self.critic1, self.critic1_target)
            self._soft_update(self.critic2, self.critic2_target)

            metrics["actor_loss"] = actor_loss.item()

        return metrics

    def _soft_update(self, source: nn.Module, target: nn.Module) -> None:
        """Soft update target network parameters.

        θ_target = τ * θ_source + (1 - τ) * θ_target

        Args:
            source: Source network
            target: Target network
        """
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, path: str) -> None:
        """Save agent parameters."""
        save_dict = {
            "actor_state_dict": self.actor.state_dict(),
            "actor_target_state_dict": self.actor_target.state_dict(),
            "critic1_state_dict": self.critic1.state_dict(),
            "critic2_state_dict": self.critic2.state_dict(),
            "critic1_target_state_dict": self.critic1_target.state_dict(),
            "critic2_target_state_dict": self.critic2_target.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic1_optimizer_state_dict": self.critic1_optimizer.state_dict(),
            "critic2_optimizer_state_dict": self.critic2_optimizer.state_dict(),
            "critic_update_count": self.critic_update_count,
        }

        torch.save(save_dict, path)

    def load(self, path: str) -> None:
        """Load agent parameters."""
        checkpoint = torch.load(path, map_location=self.device)

        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.actor_target.load_state_dict(checkpoint["actor_target_state_dict"])
        self.critic1.load_state_dict(checkpoint["critic1_state_dict"])
        self.critic2.load_state_dict(checkpoint["critic2_state_dict"])
        self.critic1_target.load_state_dict(checkpoint["critic1_target_state_dict"])
        self.critic2_target.load_state_dict(checkpoint["critic2_target_state_dict"])

        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        self.critic1_optimizer.load_state_dict(checkpoint["critic1_optimizer_state_dict"])
        self.critic2_optimizer.load_state_dict(checkpoint["critic2_optimizer_state_dict"])

        if "critic_update_count" in checkpoint:
            self.critic_update_count = checkpoint["critic_update_count"]
