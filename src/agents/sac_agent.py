"""Soft Actor-Critic (SAC) agent implementation."""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Any, Tuple
from .base_agent import BaseAgent
from ..models.networks import GaussianActor, Critic


class SACAgent(BaseAgent):
    """Soft Actor-Critic agent for continuous control.

    SAC is an off-policy actor-critic algorithm that maximizes both
    expected return and entropy, encouraging exploration.

    Reference: Haarnoja et al., 2018 - "Soft Actor-Critic: Off-Policy Maximum
    Entropy Deep Reinforcement Learning with a Stochastic Actor"
    """

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dims: Tuple[int, ...] = (256, 256),
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        automatic_entropy_tuning: bool = True,
        target_entropy: float = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        seed: int = 42
    ):
        """Initialize SAC agent.

        Args:
            observation_dim: Dimension of observation space
            action_dim: Dimension of action space
            hidden_dims: Hidden layer dimensions
            learning_rate: Learning rate
            gamma: Discount factor
            tau: Soft update coefficient for target networks
            alpha: Entropy regularization coefficient
            automatic_entropy_tuning: Whether to automatically tune alpha
            target_entropy: Target entropy for automatic tuning (default: -action_dim)
            device: Device to run on
            seed: Random seed
        """
        super().__init__(observation_dim, action_dim, hidden_dims, learning_rate, gamma, device, seed)

        self.tau = tau
        self.automatic_entropy_tuning = automatic_entropy_tuning

        # Initialize actor
        self.actor = GaussianActor(observation_dim, action_dim, hidden_dims).to(self.device)

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

        # Automatic entropy tuning
        if self.automatic_entropy_tuning:
            if target_entropy is None:
                self.target_entropy = -action_dim
            else:
                self.target_entropy = target_entropy

            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp()
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=learning_rate)
        else:
            self.alpha = alpha

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

    def update(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """Update agent using SAC algorithm.

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
            next_actions, next_log_probs = self.actor.sample(next_observations)

            q1_next_target = self.critic1_target(next_observations, next_actions)
            q2_next_target = self.critic2_target(next_observations, next_actions)
            min_q_next_target = torch.min(q1_next_target, q2_next_target)

            if isinstance(self.alpha, torch.Tensor):
                alpha_value = self.alpha.detach()
            else:
                alpha_value = self.alpha

            next_q_value = min_q_next_target - alpha_value * next_log_probs
            target_q_value = rewards + (1 - dones) * self.gamma * next_q_value

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

        # Update actor
        new_actions, log_probs = self.actor.sample(observations)

        q1_new = self.critic1(observations, new_actions)
        q2_new = self.critic2(observations, new_actions)
        min_q_new = torch.min(q1_new, q2_new)

        if isinstance(self.alpha, torch.Tensor):
            alpha_value = self.alpha.detach()
        else:
            alpha_value = self.alpha

        actor_loss = (alpha_value * log_probs - min_q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update alpha (temperature parameter)
        alpha_loss = None
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            self.alpha = self.log_alpha.exp()

        # Soft update target networks
        self._soft_update(self.critic1, self.critic1_target)
        self._soft_update(self.critic2, self.critic2_target)

        # Return metrics
        metrics = {
            "critic1_loss": critic1_loss.item(),
            "critic2_loss": critic2_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha": self.alpha.item() if isinstance(self.alpha, torch.Tensor) else self.alpha,
        }

        if alpha_loss is not None:
            metrics["alpha_loss"] = alpha_loss.item()

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
            "critic1_state_dict": self.critic1.state_dict(),
            "critic2_state_dict": self.critic2.state_dict(),
            "critic1_target_state_dict": self.critic1_target.state_dict(),
            "critic2_target_state_dict": self.critic2_target.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic1_optimizer_state_dict": self.critic1_optimizer.state_dict(),
            "critic2_optimizer_state_dict": self.critic2_optimizer.state_dict(),
        }

        if self.automatic_entropy_tuning:
            save_dict["log_alpha"] = self.log_alpha
            save_dict["alpha_optimizer_state_dict"] = self.alpha_optimizer.state_dict()

        torch.save(save_dict, path)

    def load(self, path: str) -> None:
        """Load agent parameters."""
        checkpoint = torch.load(path, map_location=self.device)

        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic1.load_state_dict(checkpoint["critic1_state_dict"])
        self.critic2.load_state_dict(checkpoint["critic2_state_dict"])
        self.critic1_target.load_state_dict(checkpoint["critic1_target_state_dict"])
        self.critic2_target.load_state_dict(checkpoint["critic2_target_state_dict"])

        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        self.critic1_optimizer.load_state_dict(checkpoint["critic1_optimizer_state_dict"])
        self.critic2_optimizer.load_state_dict(checkpoint["critic2_optimizer_state_dict"])

        if self.automatic_entropy_tuning and "log_alpha" in checkpoint:
            self.log_alpha = checkpoint["log_alpha"]
            self.alpha = self.log_alpha.exp()
            self.alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer_state_dict"])
