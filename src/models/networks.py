"""Neural network architectures for policy and value functions."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from typing import Tuple


class MLP(nn.Module):
    """Multi-layer perceptron with optional layer normalization."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Tuple[int, ...],
        output_dim: int,
        activation: nn.Module = nn.ReLU,
        layer_norm: bool = False,
        output_activation: nn.Module = None
    ):
        """Initialize MLP.

        Args:
            input_dim: Input dimension
            hidden_dims: Tuple of hidden layer dimensions
            output_dim: Output dimension
            activation: Activation function
            layer_norm: Whether to use layer normalization
            output_activation: Activation for output layer (None for no activation)
        """
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(activation())
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        if output_activation is not None:
            layers.append(output_activation())

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.network(x)


class Actor(nn.Module):
    """Deterministic policy network (for DDPG/TD3)."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dims: Tuple[int, ...] = (256, 256),
        max_action: float = 1.0
    ):
        """Initialize Actor.

        Args:
            observation_dim: Observation space dimension
            action_dim: Action space dimension
            hidden_dims: Hidden layer dimensions
            max_action: Maximum action value
        """
        super().__init__()

        self.max_action = max_action
        self.network = MLP(
            input_dim=observation_dim,
            hidden_dims=hidden_dims,
            output_dim=action_dim,
            activation=nn.ReLU,
            output_activation=nn.Tanh
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            observation: Batch of observations

        Returns:
            Actions scaled to [-max_action, max_action]
        """
        return self.max_action * self.network(observation)


class Critic(nn.Module):
    """Value network for state-action pairs."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dims: Tuple[int, ...] = (256, 256)
    ):
        """Initialize Critic.

        Args:
            observation_dim: Observation space dimension
            action_dim: Action space dimension
            hidden_dims: Hidden layer dimensions
        """
        super().__init__()

        self.network = MLP(
            input_dim=observation_dim + action_dim,
            hidden_dims=hidden_dims,
            output_dim=1,
            activation=nn.ReLU
        )

    def forward(self, observation: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            observation: Batch of observations
            action: Batch of actions

        Returns:
            Q-values for (observation, action) pairs
        """
        x = torch.cat([observation, action], dim=-1)
        return self.network(x)


class GaussianActor(nn.Module):
    """Stochastic policy network with Gaussian distribution (for PPO/SAC)."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dims: Tuple[int, ...] = (256, 256),
        log_std_min: float = -20.0,
        log_std_max: float = 2.0
    ):
        """Initialize Gaussian Actor.

        Args:
            observation_dim: Observation space dimension
            action_dim: Action space dimension
            hidden_dims: Hidden layer dimensions
            log_std_min: Minimum log standard deviation
            log_std_max: Maximum log standard deviation
        """
        super().__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # Shared backbone
        self.backbone = MLP(
            input_dim=observation_dim,
            hidden_dims=hidden_dims,
            output_dim=hidden_dims[-1],
            activation=nn.ReLU
        )

        # Mean and log_std heads
        self.mean_head = nn.Linear(hidden_dims[-1], action_dim)
        self.log_std_head = nn.Linear(hidden_dims[-1], action_dim)

    def forward(self, observation: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            observation: Batch of observations

        Returns:
            mean: Mean of action distribution
            log_std: Log standard deviation of action distribution
        """
        features = self.backbone(observation)
        mean = self.mean_head(features)
        log_std = self.log_std_head(features)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def sample(self, observation: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample action from policy.

        Args:
            observation: Batch of observations

        Returns:
            action: Sampled actions
            log_prob: Log probability of actions
        """
        mean, log_std = self.forward(observation)
        std = log_std.exp()

        # Create normal distribution and sample
        normal = Normal(mean, std)
        action = normal.rsample()  # Reparameterization trick

        # Compute log probability
        log_prob = normal.log_prob(action).sum(dim=-1, keepdim=True)

        # Apply tanh squashing
        action = torch.tanh(action)

        # Correct log_prob for tanh squashing
        log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(dim=-1, keepdim=True)

        return action, log_prob

    def get_action_and_log_prob(self, observation: torch.Tensor, deterministic: bool = False):
        """Get action and its log probability.

        Args:
            observation: Batch of observations
            deterministic: If True, return mean action

        Returns:
            action: Selected action
            log_prob: Log probability of action (None if deterministic)
        """
        if deterministic:
            mean, _ = self.forward(observation)
            return torch.tanh(mean), None
        else:
            return self.sample(observation)


class StateValueNetwork(nn.Module):
    """State value function V(s)."""

    def __init__(
        self,
        observation_dim: int,
        hidden_dims: Tuple[int, ...] = (256, 256)
    ):
        """Initialize state value network.

        Args:
            observation_dim: Observation space dimension
            hidden_dims: Hidden layer dimensions
        """
        super().__init__()

        self.network = MLP(
            input_dim=observation_dim,
            hidden_dims=hidden_dims,
            output_dim=1,
            activation=nn.ReLU
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            observation: Batch of observations

        Returns:
            State values
        """
        return self.network(observation)
