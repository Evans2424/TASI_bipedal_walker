"""Base agent class for reinforcement learning algorithms."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
import torch
import torch.nn as nn
import numpy as np


class BaseAgent(ABC):
    """Abstract base class for RL agents.

    Provides a common interface for different RL algorithms,
    ensuring consistency across implementations.
    """

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dims: Tuple[int, ...] = (256, 256),
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        seed: int = 42
    ):
        """Initialize base agent.

        Args:
            observation_dim: Dimension of observation space
            action_dim: Dimension of action space
            hidden_dims: Hidden layer dimensions for neural networks
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            device: Device to run computations on
            seed: Random seed for reproducibility
        """
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.device = torch.device(device)
        self.seed = seed

        # Set random seeds
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    @abstractmethod
    def select_action(self, observation: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Select an action given an observation.

        Args:
            observation: Current observation from environment
            deterministic: Whether to select action deterministically

        Returns:
            Selected action
        """
        pass

    @abstractmethod
    def update(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """Update agent parameters using a batch of experience.

        Args:
            batch: Dictionary containing experience batch

        Returns:
            Dictionary of training metrics
        """
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """Save agent parameters.

        Args:
            path: Path to save checkpoint
        """
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """Load agent parameters.

        Args:
            path: Path to load checkpoint from
        """
        pass

    def to_tensor(self, x: np.ndarray) -> torch.Tensor:
        """Convert numpy array to torch tensor on correct device.

        Args:
            x: Numpy array

        Returns:
            Torch tensor
        """
        return torch.FloatTensor(x).to(self.device)
