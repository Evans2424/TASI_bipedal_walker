"""Replay buffers for storing and sampling experience."""

import numpy as np
from typing import Dict, Tuple


class ReplayBuffer:
    """Replay buffer for off-policy algorithms (SAC, TD3).

    Stores transitions and allows random sampling for training.
    """

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        capacity: int = 1000000,
        seed: int = 42
    ):
        """Initialize replay buffer.

        Args:
            observation_dim: Dimension of observation space
            action_dim: Dimension of action space
            capacity: Maximum buffer size
            seed: Random seed
        """
        self.capacity = capacity
        self.observation_dim = observation_dim
        self.action_dim = action_dim

        # Initialize buffers
        self.observations = np.zeros((capacity, observation_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_observations = np.zeros((capacity, observation_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

        self.ptr = 0
        self.size = 0

        self.rng = np.random.RandomState(seed)

    def add(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_observation: np.ndarray,
        done: bool
    ) -> None:
        """Add a transition to the buffer.

        Args:
            observation: Current observation
            action: Action taken
            reward: Reward received
            next_observation: Next observation
            done: Whether episode ended
        """
        self.observations[self.ptr] = observation
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_observations[self.ptr] = next_observation
        self.dones[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        """Sample a batch of transitions.

        Args:
            batch_size: Number of transitions to sample

        Returns:
            Dictionary containing batch of transitions
        """
        indices = self.rng.randint(0, self.size, size=batch_size)

        return {
            "observations": self.observations[indices],
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "next_observations": self.next_observations[indices],
            "dones": self.dones[indices],
        }

    def __len__(self) -> int:
        """Return current buffer size."""
        return self.size


class RolloutBuffer:
    """Rollout buffer for on-policy algorithms (PPO, A2C).

    Stores complete trajectories and returns them in order.
    """

    def __init__(self, observation_dim: int, action_dim: int):
        """Initialize rollout buffer.

        Args:
            observation_dim: Dimension of observation space
            action_dim: Dimension of action space
        """
        self.observation_dim = observation_dim
        self.action_dim = action_dim

        self.observations = []
        self.actions = []
        self.rewards = []
        self.next_observations = []
        self.dones = []

    def add(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_observation: np.ndarray,
        done: bool
    ) -> None:
        """Add a transition to the buffer.

        Args:
            observation: Current observation
            action: Action taken
            reward: Reward received
            next_observation: Next observation
            done: Whether episode ended
        """
        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_observations.append(next_observation)
        self.dones.append(done)

    def get(self) -> Dict[str, np.ndarray]:
        """Get all stored transitions and clear buffer.

        Returns:
            Dictionary containing all transitions
        """
        batch = {
            "observations": np.array(self.observations, dtype=np.float32),
            "actions": np.array(self.actions, dtype=np.float32),
            "rewards": np.array(self.rewards, dtype=np.float32),
            "next_observations": np.array(self.next_observations, dtype=np.float32),
            "dones": np.array(self.dones, dtype=np.float32),
        }

        # Clear buffer
        self.clear()

        return batch

    def clear(self) -> None:
        """Clear the buffer."""
        self.observations = []
        self.actions = []
        self.rewards = []
        self.next_observations = []
        self.dones = []

    def __len__(self) -> int:
        """Return current buffer size."""
        return len(self.observations)
