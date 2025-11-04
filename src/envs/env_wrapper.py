"""Environment wrappers for Bipedal Walker."""

import gymnasium as gym
import numpy as np
from typing import Tuple, Optional


class BipedalWalkerWrapper(gym.Wrapper):
    """Wrapper for Bipedal Walker environment with additional utilities.

    Provides observation/action clipping, reward normalization options,
    and episode tracking.
    """

    def __init__(
        self,
        env: gym.Env,
        reward_scale: float = 1.0,
        clip_observations: bool = False,
        clip_actions: bool = True
    ):
        """Initialize wrapper.

        Args:
            env: Base Gymnasium environment
            reward_scale: Scale factor for rewards
            clip_observations: Whether to clip observations
            clip_actions: Whether to clip actions
        """
        super().__init__(env)
        self.reward_scale = reward_scale
        self.clip_observations = clip_observations
        self.clip_actions = clip_actions

        # Episode tracking
        self.episode_reward = 0.0
        self.episode_length = 0

    def reset(self, **kwargs) -> Tuple[np.ndarray, dict]:
        """Reset the environment.

        Returns:
            observation: Initial observation
            info: Additional information
        """
        observation, info = self.env.reset(**kwargs)

        # Reset episode tracking
        self.episode_reward = 0.0
        self.episode_length = 0

        if self.clip_observations:
            observation = np.clip(observation, -10, 10)

        return observation, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Take a step in the environment.

        Args:
            action: Action to take

        Returns:
            observation: Next observation
            reward: Reward received
            terminated: Whether episode terminated
            truncated: Whether episode was truncated
            info: Additional information
        """
        if self.clip_actions:
            action = np.clip(action, -1.0, 1.0)

        observation, reward, terminated, truncated, info = self.env.step(action)

        # Scale reward
        reward = reward * self.reward_scale

        # Clip observations if needed
        if self.clip_observations:
            observation = np.clip(observation, -10, 10)

        # Update episode tracking
        self.episode_reward += reward
        self.episode_length += 1

        # Add episode info when done
        if terminated or truncated:
            info["episode"] = {
                "r": self.episode_reward,
                "l": self.episode_length
            }

        return observation, reward, terminated, truncated, info


def make_env(
    env_id: str = "BipedalWalker-v3",
    hardcore: bool = False,
    render_mode: Optional[str] = None,
    reward_scale: float = 1.0,
    clip_observations: bool = False,
    clip_actions: bool = True,
    seed: Optional[int] = None
) -> gym.Env:
    """Create and wrap Bipedal Walker environment.

    Args:
        env_id: Environment ID
        hardcore: Whether to use hardcore mode
        render_mode: Render mode ('human', 'rgb_array', None)
        reward_scale: Scale factor for rewards
        clip_observations: Whether to clip observations
        clip_actions: Whether to clip actions
        seed: Random seed

    Returns:
        Wrapped environment
    """
    # Create base environment
    if hardcore:
        env_id = "BipedalWalkerHardcore-v3"

    env = gym.make(env_id, render_mode=render_mode)

    # Set seed if provided
    if seed is not None:
        env.reset(seed=seed)

    # Wrap environment
    env = BipedalWalkerWrapper(
        env,
        reward_scale=reward_scale,
        clip_observations=clip_observations,
        clip_actions=clip_actions
    )

    return env
