"""Environment wrappers for Bipedal Walker."""

import gymnasium as gym
import numpy as np
from typing import Tuple, Optional


class BipedalWalkerWrapper(gym.Wrapper):
    """Wrapper for Bipedal Walker environment with additional utilities.

    Provides observation/action clipping, reward normalization options,
    episode tracking, and hardcore mode enhancements.
    """

    def __init__(
        self,
        env: gym.Env,
        reward_scale: float = 1.0,
        clip_observations: bool = False,
        clip_actions: bool = True,
        normalize_observations: bool = False,
        normalize_rewards: bool = False,
        clip_normalized_obs: float = 10.0,
        clip_normalized_reward: float = 10.0,
        frame_skip: int = 1,
        smoothness_coef: float = 0.0,
        hull_angle_coef: float = 0.0,
        hull_angular_vel_coef: float = 0.0
    ):
        """Initialize wrapper.

        Args:
            env: Base Gymnasium environment
            reward_scale: Scale factor for rewards
            clip_observations: Whether to clip observations
            clip_actions: Whether to clip actions
            normalize_observations: Whether to normalize observations
            normalize_rewards: Whether to normalize rewards
            clip_normalized_obs: Clipping range for normalized observations
            clip_normalized_reward: Clipping range for normalized rewards
            frame_skip: Number of frames to skip (repeat action)
            smoothness_coef: Penalty coefficient for action smoothness
            hull_angle_coef: Penalty coefficient for hull angle
            hull_angular_vel_coef: Penalty coefficient for hull angular velocity
        """
        super().__init__(env)
        self.reward_scale = reward_scale
        self.clip_observations = clip_observations
        self.clip_actions = clip_actions
        self.normalize_observations = normalize_observations
        self.normalize_rewards = normalize_rewards
        self.clip_normalized_obs = clip_normalized_obs
        self.clip_normalized_reward = clip_normalized_reward
        self.frame_skip = frame_skip
        self.smoothness_coef = smoothness_coef
        self.hull_angle_coef = hull_angle_coef
        self.hull_angular_vel_coef = hull_angular_vel_coef

        # Episode tracking
        self.episode_reward = 0.0
        self.episode_length = 0
        self.last_action = None
        
        # Normalization statistics
        self.obs_mean = np.zeros(env.observation_space.shape[0])
        self.obs_std = np.ones(env.observation_space.shape[0])
        self.reward_mean = 0.0
        self.reward_std = 1.0

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
        self.last_action = None

        if self.clip_observations:
            observation = np.clip(observation, -10, 10)
        
        if self.normalize_observations:
            observation = (observation - self.obs_mean) / (self.obs_std + 1e-8)
            observation = np.clip(observation, -self.clip_normalized_obs, self.clip_normalized_obs)

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

        # Frame skip (repeat action)
        total_reward = 0.0
        for _ in range(self.frame_skip):
            observation, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break

        reward = total_reward

        # Apply reward shaping penalties
        if self.smoothness_coef > 0 and self.last_action is not None:
            # Penalize difference from last action (smoothness)
            action_diff = np.sum(np.abs(action - self.last_action))
            reward -= self.smoothness_coef * action_diff
        
        if self.hull_angle_coef > 0 or self.hull_angular_vel_coef > 0:
            # observation[0] is hull angle, observation[1] is angular velocity
            if len(observation) > 1:
                reward -= self.hull_angle_coef * np.abs(observation[0])
                reward -= self.hull_angular_vel_coef * np.abs(observation[1])
        
        self.last_action = action.copy()

        # Scale reward
        reward = reward * self.reward_scale

        # Normalize reward if enabled
        if self.normalize_rewards:
            reward = (reward - self.reward_mean) / (self.reward_std + 1e-8)
            reward = np.clip(reward, -self.clip_normalized_reward, self.clip_normalized_reward)

        # Clip observations if needed
        if self.clip_observations:
            observation = np.clip(observation, -10, 10)
        
        if self.normalize_observations:
            observation = (observation - self.obs_mean) / (self.obs_std + 1e-8)
            observation = np.clip(observation, -self.clip_normalized_obs, self.clip_normalized_obs)

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
    normalize_observations: bool = False,
    normalize_rewards: bool = False,
    clip_normalized_obs: float = 10.0,
    clip_normalized_reward: float = 10.0,
    frame_skip: int = 1,
    smoothness_coef: float = 0.0,
    hull_angle_coef: float = 0.0,
    hull_angular_vel_coef: float = 0.0,
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
        normalize_observations: Whether to normalize observations
        normalize_rewards: Whether to normalize rewards
        clip_normalized_obs: Clipping range for normalized observations
        clip_normalized_reward: Clipping range for normalized rewards
        frame_skip: Number of frames to skip (repeat action)
        smoothness_coef: Penalty coefficient for action smoothness
        hull_angle_coef: Penalty coefficient for hull angle
        hull_angular_vel_coef: Penalty coefficient for hull angular velocity
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
        clip_actions=clip_actions,
        normalize_observations=normalize_observations,
        normalize_rewards=normalize_rewards,
        clip_normalized_obs=clip_normalized_obs,
        clip_normalized_reward=clip_normalized_reward,
        frame_skip=frame_skip,
        smoothness_coef=smoothness_coef,
        hull_angle_coef=hull_angle_coef,
        hull_angular_vel_coef=hull_angular_vel_coef
    )

    return env
