"""Environment wrappers for BipedalWalkerHardcore based on successful implementations.

These wrappers implement techniques from research and successful implementations:
1. Frame skipping (reduces decision frequency, accelerates training)
2. Reward clipping (stabilizes learning)
3. Modified failure penalty (encourages exploration)
4. Action smoothness reward (encourages smooth, natural gaits)
5. Hull stability reward (encourages upright posture)

References:
- https://github.com/ugurcanozalp/td3-sac-bipedal-walker-hardcore-v3
- https://janak-lal.com.np/solving-bipedal-walker-hardcore-challenge-with-soft-actor-critic-algorithm/
- https://github.com/DLR-RM/rl-baselines3-zoo
- https://pylessons.com/BipedalWalker-v3-PPO (smoothness penalties)
- https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2022.1054239/full
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class SmoothActionWrapper(gym.Wrapper):
    """Wrapper that encourages smooth actions to prevent jerky movements.

    This wrapper penalizes large changes in actions between consecutive timesteps,
    encouraging the agent to learn smooth, efficient gaits instead of erratic movements.

    The smoothness penalty is calculated as the L2 norm of action differences:
        penalty = smoothness_coef * ||action_t - action_{t-1}||^2

    Args:
        env: The environment to wrap
        smoothness_coef: Coefficient for action smoothness penalty (default: 0.05)
                        Higher values encourage smoother but potentially slower movement
    """

    def __init__(self, env, smoothness_coef=0.05):
        super().__init__(env)
        self.smoothness_coef = smoothness_coef
        self.prev_action = None

    def reset(self, **kwargs):
        """Reset environment and clear action history."""
        self.prev_action = None
        return self.env.reset(**kwargs)

    def step(self, action):
        """Execute action and apply smoothness penalty.

        Args:
            action: Current action

        Returns:
            observation, modified_reward, terminated, truncated, info
        """
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Apply smoothness penalty (skip first step where prev_action is None)
        if self.prev_action is not None:
            # Calculate action difference
            action_diff = np.array(action) - np.array(self.prev_action)
            # L2 norm of difference (penalizes jerky movements)
            smoothness_penalty = self.smoothness_coef * np.sum(action_diff ** 2)
            reward -= smoothness_penalty

            # Store in info for logging
            info['smoothness_penalty'] = smoothness_penalty

        # Update previous action
        self.prev_action = action.copy() if isinstance(action, np.ndarray) else np.array(action)

        return obs, reward, terminated, truncated, info


class HullStabilityWrapper(gym.Wrapper):
    """Wrapper that encourages the hull to maintain stable, upright posture.

    BipedalWalker observations include hull angle and angular velocity.
    This wrapper penalizes excessive hull tilting and rotation, encouraging
    the agent to maintain balance and walk smoothly.

    Observation indices:
        - hull_angle: obs[4]
        - hull_angular_velocity: obs[5]

    Args:
        env: The environment to wrap
        angle_coef: Penalty coefficient for hull angle deviation (default: 0.02)
        angular_vel_coef: Penalty coefficient for angular velocity (default: 0.01)
    """

    def __init__(self, env, angle_coef=0.02, angular_vel_coef=0.01):
        super().__init__(env)
        self.angle_coef = angle_coef
        self.angular_vel_coef = angular_vel_coef

    def step(self, action):
        """Execute action and apply stability rewards.

        Args:
            action: Action to execute

        Returns:
            observation, modified_reward, terminated, truncated, info
        """
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Extract hull angle and angular velocity from observations
        # obs[4] = hull angle, obs[5] = hull angular velocity
        hull_angle = obs[4]
        hull_angular_vel = obs[5]

        # Penalize excessive tilt (want to stay upright)
        angle_penalty = self.angle_coef * (hull_angle ** 2)

        # Penalize excessive rotation (want smooth, stable movement)
        angular_vel_penalty = self.angular_vel_coef * (hull_angular_vel ** 2)

        # Apply penalties
        total_stability_penalty = angle_penalty + angular_vel_penalty
        reward -= total_stability_penalty

        # Store in info for logging
        info['hull_angle_penalty'] = angle_penalty
        info['hull_angular_vel_penalty'] = angular_vel_penalty
        info['hull_angle'] = hull_angle

        return obs, reward, terminated, truncated, info


class HardcoreRewardWrapper(gym.RewardWrapper):
    """Reward wrapper for BipedalWalkerHardcore that improves learning.

    Modifications:
    1. Clips rewards to [-10, 10] to prevent extreme fluctuations
    2. Changes game over penalty from -100 to -10 to encourage exploration
       (agents won't be overly cautious about trying risky maneuvers like jumping)

    Args:
        env: The environment to wrap
        clip_min: Minimum reward value (default: -10)
        clip_max: Maximum reward value (default: 10)
    """

    def __init__(self, env, clip_min=-10.0, clip_max=10.0):
        super().__init__(env)
        self.clip_min = clip_min
        self.clip_max = clip_max

    def reward(self, reward):
        """Modify the reward signal.

        Args:
            reward: Original reward from environment

        Returns:
            Modified and clipped reward
        """
        # Check if episode ended in failure
        # In BipedalWalker, game_over is set when the hull touches ground or goes off-screen
        if hasattr(self.env.unwrapped, 'game_over') and self.env.unwrapped.game_over:
            # Use smaller penalty to encourage exploration
            reward = -10.0

        # Clip reward to prevent large fluctuations
        reward = np.clip(reward, self.clip_min, self.clip_max)

        return reward


class FrameSkipWrapper(gym.Wrapper):
    """Frame skipping wrapper that repeats actions across multiple steps.

    Frame skipping reduces the decision frequency, which:
    - Makes the control problem easier (agent has more time to react)
    - Speeds up training (fewer decisions needed)
    - Provides temporal consistency (similar to framestacking but simpler)

    Args:
        env: The environment to wrap
        skip: Number of frames to skip (default: 4, as used in successful implementations)
    """

    def __init__(self, env, skip=4):
        super().__init__(env)
        self.skip = skip

    def step(self, action):
        """Execute action for 'skip' frames and accumulate rewards.

        Args:
            action: Action to repeat

        Returns:
            observation, total_reward, terminated, truncated, info
        """
        total_reward = 0.0
        terminated = False
        truncated = False
        info = {}

        for _ in range(self.skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward

            # Stop early if episode ends
            if terminated or truncated:
                break

        return obs, total_reward, terminated, truncated, info


class HardcoreWrapper(gym.Wrapper):
    """Combined wrapper that applies all BipedalWalkerHardcore optimizations.

    This is a convenience wrapper that combines:
    - Frame skipping (skip=4)
    - Reward clipping ([-10, 10])
    - Modified failure penalty (-10 instead of -100)
    - Action smoothness reward (encourages smooth gaits)
    - Hull stability reward (encourages upright posture)

    Args:
        env: The environment to wrap
        frame_skip: Number of frames to skip (default: 4)
        reward_clip_min: Minimum reward (default: -10)
        reward_clip_max: Maximum reward (default: 10)
        smoothness_coef: Action smoothness penalty coefficient (default: 0.05)
        angle_coef: Hull angle penalty coefficient (default: 0.02)
        angular_vel_coef: Hull angular velocity penalty coefficient (default: 0.01)
    """

    def __init__(
        self,
        env,
        frame_skip=4,
        reward_clip_min=-10.0,
        reward_clip_max=10.0,
        smoothness_coef=0.05,
        angle_coef=0.02,
        angular_vel_coef=0.01
    ):
        # Apply wrappers in order:
        # 1. Frame skip (reduces decision frequency)
        env = FrameSkipWrapper(env, skip=frame_skip)
        # 2. Action smoothness (encourages smooth gaits)
        env = SmoothActionWrapper(env, smoothness_coef=smoothness_coef)
        # 3. Hull stability (encourages upright posture)
        env = HullStabilityWrapper(env, angle_coef=angle_coef, angular_vel_coef=angular_vel_coef)
        # 4. Reward modifications (clipping and failure penalty)
        env = HardcoreRewardWrapper(env, clip_min=reward_clip_min, clip_max=reward_clip_max)

        super().__init__(env)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        return self.env.step(action)


def make_hardcore_env(
    env_id: str,
    rank: int,
    seed: int = 0,
    frame_skip: int = 4,
    smoothness_coef: float = 0.05,
    angle_coef: float = 0.02,
    angular_vel_coef: float = 0.01
):
    """Factory function to create wrapped BipedalWalkerHardcore environment.

    This creates an environment with all the optimizations from successful implementations:
    - Frame skipping (default: 4)
    - Reward clipping ([-10, 10])
    - Modified failure penalty
    - Action smoothness reward
    - Hull stability reward

    Args:
        env_id: Gym environment ID (should be 'BipedalWalker-v3')
        rank: Unique ID for this environment (for parallel training)
        seed: Random seed
        frame_skip: Number of frames to skip (default: 4)
        smoothness_coef: Action smoothness penalty (default: 0.05)
        angle_coef: Hull angle penalty (default: 0.02)
        angular_vel_coef: Hull angular velocity penalty (default: 0.01)

    Returns:
        Callable that creates the wrapped environment
    """
    def _init():
        env = gym.make(env_id, hardcore=True)
        env.reset(seed=seed + rank)
        # Apply hardcore optimizations
        env = HardcoreWrapper(
            env,
            frame_skip=frame_skip,
            smoothness_coef=smoothness_coef,
            angle_coef=angle_coef,
            angular_vel_coef=angular_vel_coef
        )
        # Monitor wrapper should be applied by the training script
        return env
    return _init
