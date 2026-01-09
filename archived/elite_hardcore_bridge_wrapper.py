"""Elite Hardcore Wrapper - BRIDGE AWARE VERSION

This wrapper extends EliteHardcoreWrapper to handle BRIDGE obstacles in custom_walker.py.

KEY FIX: Reduces penalties when agent is "waiting" for bridges to lower.

Bridge behavior:
- Bridges activate when robot within 10 units
- Wait 300 steps (6 seconds) before lowering
- Agent must stand still and maintain balance

Problem with standard Elite Hardcore:
- Penalizes standing still (smoothness, hull stability)
- No forward progress reward while waiting
- 300 steps of waiting = -30 to -50 reward!

Solution:
- Detect "waiting" state: Low velocity + upright + stable
- Reduce penalties by 80% during wait
- Add small "patience" bonus for maintaining position

This allows bridges to work while maintaining hardcore obstacle-solving ability.
"""

import numpy as np
import gymnasium as gym
from elite_hardcore_wrapper import EliteHardcoreWrapper


class EliteHardcoreBridgeWrapper(EliteHardcoreWrapper):
    """Bridge-aware version of Elite Hardcore wrapper.

    Extends EliteHardcoreWrapper with bridge-specific handling:
    - Detects when agent is "waiting" (low velocity + stable)
    - Reduces penalties during wait period
    - Adds patience bonus for maintaining position

    All other functionality identical to EliteHardcoreWrapper.
    """

    def __init__(
        self,
        env,
        # Core hardcore features (same as parent)
        frame_skip=4,
        smoothness_coef=0.2,
        hull_angle_coef=0.1,
        hull_angular_vel_coef=0.05,
        # Natural walking augmentations (same as parent)
        knee_bend_reward=0.02,
        min_bend_threshold=0.3,
        max_joint_velocity=2.0,
        velocity_penalty=0.02,
        early_steps_stability_bonus=0.01,
        early_steps_count=100,
        # Bridge-specific parameters (NEW)
        waiting_velocity_threshold=0.1,  # Consider "waiting" if vel < 0.1
        waiting_angle_threshold=0.3,  # Consider "stable" if angle < 0.3
        penalty_reduction_factor=0.2,  # Reduce penalties to 20% during wait
        patience_bonus=0.005,  # Small bonus per step for stable waiting
    ):
        # Initialize parent
        super().__init__(
            env,
            frame_skip=frame_skip,
            smoothness_coef=smoothness_coef,
            hull_angle_coef=hull_angle_coef,
            hull_angular_vel_coef=hull_angular_vel_coef,
            knee_bend_reward=knee_bend_reward,
            min_bend_threshold=min_bend_threshold,
            max_joint_velocity=max_joint_velocity,
            velocity_penalty=velocity_penalty,
            early_steps_stability_bonus=early_steps_stability_bonus,
            early_steps_count=early_steps_count,
        )

        # Bridge-specific parameters
        self.waiting_velocity_threshold = waiting_velocity_threshold
        self.waiting_angle_threshold = waiting_angle_threshold
        self.penalty_reduction_factor = penalty_reduction_factor
        self.patience_bonus = patience_bonus

        # Tracking
        self.consecutive_waiting_steps = 0

    def reset(self, **kwargs):
        """Reset environment and wrapper state."""
        self.consecutive_waiting_steps = 0
        return super().reset(**kwargs)

    def _is_waiting(self, obs):
        """Detect if agent is in 'waiting' state (for bridges).

        Waiting criteria:
        - Low horizontal velocity (< threshold)
        - Upright and stable (hull angle < threshold)
        - Low angular velocity

        This indicates agent is standing still waiting for bridge.
        """
        horizontal_velocity = abs(obs[2])  # vel_x
        hull_angle = abs(obs[0])  # hull_angle
        hull_angular_vel = abs(obs[1])  # hull_angular_velocity

        is_low_velocity = horizontal_velocity < self.waiting_velocity_threshold
        is_stable = hull_angle < self.waiting_angle_threshold
        is_steady = hull_angular_vel < 0.5

        return is_low_velocity and is_stable and is_steady

    def step(self, action):
        """Execute action with bridge-aware reward modifications."""
        # Get base step from parent (includes all elite hardcore modifications)
        obs, reward, terminated, truncated, info = super().step(action)

        # Check if agent is waiting
        is_waiting = self._is_waiting(obs)

        if is_waiting:
            self.consecutive_waiting_steps += 1

            # CRITICAL FIX: Reduce penalties during waiting
            # Parent class already applied penalties, so we ADD BACK partial amount

            # 1. Reduce smoothness penalty (parent applied -0.2 * diff²)
            if 'smoothness_penalty' in info and info['smoothness_penalty'] > 0:
                smoothness_refund = info['smoothness_penalty'] * (1.0 - self.penalty_reduction_factor)
                reward += smoothness_refund
                info['waiting_smoothness_refund'] = smoothness_refund

            # 2. Reduce hull angle penalty (parent applied -0.1 * angle²)
            if 'hull_angle_penalty' in info and info['hull_angle_penalty'] > 0:
                angle_refund = info['hull_angle_penalty'] * (1.0 - self.penalty_reduction_factor)
                reward += angle_refund
                info['waiting_angle_refund'] = angle_refund

            # 3. Reduce hull angular vel penalty (parent applied -0.05 * vel²)
            if 'hull_angular_vel_penalty' in info and info['hull_angular_vel_penalty'] > 0:
                vel_refund = info['hull_angular_vel_penalty'] * (1.0 - self.penalty_reduction_factor)
                reward += vel_refund
                info['waiting_vel_refund'] = vel_refund

            # 4. Add patience bonus (small reward for stable waiting)
            patience_reward = self.patience_bonus
            reward += patience_reward
            info['patience_bonus'] = patience_reward

            # 5. Track waiting state
            info['is_waiting'] = True
            info['consecutive_waiting_steps'] = self.consecutive_waiting_steps
        else:
            # Not waiting - reset counter
            self.consecutive_waiting_steps = 0
            info['is_waiting'] = False
            info['consecutive_waiting_steps'] = 0

        # Re-clip reward after adjustments
        reward = np.clip(reward, -10.0, 10.0)

        return obs, reward, terminated, truncated, info
