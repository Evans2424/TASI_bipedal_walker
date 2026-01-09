"""Elite Hardcore Wrapper - BRIDGE AWARE V2 (Anti-Exploit)

VERSION 2 FIXES:
- Only applies waiting logic when forward progress made (x > 10 units)
- Requires consecutive stable frames (not just one frame)
- Removed patience bonus (was being exploited)
- Stricter waiting criteria

This prevents agent from exploiting waiting detection at episode start.
"""

import numpy as np
import gymnasium as gym
from wrappers.elite_hardcore_wrapper import EliteHardcoreWrapper


class EliteHardcoreBridgeWrapperV2(EliteHardcoreWrapper):
    """Anti-exploit version of bridge-aware wrapper.

    Key changes from V1:
    - Waiting only counts after forward progress (x > min_progress)
    - Requires consecutive stable frames (min_consecutive_frames)
    - No patience bonus (removed exploit)
    - Stricter velocity/angle thresholds
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
        # Bridge-specific parameters (V2 - ANTI-EXPLOIT)
        waiting_velocity_threshold=0.05,  # Stricter: vel < 0.05 (was 0.1)
        waiting_angle_threshold=0.2,  # Stricter: angle < 0.2 (was 0.3)
        penalty_reduction_factor=0.2,  # Keep 80% reduction
        min_progress_for_waiting=10.0,  # NEW: Must reach x=10 before waiting applies
        min_consecutive_frames=8,  # NEW: Must be stable for 8 frames (2 seconds)
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

        # Bridge-specific parameters (V2)
        self.waiting_velocity_threshold = waiting_velocity_threshold
        self.waiting_angle_threshold = waiting_angle_threshold
        self.penalty_reduction_factor = penalty_reduction_factor
        self.min_progress_for_waiting = min_progress_for_waiting
        self.min_consecutive_frames = min_consecutive_frames

        # Tracking
        self.consecutive_waiting_steps = 0
        self.total_distance = 0.0  # Track forward progress

    def reset(self, **kwargs):
        """Reset environment and wrapper state."""
        self.consecutive_waiting_steps = 0
        self.total_distance = 0.0
        return super().reset(**kwargs)

    def _is_waiting(self, obs):
        """Detect if agent is legitimately waiting for bridge.

        V2 Criteria (anti-exploit):
        1. Low horizontal velocity (< threshold)
        2. Upright and stable (hull angle < threshold)
        3. Low angular velocity
        4. Has made forward progress (x > min_progress) ← NEW
        5. Stable for multiple consecutive frames ← NEW

        This prevents exploitation at episode start.
        """
        horizontal_velocity = abs(obs[2])  # vel_x
        hull_angle = abs(obs[0])  # hull_angle
        hull_angular_vel = abs(obs[1])  # hull_angular_velocity

        # Basic waiting conditions (stricter thresholds)
        is_low_velocity = horizontal_velocity < self.waiting_velocity_threshold
        is_stable = hull_angle < self.waiting_angle_threshold
        is_steady = hull_angular_vel < 0.5

        # NEW: Forward progress requirement
        has_progressed = self.total_distance > self.min_progress_for_waiting

        # All conditions must be met
        return is_low_velocity and is_stable and is_steady and has_progressed

    def step(self, action):
        """Execute action with bridge-aware reward modifications (V2)."""
        # Get base step from parent (includes all elite hardcore modifications)
        obs, reward, terminated, truncated, info = super().step(action)

        # Track forward progress (from hull_x position if available)
        # BipedalWalker obs[2] is vel_x, we integrate to get approximate position
        # More reliable: use info dict if env provides hull_x
        if 'x_position' in info:
            self.total_distance = info['x_position']
        else:
            # Approximate from velocity
            self.total_distance += obs[2] * 4  # frame_skip=4

        # Check if agent meets waiting criteria
        is_currently_stable = self._is_waiting(obs)

        if is_currently_stable:
            self.consecutive_waiting_steps += 1
        else:
            self.consecutive_waiting_steps = 0

        # Only apply waiting bonus if stable for minimum consecutive frames
        is_waiting = self.consecutive_waiting_steps >= self.min_consecutive_frames

        if is_waiting:
            # CRITICAL FIX: Reduce penalties during legitimate waiting
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

            # NO PATIENCE BONUS - Removed to prevent exploitation

            # Track waiting state
            info['is_waiting'] = True
            info['consecutive_waiting_steps'] = self.consecutive_waiting_steps
            info['total_distance'] = self.total_distance
        else:
            info['is_waiting'] = False
            info['consecutive_waiting_steps'] = 0
            info['total_distance'] = self.total_distance

        # Re-clip reward after adjustments
        reward = np.clip(reward, -10.0, 10.0)

        return obs, reward, terminated, truncated, info
