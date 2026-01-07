"""Bridge-Optimized Wrapper - Designed for Custom Walker with Bridges

PHILOSOPHY:
- Bridges require waiting, so we can't heavily penalize standing still
- Instead: Softer base penalties + forward progress as primary reward
- Movement quality through weak constraints, not strong penalties
- Let agent learn that waiting near bridges is worth it long-term

KEY DIFFERENCES FROM ELITE HARDCORE:
- Weaker smoothness penalty (0.05 vs 0.2) - allows standing still
- Weaker hull penalties (0.05, 0.02 vs 0.1, 0.05) - less punishment for waiting
- Forward progress remains main reward driver
- Natural movement from weak knee bending rewards
- No special "waiting detection" - just make waiting less painful overall
"""

import numpy as np
import gymnasium as gym


class BridgeOptimizedWrapper(gym.Wrapper):
    """Wrapper optimized for BipedalWalker with bridge obstacles.

    Design principles:
    1. Softer penalties that don't make waiting prohibitive
    2. Forward progress as primary learning signal
    3. Movement quality through weak positive shaping, not strong penalties
    4. No complex bridge detection - let agent learn naturally
    """

    def __init__(
        self,
        env,
        # Frame skip for decision frequency
        frame_skip=4,

        # SOFT penalties (bridge-compatible)
        smoothness_coef=0.05,           # REDUCED from 0.2 (4x softer)
        hull_angle_coef=0.05,           # REDUCED from 0.1 (2x softer)
        hull_angular_vel_coef=0.02,     # REDUCED from 0.05 (2.5x softer)

        # Movement quality (weak positive shaping)
        knee_bend_reward=0.01,          # Encourage knee flexion
        min_bend_threshold=0.3,         # Min angle to count as "bent"

        # Velocity smoothing (very weak)
        max_joint_velocity=3.0,         # Higher threshold (was 2.0)
        velocity_penalty=0.01,          # REDUCED from 0.02

        # Early stability bonus
        early_steps_stability_bonus=0.01,
        early_steps_count=100,
    ):
        super().__init__(env)

        # Core parameters
        self.frame_skip = frame_skip
        self.smoothness_coef = smoothness_coef
        self.hull_angle_coef = hull_angle_coef
        self.hull_angular_vel_coef = hull_angular_vel_coef

        # Movement quality
        self.knee_bend_reward = knee_bend_reward
        self.min_bend_threshold = min_bend_threshold
        self.max_joint_velocity = max_joint_velocity
        self.velocity_penalty = velocity_penalty

        # Early training
        self.early_steps_stability_bonus = early_steps_stability_bonus
        self.early_steps_count = early_steps_count

        # State tracking
        self.prev_action = None
        self.episode_steps = 0

    def reset(self, **kwargs):
        """Reset environment and wrapper state."""
        self.prev_action = None
        self.episode_steps = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        """Execute action with frame skip and apply reward shaping."""
        # Initialize
        if self.prev_action is None:
            self.prev_action = action

        total_reward = 0.0
        info = {}

        # Execute action with frame skip
        for _ in range(self.frame_skip):
            obs, reward, terminated, truncated, step_info = self.env.step(action)
            total_reward += reward

            if terminated or truncated:
                break

        # Update step counter
        self.episode_steps += 1

        # === REWARD SHAPING ===

        # 1. SOFT Action Smoothness (L2)
        # Encourage smooth actions but don't heavily penalize standing still
        action_diff = action - self.prev_action
        smoothness_penalty = self.smoothness_coef * np.sum(action_diff ** 2)
        total_reward -= smoothness_penalty
        info['smoothness_penalty'] = smoothness_penalty

        # 2. SOFT Hull Stability
        # Keep robot upright but allow some wobble during waiting
        hull_angle = obs[0]
        hull_angular_vel = obs[1]

        hull_angle_penalty = self.hull_angle_coef * (hull_angle ** 2)
        hull_angular_vel_penalty = self.hull_angular_vel_coef * (hull_angular_vel ** 2)

        total_reward -= hull_angle_penalty
        total_reward -= hull_angular_vel_penalty

        info['hull_angle_penalty'] = hull_angle_penalty
        info['hull_angular_vel_penalty'] = hull_angular_vel_penalty

        # 3. Knee Bending Reward (movement quality)
        # Encourage natural leg flexion during swing phase
        # obs[4] = hip1, obs[6] = knee1, obs[8] = hip2, obs[10] = knee2
        # obs[12] = leg1_ground_contact, obs[13] = leg2_ground_contact

        knee_reward = 0.0
        if len(obs) >= 14:
            # Leg 1
            if obs[12] < 0.5:  # Leg not in contact (swing phase)
                knee_angle = abs(obs[6])
                if knee_angle > self.min_bend_threshold:
                    knee_reward += self.knee_bend_reward

            # Leg 2
            if obs[13] < 0.5:  # Leg not in contact (swing phase)
                knee_angle = abs(obs[10])
                if knee_angle > self.min_bend_threshold:
                    knee_reward += self.knee_bend_reward

        total_reward += knee_reward
        info['knee_bend_reward'] = knee_reward

        # 4. Very Weak Joint Velocity Penalty
        # Prevent thrashing but allow quick adjustments
        velocity_penalty_total = 0.0
        if len(obs) >= 11:
            joint_velocities = [obs[5], obs[7], obs[9], obs[11]]  # Joint angular velocities
            for vel in joint_velocities:
                if abs(vel) > self.max_joint_velocity:
                    velocity_penalty_total += self.velocity_penalty * (abs(vel) - self.max_joint_velocity)

        total_reward -= velocity_penalty_total
        info['velocity_penalty'] = velocity_penalty_total

        # 5. Early Stability Bonus (first 100 steps only)
        # Help initial learning
        if self.episode_steps <= self.early_steps_count:
            if abs(hull_angle) < 0.3:  # Reasonably upright
                total_reward += self.early_steps_stability_bonus
                info['early_stability_bonus'] = self.early_steps_stability_bonus

        # Clip final reward
        total_reward = np.clip(total_reward, -10.0, 10.0)

        # Update state
        self.prev_action = action.copy()

        return obs, total_reward, terminated, truncated, info


class BridgeOptimizedWrapperStrict(BridgeOptimizedWrapper):
    """Stricter variant with slightly stronger penalties.

    Use this if the soft version allows too much instability.
    """

    def __init__(self, env, **kwargs):
        # Override with stricter defaults
        kwargs.setdefault('smoothness_coef', 0.08)        # Between soft (0.05) and hardcore (0.2)
        kwargs.setdefault('hull_angle_coef', 0.07)        # Between soft (0.05) and hardcore (0.1)
        kwargs.setdefault('hull_angular_vel_coef', 0.03)  # Between soft (0.02) and hardcore (0.05)

        super().__init__(env, **kwargs)
