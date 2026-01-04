"""Elite Hardcore Wrapper: Unified integration of proven hardcore and natural walking.

This wrapper intelligently combines:
- STRONG hardcore features (proven for obstacle navigation)
- WEAK natural walking augmentations (for gait quality)

Design Philosophy:
1. Hardcore features are non-negotiable (frame skip, strong penalties)
2. Natural walking features are augmentations (weak coefficients)
3. Conflicts resolved by choosing stronger proven approach (L2 smoothness 0.2)
4. Forward progress remains primary objective

CRITICAL FIXES (Jan 2025):
V1 → V2 (Anti-Exploit):
- Corrected observation indices to use TRUE hull angle/velocity (obs[0], obs[1])
- Fixed leg contact detection (obs[8], obs[13] instead of wrong obs[6], obs[7])
- Fixed knee angle detection (obs[6], obs[11] instead of wrong obs[9], obs[11])
- Fixed joint velocity monitoring (using speed observations obs[5,7,10,12])
- Ensured reward clipping happens LAST (prevents reward hacking)
- Made knee bending conditional on velocity > 0.1 (prevents standing still)
- Added standing still penalty to enforce forward movement

V2 → V3 (Natural Speed):
- Reduced standing penalty (0.5 → 0.1) to avoid forcing fast movement
- Lowered standing threshold (0.05 → 0.02) for gentler enforcement
- Added maximum velocity penalty (>0.6) to discourage running
- Lowered velocity thresholds for bonuses (0.1 → 0.05) to reward slower walking

V3 → V3.1 (Immediate Start):
- Added time-scaled standing penalty (6x stronger in first steps, 1x at step 50)
- Encourages agent to start moving immediately at episode reset

V3.1 → V3.2 (Truly Slow Walking):
- Lowered max_velocity threshold (0.6 → 0.35) for slow walking
- Increased running penalty strength (0.1 → 0.5) - 5x stronger enforcement
- Target velocity range now 0.2-0.35 instead of 0.3-0.6

V3.2 → V3.3 (Economics Fix - CRITICAL):
- Increased running penalty (0.5 → 5.0) - 10x stronger!
- Root cause: Base env reward (~2.6*velocity) dominated weak penalties
- Solution: Strong penalty makes slow walking economically optimal
- At 0.4 m/s: penalty -0.25 now comparable to base reward difference

V3.3 → V4 (Simplification - Back to Basics):
- REMOVED standing penalty (velocity constraints were solving video bug!)
- REMOVED running penalty (hurt performance for non-problem)
- Made knee bending unconditional (was video bug, not training issue)
- Made early stability unconditional (removed velocity check)
- Focus: Proven hardcore features + quality gait improvements
- Let agent optimize speed naturally for obstacle navigation

References:
- hardcore_wrappers.py: Frame skip, L2 smoothness, hull stability, reward clipping
- smooth_natural_wrapper.py: Knee bending, velocity limits, early stability
- BipedalWalker-v3 official observation space documentation
"""

import numpy as np
import gymnasium as gym


class EliteHardcoreWrapper(gym.Wrapper):
    """Unified wrapper combining proven hardcore with natural walking augmentations.

    Core Features (STRONG - proven for hardcore):
    - Frame skip: 4
    - L2 action smoothness: 0.2
    - Hull stability: angle=0.1, angular_vel=0.05

    Augmentations (WEAK - gait quality):
    - Knee bending reward: 0.02
    - Joint velocity limits: max=2.0, penalty=0.02
    - Early stability bonus: 0.01 for first 100 steps

    Reward clipping (CRITICAL - applied LAST):
    - Clips final reward to [-10, 10]
    - Failure penalty: -10

    IMPORTANT: Reward clipping happens AFTER all modifications to prevent
    reward hacking (agent getting bonuses for "dying with good form")

    Args:
        env: Environment to wrap
        frame_skip: Number of frames to skip (default: 4)
        smoothness_coef: L2 action smoothness penalty (default: 0.2)
        hull_angle_coef: Hull angle penalty (default: 0.1)
        hull_angular_vel_coef: Hull angular velocity penalty (default: 0.05)
        knee_bend_reward: Knee bending reward during swing (default: 0.02)
        min_bend_threshold: Minimum knee bend threshold (default: 0.3)
        max_joint_velocity: Maximum joint velocity before penalty (default: 2.0)
        velocity_penalty: Joint velocity excess penalty (default: 0.02)
        early_steps_stability_bonus: Early stability bonus (default: 0.01)
        early_steps_count: Number of early steps for stability bonus (default: 100)
    """

    def __init__(
        self,
        env,
        # Core hardcore features (STRONG)
        frame_skip=4,
        smoothness_coef=0.2,
        hull_angle_coef=0.1,
        hull_angular_vel_coef=0.05,
        # Natural walking augmentations (WEAK)
        knee_bend_reward=0.02,
        min_bend_threshold=0.3,
        max_joint_velocity=2.0,
        velocity_penalty=0.02,
        early_steps_stability_bonus=0.01,
        early_steps_count=100,
    ):
        # Apply frame skip first (from hardcore)
        env = FrameSkipWrapper(env, skip=frame_skip)

        super().__init__(env)

        # Core hardcore parameters (STRONG)
        self.smoothness_coef = smoothness_coef
        self.hull_angle_coef = hull_angle_coef
        self.hull_angular_vel_coef = hull_angular_vel_coef

        # Natural walking augmentation parameters (WEAK)
        self.knee_bend_reward = knee_bend_reward
        self.min_bend_threshold = min_bend_threshold
        self.max_joint_velocity = max_joint_velocity
        self.velocity_penalty = velocity_penalty
        self.early_steps_stability_bonus = early_steps_stability_bonus
        self.early_steps_count = early_steps_count

        # State tracking
        self.prev_action = None
        self.step_count = 0

    def reset(self, **kwargs):
        """Reset environment and wrapper state."""
        self.prev_action = None
        self.step_count = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        """Execute action and apply unified reward modifications."""
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Track all reward modifications
        modifications = {}

        # ===================================================================
        # CORE HARDCORE FEATURES (STRONG - proven for obstacles)
        # ===================================================================

        # 1. L2 Action Smoothness (STRONG - 0.2)
        # Prevents jerky movements, encourages smooth gaits
        if self.prev_action is not None:
            action_diff = np.array(action) - np.array(self.prev_action)
            smoothness_penalty = self.smoothness_coef * np.sum(action_diff ** 2)
            reward -= smoothness_penalty
            modifications['smoothness_penalty'] = smoothness_penalty
        else:
            modifications['smoothness_penalty'] = 0.0

        # 2. Hull Stability (STRONG - 0.1 angle, 0.05 angular_vel)
        # Maintains upright posture on obstacles
        # CORRECTED: Using obs[0] and obs[1] for TRUE hull angle/velocity
        hull_angle = obs[0]
        hull_angular_vel = obs[1]

        angle_penalty = self.hull_angle_coef * (hull_angle ** 2)
        angular_vel_penalty = self.hull_angular_vel_coef * (hull_angular_vel ** 2)

        reward -= (angle_penalty + angular_vel_penalty)
        modifications['hull_angle_penalty'] = angle_penalty
        modifications['hull_angular_vel_penalty'] = angular_vel_penalty

        # ===================================================================
        # NATURAL WALKING AUGMENTATIONS (WEAK - for gait quality)
        # ===================================================================

        # 3. Knee Bending During Swing (WEAK - 0.02)
        # Helps with obstacle clearance and natural gait
        # CRITICAL: Only reward when MOVING FORWARD to prevent standing still exploit
        # CORRECTED: Using proper contact sensors and knee angles
        leg1_contact = obs[8]   # leg_1_ground_contact
        leg2_contact = obs[13]  # leg_2_ground_contact
        knee1_angle = abs(obs[6])   # knee_joint_1_angle
        knee2_angle = abs(obs[11])  # knee_joint_2_angle
        horizontal_velocity = obs[2]  # vel_x

        knee_bonus = 0.0
        # V4: Simplified - reward knee bending unconditionally for natural gait
        # No velocity check needed (standing still was a video bug, not training issue)
        for leg_contact, knee_angle in [(leg1_contact, knee1_angle), (leg2_contact, knee2_angle)]:
            if leg_contact < 0.5:  # Leg in air (swing phase)
                if knee_angle >= self.min_bend_threshold:
                    knee_bonus += self.knee_bend_reward * min(knee_angle, 1.0)

        reward += knee_bonus
        modifications['knee_bonus'] = knee_bonus

        # 4. Joint Velocity Limits (WEAK - 0.02)
        # Prevents thrashing on uneven terrain
        # CORRECTED: Using actual joint speed observations
        joint_velocities = [abs(obs[5]), abs(obs[7]), abs(obs[10]), abs(obs[12])]
        # obs[5] = hip_joint_1_speed, obs[7] = knee_joint_1_speed
        # obs[10] = hip_joint_2_speed, obs[12] = knee_joint_2_speed

        velocity_excess = 0.0
        for vel in joint_velocities:
            if vel > self.max_joint_velocity:
                velocity_excess += (vel - self.max_joint_velocity)

        if velocity_excess > 0:
            vel_penalty = self.velocity_penalty * velocity_excess
            reward -= vel_penalty
            modifications['velocity_penalty'] = vel_penalty
        else:
            modifications['velocity_penalty'] = 0.0

        # 5. Early Stability Bonus (WEAK - 0.01, time-limited)
        # V4: Simplified - unconditional to help initial learning
        # Faster initial learning, helps with weird first steps
        if self.step_count < self.early_steps_count:
            hull_angle_early = abs(obs[0])
            if hull_angle_early < 0.5:  # Upright
                stability_bonus = self.early_steps_stability_bonus * (1.0 - hull_angle_early)
                reward += stability_bonus
                modifications['stability_bonus'] = stability_bonus
            else:
                modifications['stability_bonus'] = 0.0
        else:
            modifications['stability_bonus'] = 0.0

        # V4: REMOVED velocity constraints (standing/running penalties)
        # Standing still and speed were video recording bugs, not training issues!
        # Let agent optimize velocity naturally for obstacle navigation
        modifications['standing_penalty'] = 0.0
        modifications['running_penalty'] = 0.0

        # ===================================================================
        # REWARD CLIPPING (CRITICAL - MUST BE LAST!)
        # ===================================================================

        # 8. Clip final reward to [-10, 10] and apply failure penalty
        # IMPORTANT: This MUST be done AFTER all modifications to prevent reward hacking
        # If clipped earlier, agent can get bonuses for "dying with good form"
        if hasattr(self.env.unwrapped, 'game_over') and self.env.unwrapped.game_over:
            reward = -10.0
        reward = np.clip(reward, -10.0, 10.0)

        # Update state
        self.prev_action = action.copy() if isinstance(action, np.ndarray) else np.array(action)
        self.step_count += 1

        # Store modifications in info for logging
        info.update(modifications)

        return obs, reward, terminated, truncated, info


class FrameSkipWrapper(gym.Wrapper):
    """Frame skipping wrapper that repeats actions across multiple steps.

    Copied from hardcore_wrappers.py for self-contained implementation.
    """

    def __init__(self, env, skip=4):
        super().__init__(env)
        self.skip = skip

    def step(self, action):
        """Execute action for 'skip' frames and accumulate rewards."""
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
