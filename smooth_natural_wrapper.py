"""Enhanced wrapper for smooth, natural walking with consistent periodic movement."""

import numpy as np
import gymnasium as gym


class SmoothNaturalWalking(gym.Wrapper):
    """Add incentives for smooth, natural, periodic walking.

    Features:
    1. Knee bending during swing (subtle)
    2. Action smoothness (prevent jerky movements)
    3. Joint velocity limits (prevent legs moving too fast)
    4. Early stability bonus (help with initial steps)

    All bonuses are WEAK to avoid reward gaming - walking forward remains primary.
    """

    def __init__(
        self,
        env,
        # Knee bending (from before)
        knee_bend_reward=0.02,
        min_bend_threshold=0.3,
        # Action smoothness
        action_smoothness_penalty=0.05,
        # Joint velocity penalty
        max_joint_velocity=2.0,
        velocity_penalty=0.02,
        # Early stability
        early_steps_stability_bonus=0.01,
        early_steps_count=100,
    ):
        super().__init__(env)

        # Knee bending params
        self.knee_bend_reward = knee_bend_reward
        self.min_bend_threshold = min_bend_threshold

        # Action smoothness params
        self.action_smoothness_penalty = action_smoothness_penalty
        self.prev_action = None

        # Joint velocity params
        self.max_joint_velocity = max_joint_velocity
        self.velocity_penalty = velocity_penalty

        # Early stability params
        self.early_steps_stability_bonus = early_steps_stability_bonus
        self.early_steps_count = early_steps_count
        self.step_count = 0

    def reset(self, **kwargs):
        """Reset wrapper state."""
        self.prev_action = None
        self.step_count = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        modification = 0.0

        # 1. Knee bending reward (subtle - from before)
        leg1_contact = obs[6]
        leg2_contact = obs[7]
        knee1_angle = abs(obs[9])
        knee2_angle = abs(obs[11])

        knee_bonus = 0.0
        for leg_contact, knee_angle in [(leg1_contact, knee1_angle), (leg2_contact, knee2_angle)]:
            if leg_contact < 0.5:  # Leg in air (swing phase)
                knee_bonus += self.knee_bend_reward * min(knee_angle, 1.0)

        modification += knee_bonus

        # 2. Action smoothness penalty (prevent jerky movements)
        if self.prev_action is not None:
            action_change = np.abs(action - self.prev_action)
            avg_action_change = np.mean(action_change)
            smoothness_penalty = self.action_smoothness_penalty * avg_action_change
            modification -= smoothness_penalty
            info['smoothness_penalty'] = smoothness_penalty
        else:
            info['smoothness_penalty'] = 0.0

        self.prev_action = action.copy()

        # 3. Joint velocity penalty (prevent legs moving too fast)
        # obs[4], [5] = leg1 joint velocities
        # obs[8], [10] = leg2 joint velocities
        joint_velocities = [abs(obs[4]), abs(obs[5]), abs(obs[8]), abs(obs[10])]

        velocity_excess = 0.0
        for vel in joint_velocities:
            if vel > self.max_joint_velocity:
                velocity_excess += (vel - self.max_joint_velocity)

        if velocity_excess > 0:
            vel_penalty = self.velocity_penalty * velocity_excess
            modification -= vel_penalty
            info['velocity_penalty'] = vel_penalty
        else:
            info['velocity_penalty'] = 0.0

        # 4. Early stability bonus (help with initial weird steps)
        if self.step_count < self.early_steps_count:
            # Bonus for keeping hull angle stable (obs[0] is hull angle)
            hull_angle = abs(obs[0])
            if hull_angle < 0.5:  # Hull relatively upright
                stability_bonus = self.early_steps_stability_bonus * (1.0 - hull_angle)
                modification += stability_bonus
                info['stability_bonus'] = stability_bonus
            else:
                info['stability_bonus'] = 0.0
        else:
            info['stability_bonus'] = 0.0

        self.step_count += 1

        # Add all modifications to base reward
        reward += modification

        info['total_modification'] = modification
        info['knee_bonus'] = knee_bonus

        return obs, reward, terminated, truncated, info
