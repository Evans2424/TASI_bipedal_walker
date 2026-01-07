"""Bridge-Shaped Wrapper - Intelligent Reward Shaping for Bridges

PHILOSOPHY:
- Bridges are detectable in LIDAR (raised bridge = obstacle in LIDAR)
- Give IMMEDIATE positive reward for correct bridge behavior
- Solve the delayed reward problem with dense shaping
- Maintain natural movement quality throughout

KEY FEATURES:
1. Bridge Detection: Monitor LIDAR for bridge obstacles
2. Approach Reward: Slow down when bridge detected
3. Waiting Reward: Positive reward for stable waiting near bridge
4. Crossing Reward: Bonus for successfully crossing after wait
5. Movement Quality: Maintain natural gait throughout
"""

import numpy as np
import gymnasium as gym


class BridgeShapedWrapper(gym.Wrapper):
    """Intelligently shaped rewards for bridge navigation + natural movement.

    Uses LIDAR to detect bridges and provides immediate rewards for:
    - Approaching bridges cautiously
    - Stopping and waiting near bridges
    - Maintaining balance during wait
    - Crossing after bridge lowers

    Also maintains movement quality rewards for natural walking.
    """

    def __init__(
        self,
        env,
        frame_skip=4,

        # SOFT base penalties (bridge-compatible)
        smoothness_coef=0.03,           # Very soft (was 0.05)
        hull_angle_coef=0.04,           # Very soft (was 0.05)
        hull_angular_vel_coef=0.02,     # Very soft

        # Movement quality
        knee_bend_reward=0.015,         # Natural leg movement
        min_bend_threshold=0.3,

        # Bridge detection and shaping
        lidar_bridge_threshold=0.8,     # LIDAR distance indicating obstacle
        bridge_approach_distance=5.0,   # Distance to start slowing

        # Bridge behavior rewards (STRONG - makes waiting worthwhile)
        cautious_approach_bonus=0.02,   # Reward for slowing near bridge
        stable_waiting_bonus=0.03,      # Reward per step for stable waiting
        bridge_cross_bonus=2.0,         # Large bonus for successful crossing

        # Waiting criteria
        waiting_velocity_threshold=0.15,
        waiting_angle_threshold=0.3,
    ):
        super().__init__(env)

        # Frame skip
        self.frame_skip = frame_skip

        # Base penalties (very soft)
        self.smoothness_coef = smoothness_coef
        self.hull_angle_coef = hull_angle_coef
        self.hull_angular_vel_coef = hull_angular_vel_coef

        # Movement quality
        self.knee_bend_reward = knee_bend_reward
        self.min_bend_threshold = min_bend_threshold

        # Bridge detection
        self.lidar_bridge_threshold = lidar_bridge_threshold
        self.bridge_approach_distance = bridge_approach_distance

        # Bridge shaping rewards
        self.cautious_approach_bonus = cautious_approach_bonus
        self.stable_waiting_bonus = stable_waiting_bonus
        self.bridge_cross_bonus = bridge_cross_bonus

        # Waiting criteria
        self.waiting_velocity_threshold = waiting_velocity_threshold
        self.waiting_angle_threshold = waiting_angle_threshold

        # State tracking
        self.prev_action = None
        self.episode_steps = 0
        self.prev_lidar_detected_bridge = False
        self.waiting_near_bridge_steps = 0
        self.total_distance = 0.0

    def reset(self, **kwargs):
        """Reset environment and wrapper state."""
        self.prev_action = None
        self.episode_steps = 0
        self.prev_lidar_detected_bridge = False
        self.waiting_near_bridge_steps = 0
        self.total_distance = 0.0
        return self.env.reset(**kwargs)

    def _detect_bridge_in_lidar(self, obs):
        """Detect if there's a bridge obstacle ahead using LIDAR.

        LIDAR readings are obs[14:24] (10 beams).
        A raised bridge appears as a close obstacle in front beams.

        Returns:
            bridge_detected (bool): Whether bridge is detected
            min_distance (float): Minimum LIDAR distance (estimate to obstacle)
        """
        if len(obs) < 24:
            return False, 10.0

        # Front-facing LIDAR beams (indices 14-18, center beams)
        front_lidar = obs[14:19]

        # Check if any front beam detects close obstacle
        min_distance = np.min(front_lidar)

        # Bridge detected if obstacle within threshold
        bridge_detected = min_distance < self.lidar_bridge_threshold

        return bridge_detected, min_distance

    def _is_stable_waiting(self, obs):
        """Check if agent is in stable waiting position."""
        velocity_x = abs(obs[2])
        hull_angle = abs(obs[0])
        hull_angular_vel = abs(obs[1])

        is_slow = velocity_x < self.waiting_velocity_threshold
        is_upright = hull_angle < self.waiting_angle_threshold
        is_steady = hull_angular_vel < 0.5

        return is_slow and is_upright and is_steady

    def step(self, action):
        """Execute action with bridge-aware reward shaping."""
        # Initialize
        if self.prev_action is None:
            self.prev_action = action

        total_reward = 0.0
        info = {}

        # Execute with frame skip
        for _ in range(self.frame_skip):
            obs, reward, terminated, truncated, step_info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break

        self.episode_steps += 1

        # Track approximate forward distance
        self.total_distance += obs[2] * self.frame_skip

        # === BASE PENALTIES (VERY SOFT) ===

        # 1. Smoothness (very soft)
        action_diff = action - self.prev_action
        smoothness_penalty = self.smoothness_coef * np.sum(action_diff ** 2)
        total_reward -= smoothness_penalty
        info['smoothness_penalty'] = smoothness_penalty

        # 2. Hull stability (very soft)
        hull_angle = obs[0]
        hull_angular_vel = obs[1]

        hull_angle_penalty = self.hull_angle_coef * (hull_angle ** 2)
        hull_angular_vel_penalty = self.hull_angular_vel_coef * (hull_angular_vel ** 2)

        total_reward -= hull_angle_penalty
        total_reward -= hull_angular_vel_penalty

        info['hull_angle_penalty'] = hull_angle_penalty
        info['hull_angular_vel_penalty'] = hull_angular_vel_penalty

        # === MOVEMENT QUALITY ===

        # 3. Knee bending (natural gait)
        knee_reward = 0.0
        if len(obs) >= 14:
            # Leg 1
            if obs[12] < 0.5:  # Swing phase
                if abs(obs[6]) > self.min_bend_threshold:
                    knee_reward += self.knee_bend_reward
            # Leg 2
            if obs[13] < 0.5:  # Swing phase
                if abs(obs[10]) > self.min_bend_threshold:
                    knee_reward += self.knee_bend_reward

        total_reward += knee_reward
        info['knee_bend_reward'] = knee_reward

        # === BRIDGE DETECTION AND SHAPING ===

        bridge_detected, lidar_distance = self._detect_bridge_in_lidar(obs)
        is_stable = self._is_stable_waiting(obs)

        info['bridge_detected'] = bridge_detected
        info['lidar_distance'] = lidar_distance
        info['is_stable'] = is_stable

        # 4. CAUTIOUS APPROACH BONUS
        # Reward for slowing down when bridge is detected ahead
        if bridge_detected and not self.prev_lidar_detected_bridge:
            # Just detected bridge - encourage slowing down
            velocity = abs(obs[2])
            if velocity < 0.5:  # Moving slowly
                approach_bonus = self.cautious_approach_bonus
                total_reward += approach_bonus
                info['cautious_approach_bonus'] = approach_bonus

        # 5. STABLE WAITING BONUS
        # Strong positive reward for waiting stably near bridge
        if bridge_detected and is_stable:
            # Agent is waiting near bridge - GOOD!
            wait_bonus = self.stable_waiting_bonus
            total_reward += wait_bonus
            self.waiting_near_bridge_steps += 1
            info['stable_waiting_bonus'] = wait_bonus
            info['waiting_steps'] = self.waiting_near_bridge_steps
        else:
            self.waiting_near_bridge_steps = 0

        # 6. BRIDGE CROSSING BONUS
        # Large reward when bridge clears (LIDAR opens up after waiting)
        if (self.prev_lidar_detected_bridge and not bridge_detected and
            self.waiting_near_bridge_steps > 50):  # Waited at least 50 steps
            # Bridge has lowered! Give big bonus for successful wait
            cross_bonus = self.bridge_cross_bonus
            total_reward += cross_bonus
            info['bridge_cross_bonus'] = cross_bonus
            info['bridge_crossed'] = True
            self.waiting_near_bridge_steps = 0

        # Update state
        self.prev_lidar_detected_bridge = bridge_detected
        self.prev_action = action.copy()

        # Clip final reward
        total_reward = np.clip(total_reward, -10.0, 10.0)

        return obs, total_reward, terminated, truncated, info


class BridgeShapedWrapperAggressive(BridgeShapedWrapper):
    """More aggressive bridge shaping for faster learning.

    Use this variant if agent struggles to learn bridge behavior.
    """

    def __init__(self, env, **kwargs):
        # Override with more aggressive shaping
        kwargs.setdefault('stable_waiting_bonus', 0.05)      # Stronger (was 0.03)
        kwargs.setdefault('bridge_cross_bonus', 5.0)         # Much stronger (was 2.0)
        kwargs.setdefault('cautious_approach_bonus', 0.03)   # Stronger (was 0.02)

        super().__init__(env, **kwargs)
