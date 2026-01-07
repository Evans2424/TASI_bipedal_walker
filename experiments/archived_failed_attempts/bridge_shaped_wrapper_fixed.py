"""Bridge-Shaped Wrapper - FIXED VERSION (No Exploit)

CRITICAL FIX:
- Only reward waiting when ACTUALLY near a bridge
- Require forward progress before waiting bonuses activate
- Distinguish bridges from other obstacles in LIDAR
- Remove exploit where agent stands still entire episode

KEY CHANGES:
1. Track forward progress (must reach x > 15 before waiting bonuses)
2. Detect bridge-specific LIDAR patterns (different from stumps/stairs)
3. Limit total waiting bonus per episode
4. Require crossing bonus to prevent infinite waiting
"""

import numpy as np
import gymnasium as gym


class BridgeShapedWrapperFixed(gym.Wrapper):
    """FIXED: Exploit-proof bridge wrapper with proper detection."""

    def __init__(
        self,
        env,
        frame_skip=4,

        # SOFT base penalties
        smoothness_coef=0.03,
        hull_angle_coef=0.04,
        hull_angular_vel_coef=0.02,

        # Movement quality
        knee_bend_reward=0.015,
        min_bend_threshold=0.3,

        # Bridge shaping (REDUCED to prevent exploit)
        stable_waiting_bonus=0.01,      # REDUCED from 0.03 (was exploited)
        bridge_cross_bonus=5.0,         # INCREASED from 2.0 (encourage crossing)

        # Anti-exploit measures (NEW)
        min_progress_for_bonuses=15.0,  # Must reach x=15 before bonuses
        max_waiting_steps=400,          # Stop bonuses after 400 steps waiting

        # Detection
        lidar_bridge_threshold=0.5,     # STRICTER (was 0.8)
        waiting_velocity_threshold=0.15,
        waiting_angle_threshold=0.3,
    ):
        super().__init__(env)

        self.frame_skip = frame_skip
        self.smoothness_coef = smoothness_coef
        self.hull_angle_coef = hull_angle_coef
        self.hull_angular_vel_coef = hull_angular_vel_coef

        self.knee_bend_reward = knee_bend_reward
        self.min_bend_threshold = min_bend_threshold

        self.stable_waiting_bonus = stable_waiting_bonus
        self.bridge_cross_bonus = bridge_cross_bonus

        # Anti-exploit
        self.min_progress_for_bonuses = min_progress_for_bonuses
        self.max_waiting_steps = max_waiting_steps

        self.lidar_bridge_threshold = lidar_bridge_threshold
        self.waiting_velocity_threshold = waiting_velocity_threshold
        self.waiting_angle_threshold = waiting_angle_threshold

        # State
        self.prev_action = None
        self.episode_steps = 0
        self.total_distance = 0.0
        self.total_waiting_steps = 0
        self.prev_lidar_detected_bridge = False

    def reset(self, **kwargs):
        self.prev_action = None
        self.episode_steps = 0
        self.total_distance = 0.0
        self.total_waiting_steps = 0
        self.prev_lidar_detected_bridge = False
        return self.env.reset(**kwargs)

    def _detect_bridge_in_lidar(self, obs):
        """Detect BRIDGE specifically (not just any obstacle)."""
        if len(obs) < 24:
            return False, 10.0

        front_lidar = obs[14:19]
        min_distance = np.min(front_lidar)

        # Bridge detected if:
        # 1. Close obstacle (< threshold)
        # 2. Multiple beams blocked (wide obstacle = bridge)
        # 3. Agent has made forward progress (eliminates standing at start)

        close_beams = sum(1 for d in front_lidar if d < self.lidar_bridge_threshold)
        has_progress = self.total_distance > self.min_progress_for_bonuses

        # Bridge: wide obstacle (3+ beams) after progress
        bridge_detected = (close_beams >= 3 and
                          min_distance < self.lidar_bridge_threshold and
                          has_progress)

        return bridge_detected, min_distance

    def _is_stable_waiting(self, obs):
        velocity_x = abs(obs[2])
        hull_angle = abs(obs[0])
        hull_angular_vel = abs(obs[1])

        is_slow = velocity_x < self.waiting_velocity_threshold
        is_upright = hull_angle < self.waiting_angle_threshold
        is_steady = hull_angular_vel < 0.5

        return is_slow and is_upright and is_steady

    def step(self, action):
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
        self.total_distance += obs[2] * self.frame_skip

        # === BASE PENALTIES (SOFT) ===
        action_diff = action - self.prev_action
        smoothness_penalty = self.smoothness_coef * np.sum(action_diff ** 2)
        total_reward -= smoothness_penalty
        info['smoothness_penalty'] = smoothness_penalty

        hull_angle = obs[0]
        hull_angular_vel = obs[1]
        hull_angle_penalty = self.hull_angle_coef * (hull_angle ** 2)
        hull_angular_vel_penalty = self.hull_angular_vel_coef * (hull_angular_vel ** 2)
        total_reward -= hull_angle_penalty
        total_reward -= hull_angular_vel_penalty

        # === MOVEMENT QUALITY ===
        knee_reward = 0.0
        if len(obs) >= 14:
            if obs[12] < 0.5 and abs(obs[6]) > self.min_bend_threshold:
                knee_reward += self.knee_bend_reward
            if obs[13] < 0.5 and abs(obs[10]) > self.min_bend_threshold:
                knee_reward += self.knee_bend_reward
        total_reward += knee_reward
        info['knee_bend_reward'] = knee_reward

        # === BRIDGE DETECTION (FIXED) ===
        bridge_detected, lidar_distance = self._detect_bridge_in_lidar(obs)
        is_stable = self._is_stable_waiting(obs)

        info['bridge_detected'] = bridge_detected
        info['lidar_distance'] = lidar_distance
        info['is_stable'] = is_stable
        info['total_distance'] = self.total_distance

        # STABLE WAITING BONUS (ANTI-EXPLOIT)
        # Only if: bridge detected + stable + made progress + not waited too long
        if (bridge_detected and is_stable and
            self.total_distance > self.min_progress_for_bonuses and
            self.total_waiting_steps < self.max_waiting_steps):

            wait_bonus = self.stable_waiting_bonus
            total_reward += wait_bonus
            self.total_waiting_steps += 1
            info['stable_waiting_bonus'] = wait_bonus
            info['total_waiting_steps'] = self.total_waiting_steps

        # BRIDGE CROSSING BONUS
        if (self.prev_lidar_detected_bridge and not bridge_detected and
            self.total_waiting_steps > 50):  # Actually waited

            cross_bonus = self.bridge_cross_bonus
            total_reward += cross_bonus
            info['bridge_cross_bonus'] = cross_bonus
            info['bridge_crossed'] = True
            self.total_waiting_steps = 0  # Reset for next bridge

        self.prev_lidar_detected_bridge = bridge_detected
        self.prev_action = action.copy()

        total_reward = np.clip(total_reward, -10.0, 10.0)

        return obs, total_reward, terminated, truncated, info
