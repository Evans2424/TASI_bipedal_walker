"""Bridge Aggressive Wrapper - EXTREME SHAPING for Bridge Learning

AGGRESSIVE STRATEGY:
- HUGE crossing bonus (20.0) - makes waiting worth it
- Strong per-step waiting reward (0.1) - continuous feedback
- Zero penalties when near bridge - remove all discouragement
- Forward velocity bonus when not near bridge - encourage movement
- Progressive rewards for bridge approach and stopping

Goal: Make bridge waiting SO rewarding that agent prioritizes it over everything
"""

import numpy as np
import gymnasium as gym


class BridgeAggressiveWrapper(gym.Wrapper):
    """Extremely aggressive shaping to force bridge learning."""

    def __init__(
        self,
        env,
        frame_skip=4,

        # VERY SOFT base penalties (almost none)
        smoothness_coef=0.01,           # Minimal (was 0.03)
        hull_angle_coef=0.02,           # Minimal (was 0.04)
        hull_angular_vel_coef=0.01,     # Minimal (was 0.02)

        # Movement quality
        knee_bend_reward=0.02,
        min_bend_threshold=0.3,

        # AGGRESSIVE bridge shaping
        stable_waiting_reward=0.1,      # STRONG: +0.1 per step (was 0.01)
        bridge_cross_bonus=20.0,        # HUGE: +20 for crossing (was 5.0)
        bridge_detect_bonus=2.0,        # NEW: +2 for detecting bridge
        bridge_stop_bonus=3.0,          # NEW: +3 for stopping near bridge

        # Forward movement bonus (when not near bridge)
        forward_velocity_bonus=0.5,     # NEW: Encourage continuous movement

        # Anti-exploit
        min_progress_for_bonuses=10.0,  # Lower threshold (was 15.0)
        max_waiting_steps=400,

        # Detection (VERY PERMISSIVE - detect bridges easily)
        lidar_bridge_threshold=1.0,     # More permissive (was 0.5)
        waiting_velocity_threshold=0.2,  # More permissive (was 0.15)
        waiting_angle_threshold=0.4,     # More permissive (was 0.3)
    ):
        super().__init__(env)

        self.frame_skip = frame_skip
        self.smoothness_coef = smoothness_coef
        self.hull_angle_coef = hull_angle_coef
        self.hull_angular_vel_coef = hull_angular_vel_coef

        self.knee_bend_reward = knee_bend_reward
        self.min_bend_threshold = min_bend_threshold

        # Aggressive shaping
        self.stable_waiting_reward = stable_waiting_reward
        self.bridge_cross_bonus = bridge_cross_bonus
        self.bridge_detect_bonus = bridge_detect_bonus
        self.bridge_stop_bonus = bridge_stop_bonus
        self.forward_velocity_bonus = forward_velocity_bonus

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
        self.prev_bridge_detected = False
        self.bridge_stop_given = False  # Track if stop bonus given

    def reset(self, **kwargs):
        self.prev_action = None
        self.episode_steps = 0
        self.total_distance = 0.0
        self.total_waiting_steps = 0
        self.prev_bridge_detected = False
        self.bridge_stop_given = False
        return self.env.reset(**kwargs)

    def _detect_bridge_in_lidar(self, obs):
        """Permissive bridge detection."""
        if len(obs) < 24:
            return False, 10.0

        front_lidar = obs[14:19]
        min_distance = np.min(front_lidar)

        # More permissive: just need 2+ close beams
        close_beams = sum(1 for d in front_lidar if d < self.lidar_bridge_threshold)
        has_progress = self.total_distance > self.min_progress_for_bonuses

        bridge_detected = (close_beams >= 2 and
                          min_distance < self.lidar_bridge_threshold and
                          has_progress)

        return bridge_detected, min_distance

    def _is_stable_waiting(self, obs):
        """Check if waiting stably (permissive)."""
        velocity_x = abs(obs[2])
        hull_angle = abs(obs[0])

        is_slow = velocity_x < self.waiting_velocity_threshold
        is_upright = hull_angle < self.waiting_angle_threshold

        return is_slow and is_upright

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

        # Detect bridge
        bridge_detected, lidar_distance = self._detect_bridge_in_lidar(obs)
        is_stable = self._is_stable_waiting(obs)

        info['bridge_detected'] = bridge_detected
        info['is_stable'] = is_stable
        info['total_distance'] = self.total_distance

        # === BRIDGE PRIORITY: If bridge detected, focus on that ===
        if bridge_detected:
            # ZERO OUT base penalties when bridge detected
            # Agent should focus entirely on bridge, not movement quality
            info['bridge_mode'] = True

            # 1. BRIDGE DETECTION BONUS (first time detecting this bridge)
            if not self.prev_bridge_detected:
                detect_bonus = self.bridge_detect_bonus
                total_reward += detect_bonus
                info['bridge_detect_bonus'] = detect_bonus
                self.bridge_stop_given = False  # Reset for new bridge

            # 2. BRIDGE STOP BONUS (one-time reward for stopping)
            if is_stable and not self.bridge_stop_given:
                stop_bonus = self.bridge_stop_bonus
                total_reward += stop_bonus
                info['bridge_stop_bonus'] = stop_bonus
                self.bridge_stop_given = True

            # 3. STRONG WAITING REWARD (continuous, makes waiting profitable)
            if is_stable and self.total_waiting_steps < self.max_waiting_steps:
                wait_reward = self.stable_waiting_reward
                total_reward += wait_reward
                self.total_waiting_steps += 1
                info['stable_waiting_reward'] = wait_reward
                info['total_waiting_steps'] = self.total_waiting_steps

            # NO penalties when bridge detected - let agent focus on waiting
            action_diff = action - self.prev_action
            smoothness_penalty = 0.0  # ZERO during bridge
            hull_penalty = 0.0  # ZERO during bridge
            info['penalties_disabled'] = True

        else:
            # === NORMAL MOVEMENT: Apply minimal penalties + movement bonuses ===
            info['bridge_mode'] = False

            # Minimal penalties
            action_diff = action - self.prev_action
            smoothness_penalty = self.smoothness_coef * np.sum(action_diff ** 2)
            total_reward -= smoothness_penalty

            hull_angle = obs[0]
            hull_angular_vel = obs[1]
            hull_penalty = (self.hull_angle_coef * (hull_angle ** 2) +
                           self.hull_angular_vel_coef * (hull_angular_vel ** 2))
            total_reward -= hull_penalty

            # FORWARD VELOCITY BONUS (encourage movement when not at bridge)
            velocity_x = obs[2]
            if velocity_x > 0.1:  # Moving forward
                velocity_bonus = self.forward_velocity_bonus * velocity_x
                total_reward += velocity_bonus
                info['velocity_bonus'] = velocity_bonus

            # Reset waiting counter when not at bridge
            self.total_waiting_steps = 0

        # === MOVEMENT QUALITY (always applied) ===
        knee_reward = 0.0
        if len(obs) >= 14:
            if obs[12] < 0.5 and abs(obs[6]) > self.min_bend_threshold:
                knee_reward += self.knee_bend_reward
            if obs[13] < 0.5 and abs(obs[10]) > self.min_bend_threshold:
                knee_reward += self.knee_bend_reward
        total_reward += knee_reward
        info['knee_bend_reward'] = knee_reward

        # === HUGE CROSSING BONUS ===
        if self.prev_bridge_detected and not bridge_detected and self.total_waiting_steps > 20:
            # Successfully crossed bridge!
            cross_bonus = self.bridge_cross_bonus
            total_reward += cross_bonus
            info['bridge_cross_bonus'] = cross_bonus
            info['bridge_crossed'] = True
            self.total_waiting_steps = 0
            self.bridge_stop_given = False

        self.prev_bridge_detected = bridge_detected
        self.prev_action = action.copy()

        total_reward = np.clip(total_reward, -10.0, 30.0)  # Higher ceiling for big bonuses

        return obs, total_reward, terminated, truncated, info
