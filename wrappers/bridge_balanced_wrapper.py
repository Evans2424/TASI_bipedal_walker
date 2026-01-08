"""Bridge Balanced Wrapper - PROPERLY TUNED for Bridge Learning

ROOT CAUSE ANALYSIS OF PREVIOUS FAILURES:
1. Aggressive wrapper: Reward clipping destroyed crossing bonus signal
2. Fixed wrapper: Bonuses too weak, agent never discovered waiting strategy
3. Optimized wrapper: No bridge-specific shaping at all

NEW STRATEGY:
- MODERATE bonuses that don't need clipping (2-5x base rewards, not 20x)
- SIMPLE reward structure: Just waiting + crossing (no detect/stop/velocity bonuses)
- NO reward normalization (we control scale ourselves)
- KEEP minimal penalties (don't zero them out)
- RELIABLE bridge detection (strict criteria)
- DENSE waiting reward that accumulates to meaningful total

MATH:
- Base forward progress: ~1.0 per terrain section (~50 steps)
- Bridge wait: 300 steps
- Waiting bonus: +0.02/step × 300 = +6.0 total
- Crossing bonus: +8.0 (makes waiting profitable)
- Total bridge reward: ~+14 (equivalent to 14 terrain sections!)
- No clipping needed - rewards stay in reasonable range
"""

import numpy as np
import gymnasium as gym


class BridgeBalancedWrapper(gym.Wrapper):
    """Balanced shaping for bridge learning - fixes all previous issues."""

    def __init__(
        self,
        env,
        frame_skip=4,

        # MODERATE base penalties (not zero, not extreme)
        smoothness_coef=0.02,           # Soft but present (was 0.01→0.03)
        hull_angle_coef=0.03,           # Soft but present (was 0.02→0.04)
        hull_angular_vel_coef=0.015,    # Soft but present (was 0.01→0.02)

        # Movement quality
        knee_bend_reward=0.02,
        min_bend_threshold=0.3,

        # MODERATE bridge shaping (SIMPLE - just 2 bonuses)
        stable_waiting_bonus=0.02,      # +0.02/step × 300 = +6.0 total
        bridge_cross_bonus=8.0,         # +8.0 for crossing (was 5.0→20.0)
        # NO detect bonus, NO stop bonus, NO velocity bonus

        # Anti-exploit (stricter than aggressive)
        min_progress_for_bonuses=15.0,
        max_waiting_steps=400,

        # Detection (STRICT - avoid false positives)
        lidar_bridge_threshold=0.5,     # Stricter (was 1.0)
        min_close_beams=3,              # Need 3+ beams blocked (wide obstacle)
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

        # Bridge shaping - SIMPLE
        self.stable_waiting_bonus = stable_waiting_bonus
        self.bridge_cross_bonus = bridge_cross_bonus

        self.min_progress_for_bonuses = min_progress_for_bonuses
        self.max_waiting_steps = max_waiting_steps

        self.lidar_bridge_threshold = lidar_bridge_threshold
        self.min_close_beams = min_close_beams
        self.waiting_velocity_threshold = waiting_velocity_threshold
        self.waiting_angle_threshold = waiting_angle_threshold

        # State
        self.prev_action = None
        self.episode_steps = 0
        self.total_distance = 0.0
        self.total_waiting_steps = 0
        self.prev_bridge_detected = False

    def reset(self, **kwargs):
        self.prev_action = None
        self.episode_steps = 0
        self.total_distance = 0.0
        self.total_waiting_steps = 0
        self.prev_bridge_detected = False
        return self.env.reset(**kwargs)

    def _detect_bridge_in_lidar(self, obs):
        """STRICT bridge detection to avoid false positives."""
        if len(obs) < 24:
            return False, 10.0

        front_lidar = obs[14:19]  # 5 front beams
        min_distance = np.min(front_lidar)

        # Bridge = wide obstacle (3+ beams blocked) after progress
        close_beams = sum(1 for d in front_lidar if d < self.lidar_bridge_threshold)
        has_progress = self.total_distance > self.min_progress_for_bonuses

        # STRICT: require multiple beams AND progress
        bridge_detected = (close_beams >= self.min_close_beams and
                          min_distance < self.lidar_bridge_threshold and
                          has_progress)

        return bridge_detected, min_distance

    def _is_stable_waiting(self, obs):
        """Check if stably waiting near bridge."""
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

        # === BASE PENALTIES (ALWAYS APPLIED - soft but present) ===
        action_diff = action - self.prev_action
        smoothness_penalty = self.smoothness_coef * np.sum(action_diff ** 2)
        total_reward -= smoothness_penalty

        hull_angle = obs[0]
        hull_angular_vel = obs[1]
        hull_penalty = (self.hull_angle_coef * (hull_angle ** 2) +
                       self.hull_angular_vel_coef * (hull_angular_vel ** 2))
        total_reward -= hull_penalty

        info['smoothness_penalty'] = smoothness_penalty
        info['hull_penalty'] = hull_penalty

        # === MOVEMENT QUALITY ===
        knee_reward = 0.0
        if len(obs) >= 14:
            # Leg 1
            if obs[12] < 0.5 and abs(obs[6]) > self.min_bend_threshold:
                knee_reward += self.knee_bend_reward
            # Leg 2
            if obs[13] < 0.5 and abs(obs[10]) > self.min_bend_threshold:
                knee_reward += self.knee_bend_reward

        total_reward += knee_reward
        info['knee_bend_reward'] = knee_reward

        # === BRIDGE SHAPING (SIMPLE - just 2 bonuses) ===
        if bridge_detected:
            info['bridge_mode'] = True

            # 1. WAITING BONUS (continuous, accumulates to meaningful total)
            if is_stable and self.total_waiting_steps < self.max_waiting_steps:
                wait_bonus = self.stable_waiting_bonus
                total_reward += wait_bonus
                self.total_waiting_steps += 1
                info['stable_waiting_bonus'] = wait_bonus
                info['total_waiting_steps'] = self.total_waiting_steps

        else:
            info['bridge_mode'] = False
            # Reset waiting counter when not at bridge
            self.total_waiting_steps = 0

        # 2. CROSSING BONUS (big reward for success)
        if self.prev_bridge_detected and not bridge_detected and self.total_waiting_steps > 20:
            # Successfully crossed bridge!
            cross_bonus = self.bridge_cross_bonus
            total_reward += cross_bonus
            info['bridge_cross_bonus'] = cross_bonus
            info['bridge_crossed'] = True
            self.total_waiting_steps = 0

        self.prev_bridge_detected = bridge_detected
        self.prev_action = action.copy()

        # CONSERVATIVE clipping (accommodate waiting + crossing)
        # Max possible: ~0.02*300 + 8.0 = ~14.0, so clip at 20 to be safe
        total_reward = np.clip(total_reward, -10.0, 20.0)

        return obs, total_reward, terminated, truncated, info
