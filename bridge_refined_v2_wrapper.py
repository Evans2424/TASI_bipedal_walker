"""Bridge Refined V2 Wrapper - MINIMAL Changes to Working Balanced Wrapper

LESSON LEARNED FROM V1:
- Adding 6+ new bonus components broke learning (-78 reward at 7.3M steps)
- Too many objectives = conflicting gradients = no convergence
- The balanced wrapper was WORKING - don't fix what isn't broken

NEW STRATEGY (MINIMAL CHANGES):
1. Keep balanced wrapper reward structure EXACTLY the same
2. Only make 2 small changes:
   a) Slightly stronger hull penalties (better upright posture)
   b) Stricter waiting detection (requires legs closer together)
3. NO new bonus components - just tighten existing criteria

CHANGES FROM BALANCED:
- hull_angle_coef: 0.03 → 0.04 ma(10% stronger upright incentive)
- hull_angular_vel_coef: 0.015 → 0.02 (less wobbling)
- New: waiting requires legs not too spread apart (max_hip_spread check)

PRESERVED FROM BALANCED (ALL WORKING COMPONENTS):
- smoothness_coef: 0.02
- knee_bend_reward: 0.02
- stable_waiting_bonus: 0.02/step
- bridge_cross_bonus: 8.0
- All detection parameters
"""

import numpy as np
import gymnasium as gym


class BridgeRefinedV2Wrapper(gym.Wrapper):
    """Minimal refinement of balanced wrapper - tighter criteria, same reward structure."""

    def __init__(
        self,
        env,
        frame_skip=4,

        # BASE PENALTIES (slightly stronger for better posture)
        smoothness_coef=0.02,           # SAME as balanced
        hull_angle_coef=0.04,           # SLIGHTLY increased from 0.03
        hull_angular_vel_coef=0.02,     # SLIGHTLY increased from 0.015

        # MOVEMENT QUALITY (same as balanced)
        knee_bend_reward=0.02,          # SAME as balanced
        min_bend_threshold=0.3,         # SAME as balanced

        # BRIDGE SHAPING (same as balanced - WORKING)
        stable_waiting_bonus=0.02,      # SAME as balanced
        bridge_cross_bonus=8.0,         # SAME as balanced

        # ANTI-EXPLOIT (same as balanced)
        min_progress_for_bonuses=15.0,
        max_waiting_steps=400,

        # DETECTION (same as balanced)
        lidar_bridge_threshold=0.5,
        min_close_beams=3,
        waiting_velocity_threshold=0.15,
        waiting_angle_threshold=0.3,

        # NEW: Stricter waiting posture (only change to detection)
        max_hip_spread_for_waiting=0.6,  # Legs can't be too spread during waiting
    ):
        super().__init__(env)

        self.frame_skip = frame_skip
        self.smoothness_coef = smoothness_coef
        self.hull_angle_coef = hull_angle_coef
        self.hull_angular_vel_coef = hull_angular_vel_coef

        self.knee_bend_reward = knee_bend_reward
        self.min_bend_threshold = min_bend_threshold

        # Bridge shaping - SAME AS BALANCED
        self.stable_waiting_bonus = stable_waiting_bonus
        self.bridge_cross_bonus = bridge_cross_bonus

        self.min_progress_for_bonuses = min_progress_for_bonuses
        self.max_waiting_steps = max_waiting_steps

        self.lidar_bridge_threshold = lidar_bridge_threshold
        self.min_close_beams = min_close_beams
        self.waiting_velocity_threshold = waiting_velocity_threshold
        self.waiting_angle_threshold = waiting_angle_threshold

        # NEW: Stricter waiting posture requirement
        self.max_hip_spread_for_waiting = max_hip_spread_for_waiting

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
        """STRICT bridge detection - SAME AS BALANCED."""
        if len(obs) < 24:
            return False, 10.0

        front_lidar = obs[14:19]  # 5 front beams
        min_distance = np.min(front_lidar)

        close_beams = sum(1 for d in front_lidar if d < self.lidar_bridge_threshold)
        has_progress = self.total_distance > self.min_progress_for_bonuses

        bridge_detected = (close_beams >= self.min_close_beams and
                          min_distance < self.lidar_bridge_threshold and
                          has_progress)

        return bridge_detected, min_distance

    def _is_stable_waiting(self, obs):
        """Check if stably waiting - ENHANCED with posture check."""
        velocity_x = abs(obs[2])
        hull_angle = abs(obs[0])

        is_slow = velocity_x < self.waiting_velocity_threshold
        is_upright = hull_angle < self.waiting_angle_threshold

        # NEW: Check if legs are not too spread apart
        # obs[4] = hip1 angle, obs[9] = hip2 angle
        hip1_angle = obs[4]
        hip2_angle = obs[9]
        hip_spread = abs(hip1_angle - hip2_angle)
        legs_not_spread = hip_spread < self.max_hip_spread_for_waiting

        return is_slow and is_upright and legs_not_spread

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

        # === BASE PENALTIES (ALWAYS APPLIED) ===
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

        # === MOVEMENT QUALITY (SAME AS BALANCED) ===
        knee_reward = 0.0
        if len(obs) >= 14:
            # Leg 1: ground contact is obs[8]
            if obs[8] < 0.5 and abs(obs[6]) > self.min_bend_threshold:
                knee_reward += self.knee_bend_reward
            # Leg 2: ground contact is obs[13]
            if obs[13] < 0.5 and abs(obs[10]) > self.min_bend_threshold:
                knee_reward += self.knee_bend_reward

        total_reward += knee_reward
        info['knee_bend_reward'] = knee_reward

        # === BRIDGE SHAPING (SAME AS BALANCED) ===
        if bridge_detected:
            info['bridge_mode'] = True

            # WAITING BONUS - now requires better posture via _is_stable_waiting
            if is_stable and self.total_waiting_steps < self.max_waiting_steps:
                wait_bonus = self.stable_waiting_bonus
                total_reward += wait_bonus
                self.total_waiting_steps += 1
                info['stable_waiting_bonus'] = wait_bonus
                info['total_waiting_steps'] = self.total_waiting_steps

        else:
            info['bridge_mode'] = False
            self.total_waiting_steps = 0

        # CROSSING BONUS (SAME AS BALANCED)
        if self.prev_bridge_detected and not bridge_detected and self.total_waiting_steps > 20:
            cross_bonus = self.bridge_cross_bonus
            total_reward += cross_bonus
            info['bridge_cross_bonus'] = cross_bonus
            info['bridge_crossed'] = True
            self.total_waiting_steps = 0

        self.prev_bridge_detected = bridge_detected
        self.prev_action = action.copy()

        # SAME clipping as balanced
        total_reward = np.clip(total_reward, -10.0, 20.0)

        return obs, total_reward, terminated, truncated, info
