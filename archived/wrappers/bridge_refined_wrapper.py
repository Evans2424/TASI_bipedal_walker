"""Bridge Refined Wrapper - IMPROVED Movement Quality & Standing Posture

Building on the WORKING BridgeBalancedWrapper with targeted improvements:

IMPROVEMENTS OVER BALANCED WRAPPER:
1. MOVEMENT QUALITY:
   - Stronger upright posture incentive (hull angle penalty increased)
   - Periodic gait reward (alternating leg motion)
   - Symmetric leg movement bonus
   - Velocity stability reward (smooth forward motion)

2. BRIDGE WAITING POSTURE:
   - Penalize legs spread apart (hip angles too wide)
   - Reward standing upright with legs together
   - Reward slight knee bend for stable stance
   - Stronger hull upright requirement during waiting

OBSERVATION STRUCTURE (BipedalWalker):
- obs[0]: hull angle
- obs[1]: hull angular velocity  
- obs[2]: velocity x
- obs[3]: velocity y
- obs[4]: hip1 angle (joint 0)
- obs[5]: hip1 speed
- obs[6]: knee1 angle + 1.0 (joint 1)
- obs[7]: knee1 speed
- obs[8]: leg1 ground contact
- obs[9]: hip2 angle (joint 2)
- obs[10]: hip2 speed
- obs[11]: knee2 angle + 1.0 (joint 3)
- obs[12]: knee2 speed
- obs[13]: leg2 ground contact
- obs[14-23]: LIDAR (10 beams)

KEY PRINCIPLE: Make small, focused changes to the working balanced setup.
All bonuses remain in reasonable range to avoid clipping issues.
"""

import numpy as np
import gymnasium as gym


class BridgeRefinedWrapper(gym.Wrapper):
    """Refined wrapper with improved movement quality and standing posture."""

    def __init__(
        self,
        env,
        frame_skip=4,

        # === BASE PENALTIES (slightly stronger for better posture) ===
        smoothness_coef=0.02,           # Same as balanced
        hull_angle_coef=0.05,           # INCREASED from 0.03 (stronger upright)
        hull_angular_vel_coef=0.02,     # INCREASED from 0.015 (less wobbling)

        # === MOVEMENT QUALITY (NEW) ===
        # Periodic gait reward
        gait_periodicity_bonus=0.015,   # Reward alternating leg motion
        leg_symmetry_bonus=0.01,        # Reward symmetric leg positions during swing
        
        # Velocity stability
        velocity_stability_bonus=0.01,  # Reward consistent forward velocity
        target_velocity=0.5,            # Target forward velocity (normalized)
        
        # Knee bend during walking (existing, tuned)
        knee_bend_reward=0.015,         # Slightly reduced (was 0.02)
        min_bend_threshold=0.3,

        # === BRIDGE WAITING POSTURE (NEW) ===
        # Standing posture rewards
        standing_upright_bonus=0.03,    # Reward upright hull during waiting
        legs_together_bonus=0.02,       # Reward legs not spread apart
        stable_stance_bonus=0.015,      # Reward slight knee bend for stability
        
        # Posture thresholds
        max_hip_spread=0.4,             # Max allowed hip angle difference for "together"
        target_knee_angle=0.8,          # Target knee angle for stable stance (~slight bend)
        stance_hull_threshold=0.15,     # Stricter hull angle for standing (was 0.3)

        # === BRIDGE SHAPING (same as balanced - WORKING) ===
        stable_waiting_bonus=0.02,      # +0.02/step × 300 = +6.0 total
        bridge_cross_bonus=8.0,         # +8.0 for crossing

        # Anti-exploit (same as balanced)
        min_progress_for_bonuses=15.0,
        max_waiting_steps=400,

        # Detection (same as balanced)
        lidar_bridge_threshold=0.5,
        min_close_beams=3,
        waiting_velocity_threshold=0.15,
        waiting_angle_threshold=0.3,
    ):
        super().__init__(env)

        self.frame_skip = frame_skip
        
        # Base penalties
        self.smoothness_coef = smoothness_coef
        self.hull_angle_coef = hull_angle_coef
        self.hull_angular_vel_coef = hull_angular_vel_coef

        # Movement quality
        self.gait_periodicity_bonus = gait_periodicity_bonus
        self.leg_symmetry_bonus = leg_symmetry_bonus
        self.velocity_stability_bonus = velocity_stability_bonus
        self.target_velocity = target_velocity
        self.knee_bend_reward = knee_bend_reward
        self.min_bend_threshold = min_bend_threshold

        # Bridge waiting posture
        self.standing_upright_bonus = standing_upright_bonus
        self.legs_together_bonus = legs_together_bonus
        self.stable_stance_bonus = stable_stance_bonus
        self.max_hip_spread = max_hip_spread
        self.target_knee_angle = target_knee_angle
        self.stance_hull_threshold = stance_hull_threshold

        # Bridge shaping
        self.stable_waiting_bonus = stable_waiting_bonus
        self.bridge_cross_bonus = bridge_cross_bonus
        self.min_progress_for_bonuses = min_progress_for_bonuses
        self.max_waiting_steps = max_waiting_steps

        # Detection
        self.lidar_bridge_threshold = lidar_bridge_threshold
        self.min_close_beams = min_close_beams
        self.waiting_velocity_threshold = waiting_velocity_threshold
        self.waiting_angle_threshold = waiting_angle_threshold

        # State tracking
        self.prev_action = None
        self.episode_steps = 0
        self.total_distance = 0.0
        self.total_waiting_steps = 0
        self.prev_bridge_detected = False
        
        # Gait tracking (for periodicity)
        self.prev_leg1_contact = False
        self.prev_leg2_contact = False
        self.gait_phase = 0  # Track gait phase changes
        self.prev_velocity_x = 0.0

    def reset(self, **kwargs):
        self.prev_action = None
        self.episode_steps = 0
        self.total_distance = 0.0
        self.total_waiting_steps = 0
        self.prev_bridge_detected = False
        self.prev_leg1_contact = False
        self.prev_leg2_contact = False
        self.gait_phase = 0
        self.prev_velocity_x = 0.0
        return self.env.reset(**kwargs)

    def _detect_bridge_in_lidar(self, obs):
        """STRICT bridge detection to avoid false positives."""
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
        """Check if stably waiting near bridge."""
        velocity_x = abs(obs[2])
        hull_angle = abs(obs[0])

        is_slow = velocity_x < self.waiting_velocity_threshold
        is_upright = hull_angle < self.waiting_angle_threshold

        return is_slow and is_upright

    def _compute_gait_rewards(self, obs):
        """Compute rewards for periodic, natural gait."""
        rewards = {}
        total_gait_reward = 0.0

        leg1_contact = obs[8] > 0.5  # obs[8] is leg1 ground contact
        leg2_contact = obs[13] > 0.5  # obs[13] is leg2 ground contact
        
        # 1. GAIT PERIODICITY: Reward alternating leg contacts
        # Good gait = one leg lifts while other is planted
        contact_changed_leg1 = (leg1_contact != self.prev_leg1_contact)
        contact_changed_leg2 = (leg2_contact != self.prev_leg2_contact)
        
        if contact_changed_leg1 or contact_changed_leg2:
            # Reward phase changes (alternating motion)
            if contact_changed_leg1 != contact_changed_leg2:
                # Only one leg changed - good alternating gait!
                total_gait_reward += self.gait_periodicity_bonus
                rewards['gait_periodicity'] = self.gait_periodicity_bonus
            self.gait_phase += 1
        
        self.prev_leg1_contact = leg1_contact
        self.prev_leg2_contact = leg2_contact

        # 2. LEG SYMMETRY: During swing, legs should mirror each other
        hip1_angle = obs[4]   # Hip 1 angle
        hip2_angle = obs[9]   # Hip 2 angle
        
        # Good walking has opposing hip angles (one forward, one back)
        hip_opposition = hip1_angle * hip2_angle  # Negative = opposing
        if hip_opposition < -0.05:  # Legs are in opposition
            symmetry_reward = min(self.leg_symmetry_bonus, 
                                  self.leg_symmetry_bonus * abs(hip_opposition) * 2)
            total_gait_reward += symmetry_reward
            rewards['leg_symmetry'] = symmetry_reward

        # 3. VELOCITY STABILITY: Reward consistent forward velocity
        velocity_x = obs[2]
        velocity_change = abs(velocity_x - self.prev_velocity_x)
        self.prev_velocity_x = velocity_x
        
        # Reward small velocity changes and velocity near target
        if velocity_x > 0.1:  # Moving forward
            # Penalize velocity fluctuations
            stability = max(0, 1.0 - velocity_change * 5)
            # Reward being near target velocity
            velocity_match = max(0, 1.0 - abs(velocity_x - self.target_velocity) * 2)
            vel_reward = self.velocity_stability_bonus * (stability * 0.5 + velocity_match * 0.5)
            total_gait_reward += vel_reward
            rewards['velocity_stability'] = vel_reward

        return total_gait_reward, rewards

    def _compute_standing_posture_rewards(self, obs):
        """Compute rewards for good standing posture at bridges."""
        rewards = {}
        total_posture_reward = 0.0

        hull_angle = abs(obs[0])
        hip1_angle = obs[4]
        hip2_angle = obs[9]
        knee1_angle = obs[6]  # Already has +1.0 offset
        knee2_angle = obs[11]  # Already has +1.0 offset

        # 1. UPRIGHT HULL: Stricter requirement for standing
        if hull_angle < self.stance_hull_threshold:
            # More upright = more reward
            upright_factor = 1.0 - (hull_angle / self.stance_hull_threshold)
            upright_reward = self.standing_upright_bonus * upright_factor
            total_posture_reward += upright_reward
            rewards['standing_upright'] = upright_reward

        # 2. LEGS TOGETHER: Penalize wide hip spread
        hip_spread = abs(hip1_angle - hip2_angle)
        if hip_spread < self.max_hip_spread:
            # Legs closer together = more reward
            together_factor = 1.0 - (hip_spread / self.max_hip_spread)
            together_reward = self.legs_together_bonus * together_factor
            total_posture_reward += together_reward
            rewards['legs_together'] = together_reward

        # 3. STABLE STANCE: Slight knee bend for stability
        # Target is around 0.8 (slight bend), not fully extended (1.0) or fully bent (0.0)
        knee1_error = abs(knee1_angle - self.target_knee_angle)
        knee2_error = abs(knee2_angle - self.target_knee_angle)
        avg_knee_error = (knee1_error + knee2_error) / 2
        
        if avg_knee_error < 0.5:  # Reasonably close to target
            stance_factor = 1.0 - (avg_knee_error / 0.5)
            stance_reward = self.stable_stance_bonus * stance_factor
            total_posture_reward += stance_reward
            rewards['stable_stance'] = stance_reward

        return total_posture_reward, rewards

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

        # === BASE PENALTIES (ALWAYS APPLIED - slightly stronger) ===
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

        # === MOVEMENT QUALITY (NEW - only when NOT at bridge) ===
        if not bridge_detected:
            # Gait rewards only during walking
            gait_reward, gait_info = self._compute_gait_rewards(obs)
            total_reward += gait_reward
            info.update(gait_info)
            info['total_gait_reward'] = gait_reward

            # Knee bend during walking
            knee_reward = 0.0
            if len(obs) >= 14:
                # Leg 1: reward knee bend when foot not on ground
                if obs[8] < 0.5 and abs(obs[6] - 1.0) > self.min_bend_threshold:
                    knee_reward += self.knee_bend_reward
                # Leg 2: reward knee bend when foot not on ground
                if obs[13] < 0.5 and abs(obs[11] - 1.0) > self.min_bend_threshold:
                    knee_reward += self.knee_bend_reward

            total_reward += knee_reward
            info['knee_bend_reward'] = knee_reward

        # === BRIDGE SHAPING ===
        if bridge_detected:
            info['bridge_mode'] = True

            # 1. BASE WAITING BONUS (same as balanced - WORKING)
            if is_stable and self.total_waiting_steps < self.max_waiting_steps:
                wait_bonus = self.stable_waiting_bonus
                total_reward += wait_bonus
                self.total_waiting_steps += 1
                info['stable_waiting_bonus'] = wait_bonus
                info['total_waiting_steps'] = self.total_waiting_steps

            # 2. STANDING POSTURE REWARDS (NEW - during bridge waiting)
            posture_reward, posture_info = self._compute_standing_posture_rewards(obs)
            total_reward += posture_reward
            info.update(posture_info)
            info['total_posture_reward'] = posture_reward

        else:
            info['bridge_mode'] = False
            self.total_waiting_steps = 0

        # 3. CROSSING BONUS (same as balanced - WORKING)
        if self.prev_bridge_detected and not bridge_detected and self.total_waiting_steps > 20:
            cross_bonus = self.bridge_cross_bonus
            total_reward += cross_bonus
            info['bridge_cross_bonus'] = cross_bonus
            info['bridge_crossed'] = True
            self.total_waiting_steps = 0

        self.prev_bridge_detected = bridge_detected
        self.prev_action = action.copy()

        # Conservative clipping (same as balanced)
        # Max theoretical: ~6 (waiting) + 8 (cross) + ~2 (posture) = ~16, well under 20
        total_reward = np.clip(total_reward, -10.0, 20.0)

        return obs, total_reward, terminated, truncated, info
