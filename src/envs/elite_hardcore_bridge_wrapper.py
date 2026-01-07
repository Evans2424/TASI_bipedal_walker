"""Elite Hardcore Wrapper - BRIDGE AWARE VERSION

This wrapper extends the reward shaping to handle BRIDGE obstacles in custom_walker.py.

KEY FIX: Reduces penalties when agent is "waiting" for bridges to lower.

Bridge behavior:
- Bridges activate when robot within 10 units
- Wait 300 steps (6 seconds) before lowering
- Agent must stand still and maintain balance

Problem with standard reward shaping:
- Penalizes standing still (smoothness, hull stability)
- No forward progress reward while waiting
- 300 steps of waiting = -30 to -50 reward!

Solution:
- Detect "waiting" state: Low velocity + upright + stable
- Reduce penalties by 80% during wait
- Add small "patience" bonus for maintaining position

This allows bridges to work while maintaining hardcore obstacle-solving ability.
"""

import numpy as np
import gymnasium as gym


class EliteHardcoreBridgeWrapper(gym.Wrapper):
    """Bridge-aware version of reward shaping wrapper.

    Extends base wrapper with bridge-specific handling:
    - Detects when agent is "waiting" (low velocity + stable)
    - Reduces penalties during wait period
    - Adds patience bonus for maintaining position
    """

    def __init__(
        self,
        env,
        # Core hardcore features
        frame_skip=4,
        smoothness_coef=0.2,
        hull_angle_coef=0.1,
        hull_angular_vel_coef=0.05,
        # Natural walking augmentations
        knee_bend_reward=0.02,
        min_bend_threshold=0.3,
        max_joint_velocity=2.0,
        velocity_penalty=0.02,
        early_steps_stability_bonus=0.01,
        early_steps_count=100,
        # Bridge-specific parameters
        waiting_velocity_threshold=0.1,
        waiting_angle_threshold=0.3,
        waiting_angular_vel_threshold=0.5,
        penalty_reduction_factor=0.2,
        patience_bonus=0.005,
    ):
        """Initialize the wrapper.

        Args:
            env: Environment to wrap
            frame_skip: Number of frames to skip
            smoothness_coef: Smoothness penalty coefficient
            hull_angle_coef: Hull angle penalty coefficient
            hull_angular_vel_coef: Hull angular velocity penalty coefficient
            knee_bend_reward: Reward for knee bending
            min_bend_threshold: Minimum knee bend for reward
            max_joint_velocity: Maximum joint velocity
            velocity_penalty: Penalty for velocity
            early_steps_stability_bonus: Stability bonus for early steps
            early_steps_count: Number of early steps
            waiting_velocity_threshold: Velocity threshold for waiting state
            waiting_angle_threshold: Angle threshold for waiting state
            waiting_angular_vel_threshold: Angular velocity threshold for waiting
            penalty_reduction_factor: Factor to reduce penalties (0.2 = 80% reduction)
            patience_bonus: Bonus for patient waiting
        """
        super().__init__(env)
        self.frame_skip = frame_skip
        self.smoothness_coef = smoothness_coef
        self.hull_angle_coef = hull_angle_coef
        self.hull_angular_vel_coef = hull_angular_vel_coef
        self.knee_bend_reward = knee_bend_reward
        self.min_bend_threshold = min_bend_threshold
        self.max_joint_velocity = max_joint_velocity
        self.velocity_penalty = velocity_penalty
        self.early_steps_stability_bonus = early_steps_stability_bonus
        self.early_steps_count = early_steps_count
        
        # Bridge-specific
        self.waiting_velocity_threshold = waiting_velocity_threshold
        self.waiting_angle_threshold = waiting_angle_threshold
        self.waiting_angular_vel_threshold = waiting_angular_vel_threshold
        self.penalty_reduction_factor = penalty_reduction_factor
        self.patience_bonus = patience_bonus
        
        # Tracking
        self.step_count = 0
        self.consecutive_waiting_steps = 0
        self.prev_action = None

    def reset(self, **kwargs):
        """Reset environment and wrapper state."""
        self.step_count = 0
        self.consecutive_waiting_steps = 0
        self.prev_action = None
        return self.env.reset(**kwargs)

    def _is_waiting(self, obs):
        """Detect if agent is in "waiting" state.
        
        Waiting state = low velocity + upright + stable
        This happens when agent is balanced and waiting for bridge to lower.
        
        Args:
            obs: Observation from environment
            
        Returns:
            Boolean indicating if agent is waiting
        """
        # Extract state from observation
        # BipedalWalker obs: [hull_x, hull_y, hull_x_velocity, hull_y_velocity,
        #                     hull_angle, hull_angular_velocity,
        #                     joints_angles(4), joints_velocities(4)]
        
        if len(obs) < 14:
            return False
        
        # Hull state: [x, y, vx, vy, angle, angular_vel, ...]
        hull_x_vel = obs[2]
        hull_y_vel = obs[3]
        hull_angle = obs[4]
        hull_angular_vel = obs[5]
        
        # Compute velocity magnitude
        velocity = np.sqrt(hull_x_vel**2 + hull_y_vel**2)
        
        # Waiting criteria:
        # 1. Low velocity (not moving)
        # 2. Upright posture (small angle)
        # 3. Stable (low angular velocity)
        is_waiting = (
            velocity < self.waiting_velocity_threshold
            and abs(hull_angle) < self.waiting_angle_threshold
            and abs(hull_angular_vel) < self.waiting_angular_vel_threshold
        )
        
        return is_waiting

    def step(self, action):
        """Execute action with bridge-aware reward modifications.
        
        Args:
            action: Action to take
            
        Returns:
            obs, reward, terminated, truncated, info
        """
        self.step_count += 1
        
        # Execute environment step with frame skip
        # Apply reward shaping at each sub-step for consistency
        total_reward = 0.0
        info = {}
        
        for i in range(self.frame_skip):
            obs, reward, terminated, truncated, step_info = self.env.step(action)
            
            # Apply reward shaping to each frame's reward
            shaped_reward = self._apply_reward_shaping(obs, action, reward, step_info)
            total_reward += shaped_reward
            
            # Merge info from last step
            if i == self.frame_skip - 1 or terminated or truncated:
                info = step_info
            
            if terminated or truncated:
                break
        
        # Use accumulated shaped reward
        reward = total_reward
        
        # Check if waiting (based on final observation)
        is_waiting = self._is_waiting(obs)
        
        if is_waiting:
            self.consecutive_waiting_steps += 1
            
            # Reduce penalties during waiting
            if 'smoothness_penalty' in info and info['smoothness_penalty'] > 0:
                smoothness_refund = info['smoothness_penalty'] * (1.0 - self.penalty_reduction_factor)
                reward += smoothness_refund
                info['waiting_smoothness_refund'] = smoothness_refund
            
            if 'hull_angle_penalty' in info and info['hull_angle_penalty'] > 0:
                angle_refund = info['hull_angle_penalty'] * (1.0 - self.penalty_reduction_factor)
                reward += angle_refund
                info['waiting_angle_refund'] = angle_refund
            
            if 'hull_angular_vel_penalty' in info and info['hull_angular_vel_penalty'] > 0:
                angvel_refund = info['hull_angular_vel_penalty'] * (1.0 - self.penalty_reduction_factor)
                reward += angvel_refund
                info['waiting_angvel_refund'] = angvel_refund
            
            # Add patience bonus
            patience_reward = self.patience_bonus
            reward += patience_reward
            info['patience_bonus'] = patience_reward
            
            # Track waiting state
            info['is_waiting'] = True
            info['consecutive_waiting_steps'] = self.consecutive_waiting_steps
        else:
            # Not waiting - reset counter
            self.consecutive_waiting_steps = 0
            info['is_waiting'] = False
            info['consecutive_waiting_steps'] = 0
        
        # Clip reward
        reward = np.clip(reward, -10.0, 10.0)
        
        self.prev_action = action
        return obs, reward, terminated, truncated, info

    def _apply_reward_shaping(self, obs, action, reward, info):
        """Apply hardcore reward shaping to base reward.
        
        This is the standard Elite Hardcore shaping that applies
        penalties for non-natural movement.
        
        Args:
            obs: Current observation
            action: Action taken
            reward: Base reward from environment
            info: Info dict
            
        Returns:
            Modified reward
        """
        # Extract hull state from observation
        if len(obs) < 14:
            return reward
        
        hull_x_vel = obs[2]
        hull_y_vel = obs[3]
        hull_angle = obs[4]
        hull_angular_vel = obs[5]
        
        # Joint angles and velocities
        joint_angles = obs[6:10] if len(obs) >= 10 else np.zeros(4)
        joint_velocities = obs[10:14] if len(obs) >= 14 else np.zeros(4)
        
        # 1. Smoothness penalty: penalize large action changes
        if self.prev_action is not None:
            action_diff = np.abs(action - self.prev_action)
            smoothness_penalty = self.smoothness_coef * np.sum(action_diff ** 2)
            reward -= smoothness_penalty
            info['smoothness_penalty'] = smoothness_penalty
        
        # 2. Hull angle penalty: penalize falling over
        hull_angle_penalty = self.hull_angle_coef * (hull_angle ** 2)
        reward -= hull_angle_penalty
        info['hull_angle_penalty'] = hull_angle_penalty
        
        # 3. Hull angular velocity penalty: penalize spinning
        hull_angular_vel_penalty = self.hull_angular_vel_coef * (hull_angular_vel ** 2)
        reward -= hull_angular_vel_penalty
        info['hull_angular_vel_penalty'] = hull_angular_vel_penalty
        
        # 4. Knee bend reward: encourage natural walking
        knee_angles = np.abs(joint_angles[1::2])  # Lower joint angles
        knee_bend_count = np.sum(knee_angles > self.min_bend_threshold)
        knee_reward = self.knee_bend_reward * knee_bend_count
        reward += knee_reward
        info['knee_bend_reward'] = knee_reward
        
        # 5. Joint velocity penalty: prefer smooth joints
        joint_vel_penalty = self.velocity_penalty * np.sum(np.abs(joint_velocities))
        reward -= joint_vel_penalty
        info['joint_velocity_penalty'] = joint_vel_penalty
        
        # 6. Early steps stability bonus
        if self.step_count < self.early_steps_count:
            stability_bonus = self.early_steps_stability_bonus
            reward += stability_bonus
            info['stability_bonus'] = stability_bonus
        
        return reward
