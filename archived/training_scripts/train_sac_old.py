#!/usr/bin/env python3
"""Train SAC on Custom BipedalWalker with Bridges - FINAL CLEAN VERSION

This script trains using the WORKING bridge-shaped wrapper with LIDAR-based detection.

Key Features:
- Custom walker environment with BRIDGE obstacles
- Bridge-shaped wrapper with intelligent LIDAR detection
- Immediate rewards for correct bridge behavior (solves delayed reward problem)
- Natural movement quality (knee bending, soft penalties)
- Proven SAC hyperparameters from RL Zoo3

Usage:
    python train_bridge_walker.py --config configs/sac_bridge_shaped_gpu.yaml
"""

import os
import sys
import logging
import argparse
import yaml
import torch
from pathlib import Path

import gymnasium as gym
from gymnasium.envs.registration import register

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.monitor import Monitor

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wrappers.bridge_balanced_wrapper import BridgeBalancedWrapper

# Register custom walker environment
register(
    id='CustomBipedalWalker-v3',
    entry_point='custom_walker:BipedalWalker',
    max_episode_steps=2000,
    reward_threshold=300,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def linear_schedule(initial_value: float):
    """Linear learning rate schedule."""
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func


def make_env(rank: int, seed: int, hardcore: bool, wrapper_kwargs: dict, wrapper_type: str = "shaped"):
    """Create environment with appropriate wrapper."""
    def _init():
        env = gym.make("CustomBipedalWalker-v3", hardcore=hardcore)
        env.reset(seed=seed + rank)

        # Apply appropriate wrapper
        if wrapper_type == "aggressive":
            env = BridgeAggressiveWrapper(env, **wrapper_kwargs)
        elif wrapper_type == "balanced":
            env = BridgeBalancedWrapper(env, **wrapper_kwargs)
        elif wrapper_type == "refined":
            env = BridgeRefinedWrapper(env, **wrapper_kwargs)
        elif wrapper_type == "refined_v2":
            env = BridgeRefinedV2Wrapper(env, **wrapper_kwargs)
        else:  # shaped (default)
            env = BridgeShapedWrapper(env, **wrapper_kwargs)

        env = Monitor(env)
        return env
    return _init


def get_device(requested_device: str) -> str:
    """Get available device with fallback."""
    if requested_device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    return requested_device


def main():
    parser = argparse.ArgumentParser(description="Train SAC on Custom BipedalWalker with Bridges")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    args = parser.parse_args()

    try:
        # Load configuration
        logger.info("=" * 60)
        logger.info("BRIDGE WALKER TRAINING - LIDAR-BASED SHAPING")
        logger.info("=" * 60)
        logger.info(f"Loading config from: {args.config}")

        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)

        logger.info("✓ Configuration loaded")

        # Parse config
        env_config = config['env']
        agent_config = config['agent']
        buffer_config = config['buffer']
        training_config = config['training']
        experiment_config = config['experiment']
        paths_config = config['paths']

        device = get_device(experiment_config['device'])
        logger.info(f"✓ Using device: {device}")

        # Setup paths
        experiment_name = experiment_config['name'] + "_custom_bridges"
        checkpoint_dir = Path(paths_config['checkpoints']) / experiment_name
        log_dir = Path(paths_config['logs']) / experiment_name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)

        # Configuration
        num_envs = config['gpu']['num_parallel_envs']
        seed = experiment_config['seed']
        hardcore = env_config.get('hardcore', True)

        # Determine wrapper type
        use_aggressive = env_config.get('use_bridge_aggressive', False)
        use_balanced = env_config.get('use_bridge_balanced', False)
        use_refined = env_config.get('use_bridge_refined', False)
        use_refined_v2 = env_config.get('use_bridge_refined_v2', False)

        if use_refined_v2:
            wrapper_type = "refined_v2"
        elif use_refined:
            wrapper_type = "refined"
        elif use_balanced:
            wrapper_type = "balanced"
        elif use_aggressive:
            wrapper_type = "aggressive"
        else:
            wrapper_type = "shaped"

        # Wrapper parameters
        if use_refined_v2:
            # MINIMAL changes from balanced - just 3 tweaks
            wrapper_kwargs = {
                'frame_skip': env_config.get('frame_skip', 4),
                'smoothness_coef': env_config.get('smoothness_coef', 0.02),
                'hull_angle_coef': env_config.get('hull_angle_coef', 0.04),
                'hull_angular_vel_coef': env_config.get('hull_angular_vel_coef', 0.02),
                'knee_bend_reward': env_config.get('knee_bend_reward', 0.02),
                'min_bend_threshold': env_config.get('min_bend_threshold', 0.3),
                'stable_waiting_bonus': env_config.get('stable_waiting_bonus', 0.02),
                'bridge_cross_bonus': env_config.get('bridge_cross_bonus', 8.0),
                'min_progress_for_bonuses': env_config.get('min_progress_for_bonuses', 15.0),
                'max_waiting_steps': env_config.get('max_waiting_steps', 400),
                'lidar_bridge_threshold': env_config.get('lidar_bridge_threshold', 0.5),
                'min_close_beams': env_config.get('min_close_beams', 3),
                'waiting_velocity_threshold': env_config.get('waiting_velocity_threshold', 0.15),
                'waiting_angle_threshold': env_config.get('waiting_angle_threshold', 0.3),
                # Only NEW parameter
                'max_hip_spread_for_waiting': env_config.get('max_hip_spread_for_waiting', 0.6),
            }
        elif use_refined:
            wrapper_kwargs = {
                'frame_skip': env_config.get('frame_skip', 4),
                # Base penalties (slightly stronger)
                'smoothness_coef': env_config.get('smoothness_coef', 0.02),
                'hull_angle_coef': env_config.get('hull_angle_coef', 0.05),
                'hull_angular_vel_coef': env_config.get('hull_angular_vel_coef', 0.02),
                # Movement quality (NEW)
                'gait_periodicity_bonus': env_config.get('gait_periodicity_bonus', 0.015),
                'leg_symmetry_bonus': env_config.get('leg_symmetry_bonus', 0.01),
                'velocity_stability_bonus': env_config.get('velocity_stability_bonus', 0.01),
                'target_velocity': env_config.get('target_velocity', 0.5),
                'knee_bend_reward': env_config.get('knee_bend_reward', 0.015),
                'min_bend_threshold': env_config.get('min_bend_threshold', 0.3),
                # Bridge waiting posture (NEW)
                'standing_upright_bonus': env_config.get('standing_upright_bonus', 0.03),
                'legs_together_bonus': env_config.get('legs_together_bonus', 0.02),
                'stable_stance_bonus': env_config.get('stable_stance_bonus', 0.015),
                'max_hip_spread': env_config.get('max_hip_spread', 0.4),
                'target_knee_angle': env_config.get('target_knee_angle', 0.8),
                'stance_hull_threshold': env_config.get('stance_hull_threshold', 0.15),
                # Bridge shaping (same as balanced)
                'stable_waiting_bonus': env_config.get('stable_waiting_bonus', 0.02),
                'bridge_cross_bonus': env_config.get('bridge_cross_bonus', 8.0),
                'min_progress_for_bonuses': env_config.get('min_progress_for_bonuses', 15.0),
                'max_waiting_steps': env_config.get('max_waiting_steps', 400),
                'lidar_bridge_threshold': env_config.get('lidar_bridge_threshold', 0.5),
                'min_close_beams': env_config.get('min_close_beams', 3),
                'waiting_velocity_threshold': env_config.get('waiting_velocity_threshold', 0.15),
                'waiting_angle_threshold': env_config.get('waiting_angle_threshold', 0.3),
            }
        elif use_balanced:
            wrapper_kwargs = {
                'frame_skip': env_config.get('frame_skip', 4),
                'smoothness_coef': env_config.get('smoothness_coef', 0.02),
                'hull_angle_coef': env_config.get('hull_angle_coef', 0.03),
                'hull_angular_vel_coef': env_config.get('hull_angular_vel_coef', 0.015),
                'knee_bend_reward': env_config.get('knee_bend_reward', 0.02),
                'min_bend_threshold': env_config.get('min_bend_threshold', 0.3),
                'stable_waiting_bonus': env_config.get('stable_waiting_bonus', 0.02),
                'bridge_cross_bonus': env_config.get('bridge_cross_bonus', 8.0),
                'min_progress_for_bonuses': env_config.get('min_progress_for_bonuses', 15.0),
                'max_waiting_steps': env_config.get('max_waiting_steps', 400),
                'lidar_bridge_threshold': env_config.get('lidar_bridge_threshold', 0.5),
                'min_close_beams': env_config.get('min_close_beams', 3),
                'waiting_velocity_threshold': env_config.get('waiting_velocity_threshold', 0.15),
                'waiting_angle_threshold': env_config.get('waiting_angle_threshold', 0.3),
            }
        elif use_aggressive:
            wrapper_kwargs = {
                'frame_skip': env_config.get('frame_skip', 4),
                'smoothness_coef': env_config.get('smoothness_coef', 0.01),
                'hull_angle_coef': env_config.get('hull_angle_coef', 0.02),
                'hull_angular_vel_coef': env_config.get('hull_angular_vel_coef', 0.01),
                'knee_bend_reward': env_config.get('knee_bend_reward', 0.02),
                'min_bend_threshold': env_config.get('min_bend_threshold', 0.3),
                'stable_waiting_reward': env_config.get('stable_waiting_reward', 0.1),
                'bridge_cross_bonus': env_config.get('bridge_cross_bonus', 20.0),
                'bridge_detect_bonus': env_config.get('bridge_detect_bonus', 2.0),
                'bridge_stop_bonus': env_config.get('bridge_stop_bonus', 3.0),
                'forward_velocity_bonus': env_config.get('forward_velocity_bonus', 0.5),
                'min_progress_for_bonuses': env_config.get('min_progress_for_bonuses', 10.0),
                'max_waiting_steps': env_config.get('max_waiting_steps', 400),
                'lidar_bridge_threshold': env_config.get('lidar_bridge_threshold', 1.0),
                'waiting_velocity_threshold': env_config.get('waiting_velocity_threshold', 0.2),
                'waiting_angle_threshold': env_config.get('waiting_angle_threshold', 0.4),
            }
        else:
            wrapper_kwargs = {
            'frame_skip': env_config.get('frame_skip', 4),
            'smoothness_coef': env_config.get('smoothness_coef', 0.03),
            'hull_angle_coef': env_config.get('hull_angle_coef', 0.04),
            'hull_angular_vel_coef': env_config.get('hull_angular_vel_coef', 0.02),
            'knee_bend_reward': env_config.get('knee_bend_reward', 0.015),
            'min_bend_threshold': env_config.get('min_bend_threshold', 0.3),
            'stable_waiting_bonus': env_config.get('stable_waiting_bonus', 0.01),
            'bridge_cross_bonus': env_config.get('bridge_cross_bonus', 5.0),
            'min_progress_for_bonuses': env_config.get('min_progress_for_bonuses', 15.0),
            'max_waiting_steps': env_config.get('max_waiting_steps', 400),
            'lidar_bridge_threshold': env_config.get('lidar_bridge_threshold', 0.5),
            'waiting_velocity_threshold': env_config.get('waiting_velocity_threshold', 0.15),
            'waiting_angle_threshold': env_config.get('waiting_angle_threshold', 0.3),
        }

        # Print summary
        logger.info("=" * 60)
        logger.info("TRAINING CONFIGURATION")
        logger.info("=" * 60)
        logger.info(f"Environment: CustomBipedalWalker-v3 (HARDCORE + BRIDGES)")
        logger.info(f"Experiment: {experiment_name}")

        if use_refined_v2:
            wrapper_name = 'BridgeRefinedV2Wrapper (MINIMAL CHANGES)'
        elif use_refined:
            wrapper_name = 'BridgeRefinedWrapper (IMPROVED QUALITY + POSTURE)'
        elif use_balanced:
            wrapper_name = 'BridgeBalancedWrapper (PROPERLY TUNED)'
        elif use_aggressive:
            wrapper_name = 'BridgeAggressiveWrapper (EXTREME SHAPING)'
        else:
            wrapper_name = 'BridgeShapedWrapper (LIDAR-based)'
        logger.info(f"Wrapper: {wrapper_name}")

        logger.info(f"Parallel Envs: {num_envs}")
        logger.info(f"Device: {device}")
        logger.info(f"Seed: {seed}")
        logger.info("=" * 60)

        if use_refined_v2:
            logger.info("REFINED V2 WRAPPER - MINIMAL CHANGES FROM BALANCED:")
            logger.info(f"  Base Penalties (slightly stronger):")
            logger.info(f"    - Smoothness: {wrapper_kwargs['smoothness_coef']} (same)")
            logger.info(f"    - Hull: {wrapper_kwargs['hull_angle_coef']}/{wrapper_kwargs['hull_angular_vel_coef']} (0.03→0.04, 0.015→0.02)")
            logger.info(f"  Movement Quality (same as balanced):")
            logger.info(f"    - Knee Bending: {wrapper_kwargs['knee_bend_reward']}")
            logger.info(f"  Bridge Shaping (same as balanced):")
            logger.info(f"    - Waiting Bonus: +{wrapper_kwargs['stable_waiting_bonus']}/step")
            logger.info(f"    - Crossing Bonus: +{wrapper_kwargs['bridge_cross_bonus']}")
            logger.info(f"  NEW: Stricter waiting detection:")
            logger.info(f"    - max_hip_spread_for_waiting: {wrapper_kwargs['max_hip_spread_for_waiting']}")
        elif use_refined:
            logger.info("REFINED WRAPPER FEATURES:")
            logger.info(f"  Base Penalties (STRONGER for better posture):")
            logger.info(f"    - Smoothness: {wrapper_kwargs['smoothness_coef']}")
            logger.info(f"    - Hull: {wrapper_kwargs['hull_angle_coef']}/{wrapper_kwargs['hull_angular_vel_coef']}")
            logger.info(f"  MOVEMENT QUALITY (NEW):")
            logger.info(f"    - Gait Periodicity: +{wrapper_kwargs['gait_periodicity_bonus']} (alternating legs)")
            logger.info(f"    - Leg Symmetry: +{wrapper_kwargs['leg_symmetry_bonus']} (opposing hip angles)")
            logger.info(f"    - Velocity Stability: +{wrapper_kwargs['velocity_stability_bonus']} (smooth motion)")
            logger.info(f"    - Knee Bending: +{wrapper_kwargs['knee_bend_reward']}")
            logger.info(f"  BRIDGE WAITING POSTURE (NEW):")
            logger.info(f"    - Standing Upright: +{wrapper_kwargs['standing_upright_bonus']} (strict hull)")
            logger.info(f"    - Legs Together: +{wrapper_kwargs['legs_together_bonus']} (not spread)")
            logger.info(f"    - Stable Stance: +{wrapper_kwargs['stable_stance_bonus']} (slight knee bend)")
            logger.info(f"  BRIDGE SHAPING (same as balanced):")
            logger.info(f"    - Waiting Bonus: +{wrapper_kwargs['stable_waiting_bonus']}/step")
            logger.info(f"    - Crossing Bonus: +{wrapper_kwargs['bridge_cross_bonus']}")
        elif use_balanced:
            logger.info("BALANCED WRAPPER FEATURES:")
            logger.info(f"  Base Penalties (MODERATE - always applied):")
            logger.info(f"    - Smoothness: {wrapper_kwargs['smoothness_coef']}")
            logger.info(f"    - Hull: {wrapper_kwargs['hull_angle_coef']}/{wrapper_kwargs['hull_angular_vel_coef']}")
            logger.info(f"  Movement Quality:")
            logger.info(f"    - Knee Bending: {wrapper_kwargs['knee_bend_reward']}")
            logger.info(f"  SIMPLE BRIDGE SHAPING (just 2 bonuses):")
            logger.info(f"    - Waiting Bonus: +{wrapper_kwargs['stable_waiting_bonus']}/step (300 steps = +6.0 total)")
            logger.info(f"    - Crossing Bonus: +{wrapper_kwargs['bridge_cross_bonus']} (big success reward)")
            logger.info(f"  KEY FIXES:")
            logger.info(f"    - No reward normalization (we control scale)")
            logger.info(f"    - Moderate bonuses (no clipping issues)")
            logger.info(f"    - Strict detection (avoid false positives)")
            logger.info(f"    - Penalties always present (stable learning)")
        elif use_aggressive:
            logger.info("AGGRESSIVE WRAPPER FEATURES:")
            logger.info(f"  Base Penalties (MINIMAL - almost zero during bridges):")
            logger.info(f"    - Smoothness: {wrapper_kwargs['smoothness_coef']} (ZERO at bridges)")
            logger.info(f"    - Hull: {wrapper_kwargs['hull_angle_coef']}/{wrapper_kwargs['hull_angular_vel_coef']} (ZERO at bridges)")
            logger.info(f"  Movement Quality:")
            logger.info(f"    - Knee Bending: {wrapper_kwargs['knee_bend_reward']}")
            logger.info(f"    - Forward Velocity Bonus: {wrapper_kwargs['forward_velocity_bonus']} (between bridges)")
            logger.info(f"  AGGRESSIVE BRIDGE SHAPING:")
            logger.info(f"    - Detection Bonus: +{wrapper_kwargs['bridge_detect_bonus']} (first detection)")
            logger.info(f"    - Stop Bonus: +{wrapper_kwargs['bridge_stop_bonus']} (stopping near bridge)")
            logger.info(f"    - Waiting Reward: +{wrapper_kwargs['stable_waiting_reward']}/step (STRONG continuous reward)")
            logger.info(f"    - Crossing Bonus: +{wrapper_kwargs['bridge_cross_bonus']} (HUGE success reward)")
            logger.info(f"  STRATEGY:")
            logger.info(f"    - Make bridge waiting extremely rewarding")
            logger.info(f"    - Zero penalties during bridge encounter")
            logger.info(f"    - Progressive rewards guide agent to correct behavior")
        else:
            logger.info("WRAPPER FEATURES:")
            logger.info(f"  Base Penalties (Very Soft):")
            logger.info(f"    - Smoothness: {wrapper_kwargs['smoothness_coef']}")
            logger.info(f"    - Hull Stability: {wrapper_kwargs['hull_angle_coef']}/{wrapper_kwargs['hull_angular_vel_coef']}")
            logger.info(f"  Movement Quality:")
            logger.info(f"    - Knee Bending: {wrapper_kwargs['knee_bend_reward']}")
            logger.info(f"  Bridge Shaping:")
            logger.info(f"    - LIDAR Detection: threshold={wrapper_kwargs['lidar_bridge_threshold']}")
            logger.info(f"    - Waiting Bonus: +{wrapper_kwargs['stable_waiting_bonus']}/step")
            logger.info(f"    - Crossing Bonus: +{wrapper_kwargs['bridge_cross_bonus']}")

        logger.info("=" * 60)

        # Create environments
        logger.info(f"Creating {num_envs} parallel environments...")
        env = SubprocVecEnv([
            make_env(i, seed, hardcore, wrapper_kwargs, wrapper_type)
            for i in range(num_envs)
        ])
        env = VecMonitor(env)

        # Apply VecNormalize
        logger.info("Applying VecNormalize...")
        env = VecNormalize(
            env,
            norm_obs=env_config.get('normalize_observations', True),
            norm_reward=env_config.get('normalize_rewards', True),
            clip_obs=env_config.get('clip_normalized_obs', 10.0),
            clip_reward=env_config.get('clip_normalized_reward', 10.0),
            gamma=agent_config['gamma'],
        )
        logger.info("✓ Training environments created")

        # Create eval environments
        num_eval_envs = env_config.get('num_eval_envs', 5)
        logger.info(f"Creating {num_eval_envs} eval environments...")
        eval_env = SubprocVecEnv([
            make_env(i, seed + 10000, hardcore, wrapper_kwargs, wrapper_type)
            for i in range(num_eval_envs)
        ])
        eval_env = VecMonitor(eval_env)
        eval_env = VecNormalize(
            env,
            norm_obs=True,
            norm_reward=False,
            clip_obs=env_config.get('clip_normalized_obs', 10.0),
            training=False,
        )
        logger.info("✓ Evaluation environments created")

        # Create model
        logger.info("Creating SAC model...")
        learning_rate = agent_config['learning_rate']
        if agent_config.get('use_linear_schedule', False):
            learning_rate = linear_schedule(learning_rate)

        model = SAC(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            buffer_size=buffer_config['capacity'],
            learning_starts=training_config['learning_starts'],
            batch_size=buffer_config['batch_size'],
            tau=agent_config['tau'],
            gamma=agent_config['gamma'],
            train_freq=training_config['train_frequency'],
            gradient_steps=training_config['gradient_steps'],
            ent_coef=agent_config.get('alpha', 'auto'),
            target_entropy=agent_config.get('target_entropy', 'auto'),
            policy_kwargs={"net_arch": agent_config['hidden_dims']},
            tensorboard_log=str(log_dir),
            verbose=1,
            device=device,
            seed=seed,
        )
        logger.info("✓ Model created")

        # Setup callbacks
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=str(checkpoint_dir / "best_model"),
            log_path=str(log_dir),
            eval_freq=max(training_config['eval_frequency'] // num_envs, 1),
            n_eval_episodes=training_config['eval_episodes'],
            deterministic=True,
        )

        checkpoint_callback = CheckpointCallback(
            save_freq=max(training_config['save_frequency'] // num_envs, 1),
            save_path=str(checkpoint_dir),
            name_prefix='sac_model',
            save_replay_buffer=True,
            save_vecnormalize=True,
        )

        callbacks = CallbackList([checkpoint_callback, eval_callback])

        # Train
        logger.info("=" * 60)
        logger.info("STARTING TRAINING")
        logger.info("=" * 60)
        logger.info(f"Total timesteps: {training_config['total_timesteps']:,}")
        logger.info("=" * 60)

        model.learn(
            total_timesteps=training_config['total_timesteps'],
            callback=callbacks,
            log_interval=training_config.get('log_frequency', 2000),
            progress_bar=True,
        )

        # Save final model
        logger.info("Saving final model...")
        final_model_path = checkpoint_dir / "final_model"
        final_model_path.mkdir(parents=True, exist_ok=True)

        model.save(final_model_path / "sac_model")
        env.save(final_model_path / "vecnormalize.pkl")

        logger.info(f"✓ Final model saved to: {final_model_path}")
        logger.info("=" * 60)
        logger.info("TRAINING COMPLETED!")
        logger.info("=" * 60)

        env.close()
        eval_env.close()

    except KeyboardInterrupt:
        logger.info("\n Training interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"\nFatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
