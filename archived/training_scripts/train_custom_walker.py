#!/usr/bin/env python3
"""Train Elite Hardcore SAC on Custom BipedalWalker with Bridges.

This script uses the custom_walker.py environment (includes BRIDGES obstacle)
with the proven Elite Hardcore configuration.

Matches train_sb3_gpu.py functionality:
- Proper logging to console and files
- Tensorboard integration
- Intermediate checkpoints with VecNormalize
- Live training progress
- Full evaluation callbacks

Usage:
    python train_custom_walker.py --config configs/sac_elite_hardcore_gpu.yaml
    python train_custom_walker.py --config configs/sac_elite_hardcore_gpu.yaml --resume path/to/checkpoint.zip
"""

import os
import sys
import logging
import argparse
import yaml
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

import gymnasium as gym
from gymnasium.envs.registration import register

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.monitor import Monitor

from elite_hardcore_wrapper import EliteHardcoreWrapper
from bridge_optimized_wrapper import BridgeOptimizedWrapper
from bridge_shaped_wrapper import BridgeShapedWrapper

# Register custom walker environment
register(
    id='CustomBipedalWalker-v3',
    entry_point='custom_walker:BipedalWalker',
    max_episode_steps=2000,
    reward_threshold=300,
)

# Setup logging (matches train_sb3_gpu.py)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def linear_schedule(initial_value: float):
    """Linear learning rate schedule that decreases from initial_value to 0.

    Args:
        initial_value: Initial learning rate

    Returns:
        Function that computes current learning rate based on progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0 (end).

        Args:
            progress_remaining: Remaining progress (1.0 at start, 0.0 at end)

        Returns:
            Current learning rate
        """
        return progress_remaining * initial_value

    return func


def make_env(
    rank: int,
    seed: int = 0,
    hardcore: bool = True,
    use_elite_hardcore: bool = False,
    use_bridge_optimized: bool = False,
    use_bridge_shaped: bool = False,
    wrapper_kwargs: dict = None,
):
    """Create a single environment instance.

    Args:
        rank: Unique ID for this environment
        seed: Base random seed
        hardcore: Enable hardcore mode (obstacles)
        use_elite_hardcore: Whether to use elite hardcore V2 wrapper
        use_bridge_optimized: Whether to use bridge-optimized wrapper
        wrapper_kwargs: Keyword arguments for wrapper
    """
    def _init():
        # Use custom walker instead of standard BipedalWalker-v3
        env = gym.make("CustomBipedalWalker-v3", hardcore=hardcore)
        env.reset(seed=seed + rank)

        # Apply wrapper if configured
        if use_bridge_shaped:
            if wrapper_kwargs is None:
                wrapper_kwargs_local = {}
            else:
                wrapper_kwargs_local = wrapper_kwargs.copy()

            if rank == 0:  # Only log once
                logger.info(f"Applying BridgeShapedWrapper with kwargs: {wrapper_kwargs_local}")

            # Use bridge-shaped wrapper (intelligent LIDAR-based shaping)
            env = BridgeShapedWrapper(env, **wrapper_kwargs_local)

        elif use_bridge_optimized:
            if wrapper_kwargs is None:
                wrapper_kwargs_local = {}
            else:
                wrapper_kwargs_local = wrapper_kwargs.copy()

            if rank == 0:  # Only log once
                logger.info(f"Applying BridgeOptimizedWrapper with kwargs: {wrapper_kwargs_local}")

            # Use bridge-optimized wrapper (soft penalties for bridge compatibility)
            env = BridgeOptimizedWrapper(env, **wrapper_kwargs_local)

        elif use_elite_hardcore:
            if wrapper_kwargs is None:
                wrapper_kwargs_local = {}
            else:
                wrapper_kwargs_local = wrapper_kwargs.copy()

            if rank == 0:  # Only log once
                logger.info(f"Applying EliteHardcoreBridgeWrapperV2 (ANTI-EXPLOIT) with kwargs: {wrapper_kwargs_local}")

            # Use bridge-aware V2 wrapper (anti-exploit version)
            env = EliteHardcoreBridgeWrapperV2(env, **wrapper_kwargs_local)

        # Wrap with Monitor for logging
        env = Monitor(env)
        return env

    return _init


def get_device(requested_device: str) -> str:
    """Validate and get available device with fallback.

    Args:
        requested_device: Requested device (cuda, mps, cpu, or specific like cuda:0)

    Returns:
        Available device string
    """
    if requested_device.startswith('cuda'):
        if torch.cuda.is_available():
            logger.info(f"✓ Using CUDA device: {requested_device}")
            return requested_device
        else:
            logger.warning("CUDA requested but not available. Falling back to CPU.")
            return 'cpu'
    elif requested_device == 'mps':
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            logger.info("✓ Using MPS (Apple Silicon GPU)")
            return 'mps'
        else:
            logger.warning("MPS requested but not available. Falling back to CPU.")
            return 'cpu'
    elif requested_device == 'cpu':
        logger.info("Using CPU")
        return 'cpu'
    else:
        logger.warning(f"Unknown device '{requested_device}'. Falling back to CPU.")
        return 'cpu'


def main():
    parser = argparse.ArgumentParser(
        description="Train SAC on Custom BipedalWalker with Bridges (matches train_sb3_gpu.py)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train from scratch
  python train_custom_walker.py --config configs/sac_elite_hardcore_gpu.yaml

  # Resume from checkpoint
  python train_custom_walker.py --config configs/sac_elite_hardcore_gpu.yaml --resume experiments/checkpoints/sac_elite_unified_hardcore_gpu_custom_bridges/sac_model_5000000_steps.zip

  # Override device and num envs
  python train_custom_walker.py --config configs/sac_elite_hardcore_gpu.yaml --device cpu --num-envs 4
        """
    )
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--num-envs", type=int, default=None, help="Override number of parallel environments")
    parser.add_argument("--device", type=str, default=None, help="Override device (cuda, mps, cpu)")
    args = parser.parse_args()

    try:
        # Load configuration
        logger.info("=" * 60)
        logger.info("CUSTOM WALKER TRAINING - ELITE HARDCORE + BRIDGES")
        logger.info("=" * 60)

        logger.info(f"Loading config from: {args.config}")
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        logger.info("✓ Configuration loaded")

        # Extract configuration
        env_config = config['env']
        agent_config = config['agent']
        buffer_config = config['buffer']
        training_config = config['training']
        experiment_config = config['experiment']
        paths_config = config['paths']
        gpu_config = config.get('gpu', {})

        # Add GPU-specific config if not present
        if 'gpu' not in config:
            config['gpu'] = {}

        # Override with command line args
        if args.num_envs is not None:
            config['gpu']['num_parallel_envs'] = args.num_envs
            logger.info(f"Overriding num_parallel_envs to {args.num_envs}")
        if args.device is not None:
            config['experiment']['device'] = args.device
            logger.info(f"Overriding device to {args.device}")

        # Set defaults
        config['gpu'].setdefault('num_parallel_envs', 8)

        # Setup paths
        experiment_name = experiment_config['name'] + "_custom_bridges"
        checkpoint_dir = Path(paths_config['checkpoints']) / experiment_name
        log_dir = Path(paths_config['logs']) / experiment_name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)

        # Setup
        num_envs = config['gpu']['num_parallel_envs']
        device = get_device(config['experiment']['device'])
        seed = experiment_config['seed']
        num_eval_envs = env_config.get('num_eval_envs', 5)
        hardcore = env_config.get('hardcore', True)

        # Wrapper configuration
        use_elite_hardcore = env_config.get('use_elite_hardcore', False)
        use_bridge_optimized = env_config.get('use_bridge_optimized', False)
        use_bridge_shaped = env_config.get('use_bridge_shaped', False)
        wrapper_kwargs = {}

        if use_bridge_shaped:
            # Bridge-shaped wrapper parameters (LIDAR-based)
            wrapper_kwargs = {
                'frame_skip': env_config.get('frame_skip', 4),
                'smoothness_coef': env_config.get('smoothness_coef', 0.03),
                'hull_angle_coef': env_config.get('hull_angle_coef', 0.04),
                'hull_angular_vel_coef': env_config.get('hull_angular_vel_coef', 0.02),
                'knee_bend_reward': env_config.get('knee_bend_reward', 0.015),
                'min_bend_threshold': env_config.get('min_bend_threshold', 0.3),
                'lidar_bridge_threshold': env_config.get('lidar_bridge_threshold', 0.8),
                'bridge_approach_distance': env_config.get('bridge_approach_distance', 5.0),
                'cautious_approach_bonus': env_config.get('cautious_approach_bonus', 0.02),
                'stable_waiting_bonus': env_config.get('stable_waiting_bonus', 0.03),
                'bridge_cross_bonus': env_config.get('bridge_cross_bonus', 2.0),
                'waiting_velocity_threshold': env_config.get('waiting_velocity_threshold', 0.15),
                'waiting_angle_threshold': env_config.get('waiting_angle_threshold', 0.3),
            }
        elif use_bridge_optimized or use_elite_hardcore:
            wrapper_kwargs = {
                'frame_skip': env_config.get('frame_skip', 4),
                'smoothness_coef': env_config.get('smoothness_coef', 0.05),
                'hull_angle_coef': env_config.get('hull_angle_coef', 0.05),
                'hull_angular_vel_coef': env_config.get('hull_angular_vel_coef', 0.02),
                'knee_bend_reward': env_config.get('knee_bend_reward', 0.01),
                'min_bend_threshold': env_config.get('min_bend_threshold', 0.3),
                'max_joint_velocity': env_config.get('max_joint_velocity', 3.0),
                'velocity_penalty': env_config.get('velocity_penalty', 0.01),
                'early_steps_stability_bonus': env_config.get('early_steps_stability_bonus', 0.01),
                'early_steps_count': env_config.get('early_steps_count', 100),
            }

        # Print configuration summary
        logger.info("=" * 60)
        logger.info("TRAINING CONFIGURATION")
        logger.info("=" * 60)
        logger.info(f"Environment: CustomBipedalWalker-v3 (with BRIDGES)")
        logger.info(f"Experiment: {experiment_name}")
        logger.info(f"Config: {args.config}")
        logger.info(f"Checkpoint dir: {checkpoint_dir}")
        logger.info(f"Log dir: {log_dir}")
        if hardcore:
            logger.info("HARDCORE MODE ENABLED - Training with obstacles!")
            logger.info("Obstacles: GRASS, STUMP, STAIRS, PIT, **BRIDGE** (dynamic drawbridges)")

        if use_bridge_shaped:
            logger.info("*** USING BRIDGE-SHAPED WRAPPER (INTELLIGENT LIDAR SHAPING) ***")
            logger.info(f"  BASE PENALTIES (Very Soft):")
            logger.info(f"    Smoothness: {wrapper_kwargs['smoothness_coef']} (very soft)")
            logger.info(f"    Hull Stability: angle={wrapper_kwargs['hull_angle_coef']}, vel={wrapper_kwargs['hull_angular_vel_coef']}")
            logger.info(f"  MOVEMENT QUALITY:")
            logger.info(f"    Knee Bending: {wrapper_kwargs['knee_bend_reward']}")
            logger.info(f"  BRIDGE SHAPING (INTELLIGENT):")
            logger.info(f"    Bridge Detection: LIDAR threshold={wrapper_kwargs['lidar_bridge_threshold']}")
            logger.info(f"    Cautious Approach: +{wrapper_kwargs['cautious_approach_bonus']} reward for slowing")
            logger.info(f"    Stable Waiting: +{wrapper_kwargs['stable_waiting_bonus']} reward per step")
            logger.info(f"    Crossing Bonus: +{wrapper_kwargs['bridge_cross_bonus']} reward for success")
            logger.info(f"  KEY INNOVATION:")
            logger.info(f"    - Detects bridges in LIDAR")
            logger.info(f"    - Immediate reward for correct behavior (not delayed)")
            logger.info(f"    - Solves credit assignment problem")

        elif use_bridge_optimized:
            logger.info("*** USING BRIDGE-OPTIMIZED WRAPPER (SOFT PENALTIES) ***")
            logger.info(f"  CORE FEATURES (SOFT - Bridge Compatible):")
            logger.info(f"    Frame Skip: {wrapper_kwargs['frame_skip']}")
            logger.info(f"    L2 Smoothness: {wrapper_kwargs['smoothness_coef']} (REDUCED 4x from Elite)")
            logger.info(f"    Hull Stability: angle={wrapper_kwargs['hull_angle_coef']}, vel={wrapper_kwargs['hull_angular_vel_coef']} (REDUCED 2x from Elite)")
            logger.info(f"  MOVEMENT QUALITY (Weak Positive Shaping):")
            logger.info(f"    Knee Bending: {wrapper_kwargs['knee_bend_reward']}")
            logger.info(f"    Velocity Limits: max={wrapper_kwargs['max_joint_velocity']}, penalty={wrapper_kwargs['velocity_penalty']}")
            logger.info(f"    Early Stability: bonus={wrapper_kwargs['early_steps_stability_bonus']} for {wrapper_kwargs['early_steps_count']} steps")
            logger.info(f"  PHILOSOPHY:")
            logger.info(f"    - Softer penalties make 300-step bridge waits viable (~15 cost vs ~60)")
            logger.info(f"    - Forward progress drives learning, not penalties")
            logger.info(f"    - Agent learns naturally when waiting is beneficial")

        elif use_elite_hardcore:
            logger.info("*** USING ELITE HARDCORE BRIDGE WRAPPER V2 (ANTI-EXPLOIT!) ***")
            logger.info(f"  CORE FEATURES (STRONG):")
            logger.info(f"    Frame Skip: {wrapper_kwargs['frame_skip']}")
            logger.info(f"    L2 Smoothness: {wrapper_kwargs['smoothness_coef']}")
            logger.info(f"    Hull Stability: angle={wrapper_kwargs['hull_angle_coef']}, vel={wrapper_kwargs['hull_angular_vel_coef']}")
            logger.info(f"  AUGMENTATIONS (WEAK):")
            logger.info(f"    Knee Bending: {wrapper_kwargs['knee_bend_reward']}")
            logger.info(f"    Velocity Limits: max={wrapper_kwargs['max_joint_velocity']}, penalty={wrapper_kwargs['velocity_penalty']}")
            logger.info(f"    Early Stability: bonus={wrapper_kwargs['early_steps_stability_bonus']} for {wrapper_kwargs['early_steps_count']} steps")
            logger.info(f"  BRIDGE HANDLING V2 (ANTI-EXPLOIT):")
            logger.info(f"    Waiting Detection: velocity < 0.05, angle < 0.2 (STRICTER)")
            logger.info(f"    Forward Progress Req: Must reach x=10 before waiting applies")
            logger.info(f"    Consecutive Frames: Must be stable for 8 frames (2 seconds)")
            logger.info(f"    Penalty Reduction: 80% (only during legitimate bridge waits)")

        logger.info(f"Parallel Envs: {num_envs}")
        logger.info(f"Eval Envs: {num_eval_envs}")
        logger.info(f"Device: {device}")
        logger.info(f"Seed: {seed}")
        if args.resume:
            logger.info(f"Resuming from: {args.resume}")
        logger.info("=" * 60)

        # Print PyTorch info
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        if hasattr(torch.backends, 'mps'):
            logger.info(f"MPS available: {torch.backends.mps.is_available()}")

        # Create vectorized environment
        logger.info(f"Creating {num_envs} parallel HARDCORE environments with BRIDGES...")
        env = SubprocVecEnv([
            make_env(
                rank=i,
                seed=seed,
                hardcore=hardcore,
                use_elite_hardcore=use_elite_hardcore,
                use_bridge_optimized=use_bridge_optimized,
                use_bridge_shaped=use_bridge_shaped,
                wrapper_kwargs=wrapper_kwargs,
            )
            for i in range(num_envs)
        ])
        env = VecMonitor(env)
        logger.info("✓ Training environments created")

        # Apply VecNormalize if requested
        if env_config.get('normalize_observations', False) or env_config.get('normalize_rewards', False):
            logger.info("Applying VecNormalize for observation/reward normalization")
            env = VecNormalize(
                env,
                norm_obs=env_config.get('normalize_observations', True),
                norm_reward=env_config.get('normalize_rewards', True),
                clip_obs=env_config.get('clip_normalized_obs', 10.0),
                clip_reward=env_config.get('clip_normalized_reward', 10.0),
                gamma=agent_config['gamma'],
            )
            logger.info("✓ VecNormalize applied")

        # Create evaluation environment
        logger.info(f"Creating {num_eval_envs} evaluation HARDCORE environments with BRIDGES...")
        eval_env = SubprocVecEnv([
            make_env(
                rank=i,
                seed=seed + 10000,  # Different seed for eval
                hardcore=hardcore,
                use_elite_hardcore=use_elite_hardcore,
                use_bridge_optimized=use_bridge_optimized,
                use_bridge_shaped=use_bridge_shaped,
                wrapper_kwargs=wrapper_kwargs,
            )
            for i in range(num_eval_envs)
        ])
        eval_env = VecMonitor(eval_env)

        # Apply VecNormalize to eval env if used in training (but don't update stats)
        if isinstance(env, VecNormalize):
            eval_env = VecNormalize(
                eval_env,
                norm_obs=env_config.get('normalize_observations', True),
                norm_reward=False,  # Don't normalize rewards during eval
                clip_obs=env_config.get('clip_normalized_obs', 10.0),
                gamma=agent_config['gamma'],
                training=False,  # Freeze normalization stats
            )
            logger.info("✓ VecNormalize applied to eval env (training=False)")
        logger.info("✓ Evaluation environments created")

        # Create or load SAC agent
        if args.resume:
            logger.info(f"Loading model from checkpoint: {args.resume}")
            model = SAC.load(args.resume, env=env, device=device)
            logger.info("✓ Model loaded successfully")

            # Load VecNormalize stats if available
            if isinstance(env, VecNormalize):
                vecnorm_path = args.resume.replace('.zip', '_vecnormalize.pkl')
                if os.path.exists(vecnorm_path):
                    logger.info(f"Loading VecNormalize stats from: {vecnorm_path}")
                    env = VecNormalize.load(vecnorm_path, env)
                    env.training = True
                    env.norm_reward = True
                    logger.info("✓ VecNormalize stats loaded")
        else:
            logger.info("Creating new SAC model...")

            # Learning rate schedule
            learning_rate = agent_config['learning_rate']
            if agent_config.get('use_linear_schedule', False):
                logger.info(f"Using linear learning rate schedule: {agent_config['learning_rate']} → 0")
                learning_rate = linear_schedule(learning_rate)
            else:
                logger.info(f"Using constant learning rate: {learning_rate}")

            # Create SAC agent
            model = SAC(
                policy="MlpPolicy",
                env=env,
                learning_rate=learning_rate,
                buffer_size=buffer_config['capacity'],
                batch_size=buffer_config['batch_size'],
                gamma=agent_config['gamma'],
                tau=agent_config['tau'],
                ent_coef='auto' if agent_config['automatic_entropy_tuning'] else agent_config['alpha'],
                target_entropy='auto' if agent_config.get('target_entropy') is None else agent_config['target_entropy'],
                learning_starts=training_config['learning_starts'],
                train_freq=training_config['train_frequency'],
                gradient_steps=training_config['gradient_steps'],
                policy_kwargs=dict(net_arch=agent_config['hidden_dims']),
                verbose=1,
                tensorboard_log=str(log_dir),  # Enable Tensorboard logging
                device=device,
                seed=seed,
            )
            logger.info("✓ Model created successfully")

        # Setup callbacks
        logger.info(f"Setting up callbacks...")
        logger.info(f"Checkpoint directory: {checkpoint_dir}")

        # Checkpoint callback - save periodically with VecNormalize stats
        # Note: save_freq is divided by num_envs because vectorized envs count steps differently
        checkpoint_callback = CheckpointCallback(
            save_freq=max(training_config['save_frequency'] // num_envs, 1),
            save_path=str(checkpoint_dir),
            name_prefix='sac_model',
            save_replay_buffer=True,
            save_vecnormalize=True,  # CRITICAL: Save VecNormalize stats with checkpoints
        )
        logger.info(f"✓ Checkpoint callback: saving every {training_config['save_frequency']:,} steps")

        # Evaluation callback - evaluate periodically
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=str(checkpoint_dir / "best_model"),
            log_path=str(log_dir),
            eval_freq=max(training_config['eval_frequency'] // num_envs, 1),
            n_eval_episodes=training_config['eval_episodes'],
            deterministic=True,
            render=False,
        )
        logger.info(f"✓ Eval callback: evaluating every {training_config['eval_frequency']:,} steps")

        # Combine callbacks
        callback_list = CallbackList([checkpoint_callback, eval_callback])

        # Train
        logger.info("=" * 60)
        logger.info("STARTING TRAINING")
        logger.info("=" * 60)
        total_timesteps = training_config['total_timesteps']
        logger.info(f"Total timesteps: {total_timesteps:,}")
        logger.info(f"Learning starts: {training_config['learning_starts']:,}")
        logger.info(f"Eval frequency: {training_config['eval_frequency']:,}")
        logger.info(f"Save frequency: {training_config['save_frequency']:,}")
        logger.info(f"Log frequency: {training_config.get('log_frequency', 2000):,}")
        logger.info("=" * 60)

        try:
            model.learn(
                total_timesteps=total_timesteps,
                callback=callback_list,
                log_interval=4,  # Console log interval
                progress_bar=True,  # Show progress bar
            )
            logger.info("✓ Training completed successfully!")
        except KeyboardInterrupt:
            logger.warning("Training interrupted by user")
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
        finally:
            # Save final model
            logger.info("=" * 60)
            logger.info("SAVING FINAL MODEL")
            logger.info("=" * 60)

            final_model_path = checkpoint_dir / "final_model.zip"
            model.save(str(final_model_path))
            logger.info(f"✓ Model saved to: {final_model_path}")

            # Save VecNormalize stats if used
            if isinstance(env, VecNormalize):
                vecnorm_path = checkpoint_dir / "final_model_vecnormalize.pkl"
                env.save(str(vecnorm_path))
                logger.info(f"✓ VecNormalize stats saved to: {vecnorm_path}")

            # Save training info
            info_path = checkpoint_dir / "training_info.txt"
            with open(info_path, 'w') as f:
                f.write(f"Experiment: {experiment_name}\n")
                f.write(f"Config: {args.config}\n")
                f.write(f"Environment: CustomBipedalWalker-v3 (with BRIDGES)\n")
                f.write(f"Hardcore: {hardcore}\n")
                f.write(f"Elite Hardcore: {use_elite_hardcore}\n")
                f.write(f"Total timesteps: {total_timesteps:,}\n")
                f.write(f"Seed: {seed}\n")
                f.write(f"Device: {device}\n")
                f.write(f"Parallel envs: {num_envs}\n")
                f.write(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            logger.info(f"✓ Training info saved to: {info_path}")

            # Cleanup
            logger.info("Cleaning up environments...")
            env.close()
            eval_env.close()
            logger.info("✓ Cleanup complete")

        logger.info("=" * 60)
        logger.info("TRAINING COMPLETE!")
        logger.info("=" * 60)
        logger.info(f"Final model: {final_model_path}")
        logger.info(f"VecNormalize: {checkpoint_dir / 'final_model_vecnormalize.pkl'}")
        logger.info(f"Logs: {log_dir}")
        logger.info(f"Tensorboard: tensorboard --logdir {log_dir}")
        logger.info("=" * 60)

    except KeyboardInterrupt:
        logger.warning("\nTraining interrupted by user. Exiting...")
        sys.exit(0)
    except Exception as e:
        logger.error(f"\nFatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
