#!/usr/bin/env python3
"""Train TD3 on BipedalWalker - Support for Easy, Hardcore, and Bridges modes

Usage:
    # Easy mode (no obstacles)
    python scripts/training/train_generic.py --config configs/td3_easy.yaml
    
    # Hardcore mode
    python scripts/training/train_generic.py --config configs/td3_hardcore.yaml
    
    # Bridges mode
    python scripts/training/train_generic.py --config configs/td3_bridges.yaml
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

from stable_baselines3 import TD3
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, VecMonitor, DummyVecEnv
from stable_baselines3.common.callbacks import (
    EvalCallback, CheckpointCallback, CallbackList,
    StopTrainingOnRewardThreshold, StopTrainingOnNoModelImprovement
)
from stable_baselines3.common.monitor import Monitor

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.wrappers.bridge_balanced_wrapper import BridgeBalancedWrapper
from src.wrappers.hardcore_wrappers import HardcoreWrapper

# Register custom walker environment for bridges mode
register(
    id='CustomBipedalWalker-v3',
    entry_point='src.envs.custom_walker:BipedalWalker',
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


def make_env(rank: int, seed: int, config: dict):
    """Create environment based on configuration."""
    def _init():
        env_config = config['env']
        hardcore = env_config.get('hardcore', False)
        use_bridge_wrapper = env_config.get('use_bridge_wrapper', False)
        use_hardcore_wrapper = env_config.get('use_hardcore_wrapper', False)
        env_name = env_config.get('name', 'BipedalWalker-v3')
        
        # Create environment
        if use_bridge_wrapper:
            # Use custom walker with bridges
            env = gym.make("CustomBipedalWalker-v3", hardcore=True)
        else:
            # Standard BipedalWalker
            env = gym.make(env_name, hardcore=hardcore)
        
        env.reset(seed=seed + rank)
        
        # Apply wrapper based on mode
        if use_bridge_wrapper:
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
            env = BridgeBalancedWrapper(env, **wrapper_kwargs)
        elif use_hardcore_wrapper and hardcore:
            # Apply unified hardcore wrapper (includes frame skip, smoothness, stability, reward clipping)
            wrapper_kwargs = {
                'frame_skip': env_config.get('frame_skip', 4),
                'smoothness_coef': env_config.get('smoothness_coef', 0.05),
                'angle_coef': env_config.get('angle_coef', 0.02),
                'angular_vel_coef': env_config.get('angular_vel_coef', 0.01),
                'reward_clip_min': env_config.get('reward_clip_min', -10.0),
                'reward_clip_max': env_config.get('reward_clip_max', 10.0),
            }
            env = HardcoreWrapper(env, **wrapper_kwargs)
        
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
    parser = argparse.ArgumentParser(description="Train RL agent on BipedalWalker")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    args = parser.parse_args()

    try:
        # Load configuration
        logger.info("=" * 60)
        logger.info("BIPEDAL WALKER TRAINING")
        logger.info("=" * 60)
        logger.info(f"Loading config from: {args.config}")

        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)

        logger.info("✓ Configuration loaded")

        # Parse config
        env_config = config['env']
        algorithm_config = config['algorithm']
        training_config = config['training']
        checkpoint_config = config['checkpoint']
        experiment_config = config['experiment']

        # Determine mode
        hardcore = env_config.get('hardcore', False)
        use_bridges = env_config.get('use_bridge_wrapper', False)
        
        if use_bridges:
            mode = "BRIDGES (Custom Walker)"
        elif hardcore:
            mode = "HARDCORE (Standard)"
        else:
            mode = "EASY (Standard)"
        
        logger.info(f"✓ Training Mode: {mode}")

        device = get_device(experiment_config['device'])
        logger.info(f"✓ Using device: {device}")

        # Setup paths
        experiment_name = experiment_config['name']
        checkpoint_dir = Path(checkpoint_config['save_path'])
        log_dir = Path(training_config['tensorboard_log'])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"✓ Checkpoints: {checkpoint_dir}")
        logger.info(f"✓ Logs: {log_dir}")

        # Configuration
        num_envs = training_config['n_envs']
        seed = experiment_config['seed']
        total_timesteps = training_config['total_timesteps']

        logger.info(f"✓ Parallel Environments: {num_envs}")
        logger.info(f"✓ Total Timesteps: {total_timesteps:,}")
        logger.info(f"✓ Seed: {seed}")

        # Create vectorized environment
        logger.info("Creating environments...")
        env_fns = [make_env(i, seed, config) for i in range(num_envs)]
        
        if num_envs > 1:
            env = SubprocVecEnv(env_fns)
        else:
            env = DummyVecEnv(env_fns)
        
        # Wrap with VecMonitor
        env = VecMonitor(env)
        
        # Apply normalization if specified
        if env_config.get('normalize_observations', False) or env_config.get('normalize_rewards', False):
            env = VecNormalize(
                env,
                norm_obs=env_config.get('normalize_observations', False),
                norm_reward=env_config.get('normalize_rewards', False),
                clip_obs=env_config.get('clip_normalized_obs', 10.0),
                clip_reward=env_config.get('clip_normalized_reward', 10.0),
            )
            logger.info("✓ VecNormalize applied")
        
        logger.info("✓ Environments created")

        # Create evaluation environment
        logger.info("Creating evaluation environment...")
        eval_env = DummyVecEnv([make_env(0, seed + 1000, config)])
        eval_env = VecMonitor(eval_env)
        
        if env_config.get('normalize_observations', False) or env_config.get('normalize_rewards', False):
            eval_env = VecNormalize(
                eval_env,
                norm_obs=env_config.get('normalize_observations', False),
                norm_reward=False,  # Don't normalize rewards during evaluation
                clip_obs=env_config.get('clip_normalized_obs', 10.0),
                clip_reward=env_config.get('clip_normalized_reward', 10.0),
                training=False,  # Don't update stats during evaluation
            )
        
        logger.info("✓ Evaluation environment created")

        # Setup learning rate schedule
        learning_rate = algorithm_config.get('learning_rate', 3e-4)
        if isinstance(learning_rate, str) and learning_rate == "linear":
            learning_rate = linear_schedule(3e-4)
            logger.info("✓ Using linear learning rate schedule")
        else:
            logger.info(f"✓ Learning rate: {learning_rate}")

        # Create TD3 agent
        logger.info("Creating TD3 agent...")
        
        model = TD3(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            buffer_size=algorithm_config.get('buffer_size', 1000000),
            learning_starts=algorithm_config.get('learning_starts', 10000),
            batch_size=algorithm_config.get('batch_size', 256),
            tau=algorithm_config.get('tau', 0.005),
            gamma=algorithm_config.get('gamma', 0.99),
            train_freq=algorithm_config.get('train_freq', 1),
            gradient_steps=algorithm_config.get('gradient_steps', 1),
            policy_kwargs=dict(net_arch=algorithm_config.get('net_arch', [256, 256])),
            policy_delay=algorithm_config.get('policy_delay', 2),
            target_policy_noise=algorithm_config.get('target_policy_noise', 0.2),
            target_noise_clip=algorithm_config.get('target_noise_clip', 0.5),
            verbose=experiment_config.get('verbose', 1),
            device=device,
            tensorboard_log=str(log_dir),
            seed=seed,
        )
        
        logger.info("✓ TD3 agent created")

        # Setup callbacks
        logger.info("Setting up callbacks...")
        callbacks = []
        
        # Early stopping parameters from config
        early_stopping_config = training_config.get('early_stopping', {})
        use_reward_threshold = early_stopping_config.get('use_reward_threshold', False)
        reward_threshold = early_stopping_config.get('reward_threshold', 300.0)
        use_no_improvement_stop = early_stopping_config.get('use_no_improvement_stop', False)
        patience = early_stopping_config.get('patience', 10)
        min_evals = early_stopping_config.get('min_evals', 0)
        
        # Evaluation callback (always needed)
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=str(checkpoint_dir),
            log_path=str(log_dir),
            eval_freq=training_config.get('eval_freq', 10000),
            n_eval_episodes=training_config.get('eval_episodes', 10),
            deterministic=True,
            render=False,
        )
        
        # Wrap with early stopping callbacks if enabled
        if use_reward_threshold:
            logger.info(f"✓ Early stopping enabled: Reward threshold = {reward_threshold}")
            stop_callback = StopTrainingOnRewardThreshold(
                reward_threshold=reward_threshold,
                verbose=1
            )
            eval_callback = EvalCallback(
                eval_env,
                callback_after_eval=stop_callback,
                best_model_save_path=str(checkpoint_dir),
                log_path=str(log_dir),
                eval_freq=training_config.get('eval_freq', 10000),
                n_eval_episodes=training_config.get('eval_episodes', 10),
                deterministic=True,
                render=False,
            )
        
        if use_no_improvement_stop:
            logger.info(f"✓ Early stopping enabled: No improvement for {patience} evaluations")
            no_improvement_callback = StopTrainingOnNoModelImprovement(
                max_no_improvement_evals=patience,
                min_evals=min_evals,
                verbose=1
            )
            eval_callback = EvalCallback(
                eval_env,
                callback_after_eval=no_improvement_callback,
                best_model_save_path=str(checkpoint_dir),
                log_path=str(log_dir),
                eval_freq=training_config.get('eval_freq', 10000),
                n_eval_episodes=training_config.get('eval_episodes', 10),
                deterministic=True,
                render=False,
            )
        
        callbacks.append(eval_callback)
        
        # Checkpoint callback
        checkpoint_callback = CheckpointCallback(
            save_freq=training_config.get('save_freq', 50000),
            save_path=str(checkpoint_dir),
            name_prefix="td3_model",
            save_replay_buffer=checkpoint_config.get('save_replay_buffer', False),
            save_vecnormalize=True,
        )
        callbacks.append(checkpoint_callback)
        
        callback = CallbackList(callbacks)
        logger.info("✓ Callbacks configured")

        # Start training
        logger.info("=" * 60)
        logger.info("STARTING TRAINING")
        logger.info("=" * 60)
        
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=training_config.get('log_interval', 10),
            tb_log_name=experiment_name,
            progress_bar=True,  # Enable tqdm progress bar
        )

        # Save final model
        logger.info("=" * 60)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 60)
        
        final_model_path = checkpoint_dir / "final_model"
        model.save(final_model_path)
        logger.info(f"✓ Final model saved to: {final_model_path}.zip")
        
        # Save VecNormalize stats if used
        if isinstance(env, VecNormalize):
            vecnormalize_path = checkpoint_dir / "final_model_vecnormalize.pkl"
            env.save(str(vecnormalize_path))
            logger.info(f"✓ VecNormalize stats saved to: {vecnormalize_path}")
        
        logger.info("✓ Training finished successfully!")

    except Exception as e:
        logger.error(f"✗ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        try:
            env.close()
            eval_env.close()
        except:
            pass


if __name__ == "__main__":
    main()

