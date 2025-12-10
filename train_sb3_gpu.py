"""GPU-optimized training using Stable-Baselines3.

This script uses stable-baselines3's production-ready implementations
with GPU acceleration and vectorized environments.

Key advantages over custom implementation:
- Battle-tested, optimized code
- Better callbacks and logging
- Easier hyperparameter tuning
- Native GPU support
- Advanced features (HER, recurrent policies, etc.)
"""

import os
import argparse
import yaml
import torch
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, VecMonitor
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    CallbackList
)
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym


def make_env(env_id: str, rank: int, seed: int = 0, **kwargs):
    """Create a single environment with proper seeding.
    
    Args:
        env_id: Gym environment ID
        rank: Unique ID for this environment
        seed: Base random seed
        **kwargs: Additional environment arguments
    """
    def _init():
        env = gym.make(env_id)
        env.reset(seed=seed + rank)
        env = Monitor(env)
        return env
    return _init


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def train_sac_sb3(config: dict):
    """Train SAC using Stable-Baselines3.
    
    Args:
        config: Configuration dictionary
    """
    print("\n" + "="*60)
    print("STABLE-BASELINES3 SAC TRAINING")
    print("="*60)
    
    # Setup
    env_id = config['env']['name']
    num_envs = config['gpu'].get('num_parallel_envs', 8)
    device = config['experiment']['device']
    seed = config['experiment']['seed']
    
    print(f"Environment: {env_id}")
    print(f"Parallel Envs: {num_envs}")
    print(f"Device: {device}")
    print(f"Seed: {seed}")
    print("="*60 + "\n")
    
    # Create vectorized environment
    print(f"Creating {num_envs} parallel environments...")
    env = SubprocVecEnv([make_env(env_id, i, seed) for i in range(num_envs)])
    env = VecMonitor(env)
    
    # Create evaluation environment
    eval_env = SubprocVecEnv([make_env(env_id, i, seed + 1000) for i in range(5)])
    eval_env = VecMonitor(eval_env)
    
    # SAC hyperparameters from config
    agent_config = config['agent']
    
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=agent_config['learning_rate'],
        buffer_size=config['buffer']['capacity'],
        learning_starts=config['training']['learning_starts'],
        batch_size=config['buffer']['batch_size'],
        tau=agent_config['tau'],
        gamma=agent_config['gamma'],
        train_freq=config['training'].get('train_frequency', 1),
        gradient_steps=config['training'].get('gradient_steps', 1),
        ent_coef='auto' if agent_config['automatic_entropy_tuning'] else agent_config['alpha'],
        target_entropy='auto' if agent_config.get('target_entropy') is None else agent_config['target_entropy'],
        policy_kwargs=dict(
            net_arch=agent_config['hidden_dims']
        ),
        verbose=1,
        tensorboard_log=config['paths']['logs'],
        device=device,
        seed=seed
    )
    
    # Setup callbacks
    checkpoint_dir = os.path.join(config['paths']['checkpoints'], config['experiment']['name'])
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Checkpoint callback - save periodically
    checkpoint_callback = CheckpointCallback(
        save_freq=config['training']['save_frequency'] // num_envs,
        save_path=checkpoint_dir,
        name_prefix='sac_model',
        save_replay_buffer=True,
        save_vecnormalize=True
    )
    
    # Evaluation callback - evaluate periodically
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=checkpoint_dir,
        log_path=os.path.join(config['paths']['logs'], config['experiment']['name']),
        eval_freq=config['training']['eval_frequency'] // num_envs,
        n_eval_episodes=config['training']['eval_episodes'],
        deterministic=True,
        render=False
    )
    
    callback_list = CallbackList([checkpoint_callback, eval_callback])
    
    # Train
    print("\nStarting training...")
    total_timesteps = config['training']['total_timesteps']
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback_list,
        log_interval=4,
        progress_bar=True
    )
    
    # Save final model
    final_path = os.path.join(checkpoint_dir, "final_model.zip")
    model.save(final_path)
    print(f"\n✓ Training completed! Final model saved to {final_path}")
    
    # Clean up
    env.close()
    eval_env.close()


def train_ppo_sb3(config: dict):
    """Train PPO using Stable-Baselines3.
    
    Args:
        config: Configuration dictionary
    """
    print("\n" + "="*60)
    print("STABLE-BASELINES3 PPO TRAINING")
    print("="*60)
    
    # Setup
    env_id = config['env']['name']
    num_envs = config['gpu'].get('num_parallel_envs', 8)
    device = config['experiment']['device']
    seed = config['experiment']['seed']
    
    print(f"Environment: {env_id}")
    print(f"Parallel Envs: {num_envs}")
    print(f"Device: {device}")
    print(f"Seed: {seed}")
    print("="*60 + "\n")
    
    # Create vectorized environment
    print(f"Creating {num_envs} parallel environments...")
    env = SubprocVecEnv([make_env(env_id, i, seed) for i in range(num_envs)])
    env = VecMonitor(env)
    
    # Create evaluation environment
    eval_env = SubprocVecEnv([make_env(env_id, i, seed + 1000) for i in range(5)])
    eval_env = VecMonitor(eval_env)
    
    # PPO hyperparameters from config
    agent_config = config['agent']
    
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=agent_config['learning_rate'],
        n_steps=config['training']['rollout_steps'] // num_envs,
        batch_size=agent_config['mini_batch_size'],
        n_epochs=agent_config['ppo_epochs'],
        gamma=agent_config['gamma'],
        gae_lambda=agent_config['gae_lambda'],
        clip_range=agent_config['clip_epsilon'],
        vf_coef=agent_config['value_loss_coef'],
        ent_coef=agent_config['entropy_coef'],
        max_grad_norm=agent_config['max_grad_norm'],
        policy_kwargs=dict(
            net_arch=dict(pi=agent_config['hidden_dims'], vf=agent_config['hidden_dims'])
        ),
        verbose=1,
        tensorboard_log=config['paths']['logs'],
        device=device,
        seed=seed
    )
    
    # Setup callbacks
    checkpoint_dir = os.path.join(config['paths']['checkpoints'], config['experiment']['name'])
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint_callback = CheckpointCallback(
        save_freq=config['training']['save_frequency'] // num_envs,
        save_path=checkpoint_dir,
        name_prefix='ppo_model',
        save_vecnormalize=True
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=checkpoint_dir,
        log_path=os.path.join(config['paths']['logs'], config['experiment']['name']),
        eval_freq=config['training']['eval_frequency'] // num_envs,
        n_eval_episodes=config['training']['eval_episodes'],
        deterministic=True,
        render=False
    )
    
    callback_list = CallbackList([checkpoint_callback, eval_callback])
    
    # Train
    print("\nStarting training...")
    total_timesteps = config['training']['total_timesteps']
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback_list,
        log_interval=4,
        progress_bar=True
    )
    
    # Save final model
    final_path = os.path.join(checkpoint_dir, "final_model.zip")
    model.save(final_path)
    print(f"\n✓ Training completed! Final model saved to {final_path}")
    
    # Clean up
    env.close()
    eval_env.close()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="SB3 GPU-optimized training")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--num-envs", type=int, default=None, help="Number of parallel environments")
    parser.add_argument("--device", type=str, default=None, help="Device to use")
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    
    # Add GPU-specific config if not present
    if 'gpu' not in config:
        config['gpu'] = {}
    
    # Override with command line args
    if args.num_envs is not None:
        config['gpu']['num_parallel_envs'] = args.num_envs
    if args.device is not None:
        config['experiment']['device'] = args.device
    
    # Set defaults
    config['gpu'].setdefault('num_parallel_envs', 8)

    # Print configuration
    print("="*60)
    print("SB3 GPU-OPTIMIZED TRAINING")
    print("="*60)
    print(f"Agent: {config['agent']['type'].upper()}")
    print(f"Config: {args.config}")
    print(f"Total timesteps: {config['training']['total_timesteps']:,}")
    print("="*60)

    # Train based on agent type
    agent_type = config['agent']['type'].lower()
    if agent_type == 'sac':
        train_sac_sb3(config)
    elif agent_type == 'ppo':
        train_ppo_sb3(config)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


if __name__ == "__main__":
    main()
