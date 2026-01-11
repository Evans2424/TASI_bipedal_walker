"""Simplified TD3 training script for hardcore bridges using BridgeBalancedWrapper."""

import os
import sys
import yaml
import torch
import numpy as np
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.agents import TD3Agent
from src.envs.custom_walker import BipedalWalker
from src.envs.env_wrapper import BipedalWalkerWrapper
from wrappers.bridge_balanced_wrapper import BridgeBalancedWrapper
from src.utils import ReplayBuffer, Logger, set_seed


def make_env(config, eval_mode=False):
    """Create wrapped environment."""
    # Create base environment
    env = BipedalWalker(hardcore=config['env']['hardcore'])
    
    # Apply bridge wrapper
    if config['env'].get('use_bridge_wrapper', True):
        env = BridgeBalancedWrapper(
            env,
            frame_skip=config['env'].get('frame_skip', 4),
            smoothness_coef=config['env'].get('smoothness_coef', 0.15),
            hull_angle_coef=config['env'].get('hull_angle_coef', 0.08),
            hull_angular_vel_coef=config['env'].get('hull_angular_vel_coef', 0.03),
        )
    
    # Apply normalization wrapper (without duplicate frame_skip/penalties)
    env = BipedalWalkerWrapper(
        env,
        reward_scale=config['env'].get('reward_scale', 1.0),
        clip_observations=config['env'].get('clip_observations', False),
        clip_actions=config['env'].get('clip_actions', True),
        normalize_observations=config['env'].get('normalize_observations', True),
        normalize_rewards=False,  # Don't normalize - bridge wrapper clips appropriately
        clip_normalized_obs=config['env'].get('clip_normalized_obs', 10.0),
        frame_skip=1,  # Already handled by bridge wrapper
        smoothness_coef=0.0,  # Already handled by bridge wrapper
        hull_angle_coef=0.0,  # Already handled by bridge wrapper
        hull_angular_vel_coef=0.0  # Already handled by bridge wrapper
    )
    
    return env


def evaluate(agent, config, num_episodes=10):
    """Evaluate agent performance."""
    eval_env = make_env(config, eval_mode=True)
    episode_rewards = []
    
    for _ in range(num_episodes):
        obs, _ = eval_env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action = agent.select_action(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = eval_env.step(action)
            episode_reward += reward
            done = terminated or truncated
        
        episode_rewards.append(episode_reward)
    
    eval_env.close()
    return episode_rewards


def train():
    """Main training function."""
    # Load config
    config_path = 'configs/td3_hardcore_advanced_bridges.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"Loaded config from {config_path}")
    print(f"Training {config['experiment']['name']} for {config['training']['total_timesteps']} steps")
    
    # Set seed
    set_seed(config['experiment']['seed'])
    
    # Create environment
    env = make_env(config)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    print(f"Environment: obs_dim={obs_dim}, action_dim={action_dim}")
    
    # Create agent
    agent = TD3Agent(
        observation_dim=obs_dim,
        action_dim=action_dim,
        hidden_dims=tuple(config['agent']['hidden_dims']),
        learning_rate=config['agent']['learning_rate'],
        gamma=config['agent']['gamma'],
        tau=config['agent']['tau'],
        target_noise=config['agent']['target_noise'],
        noise_clip=config['agent']['noise_clip'],
        policy_update_freq=config['agent']['policy_update_freq'],
        device=config['experiment']['device'],
        seed=config['experiment']['seed']
    )
    print(f"Agent created on device: {config['experiment']['device']}")
    
    # Create replay buffer
    buffer = ReplayBuffer(
        obs_dim,
        action_dim,
        capacity=config['buffer']['capacity'],
        seed=config['experiment']['seed']
    )
    print(f"Replay buffer capacity: {config['buffer']['capacity']}")
    
    # Create logger
    log_dir = os.path.join(config['paths']['logs'], config['experiment']['name'])
    logger = Logger(config['paths']['logs'], config['experiment']['name'])
    logger.save_config(config)
    print(f"Logging to: {log_dir}")
    
    # Training setup
    obs, _ = env.reset(seed=config['experiment']['seed'])
    episode_reward = 0
    episode_length = 0
    episode_count = 0
    
    total_steps = config['training']['total_timesteps']
    learning_starts = config['training']['learning_starts']
    batch_size = config['buffer']['batch_size']
    initial_random = config['exploration']['initial_random_steps']
    
    print(f"\nStarting training...")
    print(f"Random exploration until step {initial_random}")
    print(f"Learning starts at step {learning_starts}")
    
    pbar = tqdm(total=total_steps, desc="Training")
    
    try:
        for step in range(total_steps):
            # Select action
            if step < initial_random:
                action = env.action_space.sample()
            else:
                action = agent.select_action(obs, deterministic=False)
            
            # Take step
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Store transition
            buffer.add(obs, action, reward, next_obs, done)
            
            episode_reward += reward
            episode_length += 1
            obs = next_obs
            
            # Episode done
            if done:
                logger.log_episode(episode_reward, episode_length, step)
                episode_count += 1
                
                obs, _ = env.reset()
                episode_reward = 0
                episode_length = 0
                
                # Update progress bar
                stats = logger.get_stats()
                if stats:
                    pbar.set_postfix({
                        'ep': episode_count,
                        'r100': f"{stats.get('mean_reward_100', 0):.1f}"
                    })
            
            # Train agent
            if step >= learning_starts and buffer.size >= batch_size:
                batch = buffer.sample(batch_size)
                metrics = agent.update(batch)
                
                # Log metrics
                if step % config['training']['log_frequency'] == 0:
                    logger.log_metrics(metrics, step, prefix="train")
            
            # Evaluate
            if step > 0 and step % config['training']['eval_frequency'] == 0:
                print(f"\n\nEvaluating at step {step}...")
                eval_rewards = evaluate(agent, config, config['training']['eval_episodes'])
                mean_reward = np.mean(eval_rewards)
                std_reward = np.std(eval_rewards)
                
                logger.log_scalar("eval/mean_reward", mean_reward, step)
                logger.log_scalar("eval/std_reward", std_reward, step)
                
                print(f"Eval: {mean_reward:.2f} +/- {std_reward:.2f}")
                pbar.write(f"Step {step}: Eval = {mean_reward:.2f} +/- {std_reward:.2f}")
            
            # Save checkpoint
            if step > 0 and step % config['training']['save_frequency'] == 0:
                checkpoint_dir = os.path.join(config['paths']['checkpoints'], 
                                            config['experiment']['name'])
                os.makedirs(checkpoint_dir, exist_ok=True)
                checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{step}.pt")
                agent.save(checkpoint_path)
                pbar.write(f"Saved checkpoint: {checkpoint_path}")
            
            pbar.update(1)
        
        pbar.close()
        
        # Save final model
        checkpoint_dir = os.path.join(config['paths']['checkpoints'], 
                                     config['experiment']['name'])
        os.makedirs(checkpoint_dir, exist_ok=True)
        final_path = os.path.join(checkpoint_dir, "final_model.pt")
        agent.save(final_path)
        print(f"\nTraining complete! Final model: {final_path}")
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        checkpoint_dir = os.path.join(config['paths']['checkpoints'], 
                                     config['experiment']['name'])
        os.makedirs(checkpoint_dir, exist_ok=True)
        interrupt_path = os.path.join(checkpoint_dir, "interrupted.pt")
        agent.save(interrupt_path)
        print(f"Saved interrupted model: {interrupt_path}")
    
    except Exception as e:
        print(f"\n\nERROR during training: {e}")
        import traceback
        traceback.print_exc()
        
        # Try to save model on crash
        try:
            checkpoint_dir = os.path.join(config['paths']['checkpoints'], 
                                         config['experiment']['name'])
            os.makedirs(checkpoint_dir, exist_ok=True)
            crash_path = os.path.join(checkpoint_dir, "crashed.pt")
            agent.save(crash_path)
            print(f"Saved crashed model: {crash_path}")
        except:
            print("Could not save crashed model")
    
    finally:
        env.close()
        print("\nEnvironment closed")


if __name__ == "__main__":
    train()
