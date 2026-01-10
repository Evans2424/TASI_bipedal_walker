"""Custom training script for TD3 using custom walker environment."""

import os
import sys
import argparse
import yaml
import torch
import numpy as np
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents import TD3Agent
from src.envs.env_wrapper import BipedalWalkerWrapper
from src.envs.custom_walker import BipedalWalker
from src.utils import ReplayBuffer, Logger, set_seed
from wrappers.bridge_balanced_wrapper import BridgeBalancedWrapper


def make_custom_env(
    hardcore: bool = False,
    render_mode: str = None,
    reward_scale: float = 1.0,
    clip_observations: bool = False,
    clip_actions: bool = True,
    normalize_observations: bool = False,
    normalize_rewards: bool = False,
    clip_normalized_obs: float = 10.0,
    clip_normalized_reward: float = 10.0,
    frame_skip: int = 1,
    smoothness_coef: float = 0.0,
    hull_angle_coef: float = 0.0,
    hull_angular_vel_coef: float = 0.0,
    use_bridge_wrapper: bool = True,
    seed: int = None
):
    """Create custom walker environment with bridge-aware wrapper.
    
    Args:
        hardcore: Use hardcore mode
        render_mode: Render mode ('human', 'rgb_array', None)
        reward_scale: Scale factor for rewards
        clip_observations: Whether to clip observations
        clip_actions: Whether to clip actions
        normalize_observations: Whether to normalize observations
        normalize_rewards: Whether to normalize rewards
        clip_normalized_obs: Clipping range for normalized observations
        clip_normalized_reward: Clipping range for normalized rewards
        frame_skip: Number of frames to skip
        smoothness_coef: Penalty coefficient for action smoothness
        hull_angle_coef: Penalty coefficient for hull angle
        hull_angular_vel_coef: Penalty coefficient for hull angular velocity
        use_bridge_wrapper: Use bridge-aware wrapper for better bridge learning
        seed: Random seed
        
    Returns:
        Wrapped custom environment
    """
    # Create custom walker environment
    env = BipedalWalker(render_mode=render_mode, hardcore=hardcore)
    
    # Set seed if provided
    if seed is not None:
        env.reset(seed=seed)
    
    # Apply bridge-aware wrapper for better bridge learning
    if use_bridge_wrapper:
        # First apply BridgeBalancedWrapper for bridge-specific rewards
        env = BridgeBalancedWrapper(
            env,
            frame_skip=frame_skip,
            smoothness_coef=smoothness_coef,
            hull_angle_coef=hull_angle_coef,
            hull_angular_vel_coef=hull_angular_vel_coef,
            waiting_velocity_threshold=0.15,
            waiting_angle_threshold=0.3,
        )
        # Then apply BipedalWalkerWrapper for normalization (without frame_skip/penalties since handled above)
        env = BipedalWalkerWrapper(
            env,
            reward_scale=reward_scale,
            clip_observations=clip_observations,
            clip_actions=clip_actions,
            normalize_observations=normalize_observations,
            normalize_rewards=normalize_rewards,
            clip_normalized_obs=clip_normalized_obs,
            clip_normalized_reward=clip_normalized_reward,
            frame_skip=1,  # Already handled by BridgeBalancedWrapper
            smoothness_coef=0.0,  # Already handled by BridgeBalancedWrapper
            hull_angle_coef=0.0,  # Already handled by BridgeBalancedWrapper
            hull_angular_vel_coef=0.0  # Already handled by BridgeBalancedWrapper
        )
    else:
        # Fallback to original wrapper only
        env = BipedalWalkerWrapper(
            env,
            reward_scale=reward_scale,
            clip_observations=clip_observations,
            clip_actions=clip_actions,
            normalize_observations=normalize_observations,
            normalize_rewards=normalize_rewards,
            clip_normalized_obs=clip_normalized_obs,
            clip_normalized_reward=clip_normalized_reward,
            frame_skip=frame_skip,
            smoothness_coef=smoothness_coef,
            hull_angle_coef=hull_angle_coef,
            hull_angular_vel_coef=hull_angular_vel_coef
        )
    
    return env


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file.

    Args:
        config_path: Path to config file

    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_agent(config: dict, observation_dim: int, action_dim: int):
    """Create TD3 agent based on configuration.

    Args:
        config: Configuration dictionary
        observation_dim: Observation space dimension
        action_dim: Action space dimension

    Returns:
        Initialized TD3 agent
    """
    agent_config = config['agent']

    return TD3Agent(
        observation_dim=observation_dim,
        action_dim=action_dim,
        hidden_dims=tuple(agent_config['hidden_dims']),
        learning_rate=agent_config['learning_rate'],
        gamma=agent_config['gamma'],
        tau=agent_config['tau'],
        target_noise=agent_config['target_noise'],
        noise_clip=agent_config['noise_clip'],
        policy_update_freq=agent_config['policy_update_freq'],
        device=config['experiment']['device'],
        seed=config['experiment']['seed']
    )


def evaluate(agent, config: dict, num_episodes: int = 10) -> list:
    """Evaluate agent performance.

    Args:
        agent: Trained agent
        config: Configuration dictionary
        num_episodes: Number of evaluation episodes

    Returns:
        List of episode rewards
    """
    eval_env = make_custom_env(
        hardcore=config['env']['hardcore'],
        normalize_observations=config['env'].get('normalize_observations', False),
        normalize_rewards=config['env'].get('normalize_rewards', False),
        clip_normalized_obs=config['env'].get('clip_normalized_obs', 10.0),
        clip_normalized_reward=config['env'].get('clip_normalized_reward', 10.0),
        frame_skip=config['env'].get('frame_skip', 1),
        smoothness_coef=config['env'].get('smoothness_coef', 0.0),
        hull_angle_coef=config['env'].get('hull_angle_coef', 0.0),
        hull_angular_vel_coef=config['env'].get('hull_angular_vel_coef', 0.0),
        seed=config['experiment']['seed'] + 999  # Different seed for eval
    )

    episode_rewards = []

    for _ in range(num_episodes):
        observation, _ = eval_env.reset()
        episode_reward = 0
        done = False

        while not done:
            action = agent.select_action(observation, deterministic=True)
            observation, reward, terminated, truncated, _ = eval_env.step(action)
            episode_reward += reward
            done = terminated or truncated

        episode_rewards.append(episode_reward)

    eval_env.close()
    return episode_rewards


def train_td3(config: dict):
    """Train using TD3 algorithm with custom walker.

    Args:
        config: Configuration dictionary
    """
    # Set seed
    set_seed(config['experiment']['seed'])

    # Create CUSTOM environment with bridge-aware wrapper
    env = make_custom_env(
        hardcore=config['env']['hardcore'],
        reward_scale=config['env']['reward_scale'],
        clip_observations=config['env']['clip_observations'],
        clip_actions=config['env']['clip_actions'],
        normalize_observations=config['env'].get('normalize_observations', False),
        normalize_rewards=config['env'].get('normalize_rewards', False),
        clip_normalized_obs=config['env'].get('clip_normalized_obs', 10.0),
        clip_normalized_reward=config['env'].get('clip_normalized_reward', 10.0),
        frame_skip=config['env'].get('frame_skip', 1),
        smoothness_coef=config['env'].get('smoothness_coef', 0.2),
        hull_angle_coef=config['env'].get('hull_angle_coef', 0.1),
        hull_angular_vel_coef=config['env'].get('hull_angular_vel_coef', 0.05),
        use_bridge_wrapper=True,  # Enable bridge-aware wrapper
        seed=config['experiment']['seed']
    )

    # Get dimensions
    observation_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # Create agent
    agent = create_agent(config, observation_dim, action_dim)
    
    # Load checkpoint if specified (for curriculum learning)
    if 'load_checkpoint' in config['experiment'] and config['experiment']['load_checkpoint']:
        checkpoint_path = config['experiment']['load_checkpoint']
        if os.path.exists(checkpoint_path):
            print(f"\n{'='*50}")
            print(f"Loading checkpoint from: {checkpoint_path}")
            print(f"{'='*50}\n")
            agent.load(checkpoint_path)
        else:
            print(f"\nWARNING: Checkpoint not found: {checkpoint_path}")
            print("Starting training from scratch.\n")

    # Create replay buffer
    buffer = ReplayBuffer(
        observation_dim,
        action_dim,
        capacity=config['buffer']['capacity'],
        seed=config['experiment']['seed']
    )

    # Create logger
    logger = Logger(config['paths']['logs'], config['experiment']['name'])
    logger.save_config(config)

    # Training loop
    observation, _ = env.reset()
    episode_reward = 0
    episode_length = 0
    episode_count = 0

    total_timesteps = config['training']['total_timesteps']
    learning_starts = config['training']['learning_starts']
    batch_size = config['buffer']['batch_size']

    pbar = tqdm(total=total_timesteps, desc="Training")

    for step in range(total_timesteps):
        # Select action (random for initial exploration)
        if step < config['exploration']['initial_random_steps']:
            action = env.action_space.sample()
        else:
            action = agent.select_action(observation, deterministic=False)

        # Take step
        next_observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Store transition
        buffer.add(observation, action, reward, next_observation, done)

        episode_reward += reward
        episode_length += 1

        # Update observation
        observation = next_observation

        # Handle episode end
        if done:
            logger.log_episode(episode_reward, episode_length, step)
            episode_count += 1

            observation, _ = env.reset()
            episode_reward = 0
            episode_length = 0

        # Update progress bar (less frequently for performance)
        pbar.update(1)
        if done or step % 1000 == 0:
            stats = logger.get_stats()
            if stats:
                pbar.set_postfix({
                    'episodes': episode_count,
                    'mean_reward_100': f"{stats.get('mean_reward_100', 0):.2f}"
                })

        # Update agent
        if step >= learning_starts:
            batch = buffer.sample(batch_size)
            metrics = agent.update(batch)

            if step % config['training']['log_frequency'] == 0:
                logger.log_metrics(metrics, step, prefix="train")

        # Evaluation
        if step % config['training']['eval_frequency'] == 0 and step > 0:
            eval_rewards = evaluate(agent, config, config['training']['eval_episodes'])
            logger.log_scalar("eval/mean_reward", np.mean(eval_rewards), step)
            logger.log_scalar("eval/std_reward", np.std(eval_rewards), step)

            print(f"\nStep {step}: Eval mean reward = {np.mean(eval_rewards):.2f} +/- {np.std(eval_rewards):.2f}")

        # Save checkpoint
        if step % config['training']['save_frequency'] == 0 and step > 0:
            checkpoint_dir = os.path.join(config['paths']['checkpoints'], config['experiment']['name'])
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{step}.pt")
            agent.save(checkpoint_path)
            print(f"\nSaved checkpoint to {checkpoint_path}")

        pbar.update(1)

    pbar.close()

    # Final save
    checkpoint_dir = os.path.join(config['paths']['checkpoints'], config['experiment']['name'])
    os.makedirs(checkpoint_dir, exist_ok=True)
    final_path = os.path.join(checkpoint_dir, "final_model.pt")
    agent.save(final_path)

    print(f"\nTraining completed! Final model saved to {final_path}")

    # Clean up intermediate checkpoints
    print("\nCleaning up intermediate checkpoints...")
    for file in os.listdir(checkpoint_dir):
        if file.startswith("checkpoint_") and file.endswith(".pt"):
            checkpoint_path = os.path.join(checkpoint_dir, file)
            os.remove(checkpoint_path)
            print(f"Removed {file}")
    print("Cleanup completed! Only final_model.pt is kept.")

    env.close()
    logger.close()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train TD3 agent on Custom Bipedal Walker")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/td3_hardcore_advanced.yaml",
        help="Path to config file"
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Print configuration
    print("="*50)
    print("Training Configuration - CUSTOM WALKER")
    print("="*50)
    print(f"Agent: TD3")
    print(f"Environment: Custom BipedalWalker")
    print(f"Hardcore: {config['env']['hardcore']}")
    print(f"Total timesteps: {config['training']['total_timesteps']:,}")
    print(f"Device: {config['experiment']['device']}")
    print(f"Seed: {config['experiment']['seed']}")
    print("="*50)

    # Train
    train_td3(config)


if __name__ == "__main__":
    main()
