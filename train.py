"""Main training script for Bipedal Walker."""

import os
import argparse
import yaml
import torch
import numpy as np
from tqdm import tqdm

from src.agents import PPOAgent, SACAgent
from src.envs import make_env
from src.utils import ReplayBuffer, RolloutBuffer, Logger, set_seed


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
    """Create agent based on configuration.

    Args:
        config: Configuration dictionary
        observation_dim: Observation space dimension
        action_dim: Action space dimension

    Returns:
        Initialized agent
    """
    agent_config = config['agent']
    agent_type = agent_config['type'].lower()

    common_args = {
        'observation_dim': observation_dim,
        'action_dim': action_dim,
        'hidden_dims': tuple(agent_config['hidden_dims']),
        'learning_rate': agent_config['learning_rate'],
        'gamma': agent_config['gamma'],
        'device': config['experiment']['device'],
        'seed': config['experiment']['seed']
    }

    if agent_type == 'ppo':
        return PPOAgent(
            **common_args,
            gae_lambda=agent_config['gae_lambda'],
            clip_epsilon=agent_config['clip_epsilon'],
            value_loss_coef=agent_config['value_loss_coef'],
            entropy_coef=agent_config['entropy_coef'],
            max_grad_norm=agent_config['max_grad_norm'],
            ppo_epochs=agent_config['ppo_epochs'],
            mini_batch_size=agent_config['mini_batch_size']
        )
    elif agent_type == 'sac':
        return SACAgent(
            **common_args,
            tau=agent_config['tau'],
            alpha=agent_config['alpha'],
            automatic_entropy_tuning=agent_config['automatic_entropy_tuning'],
            target_entropy=agent_config.get('target_entropy')
        )
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


def train_ppo(config: dict):
    """Train using PPO algorithm.

    Args:
        config: Configuration dictionary
    """
    # Set seed
    set_seed(config['experiment']['seed'])

    # Create environment
    env = make_env(
        env_id=config['env']['name'],
        hardcore=config['env']['hardcore'],
        reward_scale=config['env']['reward_scale'],
        clip_observations=config['env']['clip_observations'],
        clip_actions=config['env']['clip_actions'],
        seed=config['experiment']['seed']
    )

    # Get dimensions
    observation_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # Create agent
    agent = create_agent(config, observation_dim, action_dim)

    # Create buffer
    buffer = RolloutBuffer(observation_dim, action_dim)

    # Create logger
    logger = Logger(config['paths']['logs'], config['experiment']['name'])
    logger.save_config(config)

    # Training loop
    observation, _ = env.reset()
    episode_reward = 0
    episode_length = 0
    episode_count = 0

    total_timesteps = config['training']['total_timesteps']
    rollout_steps = config['training']['rollout_steps']

    pbar = tqdm(total=total_timesteps, desc="Training")

    for step in range(total_timesteps):
        # Select action
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

            # Update progress bar
            stats = logger.get_stats()
            if stats:
                pbar.set_postfix({
                    'episodes': episode_count,
                    'mean_reward_100': f"{stats.get('mean_reward_100', 0):.2f}"
                })

        # Update agent
        if len(buffer) >= rollout_steps:
            batch = buffer.get()
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


def train_sac(config: dict):
    """Train using SAC algorithm.

    Args:
        config: Configuration dictionary
    """
    # Set seed
    set_seed(config['experiment']['seed'])

    # Create environment
    env = make_env(
        env_id=config['env']['name'],
        hardcore=config['env']['hardcore'],
        reward_scale=config['env']['reward_scale'],
        clip_observations=config['env']['clip_observations'],
        clip_actions=config['env']['clip_actions'],
        seed=config['experiment']['seed']
    )

    # Get dimensions
    observation_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # Create agent
    agent = create_agent(config, observation_dim, action_dim)

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

            # Update progress bar
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


def evaluate(agent, config: dict, num_episodes: int = 10) -> list:
    """Evaluate agent performance.

    Args:
        agent: Trained agent
        config: Configuration dictionary
        num_episodes: Number of evaluation episodes

    Returns:
        List of episode rewards
    """
    eval_env = make_env(
        env_id=config['env']['name'],
        hardcore=config['env']['hardcore'],
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


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train RL agent on Bipedal Walker")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/ppo_config.yaml",
        help="Path to config file"
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Print configuration
    print("="*50)
    print("Training Configuration")
    print("="*50)
    print(f"Agent: {config['agent']['type'].upper()}")
    print(f"Environment: {config['env']['name']}")
    print(f"Hardcore: {config['env']['hardcore']}")
    print(f"Total timesteps: {config['training']['total_timesteps']:,}")
    print(f"Device: {config['experiment']['device']}")
    print(f"Seed: {config['experiment']['seed']}")
    print("="*50)

    # Train based on agent type
    agent_type = config['agent']['type'].lower()
    if agent_type == 'ppo':
        train_ppo(config)
    elif agent_type == 'sac':
        train_sac(config)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


if __name__ == "__main__":
    main()
