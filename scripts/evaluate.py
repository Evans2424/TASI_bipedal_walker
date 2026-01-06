"""Evaluation script for trained agents."""

import argparse
import yaml
import numpy as np
import torch
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.agents import PPOAgent, SACAgent, TD3Agent
from src.envs import make_env
from src.utils import set_seed

# Import for custom walker
try:
    from custom_walker import BipedalWalker
    CUSTOM_WALKER_AVAILABLE = True
except ImportError:
    CUSTOM_WALKER_AVAILABLE = False


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_agent(config: dict, observation_dim: int, action_dim: int):
    """Create agent based on configuration."""
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
        return PPOAgent(**common_args)
    elif agent_type == 'sac':
        return SACAgent(**common_args)
    elif agent_type == 'td3':
        return TD3Agent(**common_args)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


def evaluate(
    checkpoint_path: str,
    config_path: str,
    num_episodes: int = 10,
    render: bool = False,
    hardcore: bool = False
):
    """Evaluate a trained agent.

    Args:
        checkpoint_path: Path to model checkpoint
        config_path: Path to config file
        num_episodes: Number of evaluation episodes
        render: Whether to render episodes
        hardcore: Whether to use hardcore mode
    """
    # Load config
    config = load_config(config_path)
    set_seed(config['experiment']['seed'])

    # Override hardcore if specified
    if hardcore:
        config['env']['hardcore'] = True
        config['env']['name'] = 'BipedalWalkerHardcore-v3'

    # Create environment
    render_mode = 'human' if render else None
    env = make_env(
        env_id=config['env']['name'],
        hardcore=config['env']['hardcore'],
        render_mode=render_mode,
        seed=config['experiment']['seed']
    )

    # Get dimensions
    observation_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # Create and load agent
    agent = create_agent(config, observation_dim, action_dim)
    agent.load(checkpoint_path)

    print(f"\nEvaluating agent from {checkpoint_path}")
    print(f"Environment: {config['env']['name']}")
    print(f"Episodes: {num_episodes}\n")

    # Evaluate
    episode_rewards = []
    episode_lengths = []

    for episode in range(num_episodes):
        observation, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False

        while not done:
            action = agent.select_action(observation, deterministic=True)
            observation, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            episode_length += 1
            done = terminated or truncated

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Length = {episode_length}")

    # Print statistics
    print("\n" + "="*50)
    print("Evaluation Results")
    print("="*50)
    print(f"Mean reward: {np.mean(episode_rewards):.2f} +/- {np.std(episode_rewards):.2f}")
    print(f"Min reward: {np.min(episode_rewards):.2f}")
    print(f"Max reward: {np.max(episode_rewards):.2f}")
    print(f"Mean length: {np.mean(episode_lengths):.2f}")
    print("="*50)

    env.close()


def evaluate_custom(
    checkpoint_path: str,
    config_path: str,
    num_episodes: int = 10
):
    """Evaluate a trained agent on custom walker environment.

    Args:
        checkpoint_path: Path to model checkpoint
        config_path: Path to config file
        num_episodes: Number of evaluation episodes
    """
    if not CUSTOM_WALKER_AVAILABLE:
        print("Error: custom_walker module not found. Make sure custom_walker.py exists.")
        return
    
    import os
    
    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return
    
    # Load config
    config = load_config(config_path)
    set_seed(config['experiment']['seed'])

    # Create custom walker environment
    env = BipedalWalker(
        hardcore=config['env']['hardcore'],
        render_mode=None
    )
    env.reset(seed=config['experiment']['seed'] + 999)

    # Get dimensions
    observation_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # Create and load agent (TD3 for custom walker)
    agent_config = config['agent']
    agent = TD3Agent(
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
    agent.load(checkpoint_path)

    print("\n" + "="*60)
    print("Custom Walker Evaluation")
    print("="*60)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Config: {config_path}")
    print(f"Hardcore mode: {config['env']['hardcore']}")
    print(f"Number of episodes: {num_episodes}")
    print("="*60 + "\n")

    # Evaluate
    episode_rewards = []
    episode_lengths = []

    for episode in range(num_episodes):
        observation, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False

        while not done:
            action = agent.select_action(observation, deterministic=True)
            observation, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            episode_length += 1
            done = terminated or truncated

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        print(f"Episode {episode + 1:2d}: Reward = {episode_reward:7.2f}, Length = {episode_length:4d}")

    # Print statistics
    print("\n" + "="*60)
    print("Evaluation Results")
    print("="*60)
    print(f"Mean reward:     {np.mean(episode_rewards):7.2f} +/- {np.std(episode_rewards):.2f}")
    print(f"Median reward:   {np.median(episode_rewards):7.2f}")
    print(f"Min reward:      {np.min(episode_rewards):7.2f}")
    print(f"Max reward:      {np.max(episode_rewards):7.2f}")
    print(f"Mean episode length: {np.mean(episode_lengths):.1f}")
    print("="*60 + "\n")

    env.close()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Evaluate trained agent")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/ppo_config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes"
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render episodes"
    )
    parser.add_argument(
        "--hardcore",
        action="store_true",
        help="Use hardcore mode"
    )
    parser.add_argument(
        "--custom",
        action="store_true",
        help="Use custom walker environment (for bridge training)"
    )

    args = parser.parse_args()

    if args.custom:
        evaluate_custom(
            checkpoint_path=args.checkpoint,
            config_path=args.config,
            num_episodes=args.episodes
        )
    else:
        evaluate(
            checkpoint_path=args.checkpoint,
            config_path=args.config,
            num_episodes=args.episodes,
            render=args.render,
            hardcore=args.hardcore
        )


if __name__ == "__main__":
    main()
