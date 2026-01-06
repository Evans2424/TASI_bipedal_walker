"""Evaluation script for custom walker trained agents."""

import argparse
import yaml
import numpy as np
import torch
from pathlib import Path
import os
import sys

# Add repo to path
sys.path.append(str(Path(__file__).parent))

from custom_walker import BipedalWalker
from src.agents import TD3Agent
from src.utils import set_seed


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_agent(config: dict, observation_dim: int, action_dim: int):
    """Create TD3 agent based on configuration."""
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


def make_custom_env(config: dict, seed: int = None):
    """Create custom walker environment."""
    env = BipedalWalker(
        hardcore=config['env']['hardcore'],
        render_mode=None
    )
    
    # Set seed if provided
    if seed is not None:
        env.reset(seed=seed)
    
    return env


def evaluate(
    checkpoint_path: str,
    config_path: str,
    num_episodes: int = 10
):
    """Evaluate a trained agent.

    Args:
        checkpoint_path: Path to model checkpoint
        config_path: Path to config file
        num_episodes: Number of evaluation episodes
    """
    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return
    
    # Load config
    config = load_config(config_path)
    set_seed(config['experiment']['seed'])

    # Create environment
    env = make_custom_env(config, seed=config['experiment']['seed'] + 999)

    # Get dimensions
    observation_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # Create and load agent
    agent = create_agent(config, observation_dim, action_dim)
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
    parser = argparse.ArgumentParser(description="Evaluate custom walker trained agent")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/td3_hardcore_advanced_custom_walker.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes"
    )

    args = parser.parse_args()

    evaluate(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        num_episodes=args.episodes
    )


if __name__ == "__main__":
    main()
