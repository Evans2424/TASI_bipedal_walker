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

    args = parser.parse_args()

    evaluate(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        num_episodes=args.episodes,
        render=args.render,
        hardcore=args.hardcore
    )


if __name__ == "__main__":
    main()
