"""Script to record videos of trained agents."""

import argparse
import yaml
import numpy as np
from pathlib import Path
import gymnasium as gym

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


def record_video(
    checkpoint_path: str,
    config_path: str,
    output_dir: str,
    num_episodes: int = 3,
    hardcore: bool = False
):
    """Record videos of agent performance.

    Args:
        checkpoint_path: Path to model checkpoint
        config_path: Path to config file
        output_dir: Directory to save videos
        num_episodes: Number of episodes to record
        hardcore: Whether to use hardcore mode
    """
    # Load config
    config = load_config(config_path)
    set_seed(config['experiment']['seed'])

    # Override hardcore if specified
    if hardcore:
        config['env']['hardcore'] = True
        config['env']['name'] = 'BipedalWalkerHardcore-v3'

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Create environment with video recording
    env = gym.make(
        config['env']['name'],
        render_mode='rgb_array'
    )
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=output_dir,
        episode_trigger=lambda x: True,  # Record all episodes
        name_prefix=f"agent_{Path(checkpoint_path).stem}"
    )

    # Get dimensions
    observation_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # Create and load agent
    agent = create_agent(config, observation_dim, action_dim)
    agent.load(checkpoint_path)

    print(f"\nRecording videos from {checkpoint_path}")
    print(f"Output directory: {output_dir}")
    print(f"Environment: {config['env']['name']}")
    print(f"Episodes: {num_episodes}\n")

    # Record episodes
    episode_rewards = []

    for episode in range(num_episodes):
        observation, _ = env.reset()
        episode_reward = 0
        done = False

        while not done:
            action = agent.select_action(observation, deterministic=True)
            observation, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            done = terminated or truncated

        episode_rewards.append(episode_reward)
        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}")

    print(f"\nMean reward: {np.mean(episode_rewards):.2f} +/- {np.std(episode_rewards):.2f}")
    print(f"Videos saved to {output_dir}")

    env.close()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Record videos of trained agent")
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
        "--output",
        type=str,
        default="experiments/videos",
        help="Output directory for videos"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=3,
        help="Number of episodes to record"
    )
    parser.add_argument(
        "--hardcore",
        action="store_true",
        help="Use hardcore mode"
    )

    args = parser.parse_args()

    record_video(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        output_dir=args.output,
        num_episodes=args.episodes,
        hardcore=args.hardcore
    )


if __name__ == "__main__":
    main()
