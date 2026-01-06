"""Record video of custom walker trained agent."""

import argparse
import yaml
import numpy as np
import os
import sys
from pathlib import Path

# Add repo to path
sys.path.append(str(Path(__file__).parent))

from custom_walker import BipedalWalker
from src.agents import TD3Agent
from src.utils import set_seed
import gymnasium as gym
from gymnasium.wrappers import RecordVideo


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


def record_video(
    checkpoint_path: str,
    config_path: str,
    num_episodes: int = 5,
    output_dir: str = "videos"
):
    """Record video of trained agent.

    Args:
        checkpoint_path: Path to model checkpoint
        config_path: Path to config file
        num_episodes: Number of episodes to record
        output_dir: Directory to save videos
    """
    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load config
    config = load_config(config_path)
    set_seed(config['experiment']['seed'])

    # Create base environment
    base_env = BipedalWalker(
        hardcore=config['env']['hardcore'],
        render_mode="rgb_array"
    )

    # Wrap with RecordVideo
    env = RecordVideo(
        base_env,
        video_folder=output_dir,
        episode_trigger=lambda x: True,  # Record all episodes
        name_prefix="custom_walker"
    )

    # Get dimensions
    observation_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # Create and load agent
    agent = create_agent(config, observation_dim, action_dim)
    agent.load(checkpoint_path)

    print("\n" + "="*60)
    print("Recording Custom Walker Videos")
    print("="*60)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output directory: {output_dir}")
    print(f"Hardcore mode: {config['env']['hardcore']}")
    print(f"Number of episodes: {num_episodes}")
    print("="*60 + "\n")

    # Record episodes
    episode_rewards = []

    for episode in range(num_episodes):
        observation, _ = env.reset()
        episode_reward = 0
        done = False
        step = 0

        while not done:
            action = agent.select_action(observation, deterministic=True)
            observation, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            step += 1
            done = terminated or truncated

        episode_rewards.append(episode_reward)
        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Steps = {step}")

    # Print statistics
    print("\n" + "="*60)
    print("Recording Complete")
    print("="*60)
    print(f"Mean reward:     {np.mean(episode_rewards):.2f} +/- {np.std(episode_rewards):.2f}")
    print(f"Videos saved to: {os.path.abspath(output_dir)}")
    print("="*60 + "\n")

    env.close()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Record video of custom walker trained agent")
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
        default=5,
        help="Number of episodes to record"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="videos",
        help="Output directory for videos"
    )

    args = parser.parse_args()

    record_video(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        num_episodes=args.episodes,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
