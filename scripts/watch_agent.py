"""Script to watch trained agent in real-time (no video recording)."""

import argparse
import yaml
from pathlib import Path
import time

import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.agents import PPOAgent, SACAgent
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
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


def watch_agent(
    checkpoint_path: str,
    config_path: str,
    num_episodes: int = 5,
    hardcore: bool = False,
    fps: int = 50
):
    """Watch agent perform in real-time with rendering.

    Args:
        checkpoint_path: Path to model checkpoint
        config_path: Path to config file
        num_episodes: Number of episodes to watch
        hardcore: Whether to use hardcore mode
        fps: Frames per second for rendering
    """
    # Load config
    config = load_config(config_path)
    set_seed(config['experiment']['seed'])

    # Override hardcore if specified
    if hardcore:
        config['env']['hardcore'] = True
        config['env']['name'] = 'BipedalWalkerHardcore-v3'

    # Create environment with human rendering
    import gymnasium as gym
    env = gym.make(config['env']['name'], render_mode='human')

    # Get dimensions
    observation_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # Create and load agent
    agent = create_agent(config, observation_dim, action_dim)
    agent.load(checkpoint_path)

    print("="*70)
    print("WATCHING AGENT PERFORMANCE")
    print("="*70)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Environment: {config['env']['name']}")
    print(f"Episodes: {num_episodes}")
    print(f"FPS: {fps}")
    print("="*70)
    print("\nClose the window to stop watching, or wait for all episodes to complete.")
    print("="*70)

    frame_time = 1.0 / fps

    try:
        for episode in range(num_episodes):
            observation, _ = env.reset()
            episode_reward = 0
            episode_length = 0
            done = False

            print(f"\nEpisode {episode + 1}/{num_episodes} started...")

            while not done:
                # Render
                env.render()

                # Select action
                action = agent.select_action(observation, deterministic=True)

                # Take step
                observation, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward
                episode_length += 1
                done = terminated or truncated

                # Control frame rate
                time.sleep(frame_time)

            print(f"Episode {episode + 1} finished: Reward = {episode_reward:.2f}, Length = {episode_length}")

    except KeyboardInterrupt:
        print("\n\nWatching interrupted by user.")
    finally:
        env.close()
        print("\nDone watching!")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Watch trained agent in real-time")
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
        default=5,
        help="Number of episodes to watch"
    )
    parser.add_argument(
        "--hardcore",
        action="store_true",
        help="Use hardcore mode"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=50,
        help="Frames per second"
    )

    args = parser.parse_args()

    watch_agent(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        num_episodes=args.episodes,
        hardcore=args.hardcore,
        fps=args.fps
    )


if __name__ == "__main__":
    main()
