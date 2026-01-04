"""Comprehensive model analysis script."""

import argparse
import yaml
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.agents import PPOAgent, SACAgent, TD3Agent
from src.envs import make_env
from src.utils import set_seed

sns.set_style("darkgrid")


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


def analyze_model_performance(
    checkpoint_path: str,
    config_path: str,
    num_episodes: int = 50,
    save_plots: bool = True
):
    """Perform comprehensive analysis of model performance.

    Args:
        checkpoint_path: Path to model checkpoint
        config_path: Path to config file
        num_episodes: Number of episodes for analysis
        save_plots: Whether to save plots
    """
    # Load config
    config = load_config(config_path)
    set_seed(config['experiment']['seed'])

    # Create environment
    env = make_env(
        env_id=config['env']['name'],
        hardcore=config['env']['hardcore'],
        seed=config['experiment']['seed']
    )

    # Get dimensions
    observation_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # Create and load agent
    agent = create_agent(config, observation_dim, action_dim)
    agent.load(checkpoint_path)

    print("="*70)
    print("MODEL PERFORMANCE ANALYSIS")
    print("="*70)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Environment: {config['env']['name']}")
    print(f"Analyzing {num_episodes} episodes...")
    print("="*70)

    # Collect detailed episode data
    episode_rewards = []
    episode_lengths = []
    episode_actions = []
    episode_observations = []
    success_count = 0

    for episode in range(num_episodes):
        observation, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False

        ep_actions = []
        ep_obs = []

        while not done:
            action = agent.select_action(observation, deterministic=True)
            next_observation, reward, terminated, truncated, _ = env.step(action)

            ep_actions.append(action.copy())
            ep_obs.append(observation.copy())

            episode_reward += reward
            episode_length += 1
            observation = next_observation
            done = terminated or truncated

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        episode_actions.append(np.array(ep_actions))
        episode_observations.append(np.array(ep_obs))

        # Success is typically 300+ reward
        if episode_reward >= 300:
            success_count += 1

    env.close()

    # Analysis
    rewards_array = np.array(episode_rewards)
    lengths_array = np.array(episode_lengths)

    print("\n" + "="*70)
    print("OVERALL STATISTICS")
    print("="*70)
    print(f"Total Episodes: {num_episodes}")
    print(f"\nReward Statistics:")
    print(f"  Mean:    {rewards_array.mean():.2f}")
    print(f"  Std:     {rewards_array.std():.2f}")
    print(f"  Min:     {rewards_array.min():.2f}")
    print(f"  Max:     {rewards_array.max():.2f}")
    print(f"  Median:  {np.median(rewards_array):.2f}")
    print(f"\nEpisode Length Statistics:")
    print(f"  Mean:    {lengths_array.mean():.2f}")
    print(f"  Std:     {lengths_array.std():.2f}")
    print(f"  Min:     {lengths_array.min():.0f}")
    print(f"  Max:     {lengths_array.max():.0f}")
    print(f"\nSuccess Rate (reward >= 300): {success_count}/{num_episodes} ({100*success_count/num_episodes:.1f}%)")

    print("\n" + "="*70)
    print("PERFORMANCE ASSESSMENT")
    print("="*70)

    mean_reward = rewards_array.mean()

    if mean_reward >= 300:
        assessment = "EXCELLENT"
        color = "\033[92m"  # Green
        message = "Model has solved the environment! The walker successfully traverses the terrain."
    elif mean_reward >= 200:
        assessment = "GOOD"
        color = "\033[93m"  # Yellow
        message = "Model shows strong performance but hasn't fully solved the environment."
    elif mean_reward >= 0:
        assessment = "MODERATE"
        color = "\033[93m"  # Yellow
        message = "Model is making forward progress but needs more training or hyperparameter tuning."
    elif mean_reward >= -50:
        assessment = "POOR"
        color = "\033[91m"  # Red
        message = "Model is struggling. The walker is likely falling or barely moving forward."
    else:
        assessment = "VERY POOR"
        color = "\033[91m"  # Red
        message = "Model has not learned useful behavior. Consider retraining with different hyperparameters."

    print(f"{color}Assessment: {assessment}\033[0m")
    print(f"Explanation: {message}")

    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)

    if mean_reward < 0:
        print("\n⚠️  Your model is performing poorly. Here's what might help:")
        print("\n1. Training Issues:")
        print("   - Did training run long enough? Check if it reached 1-2M timesteps")
        print("   - Check TensorBoard logs for signs of learning")
        print("   - Look for increasing episode rewards during training")

        print("\n2. Hyperparameter Tuning:")
        print("   - Try reducing learning rate (e.g., 1e-4 instead of 3e-4)")
        print("   - Increase entropy coefficient for more exploration")
        print("   - Adjust PPO clipping epsilon")

        print("\n3. Environment Issues:")
        print("   - Verify the environment is working correctly")
        print("   - Try training on a simpler task first")

        print("\n4. Next Steps:")
        print("   - Visualize the agent with: python scripts/watch_agent.py")
        print("   - Check training logs in TensorBoard")
        print("   - Consider starting fresh with a new training run")

    elif mean_reward < 200:
        print("\n✓ Model shows learning but needs improvement:")
        print("\n1. Continue training for more timesteps")
        print("2. Fine-tune hyperparameters (learning rate, entropy)")
        print("3. Try a different algorithm (SAC if using PPO, or vice versa)")
        print("4. Watch the agent to identify specific failure modes")

    else:
        print("\n✓ Model is performing well!")
        print("\n1. Consider testing on hardcore mode")
        print("2. Experiment with more challenging variations")
        print("3. Use this model as a baseline for improvements")

    # Action analysis
    print("\n" + "="*70)
    print("ACTION STATISTICS")
    print("="*70)
    all_actions = np.concatenate(episode_actions)
    print(f"\nAction means: {all_actions.mean(axis=0)}")
    print(f"Action stds:  {all_actions.std(axis=0)}")
    print(f"\nNote: Actions should be in range [-1, 1]")
    print(f"      All actions mean: {all_actions.mean():.3f}")
    print(f"      All actions std:  {all_actions.std():.3f}")

    # Create plots if requested
    if save_plots:
        output_dir = Path("experiments/analysis")
        output_dir.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Plot 1: Episode rewards distribution
        axes[0, 0].hist(episode_rewards, bins=20, alpha=0.7, color='blue', edgecolor='black')
        axes[0, 0].axvline(rewards_array.mean(), color='red', linestyle='--', label=f'Mean: {rewards_array.mean():.2f}')
        axes[0, 0].axvline(300, color='green', linestyle='--', label='Success threshold: 300')
        axes[0, 0].set_xlabel('Episode Reward')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Distribution of Episode Rewards')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Episode rewards over time
        axes[0, 1].plot(episode_rewards, marker='o', linestyle='-', alpha=0.6)
        axes[0, 1].axhline(rewards_array.mean(), color='red', linestyle='--', label=f'Mean: {rewards_array.mean():.2f}')
        axes[0, 1].axhline(300, color='green', linestyle='--', label='Success threshold: 300')
        axes[0, 1].set_xlabel('Episode Number')
        axes[0, 1].set_ylabel('Episode Reward')
        axes[0, 1].set_title('Episode Rewards Over Time')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: Episode lengths
        axes[1, 0].hist(episode_lengths, bins=20, alpha=0.7, color='orange', edgecolor='black')
        axes[1, 0].axvline(lengths_array.mean(), color='red', linestyle='--', label=f'Mean: {lengths_array.mean():.0f}')
        axes[1, 0].set_xlabel('Episode Length')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Distribution of Episode Lengths')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Plot 4: Action distribution
        axes[1, 1].hist(all_actions.flatten(), bins=50, alpha=0.7, color='green', edgecolor='black')
        axes[1, 1].axvline(all_actions.mean(), color='red', linestyle='--', label=f'Mean: {all_actions.mean():.3f}')
        axes[1, 1].set_xlabel('Action Value')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Distribution of All Actions')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = output_dir / "model_analysis.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"\n📊 Analysis plots saved to: {plot_path}")
        plt.close()

    print("\n" + "="*70)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Analyze trained model performance")
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
        default=50,
        help="Number of episodes for analysis"
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Don't save plots"
    )

    args = parser.parse_args()

    analyze_model_performance(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        num_episodes=args.episodes,
        save_plots=not args.no_plots
    )


if __name__ == "__main__":
    main()
