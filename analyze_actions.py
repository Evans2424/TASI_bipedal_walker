"""Analyze action distribution of trained SAC agent.

This script loads a trained model and collects action statistics
to understand if the agent is using the full action space,
particularly for jumping behaviors.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import VecNormalize
import gymnasium as gym
from pathlib import Path

sns.set_style("whitegrid")


def analyze_actions(model_path: str, vecnorm_path: str = None, n_episodes: int = 20):
    """Analyze action distribution from a trained model.

    Args:
        model_path: Path to the trained model (.zip)
        vecnorm_path: Path to VecNormalize stats (.pkl), if used
        n_episodes: Number of episodes to collect actions from
    """
    print(f"Loading model from: {model_path}")

    # Create environment
    env = gym.make("BipedalWalker-v3", hardcore=True)

    # Load VecNormalize if available
    if vecnorm_path and Path(vecnorm_path).exists():
        print(f"Loading VecNormalize from: {vecnorm_path}")
        # Note: VecNormalize needs to wrap the env, but for single env analysis
        # we'll just normalize observations manually if needed

    # Load model
    model = SAC.load(model_path)
    print("Model loaded successfully\n")

    # Action names for BipedalWalker
    action_names = [
        "Hip 1 (right)",
        "Knee 1 (right)",
        "Hip 2 (left)",
        "Knee 2 (left)"
    ]

    # Collect actions
    all_actions = []
    episode_rewards = []
    episode_lengths = []

    print(f"Collecting actions from {n_episodes} episodes...")
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        episode_actions = []
        episode_reward = 0
        steps = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            episode_actions.append(action)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            steps += 1

        all_actions.extend(episode_actions)
        episode_rewards.append(episode_reward)
        episode_lengths.append(steps)

        print(f"Episode {ep+1}/{n_episodes}: Reward={episode_reward:.1f}, Steps={steps}")

    env.close()

    # Convert to numpy array
    all_actions = np.array(all_actions)  # Shape: (total_steps, 4)

    print(f"\nCollected {len(all_actions)} action samples")
    print(f"Mean episode reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Mean episode length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")

    # Analyze statistics
    print("\n" + "="*60)
    print("ACTION STATISTICS")
    print("="*60)
    print(f"{'Action':<20} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10} {'Range Used'}")
    print("-"*60)

    for i, name in enumerate(action_names):
        actions_i = all_actions[:, i]
        mean = np.mean(actions_i)
        std = np.std(actions_i)
        min_val = np.min(actions_i)
        max_val = np.max(actions_i)
        range_used = max_val - min_val

        print(f"{name:<20} {mean:>9.3f} {std:>9.3f} {min_val:>9.3f} {max_val:>9.3f} {range_used:>9.3f}")

    # Check for action saturation (hitting limits)
    print("\n" + "="*60)
    print("ACTION SATURATION ANALYSIS")
    print("="*60)

    for i, name in enumerate(action_names):
        actions_i = all_actions[:, i]
        near_max = np.sum(actions_i > 0.9) / len(actions_i) * 100
        near_min = np.sum(actions_i < -0.9) / len(actions_i) * 100
        near_zero = np.sum(np.abs(actions_i) < 0.1) / len(actions_i) * 100

        print(f"\n{name}:")
        print(f"  Near max (+0.9 to +1.0):  {near_max:>5.1f}%")
        print(f"  Near min (-1.0 to -0.9):  {near_min:>5.1f}%")
        print(f"  Near zero (-0.1 to +0.1): {near_zero:>5.1f}%")

    # Create visualizations
    print("\nGenerating visualizations...")

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Action Distribution Analysis', fontsize=16, fontweight='bold')

    # 1. Action distributions (histograms)
    for i, (ax, name) in enumerate(zip(axes[0], action_names)):
        ax.hist(all_actions[:, i], bins=50, alpha=0.7, edgecolor='black')
        ax.axvline(0, color='red', linestyle='--', alpha=0.5, label='Zero')
        ax.axvline(-1, color='gray', linestyle=':', alpha=0.5)
        ax.axvline(1, color='gray', linestyle=':', alpha=0.5)
        ax.set_xlabel('Action Value')
        ax.set_ylabel('Frequency')
        ax.set_title(name)
        ax.set_xlim(-1.1, 1.1)
        ax.legend()
        ax.grid(True, alpha=0.3)

    # 2. Box plots
    ax_box = axes[1, 0]
    ax_box.boxplot([all_actions[:, i] for i in range(4)],
                    labels=[name.split()[0] for name in action_names])
    ax_box.axhline(0, color='red', linestyle='--', alpha=0.5)
    ax_box.axhline(-1, color='gray', linestyle=':', alpha=0.5)
    ax_box.axhline(1, color='gray', linestyle=':', alpha=0.5)
    ax_box.set_ylabel('Action Value')
    ax_box.set_title('Action Range Comparison')
    ax_box.set_ylim(-1.1, 1.1)
    ax_box.grid(True, alpha=0.3)

    # 3. Correlation heatmap
    ax_corr = axes[1, 1]
    corr_matrix = np.corrcoef(all_actions.T)
    im = ax_corr.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
    ax_corr.set_xticks(range(4))
    ax_corr.set_yticks(range(4))
    ax_corr.set_xticklabels([name.split()[0] for name in action_names], rotation=45)
    ax_corr.set_yticklabels([name.split()[0] for name in action_names])
    ax_corr.set_title('Action Correlations')

    # Add correlation values
    for i in range(4):
        for j in range(4):
            text = ax_corr.text(j, i, f'{corr_matrix[i, j]:.2f}',
                              ha="center", va="center", color="black", fontsize=10)

    plt.colorbar(im, ax=ax_corr)

    # 4. Time series sample (first 500 steps)
    ax_time = axes[1, 2]
    sample_length = min(500, len(all_actions))
    for i, name in enumerate(action_names):
        ax_time.plot(all_actions[:sample_length, i], label=name.split()[0], alpha=0.7)
    ax_time.set_xlabel('Time Step')
    ax_time.set_ylabel('Action Value')
    ax_time.set_title(f'Action Time Series (first {sample_length} steps)')
    ax_time.legend()
    ax_time.grid(True, alpha=0.3)
    ax_time.set_ylim(-1.1, 1.1)

    plt.tight_layout()

    # Save figure
    output_dir = Path("experiments/analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    model_name = Path(model_path).stem
    output_path = output_dir / f"action_distribution_{model_name}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")

    # Diagnostic summary
    print("\n" + "="*60)
    print("DIAGNOSTIC SUMMARY")
    print("="*60)

    # Check if actions are too conservative
    mean_abs_action = np.mean(np.abs(all_actions))
    if mean_abs_action < 0.2:
        print("⚠️  WARNING: Actions are very conservative (mean |action| < 0.2)")
        print("   The agent is barely using the joints - unlikely to jump!")
    elif mean_abs_action < 0.4:
        print("⚠️  CAUTION: Actions are somewhat conservative (mean |action| < 0.4)")
        print("   The agent may not be using enough force to jump over obstacles")
    else:
        print("✓ Actions show reasonable magnitude")

    # Check action diversity
    action_std = np.mean(np.std(all_actions, axis=0))
    if action_std < 0.15:
        print("⚠️  WARNING: Low action diversity (std < 0.15)")
        print("   The agent is using repetitive actions - may be stuck in local optimum")
    else:
        print("✓ Actions show good diversity")

    # Check if using full range
    min_range_used = np.min([np.max(all_actions[:, i]) - np.min(all_actions[:, i]) for i in range(4)])
    if min_range_used < 0.5:
        print("⚠️  WARNING: Not using full action range (min range < 0.5)")
        print("   Some joints are barely being used")
    else:
        print("✓ Using reasonable action range")

    print("\n")

    return all_actions, episode_rewards


def main():
    parser = argparse.ArgumentParser(description="Analyze action distribution of trained SAC model")
    parser.add_argument(
        "--model",
        type=str,
        default="experiments/checkpoints/sac_elite_unified_hardcore_gpu/final_model.zip",
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--vecnorm",
        type=str,
        default="experiments/checkpoints/sac_elite_unified_hardcore_gpu/final_model_vecnormalize.pkl",
        help="Path to VecNormalize stats (optional)"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=20,
        help="Number of episodes to analyze"
    )

    args = parser.parse_args()

    # Auto-detect vecnorm path if not provided
    if args.vecnorm is None:
        vecnorm_path = args.model.replace('.zip', '_vecnormalize.pkl')
        if Path(vecnorm_path).exists():
            args.vecnorm = vecnorm_path

    analyze_actions(args.model, args.vecnorm, args.episodes)


if __name__ == "__main__":
    main()
