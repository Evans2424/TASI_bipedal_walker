"""Script to plot training results from TensorBoard logs."""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tensorboard.backend.event_processing import event_accumulator

sns.set_style("darkgrid")


def load_tensorboard_data(log_dir: str, tag: str):
    """Load data from TensorBoard log file.

    Args:
        log_dir: Directory containing TensorBoard logs
        tag: Tag to extract (e.g., 'episode/reward')

    Returns:
        steps: Array of step numbers
        values: Array of values
    """
    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()

    if tag not in ea.Tags()['scalars']:
        available_tags = ea.Tags()['scalars']
        raise ValueError(f"Tag '{tag}' not found. Available tags: {available_tags}")

    events = ea.Scalars(tag)
    steps = [e.step for e in events]
    values = [e.value for e in events]

    return np.array(steps), np.array(values)


def plot_training_curves(log_dirs: dict, output_path: str = None):
    """Plot training curves from multiple experiments.

    Args:
        log_dirs: Dictionary mapping experiment names to log directories
        output_path: Path to save plot (optional)
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    tags = [
        ('episode/mean_reward_100', 'Mean Episode Reward (last 100 episodes)'),
        ('episode/length', 'Episode Length'),
        ('train/actor_loss', 'Actor Loss'),
        ('train/critic_loss', 'Critic Loss')
    ]

    for idx, (tag, title) in enumerate(tags):
        ax = axes[idx // 2, idx % 2]

        for exp_name, log_dir in log_dirs.items():
            try:
                steps, values = load_tensorboard_data(log_dir, tag)
                ax.plot(steps, values, label=exp_name, alpha=0.7)
            except (ValueError, KeyError) as e:
                print(f"Warning: Could not load {tag} for {exp_name}: {e}")
                continue

        ax.set_xlabel('Training Steps')
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    else:
        plt.show()


def plot_single_experiment(log_dir: str, output_path: str = None):
    """Plot results for a single experiment.

    Args:
        log_dir: Directory containing TensorBoard logs
        output_path: Path to save plot (optional)
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    tags_and_titles = [
        ('episode/reward', 'Episode Reward'),
        ('episode/mean_reward_100', 'Mean Reward (last 100 episodes)'),
        ('episode/length', 'Episode Length'),
        ('train/entropy', 'Policy Entropy')
    ]

    for idx, (tag, title) in enumerate(tags_and_titles):
        ax = axes[idx // 2, idx % 2]

        try:
            steps, values = load_tensorboard_data(log_dir, tag)

            ax.plot(steps, values, alpha=0.6, linewidth=1)

            # Add smoothed curve
            if len(values) > 10:
                window_size = min(100, len(values) // 10)
                smoothed = np.convolve(values, np.ones(window_size)/window_size, mode='valid')
                smooth_steps = steps[:len(smoothed)]
                ax.plot(smooth_steps, smoothed, linewidth=2, label='Smoothed')

            ax.set_xlabel('Training Steps')
            ax.set_ylabel(title)
            ax.set_title(title)
            if 'Smoothed' in ax.get_legend_handles_labels()[1]:
                ax.legend()
            ax.grid(True, alpha=0.3)

        except (ValueError, KeyError) as e:
            print(f"Warning: Could not load {tag}: {e}")
            ax.set_visible(False)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    else:
        plt.show()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Plot training results")
    parser.add_argument(
        "--log-dir",
        type=str,
        required=True,
        help="Path to TensorBoard log directory"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save plot"
    )

    args = parser.parse_args()

    # Find the actual event file in the log directory
    log_path = Path(args.log_dir)
    if not log_path.exists():
        print(f"Error: Log directory {log_path} does not exist")
        return

    plot_single_experiment(str(log_path), args.output)


if __name__ == "__main__":
    main()
