"""Script to compare multiple PPO training runs (Easy vs Bridge vs Hard)."""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tensorboard.backend.event_processing import event_accumulator

EXPERIMENTS = {
    "Easy":   "logs_comparison/ppo_easy_5/", 
    "Hardcore": "logs_comparison/ppo_hard_base_3/",
    "Bridge": "logs_comparison/ppo_hard_bridge_2/"
}

COLORS = {
    "Hardcore": "tab:blue",
    "Easy":   "tab:green",
    "Bridge": "tab:orange"
}

sns.set_style("darkgrid")

def load_tensorboard_data(log_dir: str, tag: str):
    """Load and merge data from ALL TensorBoard log files in a directory."""
    path = Path(log_dir)
    
    # Check if path exists
    if not path.exists():
        print(f"⚠️ Warning: Path not found: {log_dir}")
        return None, None

    # Find all event files in the directory
    if path.is_dir():
        files = list(path.glob("events.out.tfevents*"))
    else:
        files = [path] # It's a single file

    if not files:
        return None, None

    all_steps = []
    all_values = []

    # Loop through every file and collect data
    for file_path in files:
        try:
            # We load each file individually
            ea = event_accumulator.EventAccumulator(str(file_path),
                size_guidance={'scalars': 0})
            ea.Reload()

            if tag in ea.Tags()['scalars']:
                events = ea.Scalars(tag)
                for e in events:
                    all_steps.append(e.step)
                    all_values.append(e.value)
        except Exception as e:
            print(f"⚠️ Error reading {file_path.name}: {e}")
            continue

    if not all_steps:
        return None, None

    # Convert to numpy arrays
    all_steps = np.array(all_steps)
    all_values = np.array(all_values)

    # CRITICAL: Sort by step number
    # This ensures the line connects 1.9M -> 2.0M -> 2.1M correctly
    # regardless of which file was loaded first.
    sort_indices = np.argsort(all_steps)
    sorted_steps = all_steps[sort_indices]
    sorted_values = all_values[sort_indices]

    return sorted_steps, sorted_values

def plot_comparison(output_path: str = None):
    """Plot all 15 metrics comparing all experiments in the EXPERIMENTS dict."""
    
    # 4x4 Grid Layout
    tags_layout = [
        # --- ROW 1: PERFORMANCE ---
        ('rollout/ep_rew_mean', 'Mean Reward (Training)'),
        ('eval/mean_reward', 'Mean Reward (Evaluation)'),
        ('rollout/ep_len_mean', 'Episode Length (Training)'),
        ('eval/mean_ep_length', 'Episode Length (Evaluation)'),

        # --- ROW 2: LOSSES ---
        ('train/loss', 'Total Loss'),
        ('train/value_loss', 'Value Function Loss'),
        ('train/policy_gradient_loss', 'Policy Gradient Loss'),
        ('train/entropy_loss', 'Entropy Loss'),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    print(f"--- Plotting Comparison for {len(EXPERIMENTS)} Experiments ---")

    for idx, (tag, title) in enumerate(tags_layout):
        ax = axes[idx]

        if tag is None:
            ax.set_visible(False)
            continue

        # Loop through each experiment for this specific tag
        for label, log_dir in EXPERIMENTS.items():
            steps, values = load_tensorboard_data(log_dir, tag)

            if steps is None:
                continue

            color = COLORS.get(label, None) # Get custom color or None

            # Plot Smoothed Line (Thick)
            if len(values) > 20:
                window = max(5, len(values) // 20)
                smoothed = np.convolve(values, np.ones(window)/window, mode='valid')
                smooth_steps = steps[window-1:]
                ax.plot(smooth_steps, smoothed, label=label, linewidth=2, color=color)
                
                # Optional: Plot faint raw data behind it
                # ax.plot(steps, values, alpha=0.15, linewidth=0.5, color=color)
            else:
                ax.plot(steps, values, label=label, linewidth=2, color=color)

        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.tick_params(labelsize=9)
        ax.grid(True, alpha=0.3)
        
        # Add legend only to the first plot (Top-Left) to avoid clutter
        if idx == 0:
            ax.legend(loc='upper left', fontsize=9, framealpha=0.9)

    plt.tight_layout()
    fig.suptitle(f"PPO Model Comparison: {' vs '.join(EXPERIMENTS.keys())}", fontsize=18, y=1.02)

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Comparison plot saved to {output_path}")
    else:
        plt.show()

if __name__ == "__main__":
    # You can just run the script directly, or pass an output filename
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="comparison_summary.png")
    args = parser.parse_args()

    plot_comparison(args.output)