"""Script to compare multiple TD3 training runs (Easy vs Hardcore vs Bridges)."""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tensorboard.backend.event_processing import event_accumulator

EXPERIMENTS = {
    "Easy": "experiments/logs/td3_easy/",
    "Hardcore": "experiments/logs/td3_hardcore/",
    "Bridges": "experiments/logs/td3_hardcore_bridges/"
}

COLORS = {
    "Easy": "tab:green",
    "Hardcore": "tab:blue",
    "Bridges": "tab:orange"
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
    """Plot all metrics comparing all TD3 experiments in the EXPERIMENTS dict."""
    
    # 1x3 Grid Layout for TD3 performance metrics
    tags_layout = [
        ('episode/mean_length_100', 'Mean Length - Training'),
        ('episode/mean_reward_100', 'Mean Reward - Training'),
        ('eval/mean_reward', 'Mean Reward - Eval'),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes = axes.flatten()

    print(f"--- Plotting Comparison for {len(EXPERIMENTS)} TD3 Experiments ---")

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
        ax.set_xlabel('Steps', fontsize=9)
        ax.tick_params(labelsize=9)
        ax.grid(True, alpha=0.3)
        
        # Add legend only to the first plot (Top-Left) to avoid clutter
        if idx == 0:
            ax.legend(loc='best', fontsize=9, framealpha=0.9)

    fig.suptitle(f"TD3 Model Comparison: {' vs '.join(EXPERIMENTS.keys())}", fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Comparison plot saved to {output_path}")
    else:
        plt.show()

if __name__ == "__main__":
    # You can just run the script directly, or pass an output filename
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="td3_comparison_summary.png")
    args = parser.parse_args()

    plot_comparison(args.output)
