from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt
import pandas as pd
import os

files_to_plot = {
    "PPO Easy": "logs_comparison/ppo_easy_3/events.out.tfevents.1767551118.mari.250781.0",
    "PPO Hard": "logs_comparison/ppo_hard_base_3/events.out.tfevents.1767555152.mari.297605.0",
    "PPO Hard Bridges": "logs_comparison/",
}

METRIC_TO_PLOT = 'rollout/ep_rew_mean' 
# METRIC_TO_PLOT = 'train/loss' # <--- Use this if your current file has no reward data

def plot_multiple_logs(files_dict, metric):
    plt.figure(figsize=(12, 7))
    
    # Loop through each file in your list
    for label, file_path in files_dict.items():
        print(f"Processing: {label}...")
        
        # 1. Check if file exists
        if not os.path.exists(file_path):
            print(f"  ❌ WARNING: File not found: {file_path}")
            continue
            
        # 2. Load Data
        try:
            ea = EventAccumulator(file_path, size_guidance={'scalars': 0})
            ea.Reload()
        except Exception as e:
            print(f"  ❌ Error loading file: {e}")
            continue

        # 3. Check if tag exists
        if metric not in ea.Tags()['scalars']:
            print(f"  ⚠️ Tag '{metric}' not found in {label}. Skipping.")
            print(f"     Available tags: {ea.Tags()['scalars']}")
            continue
            
        # 4. Extract Data
        scalars = ea.Scalars(metric)
        steps = [s.step for s in scalars]
        values = [s.value for s in scalars]
        
        # 5. Smooth the data (Optional but recommended for messy RL plots)
        # Smoothing makes the trend clearer. 
        if len(values) > 50:
            smoothing_window = int(len(values) / 20) # Auto-adjust window size
            values_smoothed = pd.Series(values).rolling(window=smoothing_window, min_periods=1).mean()
            plt.plot(steps, values_smoothed, label=f"{label} (Smoothed)", linewidth=2)
            # Plot transparent raw data behind it
            plt.plot(steps, values, alpha=0.2, linewidth=0.5)
        else:
            plt.plot(steps, values, label=label, linewidth=2)

    # 6. Final Formatting
    plt.title(f'Model Comparison: {metric}')
    plt.xlabel('Timesteps')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    print("Plot generated. Check the popup window.")
    plt.show()

if __name__ == "__main__":
    plot_multiple_logs(files_to_plot, METRIC_TO_PLOT)