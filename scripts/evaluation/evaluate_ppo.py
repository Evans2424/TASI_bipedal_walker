import gymnasium as gym
import numpy as np
import os
import sys
import csv
import matplotlib.pyplot as plt
from tqdm import tqdm

from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from src.envs.custom_walker import BipedalWalker

# --- CONFIGURATION ---
MODEL_PATH = "models/walker_ppo_hard_base_final.zip"
STATS_PATH = "models/vec_normalize_ppo_hard_base_final.pkl"
OUTPUT_NAME = "ppo_hard_eval"

N_EPISODES = 100
N_CPU = 1
ENV_ID = "BipedalWalkerHardcore-v3"
SUCCESS_THRESHOLD = 300.0

# Wrapper Import
try:
    from wrappers.hardcore_wrappers import HardcoreWrapper
except ImportError:
    print("❌ Error: Wrapper file not found.")
    sys.exit(1)

def make_env():
    """Helper to create a monitored, wrapped environment."""
    # env = BipedalWalker(hardcore=True)
    env = gym.make(ENV_ID, render_mode=None, hardcore=True)
    env = HardcoreWrapper(env)
    env = Monitor(env) 
    return env

def save_plots_and_csv(rewards, lengths, success_rate, output_name):
    """Generates the CSV report and Histograms."""
    csv_path = f"{output_name}.csv"
    print(f"\nSaving results to {csv_path}...")
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Value'])
        writer.writerow(['Success Rate (%)', f"{success_rate:.2f}"])
        writer.writerow(['Mean Reward', f"{np.mean(rewards):.2f}"])
        writer.writerow(['Median Reward', f"{np.median(rewards):.2f}"])
        writer.writerow(['Std Reward', f"{np.std(rewards):.2f}"])
        writer.writerow(['Min Reward', f"{np.min(rewards):.2f}"])
        writer.writerow(['Max Reward', f"{np.max(rewards):.2f}"])
        writer.writerow(['Mean Length', f"{np.mean(lengths):.2f}"])
        writer.writerow([])
        writer.writerow(['Episode', 'Reward', 'Length'])
        for i, (r, l) in enumerate(zip(rewards, lengths)):
            writer.writerow([i+1, f"{r:.2f}", l])
            
    plot_path = f"{output_name}_distribution.png"
    print(f"Generating plots to {plot_path}...")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].hist(rewards, bins=20, edgecolor='black', alpha=0.7)
    axes[0].axvline(np.mean(rewards), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(rewards):.1f}')
    axes[0].axvline(SUCCESS_THRESHOLD, color='green', linestyle='--', linewidth=2, label=f'Success: {SUCCESS_THRESHOLD}')
    axes[0].set_xlabel('Reward')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Reward Distribution', fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(lengths, bins=20, edgecolor='black', alpha=0.7, color='orange')
    axes[1].axvline(np.mean(lengths), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(lengths):.1f}')
    axes[1].set_xlabel('Episode Length')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Episode Length Distribution', fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Done!")

if __name__ == "__main__":
    print(f"--- PARALLEL EVALUATION ({N_CPU} cores) ---")
    print(f"Model: {MODEL_PATH}")

    env = SubprocVecEnv([make_env for _ in range(N_CPU)])

    if os.path.exists(STATS_PATH):
        print(f"Loading normalization stats...")
        env = VecNormalize.load(STATS_PATH, env)
        env.training = False
        env.norm_reward = False
    else:
        print("⚠️ Warning: No stats file found. Running without normalization.")

    model = PPO.load(MODEL_PATH)

    print(f"Running {N_EPISODES} episodes...")
    
    episode_rewards = []
    episode_lengths = []
    
    obs = env.reset()
    
    # Initialize the progress bar
    with tqdm(total=N_EPISODES, desc="Evaluating", unit="ep") as pbar:
        while len(episode_rewards) < N_EPISODES:
            # Predict actions (deterministic for evaluation)
            action, _ = model.predict(obs, deterministic=True)
            
            # Step all environments
            obs, rewards, dones, infos = env.step(action)
            
            # Check for finished episodes
            for i, done in enumerate(dones):
                if done:
                    info = infos[i]
                    if "episode" in info:
                        ep_info = info["episode"]
                        episode_rewards.append(ep_info['r'])
                        episode_lengths.append(ep_info['l'])
                        
                        # Update progress bar
                        pbar.update(1)
                        
                        # Stop immediately if we have enough episodes
                        if len(episode_rewards) >= N_EPISODES:
                            break

    rewards = np.array(episode_rewards)
    lengths = np.array(episode_lengths)
    
    success_count = np.sum(rewards > SUCCESS_THRESHOLD)
    success_rate = (success_count / N_EPISODES) * 100

    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    median_reward = np.median(rewards)
    min_reward = np.min(rewards)
    max_reward = np.max(rewards)
    mean_length = np.mean(lengths)

    print("\n" + "="*35)
    print(f"RESULTS ({N_EPISODES} episodes)")
    print("="*35)
    print(f"Success rate (>{SUCCESS_THRESHOLD}): {success_rate:.1f}%")
    print(f"Mean reward:       {mean_reward:.2f} +/- {std_reward:.2f}")
    print(f"Median reward:     {median_reward:.2f}")
    print(f"Min reward:        {min_reward:.2f}")
    print(f"Max reward:        {max_reward:.2f}")
    print(f"Mean length:       {mean_length:.2f}")
    print("="*35)

    save_plots_and_csv(rewards, lengths, success_rate, OUTPUT_NAME)

    env.close()