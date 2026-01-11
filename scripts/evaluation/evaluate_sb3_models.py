#!/usr/bin/env python3
"""Evaluate trained Stable Baselines3 models.

Usage:
    python scripts/evaluate_sb3_models.py --model-dir experiments/checkpoints/td3_easy
    python scripts/evaluate_sb3_models.py --model-dir experiments/checkpoints/td3_hardcore
    python scripts/evaluate_sb3_models.py --model-dir experiments/checkpoints/td3_hardcore_bridges
"""

import argparse
import os
import sys
import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium.envs.registration import register
from stable_baselines3 import TD3, SAC, PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.wrappers.bridge_balanced_wrapper import BridgeBalancedWrapper
from src.wrappers.hardcore_wrappers import HardcoreWrapper

# Import custom walker from src/envs
try:
    from src.envs import custom_walker
    CUSTOM_WALKER_AVAILABLE = True
    # Register custom walker environment
    register(
        id='CustomBipedalWalker-v3',
        entry_point='src.envs.custom_walker:BipedalWalker',
        max_episode_steps=2000,
        reward_threshold=300,
    )
except ImportError:
    CUSTOM_WALKER_AVAILABLE = False


def evaluate_model(model_dir: str, num_episodes: int = 50, output_dir: str = None):
    """Evaluate a trained model.
    
    Args:
        model_dir: Path to model directory
        num_episodes: Number of episodes to evaluate
        output_dir: Output directory for results (default: same as model_dir)
    """
    model_dir = Path(model_dir)
    
    if not model_dir.exists():
        print(f"Error: Model directory not found: {model_dir}")
        return
    
    # Find model file
    model_file = model_dir / "best_model.zip"
    if not model_file.exists():
        model_file = model_dir / "final_model.zip"
        if not model_file.exists():
            print(f"Error: No model file found in {model_dir}")
            return
    
    # Find vecnormalize file
    vecnorm_file = None
    for name in ["best_model_vecnormalize.pkl", "final_model_vecnormalize.pkl"]:
        candidate = model_dir / name
        if candidate.exists():
            vecnorm_file = candidate
            break
    
    # Determine algorithm type
    dir_name = model_dir.name.lower()
    if 'td3' in dir_name:
        algorithm_class = TD3
        algorithm_name = "TD3"
    elif 'sac' in dir_name:
        algorithm_class = SAC
        algorithm_name = "SAC"
    elif 'ppo' in dir_name:
        algorithm_class = PPO
        algorithm_name = "PPO"
    else:
        algorithm_class = TD3
        algorithm_name = "TD3"
    
    # Check if this is a bridge model
    use_bridges = 'bridge' in dir_name
    
    if use_bridges and not CUSTOM_WALKER_AVAILABLE:
        print("Error: Bridge models require custom walker")
        return
    
    # Determine environment
    if use_bridges:
        env_name = "CustomBipedalWalker-v3"
        mode = "Hardcore with Bridges"
        env_hardcore = True
    elif 'hardcore' in dir_name:
        env_name = "BipedalWalkerHardcore-v3"
        mode = "Hardcore"
        env_hardcore = True
    else:
        env_name = "BipedalWalker-v3"
        mode = "Easy"
        env_hardcore = False
    
    # Set output directory
    if output_dir is None:
        output_dir = model_dir
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("MODEL EVALUATION")
    print("=" * 70)
    print(f"Algorithm:    {algorithm_name}")
    print(f"Model:        {model_file}")
    print(f"VecNormalize: {vecnorm_file if vecnorm_file else 'None'}")
    print(f"Environment:  {env_name} ({mode})")
    print(f"Episodes:     {num_episodes}")
    print(f"Output:       {output_dir}")
    print("=" * 70)
    print()
    
    # Load model
    print(f"Loading {algorithm_name} model...")
    model = algorithm_class.load(str(model_file))
    
    # Create environment
    print("Creating environment...")
    
    if use_bridges:
        base_env = gym.make(env_name, hardcore=True)
        wrapped_env = BridgeBalancedWrapper(
            base_env,
            frame_skip=4,
            smoothness_coef=0.02,
            hull_angle_coef=0.03,
            hull_angular_vel_coef=0.015,
            knee_bend_reward=0.02,
            min_bend_threshold=0.3,
            stable_waiting_bonus=0.02,
            bridge_cross_bonus=8.0,
            min_progress_for_bonuses=15.0,
            max_waiting_steps=400,
            lidar_bridge_threshold=0.5,
            min_close_beams=3,
            waiting_velocity_threshold=0.15,
            waiting_angle_threshold=0.3,
        )
        env = Monitor(wrapped_env)
    elif env_hardcore:
        base_env = gym.make(env_name)
        wrapped_env = HardcoreWrapper(
            base_env,
            frame_skip=4,
            smoothness_coef=0.05,
            angle_coef=0.02,
            angular_vel_coef=0.01,
            reward_clip_min=-10.0,
            reward_clip_max=10.0,
        )
        env = Monitor(wrapped_env)
    else:
        env = gym.make(env_name)
    
    # Wrap in VecEnv
    vec_env = DummyVecEnv([lambda: env])
    
    # Load VecNormalize stats if available
    if vecnorm_file:
        print(f"Loading VecNormalize stats...")
        vec_env = VecNormalize.load(str(vecnorm_file), vec_env)
        vec_env.training = False
        vec_env.norm_reward = False
    
    # Evaluate
    print(f"\nEvaluating for {num_episodes} episodes...\n")
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(num_episodes):
        obs = vec_env.reset()
        episode_reward = 0
        steps = 0
        done = False
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = vec_env.step(action)
            episode_reward += reward[0]
            steps += 1
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(steps)
        
        if (episode + 1) % 10 == 0 or (episode + 1) == num_episodes:
            print(f"Episode {episode + 1:3d}: Reward = {episode_reward:8.2f}, Steps = {steps:4d}")
    
    vec_env.close()
    
    # Calculate statistics
    episode_rewards = np.array(episode_rewards)
    episode_lengths = np.array(episode_lengths)
    success_rate = np.sum(episode_rewards > 300) / num_episodes * 100
    
    # Print statistics
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    print(f"Success rate (>300): {success_rate:.1f}%")
    print(f"Mean reward:         {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Median reward:       {np.median(episode_rewards):.2f}")
    print(f"Min reward:          {np.min(episode_rewards):.2f}")
    print(f"Max reward:          {np.max(episode_rewards):.2f}")
    print(f"Mean length:         {np.mean(episode_lengths):.2f} ± {np.std(episode_lengths):.2f}")
    print("=" * 70)
    
    # Save results to CSV
    csv_path = output_dir / f"{model_dir.name}_evaluation.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Value'])
        writer.writerow(['Algorithm', algorithm_name])
        writer.writerow(['Environment', f"{env_name} ({mode})"])
        writer.writerow(['Episodes', num_episodes])
        writer.writerow(['Success Rate (%)', f"{success_rate:.2f}"])
        writer.writerow(['Mean Reward', f"{np.mean(episode_rewards):.2f}"])
        writer.writerow(['Median Reward', f"{np.median(episode_rewards):.2f}"])
        writer.writerow(['Std Reward', f"{np.std(episode_rewards):.2f}"])
        writer.writerow(['Min Reward', f"{np.min(episode_rewards):.2f}"])
        writer.writerow(['Max Reward', f"{np.max(episode_rewards):.2f}"])
        writer.writerow(['Mean Length', f"{np.mean(episode_lengths):.2f}"])
        writer.writerow(['Std Length', f"{np.std(episode_lengths):.2f}"])
        writer.writerow([])
        writer.writerow(['Episode', 'Reward', 'Length'])
        for i, (r, l) in enumerate(zip(episode_rewards, episode_lengths)):
            writer.writerow([i+1, f"{r:.2f}", l])
    print(f"\n✓ Results saved to {csv_path}")
    
    # Plot distributions
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Reward distribution
    axes[0].hist(episode_rewards, bins=20, edgecolor='black', alpha=0.7, color='steelblue')
    axes[0].axvline(np.mean(episode_rewards), color='red', linestyle='--', linewidth=2, 
                    label=f'Mean: {np.mean(episode_rewards):.1f}')
    axes[0].axvline(300, color='green', linestyle='--', linewidth=2, label='Success: 300')
    axes[0].set_xlabel('Reward', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Frequency', fontsize=12, fontweight='bold')
    axes[0].set_title(f'Reward Distribution\n{algorithm_name} - {mode}', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Length distribution
    axes[1].hist(episode_lengths, bins=20, edgecolor='black', alpha=0.7, color='darkorange')
    axes[1].axvline(np.mean(episode_lengths), color='red', linestyle='--', linewidth=2, 
                    label=f'Mean: {np.mean(episode_lengths):.0f}')
    axes[1].set_xlabel('Episode Length', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Frequency', fontsize=12, fontweight='bold')
    axes[1].set_title('Episode Length Distribution', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = output_dir / f"{model_dir.name}_evaluation_distribution.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Distribution plot saved to {plot_path}")
    plt.close()
    
    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained Stable Baselines3 model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/evaluate_sb3_models.py --model-dir experiments/checkpoints/td3_easy
  python scripts/evaluate_sb3_models.py --model-dir experiments/checkpoints/td3_hardcore --episodes 100
  python scripts/evaluate_sb3_models.py --model-dir experiments/checkpoints/td3_hardcore_bridges
        """
    )
    
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Path to model directory"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=50,
        help="Number of evaluation episodes (default: 50)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results (default: same as model-dir)"
    )
    
    args = parser.parse_args()
    
    evaluate_model(
        model_dir=args.model_dir,
        num_episodes=args.episodes,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
