#!/usr/bin/env python3
"""Visualize trained SAC models - Supports Easy, Hardcore, and Bridges modes

Usage:
    # Visualize specific checkpoint
    python evaluation/visualize.py --checkpoint experiments/checkpoints/sac_easy/best_model.zip --mode easy
    python evaluation/visualize.py --checkpoint experiments/checkpoints/sac_hardcore/best_model.zip --mode hardcore
    python evaluation/visualize.py --checkpoint experiments/checkpoints/sac_bridges/best_model.zip --mode bridges
    
    # Record video
    python evaluation/visualize.py --checkpoint path/to/model.zip --mode bridges --record --episodes 5
"""

import gymnasium as gym
from gymnasium.envs.registration import register
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from gymnasium.wrappers import RecordVideo
import time
import argparse
import os
import sys
import numpy as np
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wrappers.bridge_balanced_wrapper import BridgeBalancedWrapper

# Register custom walker for bridges mode
register(
    id='CustomBipedalWalker-v3',
    entry_point='src.envs.custom_walker:BipedalWalker',
    max_episode_steps=2000,
    reward_threshold=300,
)


def make_env(mode="easy", render_mode="human", record_video=False):
    """Create environment based on mode.
    
    Args:
        mode: 'easy', 'hardcore', or 'bridges'
        render_mode: 'human' or 'rgb_array'
        record_video: Whether to record video
    """
    def _init():
        # Determine environment and settings
        if mode == "bridges":
            env = gym.make("CustomBipedalWalker-v3", render_mode=render_mode, hardcore=True)
        elif mode == "hardcore":
            env = gym.make("BipedalWalker-v3", render_mode=render_mode, hardcore=True)
        else:  # easy
            env = gym.make("BipedalWalker-v3", render_mode=render_mode, hardcore=False)

        if record_video:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            env = RecordVideo(
                env,
                video_folder="experiments/videos",
                name_prefix=f"{mode}_walker_{timestamp}",
                episode_trigger=lambda x: True,
                fps=50
            )

        # Apply bridge wrapper only for bridges mode
        if mode == "bridges":
            env = BridgeBalancedWrapper(
                env,
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

        return env

    return _init


def find_best_model(checkpoint_dir):
    """Find best_model.zip in checkpoint directory."""
    checkpoint_path = Path(checkpoint_dir)
    
    # Look for best_model.zip
    best_model = checkpoint_path / "best_model.zip"
    if best_model.exists():
        return best_model
    
    # Look for final_model.zip
    final_model = checkpoint_path / "final_model.zip"
    if final_model.exists():
        return final_model
    
    # Look for any .zip file
    zip_files = list(checkpoint_path.glob("*.zip"))
    if zip_files:
        # Return the one with highest number
        zip_files.sort(key=lambda x: x.stem)
        return zip_files[-1]
    
    return None


def main():
    parser = argparse.ArgumentParser(description="Visualize trained SAC models")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint or checkpoint directory")
    parser.add_argument("--mode", type=str, choices=["easy", "hardcore", "bridges"], required=True,
                        help="Environment mode")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes")
    parser.add_argument("--record", action="store_true", help="Record videos")
    parser.add_argument("--deterministic", action="store_true", default=True, help="Use deterministic actions")
    args = parser.parse_args()

    print("=" * 60)
    print("SAC BIPEDAL WALKER VISUALIZATION")
    print("=" * 60)

    # Find checkpoint
    checkpoint_path = Path(args.checkpoint)
    if checkpoint_path.is_dir():
        checkpoint_path = find_best_model(checkpoint_path)
        if checkpoint_path is None:
            print(f"ERROR: No model found in {args.checkpoint}")
            return
        print(f"Found model: {checkpoint_path}")
    
    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        return

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Mode: {args.mode.upper()}")
    print(f"Episodes: {args.episodes}")
    print(f"Deterministic: {args.deterministic}")
    if args.record:
        print(f"Recording videos to: experiments/videos/")
    print("=" * 60)

    # Create environment
    render_mode = "rgb_array" if args.record else "human"
    env = DummyVecEnv([make_env(mode=args.mode, render_mode=render_mode, record_video=args.record)])

    # Try to load VecNormalize stats
    vecnorm_paths = [
        checkpoint_path.parent / f"{checkpoint_path.stem}_vecnormalize.pkl",
        checkpoint_path.parent / "best_model_vecnormalize.pkl",
        checkpoint_path.parent / "final_model_vecnormalize.pkl",
    ]
    
    vecnorm_loaded = False
    for vecnorm_path in vecnorm_paths:
        if vecnorm_path.exists():
            print(f"Loading VecNormalize from: {vecnorm_path.name}")
            env = VecNormalize.load(str(vecnorm_path), env)
            env.training = False
            env.norm_reward = False
            vecnorm_loaded = True
            break
    
    if not vecnorm_loaded:
        print("WARNING: No VecNormalize stats found - using raw observations")

    # Load model
    print(f"Loading model...")
    try:
        model = SAC.load(str(checkpoint_path), env=env)
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"ERROR loading model: {e}")
        return

    # Run episodes
    print("\nRunning episodes...\n")
    total_rewards = []
    
    for episode in range(args.episodes):
        obs = env.reset()
        episode_reward = 0
        steps = 0
        done = False
        
        while not done:
            action, _states = model.predict(obs, deterministic=args.deterministic)
            obs, reward, done, info = env.step(action)
            episode_reward += reward[0]
            steps += 1
            
            if done:
                break
        
        total_rewards.append(episode_reward)
        print(f"Episode {episode + 1}/{args.episodes}: Reward = {episode_reward:.2f}, Steps = {steps}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Mean Reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    print(f"Min Reward: {np.min(total_rewards):.2f}")
    print(f"Max Reward: {np.max(total_rewards):.2f}")
    print("=" * 60)
    
    if args.record:
        print(f"\n✓ Videos saved to: experiments/videos/")
    
    env.close()


if __name__ == "__main__":
    main()
