#!/usr/bin/env python3
"""Visualize trained bridge walker models - FINAL CLEAN VERSION

Usage:
    # Visualize latest checkpoint
    python evaluation/visualize.py

    # Visualize specific checkpoint
    python evaluation/visualize.py --checkpoint experiments/checkpoints/.../sac_model_1000000_steps.zip

    # Record video
    python evaluation/visualize.py --record --episodes 5
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
from src.envs.custom_walker import BipedalWalker

# Register custom walker
register(
    id='CustomBipedalWalker-v3',
    entry_point='src.envs.custom_walker:BipedalWalker',
    max_episode_steps=2000,
    reward_threshold=300,
)


def make_env(render_mode="human", record_video=False, hardcore=True, wrapper_type="balanced"):
    """Create environment with bridge balanced wrapper."""
    def _init():
        env = gym.make("CustomBipedalWalker-v3", render_mode=render_mode, hardcore=hardcore)

        if record_video:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            env = RecordVideo(
                env,
                video_folder="experiments/videos",
                name_prefix=f"bridge_walker_{timestamp}",
                episode_trigger=lambda x: True,
                fps=50
            )

        # Apply bridge balanced wrapper
        env = BridgeBalancedWrapper(
            env,
            frame_skip=4,
            smoothness_coef=0.05,
            hull_angle_coef=0.05,
            hull_angular_vel_coef=0.02,
            knee_bend_reward=0.01,
            max_joint_velocity=3.0,
            velocity_penalty=0.01,
        )
                bridge_cross_bonus=5.0,
            )

        return env

    return _init


def find_latest_checkpoint(checkpoint_dir="experiments/checkpoints"):
    """Find the latest checkpoint in all subdirectories."""
    checkpoint_path = Path(checkpoint_dir)
    all_checkpoints = []

    for subdir in checkpoint_path.glob("*custom_bridges"):
        for ckpt in subdir.glob("sac_model_*_steps.zip"):
            steps = int(ckpt.stem.split("_")[2])
            all_checkpoints.append((steps, ckpt))

    if not all_checkpoints:
        return None, None

    all_checkpoints.sort(key=lambda x: x[0], reverse=True)
    return all_checkpoints[0][1], all_checkpoints[0][0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes")
    parser.add_argument("--record", action="store_true", help="Record videos")
    parser.add_argument("--wrapper", type=str, choices=["shaped", "optimized"], default="auto",
                        help="Wrapper type (auto-detected from checkpoint path)")
    args = parser.parse_args()

    print("=" * 60)
    print("BRIDGE WALKER VISUALIZATION")
    print("=" * 60)

    # Find checkpoint
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        steps = int(checkpoint_path.stem.split("_")[2]) if "_steps" in checkpoint_path.stem else "unknown"
    else:
        checkpoint_path, steps = find_latest_checkpoint()
        if checkpoint_path is None:
            print("ERROR: No checkpoints found!")
            return
        print(f"Using latest checkpoint: {checkpoint_path}")

    print(f"Training steps: {steps:,}" if isinstance(steps, int) else f"Training steps: {steps}")
    print(f"Episodes: {args.episodes}")
    print(f"Mode: HARDCORE with BRIDGES")

    # Auto-detect wrapper type from checkpoint path
    wrapper_type = args.wrapper
    if wrapper_type == "auto":
        if "optimized" in str(checkpoint_path):
            wrapper_type = "optimized"
            print(f"Wrapper: BridgeOptimizedWrapper (auto-detected)")
        else:
            wrapper_type = "shaped"
            print(f"Wrapper: BridgeShapedWrapper (auto-detected)")
    else:
        print(f"Wrapper: {wrapper_type} (manual)")

    print("=" * 60)

    # Create environment
    render_mode = "rgb_array" if args.record else "human"
    env = DummyVecEnv([make_env(render_mode=render_mode, record_video=args.record,
                                 hardcore=True, wrapper_type=wrapper_type)])

    # Load VecNormalize (CRITICAL for correct evaluation)
    # Try both naming patterns
    vecnorm_path = checkpoint_path.parent / f"sac_model_vecnormalize_{steps}_steps.pkl"
    if not vecnorm_path.exists():
        vecnorm_path = checkpoint_path.parent / f"{checkpoint_path.stem}_vecnormalize.pkl"

    if vecnorm_path.exists():
        print(f"Loading VecNormalize from: {vecnorm_path.name}")
        env = VecNormalize.load(str(vecnorm_path), env)
        env.training = False
        env.norm_reward = False
        print("✓ VecNormalize loaded")
    else:
        print("⚠ WARNING: No VecNormalize found - results may be inaccurate!")
        print(f"   Looked for: {vecnorm_path.name}")

    # Load model
    print("Loading model...")
    model = SAC.load(str(checkpoint_path), env=env)
    print("✓ Model loaded!\n")

    # Run episodes
    rewards = []
    lengths = []

    for episode in range(args.episodes):
        print(f"Episode {episode + 1}/{args.episodes} - Running...")
        obs = env.reset()
        done = False
        total_reward = 0
        steps = 0

        while not done and steps < 2000:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward[0]
            steps += 1

            if done:
                break

        rewards.append(total_reward)
        lengths.append(steps)
        print(f"  Reward: {total_reward:.2f} | Length: {steps}")

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Mean Reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"Mean Length: {np.mean(lengths):.0f} ± {np.std(lengths):.0f}")
    print("=" * 60)

    # Episode distribution
    short = sum(1 for l in lengths if l < 100)
    medium = sum(1 for l in lengths if 100 <= l < 500)
    long = sum(1 for l in lengths if l >= 500)

    print(f"\nEpisode Distribution:")
    print(f"  < 100 steps:    {short}/{args.episodes} ({short/args.episodes*100:.0f}%)")
    print(f"  100-500 steps:  {medium}/{args.episodes} ({medium/args.episodes*100:.0f}%)")
    print(f"  >= 500 steps:   {long}/{args.episodes} ({long/args.episodes*100:.0f}%)")

    if args.record:
        print(f"\n✓ Videos saved to: experiments/videos")
        print("  Watch to see bridge waiting and crossing behavior!")

    env.close()


if __name__ == "__main__":
    main()
