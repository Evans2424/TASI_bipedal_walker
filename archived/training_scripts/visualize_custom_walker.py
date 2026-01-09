#!/usr/bin/env python3
"""Visualize Elite Hardcore models on Custom BipedalWalker with Bridges.

This script tests models trained on custom_walker.py (includes BRIDGES obstacle).

Usage:
    python visualize_custom_walker.py \
      --model experiments/checkpoints/sac_elite_unified_hardcore_gpu_custom_bridges/final_model/sac_model.zip \
      --vecnorm experiments/checkpoints/sac_elite_unified_hardcore_gpu_custom_bridges/final_model/vecnormalize.pkl \
      --episodes 5
"""

import gymnasium as gym
from gymnasium.envs.registration import register
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from gymnasium.wrappers import RecordVideo
import time
import argparse
import os
import numpy as np

from wrappers.elite_hardcore_wrapper import EliteHardcoreWrapper

# Register custom walker environment
register(
    id='CustomBipedalWalker-v3',
    entry_point='custom_walker:BipedalWalker',
    max_episode_steps=2000,
    reward_threshold=300,
)


def make_env(render_mode="human", record_video=False, hardcore=True):
    """Create Custom BipedalWalker environment with EliteHardcoreWrapper.

    CRITICAL: RecordVideo must wrap BEFORE EliteHardcoreWrapper to capture
    all environment frames, not just post-frame-skip frames!
    """
    def _init():
        env = gym.make("CustomBipedalWalker-v3", render_mode=render_mode, hardcore=hardcore)

        # FIXED: Apply RecordVideo BEFORE frame skip wrapper for correct speed
        if record_video:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            # RecordVideo wraps base environment (renders at 50 FPS)
            # This captures ALL frames, not just agent decision frames
            env = RecordVideo(
                env,
                video_folder="experiments/videos",
                name_prefix=f"custom_bridges_{timestamp}",
                episode_trigger=lambda x: True,
                fps=50  # Base environment FPS - will show real-time speed!
            )

        # Add elite hardcore wrapper AFTER RecordVideo
        # This way RecordVideo sees all frames, not just post-frame-skip
        env = EliteHardcoreWrapper(
            env,
            # Core hardcore features (STRONG)
            frame_skip=4,
            smoothness_coef=0.2,
            hull_angle_coef=0.1,
            hull_angular_vel_coef=0.05,
            # Natural walking augmentations (WEAK)
            knee_bend_reward=0.02,
            min_bend_threshold=0.3,
            max_joint_velocity=2.0,
            velocity_penalty=0.02,
            early_steps_stability_bonus=0.01,
            early_steps_count=100,
        )

        return env

    return _init


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--vecnorm", type=str, default=None, help="Path to VecNormalize stats")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes to run")
    parser.add_argument("--record", action="store_true", help="Record videos")
    args = parser.parse_args()

    print("=" * 60)
    print("CUSTOM WALKER VISUALIZATION - BRIDGES")
    print("=" * 60)
    print(f"Model: {args.model}")

    # Auto-detect VecNormalize
    if args.vecnorm is None:
        base_path = os.path.dirname(args.model)
        model_name = os.path.basename(args.model).replace('.zip', '')
        vecnorm_path = os.path.join(base_path, "vecnormalize.pkl")
        if os.path.exists(vecnorm_path):
            args.vecnorm = vecnorm_path

    print(f"VecNormalize: {args.vecnorm}")
    print(f"Episodes: {args.episodes}")
    print(f"Mode: HARDCORE (with BRIDGES)")
    print(f"Obstacles: GRASS, STUMP, STAIRS, PIT, **BRIDGE**")
    print("=" * 60)

    # Create environment
    render_mode = "rgb_array" if args.record else "human"
    env = DummyVecEnv([make_env(render_mode=render_mode, record_video=args.record, hardcore=True)])

    # Load VecNormalize stats
    if args.vecnorm and os.path.exists(args.vecnorm):
        print("Loading VecNormalize stats...")
        env = VecNormalize.load(args.vecnorm, env)
        env.training = False
        env.norm_reward = False

    # Load model
    print("Loading model...")
    model = SAC.load(args.model, env=env)
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

        while not done:
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

    if args.record:
        print("\n✓ Videos saved to: experiments/videos")
        print("  Note: Videos show BRIDGE obstacles (drawbridges that lower)")

    env.close()


if __name__ == "__main__":
    main()
