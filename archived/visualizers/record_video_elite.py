"""Manual video recording - saves each episode separately."""

import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import argparse
import time
import numpy as np

from wrappers.elite_hardcore_wrapper import EliteHardcoreWrapper


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--vecnorm", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=5)
    args = parser.parse_args()

    print("=" * 60)
    print("MANUAL VIDEO RECORDING")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"VecNormalize: {args.vecnorm}")
    print(f"Episodes: {args.episodes}")
    print("=" * 60)

    # Load model
    def make_eval_env():
        env = gym.make("BipedalWalker-v3", hardcore=True)
        env = EliteHardcoreWrapper(
            env,
            frame_skip=4,
            smoothness_coef=0.2,
            hull_angle_coef=0.1,
            hull_angular_vel_coef=0.05,
            knee_bend_reward=0.02,
            min_bend_threshold=0.3,
            max_joint_velocity=2.0,
            velocity_penalty=0.02,
            early_steps_stability_bonus=0.01,
            early_steps_count=100,
        )
        return env

    vec_env = DummyVecEnv([make_eval_env])
    vec_env = VecNormalize.load(args.vecnorm, vec_env)
    vec_env.training = False
    vec_env.norm_reward = False

    print("Loading model...")
    model = SAC.load(args.model, env=vec_env)
    print("✓ Model loaded!\n")

    # Record each episode
    rewards = []
    lengths = []
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    for episode in range(args.episodes):
        print(f"Episode {episode + 1}/{args.episodes} - Recording...")

        # Create rendering environment
        render_env = gym.make("BipedalWalker-v3", render_mode="rgb_array", hardcore=True)

        # Apply RecordVideo BEFORE EliteHardcoreWrapper (before frame skip!)
        # This captures ALL 50 FPS frames, not just agent decisions
        render_env = RecordVideo(
            render_env,
            video_folder="experiments/videos",
            name_prefix=f"final_{timestamp}_ep{episode}",
            episode_trigger=lambda x: x == 0,  # Only first episode in this env
            fps=50  # Captures all environment frames at 50 FPS
        )

        # Apply EliteHardcoreWrapper AFTER RecordVideo
        render_env = EliteHardcoreWrapper(
            render_env,
            frame_skip=4,
            smoothness_coef=0.2,
            hull_angle_coef=0.1,
            hull_angular_vel_coef=0.05,
            knee_bend_reward=0.02,
            min_bend_threshold=0.3,
            max_joint_velocity=2.0,
            velocity_penalty=0.02,
            early_steps_stability_bonus=0.01,
            early_steps_count=100,
        )

        # Wrap for normalization
        single_vec = DummyVecEnv([lambda: render_env])
        single_vec = VecNormalize.load(args.vecnorm, single_vec)
        single_vec.training = False
        single_vec.norm_reward = False

        # Run episode (RecordVideo handles frame capture automatically)
        obs = single_vec.reset()
        done = False
        total_reward = 0
        steps = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = single_vec.step(action)
            total_reward += reward[0]
            steps += 1

            if done:
                break

        rewards.append(total_reward)
        lengths.append(steps)
        print(f"  Reward: {total_reward:.2f} | Length: {steps}")

        single_vec.close()

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Mean Reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"Mean Length: {np.mean(lengths):.0f} ± {np.std(lengths):.0f}")
    print("=" * 60)
    print(f"\n✓ {args.episodes} videos saved to: experiments/videos")
    print(f"  Pattern: final_{timestamp}_ep*-episode-0.mp4")
    print(f"  Speed: 50 FPS (real-time, matches live visualization)")

    vec_env.close()


if __name__ == "__main__":
    main()
