"""Visualize and record trained Stable-Baselines3 models.

This script loads a trained SB3 model and runs it in the environment
with rendering enabled, allowing you to watch the agent's behavior.
"""

import os
import argparse
import numpy as np
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
import gymnasium as gym
from gymnasium.wrappers import RecordVideo


def make_env(env_id: str, render_mode: str = "human", hardcore: bool = False):
    """Create environment with optional rendering.

    Args:
        env_id: Gym environment ID
        render_mode: Rendering mode ('human' for display, 'rgb_array' for recording)
        hardcore: Enable hardcore mode for BipedalWalker
    """
    def _init():
        if 'BipedalWalker' in env_id:
            env = gym.make(env_id, render_mode=render_mode, hardcore=hardcore)
        else:
            env = gym.make(env_id, render_mode=render_mode)
        return env
    return _init


def visualize_model(
    model_path: str,
    vec_normalize_path: str = None,
    env_id: str = "BipedalWalker-v3",
    n_episodes: int = 5,
    deterministic: bool = True,
    record_video: bool = False,
    video_folder: str = "experiments/videos",
    hardcore: bool = False
):
    """Visualize a trained model.

    Args:
        model_path: Path to the model .zip file
        vec_normalize_path: Path to VecNormalize stats .pkl file (if used during training)
        env_id: Gym environment ID
        n_episodes: Number of episodes to run
        deterministic: Whether to use deterministic actions
        record_video: Whether to record videos
        video_folder: Folder to save videos
        hardcore: Enable hardcore mode for BipedalWalker
    """
    print("=" * 60)
    print("STABLE-BASELINES3 MODEL VISUALIZATION")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Environment: {env_id}")
    if hardcore:
        print("HARDCORE MODE - Environment with obstacles!")
    print(f"Episodes: {n_episodes}")
    print(f"Deterministic: {deterministic}")
    print("=" * 60)

    # Determine agent type from path
    if 'sac' in model_path.lower():
        agent_class = SAC
    elif 'ppo' in model_path.lower():
        agent_class = PPO
    else:
        raise ValueError("Cannot determine agent type from model path. Must contain 'sac' or 'ppo'")

    # Create environment
    render_mode = "rgb_array" if record_video else "human"

    # Wrap with RecordVideo if requested
    if record_video:
        os.makedirs(video_folder, exist_ok=True)
        # Use timestamp to create unique video names
        import time
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        video_prefix = f"{agent_class.__name__.lower()}_{'hardcore' if hardcore else 'normal'}_{timestamp}"

        # Create environment with video recording
        def make_video_env():
            if 'BipedalWalker' in env_id:
                base = gym.make(env_id, render_mode=render_mode, hardcore=hardcore)
            else:
                base = gym.make(env_id, render_mode=render_mode)

            # Wrap with RecordVideo
            wrapped = RecordVideo(
                base,
                video_folder=video_folder,
                episode_trigger=lambda episode_id: episode_id < n_episodes,  # Only record requested episodes
                name_prefix=video_prefix,
                disable_logger=True  # Reduce console spam
            )
            return wrapped

        env = DummyVecEnv([make_video_env])
        print(f"Recording videos to: {video_folder}")
        print(f"Video prefix: {video_prefix}")
    else:
        env = DummyVecEnv([make_env(env_id, render_mode=render_mode, hardcore=hardcore)])

    # Load VecNormalize stats if provided
    if vec_normalize_path and os.path.exists(vec_normalize_path):
        print(f"Loading VecNormalize stats from: {vec_normalize_path}")
        env = VecNormalize.load(vec_normalize_path, env)
        env.training = False  # Don't update stats during evaluation
        env.norm_reward = False  # Don't normalize rewards during evaluation

    # Load model
    print(f"Loading model...")
    try:
        # Try loading with environment first
        model = agent_class.load(model_path, env=env)
    except (KeyError, AttributeError, Exception) as e:
        # If that fails due to numpy compatibility, load with custom_objects
        print(f"Warning: {e}")
        print("Loading model with custom objects to handle numpy compatibility...")
        # Get spaces from the environment
        dummy_env = env.envs[0] if hasattr(env, 'envs') else env
        custom_objects = {
            "observation_space": dummy_env.observation_space,
            "action_space": dummy_env.action_space,
        }
        model = agent_class.load(model_path, env=env, custom_objects=custom_objects)
    print("✓ Model loaded successfully!\n")

    # Run episodes
    episode_rewards = []
    episode_lengths = []

    for episode in range(n_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0

        print(f"Episode {episode + 1}/{n_episodes} - Running...")

        while not done:
            # Get action from model
            action, _ = model.predict(obs, deterministic=deterministic)

            # Step environment
            obs, reward, done, info = env.step(action)
            episode_reward += reward[0]
            episode_length += 1

            # Check if episode is done
            if done[0]:
                break

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        print(f"  Reward: {episode_reward:.2f} | Length: {episode_length}")

    # Print statistics
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Episodes: {n_episodes}")
    print(f"Mean Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Min Reward: {np.min(episode_rewards):.2f}")
    print(f"Max Reward: {np.max(episode_rewards):.2f}")
    print(f"Mean Length: {np.mean(episode_lengths):.0f} ± {np.std(episode_lengths):.0f}")
    print("=" * 60)

    if record_video:
        print(f"\n✓ Videos saved to: {video_folder}")

    env.close()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize trained Stable-Baselines3 models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize best model
  python visualize_sb3.py --model experiments/checkpoints/sac_sb3_gpu/best_model.zip

  # Visualize specific checkpoint with VecNormalize
  python visualize_sb3.py \\
    --model experiments/checkpoints/sac_sb3_gpu/sac_model_200000_steps.zip \\
    --vec-normalize experiments/checkpoints/sac_sb3_gpu/sac_model_vecnormalize_200000_steps.pkl

  # Record videos
  python visualize_sb3.py \\
    --model experiments/checkpoints/sac_sb3_gpu/best_model.zip \\
    --record-video \\
    --episodes 3
        """
    )
    parser.add_argument("--model", type=str, required=True, help="Path to model .zip file")
    parser.add_argument("--vec-normalize", type=str, default=None,
                       help="Path to VecNormalize .pkl file (optional)")
    parser.add_argument("--env", type=str, default="BipedalWalker-v3",
                       help="Environment ID")
    parser.add_argument("--episodes", type=int, default=5,
                       help="Number of episodes to run")
    parser.add_argument("--stochastic", action="store_true",
                       help="Use stochastic actions instead of deterministic")
    parser.add_argument("--record-video", action="store_true",
                       help="Record videos of episodes")
    parser.add_argument("--video-folder", type=str, default="experiments/videos",
                       help="Folder to save videos")
    parser.add_argument("--hardcore", action="store_true",
                       help="Enable hardcore mode (obstacles) for BipedalWalker")
    args = parser.parse_args()

    # Auto-detect VecNormalize file if not provided
    vec_normalize_path = args.vec_normalize
    if vec_normalize_path is None:
        # Try to find corresponding vecnormalize file
        possible_path = args.model.replace('.zip', '_vecnormalize.pkl')
        if os.path.exists(possible_path):
            vec_normalize_path = possible_path
            print(f"Auto-detected VecNormalize file: {vec_normalize_path}")
        else:
            # If direct match doesn't exist, find most recent vecnormalize in same directory
            model_dir = os.path.dirname(args.model)
            if model_dir:
                import glob
                vecnorm_files = glob.glob(os.path.join(model_dir, "*_vecnormalize_*.pkl"))
                if vecnorm_files:
                    # Get most recent by modification time
                    vec_normalize_path = max(vecnorm_files, key=os.path.getmtime)
                    print(f"Auto-detected most recent VecNormalize file: {vec_normalize_path}")
                    print(f"  (No exact match found for {os.path.basename(args.model)})")

    # Auto-detect hardcore mode from model path
    hardcore = args.hardcore
    if not hardcore and 'hardcore' in args.model.lower():
        hardcore = True
        print(f"Auto-detected HARDCORE mode from model path")

    visualize_model(
        model_path=args.model,
        vec_normalize_path=vec_normalize_path,
        env_id=args.env,
        n_episodes=args.episodes,
        deterministic=not args.stochastic,
        record_video=args.record_video,
        video_folder=args.video_folder,
        hardcore=hardcore
    )


if __name__ == "__main__":
    main()
