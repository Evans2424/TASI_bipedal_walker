#!/usr/bin/env python3
"""Record videos from a trained Stable Baselines3 model folder.

Usage:
    python scripts/evaluation/record_sb3_video.py --model-dir experiments/checkpoints/td3_easy
    python scripts/evaluation/record_sb3_video.py --model-dir experiments/checkpoints/td3_hardcore --hardcore
    python scripts/evaluation/record_sb3_video.py --model-dir experiments/checkpoints/td3_hardcore_bridges
"""

import argparse
import os
import sys
from pathlib import Path
import gymnasium as gym
from gymnasium.envs.registration import register
from stable_baselines3 import TD3, SAC, PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

# Add project root to path
project_root = Path(__file__).parent.parent.parent
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
    print("Warning: Custom walker not available")


def record_video(model_dir: str, output_dir: str = None, num_episodes: int = 3, hardcore: bool = False):
    """Record videos from a trained model.
    
    Args:
        model_dir: Path to model directory (e.g., experiments/checkpoints/td3_easy)
        output_dir: Output directory for videos (default: experiments/videos/<model_name>)
        num_episodes: Number of episodes to record
        hardcore: Whether to use hardcore mode
    """
    model_dir = Path(model_dir)
    
    if not model_dir.exists():
        print(f"Error: Model directory not found: {model_dir}")
        return
    
    # Find model file (best_model.zip or final_model.zip)
    model_file = model_dir / "best_model.zip"
    if not model_file.exists():
        model_file = model_dir / "final_model.zip"
        if not model_file.exists():
            print(f"Error: No model file found in {model_dir}")
            print("Looking for best_model.zip or final_model.zip")
            return
    
    # Find vecnormalize file
    vecnorm_file = None
    for name in ["best_model_vecnormalize.pkl", "final_model_vecnormalize.pkl"]:
        candidate = model_dir / name
        if candidate.exists():
            vecnorm_file = candidate
            break
    
    # Determine algorithm type from directory name
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
        # Try to infer from model file name
        model_name = model_file.stem.lower()
        if 'td3' in model_name:
            algorithm_class = TD3
            algorithm_name = "TD3"
        elif 'sac' in model_name:
            algorithm_class = SAC
            algorithm_name = "SAC"
        elif 'ppo' in model_name:
            algorithm_class = PPO
            algorithm_name = "PPO"
        else:
            # Default to TD3
            algorithm_class = TD3
            algorithm_name = "TD3"
            print("Warning: Could not determine algorithm, defaulting to TD3")
    
    # Set output directory
    if output_dir is None:
        output_dir = Path("experiments/videos") / model_dir.name
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if this is a bridge model
    use_bridges = 'bridge' in dir_name
    
    # Validate bridges support
    if use_bridges and not CUSTOM_WALKER_AVAILABLE:
        print("Error: Bridge models require custom walker, but it's not available")
        return
    
    # Determine environment and mode
    if use_bridges:
        env_name = "CustomBipedalWalker-v3"
        mode = "Hardcore with Bridges"
        env_hardcore = True
    elif 'hardcore' in dir_name or hardcore:
        env_name = "BipedalWalkerHardcore-v3"
        mode = "Hardcore"
        env_hardcore = True
    else:
        env_name = "BipedalWalker-v3"
        mode = "Easy"
        env_hardcore = False
    
    print("=" * 60)
    print("RECORDING VIDEO")
    print("=" * 60)
    print(f"Algorithm: {algorithm_name}")
    print(f"Model: {model_file}")
    print(f"VecNormalize: {vecnorm_file if vecnorm_file else 'None'}")
    print(f"Environment: {env_name} ({mode})")
    print(f"Episodes: {num_episodes}")
    print(f"Output: {output_dir}")
    print("=" * 60)
    print()
    
    # Load model
    print(f"Loading {algorithm_name} model...")
    model = algorithm_class.load(str(model_file))
    
    # Create environment based on mode
    print("Creating environment...")
    
    if use_bridges:
        # Create custom walker with bridge wrapper
        base_env = gym.make(env_name, hardcore=True, render_mode='rgb_array')
        
        # Apply BridgeBalancedWrapper
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
        
        # Add Monitor wrapper
        env = Monitor(wrapped_env)
    elif env_hardcore:
        # Create hardcore environment with HardcoreWrapper
        base_env = gym.make(env_name, render_mode='rgb_array')
        
        # Apply HardcoreWrapper
        wrapped_env = HardcoreWrapper(
            base_env,
            frame_skip=4,
            smoothness_coef=0.05,
            angle_coef=0.02,
            angular_vel_coef=0.01,
            reward_clip_min=-10.0,
            reward_clip_max=10.0,
        )
        
        # Add Monitor wrapper
        env = Monitor(wrapped_env)
    else:
        # Easy mode - no wrappers
        env = gym.make(env_name, render_mode='rgb_array')
    
    # Apply video recording wrapper
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=str(output_dir),
        episode_trigger=lambda x: True,  # Record all episodes
        name_prefix=f"{algorithm_name.lower()}_{model_dir.name}"
    )
    
    # Wrap in VecEnv
    vec_env = DummyVecEnv([lambda: env])
    
    # Load VecNormalize stats if available
    if vecnorm_file:
        print(f"Loading VecNormalize stats...")
        vec_env = VecNormalize.load(str(vecnorm_file), vec_env)
        vec_env.training = False
        vec_env.norm_reward = False
    
    # Record episodes
    print(f"\nRecording {num_episodes} episodes...\n")
    episode_rewards = []
    
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
        print(f"Episode {episode + 1}: Reward = {episode_reward:8.2f}, Steps = {steps}")
    
    vec_env.close()
    
    # Summary
    import numpy as np
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Mean Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Min Reward:  {np.min(episode_rewards):.2f}")
    print(f"Max Reward:  {np.max(episode_rewards):.2f}")
    print(f"\nVideos saved to: {output_dir}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Record videos from trained Stable Baselines3 model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/evaluation/record_sb3_video.py --model-dir experiments/checkpoints/td3_easy
  python scripts/evaluation/record_sb3_video.py --model-dir experiments/checkpoints/td3_hardcore --hardcore
  python scripts/evaluation/record_sb3_video.py --model-dir experiments/checkpoints/sac_easy --episodes 5
        """
    )
    
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Path to model directory containing best_model.zip or final_model.zip"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for videos (default: experiments/videos/<model_name>)"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=3,
        help="Number of episodes to record (default: 3)"
    )
    parser.add_argument(
        "--hardcore",
        action="store_true",
        help="Force hardcore mode (auto-detected from directory name)"
    )
    
    args = parser.parse_args()
    
    record_video(
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        num_episodes=args.episodes,
        hardcore=args.hardcore
    )


if __name__ == "__main__":
    main()
