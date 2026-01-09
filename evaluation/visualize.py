#!/usr/bin/env python3
"""Visualize trained SAC models - Supports Easy, Hardcore, and Bridges modes

Usage:
    # Visualize with config (recommended - loads all wrapper settings)
    python evaluation/visualize.py --config configs/sac_elite_hardcore_gpu.yaml --checkpoint experiments/checkpoints/sac_hardcore_WORKING/final_model.zip --episodes 10
    
    # Visualize specific checkpoint (legacy mode)
    python evaluation/visualize.py --checkpoint experiments/checkpoints/sac_easy/best_model.zip --mode easy
    python evaluation/visualize.py --checkpoint experiments/checkpoints/sac_hardcore/best_model.zip --mode hardcore --hardcore
    python evaluation/visualize.py --checkpoint experiments/checkpoints/sac_bridges/best_model.zip --mode bridges
    
    # Record video
    python evaluation/visualize.py --config configs/sac_elite_hardcore_gpu.yaml --checkpoint path/to/model.zip --record --episodes 5
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
import yaml
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wrappers.bridge_balanced_wrapper import BridgeBalancedWrapper
from wrappers.elite_hardcore_wrapper import EliteHardcoreWrapper

# Register custom walker for bridges mode
register(
    id='CustomBipedalWalker-v3',
    entry_point='src.envs.custom_walker:BipedalWalker',
    max_episode_steps=2000,
    reward_threshold=300,
)


def make_env(mode="easy", render_mode="human", record_video=False, config=None, use_elite_hardcore=False):
    """Create environment based on mode.
    
    Args:
        mode: 'easy', 'hardcore', or 'bridges'
        render_mode: 'human' or 'rgb_array'
        record_video: Whether to record video
        config: Config dict with wrapper settings (if provided)
        use_elite_hardcore: Whether to use Elite Hardcore Wrapper for hardcore mode
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

        # Apply wrappers based on mode and config
        if mode == "bridges":
            # Apply bridge wrapper for bridges mode
            if config and 'env' in config:
                env_cfg = config['env']
                env = BridgeBalancedWrapper(
                    env,
                    frame_skip=env_cfg.get('frame_skip', 4),
                    smoothness_coef=env_cfg.get('smoothness_coef', 0.02),
                    hull_angle_coef=env_cfg.get('hull_angle_coef', 0.03),
                    hull_angular_vel_coef=env_cfg.get('hull_angular_vel_coef', 0.015),
                    knee_bend_reward=env_cfg.get('knee_bend_reward', 0.02),
                    min_bend_threshold=env_cfg.get('min_bend_threshold', 0.3),
                    stable_waiting_bonus=env_cfg.get('stable_waiting_bonus', 0.02),
                    bridge_cross_bonus=env_cfg.get('bridge_cross_bonus', 8.0),
                    min_progress_for_bonuses=env_cfg.get('min_progress_for_bonuses', 15.0),
                    max_waiting_steps=env_cfg.get('max_waiting_steps', 400),
                )
            else:
                # Default bridge wrapper settings
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
                )
        
        elif mode == "hardcore" and (use_elite_hardcore or (config and config.get('env', {}).get('use_elite_hardcore', False))):
            # Apply Elite Hardcore Wrapper for hardcore mode
            if config and 'env' in config:
                env_cfg = config['env']
                env = EliteHardcoreWrapper(
                    env,
                    frame_skip=env_cfg.get('frame_skip', 4),
                    smoothness_coef=env_cfg.get('smoothness_coef', 0.2),
                    hull_angle_coef=env_cfg.get('hull_angle_coef', 0.1),
                    hull_angular_vel_coef=env_cfg.get('hull_angular_vel_coef', 0.05),
                    knee_bend_reward=env_cfg.get('knee_bend_reward', 0.02),
                    min_bend_threshold=env_cfg.get('min_bend_threshold', 0.3),
                    max_joint_velocity=env_cfg.get('max_joint_velocity', 2.0),
                    velocity_penalty=env_cfg.get('velocity_penalty', 0.02),
                    early_steps_stability_bonus=env_cfg.get('early_steps_stability_bonus', 0.01),
                    early_steps_count=env_cfg.get('early_steps_count', 100)
                )
            else:
                # Default Elite Hardcore settings
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
                    early_steps_count=100
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
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML file (recommended)")
    parser.add_argument("--mode", type=str, choices=["easy", "hardcore", "bridges"], default=None,
                        help="Environment mode (auto-detected from config if not provided)")
    parser.add_argument("--hardcore", action="store_true", help="Use Elite Hardcore Wrapper (ignored if --config provided)")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes")
    parser.add_argument("--record", action="store_true", help="Record videos")
    parser.add_argument("--deterministic", action="store_true", default=True, help="Use deterministic actions")
    args = parser.parse_args()

    # Load config if provided
    config = None
    if args.config:
        print(f"Loading config from: {args.config}")
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    
    # Determine mode
    mode = args.mode
    if config and mode is None:
        # Auto-detect mode from config
        env_name = config.get('env', {}).get('name', 'BipedalWalker-v3')
        is_hardcore = config.get('env', {}).get('hardcore', False)
        use_bridge = config.get('env', {}).get('use_bridge_wrapper', False)
        
        if use_bridge:
            mode = "bridges"
        elif is_hardcore:
            mode = "hardcore"
        else:
            mode = "easy"
    
    if mode is None:
        print("ERROR: Must provide either --config or --mode")
        return

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

    # Check wrapper configuration
    use_elite = False
    if config:
        use_elite = config.get('env', {}).get('use_elite_hardcore', False)
    elif args.hardcore:
        use_elite = True

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Mode: {mode.upper()}")
    if mode == "hardcore":
        print(f"Elite Hardcore Wrapper: {'YES' if use_elite else 'NO'}")
    print(f"Episodes: {args.episodes}")
    print(f"Deterministic: {args.deterministic}")
    if args.record:
        print(f"Recording videos to: experiments/videos/")
    print("=" * 60)

    # Create environment
    render_mode = "rgb_array" if args.record else "human"
    env = DummyVecEnv([make_env(mode=mode, render_mode=render_mode, record_video=args.record, 
                                 config=config, use_elite_hardcore=use_elite)])

    # Try to load VecNormalize stats
    vecnorm_paths = [
        checkpoint_path.parent / "vec_normalize.pkl",
        checkpoint_path.parent / f"{checkpoint_path.stem}_vecnormalize.pkl",
        checkpoint_path.parent / "best_model_vecnormalize.pkl",
        checkpoint_path.parent / "final_model_vecnormalize.pkl",
    ]
    
    # Also search for pattern: sac_model_vecnormalize_XXXXXX_steps.pkl
    if checkpoint_path.stem.startswith("sac_model_"):
        # Extract step number and build vecnormalize filename
        parts = checkpoint_path.stem.split("_")
        if len(parts) >= 3 and parts[-1] == "steps":
            steps = parts[-2]
            vecnorm_name = f"sac_model_vecnormalize_{steps}_steps.pkl"
            vecnorm_paths.insert(0, checkpoint_path.parent / vecnorm_name)
    
    # Also do a wildcard search in parent directory
    import glob
    vecnorm_files = list(Path(checkpoint_path.parent).glob("*vecnormalize*.pkl"))
    vecnorm_paths.extend(vecnorm_files)
    
    vecnorm_loaded = False
    for vecnorm_path in vecnorm_paths:
        if vecnorm_path.exists():
            print(f"Loading VecNormalize from: {vecnorm_path.name}")
            try:
                env = VecNormalize.load(str(vecnorm_path), env)
                env.training = False
                env.norm_reward = False
                vecnorm_loaded = True
                print("✓ VecNormalize loaded successfully")
                break
            except Exception as e:
                print(f"Warning: Failed to load {vecnorm_path.name}: {e}")
                continue
    
    if not vecnorm_loaded:
        print("WARNING: No VecNormalize stats found - using raw observations")
        print("This will likely cause poor performance!")

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
