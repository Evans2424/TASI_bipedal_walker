"""Evaluate Stable-Baselines3 trained models."""

import argparse
import yaml
import numpy as np
import csv
import matplotlib.pyplot as plt
from pathlib import Path
import gymnasium as gym
from stable_baselines3 import SAC, TD3, PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import sys
sys.path.append(str(Path(__file__).parent.parent))

# Import wrappers
from src.wrappers.elite_hardcore_wrapper import EliteHardcoreWrapper
from src.wrappers.bridge_balanced_wrapper import BridgeBalancedWrapper

# Register custom walker for bridges mode
from gymnasium.envs.registration import register
try:
    register(
        id='CustomBipedalWalker-v3',
        entry_point='src.envs.custom_walker:BipedalWalker',
        max_episode_steps=2000,
        reward_threshold=300,
    )
except:
    pass  # Already registered


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def make_eval_env(config: dict, seed: int = 44):
    """Create evaluation environment with proper wrapper."""
    env_name = config['env']['name']
    hardcore = config['env'].get('hardcore', False)
    use_elite_hardcore = config['env'].get('use_elite_hardcore', False)  # Default to False for easy mode
    use_bridge_wrapper = config['env'].get('use_bridge_wrapper', False)  # Bridge mode
    
    def _init():
        env = gym.make(env_name, hardcore=hardcore)
        
        # Apply bridge wrapper if enabled
        if use_bridge_wrapper:
            env = BridgeBalancedWrapper(
                env,
                frame_skip=config['env'].get('frame_skip', 4),
                smoothness_coef=config['env'].get('smoothness_coef', 0.05),
                hull_angle_coef=config['env'].get('hull_angle_coef', 0.03),
                hull_angular_vel_coef=config['env'].get('hull_angular_vel_coef', 0.015),
                knee_bend_reward=config['env'].get('knee_bend_reward', 0.02),
                min_bend_threshold=config['env'].get('min_bend_threshold', 0.3),
                max_joint_velocity=config['env'].get('max_joint_velocity', 2.0),
                velocity_penalty=config['env'].get('velocity_penalty', 0.02),
                stable_waiting_bonus=config['env'].get('stable_waiting_bonus', 0.02),
                bridge_cross_bonus=config['env'].get('bridge_cross_bonus', 8.0),
                min_progress_for_bonuses=config['env'].get('min_progress_for_bonuses', 15.0),
                max_waiting_steps=config['env'].get('max_waiting_steps', 400),
                lidar_bridge_threshold=config['env'].get('lidar_bridge_threshold', 0.5),
                min_close_beams=config['env'].get('min_close_beams', 3),
                waiting_velocity_threshold=config['env'].get('waiting_velocity_threshold', 0.15),
                waiting_angle_threshold=config['env'].get('waiting_angle_threshold', 0.3),
            )
        # Apply Elite Hardcore wrapper if enabled (and not using bridge wrapper)
        elif use_elite_hardcore and hardcore:
            # Apply Elite Hardcore Wrapper
            env = EliteHardcoreWrapper(
                env,
                frame_skip=config['env'].get('frame_skip', 4),
                smoothness_coef=config['env'].get('smoothness_coef', 0.2),
                hull_angle_coef=config['env'].get('hull_angle_coef', 0.1),
                hull_angular_vel_coef=config['env'].get('hull_angular_vel_coef', 0.05),
                knee_bend_reward=config['env'].get('knee_bend_reward', 0.02),
                min_bend_threshold=config['env'].get('min_bend_threshold', 0.3),
                max_joint_velocity=config['env'].get('max_joint_velocity', 2.0),
                velocity_penalty=config['env'].get('velocity_penalty', 0.02),
                early_steps_stability_bonus=config['env'].get('early_steps_stability_bonus', 0.01),
                early_steps_count=config['env'].get('early_steps_count', 100)
            )
        
        return env
    
    env = DummyVecEnv([_init])
    return env


def evaluate_sb3_model(
    model_path: str,
    config_path: str,
    vec_normalize_path: str = None,
    num_episodes: int = 10,
    render: bool = False,
    output_name: str = None,
    seed: int = 45
):
    """Evaluate a Stable-Baselines3 trained model.
    
    Args:
        model_path: Path to .zip model file
        config_path: Path to config YAML file
        vec_normalize_path: Path to vec_normalize.pkl (if None, tries to find it automatically)
        num_episodes: Number of evaluation episodes
        render: Whether to render (not supported in vec env)
        output_name: Base name for output files
        seed: Random seed for evaluation (default: 45)
    """
    # Load config
    config = load_config(config_path)
    
    # Try to find vec_normalize.pkl if not provided
    if vec_normalize_path is None:
        model_dir = Path(model_path).parent
        model_stem = Path(model_path).stem  # e.g., "sac_model_2000000_steps"
        
        # Try multiple naming patterns
        possible_names = [
            f"{model_stem.replace('sac_model', 'sac_model_vecnormalize')}.pkl",  # Match exact name
            "vec_normalize.pkl",  # Standard name
            f"{model_stem}_vecnormalize.pkl",  # Alternative pattern
            "best_model_vecnormalize.pkl",
            "final_model_vecnormalize.pkl",
        ]
        
        vec_normalize_path = None
        for name in possible_names:
            candidate = model_dir / name
            if candidate.exists():
                vec_normalize_path = candidate
                break
        
        if vec_normalize_path is None:
            print(f"Warning: VecNormalize file not found in {model_dir}")
            print(f"Tried: {', '.join(possible_names[:3])}")
            print("Model may not work properly without normalization stats!")
    
    # Determine algorithm from model_path
    model_path_lower = str(model_path).lower()
    if 'sac' in model_path_lower:
        algorithm = SAC
        algo_name = 'SAC'
    elif 'td3' in model_path_lower:
        algorithm = TD3
        algo_name = 'TD3'
    elif 'ppo' in model_path_lower:
        algorithm = PPO
        algo_name = 'PPO'
    else:
        print("Could not determine algorithm from path. Defaulting to SAC.")
        algorithm = SAC
        algo_name = 'SAC'
    
    print("\n" + "="*70)
    print("Stable-Baselines3 Model Evaluation")
    print("="*70)
    print(f"Algorithm:       {algo_name}")
    print(f"Model:           {model_path}")
    print(f"Config:          {config_path}")
    print(f"VecNormalize:    {vec_normalize_path}")
    print(f"Environment:     {config['env']['name']}")
    print(f"Hardcore:        {config['env'].get('hardcore', False)}")
    print(f"Elite Wrapper:   {config['env'].get('use_elite_hardcore', True)}")
    print(f"Episodes:        {num_episodes}")
    print(f"Seed:            {seed}")
    print("="*70 + "\n")
    
    # Create environment
    env = make_eval_env(config, seed=seed)
    
    # Load VecNormalize if available
    if vec_normalize_path and Path(vec_normalize_path).exists():
        print(f"Loading VecNormalize statistics from {vec_normalize_path}...")
        env = VecNormalize.load(vec_normalize_path, env)
        env.training = False  # Don't update stats during evaluation
        env.norm_reward = False  # Don't normalize rewards during evaluation
    
    # Load model
    print(f"Loading {algo_name} model from {model_path}...")
    model = algorithm.load(model_path, env=env)
    
    # Evaluate
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(num_episodes):
        obs = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_reward += reward[0]  # VecEnv returns array
            episode_length += 1
            
            if done[0]:
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        print(f"Episode {episode + 1:2d}: Reward = {episode_reward:7.2f}, Length = {episode_length:4d}")
    
    # Calculate statistics
    episode_rewards = np.array(episode_rewards)
    episode_lengths = np.array(episode_lengths)
    success_rate = np.sum(episode_rewards > 300) / num_episodes * 100
    
    # Print statistics
    print("\n" + "="*70)
    print("Evaluation Results")
    print("="*70)
    print(f"Success rate (>300):     {success_rate:.1f}%")
    print(f"Mean reward:             {np.mean(episode_rewards):7.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Median reward:           {np.median(episode_rewards):7.2f}")
    print(f"Min reward:              {np.min(episode_rewards):7.2f}")
    print(f"Max reward:              {np.max(episode_rewards):7.2f}")
    print(f"Mean episode length:     {np.mean(episode_lengths):.1f}")
    print("="*70 + "\n")
    
    # Save results to CSV if output_name provided
    if output_name:
        # Create output directory if needed
        output_path = Path(output_name)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        csv_path = f"{output_name}.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Metric', 'Value'])
            writer.writerow(['Algorithm', algo_name])
            writer.writerow(['Success Rate (%)', f"{success_rate:.2f}"])
            writer.writerow(['Mean Reward', f"{np.mean(episode_rewards):.2f}"])
            writer.writerow(['Median Reward', f"{np.median(episode_rewards):.2f}"])
            writer.writerow(['Std Reward', f"{np.std(episode_rewards):.2f}"])
            writer.writerow(['Min Reward', f"{np.min(episode_rewards):.2f}"])
            writer.writerow(['Max Reward', f"{np.max(episode_rewards):.2f}"])
            writer.writerow(['Mean Length', f"{np.mean(episode_lengths):.2f}"])
            writer.writerow([])
            writer.writerow(['Episode', 'Reward', 'Length'])
            for i, (r, l) in enumerate(zip(episode_rewards, episode_lengths)):
                writer.writerow([i+1, f"{r:.2f}", l])
        print(f"Results saved to {csv_path}")
        
        # Plot distributions
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Reward distribution
        axes[0].hist(episode_rewards, bins=10, edgecolor='black', alpha=0.7)
        axes[0].axvline(np.mean(episode_rewards), color='red', linestyle='--', 
                       linewidth=2, label=f'Mean: {np.mean(episode_rewards):.1f}')
        axes[0].axvline(300, color='green', linestyle='--', linewidth=2, label='Success: 300')
        axes[0].set_xlabel('Reward', fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].set_title(f'{algo_name} Reward Distribution', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Length distribution
        axes[1].hist(episode_lengths, bins=10, edgecolor='black', alpha=0.7, color='orange')
        axes[1].axvline(np.mean(episode_lengths), color='red', linestyle='--', 
                       linewidth=2, label=f'Mean: {np.mean(episode_lengths):.1f}')
        axes[1].set_xlabel('Episode Length', fontsize=12)
        axes[1].set_ylabel('Frequency', fontsize=12)
        axes[1].set_title('Episode Length Distribution', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = f"{output_name}_distribution.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Distribution plot saved to {plot_path}")
        plt.close()
    
    env.close()
    return episode_rewards, episode_lengths


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate Stable-Baselines3 trained models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate SAC model with 10 episodes
  python scripts/evaluate_sb3.py --model experiments/checkpoints/sac_hardcore_WORKING/sac_model_9600000_steps.zip --config configs/sac_elite_hardcore_gpu.yaml --episodes 10
  
  # Evaluate with custom output name for results
  python scripts/evaluate_sb3.py --model experiments/checkpoints/sac_hardcore_WORKING/sac_model_9600000_steps.zip --config configs/sac_elite_hardcore_gpu.yaml --episodes 100 --output results/sac_evaluation
  
  # Specify VecNormalize path explicitly
  python scripts/evaluate_sb3.py --model experiments/checkpoints/sac_hardcore_WORKING/sac_model_9600000_steps.zip --config configs/sac_elite_hardcore_gpu.yaml --vec-normalize experiments/checkpoints/sac_hardcore_WORKING/vec_normalize.pkl
        """
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to SB3 model checkpoint (.zip file)"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config YAML file"
    )
    parser.add_argument(
        "--vec-normalize",
        type=str,
        default=None,
        help="Path to vec_normalize.pkl file (auto-detected if not provided)"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes (default: 10)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output base name for CSV and plots (e.g., 'results/sac_eval')"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=45,
        help="Random seed for evaluation (default: 45)"
    )
    
    args = parser.parse_args()
    
    evaluate_sb3_model(
        model_path=args.model,
        config_path=args.config,
        vec_normalize_path=args.vec_normalize,
        num_episodes=args.episodes,
        render=False,
        output_name=args.output,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
