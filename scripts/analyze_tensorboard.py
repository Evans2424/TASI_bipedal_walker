#!/usr/bin/env python3
"""Analyze TensorBoard training logs."""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def load_tensorboard_data(log_dir):
    """Load data from TensorBoard event files."""
    log_path = Path(log_dir)
    
    # Find event files in the directory or subdirectories
    event_files = list(log_path.glob('events.out.tfevents*'))
    
    # If no files in root, check subdirectories (common for SAC)
    if not event_files:
        for subdir in log_path.iterdir():
            if subdir.is_dir():
                event_files.extend(subdir.glob('events.out.tfevents*'))
    
    if not event_files:
        raise ValueError(f"No TensorBoard event files found in {log_dir}")
    
    # Use the first event file found
    event_file = str(event_files[0])
    ea = EventAccumulator(event_file)
    ea.Reload()
    
    data = {}
    for tag in ea.Tags()['scalars']:
        events = ea.Scalars(tag)
        data[tag] = {
            'steps': [e.step for e in events],
            'values': [e.value for e in events],
            'wall_time': [e.wall_time for e in events]
        }
    
    return data


def plot_training_analysis(data, save_dir):
    """Create comprehensive training analysis plots."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))
    
    # 1. Episode Rewards
    ax1 = plt.subplot(3, 3, 1)
    if 'rollout/ep_rew_mean' in data:
        steps = data['rollout/ep_rew_mean']['steps']
        values = data['rollout/ep_rew_mean']['values']
        ax1.plot(steps, values, linewidth=2, color='#2E86AB')
        ax1.set_xlabel('Training Steps')
        ax1.set_ylabel('Mean Episode Reward')
        ax1.set_title('Training Progress: Episode Rewards')
        ax1.grid(True, alpha=0.3)
        
        # Add horizontal line at 300 (success threshold)
        ax1.axhline(y=300, color='green', linestyle='--', alpha=0.5, label='Success (300)')
        ax1.legend()
    
    # 2. Evaluation Rewards
    ax2 = plt.subplot(3, 3, 2)
    if 'eval/mean_reward' in data:
        steps = data['eval/mean_reward']['steps']
        values = data['eval/mean_reward']['values']
        ax2.plot(steps, values, linewidth=2, color='#A23B72', marker='o', markersize=4)
        ax2.set_xlabel('Training Steps')
        ax2.set_ylabel('Evaluation Reward')
        ax2.set_title('Evaluation Performance')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=300, color='green', linestyle='--', alpha=0.5, label='Success (300)')
        ax2.legend()
    
    # 3. Episode Length
    ax3 = plt.subplot(3, 3, 3)
    if 'rollout/ep_len_mean' in data:
        steps = data['rollout/ep_len_mean']['steps']
        values = data['rollout/ep_len_mean']['values']
        ax3.plot(steps, values, linewidth=2, color='#F18F01')
        ax3.set_xlabel('Training Steps')
        ax3.set_ylabel('Mean Episode Length')
        ax3.set_title('Episode Length Over Time')
        ax3.grid(True, alpha=0.3)
    
    # 4. Actor Loss
    ax4 = plt.subplot(3, 3, 4)
    if 'train/actor_loss' in data:
        steps = data['train/actor_loss']['steps']
        values = data['train/actor_loss']['values']
        ax4.plot(steps, values, linewidth=1, color='#C73E1D', alpha=0.6)
        ax4.set_xlabel('Training Steps')
        ax4.set_ylabel('Actor Loss')
        ax4.set_title('Actor Loss')
        ax4.grid(True, alpha=0.3)
    
    # 5. Critic Loss
    ax5 = plt.subplot(3, 3, 5)
    if 'train/critic_loss' in data:
        steps = data['train/critic_loss']['steps']
        values = data['train/critic_loss']['values']
        ax5.plot(steps, values, linewidth=1, color='#6A4C93', alpha=0.6)
        ax5.set_xlabel('Training Steps')
        ax5.set_ylabel('Critic Loss')
        ax5.set_title('Critic Loss')
        ax5.grid(True, alpha=0.3)
    
    # 6. Learning Rate
    ax6 = plt.subplot(3, 3, 6)
    if 'train/learning_rate' in data:
        steps = data['train/learning_rate']['steps']
        values = data['train/learning_rate']['values']
        ax6.plot(steps, values, linewidth=2, color='#1D7874')
        ax6.set_xlabel('Training Steps')
        ax6.set_ylabel('Learning Rate')
        ax6.set_title('Learning Rate Schedule')
        ax6.grid(True, alpha=0.3)
    
    # 7. FPS
    ax7 = plt.subplot(3, 3, 7)
    if 'time/fps' in data:
        steps = data['time/fps']['steps']
        values = data['time/fps']['values']
        ax7.plot(steps, values, linewidth=2, color='#588B8B')
        ax7.set_xlabel('Training Steps')
        ax7.set_ylabel('FPS')
        ax7.set_title('Training Speed (FPS)')
        ax7.grid(True, alpha=0.3)
    
    # 8. Reward Histogram (final 20% of training)
    ax8 = plt.subplot(3, 3, 8)
    if 'rollout/ep_rew_mean' in data:
        values = data['rollout/ep_rew_mean']['values']
        final_rewards = values[int(len(values) * 0.8):]
        ax8.hist(final_rewards, bins=30, color='#2E86AB', alpha=0.7, edgecolor='black')
        ax8.axvline(x=300, color='green', linestyle='--', linewidth=2, label='Success (300)')
        ax8.set_xlabel('Reward')
        ax8.set_ylabel('Frequency')
        ax8.set_title('Reward Distribution (Final 20%)')
        ax8.legend()
        ax8.grid(True, alpha=0.3)
    
    # 9. Combined Rewards Plot
    ax9 = plt.subplot(3, 3, 9)
    if 'rollout/ep_rew_mean' in data:
        ax9.plot(data['rollout/ep_rew_mean']['steps'], 
                data['rollout/ep_rew_mean']['values'], 
                linewidth=2, color='#2E86AB', alpha=0.7, label='Training')
    if 'eval/mean_reward' in data:
        ax9.plot(data['eval/mean_reward']['steps'], 
                data['eval/mean_reward']['values'], 
                linewidth=2, color='#A23B72', marker='o', markersize=4, label='Evaluation')
    ax9.axhline(y=300, color='green', linestyle='--', alpha=0.5)
    ax9.set_xlabel('Training Steps')
    ax9.set_ylabel('Mean Reward')
    ax9.set_title('Training vs Evaluation Rewards')
    ax9.legend()
    ax9.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = save_dir / 'training_analysis.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to: {plot_path}")
    plt.close()
    
    # Create second plot: Training vs Evaluation Rewards only
    fig2 = plt.figure(figsize=(12, 6))
    
    if 'rollout/ep_rew_mean' in data or 'eval/mean_reward' in data:
        if 'rollout/ep_rew_mean' in data:
            plt.plot(data['rollout/ep_rew_mean']['steps'], 
                    data['rollout/ep_rew_mean']['values'], 
                    linewidth=2, color='#2E86AB', alpha=0.7, label='Training Reward')
        if 'eval/mean_reward' in data:
            plt.plot(data['eval/mean_reward']['steps'], 
                    data['eval/mean_reward']['values'], 
                    linewidth=2.5, color='#A23B72', marker='o', markersize=5, 
                    label='Evaluation Reward')
        
        plt.axhline(y=300, color='green', linestyle='--', linewidth=2, alpha=0.6, label='Success Threshold (300)')
        plt.xlabel('Training Steps', fontsize=12)
        plt.ylabel('Mean Reward', fontsize=12)
        plt.title('Training vs Evaluation Rewards', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11, loc='best')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path2 = save_dir / 'reward_comparison.png'
        plt.savefig(plot_path2, dpi=300, bbox_inches='tight')
        print(f"Saved reward comparison plot to: {plot_path2}")
        plt.close()


def print_statistics(data):
    """Print summary statistics."""
    print("\n" + "="*60)
    print("TRAINING SUMMARY STATISTICS")
    print("="*60)
    
    if 'rollout/ep_rew_mean' in data:
        values = data['rollout/ep_rew_mean']['values']
        print("\n📊 Training Rewards:")
        print(f"  Final reward: {values[-1]:.2f}")
        print(f"  Max reward: {max(values):.2f}")
        print(f"  Mean reward: {np.mean(values):.2f}")
        print(f"  Std deviation: {np.std(values):.2f}")
        
        # Success rate in final 20%
        final_rewards = values[int(len(values) * 0.8):]
        success_rate = sum(1 for r in final_rewards if r > 300) / len(final_rewards) * 100
        print(f"  Success rate (final 20%, >300): {success_rate:.1f}%")
    
    if 'eval/mean_reward' in data:
        values = data['eval/mean_reward']['values']
        print("\n🎯 Evaluation Rewards:")
        print(f"  Final eval reward: {values[-1]:.2f}")
        print(f"  Best eval reward: {max(values):.2f}")
        print(f"  Mean eval reward: {np.mean(values):.2f}")
        print(f"  Std deviation: {np.std(values):.2f}")
    
    if 'rollout/ep_len_mean' in data:
        values = data['rollout/ep_len_mean']['values']
        print("\n⏱️  Episode Length:")
        print(f"  Final length: {values[-1]:.0f}")
        print(f"  Max length: {max(values):.0f}")
        print(f"  Mean length: {np.mean(values):.0f}")
    
    if 'train/actor_loss' in data:
        values = data['train/actor_loss']['values']
        print("\n🔧 Actor Loss:")
        print(f"  Final loss: {values[-1]:.6f}")
        print(f"  Mean loss: {np.mean(values):.6f}")
    
    if 'train/critic_loss' in data:
        values = data['train/critic_loss']['values']
        print("\n🔧 Critic Loss:")
        print(f"  Final loss: {values[-1]:.6f}")
        print(f"  Mean loss: {np.mean(values):.6f}")
    
    if 'time/fps' in data:
        values = data['time/fps']['values']
        print("\n⚡ Training Speed:")
        print(f"  Mean FPS: {np.mean(values):.1f}")
        print(f"  Total steps: {data['rollout/ep_rew_mean']['steps'][-1]}")
    
    print("\n" + "="*60)


def plot_algorithm_comparison(algorithm='td3', save_dir='plots'):
    """Create comparison plot for all three environments of a given algorithm."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Define experiments based on algorithm
    if algorithm.lower() == 'td3':
        experiments = {
            'TD3 Easy': 'experiments/logs/td3_easy',
            'TD3 Hardcore': 'experiments/logs/td3_hardcore',
            'TD3 Hardcore Bridges': 'experiments/logs/td3_hardcore_bridges'
        }
        colors = {
            'TD3 Easy': '#2E86AB',
            'TD3 Hardcore': '#A23B72',
            'TD3 Hardcore Bridges': '#F18F01'
        }
        title_prefix = 'TD3'
    elif algorithm.lower() == 'sac':
        experiments = {
            'SAC Easy': 'experiments/logs/sac_easy',
            'SAC Hardcore': 'experiments/logs/sac_hardcore',
            'SAC Hardcore Bridges': 'experiments/logs/sac_elite_unified_hardcore_gpu_custom_bridges'
        }
        colors = {
            'SAC Easy': '#2E86AB',
            'SAC Hardcore': '#A23B72',
            'SAC Hardcore Bridges': '#F18F01'
        }
        title_prefix = 'SAC'
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}. Use 'td3' or 'sac'.")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Training Rewards
    ax1 = axes[0]
    for name, log_dir in experiments.items():
        if not Path(log_dir).exists():
            print(f"⚠️ Warning: {log_dir} not found, skipping...")
            continue
        
        try:
            data = load_tensorboard_data(log_dir)
            if 'rollout/ep_rew_mean' in data:
                steps = data['rollout/ep_rew_mean']['steps']
                values = data['rollout/ep_rew_mean']['values']
                ax1.plot(steps, values, linewidth=2, color=colors[name], 
                        alpha=0.8, label=name)
        except Exception as e:
            print(f"⚠️ Error loading {name}: {e}")
            continue
    
    ax1.axhline(y=300, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Success (300)')
    ax1.set_xlabel('Training Steps', fontsize=12)
    ax1.set_ylabel('Mean Training Reward', fontsize=12)
    ax1.set_title(f'{title_prefix} Training Rewards Comparison', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10, loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Evaluation Rewards
    ax2 = axes[1]
    for name, log_dir in experiments.items():
        if not Path(log_dir).exists():
            continue
        
        try:
            data = load_tensorboard_data(log_dir)
            if 'eval/mean_reward' in data:
                steps = data['eval/mean_reward']['steps']
                values = data['eval/mean_reward']['values']
                ax2.plot(steps, values, linewidth=2.5, color=colors[name], 
                        marker='o', markersize=4, alpha=0.8, label=name)
        except Exception as e:
            continue
    
    ax2.axhline(y=300, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Success (300)')
    ax2.set_xlabel('Training Steps', fontsize=12)
    ax2.set_ylabel('Mean Evaluation Reward', fontsize=12)
    ax2.set_title(f'{title_prefix} Evaluation Rewards Comparison', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10, loc='best')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = save_dir / f'{algorithm.lower()}_environments_comparison.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 {title_prefix} comparison plot saved to: {plot_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Analyze TensorBoard training logs')
    parser.add_argument('--log-dir', type=str,
                        help='Path to TensorBoard log directory')
    parser.add_argument('--save-dir', type=str, default=None,
                        help='Directory to save plots (default: same as log-dir)')
    parser.add_argument('--compare-td3', action='store_true',
                        help='Compare all three TD3 environments (easy, hardcore, bridges)')
    parser.add_argument('--compare-sac', action='store_true',
                        help='Compare all three SAC environments (easy, hardcore, bridges)')
    args = parser.parse_args()
    
    if args.compare_td3:
        # Create comparison plot for all TD3 environments
        print("Creating TD3 environments comparison plot...")
        save_dir = args.save_dir if args.save_dir else 'plots'
        plot_algorithm_comparison('td3', save_dir)
        print(f"\n✅ TD3 comparison complete!")
        return
    
    if args.compare_sac:
        # Create comparison plot for all SAC environments
        print("Creating SAC environments comparison plot...")
        save_dir = args.save_dir if args.save_dir else 'plots'
        plot_algorithm_comparison('sac', save_dir)
        print(f"\n✅ SAC comparison complete!")
        return
    
    if not args.log_dir:
        parser.error("--log-dir is required unless using --compare-td3 or --compare-sac")
    
    # Load data
    print(f"Loading TensorBoard logs from: {args.log_dir}")
    data = load_tensorboard_data(args.log_dir)
    print(f"Found {len(data)} metrics")
    
    # Print statistics
    print_statistics(data)
    
    # Create plots
    save_dir = args.save_dir if args.save_dir else args.log_dir
    plot_training_analysis(data, save_dir)
    
    print(f"\n✅ Analysis complete!")


if __name__ == '__main__':
    main()
