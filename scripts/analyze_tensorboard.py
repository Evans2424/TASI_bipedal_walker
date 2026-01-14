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
    
    # Create figure with subplots (2x2 grid)
    fig = plt.figure(figsize=(14, 10))
    
    # 1. Episode Length Over Time
    ax1 = plt.subplot(2, 2, 1)
    if 'rollout/ep_len_mean' in data:
        steps = data['rollout/ep_len_mean']['steps']
        values = data['rollout/ep_len_mean']['values']
        ax1.plot(steps, values, linewidth=2, color='#F18F01')
        ax1.set_xlabel('Training Steps')
        ax1.set_ylabel('Mean Episode Length')
        ax1.set_title('Episode Length Over Time')
        ax1.grid(True, alpha=0.3)
    
    # 2. Actor Loss
    ax2 = plt.subplot(2, 2, 2)
    if 'train/actor_loss' in data:
        steps = data['train/actor_loss']['steps']
        values = data['train/actor_loss']['values']
        ax2.plot(steps, values, linewidth=1, color='#C73E1D', alpha=0.6)
        ax2.set_xlabel('Training Steps')
        ax2.set_ylabel('Actor Loss')
        ax2.set_title('Actor Loss')
        ax2.grid(True, alpha=0.3)
    
    # 3. Critic Loss
    ax3 = plt.subplot(2, 2, 3)
    if 'train/critic_loss' in data:
        steps = data['train/critic_loss']['steps']
        values = data['train/critic_loss']['values']
        ax3.plot(steps, values, linewidth=1, color='#6A4C93', alpha=0.6)
        ax3.set_xlabel('Training Steps')
        ax3.set_ylabel('Critic Loss')
        ax3.set_title('Critic Loss')
        ax3.grid(True, alpha=0.3)
    
    # 4. Reward Distribution (final 20% of training)
    ax4 = plt.subplot(2, 2, 4)
    if 'rollout/ep_rew_mean' in data:
        values = data['rollout/ep_rew_mean']['values']
        final_rewards = values[int(len(values) * 0.8):]
        ax4.hist(final_rewards, bins=30, color='#2E86AB', alpha=0.7, edgecolor='black')
        ax4.set_xlabel('Reward')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Reward Distribution (Final 20%)')
        ax4.grid(True, alpha=0.3)
    
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
            'TD3 Easy': 'logs/td3_easy',
            'TD3 Hardcore': 'logs/td3_hardcore_test_1',
            'TD3 Hardcore Bridges': 'logs/td3_hardcore_bridges'
        }
        colors = {
            'TD3 Easy': '#2E86AB',
            'TD3 Hardcore': '#A23B72',
            'TD3 Hardcore Bridges': '#F18F01'
        }
        title_prefix = 'TD3'
    elif algorithm.lower() == 'sac':
        experiments = {
            'SAC Easy': 'logs/sac_easy/sac_easy',
            'SAC Hardcore': 'logs/SAC_1',
            'SAC Hardcore Bridges': 'logs/sac_bridges/sac_bridges_v2_decent'
        }
        colors = {
            'SAC Easy': '#2E86AB',
            'SAC Hardcore': '#A23B72',
            'SAC Hardcore Bridges': '#F18F01'
        }
        title_prefix = 'SAC'
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}. Use 'td3' or 'sac'.")
    
    fig, axes = plt.subplots(2, 1, figsize=(8, 12))
    
    # Plot 1: Training Rewards
    ax1 = axes[0]
    for name, log_dir in experiments.items():
        if not Path(log_dir).exists():
            print(f"⚠️ Warning: {log_dir} not found, skipping...")
            continue
        
        try:
            data = load_tensorboard_data(log_dir)
            # Try different tag names for training rewards
            reward_tag = None
            if 'rollout/ep_rew_mean' in data:
                reward_tag = 'rollout/ep_rew_mean'
            elif 'episode/mean_reward_100' in data:
                reward_tag = 'episode/mean_reward_100'
            
            if reward_tag:
                steps = data[reward_tag]['steps']
                values = data[reward_tag]['values']
                ax1.plot(steps, values, linewidth=2, color=colors[name], 
                        alpha=0.8, label=name)
        except Exception as e:
            print(f"⚠️ Error loading {name}: {e}")
            continue
    
    ax1.set_xlabel('Training Steps', fontsize=12)
    ax1.set_ylabel('Mean Training Reward', fontsize=12)
    ax1.set_title(f'{title_prefix} Training Rewards Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylim(-100, 350)  # Fixed scale for comparison
    ax1.legend(fontsize=10, loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Evaluation Rewards
    ax2 = axes[1]
    for name, log_dir in experiments.items():
        if not Path(log_dir).exists():
            continue
        
        try:
            data = load_tensorboard_data(log_dir)
            # Try different tag names for evaluation rewards
            eval_tag = None
            if 'eval/mean_reward' in data:
                eval_tag = 'eval/mean_reward'
            elif 'episode/reward' in data:
                # Use episode reward as proxy if no eval tag
                eval_tag = 'episode/reward'
            
            if eval_tag:
                steps = data[eval_tag]['steps']
                values = data[eval_tag]['values']
                ax2.plot(steps, values, linewidth=2.5, color=colors[name], 
                         markersize=4, alpha=0.8, label=name)
        except Exception as e:
            continue
    
    ax2.set_xlabel('Training Steps', fontsize=12)
    ax2.set_ylabel('Mean Evaluation Reward', fontsize=12)
    ax2.set_title(f'{title_prefix} Evaluation Rewards Comparison', fontsize=14, fontweight='bold')
    ax2.set_ylim(-100, 350)  # Fixed scale for comparison
    ax2.legend(fontsize=10, loc='best')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = save_dir / f'{algorithm.lower()}_environments_comparison.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 {title_prefix} comparison plot saved to: {plot_path}")
    plt.close()
    
    # Create third plot: Individual Loss Plots (3x2 grid - 6 subplots)
    fig3 = plt.figure(figsize=(9, 14))
    
    plot_idx = 1
    for name, log_dir in experiments.items():
        if not Path(log_dir).exists():
            continue
        
        try:
            data = load_tensorboard_data(log_dir)
            
            # Actor Loss subplot (left column)
            ax_actor = plt.subplot(3, 2, plot_idx)
            if 'train/actor_loss' in data:
                steps = data['train/actor_loss']['steps']
                values = data['train/actor_loss']['values']
                ax_actor.plot(steps, values, linewidth=2, color=colors[name], alpha=0.7)
                ax_actor.set_xlabel('Training Steps', fontsize=10)
                ax_actor.set_ylabel('Actor Loss', fontsize=10)
                ax_actor.set_title(f'{name} - Actor Loss', fontsize=12, fontweight='bold')
                ax_actor.grid(True, alpha=0.3)
            
            # Critic Loss subplot (right column)
            ax_critic = plt.subplot(3, 2, plot_idx + 1)
            if 'train/critic_loss' in data:
                steps = data['train/critic_loss']['steps']
                values = data['train/critic_loss']['values']
                ax_critic.plot(steps, values, linewidth=2, color=colors[name], alpha=0.7)
                ax_critic.set_xlabel('Training Steps', fontsize=10)
                ax_critic.set_ylabel('Critic Loss', fontsize=10)
                ax_critic.set_title(f'{name} - Critic Loss', fontsize=12, fontweight='bold')
                ax_critic.grid(True, alpha=0.3)
            
            plot_idx += 2
            
        except Exception as e:
            print(f"⚠️ Error loading {name}: {e}")
            continue
    
    plt.tight_layout()
    plot_path3 = save_dir / f'{algorithm.lower()}_loss_comparison.png'
    plt.savefig(plot_path3, dpi=300, bbox_inches='tight')
    print(f"📊 {title_prefix} loss comparison plot saved to: {plot_path3}")
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
