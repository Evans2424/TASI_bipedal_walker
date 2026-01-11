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
    event_files = [f for f in os.listdir(log_dir) if f.startswith('events')]
    if not event_files:
        raise ValueError(f"No TensorBoard event files found in {log_dir}")
    
    event_file = os.path.join(log_dir, event_files[0])
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


def main():
    parser = argparse.ArgumentParser(description='Analyze TensorBoard training logs')
    parser.add_argument('--log-dir', type=str, required=True,
                        help='Path to TensorBoard log directory')
    parser.add_argument('--save-dir', type=str, default=None,
                        help='Directory to save plots (default: same as log-dir)')
    args = parser.parse_args()
    
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
