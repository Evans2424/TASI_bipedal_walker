"""Compare different configuration files."""

import yaml
import argparse
from pathlib import Path
from tabulate import tabulate


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def compare_configs(config_paths: list):
    """Compare multiple configuration files."""

    configs = {}
    for path in config_paths:
        name = Path(path).stem
        configs[name] = load_config(path)

    print("="*80)
    print("CONFIGURATION COMPARISON")
    print("="*80)

    # Agent parameters
    print("\n📊 AGENT PARAMETERS:")
    agent_params = [
        ["Parameter"] + list(configs.keys()),
        ["─"*15] + ["─"*15] * len(configs),
    ]

    param_keys = [
        ('learning_rate', 'Learning Rate'),
        ('entropy_coef', 'Entropy Coef'),
        ('clip_epsilon', 'Clip Epsilon'),
        ('gamma', 'Gamma'),
        ('mini_batch_size', 'Mini Batch Size'),
    ]

    for key, label in param_keys:
        row = [label]
        for config_name, config in configs.items():
            value = config['agent'].get(key, 'N/A')
            if isinstance(value, float):
                row.append(f"{value:.1e}")
            else:
                row.append(str(value))
        agent_params.append(row)

    print(tabulate(agent_params, headers='firstrow', tablefmt='simple'))

    # Training parameters
    print("\n⚙️  TRAINING PARAMETERS:")
    training_params = [
        ["Parameter"] + list(configs.keys()),
        ["─"*15] + ["─"*15] * len(configs),
    ]

    train_keys = [
        ('total_timesteps', 'Total Timesteps'),
        ('rollout_steps', 'Rollout Steps'),
        ('eval_frequency', 'Eval Frequency'),
    ]

    for key, label in train_keys:
        row = [label]
        for config_name, config in configs.items():
            value = config['training'].get(key, 'N/A')
            if isinstance(value, int) and value >= 1000:
                row.append(f"{value/1000000:.1f}M" if value >= 1000000 else f"{value/1000:.0f}K")
            else:
                row.append(str(value))
        training_params.append(row)

    print(tabulate(training_params, headers='firstrow', tablefmt='simple'))

    # Recommendations
    print("\n💡 RECOMMENDATIONS:")
    print("─"*80)

    recommendations = {
        'ppo_config': {
            'use_case': '🔵 Default/Baseline',
            'description': 'Standard configuration for initial training',
            'when': 'First time training',
            'time': '6-12 hours (CPU)'
        },
        'ppo_conservative': {
            'use_case': '🟢 Stable & Reliable',
            'description': 'Lower learning rate, more training time',
            'when': 'Previous training failed, want stability',
            'time': '9-15 hours (CPU)'
        },
        'ppo_exploration': {
            'use_case': '🟡 High Exploration',
            'description': 'Maximize exploration to find good strategies',
            'when': 'Agent gets stuck in local optima',
            'time': '9-15 hours (CPU)'
        },
        'ppo_fast': {
            'use_case': '🟠 Quick Experiments',
            'description': 'Higher learning rate, shorter training',
            'when': 'Rapid iteration, testing ideas',
            'time': '2-4 hours (GPU), 6-8 hours (CPU)'
        },
        'sac_config': {
            'use_case': '🔵 SAC Default',
            'description': 'Standard SAC configuration',
            'when': 'Trying SAC algorithm',
            'time': '4-8 hours (CPU)'
        },
        'sac_tuned': {
            'use_case': '🟣 SAC Optimized',
            'description': 'Tuned SAC for sample efficiency',
            'when': 'PPO failed, want better algorithm',
            'time': '4-8 hours (CPU)'
        },
        'ppo_hardcore': {
            'use_case': '🔴 Hardcore Mode',
            'description': 'For BipedalWalkerHardcore-v3',
            'when': 'After mastering normal mode',
            'time': '15-24 hours (CPU)'
        }
    }

    for config_name in configs.keys():
        if config_name in recommendations:
            rec = recommendations[config_name]
            print(f"\n{rec['use_case']}: {config_name}")
            print(f"  Description: {rec['description']}")
            print(f"  When to use: {rec['when']}")
            print(f"  Expected time: {rec['time']}")

    print("\n" + "="*80)
    print("🚀 GETTING STARTED:")
    print("="*80)
    print("\n1. Based on your previous failure (mean reward: -50), try:")
    print("   python train.py --config configs/ppo_conservative.yaml")
    print("\n2. Monitor training:")
    print("   tensorboard --logdir experiments/logs")
    print("\n3. If still failing, try:")
    print("   python train.py --config configs/sac_tuned.yaml")
    print("\n" + "="*80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Compare configuration files")
    parser.add_argument(
        "--configs",
        type=str,
        nargs='+',
        default=[
            'configs/ppo_config.yaml',
            'configs/ppo_conservative.yaml',
            'configs/ppo_exploration.yaml',
            'configs/sac_tuned.yaml'
        ],
        help="Paths to config files to compare"
    )

    args = parser.parse_args()

    # Check if tabulate is available
    try:
        compare_configs(args.configs)
    except ImportError:
        print("Note: Install 'tabulate' for better formatting: pip install tabulate")
        print("\nBasic comparison:")
        for config_path in args.configs:
            config = load_config(config_path)
            name = Path(config_path).stem
            print(f"\n{name}:")
            print(f"  Learning Rate: {config['agent'].get('learning_rate', 'N/A')}")
            print(f"  Entropy Coef: {config['agent'].get('entropy_coef', 'N/A')}")
            print(f"  Total Timesteps: {config['training'].get('total_timesteps', 'N/A')}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        print("\nBasic usage:")
        print("python scripts/compare_configs.py")
