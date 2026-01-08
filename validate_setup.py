#!/usr/bin/env python3
"""
Validation script to check SAC implementation before final training.
Tests imports, configs, and environment creation for all three modes.
"""

import sys
import yaml
from pathlib import Path
import traceback

def test_section(name):
    """Print test section header."""
    print("\n" + "="*60)
    print(f"Testing: {name}")
    print("="*60)

def test_imports():
    """Test all required imports."""
    test_section("Imports")
    
    tests = [
        ("gymnasium", "import gymnasium as gym"),
        ("stable_baselines3", "from stable_baselines3 import SAC"),
        ("torch", "import torch"),
        ("wrapper", "from wrappers.bridge_balanced_wrapper import BridgeBalancedWrapper"),
        ("custom_walker", "from src.envs.custom_walker import BipedalWalker"),
    ]
    
    all_passed = True
    for name, import_stmt in tests:
        try:
            exec(import_stmt)
            print(f"✓ {name}")
        except Exception as e:
            print(f"✗ {name}: {e}")
            all_passed = False
    
    return all_passed

def test_config(config_path, mode_name):
    """Test a specific config file."""
    print(f"\nTesting config: {config_path}")
    
    required_keys = {
        'env': ['name', 'hardcore', 'normalize_observations', 'use_bridge_wrapper'],
        'algorithm': ['learning_rate', 'buffer_size', 'batch_size'],
        'training': ['total_timesteps', 'n_envs', 'eval_freq', 'tensorboard_log'],
        'checkpoint': ['save_path'],
        'experiment': ['name', 'seed', 'device']
    }
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check top-level sections
        missing = []
        for section in required_keys.keys():
            if section not in config:
                missing.append(section)
        
        if missing:
            print(f"  ✗ Missing sections: {missing}")
            return False
        
        # Check required keys in each section
        for section, keys in required_keys.items():
            for key in keys:
                if key not in config[section]:
                    print(f"  ✗ Missing key: {section}.{key}")
                    return False
        
        # Validate specific values
        if mode_name == "easy" and config['env']['hardcore']:
            print(f"  ✗ Easy mode should have hardcore=False")
            return False
        
        if mode_name == "hardcore" and not config['env']['hardcore']:
            print(f"  ✗ Hardcore mode should have hardcore=True")
            return False
        
        if mode_name == "bridges" and not config['env']['use_bridge_wrapper']:
            print(f"  ✗ Bridges mode should have use_bridge_wrapper=True")
            return False
        
        print(f"  ✓ Config valid")
        print(f"    - Mode: {mode_name}")
        print(f"    - Hardcore: {config['env']['hardcore']}")
        print(f"    - Bridge wrapper: {config['env'].get('use_bridge_wrapper', False)}")
        print(f"    - Timesteps: {config['training']['total_timesteps']:,}")
        print(f"    - Envs: {config['training']['n_envs']}")
        
        # Check early stopping
        if 'early_stopping' in config['training']:
            es = config['training']['early_stopping']
            if es.get('use_no_improvement_stop'):
                print(f"    - Early stopping: Yes (patience={es.get('patience')})")
            else:
                print(f"    - Early stopping: No")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        traceback.print_exc()
        return False

def test_configs():
    """Test all config files."""
    test_section("Configuration Files")
    
    configs = [
        ("configs/sac_easy_gpu.yaml", "easy"),
        ("configs/sac_hardcore_gpu.yaml", "hardcore"),
        ("configs/sac_bridges_gpu.yaml", "bridges"),
    ]
    
    all_passed = True
    for config_path, mode in configs:
        if not test_config(config_path, mode):
            all_passed = False
    
    return all_passed

def test_environment_creation():
    """Test creating environments for all modes."""
    test_section("Environment Creation")
    
    import gymnasium as gym
    from gymnasium.envs.registration import register
    from wrappers.bridge_balanced_wrapper import BridgeBalancedWrapper
    
    # Register custom walker
    try:
        register(
            id='CustomBipedalWalker-v3',
            entry_point='src.envs.custom_walker:BipedalWalker',
            max_episode_steps=2000,
            reward_threshold=300,
        )
    except:
        pass  # Already registered
    
    tests = [
        ("Easy", "BipedalWalker-v3", False, False),
        ("Hardcore", "BipedalWalker-v3", True, False),
        ("Bridges", "CustomBipedalWalker-v3", True, True),
    ]
    
    all_passed = True
    for name, env_id, hardcore, use_wrapper in tests:
        try:
            print(f"\nTesting {name} mode:")
            env = gym.make(env_id, hardcore=hardcore)
            
            if use_wrapper:
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
                    lidar_bridge_threshold=0.5,
                    min_close_beams=3,
                    waiting_velocity_threshold=0.15,
                    waiting_angle_threshold=0.3,
                )
                print(f"  ✓ Wrapper applied")
            
            obs, info = env.reset()
            print(f"  ✓ Environment created")
            print(f"    - Observation shape: {obs.shape}")
            print(f"    - Action space: {env.action_space}")
            
            # Test one step
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"  ✓ Step successful")
            
            env.close()
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            traceback.print_exc()
            all_passed = False
    
    return all_passed

def test_device_availability():
    """Test GPU/device availability."""
    test_section("Device Availability")
    
    import torch
    
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  - Device: {torch.cuda.get_device_name(0)}")
        print(f"  - Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    print(f"MPS available: {hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()}")
    
    return True

def test_directories():
    """Test that required directories exist or can be created."""
    test_section("Directory Structure")
    
    dirs = [
        "experiments/checkpoints/sac_easy",
        "experiments/checkpoints/sac_hardcore",
        "experiments/checkpoints/sac_bridges",
        "experiments/logs/sac_easy",
        "experiments/logs/sac_hardcore",
        "experiments/logs/sac_bridges",
        "experiments/videos",
    ]
    
    all_passed = True
    for dir_path in dirs:
        path = Path(dir_path)
        if path.exists():
            print(f"✓ {dir_path} (exists)")
        else:
            try:
                path.mkdir(parents=True, exist_ok=True)
                print(f"✓ {dir_path} (created)")
            except Exception as e:
                print(f"✗ {dir_path}: {e}")
                all_passed = False
    
    return all_passed

def main():
    """Run all validation tests."""
    print("="*60)
    print("SAC IMPLEMENTATION VALIDATION")
    print("="*60)
    print("\nThis script validates the SAC setup before training.")
    print("All tests must pass for reliable training.")
    
    results = {}
    
    # Run tests
    results['imports'] = test_imports()
    results['configs'] = test_configs()
    results['environments'] = test_environment_creation()
    results['device'] = test_device_availability()
    results['directories'] = test_directories()
    
    # Summary
    test_section("VALIDATION SUMMARY")
    
    all_passed = all(results.values())
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print("\n" + "="*60)
    if all_passed:
        print("✓ ALL TESTS PASSED - Ready for training!")
        print("="*60)
        print("\nQuick start commands:")
        print("  python training/train_sac.py --config configs/sac_easy_gpu.yaml")
        print("  python training/train_sac.py --config configs/sac_hardcore_gpu.yaml")
        print("  python training/train_sac.py --config configs/sac_bridges_gpu.yaml")
        return 0
    else:
        print("✗ SOME TESTS FAILED - Please fix errors before training")
        print("="*60)
        return 1

if __name__ == "__main__":
    sys.exit(main())
