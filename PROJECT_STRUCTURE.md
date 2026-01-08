# Bipedal Walker Project Structure

## Overview
This project implements and compares three reinforcement learning algorithms (SAC, TD3, PPO) on the BipedalWalker-v3 environment with custom bridge obstacles.

## Directory Structure

```
bipedal_walker/
├── training/               # Training scripts for each algorithm
│   ├── train_sac.py       # SAC training (primary - uses bridge_balanced_wrapper)
│   ├── train_td3.py       # TD3 training
│   └── train_ppo.py       # PPO training
│
├── wrappers/              # Environment wrappers
│   └── bridge_balanced_wrapper.py  # Main wrapper for SAC with bridge handling
│
├── evaluation/            # Evaluation and visualization
│   ├── visualize.py       # Unified visualization script
│   ├── record_video.py    # Video recording utility
│   ├── evaluate.py        # Model evaluation
│   └── evaluate_custom.py # Custom evaluation metrics
│
├── configs/               # Configuration files
│   ├── sac_bridge_balanced_gpu.yaml  # Main SAC config
│   ├── td3_hardcore.yaml             # TD3 configs
│   ├── td3_hardcore_advanced.yaml
│   ├── ppo_*.yaml                    # PPO configs
│   └── archive/                      # Old configs
│
├── src/                   # Core source code
│   ├── agents/           # RL agent implementations
│   ├── envs/             # Custom environments
│   ├── models/           # Neural network models
│   └── utils/            # Utility functions
│
├── scripts/              # Utility scripts
│   ├── analyze_model.py   # Model analysis
│   ├── check_environment.py
│   ├── compare_configs.py
│   ├── plot_results.py
│   ├── watch_agent.py
│   └── analyze_actions.py
│
├── experiments/          # Training outputs
│   ├── checkpoints/      # Model checkpoints
│   ├── logs/             # Training logs
│   ├── videos/           # Recorded videos
│   └── archived_failed_attempts/  # Old experiments
│
├── archived/             # Archived/deprecated files
│   ├── wrappers/         # Old wrapper implementations
│   ├── configs/          # Old configuration files
│   ├── training_scripts/ # Deprecated training scripts
│   ├── documentation/    # Old documentation
│   └── visualizers/      # Old visualization scripts
│
├── tests/                # Unit tests
├── notebooks/            # Jupyter notebooks for analysis
├── plots/                # Generated plots
└── report/               # LaTeX report files
```

## Main Algorithms

### 1. SAC (Soft Actor-Critic)
- **Training Script**: `training/train_sac.py`
- **Config**: `configs/sac_bridge_balanced_gpu.yaml`
- **Wrapper**: `wrappers/bridge_balanced_wrapper.py`
- **Status**: Primary implementation, working well on bridges

### 2. TD3 (Twin Delayed Deep Deterministic Policy Gradient)
- **Training Script**: `training/train_td3.py`
- **Configs**: `configs/td3_*.yaml`
- **Status**: Available for comparison

### 3. PPO (Proximal Policy Optimization)
- **Training Script**: `training/train_ppo.py`
- **Configs**: `configs/ppo_*.yaml`
- **Status**: Available for comparison

## Quick Start

### Training

```bash
# Train SAC (recommended)
python training/train_sac.py --config configs/sac_bridge_balanced_gpu.yaml

# Train TD3
python training/train_td3.py --config configs/td3_hardcore.yaml

# Train PPO
python training/train_ppo.py --config configs/ppo_gpu_optimized.yaml
```

### Visualization

```bash
# Visualize trained model
python evaluation/visualize.py

# Record video
python evaluation/visualize.py --record --episodes 5

# Visualize specific checkpoint
python evaluation/visualize.py --checkpoint experiments/checkpoints/.../model.zip
```

### Evaluation

```bash
# Evaluate model performance
python evaluation/evaluate.py --checkpoint path/to/model.zip

# Compare multiple models
python scripts/compare_configs.py
```

## Key Features

- **Custom Walker Environment**: BipedalWalker with bridge obstacles
- **Bridge-Balanced Wrapper**: Intelligent LIDAR-based bridge detection
- **GPU Acceleration**: CUDA support for faster training
- **Parallel Environments**: Multi-process training for better sampling
- **Comprehensive Logging**: TensorBoard integration and detailed metrics
- **Video Recording**: Built-in video capture for trained agents

## Configuration

All training configurations are in YAML format in the `configs/` directory. Key parameters:
- Learning rates
- Network architectures
- Replay buffer sizes
- Exploration parameters
- Reward shaping coefficients

## Results

Training checkpoints and logs are saved in `experiments/`:
- Model weights: `experiments/checkpoints/`
- TensorBoard logs: `experiments/logs/`
- Training videos: `experiments/videos/`

## Archive

Old implementations and experiments are preserved in `archived/` for reference but are not actively maintained.
