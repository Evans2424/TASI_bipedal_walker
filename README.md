# Bipedal Walker Reinforcement Learning Project

A comprehensive implementation comparing three state-of-the-art reinforcement learning algorithms (SAC, TD3, PPO) for training agents to solve the BipedalWalker environment with custom bridge obstacles.

## 🎯 Overview

This project implements and compares reinforcement learning algorithms on a custom BipedalWalker environment featuring bridge obstacles. The primary implementation uses **SAC (Soft Actor-Critic)** with a custom bridge-balanced wrapper that has proven successful in navigation.

**Key Features:**
- **Multiple RL Algorithms**: SAC (primary), TD3, PPO for comprehensive comparison
- **Custom Environment**: BipedalWalker with bridge obstacles
- **Intelligent Bridge Detection**: LIDAR-based reward shaping
- **GPU-Accelerated Training**: CUDA support for faster training
- **Comprehensive Tooling**: Evaluation, visualization, and analysis scripts
- **Clean Architecture**: Organized codebase with proper separation of concerns

## 📁 Project Structure

```
bipedal_walker/
├── training/          # Training scripts (train_sac.py, train_td3.py, train_ppo.py)
├── wrappers/          # Environment wrappers (bridge_balanced_wrapper.py)
├── evaluation/        # Visualization and evaluation tools
├── configs/           # YAML configuration files for each algorithm
├── src/               # Core source code (agents, envs, models, utils)
├── scripts/           # Utility scripts for analysis
├── experiments/       # Training outputs (checkpoints, logs, videos)
└── archived/          # Deprecated implementations (for reference)
```

See [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) for detailed structure documentation.

## 🚀 Quick Start

### Installation

```bash
# Activate virtual environment
source bipedal_venv/bin/activate  # or: conda activate TASI_project

# Install dependencies
pip install -r requirements.txt
```

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

# Evaluate performance
python evaluation/evaluate.py --checkpoint path/to/model.zip
```

## 🤖 Implemented Algorithms

### 1. **SAC (Soft Actor-Critic)** - Primary Implementation

**Status**: ✅ Working well on bridges

- **Algorithm**: Off-policy, maximum entropy RL
- **Training**: `training/train_sac.py`
- **Config**: `configs/sac_bridge_balanced_gpu.yaml`
- **Wrapper**: `wrappers/bridge_balanced_wrapper.py`
- **Features**:
  - LIDAR-based bridge detection
  - Intelligent reward shaping for bridge navigation
  - GPU-accelerated training with Stable-Baselines3
  - Parallel environments for faster sampling

### 2. **TD3 (Twin Delayed DDPG)**

**Status**: Available for comparison

- **Algorithm**: Off-policy, deterministic policy
- **Training**: `training/train_td3.py`
- **Configs**: `configs/td3_hardcore*.yaml`
- **Features**: Custom implementation with replay buffer

### 3. **PPO (Proximal Policy Optimization)**

**Status**: Available for comparison

- **Algorithm**: On-policy, policy gradient
- **Training**: `training/train_ppo.py`
- **Configs**: `configs/ppo_*.yaml`
- **Features**: Parallel environments, GAE

## 🎮 Environment

### BipedalWalker with Bridge Obstacles

- **Observation Space**: 24-dim continuous (hull state, joint positions, velocities, LIDAR)
- **Action Space**: 4-dim continuous [-1, 1] (hip and knee motor speeds)
- **Custom Features**:
  - Bridge obstacles requiring strategic navigation
  - LIDAR-based bridge detection (10 rangefinder measurements)
  - Reward shaping for natural movement and bridge crossing
  
### Reward Structure

- **Base Rewards**: Forward progress, staying upright
- **Bridge Bonuses**: Stable waiting, successful crossing
- **Penalties**: Falling (-100), excessive motor torque, jerky movements
- **Success**: 300+ points

## 📊 Configuration
learning_rate: 3e-4
gamma: 0.99              # Discount factor
gae_lambda: 0.95         # GAE parameter
clip_epsilon: 0.2        # PPO clipping parameter
value_loss_coef: 0.5     # Value loss coefficient
entropy_coef: 0.01       # Entropy bonus
ppo_epochs: 10           # Update epochs per rollout
mini_batch_size: 64      # Mini-batch size
rollout_steps: 2048      # Steps before update
```

**Training Flow:**
1. Collect rollout of experiences (2048 steps)
2. Compute advantages using GAE
3. Perform multiple epochs of mini-batch updates
4. Clip policy ratio to prevent large updates
5. Optimize policy and value function jointly

### 2. Soft Actor-Critic (SAC)

**SAC** is an off-policy algorithm that maximizes both expected return and entropy, encouraging exploration while learning.

**Key Components:**
- **Actor Network**: Stochastic Gaussian policy with reparameterization trick
- **Twin Critics**: Two Q-networks to mitigate overestimation bias
- **Target Networks**: Soft-updated target critics for stability

All configurations are in YAML format in the `configs/` directory:

- **SAC**: `sac_bridge_balanced_gpu.yaml` - Main config with proven hyperparameters
- **TD3**: `td3_hardcore*.yaml` - Various difficulty levels
- **PPO**: `ppo_*.yaml` - Different exploration/exploitation strategies

### Key Parameters

```yaml
# Example: SAC config
learning_rate: 3e-4
buffer_size: 1000000
batch_size: 256
gamma: 0.99          # Discount factor
tau: 0.005           # Soft update rate
n_envs: 8            # Parallel environments
```

## 🛠️ Utilities

### Analysis Scripts

```bash
# Analyze model behavior
python scripts/analyze_model.py --checkpoint path/to/model.zip

# Compare configurations
python scripts/compare_configs.py

# Plot training results
python scripts/plot_results.py --logdir experiments/logs/

# Watch trained agent
python scripts/watch_agent.py --checkpoint path/to/model.zip
```

### Environment Testing

```bash
# Verify environment setup
python scripts/check_environment.py
```

## 📈 Results

Training outputs are automatically saved:
- **Checkpoints**: `experiments/checkpoints/` - Model weights at regular intervals
- **Logs**: `experiments/logs/` - TensorBoard compatible training metrics
- **Videos**: `experiments/videos/` - Recorded episodes

View training progress:
```bash
tensorboard --logdir experiments/logs/
```

## 🏗️ Development

### Adding New Algorithms

1. Implement agent in `src/agents/`
2. Add configuration in `configs/`
3. Create training script in `training/`
4. Update documentation

### Testing

```bash
# Run unit tests
python -m pytest tests/
```

## 📝 Notes

- **Primary Focus**: SAC with bridge-balanced wrapper has shown the best results
- **GPU Training**: CUDA support enabled by default if available
- **Archived Code**: Old implementations preserved in `archived/` for reference
- **Reproducibility**: Set random seeds in config for reproducible results

## 🔗 References

1. Haarnoja et al. (2018) - Soft Actor-Critic
2. Fujimoto et al. (2018) - Twin Delayed DDPG (TD3)
3. Schulman et al. (2017) - Proximal Policy Optimization (PPO)
4. OpenAI Gymnasium - BipedalWalker-v3

## 📄 License

This project is for educational purposes.

---
