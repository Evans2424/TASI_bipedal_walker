# Bipedal Walker Deep Reinforcement Learning

**Comparative Analysis of TD3, SAC, and PPO for Bipedal Locomotion**

This project presents a comprehensive empirical comparison of three state-of-the-art deep reinforcement learning algorithms—Twin Delayed Deep Deterministic Policy Gradient (TD3), Soft Actor-Critic (SAC), and Proximal Policy Optimization (PPO)—applied to bipedal locomotion tasks. The study evaluates these algorithms across three environmental variants with increasing difficulty: standard terrain (easy), hardcore mode with obstacles, and hardcore mode with custom bridge obstacles.

## 🎯 Overview

Deep Reinforcement Learning has emerged as a powerful approach for solving continuous control problems like bipedal locomotion, which requires agents to learn stable walking while optimizing multiple objectives: forward progress, energy efficiency, and postural stability.

**Research Contributions:**
- Systematic comparison of TD3, SAC, and PPO on bipedal walking tasks
- Evaluation across three terrain variants (easy, hardcore, hardcore+bridges)
- Performance metrics from comprehensive evaluation framework
- Training convergence analysis and stability comparison

**Key Features:**
- **Multiple RL Algorithms**: TD3, SAC, and PPO with Stable Baselines3 implementation
- **Custom Environment**: BipedalWalker with bridge obstacles requiring sophisticated navigation
- **Intelligent Wrappers**: HardcoreWrapper and BridgeBalancedWrapper for enhanced learning
- **GPU-Accelerated Training**: CUDA/MPS support for faster training
- **Comprehensive Tooling**: Evaluation, visualization, and analysis scripts
- **Reproducible Experiments**: Fixed seeds and detailed configuration files

## 📁 Project Structure

```
TASI_bipedal_walker/
├── scripts/
│   ├── training/
│   │   ├── train_sac.py              # SAC training script
│   │   └── train_td3.py          # TD3 training script
│   ├── evaluation/
│   │   ├── evaluate_sb3_models.py    # Comprehensive model evaluation
│   │   ├── record_sb3_video.py       # Video recording from models
│   │   └── analyze_train_history.py  # Training analysis tool
│   ├── analyze_tensorboard.py        # TensorBoard log analysis
│   └── plot_train_history_td3.py     # TD3 training visualization
├── src/
│   ├── envs/
│   │   └── custom_walker.py          # Custom BipedalWalker with bridges
│   └── wrappers/
│       ├── bridge_balanced_wrapper.py # Bridge navigation wrapper
│       └── hardcore_wrappers.py       # Hardcore environment wrapper
├── configs/
│   ├── sac_easy_gpu.yaml             # SAC easy mode config
│   ├── sac_hardcore_gpu.yaml         # SAC hardcore config
│   ├── sac_bridges_gpu.yaml          # SAC bridges config
│   ├── td3_easy_new.yaml             # TD3 easy mode config
│   ├── td3_hardcore_test.yaml        # TD3 hardcore config
│   └── td3_hardcore_advanced_bridges_new.yaml  # TD3 bridges config
├── experiments/
│   ├── checkpoints/                  # Trained model checkpoints
│   ├── logs/                         # TensorBoard logs
│   └── videos/                       # Recorded episode videos
├── report/
│   ├── report.tex                    # Full research report
│   └── figures/                      # Report figures and plots
└── archived/                         # Deprecated implementations (reference only)
```

## 🚀 Quick Start

### Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  

# Install dependencies
pip install -r requirements.txt
```

### Training

#### TD3 (Twin Delayed DDPG)

TD3 addresses DDPG's overestimation bias through three key mechanisms: twin critics, delayed policy updates, and target smoothing.

```bash
# Easy mode (standard terrain)
python scripts/training/train_generic.py --config configs/td3_easy_new.yaml

# Hardcore mode (obstacles: stairs, pits, stumps)
python scripts/training/train_generic.py --config configs/td3_hardcore_test.yaml

# Hardcore with bridges (extreme obstacles with dynamic bridges)
python scripts/training/train_generic.py --config configs/td3_hardcore_advanced_bridges_new.yaml
```


#### SAC (Soft Actor-Critic)

SAC maximizes entropy-regularized rewards, balancing exploration and exploitation through automatic temperature tuning.

```bash
# Easy mode (standard terrain)
python scripts/training/train_sac.py --config configs/sac_easy_gpu.yaml

# Hardcore mode (obstacles: stairs, pits, stumps)
python scripts/training/train_sac.py --config configs/sac_hardcore_gpu.yaml

# Hardcore with bridges (extreme obstacles with dynamic bridges)
python scripts/training/train_sac.py --config configs/sac_bridges_gpu.yaml
```

**Training Outputs:**
- Model checkpoints: `experiments/checkpoints/<run_name>/`
- TensorBoard logs: `experiments/logs/<run_name>/`
- Best model: `best_model.zip` (automatically saved when reaching best performance)
- VecNormalize stats: `*_vecnormalize.pkl` (required for evaluation)

### Evaluation

Comprehensive model evaluation with detailed metrics and visualizations:

```bash
# TD3 Easy mode (50 episodes by default)
python scripts/evaluation/evaluate_sb3_models.py --model-dir experiments/checkpoints/td3_easy

# TD3 Hardcore mode (1000 episodes for robust statistics)
python scripts/evaluation/evaluate_sb3_models.py --model-dir experiments/checkpoints/td3_hardcore --episodes 1000

# TD3 Bridges mode
python scripts/evaluation/evaluate_sb3_models.py --model-dir experiments/checkpoints/td3_hardcore_bridges --episodes 1000

# SAC Easy mode
python scripts/evaluation/evaluate_sb3_models.py --model-dir experiments/checkpoints/sac_easy

# SAC Hardcore mode
python scripts/evaluation/evaluate_sb3_models.py --model-dir experiments/checkpoints/sac_hardcore --episodes 1000

# SAC Bridges mode
python scripts/evaluation/evaluate_sb3_models.py --model-dir experiments/checkpoints/sac_bridges --episodes 1000
```

**Evaluation Metrics:**
- **Performance**: Mean/median/min/max rewards, success rate (reward > 300)
- **Episode Statistics**: Mean/std/min/max episode lengths
- **Outputs**: 
  - CSV file: `<model_dir>/evaluation_results.csv` (per-episode data)
  - Plots: `<model_dir>/evaluation_plots.png` (reward and length distributions)

### Recording Videos

Record agent performance videos for visualization:

```bash
# TD3 models
python scripts/evaluation/record_sb3_video.py --model-dir experiments/checkpoints/td3_easy --episodes 3
python scripts/evaluation/record_sb3_video.py --model-dir experiments/checkpoints/td3_hardcore --episodes 3
python scripts/evaluation/record_sb3_video.py --model-dir experiments/checkpoints/td3_hardcore_bridges --episodes 3

# SAC models
python scripts/evaluation/record_sb3_video.py --model-dir experiments/checkpoints/sac_easy --episodes 3
python scripts/evaluation/record_sb3_video.py --model-dir experiments/checkpoints/sac_hardcore --episodes 3
python scripts/evaluation/record_sb3_video.py --model-dir experiments/checkpoints/sac_bridges --episodes 3
```

Videos are saved to: `experiments/videos/<model_name>/`

### Training Analysis

Analyze TensorBoard logs to visualize training progress:

```bash
# Analyze TD3 easy training
python scripts/analyze_tensorboard.py --log-dir experiments/logs/td3_easy

# Analyze SAC hardcore training
python scripts/analyze_tensorboard.py --log-dir experiments/logs/sac_hardcore

# Compare multiple TD3 runs
python scripts/plot_train_history_td3.py --output plots/td3_comparison.png
```

**Analysis Outputs:**
- Training progress plots (rewards, losses, FPS)
- Summary statistics (final performance, convergence speed)
- Saved plot: `<log_dir>/training_analysis.png`

### TensorBoard

View real-time training metrics:

```bash
tensorboard --logdir experiments/logs/
```

Navigate to `http://localhost:6006` to view:
- Episode rewards and lengths
- Actor/critic losses
- Learning rate schedule
- Training FPS

## 🎮 Environment Variants

### BipedalWalker-v3

**State Space (24D):**
- Hull state: angle, angular velocity (2D)
- Velocities: horizontal, vertical (2D)
- Joint states: 4 angles + 4 velocities (8D)
- Ground contacts: leg-ground contact flags (2D)
- LIDAR perception: rangefinder measurements (10D)

**Action Space (4D):**
- Continuous motor torques for hips and knees
- Normalized to $[-1, 1]$

**Reward Function:**
$$r_t = 130 \cdot x_{\text{progress}} - 5|\theta| - 0.00035 \sum_i |a_i| - 100 \cdot \mathbb{1}_{\text{fallen}}$$

**Success Criterion:** Mean episode reward ≥ 300

### Environment Modes

1. **Easy (Standard Terrain)**
   - Flat terrain with minor slopes
   - Tests basic locomotion skills
   - ID: `BipedalWalker-v3`

2. **Hardcore**
   - Obstacles: stairs, pits, stumps, ladders
   - Tests robustness and adaptability
   - ID: `BipedalWalkerHardcore-v3`

3. **Hardcore + Bridges**
   - All hardcore obstacles plus dynamic bridges
   - Bridges open/close after 3 seconds
   - Most challenging variant requiring sophisticated navigation
   - ID: `CustomBipedalWalker-v3` (custom implementation)

## � Custom Wrappers

### HardcoreWrapper

Enhances hardcore environments with smoothness penalties and frame skipping:

**Features:**
- Frame skip: 4 (reduces action frequency for smoother control)
- Smoothness penalty: Penalizes jerky movements
- Angle stability: Penalizes excessive hull angle changes
- Reward clipping: Prevents extreme reward values

**Location:** [src/wrappers/hardcore_wrappers.py](src/wrappers/hardcore_wrappers.py)

### BridgeBalancedWrapper

Specialized wrapper for bridge navigation with LIDAR-based detection:

**Features:**
- Bridge detection using LIDAR readings
- Reward bonuses for stable waiting before bridges
- Crossing bonuses for successful bridge navigation
- Smoothness penalties to encourage natural movement

**Location:** [src/wrappers/bridge_balanced_wrapper.py](src/wrappers/bridge_balanced_wrapper.py)

## �📊 Configuration
All configurations are in YAML format in the [configs/](configs/) directory:

### TD3 Configs
- `td3_easy_new.yaml` - Standard terrain (1M steps)
- `td3_hardcore_test.yaml` - Hardcore obstacles (2M steps)
- `td3_hardcore_advanced_bridges_new.yaml` - Bridges mode (2M steps)

### SAC Configs
- `sac_easy_gpu.yaml` - Standard terrain (1M steps)
- `sac_hardcore_gpu.yaml` - Hardcore obstacles (2M steps)
- `sac_bridges_gpu.yaml` - Bridges mode (2M steps)


## 🔗 References

1. Fujimoto, S., Hoof, H., & Meger, D. (2018). **Addressing Function Approximation Error in Actor-Critic Methods**. *ICML 2018* (TD3)
2. Haarnoja, T., Zhou, A., Abbeel, P., & Levine, S. (2018). **Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor**. *ICML 2018* (SAC)
3. Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). **Proximal Policy Optimization Algorithms**. *arXiv:1707.06347* (PPO)
4. **OpenAI Gymnasium** - BipedalWalker-v3 Environment Documentation
5. **Stable Baselines3** - PyTorch implementations of RL algorithms

## 📄 Report

For detailed analysis, methodology, and comprehensive results, see the full research report:
- [report/report.tex](report/report.tex) - LaTeX source
- [report/figures/](report/figures/) - Generated plots and figures

---

**Project**: TASI Bipedal Walker - Comparative Analysis of Deep RL Algorithms  
**Authors**: Helena Alves, José Evans, Mariana Lobão  
**Date**: December 2025

````
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
