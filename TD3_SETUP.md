# TD3 Training Setup Guide

## Overview
You now have complete support for training your Bipedal Walker agent using the **TD3 (Twin Delayed Deep Deterministic Policy Gradient)** algorithm. Here's what was created:

## Files Created/Modified

### 1. **TD3 Agent Implementation**
- **File**: `src/agents/td3_agent.py`
- **Features**:
  - Twin critics (Q1, Q2) for robust Q-value estimation
  - Target smoothing regularization (TSMR) with configurable noise
  - Delayed policy updates to address overestimation
  - Soft target network updates

### 2. **Configuration Files**
Created 4 different TD3 configuration files in the `configs/` directory:

#### a. `td3_config.yaml` (Default/Balanced)
- **Learning rate**: 3e-4
- **Buffer capacity**: 1,000,000
- **Batch size**: 256
- **Total timesteps**: 1,000,000
- **Device**: CPU
- **Use case**: General purpose training

#### b. `td3_gpu_optimized.yaml` (For GPU Training)
- **Larger networks**: [512, 512] hidden dims
- **Learning rate**: 1e-3 (higher)
- **Buffer capacity**: 2,000,000
- **Batch size**: 512
- **Total timesteps**: 2,000,000
- **Device**: CUDA
- **Use case**: Fast training on GPU

#### c. `td3_conservative.yaml` (Stability-Focused)
- **Learning rate**: 1e-4 (lower for stability)
- **Tau**: 0.001 (conservative target updates)
- **Target noise**: 0.3 (more exploration)
- **Policy update frequency**: 3 (less frequent updates)
- **Batch size**: 128
- **Use case**: When you need stable, reliable learning

#### d. `td3_aggressive.yaml` (Fast Convergence)
- **Learning rate**: 1e-3 (higher)
- **Tau**: 0.01 (aggressive target updates)
- **Target noise**: 0.1 (less exploration noise)
- **Policy update frequency**: 1 (frequent updates)
- **Batch size**: 512
- **Use case**: Quick experiments, when stability isn't a priority

### 3. **Updated train.py**
- Added `TD3Agent` to imports
- Updated `create_agent()` function to handle TD3 configuration parameters
- Added `train_td3()` function (mirrors `train_sac()` logic)
- Updated `main()` to support 'td3' agent type

### 4. **Updated src/agents/__init__.py**
- Exported `TD3Agent` for easy importing

## No Changes Required to Your Code Flow!

**Important**: You do NOT need to significantly modify `train.py` beyond what was already done. The TD3 training loop is nearly identical to SAC:
- Uses the same replay buffer system
- Same evaluation and checkpoint saving logic
- Same logging infrastructure

## How to Train with TD3

### Basic Training
```bash
python train.py --config configs/td3_config.yaml
```

### GPU-Optimized Training
```bash
python train.py --config configs/td3_gpu_optimized.yaml
```

### Conservative Training
```bash
python train.py --config configs/td3_conservative.yaml
```

### Aggressive Training
```bash
python train.py --config configs/td3_aggressive.yaml
```

## Key TD3 Hyperparameters Explained

| Parameter | Default | Range | Impact |
|-----------|---------|-------|--------|
| `learning_rate` | 3e-4 | 1e-5 to 1e-3 | Higher = faster learning but less stable |
| `tau` | 0.005 | 0.001 to 0.1 | Higher = more aggressive target updates |
| `target_noise` | 0.2 | 0.1 to 0.5 | Higher = more exploration in target actions |
| `noise_clip` | 0.5 | 0.2 to 1.0 | Limits magnitude of target smoothing noise |
| `policy_update_freq` | 2 | 1 to 5 | Update actor every N critic updates (delayed) |

## Configuration Structure

All TD3 config files follow this structure:
```yaml
env:
  name: "BipedalWalker-v3"
  hardcore: false/true
  
agent:
  type: "td3"
  hidden_dims: [256, 256]
  learning_rate: 3.0e-4
  # ... other hyperparameters
  
buffer:
  capacity: 1000000
  batch_size: 256
  
training:
  total_timesteps: 1000000
  learning_starts: 10000
  eval_frequency: 10000
  eval_episodes: 10
  # ... other training settings
```

## TD3 vs SAC vs PPO

| Feature | TD3 | SAC | PPO |
|---------|-----|-----|-----|
| Algorithm Type | Off-policy | Off-policy | On-policy |
| Sample Efficiency | High | High | Low |
| Stability | Very High | High | Medium |
| Exploration | Via target noise | Via entropy | Via policy stochasticity |
| Best For | Continuous control | Continuous control | General RL |
| Memory Usage | Moderate | Moderate | Low |

## Next Steps

1. **Try a quick test** with TD3:
   ```bash
   python train.py --config configs/td3_config.yaml
   ```

2. **Monitor training** with TensorBoard:
   ```bash
   tensorboard --logdir experiments/logs/td3_bipedal_walker
   ```

3. **Tune hyperparameters** based on your results:
   - If training is too slow: increase `learning_rate`, increase `policy_update_freq`
   - If training is unstable: decrease `learning_rate`, decrease `tau`, decrease `target_noise`
   - If exploration is poor: increase `target_noise`, increase `policy_update_freq`

## Troubleshooting

### If training diverges:
- Use `td3_conservative.yaml` as a starting point
- Reduce `learning_rate`
- Reduce `target_noise`

### If training is too slow:
- Use `td3_gpu_optimized.yaml` if you have GPU
- Increase `learning_rate`
- Increase `policy_update_freq` (update actor more often)

### If model doesn't improve:
- Increase `learning_starts` to explore longer initially
- Try `td3_aggressive.yaml` with careful monitoring
- Check that `evaluation` rewards are being logged correctly

## Files Reference

- **Agent**: `src/agents/td3_agent.py`
- **Configs**: `configs/td3_*.yaml`
- **Training Script**: `train.py` (modified)
- **Agent Registry**: `src/agents/__init__.py` (modified)

Enjoy training! 🚀
