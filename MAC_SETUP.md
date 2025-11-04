# Mac Setup Guide (Apple Silicon)

## Installation Complete! ✓

Your Bipedal Walker RL project is now fully set up on your Mac (Apple Silicon).

## What Was Done

1. ✓ Installed SWIG via Homebrew
2. ✓ Installed all Python dependencies (including Box2D)
3. ✓ Verified Gymnasium and PyTorch installation
4. ✓ Tested BipedalWalker-v3 environment
5. ✓ Updated configs for Mac (CPU mode)

## Device Configuration

Since you're on a Mac, the configs have been set to use **CPU** mode by default. You have three options:

### Option 1: CPU (Current Default - Recommended for Stability)
```yaml
device: "cpu"
```
- Most stable
- No setup required
- Training will be slower but reliable

### Option 2: MPS (Metal Performance Shaders - For Speed)
If you want to use your Mac's GPU acceleration, change device in configs:
```yaml
device: "mps"
```
- Faster than CPU
- Uses Apple Silicon GPU
- May have occasional compatibility issues

### Option 3: Remote Training
For fastest training, consider:
- Google Colab (free GPU)
- Your L40 GPU server (if available remotely)
- Cloud instances (AWS, GCP, etc.)

## Quick Start

### 1. Activate virtual environment:
```bash
source venv/bin/activate
```

### 2. Train your first agent (PPO):
```bash
python train.py --config configs/ppo_config.yaml
```

### 3. Monitor training (in another terminal):
```bash
tensorboard --logdir experiments/logs
```
Then open: http://localhost:6006

## Expected Performance on Mac

**CPU Training (Apple M-series):**
- PPO: ~500-1000 steps/sec
- SAC: ~300-600 steps/sec
- Training time: 6-12 hours for 1-2M timesteps

**MPS Training (if you switch to MPS):**
- PPO: ~2000-3000 steps/sec
- SAC: ~1000-2000 steps/sec
- Training time: 2-4 hours for 1-2M timesteps

## Testing MPS (Optional)

To test if MPS works on your Mac:
```bash
source venv/bin/activate
python -c "import torch; print('MPS available:', torch.backends.mps.is_available()); print('MPS built:', torch.backends.mps.is_built())"
```

If both are True, you can safely use `device: "mps"` in your configs.

## Common Mac Issues

### Issue: Pygame warnings
The pkg_resources deprecation warning is harmless and can be ignored.

### Issue: Training is too slow
- Switch to `device: "mps"` if available
- Reduce `total_timesteps` for faster experiments
- Use smaller networks: `hidden_dims: [128, 128]`

### Issue: Memory issues
- Reduce batch sizes in configs
- Reduce replay buffer capacity (SAC)

## Next Steps

1. **Read QUICKSTART.md** for usage guide
2. **Read README.md** for full documentation
3. **Start training**: `python train.py --config configs/ppo_config.yaml`
4. **Experiment** with hyperparameters

## File Locations

```
Your project:
├── train.py              # Main training script
├── configs/              # Algorithm configurations (CPU set)
├── scripts/              # Evaluation & visualization tools
├── src/                  # Core implementation
└── venv/                 # Virtual environment (activated)
```

## Useful Commands

```bash
# Activate environment
source venv/bin/activate

# Train PPO
python train.py --config configs/ppo_config.yaml

# Train SAC
python train.py --config configs/sac_config.yaml

# Evaluate trained model
python scripts/evaluate.py \
    --checkpoint experiments/checkpoints/ppo_bipedal_walker/final_model.pt \
    --config configs/ppo_config.yaml \
    --render

# Record videos
python scripts/record_video.py \
    --checkpoint experiments/checkpoints/ppo_bipedal_walker/final_model.pt \
    --config configs/ppo_config.yaml

# View logs
tensorboard --logdir experiments/logs
```

## Ready to Train!

Everything is set up and ready to go. Start training with:

```bash
source venv/bin/activate
python train.py --config configs/ppo_config.yaml
```

Good luck! 🦿🤖
