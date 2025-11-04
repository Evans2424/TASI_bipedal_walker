# Quick Start Guide

Get started with training your Bipedal Walker agent in 5 minutes!

## 1. Setup (One-time)

```bash
# Run the setup script
./setup.sh

# Or manually:
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 2. Train Your First Agent

### Option A: PPO (Recommended for beginners)
```bash
python train.py --config configs/ppo_config.yaml
```

### Option B: SAC (More sample-efficient)
```bash
python train.py --config configs/sac_config.yaml
```

## 3. Monitor Training

Open a new terminal and run:
```bash
tensorboard --logdir experiments/logs
```

Then visit: http://localhost:6006

## 4. Evaluate Your Agent

```bash
python scripts/evaluate.py \
    --checkpoint experiments/checkpoints/ppo_bipedal_walker/final_model.pt \
    --config configs/ppo_config.yaml \
    --episodes 10 \
    --render
```

## 5. Record Videos

```bash
python scripts/record_video.py \
    --checkpoint experiments/checkpoints/ppo_bipedal_walker/final_model.pt \
    --config configs/ppo_config.yaml \
    --episodes 3
```

## Expected Results

- **Training Time**: 2-4 hours on L40 GPU for PPO
- **Success Threshold**: 300+ reward
- **Convergence**: Usually within 1-2M timesteps

## Common Issues

**ImportError: No module named 'Box2D'**
```bash
# macOS
brew install swig
pip install box2d-py

# Ubuntu
sudo apt-get install swig
pip install box2d-py
```

**CUDA out of memory**
- Reduce batch_size in config
- Reduce hidden_dims to [128, 128]

**Slow training**
- Verify GPU is being used: check TensorBoard for GPU metrics
- Ensure CUDA version matches PyTorch installation

## Next Steps

1. Read the full README.md for detailed documentation
2. Experiment with hyperparameters in config files
3. Try hardcore mode after mastering normal mode
4. Visualize your results with plot_results.py

## File Structure Summary

```
Key Files:
├── train.py                    # Main training script
├── configs/
│   ├── ppo_config.yaml        # PPO settings
│   └── sac_config.yaml        # SAC settings
├── scripts/
│   ├── evaluate.py            # Evaluation
│   ├── record_video.py        # Video recording
│   └── plot_results.py        # Plotting
└── src/
    ├── agents/                # RL algorithms
    ├── models/                # Neural networks
    ├── envs/                  # Environment wrappers
    └── utils/                 # Utilities
```

## Support

- Check README.md for detailed documentation
- Review config files for hyperparameter options
- Examine source code in src/ for implementation details

Happy training! 🦿🤖
