#!/bin/bash
# Setup script for Bipedal Walker RL Project

echo "=========================================="
echo "Bipedal Walker RL Project Setup"
echo "=========================================="

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Found Python $python_version"

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo ""
echo "Creating experiment directories..."
mkdir -p experiments/checkpoints
mkdir -p experiments/logs
mkdir -p experiments/videos
mkdir -p data
mkdir -p notebooks

echo ""
echo "=========================================="
echo "Setup completed successfully!"
echo "=========================================="
echo ""
echo "To activate the environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "To start training, run:"
echo "  python train.py --config configs/ppo_config.yaml"
echo ""
echo "To view logs in TensorBoard:"
echo "  tensorboard --logdir experiments/logs"
echo ""
