"""Logging utilities for training metrics and visualization."""

import os
import json
import numpy as np
from typing import Dict, Any, Optional
from torch.utils.tensorboard import SummaryWriter


class Logger:
    """Logger for training metrics and TensorBoard visualization."""

    def __init__(self, log_dir: str, experiment_name: str):
        """Initialize logger.

        Args:
            log_dir: Directory to save logs
            experiment_name: Name of the experiment
        """
        self.log_dir = os.path.join(log_dir, experiment_name)
        os.makedirs(self.log_dir, exist_ok=True)

        # Initialize TensorBoard writer
        self.writer = SummaryWriter(log_dir=self.log_dir)

        # Initialize metrics storage
        self.episode_rewards = []
        self.episode_lengths = []
        self.metrics_history = {}

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """Log a scalar value to TensorBoard.

        Args:
            tag: Name of the metric
            value: Value to log
            step: Training step
        """
        self.writer.add_scalar(tag, value, step)

        # Store in history
        if tag not in self.metrics_history:
            self.metrics_history[tag] = []
        self.metrics_history[tag].append({"step": step, "value": value})

    def log_metrics(self, metrics: Dict[str, float], step: int, prefix: str = "") -> None:
        """Log multiple metrics at once.

        Args:
            metrics: Dictionary of metrics to log
            step: Training step
            prefix: Optional prefix for metric names
        """
        for key, value in metrics.items():
            tag = f"{prefix}/{key}" if prefix else key
            self.log_scalar(tag, value, step)

    def log_episode(self, episode_reward: float, episode_length: int, step: int) -> None:
        """Log episode statistics.

        Args:
            episode_reward: Total episode reward
            episode_length: Episode length
            step: Training step
        """
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_length)

        self.log_scalar("episode/reward", episode_reward, step)
        self.log_scalar("episode/length", episode_length, step)

        # Log running statistics (last 100 episodes)
        if len(self.episode_rewards) >= 10:
            recent_rewards = self.episode_rewards[-100:]
            recent_lengths = self.episode_lengths[-100:]

            self.log_scalar("episode/mean_reward_100", np.mean(recent_rewards), step)
            self.log_scalar("episode/std_reward_100", np.std(recent_rewards), step)
            self.log_scalar("episode/mean_length_100", np.mean(recent_lengths), step)

    def save_config(self, config: Dict[str, Any]) -> None:
        """Save experiment configuration.

        Args:
            config: Configuration dictionary
        """
        config_path = os.path.join(self.log_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)

    def get_stats(self) -> Dict[str, float]:
        """Get current training statistics.

        Returns:
            Dictionary of statistics
        """
        if len(self.episode_rewards) == 0:
            return {}

        recent_rewards = self.episode_rewards[-100:]

        return {
            "episodes": len(self.episode_rewards),
            "mean_reward": np.mean(self.episode_rewards),
            "mean_reward_100": np.mean(recent_rewards),
            "std_reward_100": np.std(recent_rewards),
            "max_reward": np.max(self.episode_rewards),
            "min_reward": np.min(self.episode_rewards),
        }

    def close(self) -> None:
        """Close the logger."""
        self.writer.close()

    def __del__(self):
        """Cleanup on deletion."""
        self.close()
