import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from custom_walker import BipedalWalker
import os
import torch.nn as nn

# Create folders
os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)

if __name__ == "__main__":
    RUN_NAME = "ppo_easy"
    
    # 1. Create 8 or 16 parallel environments (Speeds up training massively)
    # We use "hardcore=False" for the gym phase
    env = SubprocVecEnv([
        lambda: Monitor(BipedalWalker(hardcore=True)) 
        for _ in range(16)
    ])
    
    # 2. Add Normalization (Crucial for stability)
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

    policy_kwargs = dict(
        activation_fn=nn.Tanh,
        net_arch=dict(pi=[256, 256], vf=[256, 256])
    )

    # 3. PPO Hyperparameters optimized for BipedalWalker
    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,      # Standard start
        n_steps=2048,            # Long rollout
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.18,
        ent_coef=0.005,          # Small exploration bonus
        verbose=1,
        tensorboard_log="./logs/walker_phase1/"
    )

    print("--- PHASE 1: TRAINING THE ATHLETE (EASY MODE) ---")
    # Train for 4 Million Steps (1.5M is not enough!)
    model.learn(total_timesteps=2_000_000, tb_log_name=RUN_NAME)

    # 4. Save EVERYTHING
    model.save("models/walker_phase1_easy")
    env.save("models/vec_normalize_phase1.pkl")
    
    print("Phase 1 Complete. Now the robot should walk perfectly.")
    print(f"Training Complete. Logs saved to ./logs_comparison/{RUN_NAME}_1")
    env.close()