import gymnasium as gym
from stable_baselines3 import PPO
from custom_walker import BipedalWalker
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.monitor import Monitor  # <--- CHANGE 1: Import Monitor
import os
import torch.nn as nn
import sys

# --- CONFIGURATION ---
CONFIG = {
    "rollout_steps": 2048,
    "eval_frequency": 10000,
    "eval_episodes": 10,
    "save_frequency": 50000,
    "total_timesteps": 5_000_000,
    "num_envs": 16
}

# # Wrapper Import
# try:
#     from elite_hardcore_bridge_wrapper import EliteHardcoreBridgeWrapper 
# except ImportError:
#     print("Warning: Wrapper not found. Make sure elite_hardcore_wrapper.py is present.")
#     sys.exit(1)

# Create folders
os.makedirs("models", exist_ok=True)
os.makedirs("logs_comparison", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

if __name__ == "__main__":
    RUN_NAME = "ppo_hard_bridge_no_reward"

    print(f"--- INITIALIZING {RUN_NAME} ---")

    # 1. Create Training Envs
    # CHANGE 2: Wrap the environment in Monitor()
    # Logic: Gym -> EliteWrapper -> Monitor -> SubprocVecEnv
    train_env = SubprocVecEnv([
        lambda: Monitor(BipedalWalker(hardcore=True))
        for _ in range(CONFIG["num_envs"])
    ])
    
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.)

    # 2. Create Evaluation Env
    from stable_baselines3.common.vec_env import DummyVecEnv
    
    # CHANGE 3: Wrap the Eval env in Monitor() too
    # This ensures the EvalCallback can correctly read the episode lengths and rewards
    eval_env = DummyVecEnv([
        lambda: Monitor(BipedalWalker(hardcore=True))
    ])
    
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True, clip_obs=10.)
    eval_env.training = False 
    eval_env.norm_reward = False 

    # 3. Create Callbacks
    eval_freq_adjusted = max(1, CONFIG["eval_frequency"] // CONFIG["num_envs"])
    save_freq_adjusted = max(1, CONFIG["save_frequency"] // CONFIG["num_envs"])

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"./models/{RUN_NAME}_best/",
        log_path=f"./logs_comparison/{RUN_NAME}_eval/",
        eval_freq=eval_freq_adjusted,
        n_eval_episodes=CONFIG["eval_episodes"],
        deterministic=True,
        render=False
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq_adjusted,
        save_path='./checkpoints/',
        name_prefix=RUN_NAME,
        save_replay_buffer=False,
        save_vecnormalize=True
    )

    callback_list = CallbackList([eval_callback, checkpoint_callback])

    # 4. Define Policy
    policy_kwargs = dict(
        activation_fn=nn.Tanh,
        net_arch=dict(pi=[256, 256], vf=[256, 256])
    )

    # 5. Model Setup
    model = PPO(
        "MlpPolicy",
        train_env,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,      
        n_steps=CONFIG["rollout_steps"], 
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,          
        verbose=1,
        tensorboard_log="./logs_comparison/"
    )

    # 6. Train
    model.learn(
        total_timesteps=CONFIG["total_timesteps"], 
        callback=callback_list,
        tb_log_name=RUN_NAME
    )

    # 7. Final Save
    model.save(f"models/walker_{RUN_NAME}_final")
    train_env.save(f"models/vec_normalize_{RUN_NAME}_final.pkl")
    
    print("Done!")
    train_env.close()
    eval_env.close()