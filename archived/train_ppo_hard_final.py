import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
import os
import torch.nn as nn
import sys

# --- CONFIGURATION (Based on working SAC config) ---
CONFIG = {
    # PPO-specific settings (tuned for hardcore environments)
    "rollout_steps": 2048,          # Collect 2048 steps per env before update
    "batch_size": 256,              # Larger batches for stability
    "n_epochs": 10,                 # More epochs per update
    "learning_rate": 3e-4,          # Standard PPO learning rate
    "clip_range": 0.2,              # Standard PPO clipping
    "gae_lambda": 0.95,             # GAE parameter
    "gamma": 0.99,                  # Discount factor
    "ent_coef": 0.0,                # Entropy coefficient (start with 0)
    "vf_coef": 0.5,                 # Value function coefficient
    "max_grad_norm": 0.5,           # Gradient clipping
    
    # Training settings
    "total_timesteps": 10_000_000,  # 10M like working SAC
    "num_envs": 8,                  # 8 parallel envs like SAC
    "eval_frequency": 25000,        # Evaluate less frequently like SAC
    "eval_episodes": 10,
    "save_frequency": 100000,       # Save every 100k steps like SAC
    
    # Network architecture (similar to SAC's [400, 300])
    "pi_layers": [256, 256],        # Policy network
    "vf_layers": [256, 256],        # Value network
}

# Wrapper Import - using the same wrapper that worked for SAC
try:
    from wrappers.elite_hardcore_wrapper import EliteHardcoreWrapper
except ImportError:
    print("Error: EliteHardcoreWrapper not found in wrappers/")
    print("Make sure wrappers/elite_hardcore_wrapper.py exists")
    sys.exit(1)

# Create folders
os.makedirs("models", exist_ok=True)
os.makedirs("logs_comparison", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

if __name__ == "__main__":
    RUN_NAME = "ppo_elite_hardcore"
    
    print("=" * 60)
    print(f"PPO TRAINING - HARDCORE MODE")
    print("=" * 60)
    print(f"Run: {RUN_NAME}")
    print(f"Total timesteps: {CONFIG['total_timesteps']:,}")
    print(f"Parallel envs: {CONFIG['num_envs']}")
    print(f"Using EliteHardcoreWrapper (same as working SAC)")
    print("=" * 60)

    # 1. Create Training Environments
    # Same structure as working SAC: Gym -> EliteWrapper -> Monitor -> SubprocVecEnv -> VecNormalize
    def make_train_env(rank):
        def _init():
            env = gym.make("BipedalWalkerHardcore-v3", render_mode=None)
            # Apply the same wrapper configuration that worked for SAC
            env = EliteHardcoreWrapper(
                env,
                frame_skip=4,                    # Same as SAC
                smoothness_coef=0.2,             # Same as SAC  
                hull_angle_coef=0.1,             # Same as SAC
                hull_angular_vel_coef=0.05,      # Same as SAC
                knee_bend_reward=0.02,           # Same as SAC
                min_bend_threshold=0.3,          # Same as SAC
                max_joint_velocity=2.0,          # Same as SAC
                velocity_penalty=0.02,           # Same as SAC
                early_steps_stability_bonus=0.01,# Same as SAC
                early_steps_count=100            # Same as SAC
            )
            env = Monitor(env)
            env.reset(seed=42 + rank)
            return env
        return _init
    
    train_env = SubprocVecEnv([make_train_env(i) for i in range(CONFIG["num_envs"])])
    
    # VecNormalize settings - same as SAC
    train_env = VecNormalize(
        train_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=CONFIG["gamma"]
    )

    # 2. Create Evaluation Environment
    def make_eval_env():
        env = gym.make("BipedalWalkerHardcore-v3", render_mode=None)
        env = EliteHardcoreWrapper(
            env,
            frame_skip=4,
            smoothness_coef=0.2,
            hull_angle_coef=0.1,
            hull_angular_vel_coef=0.05,
            knee_bend_reward=0.02,
            min_bend_threshold=0.3,
            max_joint_velocity=2.0,
            velocity_penalty=0.02,
            early_steps_stability_bonus=0.01,
            early_steps_count=100
        )
        env = Monitor(env)
        return env
    
    eval_env = DummyVecEnv([make_eval_env])
    eval_env = VecNormalize(
        eval_env,
        norm_obs=True,
        norm_reward=False,   # Don't normalize rewards during eval
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=CONFIG["gamma"],
        training=False       # Don't update stats during eval
    ) 

    # 3. Create Callbacks
    # Adjust frequencies for vectorized environments
    eval_freq_adjusted = max(1, CONFIG["eval_frequency"] // CONFIG["num_envs"])
    save_freq_adjusted = max(1, CONFIG["save_frequency"] // CONFIG["num_envs"])

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"./models/{RUN_NAME}_best/",
        log_path=f"./logs_comparison/{RUN_NAME}_eval/",
        eval_freq=eval_freq_adjusted,
        n_eval_episodes=CONFIG["eval_episodes"],
        deterministic=True,
        render=False,
        verbose=1
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq_adjusted,
        save_path='./checkpoints/',
        name_prefix=RUN_NAME,
        save_replay_buffer=False,
        save_vecnormalize=True,
        verbose=1
    )

    callback_list = CallbackList([eval_callback, checkpoint_callback])

    # 4. Define Policy Network Architecture
    # Similar to SAC's [400, 300] but adapted for PPO
    policy_kwargs = dict(
        activation_fn=nn.Tanh,
        net_arch=dict(
            pi=CONFIG["pi_layers"],   # Policy network: [256, 256]
            vf=CONFIG["vf_layers"]    # Value network: [256, 256]
        )
    )

    # 5. Create PPO Model
    print("\nInitializing PPO model...")
    model = PPO(
        "MlpPolicy",
        train_env,
        policy_kwargs=policy_kwargs,
        
        # Learning settings
        learning_rate=CONFIG["learning_rate"],
        n_steps=CONFIG["rollout_steps"],
        batch_size=CONFIG["batch_size"],
        n_epochs=CONFIG["n_epochs"],
        
        # PPO-specific
        gamma=CONFIG["gamma"],
        gae_lambda=CONFIG["gae_lambda"],
        clip_range=CONFIG["clip_range"],
        clip_range_vf=None,                    # No value function clipping
        
        # Regularization
        ent_coef=CONFIG["ent_coef"],
        vf_coef=CONFIG["vf_coef"],
        max_grad_norm=CONFIG["max_grad_norm"],
        
        # Other settings
        normalize_advantage=True,               # Important for stability
        use_sde=False,                          # Don't use state-dependent exploration
        sde_sample_freq=-1,
        target_kl=None,                         # No KL divergence target
        
        verbose=1,
        tensorboard_log="./logs_comparison/",
        seed=42
    )

    print("\nModel initialized successfully!")
    print(f"Device: {model.device}")
    print(f"Policy architecture: {policy_kwargs['net_arch']}")

    # 6. Train
    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)
    
    model.learn(
        total_timesteps=CONFIG["total_timesteps"],
        callback=callback_list,
        tb_log_name=RUN_NAME,
        progress_bar=True
    )

    # 7. Final Save
    print("\n" + "=" * 60)
    print("TRAINING COMPLETED - SAVING MODELS")
    print("=" * 60)
    
    final_model_path = f"models/{RUN_NAME}_final"
    final_vecnorm_path = f"models/{RUN_NAME}_vecnormalize_final.pkl"
    
    model.save(final_model_path)
    train_env.save(final_vecnorm_path)
    
    print(f"✓ Model saved to: {final_model_path}")
    print(f"✓ VecNormalize saved to: {final_vecnorm_path}")
    print("\nDone! Training completed successfully.")
    
    # Cleanup
    train_env.close()
    eval_env.close()