import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
import os
import sys

# --- IMPORT YOUR WRAPPER ---
try:
    from elite_hardcore_bridge_wrapper import EliteHardcoreBridgeWrapper 
except ImportError:
    print("Error: Wrapper not found.")
    sys.exit(1)

# --- CONFIGURATION ---
# 1. Point to the LAST working checkpoint
# Look in your "checkpoints" folder. You will see files like:
# "ppo_hard_base_1000000_steps.zip"
# "ppo_hard_base_vecnormalize_1000000_steps.pkl"
CHECKPOINT_DIR = "./checkpoints/"
RUN_NAME = "ppo_hard_bridge"
STEPS_ALREADY_DONE = 1_250_000  # Change this to the number on your file
TOTAL_TARGET_STEPS = 5_000_000
NUM_ENVS = 16

if __name__ == "__main__":
    print(f"--- RESUMING TRAINING FROM STEP {STEPS_ALREADY_DONE} ---")

    # 1. Re-create the Environment
    # It must be identical to the original one
    env = SubprocVecEnv([
        lambda: Monitor(EliteHardcoreBridgeWrapper(gym.make("BipedalWalkerHardcore-v3", render_mode=None)))
        for _ in range(NUM_ENVS)
    ])
    
    # 2. Load the Normalization Statistics
    # We don't create a new VecNormalize; we LOAD the old one.
    stats_path = os.path.join(CHECKPOINT_DIR, f"{RUN_NAME}_vecnormalize_{STEPS_ALREADY_DONE}_steps.pkl")
    if not os.path.exists(stats_path):
        print(f"❌ Error: Could not find stats file at {stats_path}")
        sys.exit(1)
        
    env = VecNormalize.load(stats_path, env)
    env.training = True   # Ensure it keeps updating stats
    env.norm_reward = True

    # 3. Load the Model
    model_path = os.path.join(CHECKPOINT_DIR, f"{RUN_NAME}_{STEPS_ALREADY_DONE}_steps.zip")
    if not os.path.exists(model_path):
        print(f"❌ Error: Could not find model file at {model_path}")
        sys.exit(1)

    # We load the model and attach the loaded env to it
    model = PPO.load(model_path, env=env)

    # 4. Re-create Callbacks
    # (Same as before, so you keep saving new checkpoints)
    # Note: We need a new Eval Env too
    from stable_baselines3.common.vec_env import DummyVecEnv
    eval_env = DummyVecEnv([lambda: Monitor(EliteHardcoreBridgeWrapper(gym.make("BipedalWalkerHardcore-v3")))])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True, clip_obs=10.)
    eval_env.training = False
    eval_env.norm_reward = False

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"./models/{RUN_NAME}_best_resumed/",
        log_path=f"./logs_comparison/{RUN_NAME}_eval_resumed/",
        eval_freq=10000 // NUM_ENVS,
        n_eval_episodes=10,
        deterministic=True
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=50000 // NUM_ENVS,
        save_path='./checkpoints/',
        name_prefix=RUN_NAME, # Keep same name to maintain order
        save_vecnormalize=True
    )

    # 5. Calculate Remaining Steps
    steps_remaining = TOTAL_TARGET_STEPS - STEPS_ALREADY_DONE

    if steps_remaining <= 0:
        print("Training already finished!")
        sys.exit()

    print(f"Resuming for {steps_remaining} more steps...")

    # 6. Resume Training
    # reset_num_timesteps=False is CRITICAL.
    # It tells TensorBoard: "Don't start the graph at 0. Start at 1,000,000."
    model.learn(
        total_timesteps=steps_remaining, 
        callback=CallbackList([eval_callback, checkpoint_callback]),
        tb_log_name=RUN_NAME,
        reset_num_timesteps=False 
    )

    # Final Save
    model.save(f"models/walker_{RUN_NAME}_final_resumed")
    env.save(f"models/vec_normalize_{RUN_NAME}_final_resumed.pkl")
    print("Done!")