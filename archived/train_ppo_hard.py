import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from custom_walker import BipedalWalker
import os

# Create folders
os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)

if __name__ == "__main__":
    # 1. Create the Custom Hardcore Environment
    # Now we set hardcore=True to spawn the bridge and pit
    env = SubprocVecEnv([lambda: BipedalWalker(hardcore=True) for _ in range(16)])

    # 2. LOAD THE "GLASSES" (Normalization Stats)
    # Critical: The robot needs to see the world the same way it did in Phase 1
    env = VecNormalize.load("models/vec_normalize_phase1.pkl", env)
    
    # Keep updating stats (training=True) because the Bridge is a "New" thing to see
    env.training = True 
    env.norm_reward = True

    # 3. Load the Phase 1 Brain
    model = PPO.load(
        "models/walker_phase1_easy", 
        env=env,
        custom_objects={"learning_rate": 1e-4}
    )

    model.ent_coef = 0.01 

    print("--- PHASE 2: LEARNING THE HARDCORE ---")
    # Train for another 4-5 Million Steps
    model.learn(total_timesteps=5_000_000)

    model.save("models/walker_phase2")
    env.save("models/vec_normalize_phase2.pkl")
    env.close()