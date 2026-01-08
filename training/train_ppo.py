import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv

def make_env():
    return gym.make("BipedalWalker-v3")

if __name__ == "__main__":
    # Create 8 parallel environments
    env = SubprocVecEnv([make_env for _ in range(8)])

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        device="cpu",   # or "cpu"
    )

    model.learn(total_timesteps=3_000_000)

    model.save("ppo_bipedalwalker_parallel")
    env.close()

    print("Training finished!")
