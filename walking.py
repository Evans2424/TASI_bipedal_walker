from stable_baselines3 import PPO
import gymnasium as gym

env = gym.make("BipedalWalker-v3", render_mode="human")
model = PPO.load("ppo_bipedalwalker_parallel.zip")

obs, info = env.reset()
terminated = truncated = False

while not (terminated or truncated):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)

env.close()