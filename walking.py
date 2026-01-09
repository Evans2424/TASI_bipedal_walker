# viewer_hardcore.py
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from custom_walker import BipedalWalker

HARDCORE = True
# PKL = "models/vec_normalize_ppo_hard_bridge_final_resumed.pkl"
# MODEL = "models/walker_ppo_hard_bridge_final_resumed"
PKL = "models/vec_normalize_phase2_1.pkl"
MODEL = "models/walker_phase2_bridge"

# 1. Setup the Hardcore Environment
env = DummyVecEnv([lambda: BipedalWalker(hardcore=HARDCORE, render_mode="human")])

# 2. LOAD THE NEW STATS (The Critical Step)
# You must load the PKL file that was saved at the END of your hardcore training.
# Do NOT load 'vec_normalize_easy.pkl' here.
env = VecNormalize.load(PKL, env)

# 3. Lock the stats for testing
env.training = False
env.norm_reward = False

# 4. Load the Hardcore Model
model = PPO.load(MODEL)

# 5. Run
obs = env.reset()
while True:
    action, _ = model.predict(obs, deterministic=False)
    obs, _, done, _ = env.step(action)