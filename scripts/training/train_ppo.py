import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(root_dir)

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from gymnasium.envs.registration import register
import torch.nn as nn
import yaml
import argparse
from src.wrappers.hardcore_wrappers import HardcoreWrapper
from src.wrappers.bridge_balanced_wrapper import BridgeBalancedWrapper

register(
    id='CustomBipedalWalker-v3',
    entry_point='src.envs.custom_walker:BipedalWalker',
    max_episode_steps=2000,
    reward_threshold=300,
)

os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

DEFAULT_CONFIG = {
    "rollout_steps": 128,
    "eval_frequency": 50000,
    "eval_episodes": 10,
    "save_frequency": 50000,
    "total_timesteps": 5_000_000,
    "num_envs": 16,
    "master_seed": 42,
    "learning_rate": 1e-4,
    "gamma": 0.999,
    "ent_coef": 0.02,
    "use_hardcore_wrapper": "false",
    "use_bridge_wrapper": "false"
}

def load_config(args):
    config = DEFAULT_CONFIG.copy()
    
    if args.config:
        print(f"📄 Loading configuration from {args.config}...")
        with open(args.config, 'r') as f:
            yaml_config = yaml.safe_load(f)
            config.update(yaml_config)
            
    else:
        print("⚠️ No config file provided. Using DEFAULT_CONFIG.")

    return config

def make_env(rank, seed=42):
    def _init():
        env = gym.make(config["env"]["name"], hardcore=config["env"]["hardcore"])

        if config["env"]["use_hardcore_wrapper"]:
            wrapper_kwargs = {
                'smoothness_coef': config["env"].get('smoothness_coef', 0.2),
                'angle_coef': config["env"].get('hull_angle_coef', 0.1),
                'angular_vel_coef': config["env"].get('hull_angular_vel_coef', 0.05),
                'reward_clip_min': -10.0,
                'reward_clip_max': 10.0,
            }
            env = HardcoreWrapper(env, **wrapper_kwargs)

        elif config["env"]["use_bridge_wrapper"]:
            wrapper_kwargs = {
                'frame_skip': config["env"].get('frame_skip', 4),
                'smoothness_coef': config["env"].get('smoothness_coef', 0.02),
                'hull_angle_coef': config["env"].get('hull_angle_coef', 0.03),
                'hull_angular_vel_coef': config["env"].get('hull_angular_vel_coef', 0.015),
                'knee_bend_reward': config["env"].get('knee_bend_reward', 0.02),
                'min_bend_threshold': config["env"].get('min_bend_threshold', 0.3),
                'stable_waiting_bonus': config["env"].get('stable_waiting_bonus', 0.02),
                'bridge_cross_bonus': config["env"].get('bridge_cross_bonus', 8.0),
                'min_progress_for_bonuses': config["env"].get('min_progress_for_bonuses', 15.0),
                'max_waiting_steps': config["env"].get('max_waiting_steps', 400),
                'lidar_bridge_threshold': config["env"].get('lidar_bridge_threshold', 0.5),
                'min_close_beams': config["env"].get('min_close_beams', 3),
                'waiting_velocity_threshold': config["env"].get('waiting_velocity_threshold', 0.15),
                'waiting_angle_threshold': config["env"].get('waiting_angle_threshold', 0.3),
            }
            env = BridgeBalancedWrapper(env, **wrapper_kwargs)
        
        env.reset(seed=seed + rank)
        
        env = Monitor(env)
        return env
    return _init

def train_ppo(config):
    runname = config["experiment"]["name"]

    env = SubprocVecEnv([
        make_env(rank=i, seed=config["experiment"]["seed"]) 
        for i in range(config["training"]["num_envs"])
    ])

    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

    eval_env = DummyVecEnv([lambda: Monitor(gym.make(config["env"]["name"], hardcore=config["env"]["hardcore"]))])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True, clip_obs=10.)

    policy_kwargs = dict(
        activation_fn=nn.Tanh,
        net_arch=dict(pi=config["agent"]["hidden_dims"], vf=config["agent"]["hidden_dims"]),
    )

    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=config["agent"]["learning_rate"],
        n_steps=config["agent"]["n_steps"],
        batch_size=config["agent"]["batch_size"],
        n_epochs=config["agent"]["n_epochs"],
        gamma=config["agent"]["gamma"],
        gae_lambda=config["agent"]["gae_lambda"],
        clip_range=config["agent"]["clip_range"],
        verbose=1,
        tensorboard_log=f"./logs/{config['experiment']['name']}/"
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"./models/{runname}_best/",
        log_path=f"./logs/{runname}_eval/",
        eval_freq=config["training"]["eval_frequency"], # Evaluate once per update
        deterministic=True,
        render=False
    )

    print(f"Training {config['env']['name']} for {config['training']['total_timesteps']} timesteps...")

    model.learn(total_timesteps=config["training"]["total_timesteps"], tb_log_name=runname, callback=eval_callback)

    model.save(f"models/{runname}_final")
    env.save(f"models/vec_normalize_{runname}.pkl")
    
    print("Train complete.")
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BipedalWalker Hardcore")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    args = parser.parse_args()

    # Load Configuration
    config = load_config(args)
    train_ppo(config)