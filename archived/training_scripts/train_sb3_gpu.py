"""Production-ready training using Stable-Baselines3.

This script uses stable-baselines3's production-ready implementations
with hardware acceleration (CUDA/MPS/CPU) and vectorized environments.

Key advantages over custom implementation:
- Battle-tested, optimized code
- Better callbacks and logging
- Easier hyperparameter tuning
- Multi-platform GPU support (CUDA, MPS)
- Advanced features (HER, recurrent policies, etc.)

Supports:
- CUDA (NVIDIA GPUs)
- MPS (Apple Silicon)
- CPU (fallback)
"""

import os
import sys
import argparse
import yaml
import logging
from typing import Dict, Callable, Optional
import torch
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, VecMonitor
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    CallbackList
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym

# Import hardcore wrappers
from hardcore_wrappers import HardcoreWrapper
# Import BIMQ wrappers
from bimq_wrappers import BIMQWrapper, make_bimq_env
# Import natural walking wrappers
from natural_walking_wrappers import NaturalWalkingWrapper, make_natural_walking_env
# Import human gait wrappers (research-based anti-jumping)
from human_gait_wrappers import HumanGaitWrapper, make_human_gait_env
# Import elite hardcore wrapper (unified hardcore + natural walking)
from wrappers.elite_hardcore_wrapper import EliteHardcoreWrapper

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def linear_schedule(initial_value: float):
    """Linear learning rate schedule that decreases from initial_value to 0.

    Args:
        initial_value: Initial learning rate

    Returns:
        Function that computes current learning rate based on progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0 (end).

        Args:
            progress_remaining: Remaining progress (1.0 at start, 0.0 at end)

        Returns:
            Current learning rate
        """
        return progress_remaining * initial_value

    return func


def make_env(
    env_id: str,
    rank: int,
    seed: int = 0,
    hardcore: bool = False,
    use_hardcore_wrapper: bool = False,
    use_bimq_wrapper: bool = False,
    use_natural_walking_wrapper: bool = False,
    use_human_gait_wrapper: bool = False,
    use_simple_knee: bool = False,
    use_smooth_natural: bool = False,
    use_elite_hardcore: bool = False,
    wrapper_kwargs: dict = None,
    bimq_kwargs: dict = None,
    natural_walking_kwargs: dict = None,
    human_gait_kwargs: dict = None,
    simple_knee_kwargs: dict = None,
    smooth_natural_kwargs: dict = None,
    elite_hardcore_kwargs: dict = None,
    **kwargs
):
    """Create a single environment with proper seeding.

    Args:
        env_id: Gym environment ID
        rank: Unique ID for this environment
        seed: Base random seed
        hardcore: Enable hardcore mode (obstacles) for BipedalWalker
        use_hardcore_wrapper: Whether to use hardcore wrapper (frame skip, reward shaping, etc.)
        use_bimq_wrapper: Whether to use BIMQ wrapper (novel movement quality)
        use_natural_walking_wrapper: Whether to use natural walking wrapper (knee bending, speed control)
        use_human_gait_wrapper: Whether to use human gait wrapper (research-based anti-jumping)
        wrapper_kwargs: Keyword arguments for hardcore wrapper
        bimq_kwargs: Keyword arguments for BIMQ wrapper
        natural_walking_kwargs: Keyword arguments for natural walking wrapper
        human_gait_kwargs: Keyword arguments for human gait wrapper
        **kwargs: Additional environment arguments
    """
    def _init():
        # For BipedalWalker, pass hardcore parameter
        if 'BipedalWalker' in env_id:
            env = gym.make(env_id, hardcore=hardcore)
        else:
            env = gym.make(env_id)
        env.reset(seed=seed + rank)

        # Apply wrappers based on config
        if use_human_gait_wrapper:
            # Use Human Gait framework (research-based anti-jumping)
            if human_gait_kwargs is None:
                human_gait_kwargs_local = {}
            else:
                human_gait_kwargs_local = human_gait_kwargs.copy()

            if rank == 0:  # Only log once
                logger.info(f"Applying HumanGaitWrapper with kwargs: {human_gait_kwargs_local}")

            # Import required wrappers
            from hardcore_wrappers import (
                FrameSkipWrapper,
                SmoothActionWrapper,
                HardcoreRewardWrapper
            )

            # Extract parameters
            use_basic_hardcore = human_gait_kwargs_local.pop('use_basic_hardcore', True)
            frame_skip = human_gait_kwargs_local.pop('frame_skip', 4)
            smoothness_coef = human_gait_kwargs_local.pop('smoothness_coef', 0.05)

            # Apply basic hardcore optimizations if requested
            if use_basic_hardcore:
                env = FrameSkipWrapper(env, skip=frame_skip)
                env = SmoothActionWrapper(env, smoothness_coef=smoothness_coef)
                env = HardcoreRewardWrapper(env, clip_min=-10.0, clip_max=10.0)

            # Apply human gait wrapper
            env = HumanGaitWrapper(env, **human_gait_kwargs_local)

        elif use_natural_walking_wrapper:
            # Use Natural Walking framework (knee bending + speed control)
            if natural_walking_kwargs is None:
                natural_walking_kwargs_local = {}
            else:
                natural_walking_kwargs_local = natural_walking_kwargs.copy()

            if rank == 0:  # Only log once
                logger.info(f"Applying NaturalWalkingWrapper with kwargs: {natural_walking_kwargs_local}")

            # Import required wrappers
            from hardcore_wrappers import (
                FrameSkipWrapper,
                SmoothActionWrapper,
                HullStabilityWrapper,
                HardcoreRewardWrapper
            )

            # Extract parameters
            use_basic_hardcore = natural_walking_kwargs_local.pop('use_basic_hardcore', True)
            frame_skip = natural_walking_kwargs_local.pop('frame_skip', 3)
            smoothness_coef = natural_walking_kwargs_local.pop('smoothness_coef', 0.05)
            angle_coef = natural_walking_kwargs_local.pop('angle_coef', 0.02)
            angular_vel_coef = natural_walking_kwargs_local.pop('angular_vel_coef', 0.01)

            # Apply basic hardcore optimizations if requested
            if use_basic_hardcore:
                env = FrameSkipWrapper(env, skip=frame_skip)
                env = SmoothActionWrapper(env, smoothness_coef=smoothness_coef)
                env = HullStabilityWrapper(env, angle_coef=angle_coef, angular_vel_coef=angular_vel_coef)
                env = HardcoreRewardWrapper(env, clip_min=-10.0, clip_max=10.0)

            # Apply natural walking wrapper
            env = NaturalWalkingWrapper(env, **natural_walking_kwargs_local)

        elif use_bimq_wrapper:
            # Use BIMQ framework (novel approach)
            if bimq_kwargs is None:
                bimq_kwargs_local = {}
            else:
                bimq_kwargs_local = bimq_kwargs.copy()

            if rank == 0:  # Only log once
                logger.info(f"Applying BIMQWrapper with kwargs: {bimq_kwargs_local}")

            # BIMQ can optionally use basic hardcore optimizations
            # (frame skip, reward clipping) as a base
            use_hardcore_base = bimq_kwargs_local.pop('use_hardcore_base', True)
            if use_hardcore_base:
                # Apply minimal hardcore wrapper (frame skip + reward clipping only)
                env = HardcoreWrapper(
                    env,
                    frame_skip=bimq_kwargs_local.pop('frame_skip', 4),
                    reward_clip_min=bimq_kwargs_local.pop('reward_clip_min', -10.0),
                    reward_clip_max=bimq_kwargs_local.pop('reward_clip_max', 10.0),
                    # Disable hardcore smoothness/stability (BIMQ provides its own)
                    smoothness_coef=0.0,
                    angle_coef=0.0,
                    angular_vel_coef=0.0
                )

            # Apply BIMQ framework on top
            env = BIMQWrapper(env, **bimq_kwargs_local)

        elif use_simple_knee:
            # Use simple knee bending wrapper (minimal modification)
            if simple_knee_kwargs is None:
                simple_knee_kwargs_local = {}
            else:
                simple_knee_kwargs_local = simple_knee_kwargs.copy()

            if rank == 0:  # Only log once
                logger.info(f"Applying SimpleKneeBendingReward with kwargs: {simple_knee_kwargs_local}")

            from simple_knee_wrapper import SimpleKneeBendingReward
            env = SimpleKneeBendingReward(env, **simple_knee_kwargs_local)

        elif use_smooth_natural:
            # Use smooth natural walking wrapper (action smoothing + velocity limits)
            if smooth_natural_kwargs is None:
                smooth_natural_kwargs_local = {}
            else:
                smooth_natural_kwargs_local = smooth_natural_kwargs.copy()

            if rank == 0:  # Only log once
                logger.info(f"Applying SmoothNaturalWalking with kwargs: {smooth_natural_kwargs_local}")

            from smooth_natural_wrapper import SmoothNaturalWalking
            env = SmoothNaturalWalking(env, **smooth_natural_kwargs_local)

        elif use_elite_hardcore:
            # Use elite hardcore wrapper (unified hardcore + natural walking)
            if elite_hardcore_kwargs is None:
                elite_hardcore_kwargs_local = {}
            else:
                elite_hardcore_kwargs_local = elite_hardcore_kwargs.copy()

            if rank == 0:  # Only log once
                logger.info(f"Applying EliteHardcoreWrapper with kwargs: {elite_hardcore_kwargs_local}")

            env = EliteHardcoreWrapper(env, **elite_hardcore_kwargs_local)

        elif use_hardcore_wrapper:
            # Use standard hardcore wrapper
            if wrapper_kwargs is None:
                wrapper_kwargs_local = {}
            else:
                wrapper_kwargs_local = wrapper_kwargs.copy()

            if rank == 0:  # Only log once
                logger.info(f"Applying HardcoreWrapper with kwargs: {wrapper_kwargs_local}")
            env = HardcoreWrapper(env, **wrapper_kwargs_local)

        env = Monitor(env)
        return env
    return _init


def get_device(requested_device: str) -> str:
    """Validate and get available device with fallback.

    Args:
        requested_device: Requested device (cuda, mps, cpu, or specific like cuda:0)

    Returns:
        Available device string
    """
    if requested_device.startswith('cuda'):
        if torch.cuda.is_available():
            logger.info(f"✓ Using CUDA device: {requested_device}")
            return requested_device
        else:
            logger.warning("CUDA requested but not available. Falling back to CPU.")
            return 'cpu'
    elif requested_device == 'mps':
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            logger.info("✓ Using MPS (Apple Silicon GPU)")
            return 'mps'
        else:
            logger.warning("MPS requested but not available. Falling back to CPU.")
            return 'cpu'
    elif requested_device == 'cpu':
        logger.info("Using CPU")
        return 'cpu'
    else:
        logger.warning(f"Unknown device '{requested_device}'. Falling back to CPU.")
        return 'cpu'


def validate_config(config: Dict) -> None:
    """Validate that required config keys exist.

    Args:
        config: Configuration dictionary

    Raises:
        ValueError: If required keys are missing or invalid
    """
    required_keys = {
        'env': ['name'],
        'agent': ['type', 'learning_rate', 'gamma'],
        'training': ['total_timesteps'],
        'experiment': ['name', 'device', 'seed'],
        'paths': ['checkpoints', 'logs']
    }

    for section, keys in required_keys.items():
        if section not in config:
            raise ValueError(f"Missing required config section: {section}")
        for key in keys:
            if key not in config[section]:
                raise ValueError(f"Missing required key '{key}' in section '{section}'")

    # Validate agent type
    valid_agents = ['sac', 'ppo']
    if config['agent']['type'].lower() not in valid_agents:
        raise ValueError(f"Invalid agent type. Must be one of: {valid_agents}")

    logger.info("✓ Configuration validated successfully")


def load_config(config_path: str) -> Dict:
    """Load and validate configuration from YAML file.

    Args:
        config_path: Path to YAML config file

    Returns:
        Configuration dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid YAML
        ValueError: If config validation fails
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"✓ Loaded configuration from {config_path}")
        validate_config(config)
        return config
    except FileNotFoundError:
        logger.error(f"Config file not found: {config_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Invalid YAML in config file: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        raise


def train_sac_sb3(config: Dict, resume_path: Optional[str] = None):
    """Train SAC using Stable-Baselines3.

    Args:
        config: Configuration dictionary
        resume_path: Optional path to checkpoint to resume from
    """
    logger.info("=" * 60)
    logger.info("STABLE-BASELINES3 SAC TRAINING")
    logger.info("=" * 60)

    # Setup
    env_id = config['env']['name']
    num_envs = config['gpu'].get('num_parallel_envs', 8)
    device = get_device(config['experiment']['device'])
    seed = config['experiment']['seed']
    num_eval_envs = config['env'].get('num_eval_envs', 5)
    hardcore = config['env'].get('hardcore', False)

    # Wrapper settings - either hardcore, BIMQ, natural walking, human gait, simple knee, smooth natural, or elite hardcore
    use_hardcore_wrapper = config['env'].get('use_hardcore_wrapper', False)
    use_bimq_wrapper = config['env'].get('use_bimq_wrapper', False)
    use_natural_walking_wrapper = config['env'].get('use_natural_walking_wrapper', False)
    use_human_gait_wrapper = config['env'].get('use_human_gait_wrapper', False)
    use_simple_knee = config['env'].get('use_simple_knee', False)
    use_smooth_natural = config['env'].get('use_smooth_natural', False)
    use_elite_hardcore = config['env'].get('use_elite_hardcore', False)

    wrapper_kwargs = {}
    bimq_kwargs = {}
    natural_walking_kwargs = {}
    human_gait_kwargs = {}
    simple_knee_kwargs = {}
    smooth_natural_kwargs = {}
    elite_hardcore_kwargs = {}

    if use_human_gait_wrapper:
        # Human gait framework settings (research-based anti-jumping)
        human_gait_kwargs = {
            'use_basic_hardcore': config['env'].get('use_basic_hardcore', True),
            'frame_skip': config['env'].get('frame_skip', 4),
            'smoothness_coef': config['env'].get('smoothness_coef', 0.05),
            'velocity_coef': config['env'].get('velocity_coef', 3.0),
            'target_speed': config['env'].get('target_speed', 1.5),
            'speed_penalty_coef': config['env'].get('speed_penalty_coef', 0.5),
            'height_penalty_coef': config['env'].get('height_penalty_coef', 10.0),
            'airborne_penalty_coef': config['env'].get('airborne_penalty_coef', 20.0),
            'max_consecutive_airborne': config['env'].get('max_consecutive_airborne', 3),
            'single_contact_reward': config['env'].get('single_contact_reward', 1.0),
            'torque_coef': config['env'].get('torque_coef', 0.001),
            'clearance_reward_coef': config['env'].get('clearance_reward_coef', 0.5),
            'drag_penalty_coef': config['env'].get('drag_penalty_coef', 1.0),
            'orientation_coef': config['env'].get('orientation_coef', 2.0),
        }
    elif use_simple_knee:
        # Simple knee bending (minimal modification to baseline)
        simple_knee_kwargs = {
            'knee_bend_reward': config['env'].get('knee_bend_reward', 0.5),
            'straight_penalty': config['env'].get('straight_penalty', 0.3),
            'min_bend_threshold': config['env'].get('min_bend_threshold', 0.3),
        }
    elif use_smooth_natural:
        # Smooth natural walking (knee bending + action smoothing + velocity limits)
        smooth_natural_kwargs = {
            'knee_bend_reward': config['env'].get('knee_bend_reward', 0.02),
            'min_bend_threshold': config['env'].get('min_bend_threshold', 0.3),
            'action_smoothness_penalty': config['env'].get('action_smoothness_penalty', 0.05),
            'max_joint_velocity': config['env'].get('max_joint_velocity', 2.0),
            'velocity_penalty': config['env'].get('velocity_penalty', 0.02),
            'early_steps_stability_bonus': config['env'].get('early_steps_stability_bonus', 0.01),
            'early_steps_count': config['env'].get('early_steps_count', 100),
        }
    elif use_elite_hardcore:
        # Elite hardcore (unified hardcore + natural walking)
        elite_hardcore_kwargs = {
            'frame_skip': config['env'].get('frame_skip', 4),
            'smoothness_coef': config['env'].get('smoothness_coef', 0.2),
            'hull_angle_coef': config['env'].get('hull_angle_coef', 0.1),
            'hull_angular_vel_coef': config['env'].get('hull_angular_vel_coef', 0.05),
            'knee_bend_reward': config['env'].get('knee_bend_reward', 0.02),
            'min_bend_threshold': config['env'].get('min_bend_threshold', 0.3),
            'max_joint_velocity': config['env'].get('max_joint_velocity', 2.0),
            'velocity_penalty': config['env'].get('velocity_penalty', 0.02),
            'early_steps_stability_bonus': config['env'].get('early_steps_stability_bonus', 0.01),
            'early_steps_count': config['env'].get('early_steps_count', 100),
        }
    elif use_natural_walking_wrapper:
        # Natural walking framework settings
        natural_walking_kwargs = {
            'use_basic_hardcore': config['env'].get('use_basic_hardcore', True),
            'frame_skip': config['env'].get('frame_skip', 3),
            'smoothness_coef': config['env'].get('smoothness_coef', 0.05),
            'angle_coef': config['env'].get('hull_angle_coef', 0.02),
            'angular_vel_coef': config['env'].get('hull_angular_vel_coef', 0.01),
            'target_speed': config['env'].get('target_speed', 1.5),
            'speed_penalty_coef': config['env'].get('speed_penalty_coef', 0.3),
            'swing_bend_coef': config['env'].get('swing_bend_coef', 0.15),
            'stance_straight_coef': config['env'].get('stance_straight_coef', 0.1),
            'min_knee_bend': config['env'].get('min_knee_bend', 0.3),
            'max_action_magnitude': config['env'].get('max_action_magnitude', 0.7),
            'alternation_coef': config['env'].get('alternation_coef', 0.1),
        }
    elif use_bimq_wrapper:
        # BIMQ framework settings
        bimq_kwargs = {
            'use_hardcore_base': config['env'].get('use_hardcore_base', True),
            'frame_skip': config['env'].get('frame_skip', 4),
            'reward_clip_min': config['env'].get('reward_clip_min', -10.0),
            'reward_clip_max': config['env'].get('reward_clip_max', 10.0),
            'symmetry_coef': config['env'].get('symmetry_coef', 0.1),
            'periodicity_coef': config['env'].get('periodicity_coef', 0.05),
            'period_length': config['env'].get('period_length', 30),
            'straight_knee_coef': config['env'].get('straight_knee_coef', 0.1),
            'antiphase_coef': config['env'].get('antiphase_coef', 0.05),
        }
    elif use_hardcore_wrapper:
        # Standard hardcore wrapper settings
        wrapper_kwargs = {
            'frame_skip': config['env'].get('frame_skip', 4),
            'smoothness_coef': config['env'].get('smoothness_coef', 0.05),
            'angle_coef': config['env'].get('hull_angle_coef', 0.02),
            'angular_vel_coef': config['env'].get('hull_angular_vel_coef', 0.01),
        }

    logger.info(f"Environment: {env_id}")
    if hardcore:
        logger.info("HARDCORE MODE ENABLED - Training with obstacles!")
    if use_human_gait_wrapper:
        logger.info("*** USING HUMAN GAIT FRAMEWORK (RESEARCH-BASED ANTI-JUMPING) ***")
        logger.info(f"  Forward Velocity Reward: {human_gait_kwargs['velocity_coef']}")
        logger.info(f"  Target Speed: {human_gait_kwargs['target_speed']} m/s (natural walking)")
        logger.info(f"  Speed Penalty: {human_gait_kwargs['speed_penalty_coef']} (prevents running)")
        logger.info(f"  Vertical Displacement Penalty: {human_gait_kwargs['height_penalty_coef']}")
        logger.info(f"  Airborne Penalty: {human_gait_kwargs['airborne_penalty_coef']}")
        logger.info(f"  Single Contact Reward: {human_gait_kwargs['single_contact_reward']}")
        logger.info(f"  Torque Efficiency: {human_gait_kwargs['torque_coef']} (cubed)")
        logger.info(f"  Frame Skip: {human_gait_kwargs['frame_skip']}")
    elif use_natural_walking_wrapper:
        logger.info("*** USING NATURAL WALKING FRAMEWORK ***")
        logger.info(f"  Target Speed: {natural_walking_kwargs['target_speed']}")
        logger.info(f"  Max Action: {natural_walking_kwargs['max_action_magnitude']}")
        logger.info(f"  Knee Bending: swing={natural_walking_kwargs['swing_bend_coef']}, stance={natural_walking_kwargs['stance_straight_coef']}")
        logger.info(f"  Frame Skip: {natural_walking_kwargs['frame_skip']}")
    elif use_bimq_wrapper:
        logger.info("*** USING NOVEL BIMQ FRAMEWORK ***")
        logger.info(f"  Gait Symmetry: {bimq_kwargs['symmetry_coef']}")
        logger.info(f"  Periodicity: {bimq_kwargs['periodicity_coef']}")
        logger.info(f"  Biomech Constraints: knee={bimq_kwargs['straight_knee_coef']}, antiphase={bimq_kwargs['antiphase_coef']}")
    elif use_elite_hardcore:
        logger.info("*** USING ELITE HARDCORE WRAPPER (UNIFIED INTEGRATION) ***")
        logger.info(f"  CORE FEATURES (STRONG):")
        logger.info(f"    Frame Skip: {elite_hardcore_kwargs['frame_skip']}")
        logger.info(f"    L2 Smoothness: {elite_hardcore_kwargs['smoothness_coef']}")
        logger.info(f"    Hull Stability: angle={elite_hardcore_kwargs['hull_angle_coef']}, vel={elite_hardcore_kwargs['hull_angular_vel_coef']}")
        logger.info(f"  AUGMENTATIONS (WEAK):")
        logger.info(f"    Knee Bending: {elite_hardcore_kwargs['knee_bend_reward']}")
        logger.info(f"    Velocity Limits: max={elite_hardcore_kwargs['max_joint_velocity']}, penalty={elite_hardcore_kwargs['velocity_penalty']}")
        logger.info(f"    Early Stability: bonus={elite_hardcore_kwargs['early_steps_stability_bonus']} for {elite_hardcore_kwargs['early_steps_count']} steps")
    elif use_hardcore_wrapper:
        logger.info("Using standard hardcore wrapper")
    logger.info(f"Parallel Envs: {num_envs}")
    logger.info(f"Eval Envs: {num_eval_envs}")
    logger.info(f"Device: {device}")
    logger.info(f"Seed: {seed}")
    if resume_path:
        logger.info(f"Resuming from: {resume_path}")
    logger.info("=" * 60)

    try:
        # Create vectorized environment
        logger.info(f"Creating {num_envs} parallel {'HARDCORE ' if hardcore else ''}environments...")
        env = SubprocVecEnv([make_env(env_id, i, seed, hardcore=hardcore, use_hardcore_wrapper=use_hardcore_wrapper, use_bimq_wrapper=use_bimq_wrapper, use_natural_walking_wrapper=use_natural_walking_wrapper, use_human_gait_wrapper=use_human_gait_wrapper, use_simple_knee=use_simple_knee, use_smooth_natural=use_smooth_natural, use_elite_hardcore=use_elite_hardcore, wrapper_kwargs=wrapper_kwargs, bimq_kwargs=bimq_kwargs, natural_walking_kwargs=natural_walking_kwargs, human_gait_kwargs=human_gait_kwargs, simple_knee_kwargs=simple_knee_kwargs, smooth_natural_kwargs=smooth_natural_kwargs, elite_hardcore_kwargs=elite_hardcore_kwargs) for i in range(num_envs)])
        env = VecMonitor(env)

        # Apply VecNormalize if requested
        if config['env'].get('normalize_observations', False) or config['env'].get('normalize_rewards', False):
            logger.info("Applying VecNormalize for observation/reward normalization")
            env = VecNormalize(
                env,
                norm_obs=config['env'].get('normalize_observations', True),
                norm_reward=config['env'].get('normalize_rewards', True),
                clip_obs=config['env'].get('clip_normalized_obs', 10.0),
                clip_reward=config['env'].get('clip_normalized_reward', 10.0)
            )

        # Create evaluation environment
        logger.info(f"Creating {num_eval_envs} evaluation {'HARDCORE ' if hardcore else ''}environments...")
        eval_env = SubprocVecEnv([make_env(env_id, i, seed + 1000, hardcore=hardcore, use_hardcore_wrapper=use_hardcore_wrapper, use_bimq_wrapper=use_bimq_wrapper, use_natural_walking_wrapper=use_natural_walking_wrapper, use_human_gait_wrapper=use_human_gait_wrapper, use_simple_knee=use_simple_knee, use_smooth_natural=use_smooth_natural, use_elite_hardcore=use_elite_hardcore, wrapper_kwargs=wrapper_kwargs, bimq_kwargs=bimq_kwargs, natural_walking_kwargs=natural_walking_kwargs, human_gait_kwargs=human_gait_kwargs, simple_knee_kwargs=simple_knee_kwargs, smooth_natural_kwargs=smooth_natural_kwargs, elite_hardcore_kwargs=elite_hardcore_kwargs) for i in range(num_eval_envs)])
        eval_env = VecMonitor(eval_env)

        # Apply VecNormalize to eval env if used in training (but don't update stats)
        if isinstance(env, VecNormalize):
            eval_env = VecNormalize(
                eval_env,
                norm_obs=config['env'].get('normalize_observations', True),
                norm_reward=config['env'].get('normalize_rewards', True),
                clip_obs=config['env'].get('clip_normalized_obs', 10.0),
                clip_reward=config['env'].get('clip_normalized_reward', 10.0),
                training=False  # Don't update normalization stats during eval
            )
    except Exception as e:
        logger.error(f"Failed to create environments: {e}")
        raise
    
    # SAC hyperparameters from config
    agent_config = config['agent']

    try:
        # Create or load model
        if resume_path:
            logger.info(f"Loading model from checkpoint: {resume_path}")
            model = SAC.load(resume_path, env=env, device=device)
            logger.info("✓ Model loaded successfully")

            # Sync normalization stats if VecNormalize is used
            if isinstance(env, VecNormalize):
                vec_normalize_path = resume_path.replace('.zip', '_vecnormalize.pkl')
                if os.path.exists(vec_normalize_path):
                    env = VecNormalize.load(vec_normalize_path, env)
                    logger.info("✓ VecNormalize stats loaded")
        else:
            logger.info("Creating new SAC model...")

            # Setup learning rate (with optional linear schedule)
            learning_rate = agent_config['learning_rate']
            if agent_config.get('use_linear_schedule', False):
                logger.info(f"Using linear learning rate schedule: {learning_rate} -> 0")
                learning_rate = linear_schedule(learning_rate)
            else:
                logger.info(f"Using constant learning rate: {learning_rate}")

            model = SAC(
                "MlpPolicy",
                env,
                learning_rate=learning_rate,
                buffer_size=config['buffer']['capacity'],
                learning_starts=config['training']['learning_starts'],
                batch_size=config['buffer']['batch_size'],
                tau=agent_config['tau'],
                gamma=agent_config['gamma'],
                train_freq=config['training'].get('train_frequency', 1),
                gradient_steps=config['training'].get('gradient_steps', 1),
                ent_coef='auto' if agent_config['automatic_entropy_tuning'] else agent_config['alpha'],
                target_entropy='auto' if agent_config.get('target_entropy') is None else agent_config['target_entropy'],
                policy_kwargs=dict(
                    net_arch=agent_config['hidden_dims']
                ),
                verbose=1,
                tensorboard_log=config['paths']['logs'],
                device=device,
                seed=seed
            )
            logger.info("✓ Model created successfully")
    except Exception as e:
        logger.error(f"Failed to create/load model: {e}")
        env.close()
        eval_env.close()
        raise
    
    # Setup callbacks
    checkpoint_dir = os.path.join(config['paths']['checkpoints'], config['experiment']['name'])
    os.makedirs(checkpoint_dir, exist_ok=True)
    logger.info(f"Checkpoint directory: {checkpoint_dir}")

    # Checkpoint callback - save periodically
    # Note: save_freq is divided by num_envs because vectorized envs count steps differently
    checkpoint_callback = CheckpointCallback(
        save_freq=max(config['training']['save_frequency'] // num_envs, 1),
        save_path=checkpoint_dir,
        name_prefix='sac_model',
        save_replay_buffer=True,
        save_vecnormalize=True
    )

    # Evaluation callback - evaluate periodically
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=checkpoint_dir,
        log_path=os.path.join(config['paths']['logs'], config['experiment']['name']),
        eval_freq=max(config['training']['eval_frequency'] // num_envs, 1),
        n_eval_episodes=config['training']['eval_episodes'],
        deterministic=True,
        render=False
    )

    callback_list = CallbackList([checkpoint_callback, eval_callback])

    # Train
    logger.info("Starting training...")
    total_timesteps = config['training']['total_timesteps']
    logger.info(f"Total timesteps: {total_timesteps:,}")

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback_list,
            log_interval=4,
            progress_bar=True
        )
        logger.info("✓ Training completed successfully!")
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    finally:
        # Save final model and cleanup
        try:
            final_path = os.path.join(checkpoint_dir, "final_model.zip")
            model.save(final_path)
            logger.info(f"✓ Final model saved to {final_path}")

            # Save VecNormalize stats if used
            if isinstance(env, VecNormalize):
                vec_normalize_path = os.path.join(checkpoint_dir, "final_model_vecnormalize.pkl")
                env.save(vec_normalize_path)
                logger.info(f"✓ VecNormalize stats saved to {vec_normalize_path}")
        except Exception as e:
            logger.error(f"Error saving final model: {e}")

        # Clean up
        logger.info("Cleaning up environments...")
        env.close()
        eval_env.close()
        logger.info("✓ Cleanup complete")


def train_ppo_sb3(config: Dict, resume_path: Optional[str] = None):
    """Train PPO using Stable-Baselines3.

    Args:
        config: Configuration dictionary
        resume_path: Optional path to checkpoint to resume from
    """
    logger.info("=" * 60)
    logger.info("STABLE-BASELINES3 PPO TRAINING")
    logger.info("=" * 60)

    # Setup
    env_id = config['env']['name']
    num_envs = config['gpu'].get('num_parallel_envs', 8)
    device = get_device(config['experiment']['device'])
    seed = config['experiment']['seed']
    num_eval_envs = config['env'].get('num_eval_envs', 5)
    hardcore = config['env'].get('hardcore', False)

    logger.info(f"Environment: {env_id}")
    if hardcore:
        logger.info("HARDCORE MODE ENABLED - Training with obstacles!")
    logger.info(f"Parallel Envs: {num_envs}")
    logger.info(f"Eval Envs: {num_eval_envs}")
    logger.info(f"Device: {device}")
    logger.info(f"Seed: {seed}")
    if resume_path:
        logger.info(f"Resuming from: {resume_path}")
    logger.info("=" * 60)

    try:
        # Create vectorized environment
        logger.info(f"Creating {num_envs} parallel {'HARDCORE ' if hardcore else ''}environments...")
        if use_hardcore_wrapper and hardcore:
            logger.info(f"Using HardcoreWrapper with settings: {wrapper_kwargs}")
        env = SubprocVecEnv([make_env(env_id, i, seed, hardcore=hardcore, use_hardcore_wrapper=use_hardcore_wrapper, wrapper_kwargs=wrapper_kwargs) for i in range(num_envs)])
        env = VecMonitor(env)

        # Apply VecNormalize if requested
        if config['env'].get('normalize_observations', False) or config['env'].get('normalize_rewards', False):
            logger.info("Applying VecNormalize for observation/reward normalization")
            env = VecNormalize(
                env,
                norm_obs=config['env'].get('normalize_observations', True),
                norm_reward=config['env'].get('normalize_rewards', True),
                clip_obs=config['env'].get('clip_normalized_obs', 10.0),
                clip_reward=config['env'].get('clip_normalized_reward', 10.0)
            )

        # Create evaluation environment
        logger.info(f"Creating {num_eval_envs} evaluation {'HARDCORE ' if hardcore else ''}environments...")
        eval_env = SubprocVecEnv([make_env(env_id, i, seed + 1000, hardcore=hardcore, use_hardcore_wrapper=use_hardcore_wrapper, use_bimq_wrapper=use_bimq_wrapper, use_natural_walking_wrapper=use_natural_walking_wrapper, use_human_gait_wrapper=use_human_gait_wrapper, use_simple_knee=use_simple_knee, use_smooth_natural=use_smooth_natural, use_elite_hardcore=use_elite_hardcore, wrapper_kwargs=wrapper_kwargs, bimq_kwargs=bimq_kwargs, natural_walking_kwargs=natural_walking_kwargs, human_gait_kwargs=human_gait_kwargs, simple_knee_kwargs=simple_knee_kwargs, smooth_natural_kwargs=smooth_natural_kwargs, elite_hardcore_kwargs=elite_hardcore_kwargs) for i in range(num_eval_envs)])
        eval_env = VecMonitor(eval_env)

        # Apply VecNormalize to eval env if used in training
        if isinstance(env, VecNormalize):
            eval_env = VecNormalize(
                eval_env,
                norm_obs=config['env'].get('normalize_observations', True),
                norm_reward=config['env'].get('normalize_rewards', True),
                clip_obs=config['env'].get('clip_normalized_obs', 10.0),
                clip_reward=config['env'].get('clip_normalized_reward', 10.0),
                training=False
            )
    except Exception as e:
        logger.error(f"Failed to create environments: {e}")
        raise
    
    # PPO hyperparameters from config
    agent_config = config['agent']

    try:
        # Create or load model
        if resume_path:
            logger.info(f"Loading model from checkpoint: {resume_path}")
            model = PPO.load(resume_path, env=env, device=device)
            logger.info("✓ Model loaded successfully")

            # Sync normalization stats if VecNormalize is used
            if isinstance(env, VecNormalize):
                vec_normalize_path = resume_path.replace('.zip', '_vecnormalize.pkl')
                if os.path.exists(vec_normalize_path):
                    env = VecNormalize.load(vec_normalize_path, env)
                    logger.info("✓ VecNormalize stats loaded")
        else:
            logger.info("Creating new PPO model...")
            model = PPO(
                "MlpPolicy",
                env,
                learning_rate=agent_config['learning_rate'],
                n_steps=config['training']['rollout_steps'] // num_envs,
                batch_size=agent_config['mini_batch_size'],
                n_epochs=agent_config['ppo_epochs'],
                gamma=agent_config['gamma'],
                gae_lambda=agent_config['gae_lambda'],
                clip_range=agent_config['clip_epsilon'],
                vf_coef=agent_config['value_loss_coef'],
                ent_coef=agent_config['entropy_coef'],
                max_grad_norm=agent_config['max_grad_norm'],
                policy_kwargs=dict(
                    net_arch=dict(pi=agent_config['hidden_dims'], vf=agent_config['hidden_dims'])
                ),
                verbose=1,
                tensorboard_log=config['paths']['logs'],
                device=device,
                seed=seed
            )
            logger.info("✓ Model created successfully")
    except Exception as e:
        logger.error(f"Failed to create/load model: {e}")
        env.close()
        eval_env.close()
        raise

    # Setup callbacks
    checkpoint_dir = os.path.join(config['paths']['checkpoints'], config['experiment']['name'])
    os.makedirs(checkpoint_dir, exist_ok=True)
    logger.info(f"Checkpoint directory: {checkpoint_dir}")

    # Note: Frequencies are divided by num_envs for vectorized environments
    checkpoint_callback = CheckpointCallback(
        save_freq=max(config['training']['save_frequency'] // num_envs, 1),
        save_path=checkpoint_dir,
        name_prefix='ppo_model',
        save_vecnormalize=True
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=checkpoint_dir,
        log_path=os.path.join(config['paths']['logs'], config['experiment']['name']),
        eval_freq=max(config['training']['eval_frequency'] // num_envs, 1),
        n_eval_episodes=config['training']['eval_episodes'],
        deterministic=True,
        render=False
    )

    callback_list = CallbackList([checkpoint_callback, eval_callback])

    # Train
    logger.info("Starting training...")
    total_timesteps = config['training']['total_timesteps']
    logger.info(f"Total timesteps: {total_timesteps:,}")

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback_list,
            log_interval=4,
            progress_bar=True
        )
        logger.info("✓ Training completed successfully!")
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    finally:
        # Save final model and cleanup
        try:
            final_path = os.path.join(checkpoint_dir, "final_model.zip")
            model.save(final_path)
            logger.info(f"✓ Final model saved to {final_path}")

            # Save VecNormalize stats if used
            if isinstance(env, VecNormalize):
                vec_normalize_path = os.path.join(checkpoint_dir, "final_model_vecnormalize.pkl")
                env.save(vec_normalize_path)
                logger.info(f"✓ VecNormalize stats saved to {vec_normalize_path}")
        except Exception as e:
            logger.error(f"Error saving final model: {e}")

        # Clean up
        logger.info("Cleaning up environments...")
        env.close()
        eval_env.close()
        logger.info("✓ Cleanup complete")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Production-ready SB3 training with multi-platform GPU support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train from scratch
  python train_sb3_gpu.py --config configs/sac_sb3_gpu.yaml

  # Resume from checkpoint
  python train_sb3_gpu.py --config configs/sac_sb3_gpu.yaml --resume experiments/checkpoints/sac_sb3_gpu/final_model.zip

  # Override device and num envs
  python train_sb3_gpu.py --config configs/sac_sb3_gpu.yaml --device cpu --num-envs 4
        """
    )
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--num-envs", type=int, default=None, help="Override number of parallel environments")
    parser.add_argument("--device", type=str, default=None, help="Override device (cuda, mps, cpu)")
    args = parser.parse_args()

    try:
        # Load configuration
        config = load_config(args.config)

        # Add GPU-specific config if not present
        if 'gpu' not in config:
            config['gpu'] = {}

        # Override with command line args
        if args.num_envs is not None:
            config['gpu']['num_parallel_envs'] = args.num_envs
            logger.info(f"Overriding num_parallel_envs to {args.num_envs}")
        if args.device is not None:
            config['experiment']['device'] = args.device
            logger.info(f"Overriding device to {args.device}")

        # Set defaults
        config['gpu'].setdefault('num_parallel_envs', 8)

        # Print configuration summary
        logger.info("=" * 60)
        logger.info("STABLE-BASELINES3 TRAINING")
        logger.info("=" * 60)
        logger.info(f"Agent: {config['agent']['type'].upper()}")
        logger.info(f"Config: {args.config}")
        logger.info(f"Total timesteps: {config['training']['total_timesteps']:,}")
        logger.info(f"Device: {config['experiment']['device']}")
        logger.info(f"Parallel envs: {config['gpu']['num_parallel_envs']}")
        logger.info("=" * 60)

        # Verify PyTorch installation
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        if hasattr(torch.backends, 'mps'):
            logger.info(f"MPS available: {torch.backends.mps.is_available()}")

        # Train based on agent type
        agent_type = config['agent']['type'].lower()
        if agent_type == 'sac':
            train_sac_sb3(config, resume_path=args.resume)
        elif agent_type == 'ppo':
            train_ppo_sb3(config, resume_path=args.resume)
        else:
            raise ValueError(f"Unknown agent type: {agent_type}. Must be 'sac' or 'ppo'")

        logger.info("=" * 60)
        logger.info("ALL TRAINING COMPLETE")
        logger.info("=" * 60)

    except KeyboardInterrupt:
        logger.warning("\nTraining interrupted by user. Exiting...")
        sys.exit(0)
    except Exception as e:
        logger.error(f"\nFatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
