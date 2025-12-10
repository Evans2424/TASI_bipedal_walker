"""Vectorized environment wrapper for parallel training."""

import numpy as np
import gymnasium as gym
from typing import List, Tuple, Optional
from multiprocessing import Process, Pipe
import cloudpickle


def worker(remote, parent_remote, env_fn):
    """Worker process for parallel environment execution.
    
    Args:
        remote: Child end of the pipe
        parent_remote: Parent end of the pipe
        env_fn: Function that creates an environment
    """
    parent_remote.close()
    env = env_fn()
    
    while True:
        try:
            cmd, data = remote.recv()
            
            if cmd == 'step':
                obs, reward, terminated, truncated, info = env.step(data)
                done = terminated or truncated
                if done:
                    # Auto-reset on episode end
                    obs, _ = env.reset()
                remote.send((obs, reward, terminated, truncated, info))
                
            elif cmd == 'reset':
                obs, info = env.reset(seed=data)
                remote.send((obs, info))
                
            elif cmd == 'close':
                env.close()
                remote.close()
                break
                
            elif cmd == 'get_spaces':
                remote.send((env.observation_space, env.action_space))
                
            else:
                raise NotImplementedError(f"Unknown command: {cmd}")
                
        except EOFError:
            break


class VectorizedEnv:
    """Vectorized environment for parallel rollouts.
    
    Runs multiple environments in parallel using multiprocessing.
    Significantly speeds up data collection for GPU training.
    """
    
    def __init__(self, env_fns: List[callable], context: str = 'spawn'):
        """Initialize vectorized environment.
        
        Args:
            env_fns: List of functions that create environments
            context: Multiprocessing context ('spawn' or 'fork')
        """
        self.num_envs = len(env_fns)
        self.closed = False
        
        # Create pipes for communication
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(self.num_envs)])
        
        # Start worker processes
        self.processes = []
        for work_remote, remote, env_fn in zip(self.work_remotes, self.remotes, env_fns):
            args = (work_remote, remote, cloudpickle.dumps(env_fn))
            process = Process(target=self._worker_process, args=args, daemon=True)
            process.start()
            self.processes.append(process)
            work_remote.close()
        
        # Get spaces from first environment
        self.remotes[0].send(('get_spaces', None))
        self.observation_space, self.action_space = self.remotes[0].recv()
    
    @staticmethod
    def _worker_process(work_remote, remote, env_fn_serialized):
        """Wrapper for worker that deserializes the environment function."""
        env_fn = cloudpickle.loads(env_fn_serialized)
        worker(work_remote, remote, env_fn)
    
    def reset(self, seeds: Optional[List[int]] = None):
        """Reset all environments.
        
        Args:
            seeds: Optional list of seeds for each environment
            
        Returns:
            observations: Array of shape (num_envs, obs_dim)
            infos: List of info dicts
        """
        if seeds is None:
            seeds = [None] * self.num_envs
            
        for remote, seed in zip(self.remotes, seeds):
            remote.send(('reset', seed))
        
        results = [remote.recv() for remote in self.remotes]
        observations, infos = zip(*results)
        
        return np.stack(observations), list(infos)
    
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List]:
        """Step all environments.
        
        Args:
            actions: Array of shape (num_envs, action_dim)
            
        Returns:
            observations: Array of shape (num_envs, obs_dim)
            rewards: Array of shape (num_envs,)
            terminated: Array of shape (num_envs,)
            truncated: Array of shape (num_envs,)
            infos: List of info dicts
        """
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        
        results = [remote.recv() for remote in self.remotes]
        observations, rewards, terminated, truncated, infos = zip(*results)
        
        return (
            np.stack(observations),
            np.array(rewards),
            np.array(terminated),
            np.array(truncated),
            list(infos)
        )
    
    def close(self):
        """Close all environments."""
        if self.closed:
            return
        
        for remote in self.remotes:
            remote.send(('close', None))
        
        for process in self.processes:
            process.join()
        
        self.closed = True
    
    def __del__(self):
        """Cleanup on deletion."""
        if not self.closed:
            self.close()


def make_vectorized_env(env_id: str, num_envs: int, seed: int = 0, **kwargs) -> VectorizedEnv:
    """Create a vectorized environment.
    
    Args:
        env_id: Gym environment ID
        num_envs: Number of parallel environments
        seed: Base random seed
        **kwargs: Additional arguments for environment creation
        
    Returns:
        Vectorized environment
    """
    from .env_wrapper import make_env
    
    def make_env_fn(rank: int):
        def _init():
            return make_env(env_id, seed=seed + rank, **kwargs)
        return _init
    
    env_fns = [make_env_fn(i) for i in range(num_envs)]
    return VectorizedEnv(env_fns)
