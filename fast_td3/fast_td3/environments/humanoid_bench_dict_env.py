from __future__ import annotations

import gymnasium as gym

import humanoid_bench
from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.vec_env import SubprocVecEnv
import numpy as np
import torch
from loguru import logger as log

# Disable all logging below CRITICAL level
log.remove()
log.add(lambda msg: False, level="CRITICAL")


def make_env(env_name, rank, render_mode=None, seed=0):
    """
    Utility function for multiprocessed env.

    :param rank: (int) index of the subprocess
    :param seed: (int) the inital seed for RNG
    """

    if env_name in [
        "h1hand-push-v0",
        "h1-push-v0",
        "h1hand-cube-v0",
        "h1cube-v0",
        "h1hand-basketball-v0",
        "h1-basketball-v0",
        "h1hand-kitchen-v0",
        "h1-kitchen-v0",
    ]:
        max_episode_steps = 500
    else:
        max_episode_steps = 1000

    def _init():
        import humanoid_bench

        env = gym.make(env_name, render_mode=render_mode)
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
        env.unwrapped.seed(seed + rank)

        return env

    return _init


class HumanoidBenchDictEnv:
    OBS_KEYS = [
        "pelvis_position",
        "pelvis_quaternion", 
        "pelvis_linear_velocity",
        "pelvis_angular_velocity",
        "joint_positions",
        "joint_velocities",
        "joint_x",
    ]

    def __init__(self, env_name, num_envs=1, render_mode=None, device=None):
        # NOTE: HumanoidBench action space is already normalized to [-1, 1]
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sim_device = device
        self.num_envs = num_envs

        # Create the base environment
        self.envs = SubprocVecEnv(
            [make_env(env_name, i, render_mode=render_mode) for i in range(num_envs)]
        )

        if env_name in [
            "h1hand-push-v0",
            "h1-push-v0",
            "h1hand-cube-v0",
            "h1cube-v0",
            "h1hand-basketball-v0",
            "h1-basketball-v0",
            "h1hand-kitchen-v0",
            "h1-kitchen-v0",
        ]:
            self.max_episode_steps = 500
        else:
            self.max_episode_steps = 1000

        # For compatibility with MuJoCo Playground
        self.asymmetric_obs = False
        
        # Get observation space from the wrapped environment
        self.observation_space = self.envs.observation_space
        self.num_actions = self.envs.action_space.shape[-1]
        
        # Calculate flat observation size for compatibility
        self._calculate_obs_sizes()


    def reset(self):
        """Reset the environment and return dict observations as batched tensors."""
        observations = self.envs.reset()
        return self._merge_obs_array_to_dict(observations)

    def render(self):
        assert (
            self.num_envs == 1
        ), "Currently only supports single environment rendering"
        return self.envs.render()

    def step(self, actions):
        assert isinstance(actions, torch.Tensor)
        actions = actions.cpu().numpy()

        observations, rewards, dones, raw_infos = self.envs.step(actions)
        observations = self._merge_obs_array_to_dict(observations)

        # This will be used for getting 'true' next observations
        infos = dict()
        infos["observations"] = {"raw": {"obs": self._obs_to_flat(observations)}}
        truncateds = np.zeros_like(dones)
        for i in range(self.num_envs):
            if raw_infos[i].get("TimeLimit.truncated", False):
                truncateds[i] = True
                # Convert terminal_observation dict to flat tensor
                terminal_obs_dict = raw_infos[i]["terminal_observation"]
                # Convert numpy arrays to torch tensors
                terminal_obs_tensor_dict = {}
                for key, value in terminal_obs_dict.items():
                    terminal_obs_tensor_dict[key] = torch.from_numpy(value).to(
                        device=self.sim_device, dtype=torch.float
                    )
                # Now flatten this single observation using _obs_to_flat_single
                terminal_obs_flat = self._obs_to_flat_single(terminal_obs_tensor_dict)
                infos["observations"]["raw"]["obs"][i] = terminal_obs_flat

        rewards = torch.from_numpy(rewards).to(
            device=self.sim_device, dtype=torch.float
        )
        dones = torch.from_numpy(dones).to(device=self.sim_device)
        truncateds = torch.from_numpy(truncateds).to(device=self.sim_device)

        infos["time_outs"] = truncateds

        return observations, rewards, dones, infos

    def _obs_to_flat(self, observations):
        return torch.cat([
            observations["pelvis_position"],          # (B, 3)
            observations["pelvis_quaternion"],        # (B, 4)
            observations["joint_positions"],          # (B, 19)
            observations["pelvis_linear_velocity"],   # (B, 3)
            observations["pelvis_angular_velocity"],  # (B, 3)
            observations["joint_velocities"],         # (B, 19)
            observations["joint_x"].reshape(observations["joint_x"].shape[0], -1),  # (B, 19*3)
        ], dim=-1)
    
    def _obs_to_flat_single(self, observation):
        """Flatten a single observation dict (no batch dimension)."""
        return torch.cat([
            observation["pelvis_position"],          # (3,)
            observation["pelvis_quaternion"],        # (4,)
            observation["joint_positions"],          # (19,)
            observation["pelvis_linear_velocity"],   # (3,)
            observation["pelvis_angular_velocity"],  # (3,)
            observation["joint_velocities"],         # (19,)
            observation["joint_x"].reshape(-1),      # (19*3,)
        ], dim=-1)  # (flat_size,)

    
    def _calculate_obs_sizes(self):
        """Calculate observation sizes from the dict observation space."""
        if hasattr(self.observation_space, 'spaces'):
            # Dict observation space
            self.obs_sizes = {}
            total_size = 0
            for key, space in self.observation_space.spaces.items():
                size = np.prod(space.shape)
                self.obs_sizes[key] = space.shape
                total_size += size
            self.num_obs = total_size
        else:
            # Flat observation space (fallback)
            self.num_obs = self.observation_space.shape[-1]
            self.obs_sizes = None

    def _merge_obs_array_to_dict(self, obs_array):
        """
        Merge an array of dict observations into a single dict with batched tensors.
        
        This helper method:
        1. Takes an array/list of dict observations (one per environment)
        2. Stacks them into a single dict with batched numpy arrays
        3. Converts all values to torch tensors on the specified device
        
        Args:
            obs_array: Array or list of dict observations, where each dict contains
                      numpy arrays with observation components (e.g., pelvis_position,
                      joint_positions, etc.)
        
        Returns:
            Dict with same keys as input dicts, but values are torch tensors
            stacked along batch dimension (batch_size, *feature_dims)
        
        Example:
            obs_array = [
                {"pelvis_position": array([...]), "joint_positions": array([...])},
                {"pelvis_position": array([...]), "joint_positions": array([...])},
            ]
            result = self._merge_obs_array_to_dict(obs_array)
            # result = {
            #     "pelvis_position": tensor([[...], [...]], device=self.sim_device),
            #     "joint_positions": tensor([[...], [...]], device=self.sim_device),
            # }
        """
        # Stack all dicts into a single dict with batched numpy arrays
        # batched_dict = {}
        # for key in obs_array[0].keys():
        #     batched_dict[key] = np.stack([obs[key] for obs in obs_array])
        
        # Convert all numpy arrays to torch tensors
        tensor_dict = {}
        for key, value in obs_array.items():
            tensor_dict[key] = torch.from_numpy(value).to(
                device=self.sim_device, dtype=torch.float
            )
        
        return tensor_dict
