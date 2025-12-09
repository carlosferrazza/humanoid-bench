from __future__ import annotations

import gymnasium as gym

import humanoid_bench
from gymnasium.wrappers import TimeLimit
from fast_td3.environments.subproc_vec_env import SubprocVecEnv
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
        "xanchor",
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

    def _convert_obs_to_tensor(self, observations):
        """Convert dict observations from numpy to torch tensors."""
        if isinstance(observations, dict):
            tensor_obs = {}
            for key in observations:
                tensor_obs[key] = torch.from_numpy(observations[key]).to(
                    device=self.sim_device, dtype=torch.float
                )
            return tensor_obs
        else:
            return torch.from_numpy(observations).to(
                device=self.sim_device, dtype=torch.float
            )

    def _stack_dict_obs(self, obs_list):
        """Stack a list of dict observations into a single dict with batched tensors."""
        if isinstance(obs_list[0], dict):
            stacked = {}
            for key in obs_list[0]:
                stacked[key] = np.stack([obs[key] for obs in obs_list])
            return stacked
        return np.stack(obs_list)

    def reset(self):
        """Reset the environment and return dict observations."""
        observations = self.obs_to_flat(self.envs.reset())
        
        return observations

    def render(self):
        assert (
            self.num_envs == 1
        ), "Currently only supports single environment rendering"
        return self.envs.render()

    def step(self, actions):
        assert isinstance(actions, torch.Tensor)
        actions = actions.cpu().numpy()

        observations, rewards, dones, raw_infos, xanchor = self.envs.step(actions)
        observations = self.obs_to_flat(observations)

        # This will be used for getting 'true' next observations
        infos = dict()
        infos["observations"] = {"raw": {"obs": observations.copy()}}
        truncateds = np.zeros_like(dones)
        for i in range(self.num_envs):
            if raw_infos[i].get("TimeLimit.truncated", False):
                truncateds[i] = True
                infos["observations"]["raw"]["obs"][i] = raw_infos[i][
                    "terminal_observation"
                ]

        observations = torch.from_numpy(observations).to(
            device=self.sim_device, dtype=torch.float
        )
        xanchor = torch.from_numpy(xanchor).to(device=self.sim_device, dtype=torch.float)
        rewards = torch.from_numpy(rewards).to(
            device=self.sim_device, dtype=torch.float
        )
        dones = torch.from_numpy(dones).to(device=self.sim_device)
        truncateds = torch.from_numpy(truncateds).to(device=self.sim_device)
        infos["observations"]["raw"]["obs"] = torch.from_numpy(
            infos["observations"]["raw"]["obs"]
        ).to(device=self.sim_device, dtype=torch.float)
        infos["time_outs"] = truncateds

        return observations, rewards, dones, infos

    def obs_to_flat(self, observations):
        """
        Convert dict observations to a flat vector.
        Useful for MLP-based policies.
        
        Args:
            observations: Dict of observation tensors
            
        Returns:
            Flat observation tensor (excluding joint_x)
        """
        if isinstance(observations, dict):
            flat_parts = []
            for key in self.OBS_KEYS:
                if key != 'joint_x' and key in observations:
                    obs = observations[key]
                    if len(obs.shape) > 2:
                        # Flatten multi-dimensional observations
                        obs = obs.reshape(obs.shape[0], -1)
                    flat_parts.append(obs)
            return torch.cat(flat_parts, dim=-1)
        return observations
