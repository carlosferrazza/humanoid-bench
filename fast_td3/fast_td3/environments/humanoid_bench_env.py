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


class HumanoidBenchEnv:
    """Wraps HumanoidBench environment to support parallel environments."""

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
        self.asymmetric_obs = False  # For comptatibility with MuJoCo Playground
        self.num_obs = self.envs.observation_space.shape[-1]
        self.num_actions = self.envs.action_space.shape[-1]

    def reset(self):
        """Reset the environment."""
        observations = self.envs.reset()
        observations = torch.from_numpy(observations).to(
            device=self.sim_device, dtype=torch.float
        )
        return observations

    def render(self):
        assert (
            self.num_envs == 1
        ), "Currently only supports single environment rendering"
        return self.envs.render()

    def step(self, actions):
        assert isinstance(actions, torch.Tensor)
        actions = actions.cpu().numpy()

        observations, rewards, dones, raw_infos = self.envs.step(actions)

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
