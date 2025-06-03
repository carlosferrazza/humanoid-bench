from typing import Optional

import gymnasium as gym
import torch
from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import isaaclab_tasks
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


class IsaacLabEnv:
    """Wrapper for IsaacLab environments to be compatible with MuJoCo Playground"""

    def __init__(
        self,
        task_name: str,
        device: str,
        num_envs: int,
        seed: int,
        action_bounds: Optional[float] = None,
    ):
        env_cfg = parse_env_cfg(
            task_name,
            device=device,
            num_envs=num_envs,
        )
        env_cfg.seed = seed
        self.seed = seed
        self.envs = gym.make(task_name, cfg=env_cfg, render_mode=None)

        self.num_envs = self.envs.unwrapped.num_envs
        self.max_episode_steps = self.envs.unwrapped.max_episode_length
        self.action_bounds = action_bounds
        self.num_obs = self.envs.unwrapped.single_observation_space["policy"].shape[0]
        self.asymmetric_obs = "critic" in self.envs.unwrapped.single_observation_space
        if self.asymmetric_obs:
            self.num_privileged_obs = self.envs.unwrapped.single_observation_space[
                "critic"
            ].shape[0]
        else:
            self.num_privileged_obs = 0
        self.num_actions = self.envs.unwrapped.single_action_space.shape[0]

    def reset(self, random_start_init: bool = True) -> torch.Tensor:
        obs_dict, _ = self.envs.reset()
        # NOTE: decorrelate episode horizons like RSL‑RL
        if random_start_init:
            self.envs.unwrapped.episode_length_buf = torch.randint_like(
                self.envs.unwrapped.episode_length_buf, high=int(self.max_episode_steps)
            )
        return obs_dict["policy"]

    def reset_with_critic_obs(self) -> tuple[torch.Tensor, torch.Tensor]:
        obs_dict, _ = self.envs.reset()
        return obs_dict["policy"], obs_dict["critic"]

    def step(
        self, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        if self.action_bounds is not None:
            actions = torch.clamp(actions, -1.0, 1.0) * self.action_bounds
        obs_dict, rew, terminations, truncations, infos = self.envs.step(actions)
        dones = (terminations | truncations).to(dtype=torch.long)
        obs = obs_dict["policy"]
        critic_obs = obs_dict["critic"] if self.asymmetric_obs else None
        info_ret = {"time_outs": truncations, "observations": {"critic": critic_obs}}
        # NOTE: There's really no way to get the raw observations from IsaacLab
        # We just use the 'reset_obs' as next_obs, unfortunately.
        # See https://github.com/isaac-sim/IsaacLab/issues/1362
        info_ret["observations"]["raw"] = {
            "obs": obs,
            "critic_obs": critic_obs,
        }
        return obs, rew, dones, info_ret

    def render(self):
        raise NotImplementedError(
            "We don't support rendering for IsaacLab environments"
        )
