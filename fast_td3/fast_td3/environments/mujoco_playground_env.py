from mujoco_playground import registry
from mujoco_playground import wrapper_torch

import jax
import mujoco

jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)


class PlaygroundEvalEnvWrapper:
    def __init__(self, eval_env, max_episode_steps, env_name, num_eval_envs, seed):
        """
        Wrapper used for evaluation / rendering environments.
        Note that this is different from training environments that are
        wrapped with RSLRLBraxWrapper.
        """
        self.env = eval_env
        self.env_name = env_name
        self.num_envs = num_eval_envs
        self.jit_reset = jax.jit(jax.vmap(self.env.reset))
        self.jit_step = jax.jit(jax.vmap(self.env.step))

        if isinstance(self.env.unwrapped.observation_size, dict):
            self.asymmetric_obs = True
        else:
            self.asymmetric_obs = False

        self.key = jax.random.PRNGKey(seed)
        self.key_reset = jax.random.split(self.key, num_eval_envs)
        self.max_episode_steps = max_episode_steps

    def reset(self):
        self.state = self.jit_reset(self.key_reset)
        if self.asymmetric_obs:
            obs = wrapper_torch._jax_to_torch(self.state.obs["state"])
        else:
            obs = wrapper_torch._jax_to_torch(self.state.obs)
        return obs

    def step(self, actions):
        self.state = self.jit_step(self.state, wrapper_torch._torch_to_jax(actions))
        if self.asymmetric_obs:
            next_obs = wrapper_torch._jax_to_torch(self.state.obs["state"])
        else:
            next_obs = wrapper_torch._jax_to_torch(self.state.obs)
        rewards = wrapper_torch._jax_to_torch(self.state.reward)
        dones = wrapper_torch._jax_to_torch(self.state.done)
        return next_obs, rewards, dones, None

    def render_trajectory(self, trajectory):
        scene_option = mujoco.MjvOption()
        scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
        scene_option.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = False
        scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False

        frames = self.env.render(
            trajectory,
            camera="track" if "Joystick" in self.env_name else None,
            height=480,
            width=640,
            scene_option=scene_option,
        )
        return frames


def make_env(
    env_name,
    seed,
    num_envs,
    num_eval_envs,
    device_rank,
    use_tuned_reward=False,
    use_domain_randomization=False,
    use_push_randomization=False,
):
    # Make training environment
    train_env_cfg = registry.get_default_config(env_name)
    is_humanoid_task = env_name in [
        "G1JoystickRoughTerrain",
        "G1JoystickFlatTerrain",
        "T1JoystickRoughTerrain",
        "T1JoystickFlatTerrain",
    ]

    if use_tuned_reward and is_humanoid_task:
        # NOTE: Tuned reward for G1. Used for producing Figure 7 in the paper.
        # Somehow it works reasonably for T1 as well.
        # However, see `sim2real.md` for sim-to-real RL with Booster T1
        train_env_cfg.reward_config.scales.energy = -5e-5
        train_env_cfg.reward_config.scales.action_rate = -1e-1
        train_env_cfg.reward_config.scales.torques = -1e-3
        train_env_cfg.reward_config.scales.pose = -1.0
        train_env_cfg.reward_config.scales.tracking_ang_vel = 1.25
        train_env_cfg.reward_config.scales.tracking_lin_vel = 1.25
        train_env_cfg.reward_config.scales.feet_phase = 1.0
        train_env_cfg.reward_config.scales.ang_vel_xy = -0.3
        train_env_cfg.reward_config.scales.orientation = -5.0

    if is_humanoid_task and not use_push_randomization:
        train_env_cfg.push_config.enable = False
        train_env_cfg.push_config.magnitude_range = [0.0, 0.0]
    randomizer = (
        registry.get_domain_randomizer(env_name) if use_domain_randomization else None
    )
    raw_env = registry.load(env_name, config=train_env_cfg)
    train_env = wrapper_torch.RSLRLBraxWrapper(
        raw_env,
        num_envs,
        seed,
        train_env_cfg.episode_length,
        train_env_cfg.action_repeat,
        randomization_fn=randomizer,
        device_rank=device_rank,
    )

    # Make evaluation environment
    eval_env_cfg = registry.get_default_config(env_name)
    if is_humanoid_task and not use_push_randomization:
        eval_env_cfg.push_config.enable = False
        eval_env_cfg.push_config.magnitude_range = [0.0, 0.0]
    eval_env = registry.load(env_name, config=eval_env_cfg)
    eval_env = PlaygroundEvalEnvWrapper(
        eval_env, eval_env_cfg.episode_length, env_name, num_eval_envs, seed
    )

    render_env_cfg = registry.get_default_config(env_name)
    if is_humanoid_task and not use_push_randomization:
        render_env_cfg.push_config.enable = False
        render_env_cfg.push_config.magnitude_range = [0.0, 0.0]
    render_env = registry.load(env_name, config=render_env_cfg)
    render_env = PlaygroundEvalEnvWrapper(
        render_env, render_env_cfg.episode_length, env_name, 1, seed
    )

    return train_env, eval_env, render_env
