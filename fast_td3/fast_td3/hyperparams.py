import os
from dataclasses import dataclass
import tyro


@dataclass
class BaseArgs:
    # Default hyperparameters -- specifically for HumanoidBench
    # See MuJoCoPlaygroundArgs for default hyperparameters for MuJoCo Playground
    # See IsaacLabArgs for default hyperparameters for IsaacLab
    env_name: str = "h1hand-stand-v0"
    """the id of the environment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    device_rank: int = 0
    """the rank of the device"""
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    project: str = "FastTD3"
    """the project name"""
    use_wandb: bool = True
    """whether to use wandb"""
    checkpoint_path: str = None
    """the path to the checkpoint file"""
    num_envs: int = 128
    """the number of environments to run in parallel"""
    num_eval_envs: int = 128
    """the number of evaluation environments to run in parallel (only valid for MuJoCo Playground)"""
    total_timesteps: int = 150000
    """total timesteps of the experiments"""
    critic_learning_rate: float = 3e-4
    """the learning rate of the critic"""
    actor_learning_rate: float = 3e-4
    """the learning rate for the actor"""
    buffer_size: int = 1024 * 50
    """the replay memory buffer size"""
    num_steps: int = 1
    """the number of steps to use for the multi-step return"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.1
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 32768
    """the batch size of sample from the replay memory"""
    policy_noise: float = 0.001
    """the scale of policy noise"""
    std_min: float = 0.001
    """the minimum scale of noise"""
    std_max: float = 0.4
    """the maximum scale of noise"""
    learning_starts: int = 10
    """timestep to start learning"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    noise_clip: float = 0.5
    """noise clip parameter of the Target Policy Smoothing Regularization"""
    num_updates: int = 2
    """the number of updates to perform per step"""
    init_scale: float = 0.01
    """the scale of the initial parameters"""
    num_atoms: int = 101
    """the number of atoms"""
    v_min: float = -250.0
    """the minimum value of the support"""
    v_max: float = 250.0
    """the maximum value of the support"""
    critic_hidden_dim: int = 1024
    """the hidden dimension of the critic network"""
    actor_hidden_dim: int = 512
    """the hidden dimension of the actor network"""
    use_cdq: bool = True
    """whether to use Clipped Double Q-learning"""
    measure_burnin: int = 3
    """Number of burn-in iterations for speed measure."""
    eval_interval: int = 5000
    """the interval to evaluate the model"""
    render_interval: int = 5000
    """the interval to render the model"""
    compile: bool = True
    """whether to use torch.compile."""
    obs_normalization: bool = True
    """whether to enable observation normalization"""
    max_grad_norm: float = 0.0
    """the maximum gradient norm"""
    amp: bool = True
    """whether to use amp"""
    amp_dtype: str = "bf16"
    """the dtype of the amp"""
    disable_bootstrap: bool = False
    """Whether to disable bootstrap in the critic learning"""

    use_domain_randomization: bool = False
    """(Playground only) whether to use domain randomization"""
    use_push_randomization: bool = False
    """(Playground only) whether to use push randomization"""
    use_tuned_reward: bool = False
    """(Playground only) Use tuned reward for G1"""
    action_bounds: float = 1.0
    """(IsaacLab only) the bounds of the action space (-action_bounds, action_bounds)"""

    weight_decay: float = 0.1
    """the weight decay of the optimizer"""
    save_interval: int = 5000
    """the interval to save the model"""


def get_args():
    """
    Parse command-line arguments and return the appropriate Args instance based on env_name.
    """
    # First, parse all arguments using the base Args class
    base_args = tyro.cli(BaseArgs)

    # Map environment names to their specific Args classes
    # For tasks not here, default hyperparameters are used
    # See below links for available task list
    # - HumanoidBench (https://arxiv.org/abs/2403.10506)
    # - IsaacLab (https://isaac-sim.github.io/IsaacLab/main/source/overview/environments.html)
    # - MuJoCo Playground (https://arxiv.org/abs/2502.08844)
    env_to_args_class = {
        # HumanoidBench
        # NOTE: These tasks are not full list of HumanoidBench tasks
        "h1hand-reach-v0": H1HandReachArgs,
        "h1hand-balance-simple-v0": H1HandBalanceSimpleArgs,
        "h1hand-balance-hard-v0": H1HandBalanceHardArgs,
        "h1hand-pole-v0": H1HandPoleArgs,
        "h1hand-truck-v0": H1HandTruckArgs,
        "h1hand-maze-v0": H1HandMazeArgs,
        "h1hand-push-v0": H1HandPushArgs,
        "h1hand-basketball-v0": H1HandBasketballArgs,
        "h1hand-window-v0": H1HandWindowArgs,
        "h1hand-package-v0": H1HandPackageArgs,
        "h1hand-truck-v0": H1HandTruckArgs,
        # MuJoCo Playground
        # NOTE: These tasks are not full list of MuJoCo Playground tasks
        "G1JoystickFlatTerrain": G1JoystickFlatTerrainArgs,
        "G1JoystickRoughTerrain": G1JoystickRoughTerrainArgs,
        "T1JoystickFlatTerrain": T1JoystickFlatTerrainArgs,
        "T1JoystickRoughTerrain": T1JoystickRoughTerrainArgs,
        "LeapCubeReorient": LeapCubeReorientArgs,
        "LeapCubeRotateZAxis": LeapCubeRotateZAxisArgs,
        "Go1JoystickFlatTerrain": Go1JoystickFlatTerrainArgs,
        "Go1JoystickRoughTerrain": Go1JoystickRoughTerrainArgs,
        "Go1Getup": Go1GetupArgs,
        "CheetahRun": CheetahRunArgs,  # NOTE: Example config for DeepMind Control Suite
        # IsaacLab
        # NOTE: These tasks are not full list of IsaacLab tasks
        "Isaac-Lift-Cube-Franka-v0": IsaacLiftCubeFrankaArgs,
        "Isaac-Open-Drawer-Franka-v0": IsaacOpenDrawerFrankaArgs,
        "Isaac-Velocity-Flat-H1-v0": IsaacVelocityFlatH1Args,
        "Isaac-Velocity-Flat-G1-v0": IsaacVelocityFlatG1Args,
        "Isaac-Velocity-Rough-H1-v0": IsaacVelocityRoughH1Args,
        "Isaac-Velocity-Rough-G1-v0": IsaacVelocityRoughG1Args,
        "Isaac-Repose-Cube-Allegro-Direct-v0": IsaacReposeCubeAllegroDirectArgs,
        "Isaac-Repose-Cube-Shadow-Direct-v0": IsaacReposeCubeShadowDirectArgs,
    }
    # If the provided env_name has a specific Args class, use it
    if base_args.env_name in env_to_args_class:
        specific_args_class = env_to_args_class[base_args.env_name]
        # Re-parse with the specific class, maintaining any user overrides
        specific_args = tyro.cli(specific_args_class)
        return specific_args

    if base_args.env_name.startswith("h1hand-") or base_args.env_name.startswith("h1-"):
        # HumanoidBench
        specific_args = tyro.cli(HumanoidBenchArgs)
    elif base_args.env_name.startswith("Isaac-"):
        # IsaacLab
        specific_args = tyro.cli(IsaacLabArgs)
    else:
        # MuJoCo Playground
        specific_args = tyro.cli(MuJoCoPlaygroundArgs)
    return specific_args


@dataclass
class HumanoidBenchArgs(BaseArgs):
    # See HumanoidBench (https://arxiv.org/abs/2403.10506) for available task list
    total_timesteps: int = 100000


@dataclass
class H1HandReachArgs(HumanoidBenchArgs):
    env_name: str = "h1hand-reach-v0"
    v_min: float = -2000.0
    v_max: float = 2000.0


@dataclass
class H1HandBalanceSimpleArgs(HumanoidBenchArgs):
    env_name: str = "h1hand-balance-simple-v0"
    total_timesteps: int = 200000


@dataclass
class H1HandBalanceHardArgs(HumanoidBenchArgs):
    env_name: str = "h1hand-balance-hard-v0"
    total_timesteps: int = 1000000


@dataclass
class H1HandPoleArgs(HumanoidBenchArgs):
    env_name: str = "h1hand-pole-v0"
    total_timesteps: int = 150000


@dataclass
class H1HandTruckArgs(HumanoidBenchArgs):
    env_name: str = "h1hand-truck-v0"
    total_timesteps: int = 500000


@dataclass
class H1HandMazeArgs(HumanoidBenchArgs):
    env_name: str = "h1hand-maze-v0"
    v_min: float = -1000.0
    v_max: float = 1000.0


@dataclass
class H1HandPushArgs(HumanoidBenchArgs):
    env_name: str = "h1hand-push-v0"
    v_min: float = -1000.0
    v_max: float = 1000.0
    total_timesteps: int = 1000000


@dataclass
class H1HandBasketballArgs(HumanoidBenchArgs):
    env_name: str = "h1hand-basketball-v0"
    v_min: float = -2000.0
    v_max: float = 2000.0
    total_timesteps: int = 250000


@dataclass
class H1HandWindowArgs(HumanoidBenchArgs):
    env_name: str = "h1hand-window-v0"
    total_timesteps: int = 250000


@dataclass
class H1HandPackageArgs(HumanoidBenchArgs):
    env_name: str = "h1hand-package-v0"
    v_min: float = -10000.0
    v_max: float = 10000.0


@dataclass
class H1HandTruckArgs(HumanoidBenchArgs):
    env_name: str = "h1hand-truck-v0"
    v_min: float = -1000.0
    v_max: float = 1000.0


@dataclass
class MuJoCoPlaygroundArgs(BaseArgs):
    # Default hyperparameters for many of Playground environments
    v_min: float = -10.0
    v_max: float = 10.0
    buffer_size: int = 1024 * 10
    num_envs: int = 1024
    num_eval_envs: int = 1024
    gamma: float = 0.97


@dataclass
class G1JoystickFlatTerrainArgs(MuJoCoPlaygroundArgs):
    env_name: str = "G1JoystickFlatTerrain"
    total_timesteps: int = 100000


@dataclass
class G1JoystickRoughTerrainArgs(MuJoCoPlaygroundArgs):
    env_name: str = "G1JoystickRoughTerrain"
    total_timesteps: int = 100000


@dataclass
class T1JoystickFlatTerrainArgs(MuJoCoPlaygroundArgs):
    env_name: str = "T1JoystickFlatTerrain"
    total_timesteps: int = 100000


@dataclass
class T1JoystickRoughTerrainArgs(MuJoCoPlaygroundArgs):
    env_name: str = "T1JoystickRoughTerrain"
    total_timesteps: int = 100000


@dataclass
class T1LowDofJoystickFlatTerrainArgs(MuJoCoPlaygroundArgs):
    env_name: str = "T1LowDofJoystickFlatTerrain"
    total_timesteps: int = 1000000


@dataclass
class T1LowDofJoystickRoughTerrainArgs(MuJoCoPlaygroundArgs):
    env_name: str = "T1LowDofJoystickRoughTerrain"
    total_timesteps: int = 1000000


@dataclass
class CheetahRunArgs(MuJoCoPlaygroundArgs):
    # NOTE: This config will work for most DMC tasks, though we haven't tested DMC extensively.
    # Future research can consider using LayerNorm as we find it sometimes works better for DMC tasks.
    env_name: str = "CheetahRun"
    num_steps: int = 3
    v_min: float = -500.0
    v_max: float = 500.0
    std_min: float = 0.1
    policy_noise: float = 0.1


@dataclass
class Go1JoystickFlatTerrainArgs(MuJoCoPlaygroundArgs):
    env_name: str = "Go1JoystickFlatTerrain"
    total_timesteps: int = 50000
    std_min: float = 0.2
    std_max: float = 0.8
    policy_noise: float = 0.2
    num_updates: int = 8


@dataclass
class Go1JoystickRoughTerrainArgs(MuJoCoPlaygroundArgs):
    env_name: str = "Go1JoystickRoughTerrain"
    total_timesteps: int = 50000
    std_min: float = 0.2
    std_max: float = 0.8
    policy_noise: float = 0.2
    num_updates: int = 8


@dataclass
class Go1GetupArgs(MuJoCoPlaygroundArgs):
    env_name: str = "Go1Getup"
    total_timesteps: int = 50000
    std_min: float = 0.2
    std_max: float = 0.8
    policy_noise: float = 0.2
    num_updates: int = 8


@dataclass
class LeapCubeReorientArgs(MuJoCoPlaygroundArgs):
    env_name: str = "LeapCubeReorient"
    num_steps: int = 3
    policy_noise: float = 0.2
    v_min: float = -50.0
    v_max: float = 50.0
    use_cdq: bool = False


@dataclass
class LeapCubeRotateZAxisArgs(MuJoCoPlaygroundArgs):
    env_name: str = "LeapCubeRotateZAxis"
    num_steps: int = 1
    policy_noise: float = 0.2
    v_min: float = -10.0
    v_max: float = 10.0
    use_cdq: bool = False


@dataclass
class IsaacLabArgs(BaseArgs):
    v_min: float = -10.0
    v_max: float = 10.0
    buffer_size: int = 1024 * 10
    num_envs: int = 4096
    num_eval_envs: int = 4096
    action_bounds: float = 1.0
    std_max: float = 0.4
    num_atoms: int = 251
    render_interval: int = 0  # IsaacLab does not support rendering in our codebase
    total_timesteps: int = 100000


@dataclass
class IsaacLiftCubeFrankaArgs(IsaacLabArgs):
    # Value learning is unstable for Lift Cube task Due to brittle reward shaping
    # Therefore, we need to disable bootstrap from 'reset_obs' in IsaacLab
    # Higher UTD works better for manipulation tasks
    env_name: str = "Isaac-Lift-Cube-Franka-v0"
    num_updates: int = 8
    v_min: float = -50.0
    v_max: float = 50.0
    std_max: float = 0.8
    num_envs: int = 1024
    num_eval_envs: int = 1024
    action_bounds: float = 3.0
    disable_bootstrap: bool = True
    total_timesteps: int = 20000


@dataclass
class IsaacOpenDrawerFrankaArgs(IsaacLabArgs):
    # Higher UTD works better for manipulation tasks
    env_name: str = "Isaac-Open-Drawer-Franka-v0"
    v_min: float = -50.0
    v_max: float = 50.0
    num_updates: int = 8
    action_bounds: float = 3.0
    total_timesteps: int = 20000


@dataclass
class IsaacVelocityFlatH1Args(IsaacLabArgs):
    env_name: str = "Isaac-Velocity-Flat-H1-v0"
    num_steps: int = 3
    total_timesteps: int = 75000


@dataclass
class IsaacVelocityFlatG1Args(IsaacLabArgs):
    env_name: str = "Isaac-Velocity-Flat-G1-v0"
    num_steps: int = 3
    total_timesteps: int = 50000


@dataclass
class IsaacVelocityRoughH1Args(IsaacLabArgs):
    env_name: str = "Isaac-Velocity-Rough-H1-v0"
    num_steps: int = 3
    buffer_size: int = 1024 * 5  # To reduce memory usage
    total_timesteps: int = 50000


@dataclass
class IsaacVelocityRoughG1Args(IsaacLabArgs):
    env_name: str = "Isaac-Velocity-Rough-G1-v0"
    num_steps: int = 3
    buffer_size: int = 1024 * 5  # To reduce memory usage
    total_timesteps: int = 50000


@dataclass
class IsaacReposeCubeAllegroDirectArgs(IsaacLabArgs):
    env_name: str = "Isaac-Repose-Cube-Allegro-Direct-v0"
    total_timesteps: int = 100000
    v_min: float = -500.0
    v_max: float = 500.0


@dataclass
class IsaacReposeCubeShadowDirectArgs(IsaacLabArgs):
    env_name: str = "Isaac-Repose-Cube-Shadow-Direct-v0"
    total_timesteps: int = 100000
    v_min: float = -500.0
    v_max: float = 500.0
