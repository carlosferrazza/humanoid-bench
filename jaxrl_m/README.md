# A JAX Backbone for RL projects

[![DOI](https://zenodo.org/badge/552666099.svg)](https://zenodo.org/badge/latestdoi/552666099)

This project serves as a "central backbone" for an RL codebase, designed to accelerate prototyping and diagnosis of **new** algorithms (although it auxiliarily does contain reference implementations of SAC, CQL, IQL, BC). It is inspired greatly by Ilya Kostrikov's  [JaxRL](https://github.com/ikostrikov/jaxrl) codebase. 

The primary goal of the codebase is to make ease of coding up a new algorithm: towards this goal, the primary philosophy is that 

> algorithms should be single-file implementations

This means that (almost) all components of the algorithm (from update rule to network choices to hyperparameter choices) are all contained in one file (e.g. see [BC example](examples/vision_agents/continuous_bc.py) or [SAC example](examples/mujoco/sac.py)). This makes it easy to read and understand the algorithm, and also makes it easy to modify the algorithm to test out new ideas. The code is also designed to scale as easily as possible to multi-GPU / TPU setups, with simple abstractions for distributed training.


## Installation

Requires `jax`, `flax`, `optax`, `distrax`, and optionally `wandb` for logging. Clone this repository and install it (e.g. `pip install -e .`) or add to python path.

## Usage

The fastest way to understand how to use this skeleton is to see the reference SAC implementation: 

[Agent: sac.py](examples/mujoco/sac.py)

[Launcher: run_mujoco_sac.py](examples/mujoco/run_mujoco_sac.py)


### Structure

The code contains the following files:

- [jaxrl_m.common](jaxrl_m/common.py): Contains the TrainState abstraction (a fork of Flax's TrainState class with some additional syntactic features for ease of use), and some other useful general utilities (`target_update`, `shard_batch`)
- [jaxrl_m.dataset](jaxrl_m/dataset.py): Contains the Dataset class (which can store and sample from buffers containing arbitrarily nested dictionaries) and an equivalent ReplayBuffer class
- [jaxrl_m.networks](jaxrl_m/networks.py): Contains implementations of common RL networks (MLP, Critic, ValueCritic, Policy)
- [jaxrl_m.evaluation](jaxrl_m/evaluation.py): Contains code for running evaluation episodes of agents (e.g. with the `evaluate(policy, env)` function)
- [jaxrl_m.wandb](jaxrl_m/wandb.py): Contains code for easily setting up Weights & Biases for experiments
- [jaxrl_m.typing](jaxrl_m/typing.py): Useful type aliases
- [jaxrl_m.vision](jaxrl_m/vision/__init__.py): `vision.models` contains common vision models (e.g. ResNet, ResNetV2, Impala), `vision.data_augmentations` contains common augmentations (e.g. random crop, random color jitter, gaussian blur)

### Examples

Example implementations:

1. [SAC](examples/mujoco/sac.py)
2. [IQL](examples/mujoco/iql.py)
3. [Continuous BC](examples/vision_agents/continuous_bc.py)
4. [Discrete BC](examples/vision_agents/discrete_bc.py)


Example Launchers:

1. [Mujoco SAC](examples/mujoco/run_mujoco_sac.py)
2. [D4RL IQL](examples/mujoco/run_d4rl_iql.py)


### Citation

If you use this codebase in an academic work, please cite

```
@software{jaxrl_minimal,
  author       = {Dibya Ghosh},
  title        = {dibyaghosh/jaxrl\_m},
  month        = April,
  year         = 2023,
  publisher    = {Zenodo},
  version      = {v0.1},
  doi          = {10.5281/zenodo.7958265},
  url          = {https://github.com/dibyaghosh/jaxrl_m}
}
```
