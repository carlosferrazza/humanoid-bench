[![PyPI](https://img.shields.io/pypi/v/embodied.svg)](https://pypi.python.org/pypi/embodied/#history) &nbsp;
[![Docs](https://readthedocs.org/projects/embodied/badge)](https://embodied.readthedocs.org)

# Embodied

Fast reinforcement learning research.

## Overview

The goal of Embodied is to empower researchers to quickly implement new agents
at scale. Embodied achieves this by specifying an interface both for
environments and agents, allowing users to mix and match agents, envs, and
evaluation protocols. Embodied provides common building blocks that users are
encouraged to fork when more control is needed. The only dependency is Numpy
and agents can be implemented in any framework.

## Packages

```
embodied/
  core/    # Config, logging, checkpointing, simulation, wrappers
  run/     # Evaluation protocols that combine agents and environments
  envs/    # Environment suites such as Gym, Atari, DMC, Crafter
  agents/  # Agent implementations
```

## Agent API

```python
class Agent:
  @configs -> dict of configs
  __init__(obs_space, act_space, step, config)
  dataset(generator) -> generator
  policy(obs, state=None, mode='train') -> act, state
  train(data, state=None) -> state, metrics
  report(data) -> metrics
```

## Env API

```python
class Env:
  __len__() -> int
  @obs_space -> dict of spaces
  @act_space -> dict of spaces
  step(action) -> obs dict
  render() -> array
  close()
```
