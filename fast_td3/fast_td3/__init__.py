"""
Fast TD3 is a high-performance implementation of Twin Delayed Deep Deterministic Policy Gradient (TD3)
with distributional critics for reinforcement learning.
"""

# Core model components
from fast_td3.fast_td3 import Critic, DistributionalQNetwork
from fast_td3.fast_td3_utils import EmpiricalNormalization, SimpleReplayBuffer, SimpleReplayBufferGNN, DictEmpiricalNormalization
from fast_td3.fast_td3_deploy import Policy, load_policy
from fast_td3.actors import Actor, ActorEGNN, ActorEGNNDict

__all__ = [
    # Core model components
    "Actor",
    "ActorEGNN",
    "ActorEGNNDict",
    "Critic",
    "DistributionalQNetwork",
    "EmpiricalNormalization",
    "DictEmpiricalNormalization",
    "EmpiricalNormalization2D",
    "SimpleReplayBuffer",
    "SimpleReplayBufferGNN",
    "Policy",
    "load_policy"
]
