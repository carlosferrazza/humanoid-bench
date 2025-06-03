"""
Fast TD3 is a high-performance implementation of Twin Delayed Deep Deterministic Policy Gradient (TD3)
with distributional critics for reinforcement learning.
"""

# Core model components
from fast_td3.fast_td3 import Actor, Critic, DistributionalQNetwork
from fast_td3.fast_td3_utils import EmpiricalNormalization, SimpleReplayBuffer
from fast_td3.fast_td3_deploy import Policy, load_policy

__all__ = [
    # Core model components
    "Actor",
    "Critic",
    "DistributionalQNetwork",
    "EmpiricalNormalization",
    "SimpleReplayBuffer",
    "Policy",
    "load_policy",
]
