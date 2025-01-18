import os
import torch
import hydra
import pathlib
from omegaconf import OmegaConf
from tdmpc2.common.parser import parse_cfg
from tdmpc2.tdmpc2 import TDMPC2

def get_default_config(policy_path, mean_path, var_path, policy_type, task):
    """Load and parse the default config using Hydra."""
    # Initialize Hydra and load config
    with hydra.initialize(config_path='.', version_base=None):
        cfg = hydra.compose(config_name="config")
    
    # Set required parameters that would normally come from command line
    cfg.task = "humanoid_" + task
    cfg.policy_path = policy_path
    cfg.mean_path = mean_path
    cfg.var_path = var_path
    cfg.policy_type = policy_type
    cfg.steps = 10_000_000
    cfg.obs = "state"
    cfg.obs_shape = {"state": [151]}
    cfg.action_dim = 61
    cfg.episode_length = 500
    cfg.multitask = False
    
    return cfg

def get_agent(policy_path, mean_path, var_path, policy_type, task):
    """Create and return an uninitialized TD-MPC2 agent."""
    cfg = get_default_config(policy_path, mean_path, var_path, policy_type, task)
    return TDMPC2(cfg)

def load_checkpoint(agent, checkpoint_path):
    """Load state dict into agent."""
    assert checkpoint_path is not None, "Checkpoint path must be provided"
    state_dict = torch.load(checkpoint_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    agent.load(state_dict)
    return agent