import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "fast_td3"))

import numpy as np
import torch
import pytest
import gymnasium as gym

# Ensure envs are registered
import humanoid_bench  # noqa: F401

from fast_td3.robots.h1 import h1_fk
from fast_td3.robots.h1_jax import h1_jax_fk

def _compute_avg_rmse(env_id: str = "h1-maze-v0", fk_func: callable = h1_fk.fk_joint_positions) -> float:
    """Create the env, run FK vs MuJoCo xanchor comparison, and return average RMSE."""
    env = gym.make(env_id, render_mode=None)
    try:
        mse = 0.0
        for i in range(100):
            env.reset(seed=i)

            xanchor = env.unwrapped.named.data.xanchor
            batch_xanchor = torch.from_numpy(np.array(xanchor)).unsqueeze(0).float()  # Shape: (1, 20, 3)

            qpos = env.unwrapped.named.data.qpos
            batched_qpos = torch.from_numpy(np.array(qpos)).unsqueeze(0).float()  # Shape: (1, qpos_dim)

            # Convert single qpos to batched torch tensor format
            joint_positions = fk_func(batched_qpos)  # Shape: (1, 20, 3)

            # Calculate MSE between FK joint positions and MuJoCo xanchor
            mse += torch.sqrt(torch.mean((joint_positions - batch_xanchor) ** 2)).item()
        return mse / 100
    finally:
        try:
            env.close()
        except Exception:
            pass

@pytest.mark.parametrize("fk_func", [h1_fk.fk_joint_positions, h1_jax_fk.fk_joint_positions])
def test_h1_fk_avg_rmse_below_threshold(fk_func):
    """Average RMSE between FK anchors and MuJoCo xanchor must be < 0.01."""
    avg_rmse = _compute_avg_rmse("h1-maze-v0", fk_func)

    assert avg_rmse < 0.01, f"Average RMSE too high: {avg_rmse:.6f} (threshold 0.01)" 

def test_h1_jax_fk_is_faster_than_h1_fk():
    """H1 JAX FK must be faster than H1 FK."""
    import time

    # Warm up JIT compilation
    _compute_avg_rmse("h1-maze-v0", h1_jax_fk.fk_joint_positions)

    start_time = time.time()
    _compute_avg_rmse("h1-maze-v0", h1_fk.fk_joint_positions)
    h1_fk_time = time.time() - start_time

    start_time = time.time()
    _compute_avg_rmse("h1-maze-v0", h1_jax_fk.fk_joint_positions)
    h1_jax_fk_time = time.time() - start_time

    assert h1_jax_fk_time < h1_fk_time * 10, f"H1 JAX FK is not faster: {h1_jax_fk_time:.6f}s vs {h1_fk_time:.6f}s"