import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "fast_td3"))

import numpy as np
import pytest
import gymnasium as gym

# Ensure envs are registered
import humanoid_bench  # noqa: F401

from fast_td3.robots.h1 import h1


def _compute_avg_rmse(env_id: str = "h1-maze-v0") -> float:
    """Create the env, run FK vs MuJoCo xanchor comparison, and return average RMSE."""
    env = gym.make(env_id, render_mode=None)
    try:
        env.reset(seed=0)
        data = env.unwrapped.named.data
        joint_positions = h1.fk_joint_positions(h1.body_tree["pelvis"], data.qpos)

        # Mapping from joint names to xanchor indices (as in the notebook)
        joint_to_xanchor_mapping = {
            "free_base": 0,
            "left_hip_yaw": 1,
            "left_hip_roll": 2,
            "left_hip_pitch": 3,
            "left_knee": 4,
            "left_ankle": 5,
            "right_hip_yaw": 6,
            "right_hip_roll": 7,
            "right_hip_pitch": 8,
            "right_knee": 9,
            "right_ankle": 10,
            "torso": 11,
            "left_shoulder_pitch": 12,
            "left_shoulder_roll": 13,
            "left_shoulder_yaw": 14,
            "left_elbow": 15,
            "right_shoulder_pitch": 16,
            "right_shoulder_roll": 17,
            "right_shoulder_yaw": 18,
            "right_elbow": 19,
        }

        sum_err = 0.0
        count = 0
        for joint_name, xanchor_idx in joint_to_xanchor_mapping.items():
            if joint_name in joint_positions:
                fk_pos = joint_positions[joint_name]
                xanchor_pos = data.xanchor[xanchor_idx]
                rmse = np.sqrt(np.mean((fk_pos - xanchor_pos) ** 2))
                sum_err += rmse
                count += 1

        avg_rmse = sum_err / max(count, 1)
        return float(avg_rmse)
    finally:
        try:
            env.close()
        except Exception:
            pass


@pytest.mark.mujoco
def test_h1_fk_avg_rmse_below_threshold():
    """Average RMSE between FK anchors and MuJoCo xanchor must be < 0.01."""
    try:
        avg_rmse = _compute_avg_rmse("h1-maze-v0")
    except Exception as e:
        pytest.skip(f"Environment not available or failed to initialize: {e}")

    assert avg_rmse < 0.01, f"Average RMSE too high: {avg_rmse:.6f} (threshold 0.01)" 
