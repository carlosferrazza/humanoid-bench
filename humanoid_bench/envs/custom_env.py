"""
Dict observation tasks for humanoid_bench.

These tasks return observations as a flat vector (like the original tasks),
but provide an unflatten_obs service to convert to dict format when needed
for normalization and actor processing.

The observation dict (when unflattened) includes:
- pelvis_position: Root position (x, y, z)
- pelvis_quaternion: Root quaternion (w, x, y, z)
- pelvis_linear_velocity: Root linear velocity (vx, vy, vz)
- pelvis_angular_velocity: Root angular velocity (wx, wy, wz)
- joint_positions: Joint angles
- joint_velocities: Joint velocities
- joint_x: Joint anchor coordinates (3D positions)
"""

import torch
import numpy as np
from gymnasium.spaces import Box, Dict

from humanoid_bench.envs.basic_locomotion_envs import Walk, Stand, Run



class ObservationV2:
    """Mixin class that provides flat observations with unflatten service for locomotion tasks."""

    # Base task name for model_path construction (to be overridden)
    base_task_name = None

    @property
    def observation_space(self):
        return Box(
            low=-np.inf,
            high=np.inf,
            shape=(108, ),
            dtype=np.float64,
        )

    def get_obs(self) -> dict:
        """Return observations as a flat vector (matching original implementation)."""
        qpos = self._env.data.qpos.flat.copy()
        qvel = self._env.data.qvel.flat.copy()
        xanchor = self._env.data.xanchor.copy()

        # Extract pelvis state (free joint)
        pelvis_position = np.array([0.0, 0.0, 0.0])  # x, y, z
        pelvis_quaternion = qpos[3:7]  # w, x, y, z

        # Extract pelvis velocity
        pelvis_linear_velocity = qvel[:3]  # vx, vy, vz
        pelvis_angular_velocity = qvel[3:6]  # wx, wy, wz

        # Extract joint state (excluding free joint)
        joint_positions = qpos[7:]
        joint_velocities = qvel[6:]
        joint_x = xanchor[1:, :] - xanchor[0, :]  # relative to pelvis

        # Concatenate into flat vector
        return np.concatenate(
            [
                pelvis_position,
                pelvis_quaternion,
                pelvis_linear_velocity,
                pelvis_angular_velocity,
                joint_velocities,
                joint_positions,
                joint_x.flatten(),
            ]
        )

class Stand(ObservationV2, Stand):
    """Stand task with dict observations."""

    base_task_name = "stand"


class Walk(ObservationV2, Walk):
    """Walk task with dict observations."""


class Run(ObservationV2, Run):
    """Run task with dict observations."""

    base_task_name = "run"
    

def unflatten_obs(flat_obs):
    return {
        "pelvis_position": flat_obs[:, 0:3],
        "pelvis_quaternion": flat_obs[:, 3:7],
        "joint_positions": flat_obs[:, 7:26],
        "pelvis_linear_velocity": flat_obs[:, 26:29],
        "pelvis_angular_velocity": flat_obs[:, 29:32],
        "joint_velocities": flat_obs[:, 32:51],
        "joint_x": flat_obs[:, 51:].reshape(flat_obs.shape[0], -1, 3),
    }

def flatten_obs(obs_dict):
    return torch.cat([
        obs_dict["pelvis_position"],
        obs_dict["pelvis_quaternion"],
        obs_dict["joint_positions"],
        obs_dict["pelvis_linear_velocity"],
        obs_dict["pelvis_angular_velocity"],
        obs_dict["joint_velocities"],
        obs_dict["joint_x"].reshape(obs_dict["joint_x"].shape[0], -1),
    ], axis=-1)