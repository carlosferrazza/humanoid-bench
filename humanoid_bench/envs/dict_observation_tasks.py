"""
Dict observation tasks for humanoid_bench.

These tasks return observations as a dictionary instead of a flat vector,
making it easier to use with different network architectures like EGNN.

The observation dict includes:
- pelvis_position: Root position (x, y, z)
- pelvis_quaternion: Root quaternion (w, x, y, z)
- pelvis_linear_velocity: Root linear velocity (vx, vy, vz)
- pelvis_angular_velocity: Root angular velocity (wx, wy, wz)
- joint_positions: Joint angles
- joint_velocities: Joint velocities
- xanchor: Joint anchor coordinates (3D positions)
"""

import numpy as np
from gymnasium.spaces import Box, Dict

from humanoid_bench.envs.basic_locomotion_envs import Walk, Stand, Run


class DictObservationMixin:
    """Mixin class that provides dict-based observations for locomotion tasks."""

    # Base task name for model_path construction (to be overridden)
    base_task_name = None

    @property
    def observation_space(self):
        """Return a Dict observation space with separate feature types."""
        # For H1 robot: dof=26
        # qpos: 7 (free joint) + 19 (body joints) = 26
        # qvel: 6 (free joint) + 19 (body joints) = 25
        robot_dof = self.robot.dof
        
        # Joint positions (excluding free joint: 7 DoF)
        n_joint_pos = robot_dof - 7
        # Joint velocities (excluding free joint: 6 DoF)
        n_joint_vel = robot_dof - 7
        
        # xanchor: number of anchors depends on the robot
        # H1: 20 anchors, G1: varies based on model
        # We compute this dynamically in get_obs, here we use robot_dof as approximation
        n_xanchor = robot_dof - 6  # Typically (dof - 6) anchors
        
        return Dict({
            "pelvis_position": Box(
                low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64
            ),
            "pelvis_quaternion": Box(
                low=-np.inf, high=np.inf, shape=(4,), dtype=np.float64
            ),
            "pelvis_linear_velocity": Box(
                low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64
            ),
            "pelvis_angular_velocity": Box(
                low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64
            ),
            "joint_positions": Box(
                low=-np.inf, high=np.inf, shape=(n_joint_pos,), dtype=np.float64
            ),
            "joint_velocities": Box(
                low=-np.inf, high=np.inf, shape=(n_joint_vel,), dtype=np.float64
            ),
            "xanchor": Box(
                low=-np.inf, high=np.inf, shape=(n_xanchor, 3), dtype=np.float64
            ),
        })

    def get_obs(self) -> dict:
        """Return observations as a dictionary with separate feature types."""
        qpos = self._env.data.qpos.flat.copy()
        qvel = self._env.data.qvel.flat.copy()
        xanchor = self._env.data.xanchor.copy()

        # Extract pelvis state (free joint)
        pelvis_position = qpos[:3]  # x, y, z
        pelvis_quaternion = qpos[3:7]  # w, x, y, z
        
        # Extract pelvis velocity
        pelvis_linear_velocity = qvel[:3]  # vx, vy, vz
        pelvis_angular_velocity = qvel[3:6]  # wx, wy, wz
        
        # Extract joint state (excluding free joint)
        joint_positions = qpos[7:]
        joint_velocities = qvel[6:]
        
        return {
            "pelvis_position": pelvis_position,
            "pelvis_quaternion": pelvis_quaternion,
            "pelvis_linear_velocity": pelvis_linear_velocity,
            "pelvis_angular_velocity": pelvis_angular_velocity,
            "joint_positions": joint_positions,
            "joint_velocities": joint_velocities,
            "xanchor": xanchor,
        }


class StandDict(DictObservationMixin, Stand):
    """Stand task with dict observations."""
    base_task_name = "stand"


class WalkDict(DictObservationMixin, Walk):
    """Walk task with dict observations."""
    base_task_name = "walk"


class RunDict(DictObservationMixin, Run):
    """Run task with dict observations."""
    base_task_name = "run"
