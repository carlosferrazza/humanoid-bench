import jax
import jax.numpy as jnp
import torch
import numpy as np
from typing import Dict, List, Tuple, Union
from collections import defaultdict
import jax.dlpack as jdlpack
import torch._dynamo

class FastH1FKJAX:
    def __init__(self, body_tree, joint_indices, joint_axes, joint_names_ordered):
        """
        Pre-flatten body_tree into arrays for fast iterative FK using JAX.
        """
        self.body_tree = body_tree
        self.joint_indices = joint_indices
        self.joint_axes = joint_axes
        self.joint_names_ordered = joint_names_ordered

        (
            self.joint_names,
            self.parent,
            self.pos_offset,
            self.axis,
            self.joint_type,
        ) = self._flatten_tree(body_tree)

        self.name_to_index = {n: i for i, n in enumerate(self.joint_names)}
        self.output_index = {n: i for i, n in enumerate(joint_names_ordered)}

    def _flatten_tree(self, body_tree):
        joint_names = []
        parent = []
        pos_offset = []
        axis = []
        joint_type = []

        def recurse(node, parent_idx):
            jname = node.get("joint", node.get("name"))
            idx = len(joint_names)

            joint_names.append(jname)
            parent.append(parent_idx)
            
            # Convert torch tensors to numpy for JAX
            pos = node.get("pos", jnp.zeros(3))
            if hasattr(pos, 'cpu'):  # torch tensor
                pos = pos.cpu().numpy()
            pos_offset.append(pos)
            
            if jname == "free_base":
                joint_type.append("free")
                axis.append(jnp.zeros(3))
            elif jname in self.joint_axes:
                joint_type.append("revolute")
                ax = self.joint_axes[jname]
                if hasattr(ax, 'cpu'):  # torch tensor
                    ax = ax.cpu().numpy()
                axis.append(ax)
            else:
                joint_type.append("fixed")
                axis.append(jnp.zeros(3))

            for child in node.get("children", []):
                recurse(child, idx)

        root = list(body_tree.values())[0]
        recurse(root, -1)

        parent = jnp.array(parent, dtype=jnp.int32)
        pos_offset = jnp.stack(pos_offset)
        axis = jnp.stack(axis)

        return joint_names, parent, pos_offset, axis, joint_type

    def _quat_to_matrix(self, quat):
        """
        quat: [B,4] [w,x,y,z]
        return: [B,3,3]
        """
        quat = quat / (jnp.linalg.norm(quat, axis=1, keepdims=True) + 1e-9)
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        B = quat.shape[0]

        R = jnp.zeros((B, 3, 3), dtype=quat.dtype)
        R = R.at[:, 0, 0].set(1 - 2 * (y*y + z*z))
        R = R.at[:, 0, 1].set(2 * (x*y - z*w))
        R = R.at[:, 0, 2].set(2 * (x*z + y*w))
        R = R.at[:, 1, 0].set(2 * (x*y + z*w))
        R = R.at[:, 1, 1].set(1 - 2 * (x*x + z*z))
        R = R.at[:, 1, 2].set(2 * (y*z - x*w))
        R = R.at[:, 2, 0].set(2 * (x*z - y*w))
        R = R.at[:, 2, 1].set(2 * (y*z + x*w))
        R = R.at[:, 2, 2].set(1 - 2 * (x*x + y*y))
        return R

    def _axis_angle_to_matrix(self, axis, angle):
        """
        axis: [3], angle: [B]
        return: [B,3,3]
        """
        axis = axis / (jnp.linalg.norm(axis) + 1e-9)
        x, y, z = axis
        cos, sin = jnp.cos(angle), jnp.sin(angle)
        B = angle.shape[0]
        R = jnp.zeros((B, 3, 3), dtype=angle.dtype)

        R = R.at[:, 0, 0].set(cos + x*x*(1-cos))
        R = R.at[:, 0, 1].set(x*y*(1-cos) - z*sin)
        R = R.at[:, 0, 2].set(x*z*(1-cos) + y*sin)
        R = R.at[:, 1, 0].set(y*x*(1-cos) + z*sin)
        R = R.at[:, 1, 1].set(cos + y*y*(1-cos))
        R = R.at[:, 1, 2].set(y*z*(1-cos) - x*sin)
        R = R.at[:, 2, 0].set(z*x*(1-cos) - y*sin)
        R = R.at[:, 2, 1].set(z*y*(1-cos) + x*sin)
        R = R.at[:, 2, 2].set(cos + z*z*(1-cos))
        return R

    def _fk_joint_positions_jax(self, qpos):
        """
        JAX implementation of FK joint positions computation.
        qpos: [B, qpos_dim]
        return: [B, num_joints, 3] (aligned with joint_names_ordered)
        """
        B = qpos.shape[0]
        J = len(self.joint_names)

        joint_pos = jnp.zeros((B, J, 3), dtype=qpos.dtype)
        joint_rot = jnp.tile(jnp.eye(3)[None, None, :, :], (B, J, 1, 1))

        def process_joint(j, joint_pos, joint_rot):
            jname = self.joint_names[j]
            pj = self.parent[j]

            if self.joint_type[j] == "free":
                base = qpos[:, self.joint_indices[jname]]  # [B,7]
                pos, quat = base[:, :3], base[:, 3:7]
                R = self._quat_to_matrix(quat)
                joint_pos = joint_pos.at[:, j, :].set(pos)
                joint_rot = joint_rot.at[:, j, :, :].set(R)
            else:
                R_parent = joint_rot[:, pj]
                p_parent = joint_pos[:, pj]
                offset = self.pos_offset[j]
                p_local = (R_parent @ offset) + p_parent
                joint_pos = joint_pos.at[:, j, :].set(p_local)
                R_joint = R_parent
                if self.joint_type[j] == "revolute":
                    angle = qpos[:, self.joint_indices[jname]]
                    R_rel = self._axis_angle_to_matrix(self.axis[j], angle)
                    R_joint = R_parent @ R_rel
                joint_rot = joint_rot.at[:, j, :, :].set(R_joint)
            
            return joint_pos, joint_rot

        # Process joints sequentially (JAX doesn't support dynamic loops well)
        for j in range(J):
            joint_pos, joint_rot = process_joint(j, joint_pos, joint_rot)

        # Reorder into [B, ordered_J, 3]
        num_out = len(self.joint_names_ordered)
        out = jnp.zeros((B, num_out, 3), dtype=qpos.dtype)
        
        for jname, j_out in self.output_index.items():
            if jname in self.name_to_index:
                j_in = self.name_to_index[jname]
                out = out.at[:, j_out, :].set(joint_pos[:, j_in, :])

        return out

    @torch._dynamo.disable
    def fk_joint_positions(self, qpos: torch.Tensor):
        """
        Compute batched FK joint positions (ordered).
        qpos: [B, qpos_dim] (torch.Tensor)
        return: [B, num_joints, 3] (torch.Tensor) (aligned with joint_names_ordered)
        """
        # Convert torch tensor to JAX array
        # Move to CPU first, then convert to numpy, then to JAX
        qpos_jax = jdlpack.from_dlpack(torch.utils.dlpack.to_dlpack(qpos.contiguous()))

        # Compute FK with JAX
        result_jax = self._fk_joint_positions_jax(qpos_jax)
        
        # Convert back to torch tensor
        result_numpy = np.array(result_jax)
        result_torch = torch.from_numpy(result_numpy)
        
        # Move to the same device as input
        if isinstance(qpos, torch.Tensor) and qpos.is_cuda:
            result_torch = result_torch.cuda()
        
        return result_torch

    def get_joint_index_mapping(self):
        return self.output_index

    def compile_fk(self):
        """
        JIT compile the FK function for maximum speed.
        """
        self._fk_joint_positions_jax = jax.jit(self._fk_joint_positions_jax)
        return self


# Import the H1 class from the original file
from .h1 import H1

# Create JAX version of the FK instance
h1_jax_fk = FastH1FKJAX(
    H1.body_tree,
    H1.joint_indices,
    H1.joint_axes,
    H1.joint_names_ordered,
).compile_fk()
