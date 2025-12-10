"""
EGNN Actor that accepts dict observations directly.

This avoids breaking CUDA graph by not flattening dict observations,
instead extracting features directly from the structured dict.

The dict observation contains all necessary inputs including joint_x (joint coordinates).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from fast_td3.actors.gnn.egnn_dict import EGNN_dict
from humanoid_bench.envs.dict_observation_tasks import unflatten_obs

obs_shapes = {
    "pelvis_position": (3,),
    "pelvis_quaternion": (4,),
    "pelvis_linear_velocity": (3,),
    "pelvis_angular_velocity": (3,),
    "joint_positions": (19,),  # For H1 robot - 19 body joints
    "joint_velocities": (19,),  # For H1 robot - 19 body joints
    "joint_x": (19, 3),  # For H1 robot - 20 joint anchors (3D coordinates), not normalized
}

class ActorEGNNDict(nn.Module):
    """
    EGNN Actor that accepts dict observations.
    
    This version avoids dynamic tensor operations (like concatenation for flattening)
    that can break CUDA graphs, by working with dict observations directly.
    The obs dict contains all inputs including joint_x.
    """
    
    def __init__(
        self,
        num_envs: int,
        hidden_dim: int,
        batch_size: int,
        device: torch.device,
        n_layers: int,
        act_fn: str,
        env_name: str,
        robot: str = "h1",
        std_min: float = 0.05,
        std_max: float = 0.8,
        attention: bool = False,
        coords_agg: str = "mean",
        normalize: bool = False,
        tanh: bool = False,
    ):
        super().__init__()
        self.n_envs = num_envs
        self.device = device

        match act_fn:
            case "leaky_relu":
                act_fn = nn.LeakyReLU()
            case "silu":
                act_fn = nn.SiLU()
            case "relu":
                act_fn = nn.ReLU()
            case _:
                raise ValueError(f"Unknown activation function: {act_fn}")

        # EGNN for message passing (same as ActorEGNN)
        self.egnn = EGNN_dict(
            hidden_nf=hidden_dim,
            in_node_nf=2,
            in_edge_nf=0,
            out_node_nf=1,
            batch_size=batch_size,
            device=device,
            act_fn=act_fn,
            n_layers=n_layers,
            robot=robot,
            attention=attention,
            coords_agg=coords_agg,
            normalize=normalize,
            tanh=tanh,
            env_name=env_name,
        )

        # Initialize noise parameters
        noise_scales = (
            torch.rand(num_envs, 1, device=device) * (std_max - std_min) + std_min
        )
        self.register_buffer("noise_scales", noise_scales)
        self.register_buffer("std_min", torch.as_tensor(std_min, device=device))
        self.register_buffer("std_max", torch.as_tensor(std_max, device=device))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.egnn(obs)

    def explore(
        self, 
        obs: torch.Tensor,
        dones: torch.Tensor = None, 
        deterministic: bool = False
    ) -> torch.Tensor:
        # If dones is provided, resample noise for environments that are done
        if dones is not None and dones.sum() > 0:
            # Generate new noise scales for done environments
            new_scales = (
                torch.rand(self.n_envs, 1, device=self.device)
                * (self.std_max - self.std_min)
                + self.std_min
            )

            # Update only the noise scales for environments that are done
            dones_view = dones.view(-1, 1) > 0
            self.noise_scales = torch.where(dones_view, new_scales, self.noise_scales)

        # Get deterministic action
        act = self(obs)
        
        if deterministic:
            return act

        # Add exploration noise
        noise = torch.randn_like(act) * self.noise_scales
        return act + noise
