"""
EGNN Actor that accepts dict observations directly.

This avoids breaking CUDA graph by not flattening dict observations,
instead extracting features directly from the structured dict.

The dict observation contains all necessary inputs including joint_x (joint coordinates).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from fast_td3.actors.gnn.egnn import EGNN


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
        self.egnn = EGNN(
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

    def dict_to_flat_obs(self, obs_dict: dict) -> torch.Tensor:
        """
        Convert dict observations to flat tensor expected by EGNN.
        
        Args:
            obs_dict: Dictionary with keys:
                - pelvis_position: (batch, 3)
                - pelvis_quaternion: (batch, 4)
                - pelvis_linear_velocity: (batch, 3)
                - pelvis_angular_velocity: (batch, 3)
                - joint_positions: (batch, 19)
                - joint_velocities: (batch, 19)
                
        Returns:
            flat_obs: (batch, 51) tensor in the format expected by EGNN's generate_input:
                obs[:, 0:3] = pelvis_position
                obs[:, 3:7] = pelvis_quaternion
                obs[:, 7:26] = joint_positions
                obs[:, 26:29] = pelvis_linear_velocity
                obs[:, 29:32] = pelvis_angular_velocity
                obs[:, 32:51] = joint_velocities
        """
        # Validate required keys
        required_keys = ['pelvis_position', 'pelvis_quaternion', 'pelvis_linear_velocity',
                        'pelvis_angular_velocity', 'joint_positions', 'joint_velocities']
        missing_keys = [key for key in required_keys if key not in obs_dict]
        if missing_keys:
            raise ValueError(f"Missing required observation keys: {missing_keys}")
        
        batch_size = obs_dict['pelvis_position'].shape[0]
        
        # Pre-allocate tensor (total size: 3 + 4 + 19 + 3 + 3 + 19 = 51)
        # Use fixed indexing to avoid dynamic allocation
        flat_obs = torch.empty(batch_size, 51, device=self.device, dtype=obs_dict['pelvis_position'].dtype)
        
        # Fill in the tensor following EGNN's expected format
        flat_obs[:, 0:3] = obs_dict['pelvis_position']
        flat_obs[:, 3:7] = obs_dict['pelvis_quaternion']
        flat_obs[:, 7:26] = obs_dict['joint_positions']
        flat_obs[:, 26:29] = obs_dict['pelvis_linear_velocity']
        flat_obs[:, 29:32] = obs_dict['pelvis_angular_velocity']
        flat_obs[:, 32:51] = obs_dict['joint_velocities']
        
        return flat_obs

    def forward(self, obs) -> torch.Tensor:
        """
        Forward pass with observations.
        
        Args:
            obs: Dict of normalized observations including 'joint_x'
            
        Returns:
            actions: (batch, num_joints) tensor
        """
        # Obs must be a dict with joint_x inside
        if not isinstance(obs, dict):
            raise ValueError("ActorEGNNDict requires dict observations, got tensor")
        
        if 'joint_x' not in obs:
            raise ValueError("Observation dict must contain 'joint_x' key")
        
        # Extract joint_x from dict
        joint_x = obs['joint_x']
        
        # Convert dict to flat obs for EGNN (excluding joint_x)
        flat_obs = self.dict_to_flat_obs(obs)
        
        # Pass to EGNN
        result = self.egnn(flat_obs, joint_x)
        
        return result

    def explore(
        self, 
        obs,
        dones: torch.Tensor = None, 
        deterministic: bool = False
    ) -> torch.Tensor:
        """
        Exploration policy with observations.
        
        Args:
            obs: Dict of normalized observations including 'joint_x'
            dones: (batch,) boolean tensor indicating done episodes
            deterministic: If True, return deterministic actions without noise
            
        Returns:
            actions: (batch, num_joints) tensor with exploration noise
        """
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
