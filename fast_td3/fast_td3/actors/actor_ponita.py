import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.transforms import RadiusGraph

from fast_td3.actors.ponita.models.ponita import Ponita


class ActorPONITA(nn.Module):
    def __init__(
        self,
        n_obs: int,
        n_act: int, 
        num_envs: int,
        batch_size: int,
        device: torch.device,
        robot: str = "h1",
        std_min: float = 0.05,
        std_max: float = 0.8,
        n_node_feat: int = 3,
        n_edge_feat: int = 1,
        hidden_dim: int = 128, 
        output_dim: int = 1,
        output_dim_vec: int = 1,
        num_layers: int = 4,
        num_ori: int = 10,
        task_level: str = "graph",
        multiple_readouts: bool = True
    ):
        super().__init__()
        self.n_act = n_act
        self.n_envs = num_envs
        self.device = device

        self.ponita = Ponita(
            input_dim=n_node_feat,
            hidden_dim=hidden_dim,
            output_dim=n_act,
            num_layers=num_layers,
            device=device,
            robot=robot,
            batch_size=batch_size,
            num_ori=num_ori,
            task_level=task_level,
            multiple_readouts=multiple_readouts
        )

        # Initialize noise parameters
        noise_scales = (
            torch.rand(num_envs, 1, device=device) * (std_max - std_min) + std_min
        )
        self.register_buffer("noise_scales", noise_scales)
        self.register_buffer("std_min", torch.as_tensor(std_min, device=device))
        self.register_buffer("std_max", torch.as_tensor(std_max, device=device))

    def forward(self, obs, xanchor) -> torch.Tensor:
        h, pos, edge_index, batch = self.ponita.build_batched_ponita_input(obs, xanchor)
        result = self.ponita(h, pos, edge_index, batch)
        return result

    def explore(
        self, obs: torch.Tensor, xanchor: torch.Tensor, dones: torch.Tensor = None, deterministic: bool = False
    ) -> torch.Tensor:
        # If dones is provided, resample noise for environments that are done
        if dones is not None and dones.sum() > 0:
            # Generate new noise scales for done environments (one per environment)
            new_scales = (
                torch.rand(self.n_envs, 1, device=obs.device)
                * (self.std_max - self.std_min)
                + self.std_min
            )

            # Update only the noise scales for environments that are done
            dones_view = dones.view(-1, 1) > 0
            self.noise_scales = torch.where(dones_view, new_scales, self.noise_scales)

        act = self(obs, xanchor)
        if deterministic:
            return act

        noise = torch.randn_like(act) * self.noise_scales
        return act + noise