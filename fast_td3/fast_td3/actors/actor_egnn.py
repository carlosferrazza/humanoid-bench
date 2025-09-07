import torch
import torch.nn as nn
import torch.nn.functional as F

from fast_td3.actors.gnn.egnn import EGNN

class ActorEGNN(nn.Module):
    def __init__(
        self,
        n_obs: int,
        n_act: int, 
        num_envs: int,
        init_scale: float,
        hidden_dim: int,
        batch_size: int,
        device: torch.device,
        n_layers: int,
        act_fn: str,
        robot: str = "h1",
        std_min: float = 0.05,
        std_max: float = 0.8,
        n_node_feat: int = 2,
        n_edge_feat: int = 1,
        attention: bool = False,
        coords_agg: str = "mean",
        normalize: bool = False,
        tanh: bool = False,
    ):
        super().__init__()
        self.n_act = n_act
        self.n_envs = num_envs

        match act_fn:
            case "leaky_relu":
                act_fn = nn.LeakyReLU()
            case "silu":
                act_fn = nn.SiLU()
            case "relu":
                act_fn = nn.ReLU()
            case _:
                raise ValueError(f"Unknown activation function: {act_fn}")

        # EGNN for message passing
        self.egnn = EGNN(
            in_node_nf=n_node_feat,
            hidden_nf=hidden_dim,
            out_node_nf=1,
            in_edge_nf=n_edge_feat,
            batch_size=batch_size,
            device=device,
            act_fn=act_fn,
            n_layers=n_layers,
            robot=robot,
            attention=attention,
            coords_agg=coords_agg,
            normalize=normalize,
            tanh=tanh,
            init_scale=init_scale,
        )

        # Initialize noise parameters
        noise_scales = (
            torch.rand(num_envs, 1, device=device) * (std_max - std_min) + std_min
        )
        self.register_buffer("noise_scales", noise_scales)
        self.register_buffer("std_min", torch.as_tensor(std_min, device=device))
        self.register_buffer("std_max", torch.as_tensor(std_max, device=device))

    def forward(self, obs, xpos) -> torch.Tensor:
        result = self.egnn(obs, xpos)

        return result

    def explore(
        self, obs: torch.Tensor, xpos: torch.Tensor, dones: torch.Tensor = None, deterministic: bool = False
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

        act = self(obs, xpos)
        if deterministic:
            return act

        noise = torch.randn_like(act) * self.noise_scales
        return act + noise

