import torch
import torch.nn as nn
import torch.nn.functional as F

from fast_td3.actors.gnn.hepi import HEPi


class ActorHEPI(nn.Module):
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
        n_node_feat: int = 2,
        n_edge_feat: int = 0,
        hidden_dim: int = 128, 
        latent_dim: int = 16,
        num_message: int = 2,
        node_encoder_layers: int = 1, # 2
        edge_encoder_layers: int = 1,
        node_decoder_layers: int = 1,
    ):
        super().__init__()
        self.n_act = n_act
        self.n_envs = num_envs

        self.hepi = HEPi(
            input_dim_node=n_node_feat,
            input_dim_edge=1,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            output_dim=1,
            num_messages=num_message,
            node_encoder_layers=node_encoder_layers,
            edge_encoder_layers=edge_encoder_layers,
            node_decoder_layers=node_decoder_layers,
            shared_processor=False,
            device=device,
            robot=robot,
            batch_size=batch_size,
            node_type_mapping={"x": 0},
            edge_type_mapping={"x_h": 0},
            edge_level_mapping={"x_h": 0},
            output_dim_vec=1,
            message_passing=2
        )

        # Initialize noise parameters
        noise_scales = (
            torch.rand(num_envs, 1, device=device) * (std_max - std_min) + std_min
        )
        self.register_buffer("noise_scales", noise_scales)
        self.register_buffer("std_min", torch.as_tensor(std_min, device=device))
        self.register_buffer("std_max", torch.as_tensor(std_max, device=device))

    def forward(self, obs, xanchor) -> torch.Tensor:
        return self.hepi.forward(obs, xanchor)

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