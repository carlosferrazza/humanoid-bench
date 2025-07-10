import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(
        self,
        n_obs: int,
        n_act: int,
        num_envs: int,
        init_scale: float,
        hidden_dim: int,
        std_min: float = 0.05,
        std_max: float = 0.8,
        device: torch.device = None,
    ):
        super().__init__()
        self.n_act = n_act
        self.net = nn.Sequential(
            nn.Linear(n_obs, hidden_dim, device=device),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2, device=device),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4, device=device),
            nn.ReLU(),
        )
        self.fc_mu = nn.Sequential(
            nn.Linear(hidden_dim // 4, n_act, device=device),
            nn.Tanh(),
        )
        nn.init.normal_(self.fc_mu[0].weight, 0.0, init_scale)
        nn.init.constant_(self.fc_mu[0].bias, 0.0)

        noise_scales = (
            torch.rand(num_envs, 1, device=device) * (std_max - std_min) + std_min
        )
        self.register_buffer("noise_scales", noise_scales)

        self.register_buffer("std_min", torch.as_tensor(std_min, device=device))
        self.register_buffer("std_max", torch.as_tensor(std_max, device=device))
        self.n_envs = num_envs

    def forward(self, obs: torch.Tensor, xpos: torch.Tensor = None) -> torch.Tensor:

        x = obs
        x = self.net(x)
        action = self.fc_mu(x)
        return action

    def explore(
        self, obs: torch.Tensor, xpos: torch.Tensor = None, dones: torch.Tensor = None, deterministic: bool = False
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

        act = self(obs)
        if deterministic:
            return act

        noise = torch.randn_like(act) * self.noise_scales
        return act + noise

