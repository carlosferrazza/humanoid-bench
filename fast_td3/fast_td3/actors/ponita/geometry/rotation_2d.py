import torch
from typing import Optional


def random_so2_matrix(batch_size, device: Optional[str] = None) -> torch.Tensor:
    # Generate random angles
    angles = 2 * torch.pi * torch.rand(batch_size, device=device)

    # Calculate sin and cos for each angle
    cos_vals = torch.cos(angles)
    sin_vals = torch.sin(angles)

    # Construct the rotation matrices
    rotation_matrices = torch.stack([cos_vals, -sin_vals, sin_vals, cos_vals], dim=1)
    rotation_matrices = rotation_matrices.view(batch_size, 2, 2)

    return rotation_matrices


def uniform_grid_s1(num_points: int, device: Optional[str] = None) -> torch.Tensor:
    # Generate angles uniformly
    # NOTE: the last element should be one portion before 2*pi.
    angles = torch.linspace(
        start=0, 
        end=2 * torch.pi - (2 * torch.pi / num_points), 
        steps=num_points
    )

    # Calculate x and y coordinates
    x = torch.cos(angles)
    y = torch.sin(angles)

    return torch.stack((x, y), dim=1)