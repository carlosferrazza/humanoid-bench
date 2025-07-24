# Taken from: https://github.com/ThijsKuipers1995/gconv
"""
repulsion.py

Contains repulsion model.
"""


import torch
from torch import Tensor

from tqdm import trange

from typing import Callable


def columb_energy(d: Tensor, k: int = 2) -> Tensor:
    """
    Returns columb energy over given input.
    
    Arguments:
        - d: Tensor to calculate columb energy over.
        - k: Exponent term of columb energy.

    Returns:
        - Tensor containing columb energy.
    """
    return d ** (-k)


def repulse(
        grid: Tensor,
        steps: int = 200,
        step_size: float = 10,
        metric_fn: Callable = lambda x, y: x - y,
        transform_fn: Callable = lambda x: x,
        energy_fn: Callable = columb_energy,
        dist_normalization_constant: float = 1,
        alpha: float = 0.001,
        show_pbar: bool = True,
        in_place: bool = False,
    ) -> Tensor:
    """
    Performs repulsion grids defined on Sn. Will perform
    repulsion on device that grid is defined on.

    Arguments:
        - grid: Tensor of shape (N, D) of grid elements
                parameterized with a periodic parameterization, i.e.
                spherical angles on S2.
        - steps: Number times the optimimzation step will be peformed.
        - step_size: Strength of each optimization step.
        - dist_fn: Metric for which energy will be minimized.
        - transform_fn: Function that transforms grid to the parameterization
                        used by the metric.
        - energy_fn: Function used to calculate energy. Defaults to columb energy.
        - dist_normalization_constant: Optional constant for normalizing distances.
                                       Set to max distance from metric or else 1.
        - show_pbar: If True, will show progress of optimization procedure.
        - in_place: If True, will update grid in place.

    Returns:
        - Tensor of shape (N, D) of minimized grid.
    """
    
    pbar = trange(steps, disable=not show_pbar, desc="Optimizing")

    grid = grid if in_place else grid.clone()
    grid.requires_grad = True

    optimizer = torch.optim.SGD([grid], lr=step_size)

    for epoch in pbar:
        optimizer.zero_grad(set_to_none=True)

        grid_transform = transform_fn(grid)

        dists = metric_fn(grid_transform[:, None], grid_transform).sort(dim=-1)[0][:, 1:]
        energy_matrix = energy_fn(dists / dist_normalization_constant)

        mean_total_energy = energy_matrix.mean()
        mean_total_energy.backward()
        grid.grad += (steps - epoch) / steps * alpha * torch.randn(grid.grad.shape, device=grid.device)

        optimizer.step()

        pbar.set_postfix_str(f"mean total energy: {mean_total_energy.item():.3f}")

    grid.requires_grad = False

    return grid.detach()