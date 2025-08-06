# Taken from: https://github.com/ThijsKuipers1995/gconv

import torch
from torch import Tensor
from tqdm import trange
from typing import Callable, Optional


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

def uniform_grid_s2(
    n: int,
    parameterization: str = "euclidean",
    set_alpha_as_neg_gamma: bool = False,
    steps: int = 100,
    step_size: float = 0.1,
    show_pbar: bool = False,
    device: Optional[str] = None,
) -> Tensor:
    """
    Creates a uniform grid of `n` rotations on S2. Rotations will be uniform
    with respect to the geodesic distance.

    Arguments:
        - n: Number of rotations in grid.
        - parameterization: Parameterization of the returned grid elements. Must
                            be either 'spherical', 'euclidean', 'quat', 'matrix', or 'euler'. Defaults to
                            'euclidean'.
        - steps: Number of minimization steps.
        - step_size: Strength of minimization step. Default of 0.1 works well.
        - show_pbar: If True, will show progress of optimization procedure.
        - device: Device on which energy minimization will be performed and on
                  which the output grid will be defined.

    Returns:
        - Tensor containing uniform grid on SO3.
    """
    add_alpha = False
    to_so3_fn = (
        spherical_to_euler_neg_gamma if set_alpha_as_neg_gamma else spherical_to_euler
    )

    match parameterization.lower():
        case "spherical":
            param_fn = lambda x: x
        case "euclidean":
            param_fn = spherical_to_euclid
        case "euler":
            add_alpha = True
            param_fn = lambda x: x
        case "matrix":
            add_alpha = True
            param_fn = euler_to_matrix
        case "quat":
            add_alpha = True
            param_fn = euler_to_quat

    grid = random_s2((n,), device=device)

    repulsion.repulse(
        grid,
        steps=steps,
        step_size=step_size,
        alpha=0.001,
        metric_fn=geodesic_distance_s2,
        transform_fn=spherical_to_euclid,
        dist_normalization_constant=pi,
        show_pbar=show_pbar,
        in_place=True,
    )

    grid = to_so3_fn(grid) if add_alpha else grid

    return param_fn(grid)
