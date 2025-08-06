# Taken from: https://github.com/ThijsKuipers1995/gconv
"""
rotation.py

Contains functionality for working with rotations (SO3 elements)
using the Pytorch Deep Learning framework.

Limited functionality for S2 spherical coordinates.

SO3 parameterizations:
    - Euler angles:
        - Rotations are assumed to be extrinsic.
        - Parameterization `(alpha, beta, gamma)` follows ZYZ convention.
        - `alpha` (Z) in `[-pi, pi]`.
        - `beta` (Y) in `[-pi/2, pi/2]`.
        - `gamma` (Z) in `[-pi, pi]`.

    - Quaternions:
        - `q = [w, x, y, z]` with `w` denoting the real component.

    - Matrices:
        - `R' as `3x3` rotation matrices with `det(R) = 1`.

S2 parameterization:
    - Spherical coordinates:
        - Rotations are assumed to be extrinsic.
        - Parameterization `(beta, gamma)` follows YZ convention.
        - `beta` (Y) in `[-pi/2, pi/2]`.
        - `gamma` (Z) in `[-pi, pi]`.
    
    - Euclidean vectors:
        - Parametrization `r = [x, y, z]` coordinates with `||r|| = 1`.

NOTE: The domains of the Euler and spherical parameterizations may not
      always be preserved, however, this has no effect on the functionality
      of this module.

"""

from math import pi
from typing import Callable, Optional

import torch
from torch import Tensor

from . import repulsion

import warnings

###############################
# UTILITY
###############################


def _rbf_gauss(r: Tensor, width: float = 1):
    """
    Gaussian radial basis function.

    Arguments:
        - r: Tensor.
        - width: Float denoting the width of the rbf.

    Returns:
        - Tensor.
    """
    return torch.exp(-((width * r) ** 2))


def _rbf_gauss2(x, width):
    """
    2ln2 = 1.44269
    """
    return torch.exp(-(x**2 / (width**2 / 0.69314718)))


# def _rbf_gauss2(x, width):
#     return torch.exp(-(((1 / width**2) * x) ** 2))


######################################
# SO(3)
######################################

_PARAMETERIZATION: set = set(("quat", "matrix", "euler"))


def matrix_x(theta: Tensor) -> Tensor:
    """
    Returns rotation matrix around x-axis for given
    angles. New Tensor is created on the device of
    the input tensor (theta).

    Arguments:
        - theta: Tensor containing angles.

    Returns:
        - Tensor of shape '(*theta.shape, 3, 3)' of rotation matrices.
    """
    r = theta.new_empty(*theta.shape, 3, 3)

    cos_theta, sin_theta = torch.cos(theta), torch.sin(theta)

    r[..., 0, 0] = 1
    r[..., 0, 1] = 0
    r[..., 0, 2] = 0

    r[..., 1, 0] = 0
    r[..., 1, 1] = cos_theta
    r[..., 1, 2] = -sin_theta

    r[..., 2, 0] = 0
    r[..., 2, 1] = sin_theta
    r[..., 2, 2] = cos_theta

    return r


def matrix_y(theta: Tensor) -> Tensor:
    """
    Returns rotation matrix around y-axis for given
    angles. New Tensor is created on the device of
    the input tensor (theta).

    Arguments:
        - theta: Tensor containing angles.

    Returns:
        - Tensor of shape '(*theta.shape, 3, 3)' of rotation matrices.
    """
    r = theta.new_empty(*theta.shape, 3, 3)

    cos_theta, sin_theta = torch.cos(theta), torch.sin(theta)

    r[..., 0, 0] = cos_theta
    r[..., 0, 1] = 0
    r[..., 0, 2] = sin_theta

    r[..., 1, 0] = 0
    r[..., 1, 1] = 1
    r[..., 1, 2] = 0

    r[..., 2, 0] = -sin_theta
    r[..., 2, 1] = 0
    r[..., 2, 2] = cos_theta

    return r


def matrix_z(theta: Tensor) -> Tensor:
    """
    Returns rotation matrix around z-axis for given
    angles. New Tensor is created on the device of
    the input tensor (theta).

    Arguments:
        - theta: Tensor containing angles.

    Returns:
        - Tensor of shape '(*theta.shape, 3, 3)' of rotation matrices.
    """
    r = theta.new_empty(*theta.shape, 3, 3)

    cos_theta, sin_theta = torch.cos(theta), torch.sin(theta)

    r[..., 0, 0] = cos_theta
    r[..., 0, 1] = -sin_theta
    r[..., 0, 2] = 0

    r[..., 1, 0] = sin_theta
    r[..., 1, 1] = cos_theta
    r[..., 1, 2] = 0

    r[..., 2, 0] = 0
    r[..., 2, 1] = 0
    r[..., 2, 2] = 1

    return r


def quat_inverse(q: Tensor) -> Tensor:
    """
    Inverts quaternion.

    Arguments:
        - q: Tensor of shape `(..., 4)`.

    Returns:
        - Tensor of shape `(..., 4)`.
    """
    return q * q.new_tensor([1.0, -1.0, -1.0, -1.0])


def matrix_inverse(r: Tensor) -> Tensor:
    """
    Inverts rotation matrix.

    Arguments:
        - r: Tensor of shape `(..., 3, 3)`.

    Returns:
        - Tensor of shape `(..., 3, 3)`.
    """
    return r.transpose(-1, -2)


def left_apply_quat(q1: Tensor, q2: Tensor) -> Tensor:
    """
    Calculates `q = q1 * q2`.

    Arguments:
        - q1, q2: Tensors of shape `(..., 4)`.

    Returns:
        - Tensor of shape `(..., 4)`.
    """
    q = q1.new_empty(q1.shape)

    cross = torch.cross(q1[..., 1:], q2[..., 1:])

    q[..., 0] = (
        q1[..., 0] * q2[..., 0]
        - q1[..., 1] * q2[..., 1]
        - q1[..., 2] * q2[..., 2]
        - q1[..., 3] * q2[..., 3]
    )
    q[..., 1] = q1[..., 0] * q2[..., 1] + q2[..., 0], q1[..., 1] + cross[..., 0]
    q[..., 2] = q1[..., 0] * q2[..., 2] + q2[..., 0], q1[..., 2] + cross[..., 1]
    q[..., 3] = q1[..., 0] * q2[..., 3] + q2[..., 0], q1[..., 3] + cross[..., 2]

    return q


def left_apply_matrix(r1, r2):
    """
    Calculates `r1 @ r2`.

    Arguments:
        - r1, r2: Tensors of shape `(..., 3, 3)`.

    Returns:
        - Tensor of shape `(..., 3, 3)`.
    """
    return torch.matmul(r1, r2)


def euler_to_quat(g: Tensor, eps: float = 1e-6) -> Tensor:
    """
    Transforms Euler parameterization to quaternions.

    Arguments:
        - g: Tensor of shape `(..., 3)`.
        - eps: Float for preventing numerical instabilities.

    Returns:
        - Tensor of shape `(..., 4)`.
    """
    return matrix_to_quat(euler_to_matrix(g, eps=eps))


def euler_to_matrix(g: Tensor, eps: float = 1e-6) -> Tensor:
    """
    Transforms Euler parameterization to rotation matrices.

    Arguments:
        - g: Tensor of shape `(..., 3)`.
        - eps: Float for preventing numerical instabilities.

    Returns:
        - Tensor of shape `(..., 3, 3)`.
    """
    r = g.new_empty((*g.shape[:-1], 3, 3))

    cos_g = torch.cos(g)
    cos_alpha, cos_beta, cos_gamma = cos_g[..., 0], cos_g[..., 1], cos_g[..., 2]

    sin_g = torch.sin(g)
    sin_alpha, sin_beta, sin_gamma = sin_g[..., 0], sin_g[..., 1], sin_g[..., 2]

    r[..., 0, 0] = cos_alpha * cos_beta * cos_gamma - sin_alpha * sin_gamma
    r[..., 0, 1] = -cos_alpha * sin_gamma - cos_gamma * cos_beta * sin_alpha
    r[..., 0, 2] = cos_gamma * sin_beta

    r[..., 1, 0] = cos_gamma * sin_alpha + cos_beta * cos_alpha * sin_gamma
    r[..., 1, 1] = cos_gamma * cos_alpha - cos_beta * sin_alpha * sin_gamma
    r[..., 1, 2] = sin_gamma * sin_beta

    r[..., 2, 0] = -cos_alpha * sin_beta
    r[..., 2, 1] = sin_beta * sin_alpha
    r[..., 2, 2] = cos_beta

    mask = (r > eps) | (r < -eps)

    return r * mask


def matrix_to_quat(r: Tensor) -> Tensor:
    """
    Transforms rotation matrices to a quaternions.

    Arguments:
        - r: Tensor of shape `(..., 3, 3)`.

    Returns:
        - Tensor of shape `(..., 4)`.
    """
    q = r.new_empty(*r.shape[:-2], 4)

    decision = torch.diagonal(r, 0, dim1=1, dim2=2)
    decision = torch.cat((decision, torch.sum(decision, dim=-1, keepdim=True)), dim=-1)

    i = torch.argmax(decision, dim=-1)
    j = (i + 1) % 3
    k = (j + 1) % 3
    c_m = i != 3
    nc_m = ~c_m

    c_i = i[c_m]
    c_j = j[c_m]
    c_k = k[c_m]

    q[c_m, c_i + 1] = 1 - decision[c_m, 3] + 2 * r[c_m, c_i, c_i]
    q[c_m, c_j + 1] = r[c_m, c_j, c_i] + r[c_m, c_i, c_j]
    q[c_m, c_k + 1] = r[c_m, c_k, c_i] + r[c_m, c_i, c_k]
    q[c_m, 0] = r[c_m, c_k, c_j] - r[c_m, c_j, c_k]

    q[nc_m, 1] = r[nc_m, 2, 1] - r[nc_m, 1, 2]
    q[nc_m, 2] = r[nc_m, 0, 2] - r[nc_m, 2, 0]
    q[nc_m, 3] = r[nc_m, 1, 0] - r[nc_m, 0, 1]
    q[nc_m, 0] = 1 + decision[nc_m, 3]

    return q / torch.linalg.norm(q, dim=-1, keepdim=True)


def matrix_to_euler(r: Tensor, eps: float = 1e-5, no_warn: bool = False) -> Tensor:
    """
    Transforms rotation matrices to euler angles. When a gimble lock is
    detected, the third euler angle (gamma) will be set to zero. Note
    that this will still result in the correct rotation.

    Adapted from 'scipy.spatial.transform._rotation.pyx/_euler_from_matrix'.

    Arguments:
        - r: Tensor of shape `(..., 3, 3)`.
        - eps: Float for preventing numerical instabilities.
        - no_warn: Bool to display gimble lock warning, default
                   `no_warn = False`.

    Returns:
        - Tensor of shape `(..., 3)`.
    """
    g = r.new_empty((*r.shape[:-2], 3))

    # step 1, 2
    c = r.new_tensor([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

    # step 3
    res = torch.matmul(c, r)
    matrix_trans = torch.matmul(res, c.T)

    # step 4
    matrix_trans[:, 2, 2] = torch.clamp(matrix_trans[:, 2, 2], -1, 1)

    g[:, 1] = torch.acos(matrix_trans[:, 2, 2])

    # step 5, 6
    safe1 = torch.abs(g[:, 1]) >= eps
    safe2 = torch.abs(g[:, 1] - pi) >= eps
    safe = safe1 & safe2

    not_safe1 = ~safe1
    not_safe2 = ~safe2

    # step 5b
    g[safe, 2] = torch.atan2(matrix_trans[safe, 0, 2], -matrix_trans[safe, 1, 2])
    g[safe, 0] = torch.atan2(matrix_trans[safe, 2, 0], matrix_trans[safe, 2, 1])

    g[~safe, 2] = 0
    g[not_safe1, 0] = torch.atan2(
        matrix_trans[not_safe1, 1, 0] - matrix_trans[not_safe1, 0, 1],
        matrix_trans[not_safe1, 0, 0] + matrix_trans[not_safe1, 1, 1],
    )
    g[not_safe2, 0] = -torch.atan2(
        matrix_trans[not_safe2, 1, 0] + matrix_trans[not_safe2, 0, 1],
        matrix_trans[not_safe2, 0, 0] - matrix_trans[not_safe2, 1, 1],
    )

    # step 7
    adjust_and_safe = ((g[:, 1] < 0) | (g[:, 1] > pi)) & safe

    g[adjust_and_safe, 0] -= pi
    g[adjust_and_safe, 1] *= -1
    g[adjust_and_safe, 2] += pi

    g[g < -pi] += 2 * pi
    g[g >= pi] -= 2 * pi

    if not no_warn and not torch.any(safe):
        warnings.warn(
            "Gimbal lock detected. Setting third angle to zero "
            "since it is not possible to uniquely determine "
            "all angles."
        )

    return g


def quat_to_matrix(q: Tensor) -> Tensor:
    """
    Transforms quaternions to rotation matrices.

    Arguments:
        - q: Tensor of shape `(..., 4)`.

    Returns:
        - Tensor of shape `(..., 3, 3)`.
    """
    r = q.new_empty((*q.shape[:-1], 3, 3))

    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    x2, y2, z2, w2 = x * x, y * y, z * z, w * w

    xy, zw, xz = x * y, z * w, x * z
    yw, yz, xw = y * w, y * z, x * w

    r[..., 0, 0] = x2 - y2 - z2 + w2
    r[..., 0, 1] = 2 * (xy - zw)
    r[..., 0, 2] = 2 * (xz + yw)

    r[..., 1, 0] = 2 * (xy + zw)
    r[..., 1, 1] = -x2 + y2 - z2 + w2
    r[..., 1, 2] = 2 * (yz - xw)

    r[..., 2, 0] = 2 * (xz - yw)
    r[..., 2, 1] = 2 * (yz + xw)
    r[..., 2, 2] = -x2 - y2 + z2 + w2

    return r


def quat_to_euler(q: Tensor) -> Tensor:
    """
    Converts quaternions to Euler angles.

    Arguments:
        - q: Tensor of shape `(..., 4)`.

    Returns:
        - Tensor of shape `(..., 3)`.
    """
    return matrix_to_euler(quat_to_matrix(q))


def quaternion_raw_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    from: https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html#quaternion_multiply
    Multiply two quaternions.
    Usual torch rules for broadcasting apply.

    Args:
        a: Quaternions as tensor of shape (..., 4), real part first.
        b: Quaternions as tensor of shape (..., 4), real part first.

    Returns:
        The product of a and b, a tensor of quaternions shape (..., 4).
    """
    aw, ax, ay, az = torch.unbind(a, -1)
    bw, bx, by, bz = torch.unbind(b, -1)
    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw
    return torch.stack((ow, ox, oy, oz), -1)


def left_apply_to_R3(R: Tensor, grid: Tensor) -> Tensor:
    """
    Applies each rotation matrix in `R` to `grid`.

    Arguments:
        - R: Tensor of shape `(..., 3, 3)` of rotation matrices.
        - grid: Tensor of shape `(x, y, z, 3)` R3 vectors.

    Returns:
        - Tensor of shape `(..., x, y, z, 3)` of transformed
        - R3 vectors.
    """
    return (R[..., None, None, None, :, :] @ grid[..., None]).squeeze(-1)


def left_apply_to_matrix(R1: Tensor, R2: Tensor) -> Tensor:
    """
    Multiplies each matrix in `R1` to each matrix in `R2`.

    Arguments:
        - R1: Tensor of shape `(N, 3, 3)`.
        - R2: Tensor of shape `(M, 3, 3)`.

    Returns:
        - Tensor of shape `(N, M, 3, 3)` of rotation matrices.
    """
    return torch.matmul(R1[:, None], R2)


def so3_log(R: Tensor, eps: float = 1e-7) -> Tensor:
    """
    Calculates the riemannnian logarithm of each matrix in `R`.

    Adapted from https://github.com/haguettaz/ChebLieNet/blob/main/cheblienet/geometry/so.py.

    Arguments:
        - r: Tensor of shape `(..., 3, 3)`.
        - eps: Float for preventing numerical instabilities.

    Returns:
        - Tensor of shape (..., 3).
    """
    # clamping to prevent numerical instabilities
    theta = torch.acos(
        torch.clamp(
            ((R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]) - 1) / 2, -1 + eps, 1 - eps
        )
    )

    theta_sin_inv = 0.5 * theta / torch.sin(theta)

    c1 = theta_sin_inv * (R[..., 2, 1] - R[..., 1, 2])
    c2 = theta_sin_inv * (R[..., 0, 2] - R[..., 2, 0])
    c3 = theta_sin_inv * (R[..., 1, 0] - R[..., 0, 1])

    mask = theta == 0.0

    c1[mask] = 0.5 * R[mask, 2, 1] - R[mask, 1, 2]
    c2[mask] = 0.5 * R[mask, 0, 2] - R[mask, 2, 0]
    c3[mask] = 0.5 * R[mask, 1, 0] - R[mask, 0, 1]

    mask = theta == pi

    c1[mask] = pi
    c2[mask] = 0.0
    c3[mask] = 0.0

    c = torch.stack((c1, c2, c3), dim=-1)

    return c


@torch.jit.script
def geodesic_distance(
    qx: torch.Tensor, qy: torch.Tensor, eps: float = 1e-7
) -> torch.Tensor:
    """
    Calculates the geodesic distance between quaternions `qx` and `qy`.

    Arguments:
        - qx, qy: Tensors of shape `(..., 4)`.
        - eps: Float for preventing numerical instabilities.

    Returns:
        - Tensor of shape `(...)`.
    """
    return torch.acos(torch.clamp((qx * qy).sum(-1).abs(), -1 + eps, 1 - eps))


def random_quat(shape: tuple[int] | int, device: Optional[str] = None) -> Tensor:
    """
    Uniformly samples SO3 elements parameterized as quaternions.

    Arguments:
        - shape: Int or tuple denoting the shape of the output tensor.
        - device: device on which the new tensor is created.

    Returns:
        - Tensor of shape `(*shape, 4)`.
    """
    shape = shape if type(shape) is tuple else (shape,)

    q = torch.randn(*shape, 4, device=device)
    q = q / torch.linalg.norm(q, dim=-1, keepdim=True)

    return q


def random_matrix(shape: tuple[int] | int, device: Optional[str] = None) -> Tensor:
    """
    Uniformly samples SO3 elements parameterized as matrices.

    Arguments:
        - shape: Int or tuple denoting the shape of the output tensor.
        - device: device on which the new tensor is created.

    Returns:
        - Tensor of shape `(*shape, 3, 3)`.
    """
    return quat_to_matrix(random_quat(shape, device=device))


def random_euler(shape: tuple[int] | int, device: Optional[str] = None) -> Tensor:
    """
    Uniformly samples SO3 elements parameterized as euler angles (ZYZ).

    Arguments:
        - shape: Int or tuple denoting the shape of the output tensor.
        - device: device on which the new tensor is created.

    Returns:
        - Tensor of shape `(*shape, 3)`.
    """
    return quat_to_euler(random_quat(shape, device=device))


def uniform_grid(
    n: int,
    parameterization: str = "quat",
    steps: int = 200,
    step_size: Optional[float] = None,
    show_pbar: bool = True,
    device: Optional[str] = None,
) -> Tensor:
    """
    Creates a uniform grid of `n` rotations. Rotations will be uniform
    with respect to the geodesic distance.

    Arguments:
        - n: Number of rotations in grid.
        - parameterization: Parameterization of the returned grid elements. Must
                            be either 'quat', 'matrix', or 'euler'. Defaults to
                            'quat'.
        - steps: Number of minimization steps.
        - step_size: Strength of minimization step. If not provided, strength will be
                     calculated as `n ** (1 / 3` which will generally result in stable uniform
                     grids for `n` in the range of `[2, 1000]`.
        - show_pbar: If True, will show progress of optimization procedure.
        - device: Device on which energy minimization will be performed and on
                  which the output grid will be defined.

    Returns:
        - Tensor containing uniform grid on SO3.
    """
    match parameterization.lower():
        case "quat":
            param_fn = euler_to_quat
        case "matrix":
            param_fn = euler_to_matrix
        case "euler":
            param_fn = lambda x: matrix_to_euler(euler_to_matrix(x))
        case _:
            raise ValueError(f"Parameterization must be in {_PARAMETERIZATION}.")

    step_size = step_size if step_size is not None else n ** (1 / 3)

    grid = random_euler(n, device=device)

    repulsion.repulse(
        grid,
        steps=steps,
        step_size=step_size,
        alpha=0.001,
        metric_fn=geodesic_distance,
        transform_fn=euler_to_quat,
        dist_normalization_constant=pi / 2,
        show_pbar=show_pbar,
        in_place=True,
    )

    return param_fn(grid)


def nearest_neighbour_interpolation(
    rotations: Tensor,
    grid: Tensor,
    signal: Tensor,
    *,
    dist_fn: Callable = geodesic_distance,
) -> Tensor:
    """
    Performs nearest neighbour interpolation of rotations with
    respect to `signal` that is defined on `grid`.

    Arguments:
        - rotations: Tensor of shape `(N, K, 4)` containing quaternions.
        - grid: Tensor of shape `(N, L, 4)` containing quaternions
                defining a grid on SO3.
        - signal: Tensor of shape `(N, L, S)` containing the signal
                  corresponding to `grid`.
        - dist_fn: Distance function used to calculate distance between
                   rotations.

    Returns:
        - Tensor of shape `(N, K, S)`.
    """
    dims = signal.shape[2:]

    dists = dist_fn(rotations[..., :, None, :], grid[..., None, :, :])
    idx = torch.topk(dists, k=1, largest=False)[1]

    return signal.gather(1, idx.expand(-1, -1, *dims))


def barycentric_interpolation(
    rotations: Tensor,
    grid: Tensor,
    signal: Tensor,
    *,
    dist_fn=geodesic_distance,
    eps: float = 1e-3,
) -> Tensor:
    """
    Performs linear interpolation of rotations using barycentric coordinates.

    Arguments:
        - rotations: Tensor of shape `(N, K, 4)` containing quaternions.
        - grid: Tensor of shape `(N, L, 4)` containing quaternions
                defining a grid on SO3.
        - signal: Tensor of shape `(N, L, S)` containing the signal
                  corresponding to `grid`.
        - dist_fn: Distance function used to calculate distance between
                   rotations.
        - eps: Float for preventing numerical instabilities.

    Returns:
        - Tensor of shape `(N, K, S)`.
    """
    N, _, S = signal.shape
    _, H, D = rotations.shape

    dists = dist_fn(rotations[..., None, :], grid[..., None, :, :])
    dists_k, idx = torch.topk(dists, k=3, largest=False)

    simplices = (
        grid[..., None, :]
        .expand(-1, -1, D, -1)
        .gather(1, idx[..., None].expand(-1, -1, -1, D))
    )

    # to make sure all quaternions live on same shard
    simplices *= 2 * (simplices @ rotations[..., None] > 0) - 1

    bcc = torch.linalg.lstsq(simplices.transpose(-1, -2), rotations[..., None])[0].view(
        -1, 3
    )

    mask = dists_k[:, :, 0].view(-1) <= eps
    bcc[mask, 0], bcc[mask, 1], bcc[mask, 2] = 1.0, 0.0, 0.0
    bcc[bcc < 0] = 0

    bcc = bcc.view(N, H, 3)
    bcc /= bcc.sum(-1, keepdim=True)

    signal = (
        signal[..., None, :]
        .expand(-1, -1, 3, -1)
        .gather(1, idx[..., None].expand(-1, -1, -1, S))
    )

    return torch.sum(bcc[..., None] * signal, axis=-2)


def _depr_rbf_interpolation(
    rotations: Tensor,
    grid: Tensor,
    signal: Tensor,
    *,
    dist_fn: Tensor = geodesic_distance,
    rbf: Callable[[Tensor], Tensor] = _rbf_gauss2,
    width=2,
) -> Tensor:
    """
    Performs radial basis function interpolation of rotations.

    Arguments:
        Arguments:
        - rotations: Tensor of shape `(N, K, 4)` containing quaternions.
        - grid: Tensor of shape `(N, L, 4)` containing quaternions
                defining a grid on SO3.
        - signal: Tensor of shape `(N, L, S)` containing the signal
                  corresponding to `grid`.
        - dist_fn: Distance function used to calculate distance between
                   rotations.
        - rbf: Radial basis function, defaults to gaussian rbf.
        - rbf_args: Tuple containing parameter arguments for `rbf`.

    Returns:
        - Tensor of shape '(N, K, S)'.
    """
    m = rbf(dist_fn(grid[..., None, :], grid[..., None, :, :]), width)

    coeffs = torch.linalg.solve(m, signal).transpose(-1, -2)
    p = rbf(dist_fn(rotations[..., None, :], grid[..., None, :, :]), width)

    # coeffs (N, P, G)    p (N, S, G) --> (N, P, S, G)
    return torch.sum(coeffs[:, None] * p[..., None, :], axis=-1, keepdim=True).squeeze(
        -1
    )


def rbf_interpolation(
    rotations: Tensor,
    grid: Tensor,
    signal: Tensor,
    *,
    dist_fn: Tensor = geodesic_distance,
    rbf: Callable[[Tensor], Tensor] = _rbf_gauss2,
    width=2,
) -> Tensor:
    """
    Performs radial basis function interpolation of rotations.

    Arguments:
        Arguments:
        - rotations: Tensor of shape `(N, K, 4)` containing quaternions.
        - grid: Tensor of shape `(N, L, 4)` containing quaternions
                defining a grid on SO3.
        - signal: Tensor of shape `(N, L, S)` containing the signal
                  corresponding to `grid`.
        - dist_fn: Distance function used to calculate distance between
                   rotations.
        - rbf: Radial basis function, defaults to gaussian rbf.
        - rbf_args: Tuple containing parameter arguments for `rbf`.

    Returns:
        - Tensor of shape '(N, K, S)'.
    """
    m = rbf(dist_fn(grid[..., None, :], grid[..., None, :, :]), width)

    coeffs = torch.linalg.solve(m, signal).transpose(-1, -2)
    p = rbf(dist_fn(rotations[..., None, :], grid[..., None, :, :]), width)

    return p @ coeffs.transpose(-1, -2)


def quaternion_log(q: Tensor) -> Tensor:
    """
    Calculates logarithm of quaternions.

    NOTE: log of identity quaternion will be set to the zero
          vector.

    Arguments:
        - q: Tensor of shape `(N, 4)`.

    Returns:
        - Tensor of shape `(N, 4)`.
    """
    q_norm = torch.linalg.norm(q, dim=-1, keepdim=True)

    inverse_v_norm = 1 / torch.linalg.norm(q[:, 1:], dim=-1, keepdim=True)
    inverse_v_norm[inverse_v_norm == torch.inf] = 0

    log_q_norm = torch.log(q_norm)
    vector_part = (inverse_v_norm * torch.acos(q[:, 0].view(-1, 1) / q_norm)) * q[:, 1:]

    return torch.hstack((log_q_norm, vector_part))


def nearest_neighbour_distance(q: Tensor, keepdim: bool = False) -> Tensor:
    """
    Calculates the nearest neighbour distance between all elements
    in given grid of rotations `q` parameterized as quaternions.

    Arguments:
        - q: Tensor of shape (N, 4).
        - keepdims: If True, result will be of shape (N, 1), else
                    (N,). Default value is False.

    Returns:
        - Tensor of shape (N,) or (N, 1) if `keepdims = True`.
    """
    return geodesic_distance(q[:, None], q).sort()[0][:, 1]


######################################
# S2
######################################


def spherical_to_euclid(g: Tensor) -> Tensor:
    """
    Converts spherical coordinates to euclidean coordinates.

    Arguments:
        - g: Tensor of shape `(..., 2)`.

    Returns:
        - Tensor of shape `(..., 3)`.
    """
    x = g.new_empty((*g.shape[:-1], 3))

    beta = g[..., 0]
    gamma = g[..., 1]

    x[..., 0] = torch.sin(beta) * torch.cos(gamma)
    x[..., 1] = torch.sin(beta) * torch.sin(gamma)
    x[..., 2] = torch.cos(beta)

    return x


def euclid_to_spherical(x: Tensor) -> Tensor:
    """
    Converts euclidean coordinates to spherical coordinates.

    Arguments:
        - g: Tensor of shape `(..., 3)`.

    Returns:
       -  Tensor of shape `(..., 2)`.
    """
    g = x.new_empty((*x.shape[:-1], 2))

    g[..., 0] = torch.acos(x[..., 2])
    g[..., 1] = torch.atan2(x[..., 1], x[..., 0])

    return g


def random_s2(shape: tuple[int, ...], device: Optional[str] = None) -> torch.Tensor:
    """
    Generates Tensor of uniformly sampled spherical coordinates on
    S2.

    Arguments:
        - shape: Shape of the output tensor.
        - device: Device on which the new tensor is created.

    Returns:
        - Tensor of shape (*shape, 3).
    """
    x = torch.randn((*shape, 3), device=device)
    return euclid_to_spherical(x / torch.linalg.norm(x, dim=-1, keepdim=True))


def geodesic_distance_s2(r1: Tensor, r2: Tensor, eps: float = 1e-7):
    return torch.acos(torch.clamp((r1 * r2).sum(-1), -1 + eps, 1 - eps))


def spherical_to_euler(g: Tensor) -> Tensor:
    alpha = g.new_zeros(g.shape[0], 1)
    return torch.hstack((alpha, g))


def spherical_to_euler_neg_gamma(g: Tensor) -> Tensor:
    minus_gamma = g[:, 1]
    return torch.hstack((minus_gamma, g))


def uniform_grid_s2(
    n: int,
    parameterization: str = "euclidean",
    set_alpha_as_neg_gamma: bool = False,
    steps: int = 100,
    step_size: float = 0.1,
    show_pbar: bool = True,
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