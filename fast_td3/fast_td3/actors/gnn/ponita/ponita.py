import torch
import torch.nn as nn

from torch import Tensor

from typing import Optional

import math


def scatter_add(src, index, dim_size):
    out_shape = [dim_size] + list(src.shape[1:])
    out = torch.zeros(out_shape, dtype=src.dtype, device=src.device)
    dims_to_add = src.dim() - index.dim()
    for _ in range(dims_to_add):
        index = index.unsqueeze(-1)
    index_expanded = index.expand_as(src)
    return out.scatter_add_(0, index_expanded, src)


def scatter_softmax(src, index, dim_size):
    src_exp = torch.exp(src - src.max())
    sum_exp = scatter_add(src_exp, index, dim_size) + 1e-6
    return src_exp / sum_exp[index]


class GridGenerator(nn.Module):
    def __init__(
        self,
        dim: int,
        n: int,
        steps: int = 200,
        step_size: float = 0.01,
        device: torch.device = None,
        only_upper_hemisphere: bool = False,
    ):
        super(GridGenerator, self).__init__()
        self.dim = dim
        self.n = n
        self.steps = steps
        self.step_size = step_size
        self.device = device if device else torch.device("cpu")
        self.only_upper_hemisphere = only_upper_hemisphere

    def forward(self) -> torch.Tensor:
        if self.dim == 2:
            return self.generate_s1()
        elif self.dim == 3:
            return self.generate_s2()
        else:
            raise ValueError("Only S1 and S2 are supported.")

    def generate_s1(self) -> torch.Tensor:
        angles = torch.linspace(start=0, end=2 * torch.pi - (2 * torch.pi / self.n), steps=self.n)
        x = torch.cos(angles)
        y = torch.sin(angles)
        return torch.stack((x, y), dim=1)

    def generate_s2(self) -> torch.Tensor:
        if self.only_upper_hemisphere:
            return self.fibonacci_lattice_upper_hemisphere(self.n, device=self.device)
        else:
            return self.fibonacci_lattice(self.n, device=self.device)

    def fibonacci_lattice(self, n: int, offset: float = 0.5, device: Optional[str] = None) -> Tensor:
        """
        Creating ~uniform grid of points on S2 using the fibonacci spiral algorithm.

        Arguments:
            - n: Number of points.
            - offset: Strength for how much points are pushed away from the poles.
                    Default of 0.5 works well for uniformity.
        """
        if n < 1:
            raise ValueError("n must be greater than 0.")

        i = torch.arange(n, device=device)

        theta = (math.pi * i * (1 + math.sqrt(5))) % (2 * math.pi)
        phi = torch.acos(1 - 2 * (i + offset) / (n - 1 + 2 * offset))

        cos_theta, sin_theta = torch.cos(theta), torch.sin(theta)
        cos_phi, sin_phi = torch.cos(phi), torch.sin(phi)

        return torch.stack((cos_theta * sin_phi, sin_theta * sin_phi, cos_phi), dim=-1)

    def fibonacci_lattice_upper_hemisphere(
        self, n: int, offset: float = 0.5, device: Optional[str] = None
    ) -> torch.Tensor:
        if n < 1:
            raise ValueError("n must be greater than 0.")
        i = torch.arange(n, device=device)
        theta = (math.pi * i * (1 + math.sqrt(5))) % (2 * math.pi)
        phi = torch.acos(1 - 1 * (i + offset) / (n - 1 + 2 * offset))  # Adjusted to keep within [0, pi/2]
        cos_theta, sin_theta = torch.cos(theta), torch.sin(theta)
        cos_phi, sin_phi = torch.cos(phi), torch.sin(phi)
        return torch.stack((cos_theta * sin_phi, sin_theta * sin_phi, cos_phi), dim=-1)


class SeparableFiberBundleConv(nn.Module):
    """ """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_dim,
        bias=True,
        groups=1,
        attention=False,
    ):
        super().__init__()

        # Check arguments
        if groups == 1:
            self.depthwise = False
        elif groups == in_channels and groups == out_channels:
            self.depthwise = True
            self.in_channels = in_channels
            self.out_channels = out_channels
        else:
            assert ValueError(
                "Invalid option for groups, should be groups=1 or groups=in_channels=out_channels (depth-wise separable)"
            )

        # Construct kernels
        self.kernel = nn.Linear(kernel_dim, in_channels, bias=False)
        self.fiber_kernel = nn.Linear(kernel_dim, int(in_channels * out_channels / groups), bias=False)
        self.attention = attention
        if self.attention:
            key_dim = 128
            self.key_transform = nn.Linear(in_channels, key_dim)
            self.query_transform = nn.Linear(in_channels, key_dim)
            nn.init.xavier_uniform_(self.key_transform.weight)
            nn.init.xavier_uniform_(self.query_transform.weight)
            self.key_transform.bias.data.fill_(0)
            self.query_transform.bias.data.fill_(0)

        # Construct bias
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
            self.bias.data.zero_()
        else:
            self.register_parameter("bias", None)

        # Automatic re-initialization
        self.register_buffer("callibrated", torch.tensor(False))

    def forward(self, x, kernel_basis, fiber_kernel_basis, edge_index):
        """ """

        # 1. Do the spatial convolution
        message = x[edge_index[0]] * self.kernel(kernel_basis)  # [num_edges, num_ori, in_channels]
        if self.attention:
            keys = self.key_transform(x)
            queries = self.query_transform(x)
            d_k = keys.size(-1)
            att_logits = (keys[edge_index[0]] * queries[edge_index[1]]).sum(dim=-1, keepdim=True) / math.sqrt(d_k)
            att_weights = scatter_softmax(att_logits, edge_index[1], x.size(0))
            message = message * att_weights
        x_1 = scatter_add(src=message, index=edge_index[1], dim_size=x.size(0))

        # 2. Fiber (spherical) convolution
        fiber_kernel = self.fiber_kernel(fiber_kernel_basis)
        if self.depthwise:
            x_2 = torch.einsum("boc,poc->bpc", x_1, fiber_kernel) / fiber_kernel.shape[-2]
        else:
            x_2 = (
                torch.einsum(
                    "boc,podc->bpd",
                    x_1,
                    fiber_kernel.unflatten(-1, (self.out_channels, self.in_channels)),
                )
                / fiber_kernel.shape[-2]
            )

        # Re-callibrate the initializaiton
        if self.training and not (self.callibrated):
            self.callibrate(x.std(), x_1.std(), x_2.std())

        # Add bias
        if self.bias is not None:
            return x_2 + self.bias
        else:
            return x_2

    def callibrate(self, std_in, std_1, std_2):
        print("Callibrating...")
        with torch.no_grad():
            self.kernel.weight.data = self.kernel.weight.data * std_in / std_1
            self.fiber_kernel.weight.data = self.fiber_kernel.weight.data * std_1 / std_2
            self.callibrated = ~self.callibrated


class SeparableFiberBundleConvNext(nn.Module):
    """ """

    def __init__(
        self,
        channels,
        kernel_dim,
        act=nn.GELU(),
        layer_scale=1e-6,
        widening_factor=4,
        attention=False,
    ):
        super().__init__()

        self.conv = SeparableFiberBundleConv(channels, channels, kernel_dim, groups=channels, attention=attention)
        self.act_fn = act
        self.linear_1 = nn.Linear(channels, widening_factor * channels)
        self.linear_2 = nn.Linear(widening_factor * channels, channels)
        if layer_scale is not None:
            self.layer_scale = nn.Parameter(torch.ones(channels) * layer_scale)
        else:
            self.register_buffer("layer_scale", None)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x, kernel_basis, fiber_kernel_basis, edge_index):
        """ """
        input = x
        x = self.conv(x, kernel_basis, fiber_kernel_basis, edge_index)
        x = self.norm(x)
        x = self.linear_1(x)
        x = self.act_fn(x)
        x = self.linear_2(x)
        if self.layer_scale is not None:
            x = self.layer_scale * x
        x = x + input
        return x


class PolynomialFeatures(nn.Module):
    def __init__(self, degree):
        super(PolynomialFeatures, self).__init__()

        self.degree = degree

    def forward(self, x):

        polynomial_list = [x]
        for it in range(1, self.degree + 1):
            polynomial_list.append(torch.einsum("...i,...j->...ij", polynomial_list[-1], x).flatten(-2, -1))
        return torch.cat(polynomial_list, -1)


class Ponita(nn.Module):
    """Steerable E(3) equivariant (non-linear) convolutional network"""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        num_layers,
        output_dim_vec=0,
        dim=3,
        num_ori=20,
        basis_dim=None,
        degree=2,
        widening_factor=4,
        layer_scale=None,
        task_level="graph",
        multiple_readouts=False,
        last_feature_conditioning=False,
        attention=False,
        only_upper_hemisphere=False,
        **kwargs
    ):
        super().__init__()

        self.dim = dim
        self.output_dim = output_dim
        self.output_dim_vec = output_dim_vec
        self.num_ori = num_ori

        self.last_feature_conditioning = last_feature_conditioning
        self.register_buffer(
            "ori_grid",
            GridGenerator(dim, num_ori, steps=1000, only_upper_hemisphere=only_upper_hemisphere)(),
        )

        # Input output settings
        self.output_dim, self.output_dim_vec = output_dim, output_dim_vec
        self.global_pooling = task_level == "graph"

        # Activation function to use internally
        act_fn = nn.GELU()

        # Kernel basis functions and spatial window
        basis_dim = hidden_dim if (basis_dim is None) else basis_dim
        self.basis_fn = nn.Sequential(
            PolynomialFeatures(degree),
            nn.Linear(sum(2**i for i in range(1, degree + 2)), hidden_dim),  # spatial invariants signal
            act_fn,
            nn.Linear(hidden_dim, basis_dim),
            act_fn,
        )
        self.fiber_basis_fn = nn.Sequential(
            PolynomialFeatures(degree),
            nn.Linear(sum(1**i for i in range(1, degree + 2)), hidden_dim),  # spatial invariants signal
            act_fn,
            nn.Linear(hidden_dim, basis_dim),
            act_fn,
        )

        # Initial node embedding
        self.x_embedder = nn.Linear(input_dim, hidden_dim, False)

        # Make feedforward network
        self.interaction_layers = nn.ModuleList()
        self.read_out_layers = nn.ModuleList()
        for i in range(num_layers):
            self.interaction_layers.append(
                SeparableFiberBundleConvNext(
                    hidden_dim,
                    basis_dim,
                    act=act_fn,
                    layer_scale=layer_scale,
                    widening_factor=widening_factor,
                    attention=attention,
                )
            )
            if multiple_readouts or i == (num_layers - 1):
                self.read_out_layers.append(nn.Linear(hidden_dim, output_dim + output_dim_vec))
            else:
                self.read_out_layers.append(None)

    def compute_invariants(self, ori_grid, pos, edge_index):
        pos_send, pos_receive = pos[edge_index[0]], pos[edge_index[1]]  # [num_edges, 3]
        rel_pos = pos_send - pos_receive  # [num_edges, 3]
        rel_pos = rel_pos[:, None, :]  # [num_edges, 1, 3]
        ori_grid_a = ori_grid[None, :, :]  # [1, num_ori, 3]
        ori_grid_b = ori_grid[:, None, :]  # [num_ori, 1, 3]
        # Displacement along the orientation
        invariant1 = (rel_pos * ori_grid_a).sum(dim=-1, keepdim=True)  # [num_edges, num_ori, 1]
        # Displacement orthogonal to the orientation (take norm in 3D)
        if self.dim == 2:
            invariant2 = (rel_pos - invariant1 * ori_grid_a).norm(dim=-1, keepdim=True)  # [num_edges, num_ori, 1]
        elif self.dim == 3:
            invariant2 = (rel_pos - invariant1 * ori_grid_a).norm(dim=-1, keepdim=True)  # [num_edges, num_ori, 1]
        # Relative orientation
        invariant3 = (ori_grid_a * ori_grid_b).sum(dim=-1, keepdim=True)  # [num_ori, num_ori, 1]
        # Stack into spatial and orientaiton invariants separately
        spatial_invariants = torch.cat([invariant1, invariant2], dim=-1)  # [num_edges, num_ori, 2]
        orientation_invariants = invariant3  # [num_ori, num_ori, 1]
        return spatial_invariants, orientation_invariants

    def forward(self, x, pos, edge_index, batch=None):

        ori_grid = self.ori_grid.type_as(pos)
        spatial_invariants, orientation_invariants = self.compute_invariants(ori_grid, pos, edge_index)

        # This is used to condition the generative models on noise levels (passed in the last channel of the input features)
        if self.last_feature_conditioning:
            cond = x[edge_index[0], None, -1:].repeat(1, ori_grid.shape[-2], 1)  # [num_edges, num_ori, 1]
            spatial_invariants = torch.cat([spatial_invariants, cond], dim=-1)

        # Sample the kernel basis and window the spatial kernel with a smooth cut-off
        kernel_basis = self.basis_fn(spatial_invariants)  # [num_edges, num_ori, basis_dim]
        fiber_kernel_basis = self.fiber_basis_fn(orientation_invariants)  # [num_ori, num_ori, basis_dim]

        # Initial feature embeding
        x = self.x_embedder(x)

        # Interaction + readout layers
        for interaction_layer in self.interaction_layers:
            x = interaction_layer(x, kernel_basis, fiber_kernel_basis, edge_index)
        return x


def main():
    from geometry_rl.modules.pyg_models.ponita.utils.to_from_sphere import (
        scalar_to_sphere,
        vec_to_sphere,
        sphere_to_scalar,
        sphere_to_vec,
    )

    device = "cuda"

    num_ori = 4
    batch_size = 4  #  16
    num_nodes = 3  # 24
    dim = 2
    output_dim = 1
    output_dim_vec = 1
    num_layers = 4
    hidden_dim = 16

    R = torch.tensor([[0.0, 1.0], [-1.0, 0.0]], device=device)  # 90 degrees rotation

    # Set up input
    input_vec = torch.zeros(batch_size, num_nodes, dim, device=device)
    input_vec[0] = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], device=device) * 2.0
    input_vec[1] = torch.einsum("ij,bj->bi", R, input_vec[0])
    input_vec[2] = torch.einsum("ij,bj->bi", R, input_vec[1])
    input_vec[3] = torch.einsum("ij,bj->bi", R, input_vec[2])
    input_scalar = torch.ones_like(input_vec[..., 0]) * 10.0

    pos = input_vec.clone()
    pos *= 10.0

    edge_index = torch.tensor(
        [[0, 1, 3, 4, 6, 7, 9, 10], [2, 2, 5, 5, 8, 8, 11, 11]],
        dtype=torch.long,
        device=device,
    )

    model = Ponita(
        1 + 1,
        hidden_dim,
        output_dim,
        num_layers,
        dim=2,
        num_ori=num_ori,
        output_dim_vec=output_dim_vec,
        task_level="graph",
    )
    model = model.to(device)

    scalar_input_sphere = scalar_to_sphere(input_scalar, model.ori_grid)
    scalar_input_sphere = scalar_input_sphere.permute(0, 2, 1).unsqueeze(-1)
    vec_input_sphere = vec_to_sphere(input_vec, model.ori_grid)
    vec_input_sphere = vec_input_sphere.permute(0, 2, 1).unsqueeze(-1)

    input_sphere = torch.cat([scalar_input_sphere, vec_input_sphere], dim=-1)

    input_sphere = input_sphere.reshape(batch_size * num_nodes, num_ori, -1)
    pos = pos.reshape(batch_size * num_nodes, dim)

    output = model(input_sphere, pos, edge_index=edge_index)
    output = model.read_out_layers[-1](output)

    output = output.reshape(batch_size, num_nodes, num_ori, -1)

    out_scalar, out_vec = torch.split(output, [model.output_dim, model.output_dim_vec], dim=-1)
    out_scalar = sphere_to_scalar(out_scalar)
    out_vec = sphere_to_vec(out_vec.view(-1, num_ori, output_dim_vec), model.ori_grid)
    out_vec = out_vec.reshape(batch_size, num_nodes, output_dim_vec, dim)

    loss = out_scalar.mean() + out_vec.mean()
    loss.backward()

    print(out_scalar.shape, out_vec.shape)


if __name__ == "__main__":
    main()
