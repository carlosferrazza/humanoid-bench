import torch
import torch.nn as nn
from torch_geometric.nn import global_add_pool
from torch_scatter import scatter_mean
from torch_geometric.transforms import Compose
from torch_geometric.transforms import BaseTransform, Compose, RadiusGraph

from fast_td3.actors.ponita.utils.to_from_sphere import sphere_to_scalar, sphere_to_vec
from fast_td3.actors.ponita.nn.embedding import PolynomialFeatures
from fast_td3.actors.ponita.utils.windowing import PolynomialCutoff
from fast_td3.actors.ponita.transforms import (
    PositionOrientationGraph,
    SEnInvariantAttributes,
)
from fast_td3.actors.ponita.nn.convnext import ConvNext
from fast_td3.actors.ponita.nn.conv import Conv, FiberBundleConv
from fast_td3.skeleton_builder import build_edge_index_and_node_attr


# Wrapper to automatically switch between point cloud mode (num_ori = -1 or 0) and
# bundle mode (num_ori > 0).
def Ponita(
    input_dim,
    hidden_dim,
    output_dim,
    num_layers,
    device,
    output_dim_vec=0,
    radius=None,
    num_ori=20,
    basis_dim=None,
    degree=3,
    widening_factor=4,
    layer_scale=None,
    task_level="graph",
    multiple_readouts=True,
    lift_graph=True,
    aggregate_fn="sum",
    **kwargs
):
    # Select either FiberBundle mode or PointCloud mode
    PonitaClass = PonitaFiberBundle if (num_ori > 0) else PonitaPointCloud
    # Return the ponita object
    return PonitaClass(
        input_dim,
        hidden_dim,
        output_dim,
        num_layers,
        device=device,
        output_dim_vec=output_dim_vec,
        radius=radius,
        num_ori=num_ori,
        basis_dim=basis_dim,
        degree=degree,
        widening_factor=widening_factor,
        layer_scale=layer_scale,
        task_level=task_level,
        multiple_readouts=multiple_readouts,
        lift_graph=lift_graph,
        aggregate_fn=aggregate_fn,
        **kwargs
    )


class PonitaFiberBundle(nn.Module):
    """Steerable E(3) equivariant (non-linear) convolutional network"""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        num_layers,
        batch_size,
        device,
        robot="h1",
        output_dim_vec=0,
        radius=None,
        num_ori=10,
        basis_dim=None,
        degree=3,
        widening_factor=4,
        layer_scale=None,
        task_level="graph",
        multiple_readouts=True,
        aggregate_fn="mean",
        **kwargs
    ):
        super().__init__()

        # Input output settings
        self.output_dim, self.output_dim_vec = output_dim, output_dim_vec
        self.global_pooling = task_level == "graph"
        self.aggregate_fn = aggregate_fn

        # For constructing the position-orientation graph and its invariants
        self.transform = Compose(
            [PositionOrientationGraph(num_ori), SEnInvariantAttributes(separable=True)]
        )

        self.batch_size = batch_size
        self.robot = robot
        self.device = device

        # Activation function to use internally
        act_fn = torch.nn.GELU()
        self.tanh = torch.nn.Tanh().to(device)  # Reusable activation

        # Kernel basis functions and spatial window
        basis_dim = hidden_dim if (basis_dim is None) else basis_dim
        self.basis_fn = nn.Sequential(
            PolynomialFeatures(degree),
            nn.LazyLinear(hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, basis_dim),
            act_fn,
        )
        self.fiber_basis_fn = nn.Sequential(
            PolynomialFeatures(degree),
            nn.LazyLinear(hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, basis_dim),
            act_fn,
        )
        self.windowing_fn = PolynomialCutoff(radius)
        
        # Batch device transfer
        self.basis_fn.to(device)
        self.fiber_basis_fn.to(device)
        self.windowing_fn.to(device)
        
        # Move transform to device if it has a .to() method
        if hasattr(self.transform, 'to'):
            self.transform.to(device)

        # Initial node embedding
        self.x_embedder = nn.Linear(input_dim, hidden_dim, False).to(device)

        # Make feedforward network
        self.interaction_layers = nn.ModuleList()
        self.read_out_layers = nn.ModuleList()
        for i in range(num_layers):
            conv = FiberBundleConv(
                hidden_dim, hidden_dim, basis_dim, groups=hidden_dim, separable=True, device=device
            )
            layer = ConvNext(
                hidden_dim,
                conv,
                device=device,
                act=act_fn,
                layer_scale=layer_scale,
                widening_factor=widening_factor,
            )
            self.interaction_layers.append(layer)
            # self.interaction_layers.append(ConvNextR3S2(hidden_dim, basis_dim, act=act_fn, widening_factor=widening_factor, layer_scale=layer_scale))
            if multiple_readouts or i == (num_layers - 1):
                self.read_out_layers.append(
                    nn.Linear(hidden_dim, output_dim + output_dim_vec).to(device)
                )
            else:
                self.read_out_layers.append(None)

        edge_index, node_attr, num_nodes, num_edges = build_edge_index_and_node_attr(
            self.robot, self.batch_size, self.device
        )
        self.edge_index = edge_index
        self.node_attr = node_attr
        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.batch = torch.repeat_interleave(torch.arange(batch_size, device=self.device), 19)
        
        # Cache for common batch sizes (optional optimization)
        self._batch_cache = {}

    def forward(self, graph):
        # Lift and compute invariants
        graph = self.transform(graph)

        # Sample the kernel basis and window the spatial kernel with a smooth cut-off
        kernel_basis = self.basis_fn(graph.attr) * self.windowing_fn(
            graph.dists
        ).unsqueeze(-2)
        fiber_kernel_basis = self.fiber_basis_fn(graph.fiber_attr)

        # Initial feature embeding
        x = self.x_embedder(graph.x)

        # Interaction + readout layers
        readouts = []
        for interaction_layer, readout_layer in zip(
            self.interaction_layers, self.read_out_layers
        ):
            x = interaction_layer(
                x,
                graph.edge_index,
                edge_attr=kernel_basis,
                fiber_attr=fiber_kernel_basis,
                batch=graph.batch,
            )
            if readout_layer is not None:
                readouts.append(readout_layer(x))
        readout = sum(readouts) / len(readouts)

        # Read out the scalar and vector part of the output
        readout_scalar, readout_vec = torch.split(
            readout, [self.output_dim, self.output_dim_vec], dim=-1
        )

        # Read out scalar and vectoyr predictions
        output_scalar = self.scalar_readout_fn(readout_scalar, graph.batch)
        # output_vector = self.vec_readout_fn(readout_vec, graph.ori_grid, graph.batch)

        # Return predictions
        return output_scalar

    def scalar_readout_fn(self, readout_scalar, batch):
        if self.output_dim > 0:
            output_scalar = sphere_to_scalar(readout_scalar)
            if self.global_pooling:
                output_scalar = self.custom_global_add_pool(output_scalar)
        else:
            output_scalar = None
        return self.tanh(output_scalar) if output_scalar is not None else None

    def vec_readout_fn(self, readout_vec, ori_grid, batch):
        if self.output_dim_vec > 0:
            output_vector = sphere_to_vec(readout_vec, ori_grid)
            if self.global_pooling:
                output_vector = self.custom_global_add_pool(output_vector)
        else:
            output_vector = None
        return self.tanh(output_vector) if output_vector is not None else None

    def build_batched_ponita_input(self, obs: torch.Tensor, xpos: torch.Tensor):
        batch_size = obs.shape[0]

        if batch_size == self.batch_size:
            edge_index = self.edge_index
            node_attr = self.node_attr
            batch = self.batch
        else:
            assert (
                batch_size <= self.batch_size
            ), "Batch size exceeds the maximum batch size."
            
            # Use cache for common batch sizes
            if batch_size in self._batch_cache:
                edge_index, node_attr, batch = self._batch_cache[batch_size]
            else:
                # Use slicing instead of list comprehension + clone for better performance
                num_edges_total = batch_size * self.num_edges
                edge_index = torch.stack([
                    self.edge_index[0][:num_edges_total], 
                    self.edge_index[1][:num_edges_total]
                ], dim=0)
                node_attr = self.node_attr[:batch_size * self.num_nodes]
                batch = torch.repeat_interleave(torch.arange(batch_size, device=self.device), 19)
                
                # Cache if batch size is reasonable (avoid memory bloat)
                if len(self._batch_cache) < 10:  # Limit cache size
                    self._batch_cache[batch_size] = (edge_index, node_attr, batch)

        # Flatten node positions (B*N, 3) - already efficient
        x = xpos[:, 1:].reshape(-1, 3)

        if self.robot == "h1":
            h = torch.stack(
                [obs[:, 32:].reshape(-1, 1), obs[:, 7:26].reshape(-1, 1), node_attr], dim=1
            ).squeeze(2)  # (B*N, 3)
        elif self.robot == "g1":
            h = torch.stack(
                [obs[:, 50:].reshape(-1, 1), obs[:, 7:44].reshape(-1, 1), node_attr], dim=1
            ).squeeze(2)  # (B*N, 3)

        return h, x, edge_index, batch

    def custom_global_add_pool(self, x: torch.Tensor) -> torch.Tensor:
        if self.aggregate_fn == "mean":
            x = x.mean(dim=1)  # shape: [batch_size, feature_dim]
        elif self.aggregate_fn == "sum":
            x = x.sum(dim=1)   # shape: [batch_size, feature_dim]
        else:
            raise ValueError(f"Unknown aggregation function: {self.aggregate_fn}")
        
        # Make sure self.num_nodes is correctly defined
        current_batch_size = int(x.shape[0] // self.num_nodes)
        assert x.shape[0] == current_batch_size * self.num_nodes

        x_reshaped = x.view(current_batch_size, self.num_nodes)

        return x_reshaped


class PonitaPointCloud(nn.Module):
    """Steerable E(3) equivariant (non-linear) convolutional network"""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        num_layers,
        device,
        batch_size,
        robot="h1",
        output_dim_vec=0,
        radius=None,
        num_ori=-1,
        basis_dim=None,
        degree=3,
        widening_factor=4,
        layer_scale=None,
        task_level="graph",
        multiple_readouts=False,
        lift_graph=False,
        **kwargs
    ):
        super().__init__()

        # Store device
        self.device = device
        self.batch_size = batch_size
        self.robot = robot

        # Input output settings
        self.output_dim, self.output_dim_vec = output_dim, output_dim_vec
        self.global_pooling = task_level == "graph"

        # For constructing the position-orientation graph and its invariants
        self.lift_graph = lift_graph
        if lift_graph:
            self.transform = Compose(
                [
                    PositionOrientationGraph(num_ori, radius),
                    SEnInvariantAttributes(separable=False, point_cloud=True),
                ]
            )

        # Activation function to use internally
        act_fn = torch.nn.GELU()

        # Kernel basis functions and spatial window
        basis_dim = hidden_dim if (basis_dim is None) else basis_dim
        self.basis_fn = nn.Sequential(
            PolynomialFeatures(degree),
            nn.LazyLinear(hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, basis_dim),
            act_fn,
        )
        self.windowing_fn = PolynomialCutoff(radius)
        
        # Batch device transfer
        self.basis_fn.to(device)
        self.windowing_fn.to(device)

        # Initial node embedding
        self.x_embedder = nn.Linear(input_dim, hidden_dim, False).to(device)

        # Make feedforward network
        self.interaction_layers = nn.ModuleList()
        self.read_out_layers = nn.ModuleList()
        for i in range(num_layers):
            conv = Conv(hidden_dim, hidden_dim, basis_dim, groups=hidden_dim, device=device)
            layer = ConvNext(
                hidden_dim,
                conv,
                device=device,
                act=act_fn,
                layer_scale=layer_scale,
                widening_factor=widening_factor,
            )
            self.interaction_layers.append(layer)
            if multiple_readouts or i == (num_layers - 1):
                self.read_out_layers.append(
                    nn.Linear(hidden_dim, output_dim + output_dim_vec).to(device)
                )
            else:
                self.read_out_layers.append(None)

        edge_index, node_attr, num_nodes, num_edges = build_edge_index_and_node_attr(
            self.robot, self.batch_size, self.device
        )
        self.edge_index = edge_index
        self.node_attr = node_attr
        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.batch = torch.repeat_interleave(torch.arange(batch_size, device=self.device), 19)
        
        # Cache for common batch sizes (optional optimization)
        self._batch_cache = {}

    def forward(self, graph):

        # Lift and compute invariants
        if self.lift_graph:
            graph = self.transform(graph)

        # Sample the kernel basis and window the spatial kernel with a smooth cut-off
        kernel_basis = self.basis_fn(graph.attr) * self.windowing_fn(graph.dists)

        # Initial feature embeding
        x = self.x_embedder(graph.x)

        # Interaction + readout layers
        readouts = []
        for interaction_layer, readout_layer in zip(
            self.interaction_layers, self.read_out_layers
        ):
            x = interaction_layer(
                x, graph.edge_index, edge_attr=kernel_basis, batch=graph.batch
            )
            if readout_layer is not None:
                readouts.append(readout_layer(x))
        readout = sum(readouts) / len(readouts)

        # Read out the scalar and vector part of the output
        readout_scalar, readout_vec = torch.split(
            readout, [self.output_dim, self.output_dim_vec], dim=-1
        )

        # Read out scalar and vector predictions (if pos-ori cloud collect all predictions that have the same base point in R^n)
        if hasattr(graph, "scatter_projection_index"):
            output_scalar = self.scalar_readout_fn(
                readout_scalar, graph.batch, graph.scatter_projection_index
            )
            output_vector = self.vec_readout_fn(
                readout_vec, graph.pos, graph.batch, graph.scatter_projection_index
            )
        else:
            output_scalar = readout_scalar
            if self.global_pooling:
                output_scalar = global_add_pool(output_scalar, graph.batch)
            output_vector = None

        # Return predictions
        return output_scalar

    def scalar_readout_fn(self, readout_scalar, batch, scatter_projection_index):
        if self.output_dim > 0:
            # Aggregate predictions toward the base position in R^n
            output_scalar = scatter_mean(
                readout_scalar, scatter_projection_index, dim=0
            )
            if self.global_pooling:
                batch_Rn = scatter_mean(batch, scatter_projection_index, dim=0).type_as(
                    batch
                )
                output_scalar = global_add_pool(output_scalar, batch_Rn)
        else:
            output_scalar = None
        return output_scalar

    def vec_readout_fn(self, readout_vec, pos, batch, scatter_projection_index):
        if self.output_dim_vec > 0:
            # Scale each orientation with the predicted scalar and aggregate via scatter_mean
            _, ori = pos.split(int(pos.shape[-1] / 2), dim=-1)
            output_vector = scatter_mean(
                readout_vec[:, :, None] * ori[:, None, :],
                scatter_projection_index,
                dim=0,
            )
            if self.global_pooling:
                batch_Rn = scatter_mean(batch, scatter_projection_index, dim=0).type_as(
                    batch
                )
                output_vector = global_add_pool(output_vector, batch_Rn)
        else:
            output_vector = None
        return output_vector

    def custom_global_add_pool(self, x: torch.Tensor) -> torch.Tensor:
        if self.aggregate_fn == "mean":
            x = x.mean(dim=1)  # shape: [batch_size, feature_dim]
        elif self.aggregate_fn == "sum":
            x = x.sum(dim=1)   # shape: [batch_size, feature_dim]
        else:
            raise ValueError(f"Unknown aggregation function: {self.aggregate_fn}")
        
        # Make sure self.num_nodes is correctly defined
        current_batch_size = int(x.shape[0] // self.num_nodes)
        assert x.shape[0] == current_batch_size * self.num_nodes

        x_reshaped = x.view(current_batch_size, self.num_nodes)

        return x_reshaped

    def build_batched_ponita_input(self, obs: torch.Tensor, xpos: torch.Tensor):
        batch_size = obs.shape[0]

        if batch_size == self.batch_size:
            edge_index = self.edge_index
            node_attr = self.node_attr
            batch = self.batch
        else:
            assert (
                batch_size <= self.batch_size
            ), "Batch size exceeds the maximum batch size."
            
            # Use cache for common batch sizes
            if batch_size in self._batch_cache:
                edge_index, node_attr, batch = self._batch_cache[batch_size]
            else:
                # Use slicing instead of list comprehension + clone for better performance
                num_edges_total = batch_size * self.num_edges
                edge_index = torch.stack([
                    self.edge_index[0][:num_edges_total], 
                    self.edge_index[1][:num_edges_total]
                ], dim=0)
                node_attr = self.node_attr[:batch_size * self.num_nodes]
                batch = torch.repeat_interleave(torch.arange(batch_size, device=self.device), 19)
                
                # Cache if batch size is reasonable (avoid memory bloat)
                if len(self._batch_cache) < 10:  # Limit cache size
                    self._batch_cache[batch_size] = (edge_index, node_attr, batch)

        # Flatten node positions (B*N, 3) - already efficient
        x = xpos[:, 1:].reshape(-1, 3)

        if self.robot == "h1":
            h = torch.stack(
                [obs[:, 32:].reshape(-1, 1), obs[:, 7:26].reshape(-1, 1), node_attr], dim=1
            ).squeeze(2)  # (B*N, 3)
        elif self.robot == "g1":
            h = torch.stack(
                [obs[:, 50:].reshape(-1, 1), obs[:, 7:44].reshape(-1, 1), node_attr], dim=1
            ).squeeze(2)  # (B*N, 3)

        return h, x, edge_index, batch

