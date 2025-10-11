"""
Type-Aware EGNN: Enhanced heterogeneous graph implementation
This version REQUIRES heterogeneous graphs with two node types (joint and object).
Uses separate MLPs for different node/edge types for better representation learning.
"""

from torch import nn
import torch
from fast_td3.robots.graph_builder import GraphBuilder
from torch_scatter import scatter_sum, scatter_mean


env_with_object = [
    "h1-push-v0",
    "h1-basketball-v0",
    "h1-package-v0",
    "h1-sit_hard-v0",
    "h1-balance_simple-v0",
]


class TypeAwareE_GCL(nn.Module):
    """
    Type-Aware E(n) Equivariant Convolutional Layer
    Uses separate MLPs for different node/edge types for better representation learning.
    Requires edge_attr and node_attr to be provided (no fallback to homogeneous).
    """

    def __init__(
        self,
        input_nf,
        output_nf,
        hidden_nf,
        edges_in_d,
        act_fn=nn.SiLU(),
        residual=True,
        attention=False,
        normalize=False,
        coords_agg="mean",
        tanh=False,
        edge_coords_nf=1,
    ):
        super(TypeAwareE_GCL, self).__init__()
        input_edge = input_nf * 2
        self.residual = residual
        self.attention = attention
        self.normalize = normalize
        self.coords_agg = coords_agg
        self.tanh = tanh
        self.epsilon = 1e-8

        # Separate edge MLPs for joint-to-joint and joint-to-object
        self.edge_mlp_joint = nn.Sequential(
            nn.Linear(input_edge + edge_coords_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
        )
        self.edge_mlp_object = nn.Sequential(
            nn.Linear(input_edge + edge_coords_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
        )

        # Separate node MLPs for joint and object nodes
        self.node_mlp_joint = nn.Sequential(
            nn.Linear(hidden_nf + input_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf),
        )
        self.node_mlp_object = nn.Sequential(
            nn.Linear(hidden_nf + input_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf),
        )

        # Separate coord MLPs
        self.coord_mlp_joint = self._build_coord_mlp(hidden_nf, act_fn, tanh)
        self.coord_mlp_object = self._build_coord_mlp(hidden_nf, act_fn, tanh)

        if self.attention:
            self.att_mlp = nn.Sequential(nn.Linear(hidden_nf, 1), nn.Sigmoid())

    def _build_coord_mlp(self, hidden_nf, act_fn, tanh):
        """Helper to build coordinate update MLP"""
        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

        coord_mlp = []
        coord_mlp.append(nn.Linear(hidden_nf, hidden_nf))
        coord_mlp.append(act_fn)
        coord_mlp.append(layer)

        if tanh:
            coord_mlp.append(nn.Tanh())
        return nn.Sequential(*coord_mlp)

    def coord2radial(self, edge_index, coord):
        """Compute squared distance and coordinate differences"""
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.sum(coord_diff**2, 1).unsqueeze(1)

        if self.normalize:
            norm = torch.sqrt(radial).detach() + self.epsilon
            coord_diff = coord_diff / norm

        return radial, coord_diff

    def edge_model(self, source, target, radial, edge_attr):
        # Process edges with separate MLPs based on edge type (from edge_attr)
        inp = torch.cat([source, target, radial], dim=1)
        
        # edge_attr: 0 = joint-to-joint edge, 1 = joint-to-object edge
        joint_edge_mask = (edge_attr.squeeze(-1) == 0)
        object_edge_mask = (edge_attr.squeeze(-1) == 1)
        
        # Process both edge types
        joint_out = self.edge_mlp_joint(inp[joint_edge_mask])
        object_out = self.edge_mlp_object(inp[object_edge_mask])
        
        # Combine outputs
        out = torch.empty(
            source.shape[0], 
            self.edge_mlp_joint[-2].out_features,
            device=source.device,
            dtype=source.dtype
        )
        out[joint_edge_mask] = joint_out
        out[object_edge_mask] = object_out

        if self.attention:
            att_val = self.att_mlp(out)
            out = out * att_val

        return out

    def coord_model(self, coord, edge_index, coord_diff, edge_feat, edge_attr):
        """
        Type-aware coordinate update.
        Uses different update rules for joint vs object edges.
        Requires edge_attr to distinguish edge types.
        """
        
        row, col = edge_index
        
        # edge_attr: 0 = joint-to-joint edge, 1 = joint-to-object edge
        joint_edge_mask = (edge_attr.squeeze(-1) == 0)
        object_edge_mask = (edge_attr.squeeze(-1) == 1)
    
        # Compute transformations for both edge types
        trans_joint = coord_diff[joint_edge_mask] * self.coord_mlp_joint(edge_feat[joint_edge_mask])
        trans_object = coord_diff[object_edge_mask] * self.coord_mlp_object(edge_feat[object_edge_mask])
        
        # Combine transformations
        trans = torch.empty_like(coord_diff)
        trans[joint_edge_mask] = trans_joint
        trans[object_edge_mask] = trans_object

        # Aggregate coordinate updates
        if self.coords_agg == 'sum':
            agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0))
        elif self.coords_agg == 'mean':
            agg = unsorted_segment_mean(trans, row, num_segments=coord.size(0))
        else:
            raise Exception(f'Wrong coords_agg parameter: {self.coords_agg}')

        coord = coord + agg.clamp(-10, 10)
        return coord

    def node_model(self, x, edge_index, edge_feat, node_attr):
        """
        Type-aware node feature update.
        Uses different MLPs for joint and object nodes.
        Requires node_attr to distinguish node types.
        """
        
        row, col = edge_index
        agg = unsorted_segment_sum(edge_feat, row, num_segments=x.size(0))

        # Concatenate node features with aggregated edge features
        agg_input = torch.cat([x, agg], dim=1)
        
        # node_attr: 0 = joint node, 1 = object node
        joint_node_mask = (node_attr.squeeze(-1) == 0)
        object_node_mask = (node_attr.squeeze(-1) == 1)
        
        # Process both node types
        joint_out = self.node_mlp_joint(agg_input[joint_node_mask])
        object_out = self.node_mlp_object(agg_input[object_node_mask])
        
        # Combine outputs
        out = torch.empty_like(x)
        out[joint_node_mask] = joint_out
        out[object_node_mask] = object_out

        if self.residual:
            out = x + out

        return out, agg_input

    def forward(self, h, edge_index, coord, edge_attr, node_attr):
        """
        Forward pass - requires both edge_attr and node_attr.
        """
        assert edge_attr is not None, "TypeAwareE_GCL requires edge_attr"
        assert node_attr is not None, "TypeAwareE_GCL requires node_attr"
        
        row, col = edge_index

        radial, coord_diff = self.coord2radial(edge_index, coord)

        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr)

        coord = self.coord_model(coord, edge_index, coord_diff, edge_feat, edge_attr)

        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)

        return h, coord, edge_attr


class TypeAwareEGNN(nn.Module):
    """
    Enhanced EGNN with type-aware message passing for heterogeneous graphs.
    """

    def __init__(
        self,
        in_node_nf,
        hidden_nf,
        out_node_nf,
        in_edge_nf,
        device,
        batch_size,
        act_fn,
        n_layers,
        robot,
        env_name,
        object_node_nf,  # Now required, not optional
        residual=True,
        attention=False,
        normalize=False,
        tanh=False,
        coords_agg="mean",
    ):
        super(TypeAwareEGNN, self).__init__()
        
        assert object_node_nf is not None, "TypeAwareEGNN requires object_node_nf (use standard EGNN for homogeneous graphs)"
        assert env_name in env_with_object, f"TypeAwareEGNN requires object-centric environment, got {env_name}"
        
        self.in_edge_nf = in_edge_nf
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.out_node_nf = out_node_nf

        # Separate embedding layers for different node types
        self.joint_embedding_in = nn.Sequential(
            nn.Linear(in_node_nf, self.hidden_nf), act_fn
        )
        self.object_embedding_in = nn.Sequential(
            nn.Linear(object_node_nf, self.hidden_nf), act_fn
        )
        
        # Separate output heads for different node types
        self.joint_embedding_out = nn.Sequential(
            nn.Linear(self.hidden_nf, out_node_nf),
            nn.Tanh(),
        )

        self.batch_size = batch_size

        # Use type-aware layers (always with type-specific MLPs)
        self.layers = nn.ModuleList(
            [
                TypeAwareE_GCL(
                    self.hidden_nf,
                    self.hidden_nf,
                    self.hidden_nf,
                    edges_in_d=in_edge_nf,
                    act_fn=act_fn,
                    residual=residual,
                    attention=attention,
                    normalize=normalize,
                    tanh=tanh,
                    coords_agg=coords_agg,
                    edge_coords_nf=1,
                )
                for _ in range(n_layers)
            ]
        )

        self.to(self.device)

        self.graph_builder = GraphBuilder(env_name, batch_size, device, robot)
        self.robot = robot
        self.env_name = env_name

        # Pre-compute indices
        self.edge_index, self.edge_attr, self.node_attr = self.generate_index(
            batch_size, device
        )
        self._edge_cache = {}

        common_batch_sizes = [1, 4, 16, 128, 8192]
        self._precomputed_edges = {}
        for bs in common_batch_sizes:
            if bs <= batch_size:
                try:
                    self._precomputed_edges[bs] = self.generate_index(bs, device)
                except RuntimeError:
                    break

        # Always precompute mask indices for heterogeneous graphs
        self._precompute_mask_indices()

    def generate_index(self, batch_size: int, device="cuda"):
        """
        Generate edge indices, edge attributes, and node attributes for given batch size.
        Always assumes heterogeneous graph with object nodes.
        """
        assert self.env_name in env_with_object, f"TypeAwareEGNN requires object-centric env, got {self.env_name}"
        
        object_node_id = self.graph_builder.robot.OBJECT.free_object
        src, dst = zip(*self.graph_builder.robot.joint_connections_with_object)
        node_attr = [0] * (len(self.graph_builder.robot.JOINT)) + [1]  # last node is object

        edge_attr = []
        for s, d in zip(src, dst):
            if s == object_node_id or d == object_node_id:
                edge_attr.append(1)
            else:
                edge_attr.append(0)

        src = torch.tensor(src, dtype=torch.long, device=device)
        dst = torch.tensor(dst, dtype=torch.long, device=device)
        edge_attr = torch.tensor(edge_attr, dtype=torch.float, device=device)
        node_attr = torch.tensor(node_attr, dtype=torch.float, device=device)

        offsets = torch.arange(batch_size, device=device) * (
            len(self.graph_builder.robot.JOINT)
            + (1 if self.env_name in env_with_object else 0)
        )
        src_batch = (src.unsqueeze(0) + offsets.unsqueeze(1)).flatten().to(device)
        dst_batch = (dst.unsqueeze(0) + offsets.unsqueeze(1)).flatten().to(device)
        edge_attr_batch = edge_attr.repeat(batch_size).to(device).unsqueeze(-1)
        node_attr_batch = node_attr.repeat(batch_size).to(device).unsqueeze(-1)

        return torch.stack([src_batch, dst_batch]), edge_attr_batch, node_attr_batch

    def _get_cached_edges(self, current_batch_size: int):
        if current_batch_size in self._precomputed_edges:
            return self._precomputed_edges[current_batch_size]
        if current_batch_size in self._edge_cache:
            return self._edge_cache[current_batch_size]
        edge_data = self.generate_index(current_batch_size, self.device)
        self._edge_cache[current_batch_size] = edge_data
        return edge_data

    def _precompute_mask_indices(self):
        """Pre-compute joint and object node indices for efficient gathering."""
        num_joints = len(self.graph_builder.robot.JOINT)
        num_nodes_per_batch = num_joints + 1  # Always has 1 object node

        node_attr_template = torch.tensor(
            [0] * num_joints + [1],  # Last node is object
            dtype=torch.float,
            device=self.device
        )

        self.joint_indices_per_batch = torch.where(node_attr_template == 0)[0]
        self.object_indices_per_batch = torch.where(node_attr_template == 1)[0]
        self.num_nodes_per_batch = num_nodes_per_batch
        self._mask_index_cache = {}

    def _get_mask_indices(self, batch_size: int):
        if batch_size in self._mask_index_cache:
            return self._mask_index_cache[batch_size]

        batch_offsets = torch.arange(
            batch_size, device=self.device
        ) * self.num_nodes_per_batch

        joint_indices = (
            self.joint_indices_per_batch.unsqueeze(0) + batch_offsets.unsqueeze(1)
        ).flatten()

        object_indices = (
            self.object_indices_per_batch.unsqueeze(0) + batch_offsets.unsqueeze(1)
        ).flatten()

        self._mask_index_cache[batch_size] = (joint_indices, object_indices)
        return joint_indices, object_indices

    def forward(self, obs: torch.Tensor, xpos: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for heterogeneous graph with joint and object nodes.
        """
        current_batch_size = obs.shape[0]

        # Get cached edge data (always includes edge_attr and node_attr)
        edges, edge_attr, node_attr = self._get_cached_edges(current_batch_size)

        # Generate input features for joint and object nodes
        h_joint, h_object, x = self.graph_builder.generate_input_for_mixed_type(
            obs, xpos
        )

        # Embed joint and object features
        h_joint_embedded = self.joint_embedding_in(h_joint)
        h_object_embedded = self.object_embedding_in(h_object)

        # Get pre-computed indices for fast gathering
        joint_indices, object_indices = self._get_mask_indices(current_batch_size)

        # Allocate h with correct shape
        h = torch.empty(
            current_batch_size * self.num_nodes_per_batch,
            self.hidden_nf,
            device=self.device,
            dtype=h_joint_embedded.dtype,
        )

        # Assign joint and object embeddings
        h.index_copy_(0, joint_indices, h_joint_embedded)
        h.index_copy_(0, object_indices, h_object_embedded)

        # Message passing with type-aware layers
        for layer in self.layers:
            h, x, _ = layer(
                h=h, edge_index=edges, coord=x, edge_attr=edge_attr, node_attr=node_attr
            )

        # Process output with separate heads
        joint_indices, object_indices = self._get_mask_indices(current_batch_size)

        h_joint_out = self.joint_embedding_out(h[joint_indices])

        num_joints = len(self.joint_indices_per_batch)

        h_joint_out = h_joint_out.view(current_batch_size, num_joints)
        return h_joint_out


def unsorted_segment_sum(data, segment_ids, num_segments):
    return scatter_sum(data, segment_ids, dim=0, dim_size=num_segments)


def unsorted_segment_mean(data, segment_ids, num_segments):
    return scatter_mean(data, segment_ids, dim=0, dim_size=num_segments)
