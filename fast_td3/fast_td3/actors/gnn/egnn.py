from torch import nn
import torch

from fast_td3.robots.graph_builder import GraphBuilder

# Environment classification for object inclusion
env_with_object = [
    "h1-push-v0",  # medium
    "h1-basketball-v0",  # very hard
    "h1-package-v0",  # medium
    "h1-sit_hard-v0",  # hard
    "h1-balance_simple-v0",  # hard
]

env_without_object = [
    "h1-walk-v0",
    "h1-reach-v0",
    "h1-hurdle-v0",
    "h1-crawl-v0",
    "h1-maze-v0",
    "h1-highbar_simple-v0",
    "h1-stand-v0",
    "h1-run-v0",
    "h1-sit_simple-v0",
    "h1-stairs-v0",
    "h1-slide-v0",
    "h1-pole-v0",
]


class E_GCL(nn.Module):
    """
    E(n) Equivariant Convolutional Layer

    Mathematical operations:
    1. Compute squared distance: d_{ij}^2 = ||x_i - x_j||^2 (rotation/translation invariant)
    2. Edge message: m_{ij} = φ_e(h_i, h_j, d_{ij}^2, a_{ij})
    3. Coordinate update: x_i^{l+1} = x_i^l + Σ_{j∈N(i)} (x_i - x_j) * φ_x(m_{ij})
    4. Feature update: h_i^{l+1} = φ_h(h_i, Σ_{j∈N(i)} m_{ij})
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
        super(E_GCL, self).__init__()
        input_edge = input_nf * 2
        self.residual = residual
        self.attention = attention
        self.normalize = normalize
        self.coords_agg = coords_agg
        self.tanh = tanh
        self.epsilon = 1e-8

        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edge_coords_nf + edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf + edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf),
        )

        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

        coord_mlp = []
        coord_mlp.append(nn.Linear(hidden_nf, hidden_nf))
        coord_mlp.append(act_fn)
        coord_mlp.append(layer)

        if self.tanh:
            coord_mlp.append(nn.Tanh())
        self.coord_mlp = nn.Sequential(*coord_mlp)

        if self.attention:
            self.att_mlp = nn.Sequential(nn.Linear(hidden_nf, 1), nn.Sigmoid())

    def coord2radial(self, edge_index, coord):
        """
        Step 1: Compute squared distance d_{ij}^2 = ||x_i - x_j||^2
        This is rotation and translation equivariant.
        Also computes coordinate differences (x_i - x_j) for equivariant updates.
        """
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.sum(coord_diff**2, 1).unsqueeze(1)

        if self.normalize:
            norm = torch.sqrt(radial).detach() + self.epsilon
            coord_diff = coord_diff / norm

        return radial, coord_diff

    def edge_model(self, source, target, radial, edge_attr):
        """
        Step 2: Compute edge message m_{ij} = φ_e(h_i, h_j, d_{ij}^2, a_{ij}).
        Combines source node features, target node features, radial distance, and edge attributes.
        """
        if edge_attr is None:
            out = torch.cat([source, target, radial], dim=1)
        else:
            out = torch.cat([source, target, radial, edge_attr], dim=1)
        out = self.edge_mlp(out)
        if self.attention:
            att_val = self.att_mlp(out)
            out = out * att_val
        return out

    def coord_model(self, coord, edge_index, coord_diff, edge_feat):
        """
        Step 3: Coordinate update x_i^{l+1} = x_i^l + Σ_{j∈N(i)} (x_i - x_j) * φ_x(m_{ij}).
        Updates coordinates using direction vectors (x_i - x_j) weighted by scalar φ_x(m_{ij}).
        This ensures rotation equivariance: if x -> Rx, then x^{l+1} -> Rx^{l+1}.
        """
        row, col = edge_index
        trans = coord_diff * self.coord_mlp(edge_feat)
        if self.coords_agg == 'sum':
            agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0))
        elif self.coords_agg == 'mean':
            agg = unsorted_segment_mean(trans, row, num_segments=coord.size(0))
        else:
            raise Exception('Wrong coords_agg parameter' % self.coords_agg)
        coord = coord + agg.clamp(-10, 10)
        return coord

    def node_model(self, x, edge_index, edge_attr, node_attr):
        """
        Step 4: Feature update h_i^{l+1} = φ_h(h_i, Σ_{j∈N(i)} m_{ij}).
        Aggregates edge messages and updates node features.
        """
        row, col = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0))
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)
        out = self.node_mlp(agg)
        if self.residual:
            out = x + out
        return out, agg

    def forward(self, h, edge_index, coord, edge_attr=None, node_attr=None):
        row, col = edge_index

        radial, coord_diff = self.coord2radial(edge_index, coord)

        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr)

        coord = self.coord_model(coord, edge_index, coord_diff, edge_feat)

        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)

        return h, coord, edge_attr


class EGNN(nn.Module):
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
        residual=True,
        attention=False,
        normalize=False,
        tanh=False,
        coords_agg="mean",
        object_node_nf=None,
    ):
        """
        :param in_node_nf: Number of features for 'h' at the input (backward compatibility)
        :param object_node_nf: Number of features for object nodes (if different from in_node_nf)
        :param hidden_nf: Number of hidden features
        :param out_node_nf: Number of features for 'h' at the output
        :param in_edge_nf: Number of features for the edge features
        :param device: Device (e.g. 'cpu', 'cuda:0',...)
        :param act_fn: Non-linearity
        :param n_layers: Number of layer for the EGNN
        :param residual: Use residual connections, we recommend not changing this one
        :param attention: Whether using attention or not
        :param normalize: Normalizes the coordinates messages such that:
                    instead of: x^{l+1}_i = x^{l}_i + Σ(x_i - x_j)phi_x(m_ij)
                    we get:     x^{l+1}_i = x^{l}_i + Σ(x_i - x_j)phi_x(m_ij)/||x_i - x_j||
                    We noticed it may help in the stability or generalization in some future works.
                    We didn't use it in our paper.
        :param tanh: Sets a tanh activation function at the output of phi_x(m_ij). I.e. it bounds the output of
                        phi_x(m_ij) which definitely improves in stability but it may decrease in accuracy.
                        We didn't use it in our paper.
        """

        super(EGNN, self).__init__()
        self.in_edge_nf = in_edge_nf
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.out_node_nf = out_node_nf

        self.has_mixed_node_types = object_node_nf is not None

        if self.has_mixed_node_types:
            # Separate embedding layers for different node types
            self.joint_embedding_in = nn.Sequential(
                nn.Linear(in_node_nf, self.hidden_nf), act_fn
            )
            self.object_embedding_in = nn.Sequential(
                nn.Linear(object_node_nf, self.hidden_nf), act_fn
            )
        else:
            # Single embedding layer for backward compatibility
            self.embedding_in = nn.Sequential(
                nn.Linear(in_node_nf, self.hidden_nf), act_fn
            )

        self.embedding_out = nn.Sequential(
            nn.Linear(self.hidden_nf, out_node_nf),
            nn.Tanh(),
        )
        self.batch_size = batch_size
        # Use ModuleList for fast iteration and to avoid dict lookups
        self.layers = nn.ModuleList(
            [
                E_GCL(
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
        # Ensure all parameters and submodules are on the requested device
        self.to(self.device)

        # no need to learn graph builder
        self.graph_builder = GraphBuilder(env_name, batch_size, device, robot)
        self.robot = robot
        self.env_name = env_name

        # Move caching from GraphBuilder to EGNN
        self.edge_index, self.edge_attr, self.node_attr = self.generate_index(
            batch_size, device
        )
        self._edge_cache = {}
        self.num_edges = self.graph_builder.robot.joint_connections.__len__()

        # Pre-compute edge indices for common batch sizes to avoid repeated computation
        common_batch_sizes = [1, 4, 16, 128, 8192]
        self._precomputed_edges = {}
        for bs in common_batch_sizes:
            if bs <= batch_size:
                try:
                    self._precomputed_edges[bs] = self.generate_index(bs, device)
                except RuntimeError:
                    # Skip if out of memory for very large batch sizes
                    break

        # Pre-compute mask indices for mixed node types (huge speedup)
        if self.has_mixed_node_types:
            self._precompute_mask_indices()

    def generate_index(self, batch_size: int, device="cuda"):
        """Generate edge indices, edge attributes, and node attributes for given batch size."""
        edge_attr = []
        node_attr = []

        if self.env_name in env_with_object:
            object_node_id = self.graph_builder.robot.OBJECT.free_object
            src, dst = zip(*self.graph_builder.robot.joint_connections_with_object)
            node_attr = [0] * (len(self.graph_builder.robot.JOINT)) + [
                1
            ]  # last node is object node

            # Create edge_attr: 1 if edge involves object_node_id, else 0
            for s, d in zip(src, dst):
                if s == object_node_id or d == object_node_id:
                    edge_attr.append(1)
                else:
                    edge_attr.append(0)
        else:
            src, dst = zip(*self.graph_builder.robot.joint_connections)
            node_attr = [0] * (len(self.graph_builder.robot.JOINT))  # All joint nodes
            edge_attr = [0] * len(src)  # All edges are joint-to-joint

        # Unpack edge list into two tuples
        src = torch.tensor(src, dtype=torch.long, device=device)
        dst = torch.tensor(dst, dtype=torch.long, device=device)
        edge_attr = torch.tensor(edge_attr, dtype=torch.float, device=device)
        node_attr = torch.tensor(node_attr, dtype=torch.float, device=device)

        # Create batch offsets and expand edges in one operation
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
        """Optimized method to get edge indices with pre-computed cache lookup."""
        # Check pre-computed edges first (fastest)
        if current_batch_size in self._precomputed_edges:
            return self._precomputed_edges[current_batch_size]

        # Check runtime cache
        if current_batch_size in self._edge_cache:
            return self._edge_cache[current_batch_size]

        # Generate and cache
        edge_data = self.generate_index(current_batch_size, self.device)
        self._edge_cache[current_batch_size] = edge_data
        return edge_data

    def _precompute_mask_indices(self):
        """Pre-compute joint and object node indices for efficient gathering."""
        num_joints = len(self.graph_builder.robot.JOINT)
        has_object = 1 if self.env_name in env_with_object else 0
        num_nodes_per_batch = num_joints + has_object
        
        # Create a template for node attributes per batch
        node_attr_template = torch.tensor(
            [0] * num_joints + [1] * has_object,
            dtype=torch.float,
            device=self.device
        )
        
        # Store indices where joints and objects are located within each batch
        self.joint_indices_per_batch = torch.where(node_attr_template == 0)[0]
        self.object_indices_per_batch = torch.where(node_attr_template == 1)[0]
        self.num_nodes_per_batch = num_nodes_per_batch
        self._mask_index_cache = {}

    def _get_mask_indices(self, batch_size: int):
        """Get pre-computed mask indices for given batch size."""
        if batch_size in self._mask_index_cache:
            return self._mask_index_cache[batch_size]
        
        # Generate indices for all batches
        batch_offsets = torch.arange(
            batch_size, device=self.device
        ) * self.num_nodes_per_batch
        
        # Expand joint and object indices for all batches
        joint_indices = (
            self.joint_indices_per_batch.unsqueeze(0) + batch_offsets.unsqueeze(1)
        ).flatten()
        
        object_indices = (
            self.object_indices_per_batch.unsqueeze(0) + batch_offsets.unsqueeze(1)
        ).flatten()
        
        self._mask_index_cache[batch_size] = (joint_indices, object_indices)
        return joint_indices, object_indices

    def forward(self, obs: torch.Tensor, xpos: torch.Tensor) -> torch.Tensor:
        current_batch_size = obs.shape[0]

        # Get cached edge data for current batch size
        edges, edge_attr, node_attr = self._get_cached_edges(current_batch_size)
        if not self.has_mixed_node_types:
            edge_attr = None
            node_attr = None

        if self.has_mixed_node_types:
            h_joint, h_object, x = self.graph_builder.generate_input_for_mixed_type(
                obs, xpos
            )

            h_joint_embedded = self.joint_embedding_in(h_joint)
            h_object_embedded = self.object_embedding_in(h_object)

            # Get pre-computed indices for fast gathering
            joint_indices, object_indices = self._get_mask_indices(current_batch_size)

            # Allocate h with correct shape - use empty instead of zeros (faster)
            h = torch.empty(
                current_batch_size * self.num_nodes_per_batch,
                self.hidden_nf,
                device=self.device,
                dtype=h_joint_embedded.dtype,
            )

            # Use index_copy_ for faster assignment than boolean indexing
            h.index_copy_(0, joint_indices, h_joint_embedded)
            h.index_copy_(0, object_indices, h_object_embedded)
        else:
            h, x = self.graph_builder.generate_input(obs, xpos)

            h = self.embedding_in(h)

        for layer in self.layers:
            # print(f"Layer {i} x min: {x.min()}, max: {x.max()}, mean: {x.mean()}")
            h, x, edge_attr = layer(
                h=h, edge_index=edges, coord=x, edge_attr=edge_attr, node_attr=node_attr
            )

        h = self.embedding_out(h)
        h = h.view(current_batch_size, h.shape[0] // current_batch_size)

        return h[:, :19]


@torch.compile(dynamic=True)
def unsorted_segment_sum(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    return result


@torch.compile(dynamic=True)
def unsorted_segment_mean(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    count = data.new_full(result_shape, 0)
    result.scatter_add_(0, segment_ids, data)
    count.scatter_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)
