from torch import nn
import torch
from torch_scatter import scatter_sum, scatter_mean

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

    def node_model(self, x, edge_index, edge_feat, node_attr):
        """
        Step 4: Feature update h_i^{l+1} = φ_h(h_i, Σ_{j∈N(i)} m_{ij}).
        Aggregates edge messages and updates node features.
        """
        row, col = edge_index
        agg = unsorted_segment_sum(edge_feat, row, num_segments=x.size(0))
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
    ):
        """
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
        self.batch_size = batch_size
        self.has_mixed_node_types = env_name in env_with_object
        self.robot = robot
        self.env_name = env_name
        self.graph_builder = GraphBuilder(env_name, batch_size, device, robot)
        self.num_joints = len(self.graph_builder.robot.JOINT)
        self.num_edges = len(self.graph_builder.robot.joint_connections)

        # EGNN layers for local joint-to-joint processing
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
    

        if self.has_mixed_node_types:
            self.joint_embedding_in = nn.Sequential(
                nn.LazyLinear(self.hidden_nf), act_fn
            )
            # Object MLP for local processing within object cluster
            self.object_mlp = nn.Sequential(
                nn.LazyLinear(self.hidden_nf),
                act_fn
            )

            self.global_aggregation = nn.Sequential(
                nn.Linear(self.hidden_nf * 2, self.hidden_nf * 4),  # concat [joint_feat, object_feat]
                act_fn,
                nn.Linear(self.hidden_nf * 4, self.hidden_nf * 4),
                act_fn,
                nn.Linear(self.hidden_nf * 4, self.out_node_nf),
            )

            self.skip_proj = nn.Linear(self.hidden_nf * 2, self.out_node_nf)
        else:
            # Single embedding layer for backward compatibility
            self.embedding_in = nn.Sequential(
                nn.LazyLinear(self.hidden_nf), act_fn
            )
            self.embedding_out = nn.Sequential(
                nn.Linear(self.hidden_nf, out_node_nf),
                nn.Tanh(),
            )
    
        self.to(self.device)


        # Initialize edge cache - will be populated dynamically as new batch sizes are encountered
        self._edge_cache = {}

    def generate_index(self, batch_size: int, device="cuda"):
        """
        Generate joint-to-joint edge indices for given batch size.
        Since we now process objects separately with MLP, we only need joint edges for EGNN.
        """
        # Always use joint-to-joint connections only
        src, dst = zip(*self.graph_builder.robot.joint_connections)
        
        # Convert to tensors
        src = torch.tensor(src, dtype=torch.long, device=device)
        dst = torch.tensor(dst, dtype=torch.long, device=device)
        edge_attr = torch.zeros(len(src), dtype=torch.float, device=device)
        num_nodes_per_batch = len(self.graph_builder.robot.JOINT)

        # Create batch offsets and expand edges
        offsets = torch.arange(batch_size, device=device) * num_nodes_per_batch
        src_batch = (src.unsqueeze(0) + offsets.unsqueeze(1)).flatten()
        dst_batch = (dst.unsqueeze(0) + offsets.unsqueeze(1)).flatten()
        edge_attr_batch = edge_attr.repeat(batch_size).unsqueeze(-1)

        return torch.stack([src_batch, dst_batch]), edge_attr_batch, None

    def _get_cached_edges(self, current_batch_size: int):
        """
        Optimized method to get edge indices with dynamic caching.
        Automatically caches new batch sizes as they're encountered.
        """
        # Check if already cached
        if current_batch_size in self._edge_cache:
            return self._edge_cache[current_batch_size]

        # Generate, cache, and return
        edge_data = self.generate_index(current_batch_size, self.device)
        self._edge_cache[current_batch_size] = edge_data
        return edge_data

    def forward(self, obs: torch.Tensor, xpos: torch.Tensor) -> torch.Tensor:
        current_batch_size = obs.shape[0]
        edges, _, _ = self._get_cached_edges(current_batch_size)

        if self.has_mixed_node_types:
            # === STEP 1: Local Processing within Clusters ===

            # 1a. Extract joint and object features
            h_joint, h_object, x_joint, _ = self.graph_builder.generate_input_for_mixed_type(obs, xpos)

            # 1b. Process objects locally with MLP (object cluster)
            h_object_processed = self.object_mlp(h_object)  # [batch, hidden_nf]

            # 1c. Process joints locally with EGNN (actuator cluster)
            h_joint_embedded = self.joint_embedding_in(h_joint)  # [batch*num_joints, hidden_nf]
            h_joints = h_joint_embedded
            for layer in self.layers:
                h_joints, x_joint, _ = layer(h=h_joints, edge_index=edges, coord=x_joint)

            # === STEP 2: Global Aggregation via Directed Fully-Connected Inter-Edges ===
            # (object -> all joints)
            h_joints_batched = h_joints.view(current_batch_size, self.num_joints, self.hidden_nf)
            h_object_broadcasted = h_object_processed.unsqueeze(1).expand(-1, self.num_joints, -1)
            h_joint_object_concat = torch.cat([h_joints_batched, h_object_broadcasted], dim=-1)

            # MLP computes context-dependent delta for each joint
            h_delta = self.global_aggregation(h_joint_object_concat)  # [batch, num_joints, 1]

            # === Residual connection ===
            h_skip = self.skip_proj(h_joint_object_concat)  # [batch, num_joints, 1]
            h_global = h_skip + h_delta

            # Final bounded joint actions
            actions = torch.tanh(h_global)
            return actions.view(current_batch_size, self.num_joints)

        else:
            h, x = self.graph_builder.generate_input(obs, xpos)
            h = self.embedding_in(h)
            for layer in self.layers:
                h, x, _ = layer(h=h, edge_index=edges, coord=x)
            h = self.embedding_out(h)
            h = h.view(current_batch_size, self.num_joints)
            return torch.tanh(h)


def unsorted_segment_sum(data, segment_ids, num_segments):
    return scatter_sum(data, segment_ids, dim=0, dim_size=num_segments)


def unsorted_segment_mean(data, segment_ids, num_segments):
    return scatter_mean(data, segment_ids, dim=0, dim_size=num_segments)
