from torch import nn
import torch

from fast_td3.skeleton_builder import build_edge_index_and_attr


class Angle_GCL(nn.Module):
    """
    Angle-based Equivariant Convolutional Layer

    Mathematical operations for joint angles:
    1. Compute angle distance: d̃_{ij}^2 = ||sin(θ_i) - sin(θ_j)||^2 + ||cos(θ_i) - cos(θ_j)||^2
    2. Edge message: m_{ij} = φ_e(h_i, h_j, d̃_{ij}^2, a_{ij})
    3. Pose update: ψ_i^{l+1} = ψ_i^l + Σ_{j∈N(i)} (θ̃_i - θ̃_j) * φ_x(m_{ij})
    4. Feature update: h_i^{l+1} = φ_h(h_i, Σ_{j∈N(i)} m_{ij})

    where θ̃_i = [cos(θ_i), sin(θ_i)] for periodicity handling
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
        use_pose_embedding=True,
    ):
        super(Angle_GCL, self).__init__()
        input_edge = input_nf * 2
        self.residual = residual
        self.attention = attention
        self.normalize = normalize
        self.coords_agg = coords_agg
        self.tanh = tanh
        self.use_pose_embedding = use_pose_embedding
        self.epsilon = 1e-8
        edge_coords_nf = 1

        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edge_coords_nf + edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf),
        )

        if self.use_pose_embedding:
            layer = nn.Linear(hidden_nf, 1, bias=False)
            torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

            pose_mlp = []
            pose_mlp.append(nn.Linear(hidden_nf, hidden_nf))
            pose_mlp.append(act_fn)
            pose_mlp.append(layer)
            if self.tanh:
                pose_mlp.append(nn.Tanh())
            self.pose_mlp = nn.Sequential(*pose_mlp)

        if self.attention:
            self.att_mlp = nn.Sequential(nn.Linear(hidden_nf, 1), nn.Sigmoid())

    def angle2distance(self, edge_index, angles):
        """
        Step 1: Compute angle distance d̃_{ij}^2 = ||sin(θ_i) - sin(θ_j)||^2 + ||cos(θ_i) - cos(θ_j)||^2
        This handles periodicity and is invariant under joint-space translations
        Also computes trigonometric differences for equivariant updates
        """
        row, col = edge_index

        # Convert angles to trigonometric representation: [cos(θ), sin(θ)]
        cos_angles = torch.cos(angles)
        sin_angles = torch.sin(angles)
        
        trig_diff = angles[row] - angles[col]

        # Compute angle distance (periodicity-aware)
        cos_diff = cos_angles[row] - cos_angles[col]
        sin_diff = sin_angles[row] - sin_angles[col]

        radial = torch.sum(cos_diff**2, 1).unsqueeze(1) + torch.sum(
            sin_diff**2, 1
        ).unsqueeze(1)

        if self.normalize:
            norm = torch.sqrt(radial).detach() + self.epsilon
            trig_diff = trig_diff / norm

        return radial, trig_diff

    def edge_model(self, source, target, radial, edge_attr):
        """
        Step 2: Compute edge message m_{ij} = φ_e(h_i, h_j, d̃_{ij}^2, a_{ij})
        Uses angle-based distance instead of Euclidean distance
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

    def pose_model(self, pose, edge_index, pose_diff, edge_feat):
        """
        Step 3: Pose update ψ_i^{l+1} = ψ_i^l + Σ_{j∈N(i)} (θ̃_i - θ̃_j) * φ_x(m_{ij})
        Updates pose embedding using trigonometric angle differences
        """
        row, col = edge_index
        trans = pose_diff * self.pose_mlp(edge_feat)
        if self.coords_agg == "sum":
            agg = unsorted_segment_sum(trans, row, num_segments=pose.size(0))
        elif self.coords_agg == "mean":
            agg = unsorted_segment_mean(trans, row, num_segments=pose.size(0))
        else:
            raise Exception("Wrong coords_agg parameter" % self.coords_agg)
        pose = pose + agg
        return pose

    def node_model(self, x, edge_index, edge_attr, node_attr):
        """
        Step 4: Feature update h_i^{l+1} = φ_h(h_i, Σ_{j∈N(i)} m_{ij})
        Same as original EGNN - aggregates edge messages and updates node features
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

    def forward(self, h, edge_index, angles, pose=None, edge_attr=None, node_attr=None):
        """
        Forward pass for Angle-EGNN layer

        Args:
            h: Node features
            edge_index: Graph connectivity
            angles: Joint angles (N, k) where k is number of DoF per joint
            pose: Optional pose embedding (if use_pose_embedding=True)
            edge_attr: Optional edge attributes
            node_attr: Optional node attributes
        """
        row, col = edge_index
        radial, trig_diff = self.angle2distance(edge_index, angles)

        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr)

        if self.use_pose_embedding and pose is not None:
            pose = self.pose_model(pose, edge_index, trig_diff, edge_feat)

        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)

        return h, pose, edge_attr


class AngleEGNN(nn.Module):
    """
    Angle-based Equivariant Graph Neural Network

    Operates on joint angles instead of 3D coordinates while preserving
    key properties like translation invariance and permutation invariance.
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
        joint_dim=1,  # Number of DoF per joint
        residual=True,
        attention=False,
        normalize=False,
        tanh=False,
        use_pose_embedding=True,
        pose_dim=1,
    ):
        """
        Args:
            joint_dim: Number of degrees of freedom per joint
            use_pose_embedding: Whether to maintain pose embeddings
            pose_dim: Dimension of pose embedding vectors
        """
        super(AngleEGNN, self).__init__()
        self.in_edge_nf = in_edge_nf
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.joint_dim = joint_dim
        self.use_pose_embedding = use_pose_embedding
        self.pose_dim = pose_dim

        self.embedding_in = nn.Sequential(nn.Linear(in_node_nf, self.hidden_nf), act_fn)
        self.embedding_out = nn.Sequential(
            nn.Linear(self.hidden_nf, out_node_nf),
            nn.Tanh(),
        )

        if self.use_pose_embedding:
            self.pose_embedding_in = nn.Linear(
                joint_dim * 2, self.pose_dim
            )  # *2 for cos/sin

        self.layers = nn.ModuleList(
            [
                Angle_GCL(
                    self.hidden_nf,
                    self.hidden_nf,
                    self.hidden_nf,
                    edges_in_d=in_edge_nf,
                    act_fn=act_fn,
                    residual=residual,
                    attention=attention,
                    normalize=normalize,
                    tanh=tanh,
                    use_pose_embedding=use_pose_embedding,
                )
                for _ in range(n_layers)
            ]
        )
        self.to(self.device)
        self.robot = robot

        edge_index, edge_attr, num_nodes, num_edges = build_edge_index_and_attr(
            self.robot, self.batch_size, self.device
        )
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.num_nodes = num_nodes
        self.num_edges = num_edges

    def forward(self, h, angles, edges, edge_attr):
        """
        Forward pass for AngleEGNN

        Args:
            h: Node features (B*N, d)
            angles: Joint angles (B*N, k) where k is joint_dim
            edges: Edge indices
            edge_attr: Edge attributes
        """
        current_batch_size = int(h.shape[0] / self.num_nodes)

        h = self.embedding_in(h)

        pose = None
        if self.use_pose_embedding:
            # Initialize pose embedding from trigonometric representation of angles
            cos_angles = torch.cos(angles)
            sin_angles = torch.sin(angles)
            trig_repr = torch.cat([cos_angles, sin_angles], dim=-1)
            pose = self.pose_embedding_in(trig_repr)

        for layer in self.layers:
            h, pose, _ = layer(h, edges, angles, pose=pose, edge_attr=edge_attr)

        h = self.embedding_out(h)

        h = h.view(current_batch_size, self.num_nodes)

        return h

    def build_batched_angle_input(self, obs: torch.tensor, xpos: torch.tensor):
        """
        Build input for AngleEGNN from observations and joint angles

        Args:
            obs: Observation tensor (B, obs_dim)
            joint_angles: Joint angle tensor (B, N*k) where N is num_nodes, k is joint_dim
        """
        del xpos
        batch_size = obs.shape[0]
        if batch_size == self.batch_size:
            edge_index = self.edge_index
            edge_attr = self.edge_attr
        else:
            assert (
                batch_size <= self.batch_size
            ), "Batch size exceeds the maximum batch size."
            edge_index = [
                t[: batch_size * self.num_edges].clone() for t in self.edge_index
            ]
            edge_attr = self.edge_attr[: batch_size * self.num_edges].clone()

        # Build node features based on robot type
        if self.robot == "h1":
            h = obs[:, 32:].reshape(-1, 1)
            angles = obs[:, 7:26].reshape(-1, 1)
        elif self.robot == "g1":
            h = obs[:, 50:].reshape(-1, 1)
            angles = obs[:, 7:44].reshape(-1, 1)

        if self.in_edge_nf == 0:
            edge_attr = None

        return h, angles, edge_index, edge_attr


def unsorted_segment_sum(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    return result


def unsorted_segment_mean(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    count = data.new_full(result_shape, 0)
    result.scatter_add_(0, segment_ids, data)
    count.scatter_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)


def get_edges(n_nodes):
    rows, cols = [], []
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                rows.append(i)
                cols.append(j)

    edges = [rows, cols]
    return edges


def get_edges_batch(n_nodes, batch_size):
    edges = get_edges(n_nodes)
    edge_attr = torch.ones(len(edges[0]) * batch_size, 1)
    edges = [torch.LongTensor(edges[0]), torch.LongTensor(edges[1])]
    if batch_size == 1:
        return edges, edge_attr
    elif batch_size > 1:
        rows, cols = [], []
        for i in range(batch_size):
            rows.append(edges[0] + n_nodes * i)
            cols.append(edges[1] + n_nodes * i)
        edges = [torch.cat(rows), torch.cat(cols)]
    return edges, edge_attr


if __name__ == "__main__":
    # Dummy parameters
    batch_size = 8
    n_nodes = 4
    n_feat = 1
    x_dim = 3

    # Dummy variables h, x and fully connected edges
    h = torch.ones(batch_size * n_nodes, n_feat)
    x = torch.ones(batch_size * n_nodes, x_dim)
    edges, edge_attr = get_edges_batch(n_nodes, batch_size)

    # Initialize EGNN
    egnn = AngleEGNN(in_node_nf=n_feat, hidden_nf=32, out_node_nf=1, in_edge_nf=1)

    # Run EGNN
    h, x = egnn(h, x, edges, edge_attr)
