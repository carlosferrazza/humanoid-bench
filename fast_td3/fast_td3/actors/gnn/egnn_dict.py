from torch import nn
import torch

from fast_td3.robots.graph_builder import GraphBuilder
from fast_td3.actors.gnn.egnn import E_GCL, env_with_object

class EGNN_dict(nn.Module):
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
        coords_agg="mean"
    ):
        """
        :param in_node_nf: Number of features for 'h' at the input
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

        super(EGNN_dict, self).__init__()
        self.in_edge_nf = in_edge_nf
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.out_node_nf = out_node_nf
        self.batch_size = batch_size
        self.has_mixed_node_types = env_name in env_with_object
        self.robot = robot
        self.graph_builder = GraphBuilder(env_name, batch_size, device, robot)
        self.num_joints = self.graph_builder.robot.num_joints
        self.num_edges = self.graph_builder.robot.num_edges
        self._edges_cache = {}

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
                )
                for _ in range(n_layers)
            ]
        )

        self.joint_embedding_in = nn.Sequential(
            nn.Linear(in_node_nf, self.hidden_nf), act_fn
        )
        self.joint_embedding_out = nn.Sequential(
            nn.Linear(self.hidden_nf, out_node_nf),
            nn.Tanh(),
        )
        
        self.to(self.device)

    def forward(self, obs: dict) -> torch.Tensor:
        current_batch_size = obs["joint_velocities"].shape[0]
        edges = self.get_cached_edges(current_batch_size)
        
        h_joints = torch.stack([obs["joint_velocities"].reshape(-1), obs["joint_positions"].reshape(-1)], dim=1)
        x_joint = obs["joint_x"].reshape(-1, 3)

        h_joints = self.joint_embedding_in(h_joints)
        for layer in self.layers:
            h_joints, x_joint, _ = layer(h=h_joints, edge_index=edges, coord=x_joint)

        actions = self.joint_embedding_out(h_joints)

        return actions.view(current_batch_size, self.num_joints)

    def generate_index(self, batch_size: int, device="cuda"):
        src, dst = zip(*self.graph_builder.robot.joint_connections)

        src = torch.tensor(src, dtype=torch.long, device=device)
        dst = torch.tensor(dst, dtype=torch.long, device=device)

        # Create batch offsets and expand edges
        offsets = torch.arange(batch_size, device=device) * self.num_joints
        src_batch = (src.unsqueeze(0) + offsets.unsqueeze(1)).flatten()
        dst_batch = (dst.unsqueeze(0) + offsets.unsqueeze(1)).flatten()

        return torch.stack([src_batch, dst_batch])

    def get_cached_edges(self, current_batch_size: int):
        if current_batch_size in self._edges_cache:
            return self._edges_cache[current_batch_size]

        # Generate, cache, and return
        edges = self.generate_index(current_batch_size, self.device)
        self._edges_cache[current_batch_size] = edges
        return edges


@torch.jit.script
def unsorted_segment_sum(data: torch.Tensor, segment_ids: torch.Tensor, num_segments: int) -> torch.Tensor:
    """
    JIT-compiled optimized unsorted segment sum using scatter_add.
    """
    result = torch.zeros(num_segments, data.size(1), dtype=data.dtype, device=data.device)
    segment_ids_expanded = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids_expanded, data)
    return result


@torch.jit.script
def unsorted_segment_mean(data: torch.Tensor, segment_ids: torch.Tensor, num_segments: int) -> torch.Tensor:
    """
    JIT-compiled optimized unsorted segment mean with efficient counting.
    """
    result = torch.zeros(num_segments, data.size(1), dtype=data.dtype, device=data.device)
    segment_ids_expanded = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    
    # Sum values
    result.scatter_add_(0, segment_ids_expanded, data)
    
    # Count occurrences
    count = torch.zeros(num_segments, data.size(1), dtype=data.dtype, device=data.device)
    ones = torch.ones_like(data)
    count.scatter_add_(0, segment_ids_expanded, ones)
    
    # Use torch.where to handle division by zero
    return torch.where(count > 0, result / count, result)