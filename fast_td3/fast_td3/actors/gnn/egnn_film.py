"""
EGNN with FiLM (Feature-wise Linear Modulation) conditioning on root features.

Instead of concatenating root features, we use them to modulate intermediate representations.
"""
from torch import nn
import torch
from torch_scatter import scatter_sum, scatter_mean

from fast_td3.robots.graph_builder import GraphBuilder


env_with_object = [
    "h1-push-v0",
    "h1-basketball-v0",
    "h1-package-v0",
    "h1-sit_hard-v0",
    "h1-balance_simple-v0",
]


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation layer.
    
    Given context c and features h, computes:
        h_out = γ(c) ⊙ h + β(c)
    
    where γ and β are learned functions (typically linear layers).
    """
    def __init__(self, feature_dim, context_dim):
        super().__init__()
        # Learn scale (gamma) and shift (beta) from context
        self.scale_net = nn.Linear(context_dim, feature_dim)
        self.shift_net = nn.Linear(context_dim, feature_dim)
        
        # Initialize to identity: scale=1, shift=0
        nn.init.ones_(self.scale_net.weight)
        nn.init.zeros_(self.scale_net.bias)
        nn.init.zeros_(self.shift_net.weight)
        nn.init.zeros_(self.shift_net.bias)
    
    def forward(self, h, context):
        """
        Args:
            h: Features to modulate [batch*num_nodes, feature_dim]
            context: Context vector [batch*num_nodes, context_dim]
        Returns:
            Modulated features [batch*num_nodes, feature_dim]
        """
        gamma = self.scale_net(context)  # Scale parameter
        beta = self.shift_net(context)    # Shift parameter
        return gamma * h + beta


class E_GCL_FiLM(nn.Module):
    """
    E(n) Equivariant Convolutional Layer with FiLM conditioning.
    
    The root context modulates the edge and node features through FiLM layers,
    allowing the network to adaptively change its computation based on the robot's global state.
    """

    def __init__(
        self,
        input_nf,
        output_nf,
        hidden_nf,
        edges_in_d,
        context_dim,  # New: dimension of root context
        act_fn=nn.SiLU(),
        residual=True,
        attention=False,
        normalize=False,
        coords_agg="mean",
        tanh=False,
        edge_coords_nf=1,
        use_film=True,  # Can disable for ablation
    ):
        super(E_GCL_FiLM, self).__init__()
        input_edge = input_nf * 2
        self.residual = residual
        self.attention = attention
        self.normalize = normalize
        self.coords_agg = coords_agg
        self.tanh = tanh
        self.epsilon = 1e-8
        self.use_film = use_film

        # Edge MLP
        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edge_coords_nf + edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
        )
        
        # FiLM conditioning for edge features
        if use_film:
            self.edge_film = FiLMLayer(hidden_nf, context_dim)

        # Node MLP
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf + edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf),
        )
        
        # FiLM conditioning for node features
        if use_film:
            self.node_film = FiLMLayer(output_nf, context_dim)

        # Coordinate update MLP
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
        """Compute squared distance d_{ij}^2 = ||x_i - x_j||^2"""
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.sum(coord_diff**2, 1).unsqueeze(1)

        if self.normalize:
            norm = torch.sqrt(radial).detach() + self.epsilon
            coord_diff = coord_diff / norm

        return radial, coord_diff

    def edge_model(self, source, target, radial, edge_attr, context=None):
        """Compute edge message with optional FiLM conditioning"""
        if edge_attr is None:
            out = torch.cat([source, target, radial], dim=1)
        else:
            out = torch.cat([source, target, radial, edge_attr], dim=1)
        
        out = self.edge_mlp(out)
        
        # Apply FiLM modulation based on root context
        if self.use_film and context is not None:
            out = self.edge_film(out, context)
        
        if self.attention:
            att_val = self.att_mlp(out)
            out = out * att_val
        return out

    def coord_model(self, coord, edge_index, coord_diff, edge_feat):
        """Coordinate update (unchanged from original)"""
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

    def node_model(self, x, edge_index, edge_feat, node_attr, context=None):
        """Feature update with optional FiLM conditioning"""
        row, col = edge_index
        agg = unsorted_segment_sum(edge_feat, row, num_segments=x.size(0))
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)
        
        out = self.node_mlp(agg)
        
        # Apply FiLM modulation based on root context
        if self.use_film and context is not None:
            out = self.node_film(out, context)
        
        if self.residual:
            out = x + out
        return out, agg

    def forward(self, h, edge_index, coord, edge_attr=None, node_attr=None, context=None):
        """
        Args:
            context: Root context features [batch*num_nodes, context_dim]
                     These should be the same for all nodes in a batch item
        """
        row, col = edge_index

        radial, coord_diff = self.coord2radial(edge_index, coord)

        # Get context for edges (from source nodes)
        edge_context = context[row] if context is not None else None
        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr, edge_context)

        coord = self.coord_model(coord, edge_index, coord_diff, edge_feat)

        h, agg = self.node_model(h, edge_index, edge_feat, node_attr, context)

        return h, coord, edge_attr


def unsorted_segment_sum(data, segment_ids, num_segments):
    """Custom sum aggregation for segments"""
    return scatter_sum(data, segment_ids, dim=0, dim_size=num_segments)


def unsorted_segment_mean(data, segment_ids, num_segments):
    """Custom mean aggregation for segments"""
    return scatter_mean(data, segment_ids, dim=0, dim_size=num_segments)


class EGNN_FiLM(nn.Module):
    """
    EGNN with FiLM conditioning on root features.
    
    Key differences from standard EGNN:
    1. Root features are NOT concatenated to joint features
    2. Instead, root features modulate intermediate representations via FiLM
    3. Joint features start with just [pos, vel] (2 dimensions instead of 15)
    """
    
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
        use_film=True,  # Enable/disable FiLM for ablation
    ):
        super(EGNN_FiLM, self).__init__()
        self.in_edge_nf = in_edge_nf
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.out_node_nf = out_node_nf
        self.batch_size = batch_size
        self.use_film = use_film
        self.robot = robot
        self.env_name = env_name
        self.graph_builder = GraphBuilder(env_name, batch_size, device, robot)
        self.num_joints = len(self.graph_builder.robot.JOINT)
        
        # Context dimension (root features)
        self.context_dim = 13  # root pos (3) + quat (4) + root vel (6)
        
        # Joint feature dimension (just pos + vel, no root concatenation)
        self.joint_input_dim = 2  # pos + vel
        
        # Embedding for joint features (smaller input now)
        self.joint_embedding_in = nn.Sequential(
            nn.Linear(self.joint_input_dim, self.hidden_nf),
            act_fn
        )
        
        # Optional: Learn to process context before using it
        self.context_encoder = nn.Sequential(
            nn.Linear(self.context_dim, self.context_dim),
            act_fn,
            nn.Linear(self.context_dim, self.context_dim),
        )
        
        # EGNN layers with FiLM
        self.layers = nn.ModuleList([
            E_GCL_FiLM(
                self.hidden_nf,
                self.hidden_nf,
                self.hidden_nf,
                edges_in_d=in_edge_nf,
                context_dim=self.context_dim,
                act_fn=act_fn,
                residual=residual,
                attention=attention,
                normalize=normalize,
                tanh=tanh,
                coords_agg=coords_agg,
                edge_coords_nf=1,
                use_film=use_film,
            )
            for _ in range(n_layers)
        ])
        
        # Output projection
        self.embedding_out = nn.Sequential(
            nn.Linear(self.hidden_nf, out_node_nf),
            nn.Tanh(),
        )
        
        self.to(self.device)
        self._edge_cache = {}
        
        # Pre-compute edges
        self.edge_index, self.edge_attr, self.node_attr = self.generate_index(
            batch_size, device
        )
    
    def generate_index(self, batch_size: int, device="cuda"):
        """Generate joint-to-joint edge indices"""
        src, dst = zip(*self.graph_builder.robot.joint_connections)
        
        src = torch.tensor(src, dtype=torch.long, device=device)
        dst = torch.tensor(dst, dtype=torch.long, device=device)
        edge_attr = torch.zeros(len(src), dtype=torch.float, device=device)
        num_nodes_per_batch = len(self.graph_builder.robot.JOINT)

        offsets = torch.arange(batch_size, device=device) * num_nodes_per_batch
        src_batch = (src.unsqueeze(0) + offsets.unsqueeze(1)).flatten()
        dst_batch = (dst.unsqueeze(0) + offsets.unsqueeze(1)).flatten()
        
        edge_index = torch.stack([src_batch, dst_batch], dim=0)
        edge_attr_batch = edge_attr.repeat(batch_size)
        
        node_attr = None
        
        return edge_index, edge_attr_batch, node_attr
    
    def _get_cached_edges(self, current_batch_size: int):
        """Get or generate cached edges for given batch size"""
        if current_batch_size not in self._edge_cache:
            edges, edge_attr, node_attr = self.generate_index(
                current_batch_size, self.device
            )
            self._edge_cache[current_batch_size] = (edges, edge_attr, node_attr)
        return self._edge_cache[current_batch_size]
    
    def generate_input_film(self, obs: torch.tensor, xanchor: torch.tensor):
        """
        Generate input for FiLM-conditioned EGNN.
        
        Returns:
            h: Joint features [batch*19, 2] - just pos and vel
            x: Positions [batch*19, 3]
            context: Root features [batch*19, 13] - broadcasted to all joints
        """
        assert obs.shape[1] == 51, f"obs shape: {obs.shape}"
        assert xanchor.shape[1] == 20, f"xanchor shape: {xanchor.shape}"

        # Extract root features (13 values) - these become context
        root_features = torch.cat([
            obs[:, 0:7],    # root pos (3) + quat (4) = 7
            obs[:, 26:32],  # root vel (6)
        ], dim=1)  # [batch, 13]

        # Extract joint features (NO root concatenation)
        joint_pos = obs[:, 7:26].reshape(-1, 1)   # [batch*19, 1]
        joint_vel = obs[:, 32:].reshape(-1, 1)    # [batch*19, 1]
        
        h = torch.cat([joint_pos, joint_vel], dim=1)  # [batch*19, 2]

        # Broadcast root to all 19 joints as context
        context = root_features.repeat_interleave(19, dim=0)  # [batch*19, 13]

        # Positions remain relative to root
        x = (xanchor[:, 1:] - xanchor[:, [0]]).reshape(-1, 3)  # [batch*19, 3]

        return h, x, context
    
    def forward(self, obs: torch.Tensor, xanchor: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with FiLM conditioning.
        """
        current_batch_size = obs.shape[0]
        edges, _, _ = self._get_cached_edges(current_batch_size)
        
        # Generate input: joint features and root context
        h, x, context = self.generate_input_film(obs, xanchor)
        
        # Encode context
        context = self.context_encoder(context)  # [batch*19, 13]
        
        # Embed joint features
        h = self.joint_embedding_in(h)  # [batch*19, hidden_nf]
        
        # Process through FiLM-conditioned EGNN layers
        for layer in self.layers:
            h, x, _ = layer(
                h=h, 
                edge_index=edges, 
                coord=x,
                context=context  # Pass root context to modulate
            )
        
        # Output projection
        h = self.embedding_out(h)
        h = h.view(current_batch_size, self.num_joints)
        
        return h


if __name__ == "__main__":
    # Test FiLM layer
    print("Testing FiLM layer...")
    film = FiLMLayer(feature_dim=32, context_dim=13)
    h = torch.randn(10, 32)
    context = torch.randn(10, 13)
    h_modulated = film(h, context)
    print(f"Input shape: {h.shape}, Output shape: {h_modulated.shape}")
    
    # Test full EGNN with FiLM
    print("\nTesting EGNN with FiLM...")
    model = EGNN_FiLM(
        hidden_nf=64,
        out_node_nf=1,
        in_edge_nf=0,
        device="cpu",
        batch_size=4,
        act_fn=nn.SiLU(),
        n_layers=3,
        robot="h1",
        env_name="h1-walk-v0",
        use_film=True
    )
    
    obs = torch.randn(4, 51)
    xanchor = torch.randn(4, 20, 3)
    output = model(obs, xanchor)
    print(f"Output shape: {output.shape}")
    print("✓ FiLM EGNN test passed!")
