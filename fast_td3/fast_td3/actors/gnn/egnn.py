from torch import nn
import torch

from fast_td3.skeleton_builder import build_edge_index_and_attr


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
    ):
        super(E_GCL, self).__init__()
        input_edge = input_nf * 2
        self.residual = residual
        self.attention = attention
        self.normalize = normalize
        self.coords_agg = coords_agg
        self.tanh = tanh
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
        if self.coords_agg == "sum":
            agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0))
        elif self.coords_agg == "mean":
            agg = unsorted_segment_mean(trans, row, num_segments=coord.size(0))
        else:
            raise Exception("Wrong coords_agg parameter" % self.coords_agg)
        coord = coord + agg
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

        super(EGNN, self).__init__()
        self.in_edge_nf = in_edge_nf
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.embedding_in = nn.Sequential(
            nn.Linear(in_node_nf, self.hidden_nf, device=device), 
            act_fn
        )
        self.embedding_out = nn.Sequential(
            nn.Linear(self.hidden_nf, out_node_nf, device=device),
            nn.Tanh(),
        )
        self.batch_size = batch_size
        for i in range(0, n_layers):
            self.add_module(
                "gcl_%d" % i,
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
                ),
            )
        self.robot = robot

        edge_index, edge_attr, num_nodes, num_edges = build_edge_index_and_attr(
            self.robot, self.batch_size, self.device
        )
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.num_nodes = num_nodes
        self.num_edges = num_edges

    def forward(self, h, x, edges, edge_attr):
        current_batch_size = int(h.shape[0] / self.num_nodes)

        h = self.embedding_in(h)

        for i in range(0, self.n_layers):
            h, x, _ = self._modules["gcl_%d" % i](h, edges, x, edge_attr=edge_attr)

        h = self.embedding_out(h)

        h = h.view(current_batch_size, self.num_nodes)

        return h

    def build_batched_egnn_input(self, obs: torch.tensor, xpos: torch.tensor):
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
            edge_index = [
                t[: batch_size * self.num_edges].clone() for t in self.edge_index
            ]
            edge_attr = self.edge_attr[: batch_size * self.num_edges].clone()

        x = xpos[:, 1:].reshape(-1, 3)
        x_base = xpos[:, 0].repeat_interleave(self.num_nodes, dim=1).reshape(-1, 3)
        x = x - x_base

        if self.robot == "h1":
            h = torch.stack(
                [obs[:, 32:].reshape(-1, 1), obs[:, 7:26].reshape(-1, 1)], dim=1
            ).squeeze(
                2
            )  # (B*N, 2)
        elif self.robot == "g1":
            h = torch.stack(
                [obs[:, 50:].reshape(-1, 1), obs[:, 7:44].reshape(-1, 1)], dim=1
            ).squeeze(
                2
            )  # (B*N, 2)

        if self.in_edge_nf == 0:
            edge_attr = None

        return h, x, edge_index, edge_attr


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
    egnn = EGNN(in_node_nf=n_feat, hidden_nf=32, out_node_nf=1, in_edge_nf=1)

    # Run EGNN
    h, x = egnn(h, x, edges, edge_attr)
