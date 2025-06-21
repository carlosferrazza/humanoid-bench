from torch import nn
import torch
import numpy as np


class E_GCL(nn.Module):
    """
    E(n) Equivariant Convolutional Layer
    re
    """

    def __init__(
        self,
        input_nf,
        output_nf,
        hidden_nf,
        edges_in_d=0,
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

    def edge_model(self, source, target, radial, edge_attr):
        if edge_attr is None:  # Unused.
            out = torch.cat([source, target, radial], dim=1)
        else:
            out = torch.cat([source, target, radial, edge_attr], dim=1)
        out = self.edge_mlp(out)
        if self.attention:
            att_val = self.att_mlp(out)
            out = out * att_val
        return out

    def node_model(self, x, edge_index, edge_attr, node_attr):
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

    def coord_model(self, coord, edge_index, coord_diff, edge_feat):
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

    def coord2radial(self, edge_index, coord):
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.sum(coord_diff**2, 1).unsqueeze(1)

        if self.normalize:
            norm = torch.sqrt(radial).detach() + self.epsilon
            coord_diff = coord_diff / norm

        return radial, coord_diff

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
        act_fn=nn.SiLU(),
        n_layers=4,
        residual=True,
        attention=False,
        normalize=False,
        tanh=False,
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
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.embedding_in = nn.Linear(in_node_nf, self.hidden_nf, device=device)
        self.embedding_out = nn.Linear(self.hidden_nf, out_node_nf, device=device)
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
                ),
            )
        self.to(self.device)

        self.xpos_indices = [
            2,  # left_hip_yaw_link
            3,  # left_hip_roll_link
            4,  # left_hip_pitch_link
            5,  # left_knee_link
            6,  # left_ankle_link
            7,  # right_hip_yaw_link
            8,  # right_hip_roll_link
            9,  # right_hip_pitch_link
            10,  # right_knee_link
            11,  # right_ankle_link
            12,  # torso_link
            13,  # left_shoulder_pitch_link
            14,  # left_shoulder_roll_link
            15,  # left_shoulder_yaw_link
            16,  # left_elbow_link
            18,  # right_shoulder_pitch_link
            19,  # right_shoulder_roll_link
            20,  # right_shoulder_yaw_link
            21,  # right_elbow_link
        ]

        self.xpos_indices = [
            2,  # left_hip_yaw_link
            3,  # left_hip_roll_link
            4,  # left_hip_pitch_link
            5,  # left_knee_link
            6,  # left_ankle_link
            7,  # right_hip_yaw_link
            8,  # right_hip_roll_link
            9,  # right_hip_pitch_link
            10,  # right_knee_link
            11,  # right_ankle_link
            12,  # torso_link
            13,  # left_shoulder_pitch_link
            14,  # left_shoulder_roll_link
            15,  # left_shoulder_yaw_link
            16,  # left_elbow_link
            18,  # right_shoulder_pitch_link
            19,  # right_shoulder_roll_link
            20,  # right_shoulder_yaw_link
            21,  # right_elbow_link
        ]

        joint_names = [
            "left_hip_yaw",
            "left_hip_roll",
            "left_hip_pitch",
            "left_knee",
            "left_ankle",
            "right_hip_yaw",
            "right_hip_roll",
            "right_hip_pitch",
            "right_knee",
            "right_ankle",
            "torso",
            "left_shoulder_pitch",
            "left_shoulder_roll",
            "left_shoulder_yaw",
            "left_elbow",
            "right_shoulder_pitch",
            "right_shoulder_roll",
            "right_shoulder_yaw",
            "right_elbow",
        ]
        joint_idx = {name: idx for idx, name in enumerate(joint_names)}
        edge_list = [
            # Left leg
            (joint_idx["left_hip_yaw"], joint_idx["left_hip_roll"]),
            (joint_idx["left_hip_roll"], joint_idx["left_hip_pitch"]),
            (joint_idx["left_hip_pitch"], joint_idx["left_knee"]),
            (joint_idx["left_knee"], joint_idx["left_ankle"]),
            # Right leg
            (joint_idx["right_hip_yaw"], joint_idx["right_hip_roll"]),
            (joint_idx["right_hip_roll"], joint_idx["right_hip_pitch"]),
            (joint_idx["right_hip_pitch"], joint_idx["right_knee"]),
            (joint_idx["right_knee"], joint_idx["right_ankle"]),
            # Torso
            (joint_idx["torso"], joint_idx["left_hip_yaw"]),
            (joint_idx["torso"], joint_idx["right_hip_yaw"]),
            # Left arm
            (joint_idx["torso"], joint_idx["left_shoulder_pitch"]),
            (joint_idx["left_shoulder_pitch"], joint_idx["left_shoulder_roll"]),
            (joint_idx["left_shoulder_roll"], joint_idx["left_shoulder_yaw"]),
            (joint_idx["left_shoulder_yaw"], joint_idx["left_elbow"]),
            # Right arm
            (joint_idx["torso"], joint_idx["right_shoulder_pitch"]),
            (joint_idx["right_shoulder_pitch"], joint_idx["right_shoulder_roll"]),
            (joint_idx["right_shoulder_roll"], joint_idx["right_shoulder_yaw"]),
            (joint_idx["right_shoulder_yaw"], joint_idx["right_elbow"]),
        ]

        self.edge_type_encoding = torch.tensor([  
            0,  # (left_hip_yaw, left_hip_roll)      - left_leg
            0,  # (left_hip_roll, left_hip_pitch)    - left_leg
            0,  # (left_hip_pitch, left_knee)        - left_leg
            0,  # (left_knee, left_ankle)            - left_leg

            1,  # (right_hip_yaw, right_hip_roll)    - right_leg
            1,  # (right_hip_roll, right_hip_pitch)  - right_leg
            1,  # (right_hip_pitch, right_knee)      - right_leg
            1,  # (right_knee, right_ankle)          - right_leg

            2,  # (torso, left_hip_yaw)              - torso_connection
            2,  # (torso, right_hip_yaw)             - torso_connection

            3,  # (torso, left_shoulder_pitch)      - left_arm
            3,  # (left_shoulder_pitch, left_shoulder_roll) - left_arm
            3,  # (left_shoulder_roll, left_shoulder_yaw)   - left_arm
            3,  # (left_shoulder_yaw, left_elbow)            - left_arm

            4,  # (torso, right_shoulder_pitch)     - right_arm
            4,  # (right_shoulder_pitch, right_shoulder_roll) - right_arm
            4,  # (right_shoulder_roll, right_shoulder_yaw)   - right_arm
            4,  # (right_shoulder_yaw, right_elbow)            - right_arm
        ], dtype=torch.long, device=device)

        src, dst = zip(*edge_list)  # Unpack edge list into two tuples
        src = torch.tensor(src, dtype=torch.long, device=device)
        dst = torch.tensor(dst, dtype=torch.long, device=device)

        # Stack edges for batch
        src_batch = []
        dst_batch = []
        egde_attr_batch = []
        for i in range(self.batch_size):
            offset = 19 * i
            src_batch.append(src + offset)
            dst_batch.append(dst + offset)
            egde_attr_batch.append(self.edge_type_encoding)

        src_batch = torch.cat(src_batch)
        dst_batch = torch.cat(dst_batch)

        self.edge_attr = torch.cat(egde_attr_batch).unsqueeze(1)  # (B*N, 1)
        self.edge_index = [src_batch, dst_batch]

    def forward(self, h, x, edges, edge_attr):
        batch_size = int(h.shape[0] / (19))

        # Debug: Check for NaNs/Infs in input tensors
        if torch.isnan(h).any() or torch.isinf(h).any():
            print("[DEBUG] NaN or Inf detected in input h", h)
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("[DEBUG] NaN or Inf detected in input x", x)
        if edge_attr is not None and (torch.isnan(edge_attr).any() or torch.isinf(edge_attr).any()):
            print("[DEBUG] NaN or Inf detected in input edge_attr", edge_attr)

        h = self.embedding_in(h)
        for i in range(0, self.n_layers):
            h, x, _ = self._modules["gcl_%d" % i](h, edges, x, edge_attr=edge_attr)
            # Debug: Check for NaNs/Infs after each layer
            if torch.isnan(h).any() or torch.isinf(h).any():
                print(f"[DEBUG] NaN or Inf detected in h after gcl_{i}", h)
            if torch.isnan(x).any() or torch.isinf(x).any():
                print(f"[DEBUG] NaN or Inf detected in x after gcl_{i}", x)

        h = self.embedding_out(h)

        h = h.view(batch_size, 19)

        # Debug: Check for NaNs/Infs in output
        if torch.isnan(h).any() or torch.isinf(h).any():
            print("[DEBUG] NaN or Inf detected in output h", h)

        return h

    def build_batched_egnn_input(self, obs: torch.tensor, xpos: torch.tensor):
        # if torch.isnan(obs).any() or torch.isinf(obs).any():
        #     raise ValueError("Input observation contains NaN or Inf values.")
        # if torch.isnan(xpos).any() or torch.isinf(xpos).any():
        #     raise ValueError("Input xpos contains NaN or Inf values.")

        batch_size = obs.shape[0]
        if batch_size == self.batch_size:
            edge_index = self.edge_index
            edge_attr = self.edge_attr
        else:
            assert (
                batch_size <= self.batch_size
            ), "Batch size exceeds the maximum batch size."
            edge_index = [t[: batch_size * 18].clone() for t in self.edge_index]
            edge_attr = self.edge_attr[: batch_size * 18].clone()

        # x = xpos[:, self.xpos_indices].reshape(-1, 3)  # (B*N, 3)
        x = obs[:, 7:26].reshape(-1, 1)
        
        h = obs[:, 32:].reshape(-1, 1)  # (B*N, 1)

        # if torch.isnan(x).any() or torch.isinf(x).any():
        #     print("NaN or Inf detected in input x")
        # if torch.isnan(h).any() or torch.isinf(h).any():
        #     print("NaN or Inf detected in input h")

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
