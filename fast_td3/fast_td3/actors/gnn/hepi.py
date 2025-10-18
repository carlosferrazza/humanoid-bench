from typing import Dict

import torch
import torch.nn as nn
from torch_geometric.data import HeteroData

from .base_gnn import BaseGNN
from .ponita.conv import FiberBundleConv
from .ponita.hetero_fiber_conv import HeteroFiberConv
from .ponita.ponita import GridGenerator, PolynomialFeatures
from .ponita.utils.to_from_sphere import scalar_to_sphere, vec_to_sphere
from fast_td3.skeleton_builder import build_edge_index_and_attr_mpnn
# from fast_td3.robots.h1 import h1


class HEPi(BaseGNN):
    def __init__(
        self,
        input_dim_node,
        input_dim_edge,
        hidden_dim,
        latent_dim,
        output_dim,
        output_dim_vec,
        node_encoder_layers,
        edge_encoder_layers,
        node_decoder_layers,
        node_type_mapping,
        edge_type_mapping,
        edge_level_mapping,
        num_messages,
        batch_size,
        robot="h1",
        concat_global=False,
        shared_processor=False,
        shared_node_encoder=True,
        shared_edge_encoder=True,
        device="cuda",
        num_ori=16,
        basis_dim=None,
        degree=2,
        ponita_dim=3,
        only_upper_hemisphere=False,
        **ignored,
    ):
        super(BaseGNN, self).__init__()

        self.input_dim_node = input_dim_node
        self.output_dim = output_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_messages = num_messages
        self.shared_processor = shared_processor
        self.node_type_mapping = node_type_mapping
        self.edge_type_mapping = edge_type_mapping
        self.device = device
        self.concat_global = concat_global
        self.batch_size = batch_size
        self.robot = robot

        message_passing = []
        for _ in range(num_messages):
            message_passing.append(
                FiberBundleConv(
                    in_channels=latent_dim,
                    out_channels=latent_dim,
                    attr_dim=latent_dim,
                    groups=latent_dim,
                    separable=True,
                    widening_factor=4,
                )
            )
        message_passing = [message_passing]

        # Initialize graph structure - this should be set by the calling code
        edge_index, edge_attr, num_nodes, num_edges = build_edge_index_and_attr_mpnn(
            self.robot, self.batch_size, self.device
        )
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.num_nodes = num_nodes
        self.num_edges = num_edges

        """ PONITA Settings """
        self.dim = ponita_dim
        self.num_ori = num_ori
        self.output_dim = output_dim
        self.output_dim_vec = output_dim_vec

        self.register_buffer(
            "ori_grid",
            GridGenerator(
                self.dim,
                num_ori,
                steps=1000,
                only_upper_hemisphere=only_upper_hemisphere,
            )(),
        )

        # Activation function to use internally
        act_fn = torch.nn.GELU()

        # Kernel basis functions and spatial window
        basis_dim = hidden_dim if (basis_dim is None) else basis_dim
        self.basis_fn = nn.Sequential(
            PolynomialFeatures(degree),
            nn.Linear(
                sum(2**i for i in range(1, degree + 2)), hidden_dim
            ),  # spatial invariants signal
            act_fn,
            nn.Linear(hidden_dim, basis_dim),
            act_fn,
        )
        self.fiber_basis_fn = nn.Sequential(
            PolynomialFeatures(degree),
            nn.Linear(
                sum(1**i for i in range(1, degree + 2)), hidden_dim
            ),  # spatial invariants signal
            act_fn,
            nn.Linear(hidden_dim, basis_dim),
            act_fn,
        )

        self.node_encoder = nn.Linear(self.input_dim_node, latent_dim, False)

        self.processor = nn.ModuleList()
        for k in range(num_messages):
            level_processor = {}
            for l, edge_level in enumerate(edge_level_mapping):
                pl = message_passing[l][k]

                if pl is not None:
                    for src, level, dest in edge_type_mapping:
                        if level == edge_level:
                            level_processor[src, level, dest] = pl.to(device)

            self.processor.append(HeteroFiberConv(level_processor))

        input_decoder_dim = latent_dim * 2 if concat_global else latent_dim
        self.decoder = nn.Linear(input_decoder_dim, output_dim + output_dim_vec)

    def compute_invariants(self, ori_grid, pos_send, pos_receive):
        rel_pos = pos_send - pos_receive  # [num_edges, 3]
        rel_pos = rel_pos[:, None, :]  # [num_edges, 1, 3]
        ori_grid_a = ori_grid[None, :, :]  # [1, num_ori, 3]
        ori_grid_b = ori_grid[:, None, :]  # [num_ori, 1, 3]
        # Displacement along the orientation
        invariant1 = (rel_pos * ori_grid_a).sum(
            dim=-1, keepdim=True
        )  # [num_edges, num_ori, 1]
        # Displacement orthogonal to the orientation (take norm in 3D)
        invariant2 = (rel_pos - invariant1 * ori_grid_a).norm(
            dim=-1, keepdim=True
        )  # [num_edges, num_ori, 1]
        # Relative orientation
        invariant3 = (ori_grid_a * ori_grid_b).sum(
            dim=-1, keepdim=True
        )  # [num_ori, num_ori, 1]
        # Stack into spatial and orientaiton invariants separately
        spatial_invariants = torch.cat(
            [invariant1, invariant2], dim=-1
        )  # [num_edges, num_ori, 2]
        orientation_invariants = invariant3  # [num_ori, num_ori, 1]
        return spatial_invariants, orientation_invariants

    def forward(
        self,
        obs,
        xanchor,
        **kwargs,
    ) -> torch.Tensor:
        batch_size = obs.shape[0]
        if batch_size == self.batch_size:
            edge_index = self.edge_index
            edge_attr = self.edge_attr
        else:
            assert (
                batch_size <= self.batch_size
            ), "Batch size exceeds the maximum batch size."
            edge_index = self.edge_index[:, : batch_size * self.num_edges].clone()
            edge_attr = self.edge_attr[: batch_size * self.num_edges].clone()

        x = xanchor[:, 1:].reshape(-1, 3)  # (B*N, 3)

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

        # result, _ = self.one_step(
        #     graph=h1.create_joint_graph(
        #         batch_edge_index=edge_index,
        #         batch_edge_attr=edge_attr,
        #         joint_positions=x,
        #     ),
        #     u_dict=self.construct_joint_input(
        #         robot=h1, joint_features=h, joint_positions=x, device=self.device
        #     ),
        # )

        num_acts = self.num_nodes

        # return result.view(batch_size, num_acts)
        return None

    def one_step(
        self,
        graph: HeteroData,
        u_dict: Dict[str, torch.Tensor],
        u: torch.Tensor = None,
        u_properties: torch.Tensor = None,
    ):

        scalar_dict, vector_dict = u_dict
        latent_dict = {}

        for node_type in graph.node_types:
            scalar = scalar_to_sphere(scalar_dict[node_type], self.ori_grid)
            # vector = vector_dict[node_type].view(scalar.shape[0], -1, 3)
            # vector = vector[..., :2] if self.dim == 2 else vector
            vector = vector_dict[node_type].view(scalar.shape[0])
            vector = vec_to_sphere(vector, self.ori_grid)

            x = torch.cat([scalar, vector], dim=-1)
            latent_dict[node_type] = self.node_encoder(x)

        kernel_basis_dict = {}
        fiber_kernel_basis_dict = {}
        for edge_type in graph.edge_types:
            src, level, dest = edge_type
            edge_index = graph.edge_index_dict[edge_type]
            pos_src = graph[src].pos[edge_index[0]]
            pos_dest = graph[dest].pos[edge_index[1]]

            pos_src = pos_src[..., :2] if self.dim == 2 else pos_src
            pos_dest = pos_dest[..., :2] if self.dim == 2 else pos_dest
            spatial_invariants, orientation_invariants = self.compute_invariants(
                self.ori_grid, pos_src, pos_dest
            )
            kernel_basis_dict[edge_type] = self.basis_fn(spatial_invariants)
            fiber_kernel_basis_dict[edge_type] = self.fiber_basis_fn(
                orientation_invariants
            )

        """
        Encoder encodes graph (node/edge features) into latent vectors (node/edge embeddings)
        The return of processor is fed into the processor for generating new feature vectors
        """

        for i in range(self.num_messages):
            processor = self.processor if self.shared_processor else self.processor[i]
            latent_dict = processor(
                latent_dict=latent_dict,
                edge_index_dict=graph.edge_index_dict,
                edge_attr_dict=kernel_basis_dict,
                fiber_attr_dict=fiber_kernel_basis_dict,
            )

        latent = latent_dict[graph.output_mask_key]

        if self.concat_global:
            global_latent = torch.cat(
                [latent_dict[node_type] for node_type in graph.node_types], dim=0
            )
            global_latent = global_latent.mean(dim=0, keepdim=True).repeat_interleave(
                latent.shape[0], dim=0
            )
            latent = torch.cat([latent, global_latent], dim=-1)

        output = self.decoder(latent)
        out_scalar, out_vec = output.split(
            [self.output_dim, self.output_dim_vec], dim=-1
        )

        # Average over the orientations
        latent = latent.mean(dim=-2)
        out_scalar = out_scalar.mean(dim=-2)
        out_vec = (
            torch.einsum("b o c, o d -> b c d", out_vec, self.ori_grid) / self.num_ori
        )
        out = out_vec * out_scalar.unsqueeze(-1)
        if self.dim == 2:
            out = torch.cat([out, torch.zeros_like(out[..., :1])], dim=-1)
        return out.reshape(-1, out.shape[-1]), latent.reshape(-1, latent.shape[-1])

    def construct_joint_input(self, robot, joint_positions, joint_features, device):
        # joint_features: [batch_size * num_joints, 2] is the joint velocities and joint angles

        num_node_types = len(robot.node_type)  # Number of different node types
        node_type_encoding = torch.zeros(
            joint_features.shape[0], num_node_types, device=device
        )
        node_type_encoding[:, 0] = 1  # All joints are type "JOINT"

        # Combine node type encoding with joint angles
        scalar_features = torch.cat(
            [
                node_type_encoding,
            ],
            dim=1,
        )

        print(joint_positions.shape)
        print(joint_features.shape)
        vector_features = torch.cat([joint_positions, joint_features], dim=1)
        print(vector_features.shape)

        # Create the dictionaries
        scalar_dict = {robot.node_type.JOINT: scalar_features}
        vector_dict = {robot.node_type.JOINT: vector_features}

        return scalar_dict, vector_dict
