import torch
from torch_geometric.transforms import BaseTransform, RadiusGraph
from torch_geometric.typing import SparseTensor
from torch_geometric.utils import coalesce, remove_self_loops, add_self_loops
from ponita.geometry.rotation import uniform_grid_s2, random_matrix
from ponita.geometry.rotation_2d import uniform_grid_s1, random_so2_matrix
from ponita.utils.to_from_sphere import scalar_to_sphere, vec_to_sphere
import torch_geometric


class PositionOrientationGraph(BaseTransform):
    """
    A PyTorch Geometric transform that lifts a point cloud in position space, as stored in a graph data object,
    to a position-orientation space fiber bundle. The grid of orientations sampled in each fiber is shared over 
    all nodes and node features (scalars and/or vectors) are locally lifted to this grid.

    Args:
        num_ori (int): Number of orientations used to discretize the sphere.
    """

    def __init__(self, num_ori, radius=None):
        super().__init__()

        # Discretization of the orientation grid
        self.num_ori = num_ori
        self.radius = radius

        # Grid type
        if num_ori > 0:
            # Somewhat redundant but this is done to be compatible with both 2d and 3d
            self.ori_grid_s1 = uniform_grid_s1(num_ori)
            self.ori_grid_s2 = uniform_grid_s2(num_ori)
            
        if radius is not None:
            self.transform = RadiusGraph(radius, loop=True, max_num_neighbors=1000)

    def __call__(self, graph):
        """
        Apply the transform to the input graph.

        Args:
            graph (torch_geometric.data.Data): Input graph containing position (graph.pos),
                                                scalar features (optional, graph.x) with shape [num_nodes, num_features], 
                                                and vector features (optional, graph.vec) with shape [num_nodes, num_vec_features, 3]

        Returns:
            torch_geometric.data.Data: Updated graph with added orientation information (graph.ori_grid with shape [num_ori, n])
                                      and lifted feature (graph.f with shape [num_nodes, num_ori, num_vec + num_x]).
        """
        if self.num_ori == -1:
            graph = self.to_po_point_cloud(graph)
        elif self.num_ori == 0:
            graph = self.to_p_point_cloud(graph)
        else:
            graph = self.to_po_fiber_bundle(graph)
        loop = True  # Hard-code that self-interactions are always present
        if self.radius is not None:
            graph.edge_index = torch_geometric.nn.radius_graph(graph.pos[:,:graph.n], self.radius, graph.batch, loop, max_num_neighbors=1000)
        else:
            if loop:
                graph.edge_index = coalesce(add_self_loops(graph.edge_index)[0])
        return graph

    def to_po_fiber_bundle(self, graph):
        """
        Internal method to add orientation information to the input graph.

        Args:
            graph (torch_geometric.data.Data): Input graph containing position (graph.pos),
                                                scalar features (optional, graph.x with shape [num_nodes, num_scalars]), and
                                                vector features (optional, graph.vec with shape [num_nodes, num_vec, n]).

        Returns:
            torch_geometric.data.Data: Updated graph with added orientation information (graph.ori_grid) and features (graph.f).
        """
        graph.n = graph.pos.size(1)
        graph.ori_grid = (self.ori_grid_s1 if (graph.n == 2) else self.ori_grid_s2).type_as(graph.pos)
        graph.num_ori = self.num_ori
        

        # Lift input features to spheres
        inputs = []
        if hasattr(graph, "x"):    inputs.append(scalar_to_sphere(graph.x, graph.ori_grid))
        if hasattr(graph, "vec"):  inputs.append(vec_to_sphere(graph.vec, graph.ori_grid))
        graph.x = torch.cat(inputs, dim=-1)  # [num_nodes, num_ori, input_dim + input_dim_vec]

        # Return updated graph
        return graph

    def to_po_point_cloud(self, graph):
        graph.n = graph.pos.size(1)
        
        # -----------  The relevant items in the original graph

        # We should remove self-loops because those cannot define directions
        input_edge_index = remove_self_loops(coalesce(graph.edge_index, num_nodes=graph.num_nodes))[0]
        # The other relevant items
        pos = graph.pos
        batch = graph.batch
        source, target = input_edge_index

        # ----------- Lifted positions (each original edge now becomes a node)
        
        # Compute direction vectors from the edge_index
        pos_s, pos_t = pos[source], pos[target]
        dist = (pos_s - pos_t).norm(dim=-1, keepdim=True)
        ori_t = (pos_s - pos_t) / dist
        
        # Target position as base node
        graph.pos = torch.cat([pos_t, ori_t], dim=-1)  # [4D, or 6D position-orientation element]
        
        # Each edge in the original graph will become a new node with the following index
        lifted_index = torch.arange(source.size(0), device=source.device)  # lifted idx
        
        # ----------- Lift the edge_index

        # For the new_edge_index we do allow for self-interactions
        base_edge_index = coalesce(add_self_loops(input_edge_index)[0])
        base_source = base_edge_index[0]
        base_target = base_edge_index[1]

        # The following is used as lookup table for connecting lifted idx to base idx
        # We use SparseTensor for this, which codes the triplets (row_idx, col_idx, value)
        # Here we define the triplet as (base_idx, the_original_node_sending_to_this, lifted_idx)
        # In particular the combination base_idx -> lifted_idx is going to be useful to lookup which
        # lifted nodes are associated with a base node
        num_base = pos.size(0)
        baseidx_source_liftidx = SparseTensor(row=target, col=source, value=lifted_index, sparse_sizes=(num_base, num_base))

        # Determine the number of lifted_idx at each base node
        num_ori_at_base = baseidx_source_liftidx.set_value(None).sum(dim=1).to(torch.long)
        
        # We now take the base_edge_index as starting point
        # We take the base indices of this edge_index and look up which lifted points are 
        # associated with these base indices. This will form the set of sending lifted indices
        lifted_source = baseidx_source_liftidx[base_source].storage.value()  # [648000 = 72000 * 9]
        
        # Then check the base nodes at the receiving end
        base_target = base_target.repeat_interleave(num_ori_at_base[base_source])  # [64800]
        
        # Lookup all the lifted indices at the receiving (target) node
        lifted_target = baseidx_source_liftidx[base_target].storage.value()  # [5832000 = 648000 * 9]
        
        # Repeat the lifted source the number of times that it has to send to the receiving target
        lifted_source = lifted_source.repeat_interleave(num_ori_at_base[base_target])
        
        # Now we're done
        lifted_edge_index = torch.stack([lifted_source, lifted_target])
        graph.edge_index = lifted_edge_index

        # ----------- Lift the batch

        if hasattr(graph, "batch"):
            if graph.batch is not None:
                graph.batch = batch[input_edge_index[1]].clone().contiguous()

        # ----------- Lift the scalar and vector features, overwrite x
        inputs = []
        if hasattr(graph, "x"):    
            inputs.append(graph.x[input_edge_index[1]])
        if hasattr(graph, "vec"):  
            inputs.append(torch.einsum('bcd,bd->bc', graph.vec[input_edge_index[1]], graph.pos[:,graph.n:]))
        graph.x = torch.cat(inputs, dim=-1)  # [num_lifted_nodes, num_channels]

        # ----------- Utility to be able to project back to the base node (e.g. via scatter collect)

        graph.scatter_projection_index = input_edge_index[1]
        

        return graph
    
    def to_p_point_cloud(self, graph):
        graph.n = graph.pos.size(1)

        # Otherwise do nothing because the graph is already assumed to be a position point cloud
        # I.e., it already has graph.pos, graph.x, graph.batch and possibly graph.edge_index
        # However, we need to add scatter_projection_index for compatibility with the ponita method
        graph.scatter_projection_index = torch.arange(0,graph.pos.size(0)).type_as(graph.batch)

        return graph