import torch
from torch_geometric.transforms import BaseTransform
from ponita.geometry.rotation import random_matrix as random_so3_matrix
from ponita.geometry.rotation_2d import random_so2_matrix

class RandomRotate(BaseTransform):
    """
    A PyTorch Geometric transform that randomly rotates each point cloud in a batch of graphs
    by sampling rotations from a uniform distribution over the Special Orthogonal Group SO(3).

    Args:
        None
    """

    def __init__(self, attr_list, n=3):
        super().__init__()
        self.attr_list = attr_list
        self.random_rotation_matrix_fn = random_so2_matrix if (n == 2) else random_so3_matrix

    def __call__(self, graph):
        """
        Apply the random rotation transform to the input graph.

        Args:
            graph (torch_geometric.data.Data): Input graph containing position (graph.pos),
                                                and optionally, batch information (graph.batch).

        Returns:
            torch_geometric.data.Data: Updated graph with randomly rotated positions.
                                      The positions in the graph are rotated using a random rotation matrix
                                      sampled from a uniform distribution over SO(3).
        """
        rand_rot = self.random_rotation(graph)
        return self.rotate_graph(graph, rand_rot)

    def rotate_graph(self, graph, rand_rot):
        for attr in self.attr_list:
            if hasattr(graph, attr):
                setattr(graph, attr, self.rotate_attr(getattr(graph, attr), rand_rot))
        return graph
        
    def rotate_attr(self, attr, rand_rot):
            rand_rot = rand_rot.type_as(attr)
            if len(rand_rot.shape)==3:
                if len(attr.shape)==2:
                    return torch.einsum('bij,bj->bi', rand_rot, attr)
                else:
                    return torch.einsum('bij,bcj->bci', rand_rot, attr)
            else:
                if len(attr.shape)==2:
                    return torch.einsum('ij,bj->bi', rand_rot, attr)
                else:
                    return torch.einsum('ij,bcj->bci', rand_rot, attr)
    
    def random_rotation(self, graph):
        if graph.batch is not None:
            batch_size = graph.batch.max() + 1
            random_rotation_matrix = self.random_rotation_matrix_fn(batch_size).to(graph.batch.device)
            random_rotation_matrix = random_rotation_matrix[graph.batch]
        else:
            random_rotation_matrix = self.random_rotation_matrix(1)[0]
        return random_rotation_matrix
