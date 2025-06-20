import torch
import unittest
import numpy as np
from egnn_clean import EGNN, get_edges_batch


class TestBatchedEgnnInput(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2  # Set batch size for testing
        self.egnn = EGNN(in_node_nf=1, hidden_nf=32, out_node_nf=1, in_edge_nf=1, device="cuda:0", batch_size=self.batch_size)

    def test_single_batch_input(self):
        # Create mock input data
        obs = torch.ones(1, 51).to("cuda:0")  # 32 + 19 = 51 features per observation
        xpos = torch.ones(1, 22, 3).to("cuda:0")  # 22 nodes, 3D positions
        xpos[:, 3, :] = torch.tensor([2, 2, 2])
        
        # Get output
        h, x, edge_index, edge_attr = self.egnn.build_batched_egnn_input(obs, xpos)
        
        # Test shapes
        self.assertEqual(h.shape, (19, 1))  # 19 nodes, 1 feature per node
        self.assertEqual(x.shape, (19, 3))  # 19 nodes, 3D positions
        self.assertEqual(edge_index.shape, (2, 18))  # 19 edges
        self.assertEqual(edge_attr.shape, (18, 1))  # 19 edges, 1 attribute per edge

    def test_full_batch_input(self):
        # Create mock input data for full batch
        obs = torch.randn(self.batch_size, 51).to("cuda:0")  # Random observations
        xpos = torch.randn(self.batch_size, 22, 3).to("cuda:0")  # Random positions

        # Get output
        h, x, edge_index, edge_attr = self.egnn.build_batched_egnn_input(obs, xpos)

        # Test shapes
        self.assertEqual(h.shape, (self.batch_size * 19, 1))
        self.assertEqual(x.shape, (self.batch_size * 19, 3))
        self.assertEqual(edge_index.shape, (2, self.batch_size * 18))
        self.assertEqual(edge_attr.shape, (self.batch_size * 18, 1))


    def test_batch_independence(self):
        # Create two identical batches
        obs = torch.randn(2, 51).to("cuda:0")
        obs[1] = obs[0]  # Make second batch identical to first
        xpos = torch.randn(2, 22, 3).to("cuda:0")
        xpos[1] = xpos[0]  # Make second batch identical to first
        
        # Get output
        h, x, edge_index, edge_attr = self.egnn.build_batched_egnn_input(obs, xpos)
        
        # Test that the first half of the output matches the second half
        num_nodes = 19
        num_edges = 18
        
        # Check node features
        self.assertTrue(torch.allclose(h[:num_nodes], h[num_nodes:]))
        self.assertTrue(torch.allclose(x[:num_nodes], x[num_nodes:]))
        
        # Check edge attributes
        # self.assertTrue(torch.allclose(edge_attr[:num_edges], edge_attr[num_edges:]))


if __name__ == '__main__':
    unittest.main()
