import unittest
from chex import assert_equal
import torch
import sys
import os

# Add the fast_td3 module to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '../fast_td3'))

from fast_td3.robots.graph_builder import GraphBuilder, env_with_object


# class TestGraphBuilder(unittest.TestCase):    
#     def test_generate_index_no_object(self):
#         """Test generate_index for environments without objects, single batch."""
#         graph_builder = GraphBuilder("h1-run-v0", 1, "cpu")
#         # case without object, single batch
#         edge_index, _, _ = graph_builder._generate_index(1, "cpu")
#         assert torch.equal(edge_index, torch.tensor(
#             [[ 0,  0,  0,  0,  1,  1,  1,  1,  2,  2,  2,  2,  3,  3,  3,  3,  4,  5,
#             5,  5,  5,  6,  6,  6,  6,  7,  7,  7,  7,  8,  8,  8,  8,  9, 10, 10,
#             10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 12, 12, 12, 12,
#             13, 13, 13, 13, 14, 14, 14, 15, 15, 15, 15, 16, 16, 16, 16, 17, 17, 17,
#             17, 18, 18, 18],
#             [10,  1,  2,  3, 10,  0,  2,  3, 10,  0,  1,  3,  0,  1,  2,  4,  3, 10,
#             6,  7,  8, 10,  5,  7,  8, 10,  5,  6,  8,  5,  6,  7,  9,  8,  0,  5,
#             1,  6,  2,  7, 11, 15, 12, 16, 13, 17, 10, 12, 13, 14, 10, 11, 13, 14,
#             10, 12, 11, 14, 12, 11, 13, 10, 16, 17, 18, 10, 15, 17, 18, 10, 16, 15,
#             18, 16, 15, 17]]))

#     def test_generate_index(self):
#         """Test generate_index for environments without objects, single batch."""
#         graph_builder = GraphBuilder("h1-sit_hard-v0", 1, "cpu")
#         # case with object, single batch
#         edge_index, edge_attr, node_attr = graph_builder._generate_index(1, "cpu")
#         assert torch.equal(edge_index, torch.tensor(
#             [[ 0,  0,  0,  0,  0,  1,  1,  1,  1,  1,  2,  2,  2,  2,  2,  3,  3,  3,
#             3,  3,  4,  4,  5,  5,  5,  5,  5,  6,  6,  6,  6,  6,  7,  7,  7,  7,
#             7,  8,  8,  8,  8,  8,  9,  9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
#             10, 10, 10, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13,
#             14, 14, 14, 14, 15, 15, 15, 15, 15, 16, 16, 16, 16, 16, 17, 17, 17, 17,
#             17, 18, 18, 18, 18],
#             [ 1,  2,  3, 10, 19,  0,  2,  3, 10, 19,  0,  1,  3, 10, 19,  0,  1,  2,
#             4, 19,  3, 19,  6,  7,  8, 10, 19,  5,  7,  8, 10, 19,  5,  6,  8, 10,
#             19,  5,  6,  7,  9, 19,  8, 19,  0,  1,  2,  5,  6,  7, 11, 12, 13, 15,
#             16, 17, 19, 10, 12, 13, 14, 19, 10, 11, 13, 14, 19, 10, 11, 12, 14, 19,
#             11, 12, 13, 19, 10, 16, 17, 18, 19, 10, 15, 17, 18, 19, 10, 15, 16, 18,
#             19, 15, 16, 17, 19]]))

#         for i in range(edge_attr.shape[0]):
#             if edge_attr[i] == 1:
#                 assert edge_index[1, i] == 19
#             else:
#                 assert edge_index[1, i] != 19
