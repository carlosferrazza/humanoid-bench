import torch

from fast_td3.robots.h1 import H1
from fast_td3.robots.g1 import G1


def build_edge_index_and_attr(robot_name, batch_size, device):
    if robot_name == "h1":
        robot = H1
    else:
        robot = G1

    edge_list = robot.edge_list
    src, dst = zip(*edge_list)  # Unpack edge list into two tuples
    src = torch.tensor(src, dtype=torch.long)
    dst = torch.tensor(dst, dtype=torch.long)

    # Create batch offsets and expand edges in one operation
    offsets = torch.arange(batch_size) * robot.num_nodes
    src_batch = (src.unsqueeze(0) + offsets.unsqueeze(1)).flatten().to(device)
    dst_batch = (dst.unsqueeze(0) + offsets.unsqueeze(1)).flatten().to(device)

    edge_index = torch.stack([src_batch, dst_batch])

    # Normalize edge types as a tensor
    # edge_types = torch.tensor(robot.edge_type_encoding, dtype=torch.float32, device=device)
    # normalized_edge_type_encoding = edge_types / edge_types.max()
    # edge_attr = (
    #     normalized_edge_type_encoding
    #     .repeat(batch_size)
    #     .unsqueeze(-1)
    # )

    return edge_index, robot.num_nodes, robot.num_edges


def build_edge_index_and_node_attr(robot_name, batch_size, device):
    if robot_name == "h1":
        robot = H1
    else:
        robot = G1
    
    edge_list = robot.edge_list
    src, dst = zip(*edge_list)  # Unpack edge list into two tuples
    src = torch.tensor(src, dtype=torch.long)
    dst = torch.tensor(dst, dtype=torch.long)

    # Create batch offsets and expand edges in one operation
    offsets = torch.arange(batch_size) * robot.num_nodes
    src_batch = (src.unsqueeze(0) + offsets.unsqueeze(1)).flatten().to(device)
    dst_batch = (dst.unsqueeze(0) + offsets.unsqueeze(1)).flatten().to(device)

    # Create edge attributes for all batches at once
    edge_index = torch.stack([src_batch, dst_batch])
    node_attr = (
        torch.tensor(robot.node_type_encoding)
        .repeat(batch_size)
        .unsqueeze(-1)
        .to(device)
    )

    return edge_index, node_attr, robot.num_nodes, robot.num_edges


def build_edge_index_and_attr_mpnn(robot_name, batch_size, device):
    robot = None
    if robot_name == "h1":
        robot = H1
    else:
        robot = G1

    edge_list = robot.edge_list
    src, dst = zip(*edge_list)  # Unpack edge list into two tuples
    src = torch.tensor(src)
    dst = torch.tensor(dst)

    # Create batch offsets and expand edges in one operation
    offsets = torch.arange(batch_size) * robot.num_nodes
    src_batch = (src.unsqueeze(0) + offsets.unsqueeze(1)).flatten().to(device)
    dst_batch = (dst.unsqueeze(0) + offsets.unsqueeze(1)).flatten().to(device)

    # Create edge attributes for all batches at once
    edge_index = torch.stack([src_batch, dst_batch], dim=0).to(device)
    edge_attr = (
        torch.tensor(robot.edge_type_encoding, dtype=torch.float)
        .repeat(batch_size)
        .unsqueeze(-1)
        .to(device)
    )

    return edge_index, edge_attr, robot.num_nodes, robot.num_edges

