import torch

from fast_td3.robots.h1 import H1

def build_edge_index_and_attr(robot_name, batch_size, device):
    robot = None
    if robot_name == "h1":
        robot = H1
    else:
        raise NotImplementedError(f"Robot {robot_name} is not supported yet.")
    
    edge_list = robot.edge_list
    src, dst = zip(*edge_list)  # Unpack edge list into two tuples
    src = torch.tensor(src, dtype=torch.long)
    dst = torch.tensor(dst, dtype=torch.long)

    # Create batch offsets and expand edges in one operation
    offsets = torch.arange(batch_size) * 19
    src_batch = (src.unsqueeze(0) + offsets.unsqueeze(1)).flatten().to(device)
    dst_batch = (dst.unsqueeze(0) + offsets.unsqueeze(1)).flatten().to(device)

    # Create edge attributes for all batches at once
    edge_index = torch.stack([src_batch, dst_batch])
    edge_attr = torch.tensor(robot.edge_type_encoding).repeat(batch_size).unsqueeze(-1).to(device)

    return edge_index, edge_attr
