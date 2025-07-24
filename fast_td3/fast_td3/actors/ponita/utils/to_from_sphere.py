import torch


def vec_to_sphere(vec, ori_grid):
    return torch.einsum('bcd,nd->bnc', vec, ori_grid)  # [num_nodes, num_ori, num_vec]
    
def scalar_to_sphere(scalar, ori_grid):
    return scalar.unsqueeze(-2).repeat_interleave(ori_grid.shape[-2], dim=-2)  # [num_nodes, num_ori, num_scalars]

def sphere_to_vec(spherical_signal, ori_grid):
    return torch.einsum('bnc,nd->bcd',spherical_signal, ori_grid) / ori_grid.shape[-2]  # [num_nodes, num_vec, n]

def sphere_to_scalar(spherical_signal):
    return spherical_signal.mean(dim=-2)  # [num_nodes, num_scalars]