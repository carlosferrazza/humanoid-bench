import torch


def invariant_attr_rn(pos, edge_index):
    pos_send, pos_receive = pos[edge_index[0]], pos[edge_index[1]] # [num_edges, n]
    dists = (pos_send - pos_receive).norm(dim=-1, keepdim=True)    # [num_edges, 1]
    return dists


def invariant_attr_r3s2_fiber_bundle(pos, ori_grid, edge_index, separable=False):
    pos_send, pos_receive = pos[edge_index[0]], pos[edge_index[1]]                # [num_edges, 3]
    rel_pos = (pos_send - pos_receive)                                            # [num_edges, 3]

    # Convenient shape
    rel_pos = rel_pos[:, None, :]                                                 # [num_edges, 1, 3]
    ori_grid_a = ori_grid[None,:,:]                                               # [1, num_ori, 3]
    ori_grid_b = ori_grid[:, None,:]                                              # [num_ori, 1, 3]

    invariant1 = (rel_pos * ori_grid_a).sum(dim=-1, keepdim=True)                 # [num_edges, num_ori, 1]
    invariant2 = (rel_pos - invariant1 * ori_grid_a).norm(dim=-1, keepdim=True)   # [num_edges, num_ori, 1]
    invariant3 = (ori_grid_a * ori_grid_b).sum(dim=-1, keepdim=True)              # [num_ori, num_ori, 1]
    
    # Note: We could apply the acos = pi/2 - asin, which is differentiable at -1 and 1
    # But found that this mapping is unnecessary as it is monotonic and mostly linear 
    # anyway, except close to -1 and 1. Not applying the arccos worked just as well.
    # invariant3 = torch.pi / 2 - torch.asin(invariant3.clamp(-1.,1.))
    
    if separable:
        return torch.cat([invariant1, invariant2],dim=-1), invariant3             # [num_edges, num_ori, 2], [num_ori, num_ori, 1]
    else:
        invariant1 = invariant1[:,:,None,:].expand(-1,-1,ori_grid.shape[0],-1)    # [num_edges, num_ori, num_ori, 1]
        invariant2 = invariant2[:,:,None,:].expand(-1,-1,ori_grid.shape[0],-1)    # [num_edges, num_ori, num_ori, 1]
        invariant3 = invariant3[None,:,:,:].expand(invariant1.shape[0],-1,-1,-1)  # [num_edges, num_ori, num_ori, 1]
        return torch.cat([invariant1, invariant2, invariant3],dim=-1)             # [num_edges, num_ori, num_ori, 3]
    
def invariant_attr_r3s2_point_cloud(pos, edge_index):
    pos_send, pos_receive = pos[edge_index[0],:3], pos[edge_index[1],:3]          # [num_edges, 3]
    ori_send, ori_receive = pos[edge_index[0],3:], pos[edge_index[1],3:]          # [num_edges, 3]
    rel_pos = pos_send - pos_receive                                              # [num_edges, 3]

    invariant1 = torch.sum(rel_pos * ori_receive, dim=-1, keepdim=True)
    invariant2 = (rel_pos - ori_receive * invariant1).norm(dim=-1, keepdim=True)
    invariant3 = torch.sum(ori_send * ori_receive, dim=-1, keepdim=True)

    return torch.cat([invariant1, invariant2, invariant3],dim=-1)             # [num_edges, num_ori, num_ori, 3]

def invariant_attr_r2s1_fiber_bundle(pos, ori_grid, edge_index, separable=False):
    pos_send, pos_receive = pos[edge_index[0]], pos[edge_index[1]]                # [num_edges, 3]
    rel_pos = (pos_send - pos_receive)                                            # [num_edges, 3]

    # Convenient shape
    rel_pos = rel_pos[:, None, :]                                                 # [num_edges, 1, 3]
    ori_grid_a = ori_grid[None,:,:]                                               # [1, num_ori, 3]
    ori_grid_b = ori_grid[:, None,:]                                              # [num_ori, 1, 3]

    # Note ori_grid consists of tuples (ori[0], ori[1]) = (cos t, sin t)
    # A transposed rotation (cos t, sin t \\ - sin t, cos t) is then 
    # acchieved as (ori[0], ori[1] \\ -ori[1], ori[0]):
    invariant1 = (rel_pos[...,0] * ori_grid_a[...,0] + rel_pos[...,1] * ori_grid_a[...,1]).unsqueeze(-1)
    invariant2 = (- rel_pos[...,0] * ori_grid_a[...,1] + rel_pos[...,1] * ori_grid_a[...,0]).unsqueeze(-1)
    invariant3 = (ori_grid_a * ori_grid_b).sum(dim=-1, keepdim=True)              # [num_ori, num_ori, 1]
    
    # Note: We could apply the acos = pi/2 - asin, which is differentiable at -1 and 1
    # But found that this mapping is unnecessary as it is monotonic and mostly linear 
    # anyway, except close to -1 and 1. Not applying the arccos worked just as well.
    # invariant3 = torch.pi / 2 - torch.asin(invariant3.clamp(-1.,1.))
    
    if separable:
        return torch.cat([invariant1, invariant2],dim=-1), invariant3             # [num_edges, num_ori, 2], [num_ori, num_ori, 1]
    else:
        invariant1 = invariant1[:,:,None,:].expand(-1,-1,ori_grid.shape[0],-1)    # [num_edges, num_ori, num_ori, 1]
        invariant2 = invariant2[:,:,None,:].expand(-1,-1,ori_grid.shape[0],-1)    # [num_edges, num_ori, num_ori, 1]
        invariant3 = invariant3[None,:,:,:].expand(invariant1.shape[0],-1,-1,-1)  # [num_edges, num_ori, num_ori, 1]
        return torch.cat([invariant1, invariant2, invariant3],dim=-1)             # [num_edges, num_ori, num_ori, 3]
        
def invariant_attr_r2s1_point_cloud(pos, edge_index):
    pos_send, pos_receive = pos[edge_index[0],:2], pos[edge_index[1],:2]          # [num_edges, 2]
    ori_send, ori_receive = pos[edge_index[0],2:], pos[edge_index[1],2:]          # [num_edges, 2]

    rel_pos = pos_send - pos_receive                                              # [num_edges, 2]
    invariant1 = rel_pos[...,0] * ori_receive[...,0] + rel_pos[...,1] * ori_receive[...,1]
    invariant2 = - rel_pos[...,0] * ori_receive[...,1] + rel_pos[...,1] * ori_receive[...,0]
    invariant3 = torch.sum(ori_send * ori_receive, dim=-1, keepdim=False)

    return torch.stack([invariant1, invariant2, invariant3],dim=-1)             # [num_edges, num_ori, num_ori, 3]    
