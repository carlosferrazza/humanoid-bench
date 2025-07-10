from typing import Dict, List, Tuple

from torch import Tensor
from torch_geometric.nn.conv import HeteroConv
from torch_geometric.nn.conv.hetero_conv import group
from torch_geometric.typing import EdgeType, NodeType


class HeteroFiberConv(HeteroConv):
    def forward(
        self,
        latent_dict: Dict[NodeType, Tensor],
        edge_index_dict: Dict[EdgeType, Tensor],
        edge_attr_dict: Dict[EdgeType, Tensor],
        fiber_attr_dict: Dict[EdgeType, Tensor],
    ) -> Dict[NodeType, Tensor]:
        r"""Runs the forward pass of the module.

        Args:
            x_dict (Dict[str, torch.Tensor]): A dictionary holding node feature
                information for each individual node type.
            edge_index_dict (Dict[Tuple[str, str, str], torch.Tensor]): A
                dictionary holding graph connectivity information for each
                individual edge type, either as a :class:`torch.Tensor` of
                shape :obj:`[2, num_edges]` or a
                :class:`torch_sparse.SparseTensor`.
            edge_attr_dict (Dict[Tuple[str, str, str], torch.Tensor]): A
                dictionary holding edge feature information for each individual
                edge type.
        """
        out_dst_dict: Dict[str, List[Tensor]] = {}

        for edge_type, conv in self.convs.items():
            src, rel, dst = edge_type

            args = []
            if src == dst:
                args.append(latent_dict[src])
            elif src in latent_dict and dst in latent_dict:
                args.append((latent_dict[src], latent_dict[dst]))

            kwargs = {}
            kwargs["edge_index"] = edge_index_dict[edge_type]
            kwargs["edge_attr"] = edge_attr_dict[edge_type]
            kwargs["fiber_attr"] = fiber_attr_dict[edge_type]

            # Skip computation if edge_index is empty.
            if kwargs["edge_index"].nelement() == 0:
                continue

            out = conv(*args, **kwargs)

            if isinstance(out, tuple):
                out_src, out_dst = out
            else:
                out_src = out_dst = out

            if dst not in out_dst_dict:
                out_dst_dict[dst] = [out_dst]
            else:
                out_dst_dict[dst].append(out_dst)

        for key, value in out_dst_dict.items():
            latent_dict[key] = group(value, self.aggr)

        return latent_dict
