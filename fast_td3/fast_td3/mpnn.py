import torch
import torch_scatter
from torch.nn import LayerNorm, Linear, ReLU, Sequential
from torch_geometric.nn.conv import MessagePassing


class ProcessorLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, device, update_edge=True, args="sum", **kwargs):
        super(ProcessorLayer, self).__init__(**kwargs)
        """
        in_channels: dim of node embeddings [128], out_channels: dim of edge embeddings [128]

        """

        # Note that the node and edge encoders both have the same hidden dimension
        # size. This means that the input of the edge processor will always be
        # three times the specified hidden dimension
        # (input: adjacent node embeddings and self embeddings)
        self.update_edge = update_edge
        self.args = args
        self.device = device

        if self.update_edge:
            self.edge_mlp = Sequential(
                Linear(3 * in_channels, out_channels),
                ReLU(),
                Linear(out_channels, out_channels),
                LayerNorm(out_channels),
            ).to(self.device)

        self.node_mlp = Sequential(
            Linear(2 * in_channels, out_channels),
            ReLU(),
            Linear(out_channels, out_channels),
            LayerNorm(out_channels),
        ).to(self.device)

        self.reset_parameters()

    def reset_parameters(self):
        """
        reset parameters for stacked MLP layers
        """
        if self.update_edge:
            self.edge_mlp[0].reset_parameters()
            self.edge_mlp[2].reset_parameters()

        self.node_mlp[0].reset_parameters()
        self.node_mlp[2].reset_parameters()

    def forward(self, x, edge_index, edge_attr, size=None):
        """
        Handle the pre and post-processing of node features/embeddings,
        as well as initiates message passing by calling the propagate function.

        Note that message passing and aggregation are handled by the propagate
        function, and the update

        x has shpae [node_num , in_channels] (node embeddings)
        edge_index: [2, edge_num]
        edge_attr: [E, in_channels]

        """

        # Check if x is a tuple (bipartite graph) or a tensor (standard graph)
        if isinstance(x, tuple):
            x_src, x_dst = x  # For bipartite graphs, x is a tuple (x_src, x_dst)
        else:
            x_src = x_dst = x  # For standard graphs, source and destination nodes are the same

        # Propagate method needs to be called accordingly
        out, updated_edges = self.propagate(
            edge_index,
            size=size,
            x=(x_src, x_dst),
            edge_attr=edge_attr,
            dim_size=x_dst.size(0),
        )

        if isinstance(x, tuple):
            # For bipartite graphs, 'out' contains aggregated messages at destination nodes
            updated_dst = torch.cat([x_dst, out], dim=1)
            updated_dst = x_dst + self.node_mlp(updated_dst)  # Residual connection
            updated_nodes = (x_src, updated_dst)
        else:
            # Handle nodes for standard graphs
            updated_nodes = torch.cat([x, out], dim=1)
            updated_nodes = x + self.node_mlp(updated_nodes)  # Residual connection

        return updated_nodes, updated_edges

    def message(self, x_i, x_j, edge_attr):
        """
        target_node: x_i has the shape of [E, in_channels]
        source_node: x_j has the shape of [E, in_channels]
        target_edge: edge_attr has the shape of [E, out_channels]

        The messages that are passed are the raw embeddings. These are not processed.
        """

        if self.update_edge:
            updated_edges = torch.cat([x_i, x_j, edge_attr], dim=1)  # tmp_emb has the shape of [E, 3 * in_channels]
            updated_edges = self.edge_mlp(updated_edges) + edge_attr
        else:
            updated_edges = (x_i, x_j, edge_attr)

        return updated_edges

    def aggregate(self, updated_edges, edge_index, dim_size=None):
        """
        First we aggregate from neighbors (i.e., adjacent nodes) through concatenation,
        then we aggregate self message (from the edge itself). This is streamlined
        into one operation here.
        """

        # The axis along which to index number of nodes.
        node_dim = 0

        if self.update_edge:
            out = torch_scatter.scatter(
                updated_edges,
                edge_index[1, :].to(torch.int64),
                dim=node_dim,
                dim_size=dim_size,
                reduce=self.args,
            ).to(self.device)
        else:
            x_i, x_j, edge_attr = updated_edges
            out = torch_scatter.scatter(
                x_j * edge_attr,
                edge_index[1, :].to(torch.int64),
                dim=node_dim,
                dim_size=dim_size,
                reduce=self.args,
            ).to(self.device)
            updated_edges = edge_attr

        return out, updated_edges
