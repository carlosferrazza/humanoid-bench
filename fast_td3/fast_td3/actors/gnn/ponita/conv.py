import torch
import torch_scatter
import torch_geometric
from torch.nn import LayerNorm, Linear, ReLU, Sequential


class FiberBundleConv(torch_geometric.nn.MessagePassing):
    """ """

    def __init__(
        self,
        in_channels,
        out_channels,
        attr_dim,
        bias=True,
        aggr="add",
        separable=True,
        groups=1,
        widening_factor=4,
    ):
        if aggr == "AttentionalAggregation":
            aggr_kwargs = {"gate_nn": Sequential(Linear(in_channels, in_channels), ReLU())}
        else:
            aggr_kwargs = {}

        super().__init__(node_dim=0, aggr=aggr, aggr_kwargs=aggr_kwargs)

        # Check arguments
        if groups == 1:
            self.depthwise = False
        elif groups == in_channels and groups == out_channels:
            self.depthwise = True
            self.in_channels = in_channels
            self.out_channels = out_channels
        else:
            assert ValueError(
                "Invalid option for groups, should be groups=1 or groups=in_channels=out_channels (depth-wise separable)"
            )

        # Construct kernels
        self.separable = separable
        if self.separable:
            self.kernel = torch.nn.Linear(attr_dim, in_channels, bias=False)
            self.fiber_kernel = torch.nn.Linear(attr_dim, int(in_channels * out_channels / groups), bias=False)
        else:
            self.kernel = torch.nn.Linear(attr_dim, int(in_channels * out_channels / groups), bias=False)

        # Construct bias
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_channels))
            self.bias.data.zero_()
        else:
            self.register_parameter("bias", None)

        # Automatic re-initialization
        self.register_buffer("callibrated", torch.tensor(False))

        if aggr == "AttentionalAggregation":
            self.vmap_aggr = torch.vmap(self.aggr_module, in_dims=(1, None), out_dims=1)
        else:
            self.vmap_aggr = None

        self.args = "sum"
        self.node_mlp = Sequential(
            LayerNorm(in_channels),
            Linear(in_channels, out_channels * widening_factor),
            torch.nn.GELU(),
            Linear(out_channels * widening_factor, out_channels),
        )

    def forward(self, x, edge_index, edge_attr, fiber_attr=None, size=None, **kwargs):
        """ """
        if isinstance(x, tuple):
            x_src, x_dst = x  # For bipartite graphs, x is a tuple (x_src, x_dst)
        else:
            x_src = x_dst = x  # For standard graphs, source and destination nodes are the same

        # Do the convolutions: 1. Spatial conv, 2. Spherical conv
        kernel = self.kernel(edge_attr)
        x_1 = self.propagate(
            edge_index,
            size=size,
            x=(x_src, x_dst),
            kernel=kernel,
            dim_size=x_dst.size(0),
        )
        if self.separable:
            fiber_kernel = self.fiber_kernel(fiber_attr)
            if self.depthwise:
                x_2 = torch.einsum("boc,opc->bpc", x_1, fiber_kernel) / fiber_kernel.shape[-2]
            else:
                x_2 = (
                    torch.einsum(
                        "boc,opdc->bpd",
                        x_1,
                        fiber_kernel.unflatten(-1, (self.out_channels, self.in_channels)),
                    )
                    / fiber_kernel.shape[-2]
                )
        else:
            x_2 = x_1

        # Re-callibrate the initializaiton
        if self.training and not (self.callibrated):
            self.callibrate(x_dst.std(), x_1.std(), x_2.std())

        # Add bias
        if self.bias is not None:
            x_2 = x_2 + self.bias

        # Apply convnext architecture
        updated_dst = x_dst + self.node_mlp(x_2)
        return (x_src, updated_dst)

    def message(self, x_i, x_j, kernel):
        if self.separable:
            return kernel * x_j
        else:
            if self.depthwise:
                return torch.einsum("bopc,boc->bpc", kernel, x_j)
            else:
                return torch.einsum(
                    "bopdc,boc->bpd",
                    kernel.unflatten(-1, (self.out_channels, self.in_channels)),
                    x_j,
                )

    def aggregate(self, edge_attr, edge_index, dim_size=None):
        """
        First we aggregate from neighbors (i.e., adjacent nodes) through concatenation,
        then we aggregate self message (from the edge itself). This is streamlined
        into one operation here.
        """

        # The axis along which to index number of nodes.
        node_dim = 0

        if self.vmap_aggr:
            out = self.vmap_aggr(edge_attr, edge_index[1, :], dim_size=dim_size, dim=node_dim)
        else:
            out = torch_scatter.scatter(
                edge_attr,
                edge_index[1, :].to(torch.int64),
                dim=node_dim,
                dim_size=dim_size,
                reduce=self.args,
            )

        return out

    def callibrate(self, std_in, std_1, std_2):
        print("Callibrating...")
        with torch.no_grad():
            self.kernel.weight.data = self.kernel.weight.data * std_in / std_1
            if self.separable:
                self.fiber_kernel.weight.data = self.fiber_kernel.weight.data * std_1 / std_2
            self.callibrated = ~self.callibrated
