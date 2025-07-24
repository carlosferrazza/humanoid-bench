import torch
import torch_geometric
from typing import Optional


class Conv(torch_geometric.nn.MessagePassing):
    """
    """
    def __init__(self, in_channels, out_channels, attr_dim, bias=True, aggr="add", groups=1):
        super().__init__(node_dim=0, aggr=aggr)
        
        # Check arguments
        if groups==1:
            self.depthwise = False
        elif groups==in_channels and groups==out_channels:
            self.depthwise = True
            self.in_channels = in_channels
            self.out_channels = out_channels
        else:
            assert ValueError('Invalid option for groups, should be groups=1 or groups=in_channels=out_channels (depth-wise separable)')
        
        # Construct kernel and bias
        self.kernel = torch.nn.Linear(attr_dim, int(in_channels * out_channels / groups), bias=False)
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_channels))
            self.bias.data.zero_()
        else:
            self.register_parameter('bias', None)

        # Automatic re-initialization
        self.register_buffer("callibrated", torch.tensor(False))
        
    def forward(self, x, edge_index, edge_attr, **kwargs):
        """
        """
        # Sample the convolution kernels
        kernel = self.kernel(edge_attr)

        # Do the convolution
        out = self.propagate(edge_index, x=x, kernel=kernel)

        # Re-callibrate the initializaiton
        if self.training and not(self.callibrated):
            self.callibrate(x.std(), out.std())

        # Add bias
        if self.bias is not None:
            return out + self.bias
        else:  
            return out

    def message(self, x_j, kernel):
        if self.depthwise:
            return kernel * x_j
        else:
            return torch.einsum('boi,bi->bo', kernel.unflatten(-1, (self.out_channels, self.in_channels)), x_j)
    
    def callibrate(self, std_in, std_out):
        print('Callibrating...')
        with torch.no_grad():
            self.kernel.weight.data = self.kernel.weight.data * std_in/std_out
            self.callibrated = ~self.callibrated


class FiberBundleConv(torch_geometric.nn.MessagePassing):
    """
    """
    def __init__(self, in_channels, out_channels, attr_dim, bias=True, aggr="add", separable=True, groups=1):
        super().__init__(node_dim=0, aggr=aggr)

        # Check arguments
        if groups==1:
            self.depthwise = False
        elif groups==in_channels and groups==out_channels:
            self.depthwise = True
            self.in_channels = in_channels
            self.out_channels = out_channels
        else:
            assert ValueError('Invalid option for groups, should be groups=1 or groups=in_channels=out_channels (depth-wise separable)')

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
            self.register_parameter('bias', None)
        
        # Automatic re-initialization
        self.register_buffer("callibrated", torch.tensor(False))
        
    def forward(self, x, edge_index, edge_attr, fiber_attr=None, **kwargs):
        """
        """

        # Do the convolutions: 1. Spatial conv, 2. Spherical conv
        kernel = self.kernel(edge_attr)
        x_1 = self.propagate(edge_index, x=x, kernel=kernel)
        if self.separable:
            fiber_kernel = self.fiber_kernel(fiber_attr)
            if self.depthwise:
                x_2 = torch.einsum('boc,opc->bpc', x_1, fiber_kernel) / fiber_kernel.shape[-2]
            else:
                x_2 = torch.einsum('boc,opdc->bpd', x_1, fiber_kernel.unflatten(-1, (self.out_channels, self.in_channels))) / fiber_kernel.shape[-2]
        else:
            x_2 = x_1

        # Re-callibrate the initializaiton
        if self.training and not(self.callibrated):
            self.callibrate(x.std(), x_1.std(), x_2.std())

        # Add bias
        if self.bias is not None:
            return x_2 + self.bias
        else:  
            return x_2

    def message(self, x_j, kernel):
        if self.separable:
            return kernel * x_j
        else:
            if self.depthwise:
                return torch.einsum('bopc,boc->bpc', kernel, x_j)
            else:
                return torch.einsum('bopdc,boc->bpd', kernel.unflatten(-1, (self.out_channels, self.in_channels)), x_j)
    
    def callibrate(self, std_in, std_1, std_2):
        print('Callibrating...')
        with torch.no_grad():
            self.kernel.weight.data = self.kernel.weight.data * std_in/std_1
            if self.separable:
                self.fiber_kernel.weight.data = self.fiber_kernel.weight.data * std_1/std_2
            self.callibrated = ~self.callibrated
