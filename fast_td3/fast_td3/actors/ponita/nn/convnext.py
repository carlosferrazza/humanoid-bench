import torch


class ConvNext(torch.nn.Module):
    """
    """
    def __init__(self, channels, conv, device, act=torch.nn.GELU(), layer_scale=1e-6, widening_factor=4): 
        super().__init__()

        self.conv = conv
        self.act_fn = act
        self.device = device
        self.linear_1 = torch.nn.Linear(channels, widening_factor * channels).to(device)
        self.linear_2 = torch.nn.Linear(widening_factor * channels, channels).to(device)
        if layer_scale is not None:
            self.layer_scale = torch.nn.Parameter(torch.ones(channels) * layer_scale).to(device)
        else:
            self.register_buffer('layer_scale', None)
        self.norm = torch.nn.LayerNorm(channels).to(device)

    def forward(self, x, edge_index, edge_attr, **kwargs):
        """
        """
        input = x
        x = self.conv(x, edge_index, edge_attr, **kwargs)
        x = self.norm(x)
        x = self.linear_1(x)
        x = self.act_fn(x)
        x = self.linear_2(x)
        if self.layer_scale is not None:
            x = self.layer_scale * x
        if input.shape == x.shape: 
            x = x + input
        return x