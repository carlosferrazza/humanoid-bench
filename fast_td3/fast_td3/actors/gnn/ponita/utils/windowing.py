import torch


class PolynomialCutoff(torch.nn.Module):
    """
    Klicpera, J.; Groß, J.; Günnemann, S. Directional Message Passing for Molecular Graphs; ICLR 2020.
    Equation (8)
    """

    p: torch.Tensor
    r_max: torch.Tensor

    def __init__(self, r_max, p=6):
        super().__init__()
        if r_max is not None:
            self.register_buffer("p", torch.tensor(p, dtype=torch.get_default_dtype()))
            self.register_buffer("r_max", torch.tensor(r_max, dtype=torch.get_default_dtype()))
        else:
            self.r_max = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.r_max is not None:
            envelope = (
                1.0
                - ((self.p + 1.0) * (self.p + 2.0) / 2.0) * torch.pow(x / self.r_max, self.p)
                + self.p * (self.p + 2.0) * torch.pow(x / self.r_max, self.p + 1)
                - (self.p * (self.p + 1.0) / 2) * torch.pow(x / self.r_max, self.p + 2)
            )
            return envelope * (x < self.r_max)
        else:
            return torch.ones_like(x)

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p}, r_max={self.r_max})"
