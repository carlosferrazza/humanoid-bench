import torch


class PolynomialFeatures(torch.nn.Module):
    def __init__(self, degree):
        super(PolynomialFeatures, self).__init__()

        self.degree = degree

    def forward(self, x):

        polynomial_list = [x]
        for it in range(1, self.degree):
            polynomial_list.append(torch.einsum('...i,...j->...ij', polynomial_list[-1], x).flatten(-2,-1))
        return torch.cat(polynomial_list, -1)


class RandomFourierFeatures(torch.nn.Module):
    def __init__(self, out_dim, sigma, symmetric=None):
        super(RandomFourierFeatures, self).__init__()

        self.out_dim = out_dim
        if out_dim % 2 != 0:
            self.compensation = 1
        else:
            self.compensation = 0
        self.num_frequencies = int(out_dim / 2) + self.compensation

        if symmetric is None:
            symmetric = [False] * len(sigma)
        self.unconstraint_idx = [i for i, x in enumerate(symmetric) if not(x)]
        self.constraint_idx = [i for i, x in enumerate(symmetric) if x]

        self.sigma = sigma
        if len(self.unconstraint_idx) > 0:
            self.frequencies_unconstraint = torch.stack([self.random_frequencies(self.sigma[i], self.num_frequencies) for i in self.unconstraint_idx],dim=0)
        else:
            self.frequencies_unconstraint = None
        if len(self.constraint_idx) > 0:
            self.frequencies_constraint = torch.stack([self.random_frequencies(self.sigma[i], 2 * self.num_frequencies) for i in self.constraint_idx],dim=0)
        else:
            self.frequencies_constraint = None
        
    def random_frequencies(self, sigma, num_frequencies):
            if type(sigma)==float:
                # Continuous frequencies, sigma is interpreted as the std of the gaussian distribution from which we sample
                return torch.randn(num_frequencies) * math.sqrt(1/2) * sigma
            elif type(sigma)==int:
                # Integer frequencies, now sigma is interpreted as the integer band-limit (max frequency)
                return torch.randint(-sigma, sigma,(num_frequencies,))
        
    def forward(self, x):
        
        # Mix unconstraint terms
        if self.frequencies_unconstraint is not None:
            unconstraint_proj = x[..., self.unconstraint_idx] @ self.frequencies_unconstraint.type_as(x)
            out = torch.cat([unconstraint_proj.cos(),unconstraint_proj.sin()], dim=-1)
        else:
            out = torch.ones((1, self.num_frequencies * 2))
        
        # Tensor product for contraint terms
        if self.frequencies_constraint is not None:
            out = out * torch.einsum('...d,di->...di', x[...,self.constraint_idx], self.frequencies_constraint.type_as(x)).cos().prod(dim=-2)

        # Crop to dimension (if necessary), and return output
        if self.compensation:
            out = out[..., :-1]
        return out