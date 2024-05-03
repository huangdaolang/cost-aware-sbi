import torch
from sbi.utils.torchutils import *


class HomoSIR(object):
    def __init__(self, name="homo_sir",
                 k=1,
                 prior_lower=1.0,
                 prior_upper=10.0,
                 theta_dim=1,
                 x_dim=1,
                 prior_start=None):
        self.name = name
        self.k = k
        self.N = 10000
        self.theta_dim = theta_dim
        self.x_dim = x_dim
        self.prior = Uniform(prior_lower * torch.ones(1), prior_upper * torch.ones(1))

    def sample_theta(self, size):
        return self.prior.sample(size)

    def __call__(self, thetas):
        l = thetas[0]
        S = self.N - 1
        y = 1
        while y > 0:
            y -= 1
            if self.k == 0:
                I = torch.tensor(1.0)
            else:
                I = torch.distributions.Gamma(self.k, self.k).sample()
            Z = torch.poisson(l * I).int()  # number of infectious contacts

            if Z > 0:
                for _ in range(Z):
                    u = torch.rand(1)
                    if u < (S / self.N):
                        S -= 1
                        y += 1

        return self.N - S


if __name__ == "__main__":
    l = 3
    k = 1
    print(HomoSIR())

