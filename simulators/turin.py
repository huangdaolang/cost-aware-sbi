import torch
from sbi.utils.torchutils import *
import numpy as np
import os


class Turin(object):
    def __init__(self, name="turin",
                 k=1,
                 theta_1_lower=1e-9,
                 theta_1_upper=1e-8,
                 theta_2_lower=1e-9,
                 theta_2_upper=1e-8,
                 theta_3_lower=1e9,
                 theta_3_upper=1e10,
                 theta_4_lower=1e-10,
                 theta_4_upper=1e-9,
                 theta_dim=4,
                 x_dim=801,
                 inference=None):
        self.path = os.path.dirname(os.path.realpath(__file__))
        self.name = name
        self.k = k
        self.theta_dim = theta_dim
        self.x_dim = x_dim
        self.prior = [Uniform(torch.tensor([theta_1_lower]), torch.tensor([theta_1_upper])),
                      Uniform(torch.tensor([theta_2_lower]), torch.tensor([theta_2_upper])),
                      Uniform(torch.tensor([theta_3_lower]), torch.tensor([theta_3_upper])),
                      Uniform(torch.tensor([theta_4_lower]), torch.tensor([theta_4_upper]))]

        self.theta_true = torch.tensor([4e-9, 7e-9, 5e9, 3e-10])
        self.x_obs = torch.tensor(np.load(f'{self.path}/../data/turin_x_obs.npy'))

        self.cost_alpha = torch.tensor(np.load(f'{self.path}/../data/turin_cost_alpha.npy'))
        self.cost_beta = torch.tensor(np.load(f'{self.path}/../data/turin_cost_beta.npy'))

    def sample_theta(self, size):
        return NotImplementedError

    def __call__(self, thetas):
        return NotImplementedError

    def load_npe_data(self):
        x = torch.tensor(np.load(f'{self.path}/../data/turin_x_nle.npy'))
        theta = torch.tensor(np.load(f'{self.path}/../data/turin_theta_nle.npy'))

        return x, theta

    def load_ca_npe_data(self, mixture=True, k=2):
        if mixture:
            x = torch.tensor(np.load(f'{self.path}/../data/turin_x_ca_nle_mixture.npy'))
            theta = torch.tensor(np.load(f'{self.path}/../data/turin_theta_ca_nle_mixture.npy'))
            w = torch.tensor(np.load(f'{self.path}/../data/turin_weights_mixture.npy'))
        else:
            x = torch.tensor(np.load(f'{self.path}/../data/turin_x_ca_nle_k2.npy'))
            theta = torch.tensor(np.load(f'{self.path}/../data/turin_theta_ca_nle_k2.npy'))
            w = self.cost_function(theta[:, 2], k)

        return x, theta, w

    def cost_function(self, theta, k):
        return (self.cost_alpha * theta + self.cost_beta)**k


if __name__ == "__main__":
    turin = Turin()
    turin.load_ca_npe_data(True)


