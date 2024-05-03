import torch
from sbi.utils.torchutils import *
import torch.distributions as dist


class BernSIR(object):
    def __init__(self,
                 name="bern_sir",
                 population=100,
                 p_lower=0.1,
                 p_upper=1.0,
                 beta_lower=0.1,
                 beta_upper=1.0,
                 gamma_lower=0.1,
                 gamma_upper=1.0,
                 theta_dim=3,
                 x_dim=12,
                 num_bins=10,
                 prior_start=None):

        self.name = name
        self.N = population
        self.theta_dim = theta_dim
        self.x_dim = x_dim
        self.num_bins = num_bins
        self.prior = [Uniform(torch.tensor([beta_lower]), torch.tensor([beta_upper])),
                      # dist.Gamma(concentration=torch.tensor([gamma_concentration]), rate=torch.tensor([gamma_rate])),
                      Uniform(torch.tensor([gamma_lower]), torch.tensor([gamma_upper])),
                      # dist.Beta(torch.tensor([p_concentration_1]), torch.tensor([p_concentration_2]))]
                      Uniform(torch.tensor([p_lower]), torch.tensor([p_upper]))]

    def sample_theta(self, size):
        beta = self.prior[0].sample(size).reshape(-1, 1)
        gamma = self.prior[1].sample(size).reshape(-1, 1)
        p = self.prior[2].sample(size).reshape(-1, 1)

        return torch.cat([beta, gamma, p], dim=1)

    def __call__(self, thetas):
        beta = thetas[0]
        gamma = thetas[1]
        p = thetas[2]

        t = torch.tensor(0.0)
        MAT = torch.distributions.Bernoulli(torch.tensor([p])).sample((self.N, self.N)).squeeze()
        rowM = MAT.sum(dim=1)
        I = torch.zeros(self.N)
        I[0] = 1  # Set individual 1 infectious
        times = []  # Use a list for dynamic append
        count = 0  # Number of recoveries observed

        while (I == 1).sum() > 0:
            rec = (I == 1).sum()
            infe = torch.mv(MAT, (I == 1).float()).sum()
            rate = gamma * rec + beta * infe
            t += torch.distributions.Exponential(1 / rate).sample()
            u = torch.rand(1)

            if u <= beta * infe / (gamma * rec + beta * infe):
                S = MAT @ (I == 1).float()  # Project infection probabilities
                K = torch.multinomial(S, 1)  # Select an infectious individual
                J = torch.multinomial(MAT[K], 1)  # Select a susceptible contact
                if I[J] == 0:
                    I[J] = 1  # Infect the chosen susceptible
            else:
                S = (I == 1).float()
                K = torch.multinomial(S, 1)  # Select a recovering individual
                I[K] = 2  # Recover the chosen individual
                count += 1
                times.append(t.item())
        T = times[-1]
        removals_per_bin = torch.histc(torch.tensor(times), bins=self.num_bins, min=0, max=T)
        x = torch.cat([removals_per_bin, torch.tensor([count]), torch.tensor([T])])

        return x


if __name__ == "__main__":
    simulator = BernSIR()
    beta = 0.5
    gamma = 5
    p = 0.8
    result = simulator(torch.tensor([beta, gamma, p]))
    print(result)

